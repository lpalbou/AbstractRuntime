"""Wave-2 model-management relay: MODEL LOCK + CONTEXT ESTIMATE (ADR 0007:
relay core-owned truth, never infer).

Covers:
- host-facade optional contract for the three new methods (older clients keep
  binding; unsupported degrades to the structured envelope),
- the dual calling convention (payload mapping and/or kwargs; kwargs win; the
  first positional is ONLY ever the payload mapping),
- remote wire shapes (lock/unlock POST bodies, context_estimate GET, the 409
  model_locked conversion, force riding the unload body only when true),
- client-side lock semantics on the local clients (refusal without force,
  force unlock-then-unload, the best-effort ollama keep_alive knob, lock
  state surviving and surfacing across list calls),
- MODEL_RESIDENCY effect ops `lock`/`unlock` (durable, soft-fail semantics),
- passthrough of the new core record fields through local claims and the
  sweep merge (sweep-only rows: `lockable: false` stamped if absent),
- the guarded estimator import.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractruntime import Effect, EffectType, Runtime, RunState, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore import host_facade
from abstractruntime.integrations.abstractcore.host_facade import AbstractCoreHostFacade
from abstractruntime.integrations.abstractcore.effect_handlers import make_model_residency_handler
import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
    _local_provider_residency_claim,
)
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeOllamaProvider:
    """Records `load_model` keep_alive knob calls and answers verified
    residency, in the `_LoadableCoreResidencyProvider` idiom."""

    def __init__(self, *, model: str, loaded: bool = True) -> None:
        self.model = model
        self.loaded = loaded
        self.load_model_calls: List[Dict[str, Any]] = []
        self.unload_model_calls: List[Dict[str, Any]] = []
        self.raise_on_load = False
        self.raise_on_unload = False

    def get_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "task": "text_generation",
            "provider": "ollama",
            "model": str(kwargs.get("model") or self.model),
            "provider_residency_verified": True,
            "provider_resident": bool(self.loaded),
            "loaded": bool(self.loaded),
            "state": "loaded" if self.loaded else "not_loaded",
            "source": "abstractcore.provider.ollama",
        }

    def load_model(self, model_name: str, **kwargs: Any) -> Dict[str, Any]:
        if self.raise_on_load:
            raise RuntimeError("keep_alive knob exploded")
        self.loaded = True
        self.load_model_calls.append({"model": str(model_name), "kwargs": dict(kwargs)})
        return {"supported": True, "operation": "load", "model": str(model_name)}

    def unload_model(self, model_name: str, **kwargs: Any) -> Dict[str, Any]:
        if self.raise_on_unload:
            raise RuntimeError("provider unload exploded")
        self.loaded = False
        self.unload_model_calls.append({"model": str(model_name), "kwargs": dict(kwargs)})
        return {"supported": True, "operation": "unload", "model": str(model_name)}


def _local_client(*, provider: str, model: str, provider_instance: Any) -> LocalAbstractCoreLLMClient:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = provider
    client._model = model
    client._llm_kwargs = {}
    client._llm = provider_instance
    return client


class _LockRecordingSender:
    """Recording request sender in the `_ResidencySender` idiom; optional
    scripted exceptions for the raise_for_status pattern."""

    def __init__(
        self,
        *,
        get_response: Optional[Dict[str, Any]] = None,
        post_response: Optional[Dict[str, Any]] = None,
        post_raises: Optional[Exception] = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._get_response = dict(get_response or {"ok": True})
        self._post_response = dict(post_response or {"ok": True})
        self._post_raises = post_raises

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers), "timeout": timeout})
        return dict(self._get_response)

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        self.calls.append({"method": "POST", "url": url, "headers": dict(headers), "json": dict(json), "timeout": timeout})
        if self._post_raises is not None:
            raise self._post_raises
        return dict(self._post_response)


class _FakeHttpStatusError(Exception):
    """Stands in for httpx.HTTPStatusError: str(exc) + `.response`."""

    def __init__(self, message: str, *, response: Any) -> None:
        super().__init__(message)
        self.response = response


class _FakeErrorResponse:
    def __init__(self, *, status_code: int, json_body: Any = None) -> None:
        self.status_code = status_code
        self._json_body = json_body

    def json(self) -> Any:
        if self._json_body is None:
            raise ValueError("no body")
        return self._json_body


# ---------------------------------------------------------------------------
# Facade optional contract + calling conventions
# ---------------------------------------------------------------------------


class _RequiredOnlyClient:
    """Implements ONLY the required host-control contract (an older client)."""

    def __getattr__(self, name: str) -> Any:
        if name in host_facade._HOST_CONTROL_METHODS:
            def _method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                return {"ok": True, "method": name}

            return _method
        raise AttributeError(name)


class _LockRecordingClient(_RequiredOnlyClient):
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any, Dict[str, Any]]] = []

    def lock_model_residency(self, payload: Any = None, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("lock_model_residency", payload, dict(kwargs)))
        return {"ok": True, "locked": True}

    def unlock_model_residency(self, payload: Any = None, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("unlock_model_residency", payload, dict(kwargs)))
        return {"ok": True, "locked": False}

    def get_context_estimate(self, payload: Any = None, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("get_context_estimate", payload, dict(kwargs)))
        return {"ok": True, "confidence": "estimated"}


def test_lock_and_estimate_methods_are_optional_in_the_facade_contract() -> None:
    for name in ("lock_model_residency", "unlock_model_residency", "get_context_estimate"):
        assert name not in host_facade._HOST_CONTROL_METHODS
        assert name in host_facade._OPTIONAL_HOST_CONTROL_METHODS

    facade = AbstractCoreHostFacade(SimpleNamespace(_abstractcore_llm_client=_RequiredOnlyClient()))

    locked = facade.lock_model_residency(provider="ollama", model="llama3")
    unlocked = facade.unlock_model_residency(provider="ollama", model="llama3")
    estimate = facade.get_context_estimate(provider="mlx", model="qwen")

    for payload in (locked, unlocked, estimate):
        assert payload["ok"] is False
        assert payload["supported"] is False
        assert "does not implement" in payload["error"]
    assert locked["operation"] == "lock_model_residency"
    assert unlocked["operation"] == "unlock_model_residency"
    assert estimate["operation"] == "get_context_estimate"


def test_facade_relays_both_calling_conventions_verbatim() -> None:
    client = _LockRecordingClient()
    facade = AbstractCoreHostFacade(SimpleNamespace(_abstractcore_llm_client=client))

    facade.lock_model_residency({"provider": "ollama", "model": "llama3"})
    facade.unlock_model_residency(provider="ollama", model="llama3")
    facade.get_context_estimate({"provider": "mlx", "model": "a"}, model="b", context_length=4096)

    assert client.calls == [
        ("lock_model_residency", {"provider": "ollama", "model": "llama3"}, {}),
        ("unlock_model_residency", None, {"provider": "ollama", "model": "llama3"}),
        ("get_context_estimate", {"provider": "mlx", "model": "a"}, {"model": "b", "context_length": 4096}),
    ]


def test_first_positional_is_only_ever_the_payload_mapping() -> None:
    client = _local_client(provider="ollama", model="llama3", provider_instance=_FakeOllamaProvider(model="llama3"))

    with pytest.raises(TypeError):
        client.lock_model_residency("ollama")  # a bare string is not a payload
    with pytest.raises(TypeError):
        client.get_context_estimate(["mlx", "qwen"])


# ---------------------------------------------------------------------------
# Remote wire shapes
# ---------------------------------------------------------------------------


def test_remote_lock_and_unlock_post_core_endpoints_with_selector_bodies() -> None:
    sender = _LockRecordingSender(
        post_response={
            "ok": True,
            "locked": True,
            "runtime_id": "rid-1",
            "provider": "ollama",
            "model": "qwen3:4b",
            "provider_side": {"supported": True, "applied": True},
        }
    )
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test/v1",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    locked = client.lock_model_residency(
        provider="ollama",
        model="qwen3:4b",
        base_url="http://provider.test/v1",
        provider_api_key="sekret",
    )
    unlocked = client.unlock_model_residency({"runtime_id": "rid-1"})

    assert sender.calls[0]["url"] == "http://endpoint.test/acore/models/lock"
    assert sender.calls[0]["json"] == {
        "provider": "ollama",
        "model": "qwen3:4b",
        "base_url": "http://provider.test/v1",
    }
    assert sender.calls[0]["headers"]["X-AbstractCore-Provider-API-Key"] == "sekret"
    assert sender.calls[1]["url"] == "http://endpoint.test/acore/models/unlock"
    assert sender.calls[1]["json"] == {"runtime_id": "rid-1"}
    assert locked["locked"] is True
    assert locked["provider_side"] == {"supported": True, "applied": True}
    assert locked["operation"] == "lock"
    assert locked["success"] is True
    assert unlocked["operation"] == "unlock"


def test_remote_lock_merges_kwargs_over_payload() -> None:
    sender = _LockRecordingSender(post_response={"ok": True, "locked": True})
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    client.lock_model_residency({"provider": "ollama", "model": "stale"}, model="qwen3:4b")

    assert sender.calls[0]["json"] == {"provider": "ollama", "model": "qwen3:4b"}


def test_remote_context_estimate_gets_query_params_and_relays_verbatim() -> None:
    estimate = {
        "ok": True,
        "provider": "huggingface",
        "model": "qwen3.gguf",
        "confidence": "calibrated",
        "calibrated_context_length": 16384,
        "predicted_max_context": 16384,
        "notes": ["calibrated entry wins"],
    }
    sender = _LockRecordingSender(get_response=estimate)
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test/v1",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    result = client.get_context_estimate(provider="huggingface", model="qwen3.gguf", context_length=32768)

    assert sender.calls[0]["method"] == "GET"
    assert sender.calls[0]["url"] == (
        "http://endpoint.test/acore/models/context_estimate"
        "?provider=huggingface&model=qwen3.gguf&context_length=32768"
    )
    assert result == estimate


def test_remote_unload_409_is_converted_to_the_model_locked_payload() -> None:
    body = {"ok": False, "error": "model_locked", "detail": "runtime rid-1 is locked", "runtime_id": "rid-1"}
    sender = _LockRecordingSender(
        post_raises=_FakeHttpStatusError(
            "Client error '409 Conflict'",
            response=_FakeErrorResponse(status_code=409, json_body=body),
        )
    )
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    result = client.unload_model_residency(task="text_generation", runtime_id="rid-1")

    assert result["ok"] is False
    assert result["error"] == "model_locked"
    assert result["detail"] == "runtime rid-1 is locked"
    assert result["runtime_id"] == "rid-1"
    assert result["status_code"] == 409
    assert result["operation"] == "unload"
    assert result["success"] is False
    assert result["unloaded"] is False


def test_remote_unload_409_without_accessible_body_synthesizes_model_locked() -> None:
    sender = _LockRecordingSender(
        post_raises=_FakeHttpStatusError(
            "Client error '409 Conflict'",
            response=_FakeErrorResponse(status_code=409, json_body=None),
        )
    )
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    result = client.unload_model_residency(task="text_generation", provider="ollama", model="qwen3:4b")

    assert result["ok"] is False
    assert result["error"] == "model_locked"
    assert result["status_code"] == 409
    assert result["unloaded"] is False


def test_remote_unload_includes_force_in_body_only_when_true() -> None:
    sender = _LockRecordingSender(post_response={"ok": True, "unloaded": True})
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    client.unload_model_residency(task="text_generation", runtime_id="rid-1")
    client.unload_model_residency(task="text_generation", runtime_id="rid-1", force=True)
    client.unload_model_residency(task="text_generation", runtime_id="rid-1", force=False)

    assert "force" not in sender.calls[0]["json"]
    assert sender.calls[1]["json"]["force"] is True
    assert "force" not in sender.calls[2]["json"]


def test_remote_non_409_errors_keep_the_relay_error_payload() -> None:
    sender = _LockRecordingSender(
        post_raises=_FakeHttpStatusError(
            "Client error '404 Not Found'",
            response=_FakeErrorResponse(
                status_code=404,
                json_body={"error": {"message": "Loaded model runtime not found.", "type": "not_found"}},
            ),
        )
    )
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    result = client.unload_model_residency(task="text_generation", runtime_id="missing")

    assert result["ok"] is False
    assert result["status_code"] == 404
    assert result["error"] != "model_locked"
    # The client DOES implement the op — a genuine transport/server error must
    # stay distinguishable from the facade's not-implemented degradation.
    assert result["supported"] is True


# ---------------------------------------------------------------------------
# Local lock semantics
# ---------------------------------------------------------------------------


def test_local_lock_sets_flag_and_applies_the_ollama_keep_alive_knob() -> None:
    provider = _FakeOllamaProvider(model="llama3")
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    result = client.lock_model_residency()

    assert result == {
        "ok": True,
        "operation": "lock",
        "locked": True,
        "runtime_id": "local:text_generation:ollama:llama3",
        "provider": "ollama",
        "model": "llama3",
        "provider_side": {"supported": True, "applied": True},
        "diagnostics": {"source": "abstractruntime.local"},
    }
    assert provider.load_model_calls == [{"model": "llama3", "kwargs": {"keep_alive": -1}}]


def test_local_locked_pair_refuses_unload_without_force_and_survives_list_calls() -> None:
    provider = _FakeOllamaProvider(model="llama3")
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    client.lock_model_residency({"provider": "ollama", "model": "llama3"})
    refused = client.unload_model_residency(task="text_generation")

    assert refused["ok"] is False
    assert refused["error"] == "model_locked"
    assert refused["provider"] == "ollama"
    assert refused["model"] == "llama3"
    assert refused["unloaded"] is False
    assert "force" in refused["detail"]
    assert provider.unload_model_calls == []

    listed = client.list_model_residency(task="text_generation")
    record = listed["models"][0]
    assert record["locked"] is True
    assert record["lockable"] is True

    # The listing must not clear the client-side lock.
    still_refused = client.unload_model_residency(task="text_generation")
    assert still_refused["error"] == "model_locked"
    assert provider.unload_model_calls == []


def test_local_force_unload_unlocks_then_unloads() -> None:
    provider = _FakeOllamaProvider(model="llama3")
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    client.lock_model_residency()
    result = client.unload_model_residency(task="text_generation", force=True)

    assert result["ok"] is True
    assert result["unloaded"] is True
    assert provider.unload_model_calls == [{"model": "llama3", "kwargs": {}}]
    assert client.list_model_residency(task="text_generation")["models"][0]["locked"] is False


def test_local_unlock_clears_the_flag_and_restores_the_ollama_default_keep_alive() -> None:
    provider = _FakeOllamaProvider(model="llama3")
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    client.lock_model_residency()
    result = client.unlock_model_residency()

    assert result["ok"] is True
    assert result["locked"] is False
    assert result["provider_side"] == {"supported": True, "applied": True}
    assert provider.load_model_calls[-1] == {"model": "llama3", "kwargs": {"keep_alive": "5m"}}

    unloaded = client.unload_model_residency(task="text_generation")
    assert unloaded["ok"] is True
    assert provider.unload_model_calls == [{"model": "llama3", "kwargs": {}}]


def test_local_lock_on_non_ollama_provider_reports_no_provider_side_knob() -> None:
    class _LmStudioProvider(_FakeOllamaProvider):
        pass

    provider = _LmStudioProvider(model="qwen3-4b")
    client = _local_client(provider="lmstudio", model="qwen3-4b", provider_instance=provider)

    result = client.lock_model_residency()

    assert result["ok"] is True
    assert result["locked"] is True
    assert result["provider_side"] == {"supported": False, "applied": False}
    assert provider.load_model_calls == []
    assert client.unload_model_residency(task="text_generation")["error"] == "model_locked"


def test_local_lock_survives_a_failing_ollama_knob_and_reports_it() -> None:
    provider = _FakeOllamaProvider(model="llama3")
    provider.raise_on_load = True
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    result = client.lock_model_residency()

    assert result["ok"] is True
    assert result["locked"] is True
    assert result["provider_side"]["supported"] is True
    assert result["provider_side"]["applied"] is False
    assert "keep_alive=-1" in result["provider_side"]["detail"]
    # The registry-analog flag is the enforcement truth: unload still refuses.
    assert client.unload_model_residency(task="text_generation")["error"] == "model_locked"


def test_local_lock_refuses_an_unknown_pair_and_non_text_tasks() -> None:
    client = _local_client(provider="ollama", model="llama3", provider_instance=_FakeOllamaProvider(model="llama3"))

    missing = client.lock_model_residency(provider="mlx", model="other")
    media = client.lock_model_residency(task="tts", provider="omnivoice", model="supertonic-3")

    assert missing["ok"] is False
    assert "not found" in missing["error"]
    assert media["ok"] is False
    assert media["supported"] is False
    assert "text_generation" in media["error"]


def test_local_lock_refuses_a_non_resident_pair() -> None:
    """LOCK RULE (core parity): a warm pool client alone is configuration, not
    memory — locking a pair the provider cannot verify RESIDENT refuses with
    `model_not_resident` instead of presenting configured as loaded."""
    provider = _FakeOllamaProvider(model="llama3", loaded=False)
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    refused = client.lock_model_residency()

    assert refused["ok"] is False
    assert refused["error"] == "model_not_resident"
    assert "load it first" in refused["detail"]
    assert "lock:true" in refused["detail"]
    assert refused["diagnostics"]["reason"] == "model_not_resident"
    # No flag, no knob, and unload needs no force.
    assert client._locked_model_residency == set()
    assert provider.load_model_calls == []
    unloaded = client.unload_model_residency(task="text_generation")
    assert unloaded.get("error") != "model_locked"


def test_multilocal_lock_refuses_a_non_resident_pooled_pair(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")
    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    pooled_provider = client._clients[("ollama", "llama3")]._llm
    pooled_provider.loaded = False  # evicted server-side; pool client still warm

    refused = client.lock_model_residency(provider="ollama", model="llama3")

    assert refused["ok"] is False
    assert refused["error"] == "model_not_resident"
    assert client._locked_model_residency == set()
    # The keep_alive knob never fired (it rides a LOAD request).
    assert all(call["kwargs"].get("keep_alive") != -1 for call in pooled_provider.load_model_calls)


def test_local_unlock_of_an_evicted_locked_pair_never_loads_it_back() -> None:
    """Unlock must reach a locked-but-since-evicted pair (no stranded locks)
    WITHOUT the ollama keep_alive restore re-loading the evicted model."""
    provider = _FakeOllamaProvider(model="llama3", loaded=True)
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)
    client.lock_model_residency()
    calls_after_lock = list(provider.load_model_calls)

    provider.loaded = False  # server evicted the model behind the lock
    unlocked = client.unlock_model_residency()

    assert unlocked["ok"] is True
    assert unlocked["locked"] is False
    assert unlocked["provider_side"]["supported"] is True
    assert unlocked["provider_side"]["applied"] is False
    assert "not resident" in unlocked["provider_side"]["detail"]
    assert provider.load_model_calls == calls_after_lock  # no load side effect


def test_local_unlock_with_unverifiable_residency_skips_restore_with_honest_detail() -> None:
    """UNKNOWN residency is not evidence of eviction: a transient probe
    failure still skips the keep_alive restore (the safe act — the knob rides
    a load request), but the detail says "unverified", never "not resident"."""
    provider = _FakeOllamaProvider(model="llama3", loaded=True)
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)
    client.lock_model_residency()
    calls_after_lock = list(provider.load_model_calls)

    def _boom(**kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("probe transport failed")

    provider.get_model_residency = _boom  # type: ignore[method-assign]
    unlocked = client.unlock_model_residency()

    assert unlocked["ok"] is True
    assert unlocked["locked"] is False
    assert unlocked["provider_side"]["supported"] is True
    assert unlocked["provider_side"]["applied"] is False
    assert "unverified" in unlocked["provider_side"]["detail"]
    assert "not resident" not in unlocked["provider_side"]["detail"]
    assert provider.load_model_calls == calls_after_lock  # no load side effect


def test_local_rows_pinned_is_a_truthful_alias_of_locked_never_the_default_flag() -> None:
    """The operator defect: DEFAULT models presented as pinned/loaded. `pinned`
    must equal `locked` (the enforcement truth); `default` alone carries the
    default-identity pair."""
    provider = _FakeOllamaProvider(model="llama3", loaded=True)
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    record = client.list_model_residency(task="text_generation")["models"][0]
    assert record["default"] is True
    assert record["locked"] is False
    assert record["pinned"] is False  # default != pinned: nothing is locked

    client.lock_model_residency()
    locked_record = client.list_model_residency(task="text_generation")["models"][0]
    assert locked_record["locked"] is True
    assert locked_record["pinned"] is True  # alias tracks the lock, not the default
    assert locked_record["default"] is True


def test_reload_of_a_locked_pair_reports_locked_and_pinned_on_the_runtime_record() -> None:
    """Alias parity on LOAD/UNLOAD response records too: a warm re-load of a
    LOCKED pair must not answer `runtime.pinned: false` beside a live lock."""
    provider = _FakeOllamaProvider(model="llama3", loaded=True)
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)
    client.lock_model_residency()

    result = client.load_model_residency(task="text_generation")  # warm no-op re-load

    runtime = result["runtime"]
    assert runtime["locked"] is True
    assert runtime["pinned"] is True
    assert runtime["lockable"] is True
    assert runtime["default"] is True

    forced = client.unload_model_residency(task="text_generation", force=True)
    assert forced["runtime"]["locked"] is False
    assert forced["runtime"]["pinned"] is False


def test_multilocal_reload_of_a_locked_pair_reports_locked_on_the_runtime_record(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")
    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    client.lock_model_residency(provider="ollama", model="llama3")

    result = client.load_model_residency(task="text_generation", provider="ollama", model="llama3")

    assert result["runtime"]["locked"] is True
    assert result["runtime"]["pinned"] is True
    assert result["runtime"]["lockable"] is True


def test_multilocal_rows_pinned_tracks_lock_state_per_pair(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")
    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    client.lock_model_residency(provider="ollama", model="llama3")

    by_model = {record["model"]: record for record in client.list_model_residency(task="text_generation")["models"]}

    assert by_model["llama3"]["locked"] is True and by_model["llama3"]["pinned"] is True
    assert by_model["llama3"]["default"] is False
    assert by_model["default"]["locked"] is False and by_model["default"]["pinned"] is False
    assert by_model["default"]["default"] is True


def test_remote_lock_409_is_converted_to_the_model_not_resident_payload() -> None:
    """Core's lock route refuses non-resident models with HTTP 409; the remote
    client relays the envelope as a payload (same idiom as the unload 409)."""
    body = {
        "ok": False,
        "error": "model_not_resident",
        "detail": "Model lmstudio/qwen is not resident in provider memory; load it first (POST /acore/models/load with lock:true) before locking.",
        "runtime_id": "rid-1",
    }
    error = _FakeHttpStatusError(
        "Client error '409 Conflict'",
        response=_FakeErrorResponse(status_code=409, json_body=body),
    )
    sender = _LockRecordingSender(post_raises=error)
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test",
        model="lmstudio/qwen",
        request_sender=sender,
    )

    result = client.lock_model_residency({"provider": "lmstudio", "model": "qwen"})

    assert result["ok"] is False
    assert result["error"] == "model_not_resident"
    assert result["status_code"] == 409
    assert result["locked"] is False
    assert result["runtime_id"] == "rid-1"
    assert "lock:true" in result["detail"]
    assert result["operation"] == "lock"
    assert result["success"] is False


def test_context_estimate_is_advisory_no_local_load_path_references_it() -> None:
    """The estimator is a HINT: no load/unload path in the runtime clients
    gates on it (asserted at the source level, both client classes)."""
    import inspect

    for method in (
        LocalAbstractCoreLLMClient.load_model_residency,
        LocalAbstractCoreLLMClient.unload_model_residency,
        MultiLocalAbstractCoreLLMClient.load_model_residency,
        MultiLocalAbstractCoreLLMClient.unload_model_residency,
    ):
        src = inspect.getsource(method)
        assert "estimate_context_fit" not in src
        assert "_local_context_estimate" not in src
        assert "get_context_estimate" not in src


def test_local_lock_kwargs_win_over_payload_fields() -> None:
    provider = _FakeOllamaProvider(model="llama3")
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    result = client.lock_model_residency({"provider": "ollama", "model": "stale"}, model="llama3")

    assert result["ok"] is True
    assert result["model"] == "llama3"


class _DummyLocal:
    def __init__(self, *, provider: str, model: str, llm_kwargs: Dict[str, Any], artifact_store: Any) -> None:
        _ = llm_kwargs, artifact_store
        self._provider = provider
        self._model = model
        self._llm = _FakeOllamaProvider(model=model)

    def get_model_capabilities(self) -> Dict[str, Any]:
        return {}


def test_multilocal_lock_enforces_unload_refusal_and_force_on_pooled_pairs(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")
    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    pooled_provider = client._clients[("ollama", "llama3")]._llm

    locked = client.lock_model_residency(provider="ollama", model="llama3")
    refused = client.unload_model_residency(task="text_generation", provider="ollama", model="llama3")
    refused_by_runtime_id = client.unload_model_residency(
        task="text_generation",
        runtime_id="local:text_generation:ollama:llama3",
    )
    listed = client.list_model_residency(task="text_generation")
    forced = client.unload_model_residency(task="text_generation", provider="ollama", model="llama3", force=True)

    assert locked["ok"] is True
    assert locked["provider_side"] == {"supported": True, "applied": True}
    assert pooled_provider.load_model_calls[-1] == {"model": "llama3", "kwargs": {"keep_alive": -1}}
    assert refused["error"] == "model_locked"
    assert refused_by_runtime_id["error"] == "model_locked"
    by_model = {record["model"]: record for record in listed["models"]}
    assert by_model["llama3"]["locked"] is True
    assert by_model["llama3"]["lockable"] is True
    assert by_model["default"]["locked"] is False
    assert by_model["default"]["lockable"] is True
    assert forced["ok"] is True
    assert forced["unloaded"] is True
    assert pooled_provider.unload_model_calls == [{"model": "llama3", "kwargs": {}}]


def test_multilocal_lock_requires_a_warm_pair_but_unlock_reaches_locked_evicted_pairs(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")

    missing = client.lock_model_residency(provider="ollama", model="never-loaded")
    assert missing["ok"] is False
    assert "not found" in missing["error"]

    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    client.lock_model_residency(provider="ollama", model="llama3")
    # Simulate pool eviction of a locked pair: the unlock must still reach it.
    client._clients.pop(("ollama", "llama3"), None)
    unlocked = client.unlock_model_residency(provider="ollama", model="llama3")

    assert unlocked["ok"] is True
    assert unlocked["locked"] is False
    # The knob has no provider instance to reach; best-effort is reported.
    assert unlocked["provider_side"]["supported"] is True
    assert unlocked["provider_side"]["applied"] is False


# ---------------------------------------------------------------------------
# Effect ops
# ---------------------------------------------------------------------------


class _LockControl:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any, Dict[str, Any]]] = []

    def lock_model_residency(self, payload: Any = None, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("lock", payload, dict(kwargs)))
        return {
            "ok": True,
            "locked": True,
            "runtime_id": "rid-1",
            "provider": "ollama",
            "model": "llama3",
            "provider_side": {"supported": True, "applied": True},
        }

    def unlock_model_residency(self, payload: Any = None, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("unlock", payload, dict(kwargs)))
        return {
            "ok": True,
            "locked": False,
            "runtime_id": "rid-1",
            "provider": "ollama",
            "model": "llama3",
            "provider_side": {"supported": True, "applied": True},
        }


class _ForceRecordingControl:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def unload_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"ok": True, "operation": "unload", "unloaded": True}


class _LockedRefusalControl:
    def unload_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return {
            "ok": False,
            "operation": "unload",
            "error": "model_locked",
            "detail": "runtime rid-1 is locked",
            "runtime_id": "rid-1",
            "unloaded": False,
        }


class _NoLockControl:
    pass


def _run_residency_workflow(control: Any, payload: Dict[str, Any]) -> RunState:
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.MODEL_RESIDENCY: make_model_residency_handler(control=control)},
    )

    def call(run: RunState, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="call",
            effect=Effect(type=EffectType.MODEL_RESIDENCY, payload=payload, result_key="residency"),
            next_node="done",
        )

    def done(run: RunState, ctx: Any) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"residency": run.vars.get("residency")})

    workflow = WorkflowSpec("wf_model_residency_lock", "call", {"call": call, "done": done})
    run_id = runtime.start(workflow=workflow)
    return runtime.tick(workflow=workflow, run_id=run_id)


def test_effect_lock_and_unlock_ops_route_to_the_client_lock_methods() -> None:
    control = _LockControl()
    locked_state = _run_residency_workflow(
        control,
        {"operation": "lock", "provider": "ollama", "model": "llama3"},
    )
    unlocked_state = _run_residency_workflow(
        control,
        {"operation": "unlock", "runtime_id": "rid-1"},
    )

    assert locked_state.status == RunStatus.COMPLETED
    locked = locked_state.output["residency"]
    assert locked["ok"] is True
    assert locked["locked"] is True
    assert locked["operation"] == "lock"
    assert locked["success"] is True
    assert locked["affected_models"] == []

    assert unlocked_state.status == RunStatus.COMPLETED
    unlocked = unlocked_state.output["residency"]
    assert unlocked["locked"] is False
    assert unlocked["operation"] == "unlock"

    assert control.calls == [
        ("lock", None, {"provider": "ollama", "model": "llama3"}),
        ("unlock", None, {"runtime_id": "rid-1"}),
    ]


def test_effect_lock_without_client_support_soft_fails_unless_required() -> None:
    soft_state = _run_residency_workflow(
        _NoLockControl(),
        {"operation": "lock", "provider": "ollama", "model": "llama3", "required": False},
    )
    hard_state = _run_residency_workflow(
        _NoLockControl(),
        {"operation": "lock", "provider": "ollama", "model": "llama3", "required": True},
    )

    assert soft_state.status == RunStatus.COMPLETED
    result = soft_state.output["residency"]
    assert result["ok"] is False
    assert result["status_hint"] == "warning"
    assert result["degraded"] is True
    assert "does not expose" in result["error"]
    assert hard_state.status == RunStatus.FAILED


def test_effect_unload_passes_force_only_when_authored() -> None:
    control = _ForceRecordingControl()
    _run_residency_workflow(control, {"operation": "unload", "runtime_id": "rid-1"})
    _run_residency_workflow(control, {"operation": "unload", "runtime_id": "rid-1", "force": True})

    assert "force" not in control.calls[0]
    assert control.calls[1]["force"] is True


def test_effect_unload_model_locked_refusal_keeps_soft_fail_semantics() -> None:
    soft_state = _run_residency_workflow(
        _LockedRefusalControl(),
        {"operation": "unload", "runtime_id": "rid-1", "required": False},
    )
    hard_state = _run_residency_workflow(
        _LockedRefusalControl(),
        {"operation": "unload", "runtime_id": "rid-1", "required": True},
    )

    assert soft_state.status == RunStatus.COMPLETED
    result = soft_state.output["residency"]
    assert result["ok"] is False
    assert result["error"] == "model_locked"
    assert result["status_hint"] == "warning"
    assert result["degraded"] is True
    assert hard_state.status == RunStatus.FAILED
    assert "model_locked" in str(hard_state.error)


# ---------------------------------------------------------------------------
# Field passthrough (claims + sweep merge)
# ---------------------------------------------------------------------------


def _install_fake_stamp_helpers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    modalities_by_model: Optional[Dict[str, List[str]]] = None,
    host: Optional[Dict[str, Any]] = None,
) -> None:
    """Hermetic stand-ins for core's stamping truth sources (the real sibling
    registry answers for real model names, which would make these assertions
    depend on its catalog content)."""
    table = {str(k): list(v) for k, v in dict(modalities_by_model or {}).items()}
    fake_caps = types.ModuleType("abstractcore.providers.model_capabilities")
    fake_caps.modalities_for_model = lambda model: table.get(str(model))
    monkeypatch.setitem(sys.modules, "abstractcore.providers.model_capabilities", fake_caps)
    identity = dict(host or {"host_id": "feedfacecafe", "host_name": "hermetic.local", "kind": "local"})
    fake_hostinfo = types.ModuleType("abstractcore.utils.hostinfo")
    fake_hostinfo.get_host_identity = lambda: dict(identity)
    monkeypatch.setitem(sys.modules, "abstractcore.utils.hostinfo", fake_hostinfo)


def test_new_core_record_fields_pass_through_the_local_claim_blocked_set() -> None:
    class _RichClaimProvider:
        def get_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
            _ = kwargs
            return {
                "provider_residency_verified": True,
                "provider_resident": True,
                "state": "loaded",
                "source": "abstractcore.provider.test",
                "locked": True,
                "lockable": False,
                "locked_at": 123.0,
                "modalities": ["input.text", "output.text"],
                "calibrated_context_length": 16384,
                "context_calibrated": True,
                "host_id": "abc123def456",
                "host_name": "studio.local",
                "expires_at": "2026-08-27T00:00:00Z",
            }

    claim = _local_provider_residency_claim(
        provider="huggingface",
        model="qwen3.gguf",
        provider_instance=_RichClaimProvider(),
    )

    # Lock state is runtime-owned enforcement truth: claims are BLOCKED from
    # supplying it (mirrors core's post-fix claim-blocked set).
    assert "locked" not in claim
    assert "lockable" not in claim
    assert "locked_at" not in claim
    # Core-owned truth fields flow through.
    assert claim["modalities"] == ["input.text", "output.text"]
    assert claim["calibrated_context_length"] == 16384
    assert claim["context_calibrated"] is True
    assert claim["host_id"] == "abc123def456"
    assert claim["host_name"] == "studio.local"
    assert claim["expires_at"] == "2026-08-27T00:00:00Z"


def test_calibration_and_identity_claim_fields_reach_local_records(monkeypatch) -> None:
    # Pin a registry MISS so the runtime's own modalities stamp stays silent
    # and the CLAIM-relayed value survives (registry hits overwrite, like core).
    _install_fake_stamp_helpers(monkeypatch, modalities_by_model={})

    class _CalibratedProvider(_FakeOllamaProvider):
        def get_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
            claim = super().get_model_residency(**kwargs)
            claim["context_calibrated"] = True
            claim["calibrated_context_length"] = 8192
            claim["modalities"] = ["input.text", "output.text"]
            claim["host_id"] = "abc123def456"
            claim["host_name"] = "studio.local"
            # A claim can never flip managed-row lock state.
            claim["locked"] = True
            claim["lockable"] = False
            claim["locked_at"] = 123.0
            return claim

    client = _local_client(
        provider="ollama",
        model="llama3",
        provider_instance=_CalibratedProvider(model="llama3"),
    )

    record = client.list_model_residency(task="text_generation")["models"][0]

    assert record["context_calibrated"] is True
    assert record["calibrated_context_length"] == 8192
    assert record["modalities"] == ["input.text", "output.text"]
    assert record["host_id"] == "abc123def456"
    assert record["host_name"] == "studio.local"
    # Managed rows carry the CLIENT-side lock truth, not a claim's.
    assert record["locked"] is False
    assert record["lockable"] is True
    assert "locked_at" not in record


def test_sweep_only_rows_are_stamped_not_lockable_and_relay_new_fields_verbatim(monkeypatch) -> None:
    # Registry MISS pinned: sweep-relayed modalities survive verbatim (a hit
    # would overwrite them with registry truth — core's own sweep behavior).
    _install_fake_stamp_helpers(
        monkeypatch,
        modalities_by_model={},
        host={"host_id": "feedfacecafe", "host_name": "hermetic.local", "kind": "local"},
    )
    monkeypatch.setattr(
        llm_mod,
        "_sweep_host_loaded_models",
        lambda: [
            {
                "provider": "ollama",
                "model": "granite4:small",
                "resident": True,
                "loaded": True,
                "modalities": ["input.text", "output.text"],
                "host_id": "abc123def456",
                "host_name": "studio.local",
                "expires_at": "2026-08-27T00:00:00Z",
            },
            {
                "provider": "ollama",
                "model": "qwen3:latest",
                "resident": True,
                "loaded": True,
                "lockable": True,  # an explicit sweep value is relayed, not clobbered
            },
        ],
    )
    client = _local_client(
        provider="lmstudio",
        model="unrelated",
        provider_instance=_FakeOllamaProvider(model="unrelated"),
    )

    result = client.list_model_residency(task="text_generation", provider="ollama")

    by_model = {record["model"]: record for record in result["models"]}
    granite = by_model["granite4:small"]
    assert granite["source"] == "provider_server"
    assert granite["lockable"] is False  # stamped when absent, matching core
    assert granite["modalities"] == ["input.text", "output.text"]
    assert granite["host_id"] == "abc123def456"
    assert granite["host_name"] == "studio.local"
    assert granite["expires_at"] == "2026-08-27T00:00:00Z"
    assert by_model["qwen3:latest"]["lockable"] is True


# ---------------------------------------------------------------------------
# Local-lane modalities + host identity stamps (E2E fix: gateway LOCAL mode)
# ---------------------------------------------------------------------------


def test_local_managed_rows_carry_core_modalities_and_host_identity(monkeypatch) -> None:
    _install_fake_stamp_helpers(
        monkeypatch,
        modalities_by_model={"llama3": ["input.text", "input.image", "input.video", "output.text"]},
    )
    client = _local_client(provider="ollama", model="llama3", provider_instance=_FakeOllamaProvider(model="llama3"))

    record = client.list_model_residency(task="text_generation")["models"][0]

    assert record["modalities"] == ["input.text", "input.image", "input.video", "output.text"]
    assert "modalities_note" not in record
    assert record["host_id"] == "feedfacecafe"
    assert record["host_name"] == "hermetic.local"


def test_registry_miss_omits_modalities_but_still_stamps_host_identity(monkeypatch) -> None:
    _install_fake_stamp_helpers(monkeypatch, modalities_by_model={})
    client = _local_client(provider="ollama", model="llama3", provider_instance=_FakeOllamaProvider(model="llama3"))

    record = client.list_model_residency(task="text_generation")["models"][0]

    assert "modalities" not in record  # never the text-only guess
    assert record["host_id"] == "feedfacecafe"
    assert record["host_name"] == "hermetic.local"


def test_vision_unusable_provider_strips_input_image_with_note(monkeypatch) -> None:
    _install_fake_stamp_helpers(
        monkeypatch,
        modalities_by_model={"llama3": ["input.text", "input.image", "output.text"]},
    )
    provider = _FakeOllamaProvider(model="llama3")
    provider._vision_usable = False
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    record = client.list_model_residency(task="text_generation")["models"][0]

    assert record["modalities"] == ["input.text", "output.text"]
    assert record["modalities_note"] == "vision_unusable"


def test_hostile_claim_cannot_override_registry_modalities_on_managed_rows(monkeypatch) -> None:
    _install_fake_stamp_helpers(
        monkeypatch,
        modalities_by_model={"llama3": ["input.text", "output.text"]},
    )

    class _HostileClaimProvider(_FakeOllamaProvider):
        def get_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
            claim = super().get_model_residency(**kwargs)
            claim["modalities"] = ["input.text", "input.image", "input.audio", "output.text"]
            return claim

    client = _local_client(
        provider="ollama",
        model="llama3",
        provider_instance=_HostileClaimProvider(model="llama3"),
    )

    record = client.list_model_residency(task="text_generation")["models"][0]

    # The registry hit is stamped AFTER the claim merge (core's ordering):
    # a claim cannot inflate the modalities of a managed row.
    assert record["modalities"] == ["input.text", "output.text"]


def test_sweep_only_rows_get_modalities_and_host_identity_stamps(monkeypatch) -> None:
    _install_fake_stamp_helpers(
        monkeypatch,
        modalities_by_model={"granite4:small": ["input.text", "output.text"]},
    )
    monkeypatch.setattr(
        llm_mod,
        "_sweep_host_loaded_models",
        lambda: [
            {"provider": "ollama", "model": "granite4:small", "resident": True, "loaded": True},
            {
                "provider": "ollama",
                "model": "qwen3:latest",
                "resident": True,
                "loaded": True,
                "host_id": "remote-host-11",  # already attributed: setdefault keeps it
                "host_name": "elsewhere.local",
            },
        ],
    )
    client = _local_client(
        provider="lmstudio",
        model="unrelated",
        provider_instance=_FakeOllamaProvider(model="unrelated"),
    )

    result = client.list_model_residency(task="text_generation", provider="ollama")

    by_model = {record["model"]: record for record in result["models"]}
    granite = by_model["granite4:small"]
    assert granite["modalities"] == ["input.text", "output.text"]
    assert granite["host_id"] == "feedfacecafe"
    assert granite["host_name"] == "hermetic.local"
    attributed = by_model["qwen3:latest"]
    assert attributed["host_id"] == "remote-host-11"
    assert attributed["host_name"] == "elsewhere.local"
    assert "modalities" not in attributed  # registry miss stays omitted


def test_remote_rows_are_never_stamped_with_the_client_host_identity(monkeypatch) -> None:
    # Even with the helpers importable, the remote lane must relay the
    # server's rows untouched — the server stamps its OWN identity.
    _install_fake_stamp_helpers(
        monkeypatch,
        modalities_by_model={"qwen3:4b": ["input.text", "output.text"]},
    )
    sender = _LockRecordingSender(
        get_response={
            "ok": True,
            "models": [{"task": "text_generation", "provider": "ollama", "model": "qwen3:4b", "loaded": True}],
        }
    )
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    record = client.list_model_residency(task="text_generation")["models"][0]

    assert "host_id" not in record
    assert "host_name" not in record
    assert "modalities" not in record


def test_stamp_import_failure_omits_fields_and_never_fails_the_listing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "abstractcore.providers.model_capabilities", None)
    monkeypatch.setitem(sys.modules, "abstractcore.utils.hostinfo", None)
    client = _local_client(provider="ollama", model="llama3", provider_instance=_FakeOllamaProvider(model="llama3"))

    result = client.list_model_residency(task="text_generation")

    assert result["ok"] is True
    record = result["models"][0]
    assert "modalities" not in record
    assert "host_id" not in record
    assert "host_name" not in record


# ---------------------------------------------------------------------------
# Local context estimate (guarded import)
# ---------------------------------------------------------------------------


def test_local_context_estimate_relays_the_core_estimator(monkeypatch) -> None:
    calls: List[Tuple[str, str, Dict[str, Any]]] = []
    estimate = {
        "ok": True,
        "provider": "mlx",
        "model": "qwen3-4b",
        "confidence": "estimated",
        "predicted_max_context": 65536,
        "notes": [],
    }

    fake_module = types.ModuleType("abstractcore.utils.context_estimate")

    def _fake_estimate(provider: str, model: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append((provider, model, dict(kwargs)))
        return dict(estimate)

    fake_module.estimate_context_fit = _fake_estimate
    monkeypatch.setitem(sys.modules, "abstractcore.utils.context_estimate", fake_module)

    client = _local_client(provider="mlx", model="qwen3-4b", provider_instance=object())
    defaulted = client.get_context_estimate()
    explicit = client.get_context_estimate(
        {"provider": "huggingface", "model": "stale.gguf", "context_length": "8192"},
        model="qwen3.gguf",
        base_url="http://127.0.0.1:11434",
    )

    assert defaulted == estimate
    assert explicit == estimate
    assert calls == [
        ("mlx", "qwen3-4b", {"context_length": None}),
        ("huggingface", "qwen3.gguf", {"context_length": 8192, "base_url": "http://127.0.0.1:11434"}),
    ]


def test_multilocal_context_estimate_defaults_to_the_pool_default_identity(monkeypatch) -> None:
    calls: List[Tuple[str, str, Dict[str, Any]]] = []
    fake_module = types.ModuleType("abstractcore.utils.context_estimate")

    def _fake_estimate(provider: str, model: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append((provider, model, dict(kwargs)))
        return {"ok": True, "confidence": "unknown", "predicted_max_context": None, "notes": []}

    fake_module.estimate_context_fit = _fake_estimate
    monkeypatch.setitem(sys.modules, "abstractcore.utils.context_estimate", fake_module)

    client = object.__new__(MultiLocalAbstractCoreLLMClient)
    client._default_provider = "lmstudio"
    client._default_model = "qwen/qwen3-4b"

    result = client.get_context_estimate()

    assert result["ok"] is True
    assert calls == [("lmstudio", "qwen/qwen3-4b", {"context_length": None})]


def test_local_context_estimate_without_the_core_module_degrades_to_unsupported(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "abstractcore.utils.context_estimate", None)

    client = _local_client(provider="mlx", model="qwen3-4b", provider_instance=object())
    result = client.get_context_estimate()

    assert result["ok"] is False
    assert result["supported"] is False
    assert result["operation"] == "context_estimate"
    assert "unavailable" in result["error"]


def test_local_context_estimate_requires_a_provider_and_model() -> None:
    client = object.__new__(MultiLocalAbstractCoreLLMClient)
    client._default_provider = ""
    client._default_model = ""

    result = client.get_context_estimate()

    assert result["ok"] is False
    assert "requires provider and model" in result["error"]


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_foreign_runtime_id_never_falls_back_to_the_default_pair() -> None:
    """A non-empty runtime_id that addresses nothing local must be refused —
    silently falling back would lock/unlock a runtime the caller never named."""
    provider = _FakeOllamaProvider(model="llama3")
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)

    missing_lock = client.lock_model_residency(runtime_id="rid-core-42")

    assert missing_lock["ok"] is False
    assert "not found" in missing_lock["error"]
    assert missing_lock["runtime_id"] == "rid-core-42"
    assert client.list_model_residency(task="text_generation")["models"][0]["locked"] is False
    assert provider.load_model_calls == []  # no knob fired for the default pair

    client.lock_model_residency()
    missing_unlock = client.unlock_model_residency(runtime_id="rid-core-42")

    assert missing_unlock["ok"] is False
    assert "not found" in missing_unlock["error"]
    # The default pair the caller never addressed is STILL locked.
    assert client.list_model_residency(task="text_generation")["models"][0]["locked"] is True
    assert client.unload_model_residency(task="text_generation")["error"] == "model_locked"


def test_multilocal_foreign_runtime_id_is_refused_without_touching_the_default(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")

    result = client.lock_model_residency(runtime_id="local:text_generation:broken")

    assert result["ok"] is False
    assert "not found" in result["error"]
    assert client._locked_model_residency == set()


def test_local_failed_force_unload_keeps_the_lock() -> None:
    provider = _FakeOllamaProvider(model="llama3")
    provider.raise_on_unload = True
    client = _local_client(provider="ollama", model="llama3", provider_instance=provider)
    client.lock_model_residency()

    result = client.unload_model_residency(task="text_generation", force=True)

    assert result["ok"] is False
    assert ("ollama", "llama3") in client._locked_model_residency
    refused = client.unload_model_residency(task="text_generation")
    assert refused["error"] == "model_locked"

    # Once the provider recovers, the same force unload succeeds and unlocks.
    provider.raise_on_unload = False
    recovered = client.unload_model_residency(task="text_generation", force=True)
    assert recovered["ok"] is True
    assert ("ollama", "llama3") not in client._locked_model_residency


def test_multilocal_failed_force_unload_keeps_the_lock(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")
    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    client._clients[("ollama", "llama3")]._llm.raise_on_unload = True
    client.lock_model_residency(provider="ollama", model="llama3")

    result = client.unload_model_residency(task="text_generation", provider="ollama", model="llama3", force=True)

    assert result["ok"] is False
    assert ("ollama", "llama3") in client._locked_model_residency
    refused = client.unload_model_residency(task="text_generation", provider="ollama", model="llama3")
    assert refused["error"] == "model_locked"


def test_locked_pairs_survive_the_default_repoint_pool_eviction(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")
    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    client.load_model_residency(task="text_generation", provider="ollama", model="other")
    locked_client = client._clients[("ollama", "llama3")]
    client.lock_model_residency(provider="ollama", model="llama3")

    changed = client.set_default_provider_model(provider="lmstudio", model="new-default")

    assert changed is True
    # The LOCKED pair's client instance (and its resident weights) survived.
    assert client._clients.get(("ollama", "llama3")) is locked_client
    # Unlocked pairs are still evicted.
    assert ("ollama", "other") not in client._clients
    assert ("ollama", "default") not in client._clients
    # The lock is still enforced after the repoint.
    refused = client.unload_model_residency(task="text_generation", provider="ollama", model="llama3")
    assert refused["error"] == "model_locked"


def test_locked_pair_becoming_the_new_default_is_evicted_loudly_and_unlocked(monkeypatch) -> None:
    """Keeping the stale entry would serve the NEW default identity with the
    OLD connection kwargs (the 2026-07-31 misroute) — this one pair is evicted
    and its dangling flag cleared so a re-lock binds the rebuilt client."""
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")
    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    stale_client = client._clients[("ollama", "llama3")]
    client.lock_model_residency(provider="ollama", model="llama3")

    client.set_default_provider_model(provider="ollama", model="llama3")

    assert client._clients[("ollama", "llama3")] is not stale_client  # rebuilt
    assert ("ollama", "llama3") not in client._locked_model_residency  # no dangling flag
    unrefused = client.unload_model_residency(task="text_generation", provider="ollama", model="llama3")
    assert unrefused.get("error") != "model_locked"
    # Lock rule: the unloaded model is no longer resident, so a bare re-lock
    # refuses; re-load first, then the re-lock binds the rebuilt client.
    not_resident = client.lock_model_residency(provider="ollama", model="llama3")
    assert not_resident["ok"] is False
    assert not_resident["error"] == "model_not_resident"
    client.load_model_residency(task="text_generation", provider="ollama", model="llama3")
    relocked = client.lock_model_residency(provider="ollama", model="llama3")
    assert relocked["ok"] is True
    assert relocked["locked"] is True


def test_remote_lock_and_unlock_refuse_non_text_tasks_before_posting() -> None:
    sender = _LockRecordingSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    locked = client.lock_model_residency(task="tts", provider="omnivoice", model="supertonic-3")
    unlocked = client.unlock_model_residency({"task": "image_generation", "provider": "mflux", "model": "flux"})

    assert locked["ok"] is False
    assert locked["supported"] is False
    assert "text_generation" in locked["error"]
    assert unlocked["ok"] is False
    assert "text_generation" in unlocked["error"]
    assert sender.calls == []  # nothing was relayed onto a text runtime


def test_lock_on_the_plain_pair_is_honored_for_an_override_keyed_client(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="ollama", model="default")
    client._get_client("ollama", "llama3", llm_kwargs_override={"base_url": "http://other:11434/v1"})
    override_provider = next(iter(client._override_clients.values()))._llm

    locked = client.lock_model_residency(provider="ollama", model="llama3")
    refused = client.unload_model_residency(task="text_generation", provider="ollama", model="llama3")

    assert locked["ok"] is True  # the override pair is a warm, lockable pair
    assert refused["error"] == "model_locked"
    assert override_provider.unload_model_calls == []
