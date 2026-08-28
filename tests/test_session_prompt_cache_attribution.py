"""Session attribution stamping for session-scoped prompt-cache keys.

The LLM_CALL handler rides `_prompt_cache_attribution` alongside the DERIVED
session-scoped key; clients stamp it onto the provider cache entry AFTER the
generate that creates it (`prompt_cache_update_key_meta` rejects missing
keys). Stamping is best-effort, idempotent per client instance, and never
breaks the LLM call.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
)


class _StampRecordingProvider:
    """Entries exist only after generate touched the key — mirrors core's
    `prompt_cache_update_key_meta` missing-key rejection."""

    def __init__(self) -> None:
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.key_meta_calls: List[Tuple[str, Dict[str, Any]]] = []
        self.rejected: List[str] = []
        self.generate_calls: List[Dict[str, Any]] = []

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_operation(self, operation: str) -> bool:
        _ = operation
        return False  # skip the module-preparation lane entirely

    def generate(self, **kwargs: Any) -> Dict[str, Any]:
        self.generate_calls.append(dict(kwargs))
        key = str(kwargs.get("prompt_cache_key") or "").strip()
        if key:
            self.entries.setdefault(key, {})
        return {"content": "ok"}

    def prompt_cache_update_key_meta(self, key: Any, **updates: Any) -> bool:
        key_s = str(key)
        if key_s not in self.entries:
            self.rejected.append(key_s)
            return False
        meta = self.entries[key_s]
        for k, v in updates.items():
            if v is not None:
                meta[k] = v
        self.key_meta_calls.append((key_s, {k: v for k, v in updates.items() if v is not None}))
        return True


def _local_client(provider: Any) -> LocalAbstractCoreLLMClient:
    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"  # type: ignore[attr-defined]
    client._model = "test-model"  # type: ignore[attr-defined]
    client._llm = provider  # type: ignore[attr-defined]
    client._llm_kwargs = {}  # type: ignore[attr-defined]
    client._artifact_store = None  # type: ignore[attr-defined]
    client._prompt_cache_state_lock = threading.Lock()  # type: ignore[attr-defined]
    client._prompt_cache_state = {}  # type: ignore[attr-defined]
    return client


_ATTRIBUTION = {
    "session_id": "sess-1",
    "run_id": "run-1",
    "workflow_id": "wf-1",
    "node_id": "node-a",
    "namespace": "session",
}


def test_local_stamp_lands_after_first_generate_and_merges_meta() -> None:
    provider = _StampRecordingProvider()
    client = _local_client(provider)

    result = client.generate(
        prompt="hello",
        params={"prompt_cache_key": "session:abc", "_prompt_cache_attribution": dict(_ATTRIBUTION)},
    )

    assert result["content"] == "ok"
    # Placement: the entry only exists after generate, so a pre-generate stamp
    # would have been rejected. No rejection means the stamp ran afterwards.
    assert provider.rejected == []
    assert provider.key_meta_calls == [("session:abc", dict(_ATTRIBUTION))]
    assert provider.entries["session:abc"] == dict(_ATTRIBUTION)
    # The rider never leaks into the provider generate kwargs.
    assert "_prompt_cache_attribution" not in provider.generate_calls[-1]


def test_local_stamp_repeats_per_generate_and_survives_lru_eviction() -> None:
    """No done-set: core's cache store is an LRU, so an evicted-then-recreated
    key must be re-stamped by the next generate that uses it. The meta merge
    itself is idempotent."""
    provider = _StampRecordingProvider()
    client = _local_client(provider)

    for _ in range(2):
        client.generate(
            prompt="hello",
            params={"prompt_cache_key": "session:abc", "_prompt_cache_attribution": dict(_ATTRIBUTION)},
        )
    assert len(provider.key_meta_calls) == 2
    assert provider.entries["session:abc"] == dict(_ATTRIBUTION)

    # LRU eviction drops the entry AND its meta; the next generate recreates
    # the entry and the unconditional stamp restores attribution.
    provider.entries.pop("session:abc")
    client.generate(
        prompt="hello",
        params={"prompt_cache_key": "session:abc", "_prompt_cache_attribution": dict(_ATTRIBUTION)},
    )
    assert provider.rejected == []
    assert provider.entries["session:abc"] == dict(_ATTRIBUTION)


def test_local_stamp_noop_without_attribution_or_session_id() -> None:
    provider = _StampRecordingProvider()
    client = _local_client(provider)

    client.generate(prompt="hello", params={"prompt_cache_key": "session:abc"})
    client.generate(
        prompt="hello",
        params={
            "prompt_cache_key": "session:abc",
            "_prompt_cache_attribution": {"run_id": "run-1"},
        },
    )
    client.generate(prompt="hello", params={"_prompt_cache_attribution": dict(_ATTRIBUTION)})

    assert provider.key_meta_calls == []
    assert provider.rejected == []


def test_local_stamp_failure_never_breaks_the_call() -> None:
    class _ExplodingProvider(_StampRecordingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.meta_attempts = 0

        def prompt_cache_update_key_meta(self, key: Any, **updates: Any) -> bool:
            self.meta_attempts += 1
            raise RuntimeError("meta store exploded")

    provider = _ExplodingProvider()
    client = _local_client(provider)

    for _ in range(2):
        result = client.generate(
            prompt="hello",
            params={"prompt_cache_key": "session:abc", "_prompt_cache_attribution": dict(_ATTRIBUTION)},
        )
        assert result["content"] == "ok"
    # Failures stay retryable: every generate attempts the stamp again.
    assert provider.meta_attempts == 2


def test_llm_call_handler_rides_attribution_with_the_derived_key_only() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
            _ = (prompt, messages, system_prompt, media, tools)
            self.calls.append({"params": dict(params or {})})
            return {"content": "ok"}

    llm = _CapturingLLM()
    handler = make_llm_call_handler(llm=llm)
    run = RunState.new(
        workflow_id="wf-cache",
        entry_node="node-a",
        session_id="sess-cache",
        vars={"_runtime": {"prompt_cache": True}},
    )
    run.current_node = "node-a"

    outcome = handler(run, Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello", "params": {}}), None)
    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert "prompt_cache_key" in params
    assert params["_prompt_cache_attribution"] == {
        "session_id": "sess-cache",
        "run_id": run.run_id,
        "workflow_id": "wf-cache",
        "node_id": "node-a",
        "namespace": "session",
    }

    # Explicit caller-owned keys may be shared across sessions: no rider.
    outcome = handler(
        run,
        Effect(
            type=EffectType.LLM_CALL,
            payload={"prompt": "hello", "params": {"prompt_cache_key": "explicit:key"}},
        ),
        None,
    )
    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert params["prompt_cache_key"] == "explicit:key"
    assert "_prompt_cache_attribution" not in params


class _RemoteSender:
    def __init__(
        self,
        *,
        post_responses: List[Any],
        fail_on_key_meta: bool = False,
        key_meta_status_code: int | None = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._post_responses = list(post_responses)
        self._fail_on_key_meta = fail_on_key_meta
        self._key_meta_status_code = key_meta_status_code

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers), "timeout": timeout})
        return {"ok": True}

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Any:
        self.calls.append({"method": "POST", "url": url, "headers": dict(headers), "json": dict(json), "timeout": timeout})
        if url.endswith("/acore/prompt_cache/key_meta"):
            if self._key_meta_status_code is not None:
                # httpx raise_for_status shape: the error carries the response.
                exc = RuntimeError(f"HTTP {self._key_meta_status_code}")
                exc.response = SimpleNamespace(status_code=self._key_meta_status_code)  # type: ignore[attr-defined]
                raise exc
            if self._fail_on_key_meta:
                raise RuntimeError("key_meta route unavailable")
        return self._post_responses.pop(0)


_CHAT_RESPONSE = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"total_tokens": 3},
    "model": "qwen3-4b",
}


def test_remote_stamp_posts_key_meta_with_provider_model_selector_after_each_call() -> None:
    sender = _RemoteSender(
        post_responses=[
            dict(_CHAT_RESPONSE),
            {"ok": True, "supported": True, "operation": "key_meta"},
            dict(_CHAT_RESPONSE),
            {"ok": True, "supported": True, "operation": "key_meta"},
        ]
    )
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test/v1",
        model="qwen3-4b",
        timeout_s=30,
        request_sender=sender,
    )

    result = client.generate(
        prompt="hello",
        params={
            "_provider": "mlx",
            "_model": "qwen3-4b",
            "prompt_cache_key": "session:abc",
            "_prompt_cache_attribution": dict(_ATTRIBUTION),
        },
    )

    assert result["content"] == "ok"
    assert [c["url"] for c in sender.calls] == [
        "http://core.test/v1/chat/completions",
        "http://core.test/acore/prompt_cache/key_meta",
    ]
    assert sender.calls[1]["json"] == {
        "provider": "mlx",
        "model": "qwen3-4b",
        "key": "session:abc",
        "meta": dict(_ATTRIBUTION),
    }

    # No done-set: every generate that used the derived key re-stamps (an
    # evicted-then-recreated server-side key must regain its attribution).
    client.generate(
        prompt="hello again",
        params={
            "_provider": "mlx",
            "_model": "qwen3-4b",
            "prompt_cache_key": "session:abc",
            "_prompt_cache_attribution": dict(_ATTRIBUTION),
        },
    )
    assert sum(1 for c in sender.calls if c["url"].endswith("/key_meta")) == 2


def test_remote_stamp_transient_failure_never_breaks_the_call_and_retries() -> None:
    sender = _RemoteSender(post_responses=[dict(_CHAT_RESPONSE), dict(_CHAT_RESPONSE)], fail_on_key_meta=True)
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test/v1",
        model="qwen3-4b",
        timeout_s=30,
        request_sender=sender,
    )

    for _ in range(2):
        result = client.generate(
            prompt="hello",
            params={
                "_provider": "mlx",
                "_model": "qwen3-4b",
                "prompt_cache_key": "session:abc",
                "_prompt_cache_attribution": dict(_ATTRIBUTION),
            },
        )
        assert result["content"] == "ok"

    # A transient failure (no HTTP status) is NOT negative-cached: the next
    # generate retries the stamp.
    assert client._prompt_cache_key_meta_route_unsupported is False
    assert sum(1 for c in sender.calls if c["url"].endswith("/key_meta")) == 2


def test_remote_stamp_negative_caches_a_missing_key_meta_route() -> None:
    sender = _RemoteSender(
        post_responses=[dict(_CHAT_RESPONSE), dict(_CHAT_RESPONSE)],
        key_meta_status_code=404,
    )
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test/v1",
        model="qwen3-4b",
        timeout_s=30,
        request_sender=sender,
    )

    for _ in range(2):
        result = client.generate(
            prompt="hello",
            params={
                "_provider": "mlx",
                "_model": "qwen3-4b",
                "prompt_cache_key": "session:abc",
                "_prompt_cache_attribution": dict(_ATTRIBUTION),
            },
        )
        assert result["content"] == "ok"

    # An unambiguous 404 marks the route unsupported once; no further POSTs.
    assert client._prompt_cache_key_meta_route_unsupported is True
    assert sum(1 for c in sender.calls if c["url"].endswith("/key_meta")) == 1


def test_remote_stamp_skipped_without_a_provider_qualified_model() -> None:
    sender = _RemoteSender(post_responses=[dict(_CHAT_RESPONSE)])
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test/v1",
        model="qwen3-4b",  # no provider half: no key_meta selector exists
        timeout_s=30,
        request_sender=sender,
    )

    result = client.generate(
        prompt="hello",
        params={
            "prompt_cache_key": "session:abc",
            "_prompt_cache_attribution": dict(_ATTRIBUTION),
        },
    )

    assert result["content"] == "ok"
    assert [c["url"] for c in sender.calls] == ["http://core.test/v1/chat/completions"]
