from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []
        # Minimal stand-in for the provider's in-process prompt-cache store meta.
        self.key_meta: Dict[str, Dict[str, Any]] = {}

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_key_meta(self, key: Any) -> Dict[str, Any]:
        return dict(self.key_meta.get(str(key)) or {})

    def prompt_cache_update_key_meta(self, key: Any, **updates: Any) -> bool:
        meta = self.key_meta.setdefault(str(key), {})
        for k, v in updates.items():
            if v is not None:
                meta[k] = v
        return True

    def get_prompt_cache_capabilities(self) -> Dict[str, Any]:
        return {
            "supported": True,
            "mode": "local_control_plane",
            "supports_set": True,
            "supports_clear": True,
            "supports_update": True,
            "supports_fork": True,
            "supports_prepare_modules": True,
            "supports_stats": True,
            "supports_ttl": True,
        }

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
    ) -> Dict[str, Any]:
        _ = (make_default, ttl_s, version)
        derived: List[Dict[str, Any]] = []
        prefix_seed = "seed"
        for m in modules:
            payload = {
                "module_id": m.get("module_id"),
                "system_prompt": m.get("system_prompt"),
                "tools": m.get("tools"),
                "prompt": m.get("prompt"),
                "messages": m.get("messages"),
                "add_generation_prompt": bool(m.get("add_generation_prompt")),
                "scope": m.get("scope"),
            }
            raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            module_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            prefix_seed = hashlib.sha256((prefix_seed + module_hash).encode("utf-8")).hexdigest()
            cache_key = f"{namespace}:{prefix_seed[:16]}"
            derived.append({"module_id": m.get("module_id"), "module_hash": module_hash, "cache_key": cache_key})
        self.calls.append(("prepare_modules", namespace, modules))
        return {"supported": True, "namespace": namespace, "modules": derived, "final_cache_key": derived[-1]["cache_key"]}

    def get_prompt_cache_stats(self) -> Dict[str, Any]:
        self.calls.append(("stats",))
        return {"entries": 1, "keys": ["sess:abc"]}

    def prompt_cache_clear(self, key: Optional[str] = None) -> bool:
        self.calls.append(("clear", key))
        if key is None:
            self.key_meta.clear()
        else:
            self.key_meta.pop(str(key), None)
        return True

    def prompt_cache_fork(
        self,
        from_key: str,
        to_key: str,
        *,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> bool:
        _ = (make_default, ttl_s, kwargs)
        self.calls.append(("fork", from_key, to_key))
        self.key_meta.setdefault(str(to_key), {})["forked_from"] = str(from_key)
        return True

    def prompt_cache_update(
        self,
        key: str,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> bool:
        _ = (prompt, system_prompt, tools, add_generation_prompt, kwargs)
        self.calls.append(("update", key, list(messages or [])))
        return True


class _KeyOnlyProvider:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []

    def supports_prompt_cache(self) -> bool:
        return True

    def get_prompt_cache_capabilities(self) -> Dict[str, Any]:
        return {
            "supported": True,
            "mode": "keyed",
            "supports_set": True,
            "supports_clear": True,
            "supports_update": False,
            "supports_fork": False,
            "supports_prepare_modules": False,
            "supports_stats": True,
            "supports_ttl": False,
        }

    def prompt_cache_supports_operation(self, operation: str) -> bool:
        self.calls.append(("supports_operation", operation))
        return False

    def prompt_cache_prepare_modules(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        self.calls.append(("prepare_modules",))
        raise AssertionError("prompt_cache_prepare_modules should not be called for key-only providers")


def _new_client_for_cache_tests(provider: _FakeProvider):
    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient

    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    client._llm = provider  # type: ignore[attr-defined]
    client._prompt_cache_state_lock = threading.Lock()  # type: ignore[attr-defined]
    client._prompt_cache_state = {}  # type: ignore[attr-defined]
    return client


def test_prompt_cache_forks_once_and_leaves_the_message_lane_to_generate() -> None:
    """2026-08-02: this method prepares the (system+tools) prefix and forks the session key
    ONCE. It must not append messages — `_prepare_cache_delta_feed` owns the message lane
    (token-level LCP + trim), and every append here was work the delta feed had to undo."""
    provider = _FakeProvider()
    client = _new_client_for_cache_tests(provider)

    key = "sess:abc"
    sys = "SYSTEM"
    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}]
    m1 = {"role": "user", "content": "hi"}
    m2 = {"role": "assistant", "content": "hello"}

    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key,
        system_prompt=sys,
        tools=tools,
        messages=[m1],
    )

    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]
    # SEPARATE `system` and `tools` BLOCS (restored 2026-08-03). A bloc is an
    # independently-keyed slice of one rendered conversation; `system` alone is the bloc
    # many agents and sessions share, and it must keep its own key. Merging the two
    # collapsed that abstraction to work around a RENDER bug (chat templates fold the tool
    # instructions into the single system turn, so two standalone renders emitted two
    # consecutive `<|im_start|>system` blocks). The bug is fixed where it lives, in
    # `BaseProvider.prompt_cache_plan_bloc_chain`, which cuts ONE cumulative render at
    # successor-independent token boundaries — measured reuse 46% -> 99% with the blocs
    # still separate. Do NOT re-merge these to fix a rendering problem.
    prepared_modules = provider.calls[0][2]
    assert len(prepared_modules) == 2
    assert prepared_modules[0]["module_id"] == "system"
    assert prepared_modules[0]["system_prompt"] == sys
    assert "tools" not in prepared_modules[0]
    assert prepared_modules[1]["module_id"] == "tools"
    assert prepared_modules[1]["tools"] == tools

    provider.calls.clear()
    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key,
        system_prompt=sys,
        tools=tools,
        messages=[m1, m2],
    )

    assert [c[0] for c in provider.calls] == ["prepare_modules"]


def test_prompt_cache_recognizes_a_prepared_key_from_a_fresh_client() -> None:
    """The runtime builds a fresh LLMClient per llm_call effect, so the per-instance state
    dict is empty on essentially every call. The prepared-ness check must come from the
    provider's own cache meta, or the session cache is cleared before every generate."""
    provider = _FakeProvider()
    client = _new_client_for_cache_tests(provider)

    key = "sess:abc"
    sys = "SYSTEM"

    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key, system_prompt=sys, tools=None, messages=[{"role": "user", "content": "hi"}]
    )
    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]

    provider.calls.clear()
    fresh_client = _new_client_for_cache_tests(provider)  # same provider, new client instance
    fresh_client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key, system_prompt=sys, tools=None, messages=[{"role": "user", "content": "hi"}]
    )
    assert [c[0] for c in provider.calls] == ["prepare_modules"]


def test_prompt_cache_rebuilds_on_tools_change() -> None:
    provider = _FakeProvider()
    client = _new_client_for_cache_tests(provider)

    key = "sess:abc"
    sys = "SYSTEM"
    tools1 = [{"type": "function", "function": {"name": "t1", "description": "d", "parameters": {"type": "object"}}}]
    tools2 = [{"type": "function", "function": {"name": "t2", "description": "d", "parameters": {"type": "object"}}}]
    msgs = [{"role": "user", "content": "hi"}]

    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key,
        system_prompt=sys,
        tools=tools1,
        messages=msgs,
    )

    provider.calls.clear()
    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key,
        system_prompt=sys,
        tools=tools2,
        messages=msgs,
    )

    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]


def test_prompt_cache_does_not_clear_on_history_divergence() -> None:
    """A rewritten transcript tail (loop counters, edits, truncation, a sibling sub-run
    sharing the derived key) must NOT destroy the session cache. `prompt_cache_clear` also
    drops MLX's hybrid snapshot for the key; the provider's token-level LCP+trim already
    handles a divergent tail correctly and keeps the shared prefix."""
    provider = _FakeProvider()
    client = _new_client_for_cache_tests(provider)

    key = "sess:abc"
    sys = "SYSTEM"
    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}]
    m1 = {"role": "user", "content": "hi"}
    m2 = {"role": "assistant", "content": "hello"}

    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key,
        system_prompt=sys,
        tools=tools,
        messages=[m1, m2],
    )

    provider.calls.clear()
    # Truncate history (not a prefix-extension).
    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key,
        system_prompt=sys,
        tools=tools,
        messages=[m1],
    )

    assert [c[0] for c in provider.calls] == ["prepare_modules"]
    assert "clear" not in [c[0] for c in provider.calls]


def test_prompt_cache_skips_module_preparation_for_keyed_provider() -> None:
    provider = _KeyOnlyProvider()
    client = _new_client_for_cache_tests(provider)

    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key="sess:abc",
        system_prompt="SYSTEM",
        tools=None,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert provider.calls == [("supports_operation", "prepare_modules")]


def test_local_prompt_cache_control_plane_payloads_are_structured() -> None:
    provider = _FakeProvider()
    client = _new_client_for_cache_tests(provider)

    stats = client.get_prompt_cache_stats()
    assert stats["supported"] is True
    assert stats["operation"] == "stats"
    assert stats["capabilities"]["mode"] == "local_control_plane"
    assert stats["stats"]["entries"] == 1

    prepared = client.prompt_cache_prepare_modules(
        namespace="tenant:model",
        modules=[{"module_id": "system", "system_prompt": "SYSTEM"}],
    )
    assert prepared["supported"] is True
    assert prepared["operation"] == "prepare_modules"
    assert prepared["capabilities"]["mode"] == "local_control_plane"


def test_local_prompt_cache_control_plane_reports_unsupported_for_keyed_provider() -> None:
    provider = _KeyOnlyProvider()
    client = _new_client_for_cache_tests(provider)

    prepared = client.prompt_cache_prepare_modules(
        namespace="tenant:model",
        modules=[{"module_id": "system", "system_prompt": "SYSTEM"}],
    )
    assert prepared["supported"] is False
    assert prepared["operation"] == "prepare_modules"
    assert prepared["code"] == "prompt_cache_unsupported"
    assert prepared["capabilities"]["mode"] == "keyed"


class _FakeSender:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, Dict[str, Any]]] = []

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
        _ = (headers, timeout)
        self.calls.append(("GET", url, {}))
        if url.endswith("/acore/prompt_cache/capabilities"):
            return {
                "supported": True,
                "operation": "capabilities",
                "capabilities": {"supported": True, "mode": "keyed"},
            }
        return {
            "supported": True,
            "operation": "stats",
            "capabilities": {"supported": True, "mode": "local_control_plane"},
            "stats": {"entries": 2},
        }

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        _ = (headers, timeout)
        self.calls.append(("POST", url, dict(json)))
        return {
            "supported": True,
            "operation": url.rsplit("/", 1)[-1],
            "capabilities": {"supported": True, "mode": "local_control_plane"},
            "ok": True,
        }


def test_remote_prompt_cache_control_plane_proxies_endpoint() -> None:
    from abstractruntime.integrations.abstractcore.llm_client import RemoteAbstractCoreLLMClient

    sender = _FakeSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint",
        model="stub-model",
        request_sender=sender,
    )

    caps = client.get_prompt_cache_capabilities(provider="stub", model="stub-model")
    assert caps["supported"] is True
    assert caps["capabilities"]["mode"] == "keyed"

    stats = client.get_prompt_cache_stats(provider="stub", model="stub-model")
    assert stats["supported"] is True
    assert stats["stats"]["entries"] == 2

    updated = client.prompt_cache_update(
        key="k1",
        prompt="hello",
        provider="stub",
        model="stub-model",
    )
    assert updated["supported"] is True
    assert updated["ok"] is True

    assert sender.calls == [
        ("GET", "http://endpoint/acore/prompt_cache/capabilities", {}),
        ("GET", "http://endpoint/acore/prompt_cache/stats", {}),
        ("POST", "http://endpoint/acore/prompt_cache/update", {"key": "k1", "prompt": "hello", "add_generation_prompt": False}),
    ]


def test_llm_call_derives_prompt_cache_key_from_effective_client_identity() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import (
        _derive_prompt_cache_key,
        make_llm_call_handler,
    )

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
            self.calls.append(
                {
                    "prompt": prompt,
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "media": media,
                    "tools": tools,
                    "params": dict(params or {}),
                }
            )
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

    effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello", "params": {}}, result_key="llm")
    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert params["prompt_cache_key"] == _derive_prompt_cache_key(
        namespace="session",
        session_id="sess-cache",
        provider="stub-provider",
        model="default-model",
        workflow_id="wf-cache",
        node_id="node-a",
    )

    effect_override = Effect(
        type=EffectType.LLM_CALL,
        payload={"prompt": "hello", "provider": "other-provider", "model": "other-model", "params": {}},
        result_key="llm",
    )
    outcome = handler(run, effect_override, None)

    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert params["prompt_cache_key"] == _derive_prompt_cache_key(
        namespace="session",
        session_id="sess-cache",
        provider="other-provider",
        model="other-model",
        workflow_id="wf-cache",
        node_id="node-a",
    )


def test_llm_call_does_not_derive_prompt_cache_key_for_generated_media_outputs() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
            self.calls.append({"params": dict(params or {})})
            return {"outputs": {"music": []}, "metadata": {}}

    llm = _CapturingLLM()
    handler = make_llm_call_handler(llm=llm, artifact_store=InMemoryArtifactStore())
    run = RunState.new(
        workflow_id="wf-cache",
        entry_node="node-a",
        session_id="sess-cache",
        vars={"_runtime": {"prompt_cache": True}},
    )
    run.current_node = "node-a"

    outcome = handler(
        run,
        Effect(
            type=EffectType.LLM_CALL,
            payload={"prompt": "make music", "params": {"output": {"modality": "music", "task": "music_generation"}}},
        ),
        None,
    )

    assert outcome.status == "completed"
    assert "prompt_cache_key" not in llm.calls[-1]["params"]


class _KeyCapturingLLM:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def default_prompt_cache_identity(self) -> Tuple[str, str]:
        return "stub-provider", "default-model"

    def generate(self, *, prompt, messages, system_prompt, media, tools, params):
        self.calls.append({"params": dict(params or {})})
        return {"content": "ok"}


def _run_cache_probe(monkeypatch, *, vars: Dict[str, Any], session_id: Optional[str] = "sess-cache") -> Dict[str, Any]:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    llm = _KeyCapturingLLM()
    handler = make_llm_call_handler(llm=llm)
    run = RunState.new(workflow_id="wf-cache", entry_node="node-a", session_id=session_id, vars=vars)
    run.current_node = "node-a"

    outcome = handler(run, Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello", "params": {}}), None)
    assert outcome.status == "completed"
    return llm.calls[-1]["params"]


def test_llm_call_prompt_cache_defaults_on_for_session_scoped_runs(monkeypatch) -> None:
    """Backlog 0212: caching defaults ON. The derived key is session-scoped (requires a
    session_id), so the default cannot cause cross-session reuse."""
    monkeypatch.delenv("ABSTRACTRUNTIME_PROMPT_CACHE", raising=False)
    monkeypatch.delenv("ABSTRACTGATEWAY_PROMPT_CACHE", raising=False)

    params = _run_cache_probe(monkeypatch, vars={})
    assert "prompt_cache_key" in params


def test_llm_call_prompt_cache_default_requires_session_id(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACTRUNTIME_PROMPT_CACHE", raising=False)

    params = _run_cache_probe(monkeypatch, vars={}, session_id=None)
    assert "prompt_cache_key" not in params


def test_llm_call_ignores_gateway_prompt_cache_env(monkeypatch) -> None:
    """Ownership boundary: the runtime must not read Gateway-owned env names. With caching
    default-ON, a gateway-side opt-OUT env must have no effect here (Gateway translates its
    env into `_runtime.prompt_cache` explicitly)."""
    monkeypatch.setenv("ABSTRACTGATEWAY_PROMPT_CACHE", "0")
    monkeypatch.delenv("ABSTRACTRUNTIME_PROMPT_CACHE", raising=False)

    params = _run_cache_probe(monkeypatch, vars={})
    assert "prompt_cache_key" in params


def test_llm_call_honors_runtime_prompt_cache_env(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACTRUNTIME_PROMPT_CACHE", "1")

    params = _run_cache_probe(monkeypatch, vars={})
    assert "prompt_cache_key" in params


def test_llm_call_runtime_prompt_cache_env_opt_out(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACTRUNTIME_PROMPT_CACHE", "0")

    params = _run_cache_probe(monkeypatch, vars={})
    assert "prompt_cache_key" not in params


def test_llm_call_runtime_prompt_cache_false_disables(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACTRUNTIME_PROMPT_CACHE", raising=False)

    params = _run_cache_probe(monkeypatch, vars={"_runtime": {"prompt_cache": False}})
    assert "prompt_cache_key" not in params


def test_llm_call_runtime_prompt_cache_enabled_false_dict_disables(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACTRUNTIME_PROMPT_CACHE", raising=False)

    params = _run_cache_probe(monkeypatch, vars={"_runtime": {"prompt_cache": {"enabled": False}}})
    assert "prompt_cache_key" not in params


def test_llm_call_preserves_explicit_matching_prompt_cache_key_with_binding() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
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

    binding = {"binding_id": "bind-1", "key": "bloc:orbit"}
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "params": {
                "prompt_cache_key": "bloc:orbit",
                "prompt_cache_binding": binding,
            },
        },
        result_key="llm",
    )

    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    assert llm.calls[-1]["params"]["prompt_cache_key"] == "bloc:orbit"
    assert llm.calls[-1]["params"]["prompt_cache_binding"] == binding


def test_llm_call_uses_binding_key_without_deriving_competing_prompt_cache_key() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
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

    binding = {"binding_id": "bind-1", "key": "bloc:orbit"}
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "params": {
                "prompt_cache_binding": binding,
            },
        },
        result_key="llm",
    )

    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    assert llm.calls[-1]["params"]["prompt_cache_key"] == "bloc:orbit"
    assert llm.calls[-1]["params"]["prompt_cache_binding"] == binding


def test_llm_call_keyless_binding_does_not_inject_derived_prompt_cache_key() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
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

    binding = {"binding_id": "bind-1"}
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "params": {
                "prompt_cache_binding": binding,
            },
        },
        result_key="llm",
    )

    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert params["prompt_cache_binding"] == binding
    assert "prompt_cache_key" not in params


def test_llm_call_fails_fast_when_explicit_prompt_cache_key_mismatches_binding_key() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
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

    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "params": {
                "prompt_cache_key": "bloc:other",
                "prompt_cache_binding": {"binding_id": "bind-1", "key": "bloc:orbit"},
            },
        },
        result_key="llm",
    )

    with pytest.raises(ValueError, match="prompt_cache_key and prompt_cache_binding.key must match"):
        handler(run, effect, None)

    assert llm.calls == []


def test_llm_call_normalizes_expected_prompt_cache_binding_alias() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
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

    binding = {"binding_id": "bind-2", "key": "bloc:alias"}
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "params": {
                "expected_prompt_cache_binding": binding,
            },
        },
        result_key="llm",
    )

    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert params["prompt_cache_key"] == "bloc:alias"
    assert params["prompt_cache_binding"] == binding
    assert "expected_prompt_cache_binding" not in params


def test_llm_call_routes_string_binding_to_prompt_cache_key() -> None:
    """Bloc-seam adversary A-3 (2026-07-13; agent seat c1670 convention —
    one meaning per name): a bare STRING binding is cache-key intent and
    routes to prompt_cache_key; it must NOT be coerced into a
    {"binding_id": ...} dict (the shape core refuses as
    prompt_cache_binding_bare_string — the 2026-07-11 live-visit collision).
    Dict bindings keep strict durable-bloc verification semantics."""
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class _CapturingLLM:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def default_prompt_cache_identity(self) -> Tuple[str, str]:
            return "stub-provider", "default-model"

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
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

    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "params": {"prompt_cache_binding": "session:my-visit"},
        },
        result_key="llm",
    )
    outcome = handler(run, effect, None)
    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert params["prompt_cache_key"] == "session:my-visit"
    assert "prompt_cache_binding" not in params

    # A conflicting explicit key still refuses loudly.
    effect_conflict = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "params": {
                "prompt_cache_binding": "session:my-visit",
                "prompt_cache_key": "other-key",
            },
        },
        result_key="llm",
    )
    with pytest.raises(ValueError, match="string prompt_cache_binding"):
        handler(run, effect_conflict, None)


def test_remote_params_route_string_binding_to_prompt_cache_key() -> None:
    """The remote client's params normalization mirrors the handler seam."""
    from abstractruntime.integrations.abstractcore.llm_client import (
        _normalize_prompt_cache_binding_params,
    )

    out = _normalize_prompt_cache_binding_params({"prompt_cache_binding": "session:x"})
    assert out["prompt_cache_key"] == "session:x"
    assert "prompt_cache_binding" not in out

    binding = {"binding_id": "bind-1", "key": "bloc:orbit"}
    out2 = _normalize_prompt_cache_binding_params({"prompt_cache_binding": binding})
    assert out2["prompt_cache_binding"] == binding
    assert out2["prompt_cache_key"] == "bloc:orbit"


def test_local_bloc_host_methods_use_runtime_owned_root_and_structured_payloads(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from types import SimpleNamespace

    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
    from abstractruntime.integrations.abstractcore import llm_client as llm_client_module

    class _FakeBlocRecord:
        def __init__(self, *, sha256: str, bloc_id: int, path: str, content_sha256: str) -> None:
            self.sha256 = sha256
            self.bloc_id = bloc_id
            self.path = path
            self.content_sha256 = content_sha256

        def to_dict(self) -> Dict[str, Any]:
            return {
                "sha256": self.sha256,
                "bloc_id": self.bloc_id,
                "path": self.path,
                "content_sha256": self.content_sha256,
            }

    class _FakeBlocStore:
        stores: Dict[str, Dict[str, _FakeBlocRecord]] = {}

        def __init__(self, *, root_dir) -> None:
            self.root_dir = root_dir
            self._records = self.stores.setdefault(str(root_dir), {})

        def upsert(self, *, file_meta: Dict[str, Any], content: str, relpath_base=None, summary=None, keywords=None):
            _ = (content, relpath_base, summary, keywords)
            sha = str(file_meta["sha256"])
            existing = self._records.get(sha)
            bloc_id = existing.bloc_id if existing is not None else (len(self._records) + 1)
            record = _FakeBlocRecord(
                sha256=sha,
                bloc_id=bloc_id,
                path=str(file_meta["path"]),
                content_sha256=str(file_meta["content_sha256"]),
            )
            self._records[sha] = record
            return record

        def ensure_bloc_ids(self) -> int:
            return 0

        def get(self, sha256: str):
            return self._records.get(str(sha256))

        def get_by_bloc_id(self, bloc_id: int):
            for record in self._records.values():
                if record.bloc_id == int(bloc_id):
                    return record
            return None

    class _FakeManifest:
        def __init__(self, *, bloc_id: int) -> None:
            self.bloc_id = bloc_id

        def to_dict(self) -> Dict[str, Any]:
            return {"binding_id": f"bind-{self.bloc_id}", "bloc_id": self.bloc_id}

    seen: Dict[str, Any] = {}

    def _fake_ensure_bloc_kv_artifact(*, provider, store, model, record, artifact_path=None, force_rebuild=False, debug=False):
        seen["ensure"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "model": model,
            "sha256": record.sha256,
            "artifact_path": artifact_path,
            "force_rebuild": force_rebuild,
            "debug": debug,
        }
        manifest = _FakeManifest(bloc_id=record.bloc_id)
        return SimpleNamespace(
            artifact_path=store.root_dir / "kv" / f"{record.sha256}.artifact",
            manifest_path=store.root_dir / "kv" / f"{record.sha256}.manifest.json",
            manifest=manifest,
            compiled=True,
            rebuilt=False,
            source_cache_key="tmp:bloc",
            binding_id=f"bind-{record.bloc_id}",
            prompt_cache_binding={"binding_id": f"bind-{record.bloc_id}", "key": "stable:orbit"},
            debug={"phase": "ensure"} if debug else None,
        )

    def _fake_load_bloc_kv_artifact(*, provider, store, model, record, artifact_path=None, stable_cache_key=None, key=None, make_default=False, force_rebuild=False, debug=False):
        seen["load"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "model": model,
            "sha256": record.sha256,
            "stable_cache_key": stable_cache_key,
            "key": key,
            "make_default": make_default,
            "force_rebuild": force_rebuild,
            "debug": debug,
        }
        manifest = _FakeManifest(bloc_id=record.bloc_id)
        return SimpleNamespace(
            artifact_path=store.root_dir / "kv" / f"{record.sha256}.artifact",
            manifest_path=store.root_dir / "kv" / f"{record.sha256}.manifest.json",
            manifest=manifest,
            key=key or "work:orbit",
            stable_cache_key=stable_cache_key,
            compiled=False,
            loaded=True,
            reloaded_stable_key=bool(stable_cache_key),
            forked_from=stable_cache_key if stable_cache_key and key and key != stable_cache_key else None,
            binding_id=f"bind-{record.bloc_id}",
            prompt_cache_binding={"binding_id": f"bind-{record.bloc_id}", "key": key or "work:orbit"},
            debug={"phase": "load"} if debug else None,
        )

    def _fake_read_bloc_kv_manifest(*, provider, store, model, record, artifact_path=None):
        seen["manifest"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "model": model,
            "sha256": record.sha256,
            "artifact_path": artifact_path,
        }
        return _FakeManifest(bloc_id=record.bloc_id)

    def _fake_list_bloc_kv_artifacts(*, store, sha256=None, bloc_id=None, provider=None, model=None):
        seen["kv_list"] = {
            "root_dir": str(store.root_dir),
            "sha256": sha256,
            "bloc_id": bloc_id,
            "provider": provider,
            "model": model,
        }
        record = store.get(sha256) if isinstance(sha256, str) and sha256 else store.get_by_bloc_id(bloc_id)
        if record is None:
            return []
        return [
            {
                "artifact_path": str(store.root_dir / "kv" / f"{record.sha256}.artifact"),
                "manifest_path": str(store.root_dir / "kv" / f"{record.sha256}.manifest.json"),
                "provider": provider or "mlx",
                "model": model or "mlx-community/Qwen3-4B",
                "manifest": {"bloc_sha256": record.sha256, "provider": provider or "mlx", "model": model or "mlx-community/Qwen3-4B"},
            }
        ]

    def _fake_delete_bloc_kv_artifact(
        *,
        provider,
        store,
        sha256=None,
        bloc_id=None,
        provider_name=None,
        model=None,
        artifact_path=None,
        clear_loaded=False,
        force=False,
        dry_run=False,
        debug=False,
    ):
        seen["kv_delete"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "sha256": sha256,
            "bloc_id": bloc_id,
            "provider_name": provider_name,
            "model": model,
            "artifact_path": artifact_path,
            "clear_loaded": clear_loaded,
            "force": force,
            "dry_run": dry_run,
            "debug": debug,
        }
        return SimpleNamespace(
            to_dict=lambda: {
                "operation": "kv_delete",
                "deleted": not dry_run,
                "dry_run": dry_run,
                "artifact_path": artifact_path,
                "live_bindings": [],
            }
        )

    def _fake_prune_bloc_kv_artifacts(
        *,
        provider,
        store,
        sha256=None,
        bloc_id=None,
        provider_name=None,
        model=None,
        clear_loaded=False,
        force=False,
        dry_run=False,
        debug=False,
    ):
        seen["kv_prune"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "sha256": sha256,
            "bloc_id": bloc_id,
            "provider_name": provider_name,
            "model": model,
            "clear_loaded": clear_loaded,
            "force": force,
            "dry_run": dry_run,
            "debug": debug,
        }
        return [
            SimpleNamespace(
                to_dict=lambda: {
                    "operation": "kv_delete",
                    "deleted": not dry_run,
                    "dry_run": dry_run,
                    "artifact_path": str(store.root_dir / "kv" / "pruned.artifact"),
                    "live_bindings": [],
                }
            )
        ]

    def _fake_delete_bloc(*, provider, store, sha256=None, bloc_id=None, delete_kv=True, clear_loaded=False, force=False, dry_run=False):
        seen["bloc_delete"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "sha256": sha256,
            "bloc_id": bloc_id,
            "delete_kv": delete_kv,
            "clear_loaded": clear_loaded,
            "force": force,
            "dry_run": dry_run,
        }
        record = store.get(sha256) if isinstance(sha256, str) and sha256 else store.get_by_bloc_id(bloc_id)
        return SimpleNamespace(
            to_dict=lambda: {
                "operation": "bloc_delete",
                "deleted": not dry_run,
                "dry_run": dry_run,
                "record": record.to_dict() if record is not None else None,
                "kv_results": [],
                "live_bindings": [],
            }
        )

    monkeypatch.setattr(
        llm_client_module,
        "_load_abstractcore_bloc_api",
        lambda: {
            "FileBlocStore": _FakeBlocStore,
            "ensure_bloc_kv_artifact": _fake_ensure_bloc_kv_artifact,
            "load_bloc_kv_artifact": _fake_load_bloc_kv_artifact,
            "read_bloc_kv_manifest": _fake_read_bloc_kv_manifest,
            "list_bloc_kv_artifacts": _fake_list_bloc_kv_artifacts,
            "find_bloc_kv_live_bindings": lambda **kwargs: [],
            "delete_bloc_kv_artifact": _fake_delete_bloc_kv_artifact,
            "prune_bloc_kv_artifacts": _fake_prune_bloc_kv_artifacts,
            "delete_bloc": _fake_delete_bloc,
        },
    )

    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    client._llm = object()  # type: ignore[attr-defined]
    client._provider = "mlx"  # type: ignore[attr-defined]
    client._model = "mlx-community/Qwen3-4B"  # type: ignore[attr-defined]
    client._bloc_root_dir = tmp_path / "runtime-blocs"  # type: ignore[attr-defined]

    upserted = client.upsert_text_bloc(path="notes/orbit.txt", content="Orbit notes")
    record = client.get_bloc_record(bloc_id=1)
    listed = client.list_blocs(bloc_id=1)
    manifest = client.get_bloc_kv_manifest(bloc_id=1)
    ensured = client.ensure_bloc_kv_artifact(bloc_id=1, debug=True)
    loaded = client.load_bloc_kv_artifact(bloc_id=1, key="work:orbit", stable_cache_key="stable:orbit", debug=True)
    artifacts = client.list_bloc_kv_artifacts(bloc_id=1)
    deleted_artifact = client.delete_bloc_kv_artifact(
        bloc_id=1,
        artifact_path=str(artifacts["artifacts"][0]["artifact_path"]),
        clear_loaded=True,
        dry_run=True,
        debug=True,
    )
    pruned = client.prune_bloc_kv_artifacts(bloc_id=1, force=True, debug=True)
    deleted_bloc = client.delete_bloc(bloc_id=1, clear_loaded=True, dry_run=True)

    assert upserted["record"]["bloc_id"] == 1
    assert record["record"]["sha256"] == upserted["record"]["sha256"]
    assert listed["records"][0]["bloc_id"] == 1
    assert manifest["manifest"]["binding_id"] == "bind-1"
    assert ensured["artifact"]["binding_id"] == "bind-1"
    assert loaded["artifact"]["prompt_cache_binding"]["key"] == "work:orbit"
    assert artifacts["artifacts"][0]["provider"] == "mlx"
    assert "dry_run" in deleted_artifact.get("result", deleted_artifact), deleted_artifact
    artifact_result = deleted_artifact.get("result", deleted_artifact)
    assert "dry_run" in artifact_result, deleted_artifact
    assert artifact_result["dry_run"] is True
    assert pruned["results"][0]["artifact_path"].endswith("pruned.artifact")
    assert deleted_bloc["result"]["record"]["bloc_id"] == 1
    assert seen["ensure"]["root_dir"] == str(tmp_path / "runtime-blocs")
    assert seen["load"]["root_dir"] == str(tmp_path / "runtime-blocs")
    assert seen["kv_list"]["bloc_id"] == 1
    assert seen["kv_delete"]["clear_loaded"] is True
    assert seen["kv_prune"]["force"] is True
    assert seen["bloc_delete"]["delete_kv"] is True


def test_multilocal_bloc_host_methods_use_selected_local_client_and_runtime_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient
    from abstractruntime.integrations.abstractcore import llm_client as llm_client_module

    created_clients: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}

    class _FakeLocalClient:
        def __init__(self, *, provider, model, llm_kwargs=None, artifact_store=None, bloc_root_dir=None) -> None:
            _ = (llm_kwargs, artifact_store)
            self._provider = provider
            self._model = model
            self._llm = {"provider": provider, "model": model}
            created_clients.append(
                {
                    "provider": provider,
                    "model": model,
                    "bloc_root_dir": str(bloc_root_dir) if bloc_root_dir is not None else None,
                }
            )

    class _FakeBlocRecord:
        def __init__(self, *, sha256: str, bloc_id: int, path: str, content_sha256: str) -> None:
            self.sha256 = sha256
            self.bloc_id = bloc_id
            self.path = path
            self.content_sha256 = content_sha256

        def to_dict(self) -> Dict[str, Any]:
            return {
                "sha256": self.sha256,
                "bloc_id": self.bloc_id,
                "path": self.path,
                "content_sha256": self.content_sha256,
            }

    class _FakeBlocStore:
        stores: Dict[str, Dict[str, _FakeBlocRecord]] = {}

        def __init__(self, *, root_dir) -> None:
            self.root_dir = root_dir
            self._records = self.stores.setdefault(str(root_dir), {})

        def upsert(self, *, file_meta: Dict[str, Any], content: str, relpath_base=None, summary=None, keywords=None):
            _ = (content, relpath_base, summary, keywords)
            sha = str(file_meta["sha256"])
            existing = self._records.get(sha)
            bloc_id = existing.bloc_id if existing is not None else (len(self._records) + 1)
            record = _FakeBlocRecord(
                sha256=sha,
                bloc_id=bloc_id,
                path=str(file_meta["path"]),
                content_sha256=str(file_meta["content_sha256"]),
            )
            self._records[sha] = record
            return record

        def ensure_bloc_ids(self) -> int:
            return 0

        def get(self, sha256: str):
            return self._records.get(sha256)

        def get_by_bloc_id(self, bloc_id: int):
            for record in self._records.values():
                if record.bloc_id == bloc_id:
                    return record
            return None

    class _FakeManifest:
        def __init__(self, *, bloc_id: int) -> None:
            self.bloc_id = bloc_id

        def to_dict(self) -> Dict[str, Any]:
            return {"bloc_id": self.bloc_id, "binding_id": "bind-1"}

    class _FakeEnsureResult:
        def __init__(self) -> None:
            self.artifact_path = tmp_path / "artifact.bin"
            self.manifest_path = tmp_path / "artifact.json"
            self.compiled = True
            self.rebuilt = False
            self.source_cache_key = "stable:orbit"
            self.binding_id = "bind-1"
            self.prompt_cache_binding = {"binding_id": "bind-1", "key": "stable:orbit"}
            self.manifest = _FakeManifest(bloc_id=1)
            self.debug = None

    class _FakeLoadResult:
        def __init__(self, *, key: str | None) -> None:
            resolved_key = key or "work:orbit"
            self.artifact_path = tmp_path / "artifact.bin"
            self.manifest_path = tmp_path / "artifact.json"
            self.compiled = False
            self.loaded = True
            self.reloaded_stable_key = False
            self.key = resolved_key
            self.stable_cache_key = "stable:orbit"
            self.forked_from = None
            self.binding_id = "bind-1"
            self.prompt_cache_binding = {"binding_id": "bind-1", "key": resolved_key}
            self.manifest = _FakeManifest(bloc_id=1)
            self.debug = None

    def _fake_ensure_bloc_kv_artifact(*, provider, store, model, record, artifact_path=None, force_rebuild=False, debug=False):
        seen["ensure"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "model": model,
            "sha256": record.sha256,
            "artifact_path": artifact_path,
            "force_rebuild": force_rebuild,
            "debug": debug,
        }
        return _FakeEnsureResult()

    def _fake_load_bloc_kv_artifact(
        *,
        provider,
        store,
        model,
        record,
        artifact_path=None,
        stable_cache_key=None,
        key=None,
        make_default=False,
        force_rebuild=False,
        debug=False,
    ):
        seen["load"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "model": model,
            "sha256": record.sha256,
            "artifact_path": artifact_path,
            "stable_cache_key": stable_cache_key,
            "key": key,
            "make_default": make_default,
            "force_rebuild": force_rebuild,
            "debug": debug,
        }
        return _FakeLoadResult(key=key)

    def _fake_read_bloc_kv_manifest(*, provider, store, model, record, artifact_path=None):
        seen["manifest"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "model": model,
            "sha256": record.sha256,
            "artifact_path": artifact_path,
        }
        return _FakeManifest(bloc_id=record.bloc_id)

    def _fake_list_bloc_kv_artifacts(*, store, sha256=None, bloc_id=None, provider=None, model=None):
        seen["kv_list"] = {
            "root_dir": str(store.root_dir),
            "sha256": sha256,
            "bloc_id": bloc_id,
            "provider": provider,
            "model": model,
        }
        record = store.get(sha256) if isinstance(sha256, str) and sha256 else store.get_by_bloc_id(bloc_id)
        if record is None:
            return []
        return [
            {
                "artifact_path": str(store.root_dir / "kv" / f"{record.sha256}.artifact"),
                "manifest_path": str(store.root_dir / "kv" / f"{record.sha256}.manifest.json"),
                "provider": provider or "mlx",
                "model": model or "mlx-community/Qwen3-4B",
                "manifest": {"bloc_sha256": record.sha256, "provider": provider or "mlx", "model": model or "mlx-community/Qwen3-4B"},
            }
        ]

    def _fake_delete_bloc_kv_artifact(
        *,
        provider,
        store,
        sha256=None,
        bloc_id=None,
        provider_name=None,
        model=None,
        artifact_path=None,
        clear_loaded=False,
        force=False,
        dry_run=False,
        debug=False,
    ):
        seen["kv_delete"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "sha256": sha256,
            "bloc_id": bloc_id,
            "provider_name": provider_name,
            "model": model,
            "artifact_path": artifact_path,
            "clear_loaded": clear_loaded,
            "force": force,
            "dry_run": dry_run,
            "debug": debug,
        }
        return SimpleNamespace(
            to_dict=lambda: {
                "operation": "kv_delete",
                "deleted": not dry_run,
                "dry_run": dry_run,
                "artifact_path": artifact_path,
                "live_bindings": [],
            }
        )

    def _fake_delete_bloc(*, provider, store, sha256=None, bloc_id=None, delete_kv=True, clear_loaded=False, force=False, dry_run=False):
        seen["bloc_delete"] = {
            "provider": provider,
            "root_dir": str(store.root_dir),
            "sha256": sha256,
            "bloc_id": bloc_id,
            "delete_kv": delete_kv,
            "clear_loaded": clear_loaded,
            "force": force,
            "dry_run": dry_run,
        }
        record = store.get(sha256) if isinstance(sha256, str) and sha256 else store.get_by_bloc_id(bloc_id)
        return SimpleNamespace(
            to_dict=lambda: {
                "operation": "bloc_delete",
                "deleted": not dry_run,
                "dry_run": dry_run,
                "record": record.to_dict() if record is not None else None,
                "kv_results": [],
                "live_bindings": [],
            }
        )

    monkeypatch.setattr(llm_client_module, "LocalAbstractCoreLLMClient", _FakeLocalClient)
    monkeypatch.setattr(
        llm_client_module,
        "_load_abstractcore_bloc_api",
        lambda: {
            "FileBlocStore": _FakeBlocStore,
            "ensure_bloc_kv_artifact": _fake_ensure_bloc_kv_artifact,
            "load_bloc_kv_artifact": _fake_load_bloc_kv_artifact,
            "read_bloc_kv_manifest": _fake_read_bloc_kv_manifest,
            "list_bloc_kv_artifacts": _fake_list_bloc_kv_artifacts,
            "find_bloc_kv_live_bindings": lambda **kwargs: [],
            "delete_bloc_kv_artifact": _fake_delete_bloc_kv_artifact,
            "prune_bloc_kv_artifacts": lambda **kwargs: [],
            "delete_bloc": _fake_delete_bloc,
        },
    )

    client = MultiLocalAbstractCoreLLMClient(
        provider="mlx",
        model="mlx-community/Qwen3-4B",
        bloc_root_dir=tmp_path / "runtime-blocs",
    )

    upserted = client.upsert_text_bloc(path="notes/orbit.txt", content="Orbit notes")
    bloc_id = upserted["record"]["bloc_id"]
    listed = client.list_blocs(bloc_id=bloc_id)
    manifest = client.get_bloc_kv_manifest(bloc_id=bloc_id)
    ensured = client.ensure_bloc_kv_artifact(bloc_id=bloc_id, debug=True)
    loaded = client.load_bloc_kv_artifact(
        bloc_id=bloc_id,
        key="work:orbit",
        stable_cache_key="stable:orbit",
        debug=True,
    )
    artifacts = client.list_bloc_kv_artifacts(bloc_id=bloc_id)
    deleted_artifact = client.delete_bloc_kv_artifact(
        bloc_id=bloc_id,
        artifact_path=str(artifacts["artifacts"][0]["artifact_path"]),
        provider="mlx",
        model="mlx-community/Qwen3-4B",
        clear_loaded=True,
        dry_run=True,
        debug=True,
    )
    deleted_bloc = client.delete_bloc(bloc_id=bloc_id, clear_loaded=True, dry_run=True)

    assert bloc_id >= 1
    assert listed["records"][0]["bloc_id"] == bloc_id
    assert manifest["manifest"]["binding_id"] == "bind-1"
    assert ensured["artifact"]["binding_id"] == "bind-1"
    assert loaded["artifact"]["prompt_cache_binding"]["key"] == "work:orbit"
    assert artifacts["artifacts"][0]["provider"] == "mlx"
    artifact_result = deleted_artifact.get("result", deleted_artifact)
    assert "dry_run" in artifact_result, deleted_artifact
    assert artifact_result["dry_run"] is True
    assert deleted_bloc["result"]["record"]["bloc_id"] == bloc_id
    assert seen["manifest"]["root_dir"] == str(tmp_path / "runtime-blocs")
    assert seen["ensure"]["root_dir"] == str(tmp_path / "runtime-blocs")
    assert seen["load"]["root_dir"] == str(tmp_path / "runtime-blocs")
    assert seen["kv_list"]["bloc_id"] == bloc_id
    assert seen["kv_delete"]["provider_name"] == "mlx"
    assert seen["bloc_delete"]["delete_kv"] is False
    assert seen["manifest"]["provider"] == {"provider": "mlx", "model": "mlx-community/Qwen3-4B"}
    assert created_clients[0]["bloc_root_dir"] == str(tmp_path / "runtime-blocs")
