from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Dict, List, Optional, Tuple


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []

    def supports_prompt_cache(self) -> bool:
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


def test_prompt_cache_modules_appends_incrementally() -> None:
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

    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork", "update"]
    assert provider.calls[-1][2] == [m1]

    provider.calls.clear()
    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key,
        system_prompt=sys,
        tools=tools,
        messages=[m1, m2],
    )

    assert [c[0] for c in provider.calls] == ["prepare_modules", "update"]
    assert provider.calls[-1][2] == [m2]


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

    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork", "update"]


def test_prompt_cache_rebuilds_on_history_divergence() -> None:
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
    # Truncate history (not a prefix-extension): should rebuild from prefix and append full list.
    client._maybe_prepare_prompt_cache(  # type: ignore[attr-defined]
        prompt_cache_key=key,
        system_prompt=sys,
        tools=tools,
        messages=[m1],
    )

    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork", "update"]
    assert provider.calls[-1][2] == [m1]


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


def test_llm_call_ignores_gateway_prompt_cache_env(monkeypatch) -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    monkeypatch.setenv("ABSTRACTGATEWAY_PROMPT_CACHE", "1")
    monkeypatch.delenv("ABSTRACTRUNTIME_PROMPT_CACHE", raising=False)

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
    run = RunState.new(workflow_id="wf-cache", entry_node="node-a", session_id="sess-cache", vars={})
    run.current_node = "node-a"

    outcome = handler(run, Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello", "params": {}}), None)

    assert outcome.status == "completed"
    assert "prompt_cache_key" not in llm.calls[-1]["params"]


def test_llm_call_honors_runtime_prompt_cache_env(monkeypatch) -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    monkeypatch.setenv("ABSTRACTRUNTIME_PROMPT_CACHE", "1")

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
    run = RunState.new(workflow_id="wf-cache", entry_node="node-a", session_id="sess-cache", vars={})
    run.current_node = "node-a"

    outcome = handler(run, Effect(type=EffectType.LLM_CALL, payload={"prompt": "hello", "params": {}}), None)

    assert outcome.status == "completed"
    assert "prompt_cache_key" in llm.calls[-1]["params"]
