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
