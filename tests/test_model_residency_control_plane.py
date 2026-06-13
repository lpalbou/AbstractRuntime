from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from abstractruntime import Effect, EffectType, Runtime, RunState, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore.effect_handlers import (
    build_effect_handlers,
    make_model_residency_handler,
)
from abstractruntime.integrations.abstractcore.llm_client import (
    HttpResponse,
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
)
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


class _ResidencySender:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers), "timeout": timeout})
        return {"ok": True, "models": []}

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        self.calls.append({"method": "POST", "url": url, "headers": dict(headers), "json": dict(json), "timeout": timeout})
        return {"ok": True, "runtime": {"runtime_id": "rid-1"}, "loaded_new": True}


def test_remote_residency_and_prompt_cache_use_root_core_urls_with_v1_base() -> None:
    sender = _ResidencySender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://endpoint.test/v1/",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    client.get_prompt_cache_stats(base_url="http://provider.test/v1")
    client.list_model_residency(task="image_generation", provider="mflux", model="flux")
    client.load_model_residency(
        task="image_generation",
        provider="mflux",
        model="flux",
        options={"steps": 4},
        pin=True,
        base_url="http://provider.test/v1",
        timeout_s=3600,
        provider_api_key="sekret",
    )

    assert sender.calls[0]["url"] == "http://endpoint.test/acore/prompt_cache/stats?base_url=http%3A%2F%2Fprovider.test%2Fv1"
    assert sender.calls[1]["url"] == (
        "http://endpoint.test/acore/models/loaded?task=image_generation&provider=mflux&model=flux"
    )
    assert sender.calls[2]["url"] == "http://endpoint.test/acore/models/load"
    assert sender.calls[2]["headers"]["X-AbstractCore-Provider-API-Key"] == "sekret"
    assert sender.calls[2]["json"] == {
        "task": "image_generation",
        "provider": "mflux",
        "model": "flux",
        "options": {"steps": 4},
        "pin": True,
        "base_url": "http://provider.test/v1",
        "timeout_s": 3600,
    }


class _RemoteMediaSender:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
        raise AssertionError("not used")

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Any:
        self.calls.append({"method": "POST", "url": url, "headers": dict(headers), "json": dict(json), "timeout": timeout})
        if url.endswith("/images/generations"):
            return {"created": 1, "data": [{"b64_json": base64.b64encode(b"png").decode("ascii")}]}
        return HttpResponse(
            body={"model": json["model"], "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            headers={},
        )


def test_remote_media_and_chat_do_not_duplicate_v1_when_base_url_contains_v1() -> None:
    sender = _RemoteMediaSender()
    store = InMemoryArtifactStore()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test/v1",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(prompt="draw", params={"output": {"modality": "image", "format": "png"}})
    client.generate(prompt="hello")

    assert sender.calls[0]["url"] == "http://core.test/v1/images/generations"
    assert sender.calls[1]["url"] == "http://core.test/v1/chat/completions"


class _NoResidencyClient:
    pass


class _NotFoundResidencyClient:
    def __init__(self) -> None:
        self.calls = 0

    def unload_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        return {"ok": False, "operation": "unload", "status_code": 404, "error": "runtime not_found"}


class _CountingResidencyClient:
    def __init__(self) -> None:
        self.calls = 0

    def load_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        return {
            "ok": True,
            "operation": "load",
            "runtime": {"runtime_id": "rid", "loaded": True},
            "diagnostics": {"token": object()},
        }


class _UnverifiedLoadClient:
    def load_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return {
            "ok": True,
            "operation": "load",
            "runtime": {
                "runtime_id": "rid",
                "loaded": False,
                "state": "provider_not_loaded",
                "provider_residency_verified": True,
                "provider_resident": False,
            },
        }


def _run_residency_workflow(control: Any, payload: Dict[str, Any], *, stores: tuple[Any, Any] | None = None) -> tuple[Runtime, str, RunState]:
    run_store, ledger_store = stores or (InMemoryRunStore(), InMemoryLedgerStore())
    runtime = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
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

    workflow = WorkflowSpec("wf_model_residency", "call", {"call": call, "done": done})
    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id)
    return runtime, run_id, state


def test_model_residency_required_false_completes_with_warning() -> None:
    _, _, state = _run_residency_workflow(
        _NoResidencyClient(),
        {"operation": "load", "task": "image_generation", "required": False},
    )

    assert state.status == RunStatus.COMPLETED
    assert state.output["residency"]["ok"] is False
    assert state.output["residency"]["status_hint"] == "warning"
    assert state.output["residency"]["degraded"] is True
    assert "does not expose" in state.output["residency"]["error"]


def test_model_residency_required_true_fails_the_step() -> None:
    _, _, state = _run_residency_workflow(
        _NoResidencyClient(),
        {"operation": "load", "task": "image_generation", "required": True},
    )

    assert state.status == RunStatus.FAILED
    assert "does not expose" in str(state.error)


def test_model_residency_load_requires_verified_loaded_truth() -> None:
    _, _, soft_state = _run_residency_workflow(
        _UnverifiedLoadClient(),
        {"operation": "load", "task": "text_generation", "required": False},
    )
    _, _, hard_state = _run_residency_workflow(
        _UnverifiedLoadClient(),
        {"operation": "load", "task": "text_generation", "required": True},
    )

    assert soft_state.status == RunStatus.COMPLETED
    result = soft_state.output["residency"]
    assert result["ok"] is False
    assert result["status_hint"] == "warning"
    assert result["degraded"] is True
    assert "without a loaded model" in result["warnings"][0]
    assert hard_state.status == RunStatus.FAILED
    assert "without a loaded model" in str(hard_state.error)


def test_model_residency_unload_not_found_is_idempotent_unless_required() -> None:
    _, _, soft_state = _run_residency_workflow(
        _NotFoundResidencyClient(),
        {"operation": "unload", "runtime_id": "missing", "required": False},
    )
    _, _, hard_state = _run_residency_workflow(
        _NotFoundResidencyClient(),
        {"operation": "unload", "runtime_id": "missing", "required": True},
    )

    assert soft_state.status == RunStatus.COMPLETED
    assert soft_state.output["residency"]["ok"] is True
    assert soft_state.output["residency"]["unloaded"] is False
    assert hard_state.status == RunStatus.FAILED


def test_completed_model_residency_effect_replays_ledger_result_without_recalling_core() -> None:
    stores = (InMemoryRunStore(), InMemoryLedgerStore())
    control = _CountingResidencyClient()
    runtime, run_id, state = _run_residency_workflow(
        control,
        {"operation": "load", "task": "text_generation", "provider": "mlx", "model": "qwen"},
        stores=stores,
    )
    assert state.status == RunStatus.COMPLETED
    assert control.calls == 1

    run = stores[0].load(run_id)
    assert run is not None
    run.status = RunStatus.RUNNING
    run.current_node = "call"
    run.output = None
    run.vars.pop("residency", None)
    stores[0].save(run)

    state2 = runtime.tick(
        workflow=WorkflowSpec(
            "wf_model_residency",
            "call",
            {
                "call": lambda run, ctx: StepPlan(
                    node_id="call",
                    effect=Effect(
                        type=EffectType.MODEL_RESIDENCY,
                        payload={"operation": "load", "task": "text_generation", "provider": "mlx", "model": "qwen"},
                        result_key="residency",
                    ),
                    next_node="done",
                ),
                "done": lambda run, ctx: StepPlan(node_id="done", complete_output={"residency": run.vars.get("residency")}),
            },
        ),
        run_id=run_id,
    )
    assert state2.status == RunStatus.COMPLETED
    assert control.calls == 1
    assert isinstance(state2.output["residency"]["diagnostics"]["token"], str)


class _FakeResidentCapability:
    def __init__(self) -> None:
        self.loads: List[Dict[str, Any]] = []
        self.unloads: List[Dict[str, Any]] = []
        self.loaded = False

    def load_resident_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.loads.append(dict(request))
        self.loaded = True
        return {
            "task": request["task"],
            "provider": request.get("provider"),
            "model": request.get("model"),
            "load_id": f"local:{request['task']}:{request.get('provider')}:{request.get('model')}",
            "loaded": True,
            "loaded_new": True,
            "resident": True,
            "state": "resident",
        }

    def list_loaded_models(self, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        _ = filters
        if not self.loaded:
            return []
        last = self.loads[-1]
        return [
            {
                "task": last["task"],
                "provider": last.get("provider"),
                "model": last.get("model"),
                "load_id": f"local:{last['task']}:{last.get('provider')}:{last.get('model')}",
                "loaded": True,
                "state": "resident",
            }
        ]

    def unload_resident_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.unloads.append(dict(request))
        self.loaded = False
        return {
            "task": request["task"],
            "provider": request.get("provider"),
            "model": request.get("model"),
            "load_id": request.get("load_id") or request.get("runtime_id"),
            "loaded": False,
            "unloaded": True,
            "state": "unloaded",
        }


class _FakeCapabilityCore:
    def __init__(self) -> None:
        self.voice = _FakeResidentCapability()
        self.audio = _FakeResidentCapability()
        self.music = _FakeResidentCapability()
        self.vision = _FakeResidentCapability()


def test_local_media_residency_delegates_to_abstractcore_capability_plugins() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "qwen"
    fake_core = _FakeCapabilityCore()
    client._capability_residency_core = fake_core

    loaded = client.load_model_residency(
        task="tts",
        provider="omnivoice",
        model="supertonic-3",
        options={"voice": "M1"},
        pin=True,
    )
    listed = client.list_model_residency(task="tts", provider="omnivoice", model="supertonic-3")
    unloaded = client.unload_model_residency(task="tts", provider="omnivoice", model="supertonic-3", runtime_id="rid")

    assert loaded["ok"] is True
    assert loaded["supported"] is True
    assert loaded["operation"] == "load"
    assert loaded["task"] == "tts"
    assert loaded["runtime"]["provider"] == "omnivoice"
    assert loaded["runtime"]["model"] == "supertonic-3"
    assert loaded["affected_models"][0]["action"] == "loaded"
    assert fake_core.voice.loads == [
        {
            "task": "tts",
            "provider": "omnivoice",
            "model": "supertonic-3",
            "options": {"voice": "M1"},
            "pin": True,
        }
    ]
    assert listed["ok"] is True
    assert listed["models"][0]["provider"] == "omnivoice"
    assert unloaded["ok"] is True
    assert unloaded["unloaded"] is True
    assert fake_core.voice.unloads[-1]["runtime_id"] == "rid"


def test_local_media_residency_does_not_inherit_text_model_default() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "lmstudio"
    client._model = "qwen/qwen3.6-35b-a3b"
    fake_core = _FakeCapabilityCore()
    client._capability_residency_core = fake_core

    result = client.load_model_residency(task="tts", provider="omnivoice")

    assert result["ok"] is True
    assert fake_core.voice.loads[-1]["provider"] == "omnivoice"
    assert fake_core.voice.loads[-1]["model"] is None
    assert result["runtime"]["model"] is None


def test_local_residency_list_without_task_includes_media_capability_rows() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "lmstudio"
    client._model = "qwen/qwen3.6-35b-a3b"
    fake_core = _FakeCapabilityCore()
    client._capability_residency_core = fake_core

    client.load_model_residency(task="tts", provider="omnivoice")
    listed = client.list_model_residency()

    rows = listed["models"]
    voice_rows = [row for row in rows if row.get("task") == "tts" and row.get("provider") == "omnivoice"]
    assert voice_rows
    assert voice_rows[0]["loaded"] is True
    assert voice_rows[0]["resident"] is True
    assert voice_rows[0]["provider_resident"] is True
    assert voice_rows[0]["provider_residency_verified"] is True
    assert listed["diagnostics"]["task_counts"]["tts"] == 1


def test_multilocal_media_residency_delegates_to_abstractcore_capability_plugins() -> None:
    client = object.__new__(MultiLocalAbstractCoreLLMClient)
    client._default_provider = "lmstudio"
    client._default_model = "qwen"
    client._clients = {}
    fake_core = _FakeCapabilityCore()
    client._capability_residency_core = fake_core

    result = client.load_model_residency(task="music_generation", provider="acestep", model="acestep-v15")

    assert result["ok"] is True
    assert result["supported"] is True
    assert result["diagnostics"]["source"] == "abstractruntime.multilocal"
    assert fake_core.music.loads[-1]["task"] == "music_generation"


def test_multilocal_media_residency_does_not_inherit_text_model_default() -> None:
    client = object.__new__(MultiLocalAbstractCoreLLMClient)
    client._default_provider = "lmstudio"
    client._default_model = "qwen/qwen3.6-35b-a3b"
    client._clients = {}
    fake_core = _FakeCapabilityCore()
    client._capability_residency_core = fake_core

    result = client.load_model_residency(task="tts", provider="omnivoice")

    assert result["ok"] is True
    assert fake_core.voice.loads[-1]["provider"] == "omnivoice"
    assert fake_core.voice.loads[-1]["model"] is None
    assert result["runtime"]["model"] is None


def test_multilocal_residency_list_without_task_includes_media_capability_rows() -> None:
    client = object.__new__(MultiLocalAbstractCoreLLMClient)
    client._default_provider = "lmstudio"
    client._default_model = "qwen/qwen3.6-35b-a3b"
    client._clients = {}
    fake_core = _FakeCapabilityCore()
    client._capability_residency_core = fake_core

    client.load_model_residency(task="tts", provider="omnivoice")
    listed = client.list_model_residency()

    assert any(
        row.get("task") == "tts"
        and row.get("provider") == "omnivoice"
        and row.get("loaded") is True
        and row.get("provider_resident") is True
        for row in listed["models"]
    )
    assert listed["diagnostics"]["task_counts"]["tts"] == 1


def test_local_media_residency_fails_truthfully_when_capability_plugin_does_not_load() -> None:
    class _FailingResidentCapability(_FakeResidentCapability):
        def load_resident_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
            self.loads.append(dict(request))
            return {
                "task": request["task"],
                "provider": request.get("provider"),
                "model": request.get("model"),
                "loaded": False,
                "state": "failed",
                "error": {"code": "load_failed", "message": "provider extra is missing"},
            }

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "qwen"
    client._capability_residency_core = SimpleNamespace(voice=_FailingResidentCapability())

    result = client.load_model_residency(task="tts", provider="omnivoice", model="supertonic-3")

    assert result["ok"] is False
    assert result["supported"] is True
    assert result["status_hint"] == "warning"
    assert result["degraded"] is True
    assert result["runtime"]["state"] == "failed"
    assert "provider extra is missing" in result["error"]


class _CoreResidencyProvider:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = dict(payload)
        self.calls: List[Dict[str, Any]] = []

    def get_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        return dict(self.payload)


class _LoadableCoreResidencyProvider:
    def __init__(self, *, provider: str = "lmstudio", model: str = "model", loaded: bool = False) -> None:
        self.provider = provider
        self.model = model
        self.loaded = loaded
        self.residency_calls: List[Dict[str, Any]] = []
        self.load_model_calls: List[Dict[str, Any]] = []
        self.unload_model_calls: List[Dict[str, Any]] = []

    def get_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        self.residency_calls.append(dict(kwargs))
        return {
            "task": str(kwargs.get("task") or "text_generation"),
            "provider": self.provider,
            "model": str(kwargs.get("model") or self.model),
            "provider_residency_verified": True,
            "provider_resident": bool(self.loaded),
            "loaded": bool(self.loaded),
            "state": "loaded" if self.loaded else "not_loaded",
            "source": "abstractcore.provider.test",
        }

    def load_model(self, model_name: str, **kwargs: Any) -> Dict[str, Any]:
        self.loaded = True
        self.load_model_calls.append({"model": str(model_name), "kwargs": dict(kwargs)})
        return {"supported": True, "operation": "load", "model": str(model_name)}

    def unload_model(self, model_name: str, **kwargs: Any) -> Dict[str, Any]:
        self.loaded = False
        self.unload_model_calls.append({"model": str(model_name), "kwargs": dict(kwargs)})
        return {"supported": True, "operation": "unload", "model": str(model_name)}


def test_local_residency_uses_abstractcore_contract_loaded() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "mlx-community/Qwen3.6-27B-4bit"
    client._llm_kwargs = {}
    provider = _CoreResidencyProvider(
        {
            "provider_residency_verified": True,
            "provider_resident": True,
            "state": "loaded",
            "source": "abstractcore.provider.mlx",
        }
    )
    client._llm = provider

    result = client.list_model_residency(task="text_generation")
    record = result["models"][0]

    assert result["ok"] is True
    assert provider.calls == [{"task": "text_generation", "model": "mlx-community/Qwen3.6-27B-4bit"}]
    assert record["runtime_cached"] is True
    assert record["cache_state"] == "runtime_client_cached"
    assert record["provider_residency_verified"] is True
    assert record["provider_residency_source"] == "abstractcore.provider.mlx"
    assert record["provider_state"] == "loaded"
    assert record["resident"] is True
    assert record["loaded"] is True
    assert record["state"] == "provider_loaded"


def test_local_residency_uses_abstractcore_contract_not_loaded() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "lmstudio"
    client._model = "qwen/qwen3.5-35b-a3b"
    client._llm_kwargs = {}
    client._llm = _CoreResidencyProvider(
        {
            "provider_residency_verified": True,
            "provider_resident": False,
            "state": "not_loaded",
            "source": "abstractcore.provider.lmstudio.native_rest",
            "provider_instance_ids": [],
        }
    )

    record = client.list_model_residency(task="text_generation")["models"][0]

    assert record["provider_residency_verified"] is True
    assert record["provider_resident"] is False
    assert record["provider_residency_source"] == "abstractcore.provider.lmstudio.native_rest"
    assert record["provider_instance_ids"] == []
    assert record["resident"] is False
    assert record["loaded"] is False
    assert record["state"] == "provider_not_loaded"


def test_local_text_load_invokes_provider_load_when_not_resident() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "lmstudio"
    client._model = "gemma-3-1b-it"
    client._llm_kwargs = {}
    provider = _LoadableCoreResidencyProvider(model="gemma-3-1b-it", loaded=False)
    client._llm = provider

    result = client.load_model_residency(
        task="text_generation",
        provider="lmstudio",
        model="gemma-3-1b-it",
        options={"keep_alive": "1h"},
        pin=True,
    )

    assert result["ok"] is True
    assert result["loaded_new"] is True
    assert result["provider_loaded_new"] is True
    assert result["runtime"]["loaded"] is True
    assert result["runtime"]["provider_resident"] is True
    assert provider.load_model_calls == [
        {"model": "gemma-3-1b-it", "kwargs": {"keep_alive": "1h", "pin": True}}
    ]


def test_local_residency_without_core_contract_fails_closed() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "mlx-community/Qwen3.6-27B-4bit"
    client._llm_kwargs = {}
    client._llm = object()

    record = client.list_model_residency(task="text_generation")["models"][0]

    assert record["runtime_cached"] is True
    assert record["provider_residency_verified"] is False
    assert record["provider_resident"] is None
    assert record["provider_residency_source"] == "abstractcore.provider"
    assert record["resident"] is False
    assert record["loaded"] is False
    assert record["state"] == "provider_residency_unknown"
    assert "does not expose" in record["warnings"][0]


def test_local_lmstudio_default_unload_calls_provider_and_keeps_runtime_cache() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "lmstudio"
    client._model = "qwen/qwen3.5-35b-a3b"
    client._llm_kwargs = {}
    provider = _LoadableCoreResidencyProvider(model="qwen/qwen3.5-35b-a3b", loaded=True)
    client._llm = provider

    result = client.unload_model_residency(task="text_generation", provider="lmstudio", model="qwen/qwen3.5-35b-a3b")

    assert result["ok"] is True
    assert result["success"] is True
    assert result["unloaded"] is True
    assert result["runtime_cache_unloaded"] is False
    assert result["runtime"]["provider_residency_verified"] is True
    assert result["runtime"]["loaded"] is False
    assert result["affected_models"][0]["action"] == "unloaded"
    assert provider.unload_model_calls == [{"model": "qwen/qwen3.5-35b-a3b", "kwargs": {}}]


def test_local_openai_compatible_residency_does_not_infer_from_base_url() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "openai-compatible"
    client._model = "local-model"
    client._llm_kwargs = {}
    client._llm = SimpleNamespace(base_url="http://127.0.0.1:8000/v1")

    record = client.list_model_residency(task="text_generation")["models"][0]

    assert record["provider_residency_verified"] is False
    assert record["provider_residency_source"] == "abstractcore.provider"
    assert record["resident"] is False
    assert record["loaded"] is False
    assert record["state"] == "provider_residency_unknown"


def test_multilocal_text_residency_lists_loads_and_unloads_cached_clients(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_mod

    created: Dict[tuple[str, str], Any] = {}

    class _DummyLocal:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Dict[str, Any], artifact_store: Any) -> None:
            _ = llm_kwargs, artifact_store
            self._provider = provider
            self._model = model
            self._llm = _LoadableCoreResidencyProvider(provider=provider, model=model, loaded=True)
            created[(provider, model)] = self

        def get_model_capabilities(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="default")

    loaded = client.load_model_residency(task="text_generation", provider="mlx", model="other")
    listed = client.list_model_residency(task="text_generation", provider="mlx")
    unloaded = client.unload_model_residency(task="text_generation", provider="mlx", model="other")
    missing = client.unload_model_residency(task="text_generation", provider="mlx", model="other")

    assert loaded["ok"] is True
    assert loaded["loaded_new"] is True
    assert loaded["runtime_cache_loaded_new"] is True
    assert {m["model"] for m in listed["models"]} == {"default", "other"}
    assert unloaded["ok"] is True
    assert unloaded["success"] is True
    assert unloaded["unloaded"] is True
    assert unloaded["runtime_cache_unloaded"] is True
    assert unloaded["runtime"]["runtime_cached"] is False
    assert unloaded["runtime"]["provider_residency_verified"] is True
    assert unloaded["runtime"]["provider_resident"] is False
    assert unloaded["runtime"]["loaded"] is False
    assert unloaded["runtime"]["state"] == "provider_not_loaded"
    assert unloaded["affected_models"][0]["action"] == "unloaded"
    assert created[("mlx", "other")]._llm.unload_model_calls == [{"model": "other", "kwargs": {}}]
    assert missing["ok"] is True
    assert missing["unloaded"] is False
    assert missing["affected_models"][0]["action"] == "not_found"


def test_multilocal_text_load_invokes_provider_load_when_runtime_cache_is_not_provider_resident(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_mod

    created: List[Any] = []

    class _DummyLocal:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Dict[str, Any], artifact_store: Any) -> None:
            _ = llm_kwargs, artifact_store
            self._provider = provider
            self._model = model
            self._llm = _LoadableCoreResidencyProvider(provider=provider, model=model, loaded=False)
            created.append(self)

        def get_model_capabilities(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="default")

    result = client.load_model_residency(
        task="text_generation",
        provider="lmstudio",
        model="gemma-3-1b-it",
        options={"keep_alive": "1h"},
        pin=True,
    )

    assert result["ok"] is True
    assert result["loaded_new"] is True
    assert result["provider_loaded_new"] is True
    assert result["runtime_cache_loaded_new"] is True
    assert result["runtime"]["loaded"] is True
    loaded_provider = created[-1]._llm
    assert loaded_provider.load_model_calls == [
        {"model": "gemma-3-1b-it", "kwargs": {"keep_alive": "1h", "pin": True}}
    ]


def test_multilocal_uncached_server_control_unload_calls_provider_without_runtime_cache(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_mod

    created: List[Any] = []

    class _DummyLocal:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Dict[str, Any], artifact_store: Any) -> None:
            _ = llm_kwargs, artifact_store
            self._provider = provider
            self._model = model
            self._llm = _LoadableCoreResidencyProvider(provider=provider, model=model, loaded=True)
            created.append(self)

        def get_model_capabilities(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    monkeypatch.setattr(llm_mod, "_provider_supports_uncached_text_residency", lambda provider: provider == "lmstudio")
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="default")

    result = client.unload_model_residency(
        task="text_generation",
        provider="lmstudio",
        model="gemma-3-1b-it",
    )
    listed = client.list_model_residency(task="text_generation", provider="lmstudio")

    assert result["ok"] is True
    assert result["success"] is True
    assert result["unloaded"] is True
    assert result["runtime_cache_unloaded"] is False
    assert result["runtime"]["runtime_cached"] is False
    assert result["runtime"]["provider_residency_verified"] is True
    assert result["runtime"]["provider_resident"] is False
    assert result["runtime"]["state"] == "provider_not_loaded"
    assert result["affected_models"][0]["action"] == "unloaded"
    assert created[-1]._provider == "lmstudio"
    assert created[-1]._llm.unload_model_calls == [{"model": "gemma-3-1b-it", "kwargs": {}}]
    assert listed["models"] == []


def test_multilocal_default_unload_calls_provider_but_keeps_runtime_cache(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_mod

    class _DummyLocal:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Dict[str, Any], artifact_store: Any) -> None:
            _ = llm_kwargs, artifact_store
            self._provider = provider
            self._model = model
            self._llm = _LoadableCoreResidencyProvider(provider=provider, model=model, loaded=True)

        def get_model_capabilities(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="default")
    provider = client._default_client._llm

    result = client.unload_model_residency(task="text_generation", provider="lmstudio", model="default")
    listed = client.list_model_residency(task="text_generation", provider="lmstudio")

    assert result["ok"] is True
    assert result["success"] is True
    assert result["unloaded"] is True
    assert result["runtime_cache_unloaded"] is False
    assert result["runtime"]["runtime_cached"] is True
    assert result["runtime"]["loaded"] is False
    assert result["affected_models"][0]["action"] == "unloaded"
    assert provider.unload_model_calls == [{"model": "default", "kwargs": {}}]
    assert listed["models"][0]["model"] == "default"
    assert listed["models"][0]["loaded"] is False


def test_multilocal_text_load_fails_truthfully_when_provider_cannot_load(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_mod

    class _DummyLocal:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Dict[str, Any], artifact_store: Any) -> None:
            _ = llm_kwargs, artifact_store
            self._provider = provider
            self._model = model
            self._llm = _CoreResidencyProvider(
                {
                    "provider_residency_verified": True,
                    "provider_resident": False,
                    "state": "not_loaded",
                    "source": "abstractcore.provider.test",
                }
            )

        def get_model_capabilities(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="default")

    result = client.load_model_residency(task="text_generation", provider="lmstudio", model="gemma-3-1b-it")
    listed = client.list_model_residency(task="text_generation", provider="lmstudio")

    assert result["ok"] is False
    assert result["runtime_cache_loaded_new"] is True
    assert result["runtime"]["runtime_cached"] is False
    assert result["runtime"]["loaded"] is False
    assert "load_model" in result["error"]
    assert listed["models"] == []


def test_model_residency_capabilities_describe_core_backed_truth_by_task() -> None:
    local = object.__new__(LocalAbstractCoreLLMClient)
    multi = object.__new__(MultiLocalAbstractCoreLLMClient)
    remote = RemoteAbstractCoreLLMClient(server_base_url="http://core.test", model="openai/gpt-4o-mini")

    local_caps = local.get_model_residency_capabilities()
    multi_caps = multi.get_model_residency_capabilities()
    remote_caps = remote.get_model_residency_capabilities()

    assert local_caps["tasks"]["text_generation"]["supported"] is True
    assert local_caps["tasks"]["text_generation"]["truth_source"] == "abstractcore.provider.get_model_residency"
    assert local_caps["tasks"]["image_generation"]["supported"] is True
    assert local_caps["tasks"]["image_generation"]["truth_source"] == "abstractcore.capability_plugin"
    assert local_caps["tasks"]["image_to_image"]["supported"] is True
    assert local_caps["tasks"]["image_to_image"]["shares_backend_cache_with"] == "image_generation"
    assert local_caps["tasks"]["image_upscale"]["supported"] is True
    assert local_caps["tasks"]["image_upscale"]["shares_backend_cache_with"] == "image_generation"
    assert local_caps["tasks"]["video_generation"]["supported"] is True
    assert local_caps["tasks"]["text_to_video"]["supported"] is True
    assert local_caps["tasks"]["image_to_video"]["supported"] is True
    assert local_caps["tasks"]["tts"]["supported"] is True
    assert local_caps["tasks"]["stt"]["supported"] is True
    assert local_caps["tasks"]["music_generation"]["supported"] is True
    assert multi_caps["tasks"]["text_generation"]["loads_other_models"] is True
    assert multi_caps["tasks"]["tts"]["local_media_residency_backend"] == "capability_plugin"
    assert remote_caps["relay_only"] is True
    assert remote_caps["tasks"]["image_generation"]["truth_source"] == "abstractcore.server./acore/models"
    assert remote_caps["tasks"]["image_to_image"]["truth_source"] == "abstractcore.server./acore/models"
    assert remote_caps["tasks"]["image_to_image"]["supported"] is True
    assert remote_caps["tasks"]["image_upscale"]["supported"] is True
    assert remote_caps["tasks"]["video_generation"]["truth_source"] == "abstractcore.server./acore/models"
    assert remote_caps["tasks"]["text_to_video"]["supported"] is True
    assert remote_caps["tasks"]["image_to_video"]["supported"] is True
    assert remote_caps["tasks"]["music_generation"]["supported"] is False


def test_build_effect_handlers_registers_model_residency_handler() -> None:
    handlers = build_effect_handlers(llm=_NoResidencyClient())
    assert EffectType.MODEL_RESIDENCY in handlers


def test_visual_model_residency_node_compiles_to_runtime_effect() -> None:
    spec = compile_visualflow(
        {
            "id": "vf_model_residency",
            "name": "vf_model_residency",
            "entryNode": "node",
            "nodes": [
                {
                    "id": "node",
                    "type": "model_residency",
                    "data": {
                        "nodeType": "model_residency",
                        "effectConfig": {
                            "operation": "load",
                            "task": "image_generation",
                            "provider": "mflux",
                            "model": "flux",
                            "required": False,
                        },
                    },
                }
            ],
            "edges": [],
        }
    )
    run = RunState(
        run_id="run",
        workflow_id=str(spec.workflow_id),
        status=RunStatus.RUNNING,
        current_node="node",
        vars={"_temp": {}},
    )

    plan = spec.nodes["node"](run, {})

    assert plan.effect is not None
    assert plan.effect.type == EffectType.MODEL_RESIDENCY
    assert plan.effect.payload["operation"] == "load"
    assert plan.effect.payload["task"] == "image_generation"
    assert plan.effect.payload["provider"] == "mflux"
    assert plan.effect.payload["model"] == "flux"
