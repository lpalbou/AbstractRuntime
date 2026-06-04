from __future__ import annotations

from types import SimpleNamespace


def _ctx():
    return SimpleNamespace(now_iso=lambda: "2026-06-03T00:00:00Z")


def test_provider_models_node_filters_by_capability_route(monkeypatch) -> None:
    from abstractruntime.core.models import RunState, RunStatus
    from abstractruntime.visualflow_compiler import compile_visualflow

    def fake_models(provider: str):
        assert provider == "lmstudio"
        return ["text-model", "vision-model"]

    def fake_filter(models, *, capability_routes=None, **_kwargs):
        assert models == ["text-model", "vision-model"]
        assert capability_routes == "input.image,output.text"
        return ["vision-model"]

    monkeypatch.setattr("abstractcore.providers.registry.get_available_models_for_provider", fake_models)
    monkeypatch.setattr("abstractcore.providers.model_capabilities.filter_models_by_capabilities", fake_filter)

    raw = {
        "id": "vf",
        "name": "vf",
        "entryNode": "catalog",
        "nodes": [
            {
                "id": "catalog",
                "type": "provider_models",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "provider_models",
                    "providerModelsConfig": {"provider": "lmstudio", "capabilityRoute": "input.image,output.text"},
                },
            }
        ],
        "edges": [],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="catalog", vars={})
    plan = spec.get_node("catalog")(run, _ctx())

    assert plan.complete_output == {"provider": "lmstudio", "models": ["vision-model"], "success": True}


def test_provider_models_node_invalid_capability_route_fails_closed(monkeypatch) -> None:
    from abstractruntime.core.models import RunState, RunStatus
    from abstractruntime.visualflow_compiler import compile_visualflow

    monkeypatch.setattr("abstractcore.providers.registry.get_available_models_for_provider", lambda _provider: ["text-model"])

    def fake_filter(_models, *, capability_routes=None, **_kwargs):
        assert capability_routes == "not-a-route"
        raise ValueError("Unsupported capability route")

    monkeypatch.setattr("abstractcore.providers.model_capabilities.filter_models_by_capabilities", fake_filter)

    raw = {
        "id": "vf",
        "name": "vf",
        "entryNode": "catalog",
        "nodes": [
            {
                "id": "catalog",
                "type": "provider_models",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "provider_models",
                    "providerModelsConfig": {"provider": "lmstudio", "capabilityRoute": "not-a-route"},
                },
            }
        ],
        "edges": [],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="catalog", vars={})
    plan = spec.get_node("catalog")(run, _ctx())

    assert plan.complete_output["provider"] == "lmstudio"
    assert plan.complete_output["models"] == []
    assert "Invalid capability_route filter" in plan.complete_output["error"]


def test_visualflow_llm_call_includes_thinking_from_config() -> None:
    from abstractruntime.core.models import RunState, RunStatus
    from abstractruntime.visualflow_compiler import compile_visualflow

    raw = {
        "id": "vf",
        "name": "vf",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {"outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "call",
                "type": "llm_call",
                "position": {"x": 0, "y": 0},
                "data": {"effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0, "thinking": "high"}},
            },
        ],
        "edges": [{"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"}],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    plan1 = spec.get_node("start")(run, _ctx())
    run.current_node = plan1.next_node
    plan2 = spec.get_node("call")(run, _ctx())

    assert plan2.effect is not None
    assert plan2.effect.payload["params"]["thinking"] == "high"


def test_visualflow_llm_call_thinking_pin_overrides_config() -> None:
    from abstractruntime.core.models import RunState, RunStatus
    from abstractruntime.visualflow_compiler import compile_visualflow

    raw = {
        "id": "vf",
        "name": "vf",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "thinking", "label": "thinking", "type": "string"},
                    ]
                },
            },
            {
                "id": "call",
                "type": "llm_call",
                "position": {"x": 0, "y": 0},
                "data": {"effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0, "thinking": "high"}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
            {"id": "e2", "source": "start", "sourceHandle": "thinking", "target": "call", "targetHandle": "thinking"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={"thinking": "low"})
    plan1 = spec.get_node("start")(run, _ctx())
    run.current_node = plan1.next_node
    plan2 = spec.get_node("call")(run, _ctx())

    assert plan2.effect is not None
    assert plan2.effect.payload["params"]["thinking"] == "low"


def test_visualflow_agent_subworkflow_carries_thinking_from_config() -> None:
    from abstractruntime.core.models import RunState, RunStatus
    from abstractruntime.visualflow_compiler import compile_visualflow

    raw = {
        "id": "vf",
        "name": "vf",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {"outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "agent",
                "type": "agent",
                "position": {"x": 0, "y": 0},
                "data": {"agentConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0, "thinking": "high"}},
            },
        ],
        "edges": [{"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "agent", "targetHandle": "exec-in"}],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    plan1 = spec.get_node("start")(run, _ctx())
    run.current_node = plan1.next_node
    plan2 = spec.get_node("agent")(run, _ctx())

    assert plan2.effect is not None
    assert plan2.effect.type.value == "start_subworkflow"
    sub_vars = plan2.effect.payload["vars"]
    assert sub_vars["_runtime"]["thinking"] == "high"
