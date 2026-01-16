from __future__ import annotations

from types import SimpleNamespace


def _ctx():
    return SimpleNamespace(now_iso=lambda: "2026-01-16T00:00:00Z")


def test_visualflow_llm_call_uses_prompt_var() -> None:
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
                        {"id": "prompt", "label": "prompt", "type": "string"},
                        {"id": "provider", "label": "provider", "type": "provider"},
                        {"id": "model", "label": "model", "type": "model"},
                    ]
                },
            },
            {
                "id": "call",
                "type": "llm_call",
                "position": {"x": 0, "y": 0},
                "data": {"effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    run.vars.update({"prompt": "Hello", "provider": "lmstudio", "model": "unit-test-model"})

    plan1 = spec.get_node("start")(run, _ctx())
    assert plan1.next_node == "call"
    run.current_node = plan1.next_node

    plan2 = spec.get_node("call")(run, _ctx())
    assert plan2.effect is not None
    assert plan2.effect.type.value == "llm_call"
    payload = dict(plan2.effect.payload or {})
    assert payload.get("prompt") == "Hello"


def test_visualflow_llm_call_does_not_use_request_var() -> None:
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
                        {"id": "prompt", "label": "prompt", "type": "string"},
                    ]
                },
            },
            {
                "id": "call",
                "type": "llm_call",
                "position": {"x": 0, "y": 0},
                "data": {"effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    run.vars.update({"request": "Hello"})

    plan1 = spec.get_node("start")(run, _ctx())
    assert plan1.next_node == "call"
    run.current_node = plan1.next_node

    plan2 = spec.get_node("call")(run, _ctx())
    assert plan2.effect is not None
    payload = dict(plan2.effect.payload or {})
    assert payload.get("prompt") == ""


def test_visualflow_agent_uses_prompt_var() -> None:
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
                        {"id": "prompt", "label": "prompt", "type": "string"},
                    ]
                },
            },
            {
                "id": "agent",
                "type": "agent",
                "position": {"x": 0, "y": 0},
                "data": {"agentConfig": {"provider": "lmstudio", "model": "unit-test-model", "tools": []}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "agent", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    run.vars.update({"prompt": "Hello"})

    plan1 = spec.get_node("start")(run, _ctx())
    assert plan1.next_node == "agent"
    run.current_node = plan1.next_node

    plan2 = spec.get_node("agent")(run, _ctx())
    assert plan2.effect is not None
    assert plan2.effect.type.value == "start_subworkflow"
    payload = dict(plan2.effect.payload or {})
    sub_vars = payload.get("vars")
    assert isinstance(sub_vars, dict)
    ctx = sub_vars.get("context")
    assert isinstance(ctx, dict)
    assert ctx.get("task") == "Hello"


def test_visualflow_agent_does_not_use_request_var() -> None:
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
                        {"id": "prompt", "label": "prompt", "type": "string"},
                    ]
                },
            },
            {
                "id": "agent",
                "type": "agent",
                "position": {"x": 0, "y": 0},
                "data": {"agentConfig": {"provider": "lmstudio", "model": "unit-test-model", "tools": []}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "agent", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    run.vars.update({"request": "Hello"})

    plan1 = spec.get_node("start")(run, _ctx())
    assert plan1.next_node == "agent"
    run.current_node = plan1.next_node

    plan2 = spec.get_node("agent")(run, _ctx())
    assert plan2.effect is not None
    payload = dict(plan2.effect.payload or {})
    sub_vars = payload.get("vars")
    assert isinstance(sub_vars, dict)
    ctx = sub_vars.get("context")
    assert isinstance(ctx, dict)
    assert ctx.get("task") == ""
