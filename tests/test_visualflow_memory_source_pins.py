from __future__ import annotations

from types import SimpleNamespace


def _ctx():
    return SimpleNamespace(now_iso=lambda: "2026-01-20T00:00:00Z")


def test_visualflow_llm_call_does_not_schedule_memory_when_disabled() -> None:
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
    run.current_node = plan1.next_node

    plan2 = spec.get_node("call")(run, _ctx())
    assert plan2.effect is not None
    assert plan2.effect.type.value == "llm_call"


def test_visualflow_llm_call_schedules_kg_query_when_enabled_and_injects_active_memory() -> None:
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
                "data": {
                    "effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0},
                    "pinDefaults": {"use_kg_memory": True, "recall_level": "standard"},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    run.vars.update({"prompt": "noonien", "provider": "lmstudio", "model": "unit-test-model"})

    plan1 = spec.get_node("start")(run, _ctx())
    run.current_node = plan1.next_node

    # Phase 1: KG query is scheduled before the LLM call.
    plan2 = spec.get_node("call")(run, _ctx())
    assert plan2.effect is not None
    assert plan2.effect.type.value == "memory_kg_query"
    assert plan2.next_node == "call"

    # Simulate runtime storing the KG query result at the configured result_key.
    run.vars.setdefault("_temp", {}).setdefault("memory_sources", {}).setdefault("call", {})["kg_query"] = {
        "active_memory_text": "## KG ACTIVE MEMORY\n\n- ex:person-noonien-soong —schema:name→ doctor noonien soong",
    }

    # Phase 2: now we emit the LLM call and inject the active memory block.
    plan3 = spec.get_node("call")(run, _ctx())
    assert plan3.effect is not None
    assert plan3.effect.type.value == "llm_call"
    payload = dict(plan3.effect.payload or {})
    sys_text = payload.get("system_prompt") or ""
    assert isinstance(sys_text, str)
    assert "## KG ACTIVE MEMORY" in sys_text


def test_visualflow_llm_call_schedules_span_query_then_rehydrate_then_injects_messages() -> None:
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
                "data": {
                    "effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0},
                    "pinDefaults": {"use_span_memory": True},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    run.vars.update({"prompt": "memory query", "provider": "lmstudio", "model": "unit-test-model", "context": {"messages": []}})

    plan1 = spec.get_node("start")(run, _ctx())
    run.current_node = plan1.next_node

    # Phase 1: span metadata query.
    plan2 = spec.get_node("call")(run, _ctx())
    assert plan2.effect is not None
    assert plan2.effect.type.value == "memory_query"
    assert plan2.next_node == "call"

    run.vars.setdefault("_temp", {}).setdefault("memory_sources", {}).setdefault("call", {})["span_query"] = {
        "results": [{"meta": {"span_ids": ["s1"]}}],
    }

    # Phase 2: rehydrate span into context.
    plan3 = spec.get_node("call")(run, _ctx())
    assert plan3.effect is not None
    assert plan3.effect.type.value == "memory_rehydrate"
    assert plan3.next_node == "call"

    # Simulate rehydration outcome and injected messages in the run context.
    run.vars["_temp"]["memory_sources"]["call"]["span_rehydrate"] = {"inserted": 1, "skipped": 0}
    run.vars["context"]["messages"] = [
        {
            "role": "user",
            "content": "Earlier: ...",
            "metadata": {"rehydrated": True, "source_artifact_id": "s1", "message_id": "m1"},
        }
    ]

    # Phase 3: LLM call includes the rehydrated messages even when include_context is false.
    plan4 = spec.get_node("call")(run, _ctx())
    assert plan4.effect is not None
    assert plan4.effect.type.value == "llm_call"
    payload = dict(plan4.effect.payload or {})
    msgs = payload.get("messages")
    assert isinstance(msgs, list)
    assert any(isinstance(m, dict) and m.get("content") == "Earlier: ..." for m in msgs)


def test_visualflow_agent_schedules_kg_query_before_starting_subworkflow() -> None:
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
                "id": "agent",
                "type": "agent",
                "position": {"x": 0, "y": 0},
                "data": {
                    "agentConfig": {"provider": "lmstudio", "model": "unit-test-model", "tools": []},
                    "pinDefaults": {"use_kg_memory": True, "recall_level": "standard", "use_session_attachments": False},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "agent", "targetHandle": "exec-in"},
        ],
    }

    spec = compile_visualflow(raw)
    run = RunState(run_id="r1", workflow_id=spec.workflow_id, status=RunStatus.RUNNING, current_node="start", vars={})
    run.vars.update({"prompt": "noonien", "provider": "lmstudio", "model": "unit-test-model", "context": {"messages": []}})

    plan1 = spec.get_node("start")(run, _ctx())
    run.current_node = plan1.next_node

    # Phase 1: KG query is scheduled.
    plan2 = spec.get_node("agent")(run, _ctx())
    assert plan2.effect is not None
    assert plan2.effect.type.value == "memory_kg_query"

    # Simulate completed KG query and proceed to starting the agent subworkflow.
    run.vars.setdefault("_temp", {}).setdefault("memory_sources", {}).setdefault("agent", {})["kg_query"] = {
        "active_memory_text": "## KG ACTIVE MEMORY\n\n- ex:person-noonien-soong —schema:name→ doctor noonien soong",
    }

    plan3 = spec.get_node("agent")(run, _ctx())
    assert plan3.effect is not None
    assert plan3.effect.type.value == "start_subworkflow"

    sub_vars = dict(plan3.effect.payload or {}).get("vars")
    assert isinstance(sub_vars, dict)
    runtime_ns = sub_vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    control = runtime_ns.get("control")
    assert isinstance(control, dict)
    assert control.get("include_session_attachments_index") is False

