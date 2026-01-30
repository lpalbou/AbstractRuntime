def test_visual_agent_use_context_inherits_parent_attachments_into_subflow_context():
    """When an Agent node uses context, it should inherit context.attachments into the child run.

    Attachments are stored durably in the parent run's context namespace and should be available
    to the agent's ReAct subworkflow without requiring explicit wiring.
    """
    from abstractruntime.core.models import EffectType, RunState
    from abstractruntime.visualflow_compiler.compiler import compile_visualflow

    spec = compile_visualflow(
        {
            "id": "test-flow",
            "name": "test",
            "nodes": [
                {
                    "id": "node-agent",
                    "type": "agent",
                    "data": {"agentConfig": {"provider": "lmstudio", "model": "dummy"}},
                }
            ],
            "edges": [],
            "entryNode": "node-agent",
        }
    )

    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={
            "context": {"attachments": [{"$artifact": "a1", "filename": "notes.txt"}]},
            "_last_output": {"prompt": "Use the attached file.", "include_context": True},
        },
    )

    plan = spec.nodes["node-agent"](run, None)
    assert plan.effect is not None
    assert plan.effect.type == EffectType.START_SUBWORKFLOW

    payload = dict(plan.effect.payload or {})
    sub_vars = payload.get("vars")
    assert isinstance(sub_vars, dict)
    sub_ctx = sub_vars.get("context")
    assert isinstance(sub_ctx, dict)
    assert sub_ctx.get("attachments") == [{"$artifact": "a1", "filename": "notes.txt"}]


def test_visual_agent_inherits_parent_audio_policy_into_subflow_runtime():
    """Agent subworkflows should inherit run-scoped media policies (e.g. audio_policy)."""
    from abstractruntime.core.models import EffectType, RunState
    from abstractruntime.visualflow_compiler.compiler import compile_visualflow

    spec = compile_visualflow(
        {
            "id": "test-flow",
            "name": "test",
            "nodes": [
                {
                    "id": "node-agent",
                    "type": "agent",
                    "data": {"agentConfig": {"provider": "lmstudio", "model": "dummy"}},
                }
            ],
            "edges": [],
            "entryNode": "node-agent",
        }
    )

    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={
            "_runtime": {"audio_policy": "auto", "stt_language": "fr"},
            "_last_output": {"prompt": "Use the attached audio.", "include_context": False},
        },
    )

    plan = spec.nodes["node-agent"](run, None)
    assert plan.effect is not None
    assert plan.effect.type == EffectType.START_SUBWORKFLOW

    payload = dict(plan.effect.payload or {})
    sub_vars = payload.get("vars")
    assert isinstance(sub_vars, dict)
    sub_rt = sub_vars.get("_runtime")
    assert isinstance(sub_rt, dict)
    assert sub_rt.get("audio_policy") == "auto"
    assert sub_rt.get("stt_language") == "fr"
