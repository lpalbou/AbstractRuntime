from __future__ import annotations


def test_visualflow_llm_call_context_attachments_are_forwarded_as_media() -> None:
    """VisualFlow LLM Call should map `context.attachments` -> `payload.media`."""
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler import compile_visualflow

    spec = compile_visualflow(
        {
            "id": "vf",
            "name": "vf",
            "entryNode": "call",
            "nodes": [
                {
                    "id": "call",
                    "type": "llm_call",
                    "data": {"effectConfig": {"provider": "lmstudio", "model": "unit-test-model"}},
                }
            ],
            "edges": [],
        }
    )

    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="call",
        vars={
            "_last_output": {
                "prompt": "Hello",
                "context": {
                    "attachments": [{"$artifact": "a1", "filename": "notes.txt"}],
                    "messages": [{"role": "system", "content": "ctx-system"}],
                },
            }
        },
    )

    plan = spec.get_node("call")(run, None)
    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    assert payload.get("media") == [{"$artifact": "a1", "filename": "notes.txt"}]

