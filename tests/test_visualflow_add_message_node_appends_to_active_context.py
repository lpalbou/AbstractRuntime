def test_visualflow_add_message_node_appends_to_active_context_messages():
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler.compiler import compile_visualflow

    spec = compile_visualflow(
        {
            "id": "test-flow",
            "name": "test",
            "nodes": [
                {
                    "id": "node-add",
                    "type": "add_message",
                    "data": {
                        "inputs": [
                            {"id": "exec-in", "type": "execution"},
                            {"id": "role", "type": "string"},
                            {"id": "content", "type": "string"},
                        ],
                        "outputs": [
                            {"id": "exec-out", "type": "execution"},
                            {"id": "message", "type": "object"},
                            {"id": "context", "type": "object"},
                            {"id": "task", "type": "string"},
                            {"id": "messages", "type": "array"},
                        ],
                    },
                }
            ],
            "edges": [],
            "entryNode": "node-add",
        }
    )

    handler = spec.nodes["node-add"]

    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-add",
        vars={"context": {"task": "", "messages": []}, "_last_output": {"role": "user", "content": "hello"}},
    )

    handler(run, None)

    ctx = run.vars.get("context")
    assert isinstance(ctx, dict)
    msgs = ctx.get("messages")
    assert isinstance(msgs, list)
    assert len(msgs) == 1
    assert msgs[0].get("role") == "user"
    assert msgs[0].get("content") == "hello"
    assert isinstance(msgs[0].get("timestamp"), str)
    meta = msgs[0].get("metadata")
    assert isinstance(meta, dict)
    assert isinstance(meta.get("message_id"), str)
