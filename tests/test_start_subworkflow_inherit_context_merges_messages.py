def test_start_subworkflow_inherit_context_merges_parent_context_messages():
    """When inherit_context is enabled, the child run should see the parent's latest context.

    This guards against recursive workflows passing a stale `context.messages` pin while
    expecting inherit_context to carry the up-to-date conversation history.
    """
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler.compiler import compile_visualflow

    spec = compile_visualflow(
        {
            "id": "root",
            "name": "root",
            "nodes": [
                {
                    "id": "node-sub",
                    "type": "subflow",
                    "data": {
                        "subflowId": "child",
                        "inputs": [
                            {"id": "exec-in", "type": "execution"},
                            {"id": "inherit_context", "type": "boolean"},
                            {"id": "context", "type": "object"},
                        ],
                        "outputs": [{"id": "exec-out", "type": "execution"}],
                        "pinDefaults": {"inherit_context": True},
                    },
                }
            ],
            "edges": [],
            "entryNode": "node-sub",
        }
    )

    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-sub",
        vars={
            "context": {
                "attachments": [{"$artifact": "a1", "filename": "notes.txt"}],
                "messages": [
                    {"role": "user", "content": "parent-1", "metadata": {"message_id": "m1"}},
                    {"role": "assistant", "content": "parent-2", "metadata": {"message_id": "m2"}},
                    {"role": "assistant", "content": "parent-latest", "metadata": {"message_id": "m3"}},
                ]
            },
            "_last_output": {
                # Stale child context contains only a subset of the parent messages.
                "context": {
                    "messages": [
                        {"role": "system", "content": "child-system"},
                        {"role": "user", "content": "parent-1", "metadata": {"message_id": "m1"}},
                    ]
                },
                "inherit_context": True,
            },
        },
    )

    plan = spec.nodes["node-sub"](run, None)
    eff = plan.effect
    assert eff is not None
    payload = eff.payload
    assert isinstance(payload, dict)
    vars_d = payload.get("vars")
    assert isinstance(vars_d, dict)
    ctx = vars_d.get("context")
    assert isinstance(ctx, dict)
    msgs = ctx.get("messages")
    assert isinstance(msgs, list)
    attachments = ctx.get("attachments")
    assert attachments == [{"$artifact": "a1", "filename": "notes.txt"}]

    # Child system message remains first.
    assert msgs[0].get("role") == "system"
    assert msgs[0].get("content") == "child-system"

    contents = [m.get("content") for m in msgs if isinstance(m, dict)]
    assert "parent-latest" in contents

    # Dedup by message_id: parent-1 should appear once.
    m1_count = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata")
        if isinstance(meta, dict) and meta.get("message_id") == "m1":
            m1_count += 1
    assert m1_count == 1
