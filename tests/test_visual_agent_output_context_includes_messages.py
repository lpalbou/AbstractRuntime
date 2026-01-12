def test_visual_agent_output_context_includes_messages_from_subworkflow():
    """Agent node result.context should include the agent's accumulated messages.

    This enables stateful VisualFlow graphs that pass the agent output context into
    subsequent loop iterations or subflows.
    """
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler.compiler import compile_visualflow

    spec = compile_visualflow(
        {
            "id": "test-flow",
            "name": "test",
            "nodes": [
                {
                    "id": "node-agent",
                    "type": "agent",
                    "data": {
                        "agentConfig": {},
                    },
                }
            ],
            "edges": [],
            "entryNode": "node-agent",
        }
    )

    handler = spec.nodes["node-agent"]

    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={"context": {}},
    )

    run.vars["_temp"] = {
        "agent": {
            "node-agent": {
                "phase": "subworkflow",
                "resolved_inputs": {
                    "task": "Do the thing.",
                    "context": {"foo": "bar"},
                    "provider": "lmstudio",
                    "model": "dummy",
                    "include_context": False,
                    "tools": [],
                },
                "sub": {
                    "sub_run_id": "sub-1",
                    "output": {
                        "answer": "Done.",
                        "iterations": 1,
                        "messages": [
                            {"role": "user", "content": "Do the thing.", "metadata": {"message_id": "msg_u"}},
                            {"role": "assistant", "content": "Done.", "metadata": {"message_id": "msg_a"}},
                        ],
                    },
                    "node_traces": {},
                },
            }
        }
    }

    handler(run, None)

    temp = run.vars.get("_temp")
    assert isinstance(temp, dict)
    effects = temp.get("effects")
    assert isinstance(effects, dict)
    result = effects.get("node-agent")
    assert isinstance(result, dict)

    ctx = result.get("context")
    assert isinstance(ctx, dict)
    assert ctx.get("task") == "Do the thing."
    assert ctx.get("foo") == "bar"

    messages = ctx.get("messages")
    assert isinstance(messages, list)
    assert [m.get("role") for m in messages] == ["user", "assistant"]
    assert messages[0].get("content") == "Do the thing."
    assert messages[1].get("content") == "Done."

