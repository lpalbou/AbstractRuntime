def test_visual_agent_use_context_persists_turn_into_parent_context():
    """Agent nodes with include_context should persist (task, answer) into vars.context.messages.

    This is required for recursive subflows that re-invoke the agent with "inherit_context":
    the child run must see prior turns from the parent run's active context.
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
                    "task": "Please run the game to test its functionality.",
                    "context": {},
                    "provider": "lmstudio",
                    "model": "dummy",
                    "include_context": True,
                    "tools": [],
                },
                "sub": {
                    "sub_run_id": "sub-1",
                    "output": {"answer": "Sure — running `python main.py` now.", "iterations": 1},
                    "node_traces": {},
                },
            }
        }
    }

    handler(run, None)

    ctx = run.vars.get("context")
    assert isinstance(ctx, dict)
    messages = ctx.get("messages")
    assert isinstance(messages, list)
    assert [m.get("role") for m in messages] == ["user", "assistant"]
    assert messages[0].get("content") == "Please run the game to test its functionality."
    assert messages[1].get("content") == "Sure — running `python main.py` now."
