def test_visual_agent_use_context_persists_tool_observations_into_parent_context():
    """When include_context is enabled, persist tool evidence into parent context.messages.

    This is required for outer loops (e.g. RALPH) that re-invoke the Agent node and
    need continuity of evidence (files read/written, commands run, etc.).
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
                    "task": "List files in the repo.",
                    "context": {},
                    "provider": "lmstudio",
                    "model": "dummy",
                    "include_context": True,
                    "tools": ["list_files"],
                },
                "sub": {
                    "sub_run_id": "sub-1",
                    "output": {"answer": "I listed the files.", "iterations": 2},
                    "node_traces": {
                        "act": {
                            "steps": [
                                {
                                    "ts": "2026-01-13T00:00:00+00:00",
                                    "node_id": "act",
                                    "status": "completed",
                                    "effect": {
                                        "type": "tool_calls",
                                        "payload": {
                                            "tool_calls": [
                                                {"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c1"}
                                            ]
                                        },
                                    },
                                    "result": {
                                        "results": [
                                            {
                                                "call_id": "c1",
                                                "name": "list_files",
                                                "success": True,
                                                "output": "file_a.py\\nfile_b.py",
                                                "error": None,
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    },
                },
            }
        }
    }

    handler(run, None)

    ctx = run.vars.get("context")
    assert isinstance(ctx, dict)
    messages = ctx.get("messages")
    assert isinstance(messages, list)

    # user task, tool observation transcript, assistant answer
    assert [m.get("role") for m in messages] == ["user", "assistant", "assistant"]
    assert messages[0].get("content") == "List files in the repo."
    assert "[list_files]:" in str(messages[1].get("content") or "")
    assert messages[2].get("content") == "I listed the files."

