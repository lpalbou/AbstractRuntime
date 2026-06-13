def test_visual_agent_unstructured_result_is_minimal_and_messages_live_in_scratchpad():
    """Agent unstructured output should avoid duplicating transcript/context in `result`.

    The user-facing answer is surfaced on the `response` output pin, while the agent-internal
    transcript is stored under the scratchpad (runtime-owned).
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

    assert result.get("success") is True
    assert result.get("context") is None
    assert result.get("messages") is None

    agent_ns = temp.get("agent")
    assert isinstance(agent_ns, dict)
    bucket = agent_ns.get("node-agent")
    assert isinstance(bucket, dict)
    scratchpad = bucket.get("scratchpad")
    assert isinstance(scratchpad, dict)
    assert scratchpad.get("task") == "Do the thing."
    assert scratchpad.get("context_extra") == {"foo": "bar"}
    messages = scratchpad.get("messages")
    assert isinstance(messages, list)
    assert [m.get("role") for m in messages] == ["user", "assistant"]
    assert messages[0].get("content") == "Do the thing."
    assert messages[1].get("content") == "Done."


def test_visual_agent_structured_result_sync_exposes_data_pin():
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    vf = load_visualflow_json(
        {
            "id": "vf-agent-structured-sync",
            "name": "vf-agent-structured-sync",
            "entryNode": "node-agent",
            "nodes": [
                {
                    "id": "node-agent",
                    "type": "agent",
                    "data": {
                        "nodeType": "agent",
                        "pinDefaults": {
                            "resp_schema": {
                                "type": "object",
                                "properties": {
                                    "choice": {"type": "string", "enum": ["summarize", "classify"]},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["choice"],
                            }
                        },
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "response", "label": "response", "type": "string"},
                            {"id": "data", "label": "data", "type": "object"},
                            {"id": "meta", "label": "meta", "type": "object"},
                            {"id": "scratchpad", "label": "scratchpad", "type": "object"},
                        ],
                    },
                }
            ],
            "edges": [],
        }
    )
    flow = visual_to_flow(vf)
    run = RunState.new(workflow_id=flow.flow_id, entry_node="node-agent")
    structured_data = {"choice": "summarize", "confidence": 0.95}
    run.vars["_temp"] = {
        "effects": {"node-agent": structured_data},
        "agent": {
            "node-agent": {
                "output_mode": "structured",
                "answer": "The user asked for a summary.",
                "provider": "lmstudio",
                "model": "qwen/qwen3.6-35b-a3b",
                "scratchpad": {
                    "node_id": "node-agent",
                    "steps": [],
                    "messages": [{"role": "assistant", "content": "The user asked for a summary."}],
                },
            }
        },
    }

    _sync_effect_results_to_node_outputs(run, flow)

    out = flow._node_outputs["node-agent"]  # type: ignore[attr-defined]
    assert out["response"] == '{"choice":"summarize","confidence":0.95}'
    assert out["data"] == structured_data
    assert out["meta"]["output_mode"] == "structured"
    assert out["meta"]["provider"] == "lmstudio"
    assert out["meta"]["model"] == "qwen/qwen3.6-35b-a3b"
