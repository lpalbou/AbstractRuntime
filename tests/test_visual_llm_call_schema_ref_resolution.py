from __future__ import annotations

from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json


def test_visual_llm_call_resolves_abstractsemantics_schema_ref() -> None:
    from abstractsemantics import KG_ASSERTION_SCHEMA_REF_V0  # type: ignore

    vf = load_visualflow_json(
        {
            "id": "test-llm-schema-ref",
            "name": "test-llm-schema-ref",
            "description": "",
            "interfaces": [],
            "nodes": [
                {
                    "id": "node-1",
                    "type": "on_flow_start",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "on_flow_start",
                        "label": "Start",
                        "inputs": [],
                        "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                    },
                },
                {
                    "id": "node-2",
                    "type": "llm_call",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "nodeType": "llm_call",
                        "label": "LLM Call",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "provider", "label": "provider", "type": "provider"},
                            {"id": "model", "label": "model", "type": "model"},
                            {"id": "prompt", "label": "prompt", "type": "string"},
                            {"id": "response_schema", "label": "structured_output", "type": "object"},
                        ],
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "response", "label": "response", "type": "string"},
                            {"id": "result", "label": "result", "type": "object"},
                        ],
                        "effectConfig": {
                            "provider": "lmstudio",
                            "model": "qwen/qwen3-next-80b",
                            "structured_output_fallback": False,
                        },
                    },
                },
            ],
            "edges": [
                {
                    "id": "edge-1",
                    "source": "node-1",
                    "sourceHandle": "exec-out",
                    "target": "node-2",
                    "targetHandle": "exec-in",
                    "animated": True,
                }
            ],
            "entryNode": "node-1",
        }
    )

    flow = visual_to_flow(vf)
    handler = flow.nodes["node-2"].handler

    out = handler(
        {
            "provider": "lmstudio",
            "model": "qwen/qwen3-next-80b",
            "prompt": "hi",
            "response_schema": {"$ref": KG_ASSERTION_SCHEMA_REF_V0},
        }
    )
    pending = out["_pending_effect"]
    schema = pending["response_schema"]

    assert isinstance(schema, dict)
    assert "$ref" not in schema  # must be resolved
    assert schema.get("type") == "object"
    assert "assertions" in schema.get("properties", {})
