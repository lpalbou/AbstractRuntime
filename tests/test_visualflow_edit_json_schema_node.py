from __future__ import annotations

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def test_visualflow_edit_json_schema_node_adds_fields_without_modifying_existing_schema() -> None:
    flow = {
        "id": "edit-json-schema",
        "name": "edit-json-schema",
        "entryNode": "node-1",
        "nodes": [
            {
                "id": "node-1",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "base-schema",
                "type": "json_schema",
                "data": {
                    "inputs": [],
                    "outputs": [{"id": "value", "label": "schema", "type": "object"}],
                    "literalValue": {
                        "type": "object",
                        "title": "Evaluation",
                        "properties": {"score": {"type": "number"}},
                        "required": ["score"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "id": "edit-schema",
                "type": "edit_json_schema",
                "data": {
                    "inputs": [{"id": "schema", "label": "schema", "type": "object"}],
                    "outputs": [{"id": "schema", "label": "schema", "type": "object"}],
                    "literalValue": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer"},
                            "verdict": {"type": "string", "enum": ["pass", "fail"]},
                        },
                        "required": ["score", "verdict"],
                    },
                },
            },
            {
                "id": "node-3",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "schema", "label": "schema", "type": "object"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {
                "id": "e-exec",
                "source": "node-1",
                "target": "node-3",
                "sourceHandle": "exec-out",
                "targetHandle": "exec-in",
            },
            {
                "id": "e-base-schema",
                "source": "base-schema",
                "target": "edit-schema",
                "sourceHandle": "value",
                "targetHandle": "schema",
            },
            {
                "id": "e-schema",
                "source": "edit-schema",
                "target": "node-3",
                "sourceHandle": "schema",
                "targetHandle": "schema",
            },
        ],
    }

    spec = compile_visualflow(flow)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(workflow=spec, vars={})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output.get("schema") == {
        "type": "object",
        "title": "Evaluation",
        "properties": {
            "score": {"type": "number"},
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
        },
        "required": ["score", "verdict"],
        "additionalProperties": False,
    }
