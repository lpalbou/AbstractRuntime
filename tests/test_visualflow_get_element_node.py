from __future__ import annotations

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def test_visualflow_get_element_node_in_range_negative_and_out_of_range() -> None:
    flow = {
        "id": "get-element",
        "name": "get-element",
        "entryNode": "node-1",
        "nodes": [
            {
                "id": "node-1",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "node-2",
                "type": "get_element",
                "data": {
                    "inputs": [
                        {"id": "array", "label": "array", "type": "array"},
                        {"id": "index", "label": "index", "type": "number"},
                        {"id": "default", "label": "default", "type": "any"},
                    ],
                    "outputs": [
                        {"id": "result", "label": "result", "type": "any"},
                        {"id": "found", "label": "found", "type": "boolean"},
                    ],
                    "pinDefaults": {"array": ["a", "b", "c"], "index": 1, "default": "x"},
                },
            },
            {
                "id": "node-3",
                "type": "get_element",
                "data": {
                    "inputs": [
                        {"id": "array", "label": "array", "type": "array"},
                        {"id": "index", "label": "index", "type": "number"},
                        {"id": "default", "label": "default", "type": "any"},
                    ],
                    "outputs": [
                        {"id": "result", "label": "result", "type": "any"},
                        {"id": "found", "label": "found", "type": "boolean"},
                    ],
                    "pinDefaults": {"array": ["a", "b", "c"], "index": -1, "default": "x"},
                },
            },
            {
                "id": "node-4",
                "type": "get_element",
                "data": {
                    "inputs": [
                        {"id": "array", "label": "array", "type": "array"},
                        {"id": "index", "label": "index", "type": "number"},
                        {"id": "default", "label": "default", "type": "any"},
                    ],
                    "outputs": [
                        {"id": "result", "label": "result", "type": "any"},
                        {"id": "found", "label": "found", "type": "boolean"},
                    ],
                    "pinDefaults": {"array": ["a", "b", "c"], "index": 99, "default": "x"},
                },
            },
            {
                "id": "node-5",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "in_range", "label": "in_range", "type": "any"},
                        {"id": "in_range_found", "label": "in_range_found", "type": "boolean"},
                        {"id": "negative", "label": "negative", "type": "any"},
                        {"id": "negative_found", "label": "negative_found", "type": "boolean"},
                        {"id": "oob", "label": "oob", "type": "any"},
                        {"id": "oob_found", "label": "oob_found", "type": "boolean"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e-exec", "source": "node-1", "target": "node-5", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e-a", "source": "node-2", "target": "node-5", "sourceHandle": "result", "targetHandle": "in_range"},
            {"id": "e-af", "source": "node-2", "target": "node-5", "sourceHandle": "found", "targetHandle": "in_range_found"},
            {"id": "e-b", "source": "node-3", "target": "node-5", "sourceHandle": "result", "targetHandle": "negative"},
            {"id": "e-bf", "source": "node-3", "target": "node-5", "sourceHandle": "found", "targetHandle": "negative_found"},
            {"id": "e-c", "source": "node-4", "target": "node-5", "sourceHandle": "result", "targetHandle": "oob"},
            {"id": "e-cf", "source": "node-4", "target": "node-5", "sourceHandle": "found", "targetHandle": "oob_found"},
        ],
    }

    spec = compile_visualflow(flow)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(workflow=spec, vars={})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output.get("success") is True
    assert "result" not in state.output
    assert state.output["in_range"] == "b"
    assert state.output["in_range_found"] is True
    assert state.output["negative"] == "c"
    assert state.output["negative_found"] is True
    assert state.output["oob"] == "x"
    assert state.output["oob_found"] is False
