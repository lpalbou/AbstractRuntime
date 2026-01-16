from __future__ import annotations

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def test_visualflow_get_random_element_node_non_empty_and_empty() -> None:
    flow = {
        "id": "get-random-element",
        "name": "get-random-element",
        "entryNode": "node-1",
        "nodes": [
            {
                "id": "node-1",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "node-2",
                "type": "get_random_element",
                "data": {
                    "inputs": [
                        {"id": "array", "label": "array", "type": "array"},
                        {"id": "default", "label": "default", "type": "any"},
                    ],
                    "outputs": [
                        {"id": "result", "label": "result", "type": "any"},
                        {"id": "found", "label": "found", "type": "boolean"},
                    ],
                    "pinDefaults": {"array": ["a", "b", "c"], "default": "x"},
                },
            },
            {
                "id": "node-3",
                "type": "get_random_element",
                "data": {
                    "inputs": [
                        {"id": "array", "label": "array", "type": "array"},
                        {"id": "default", "label": "default", "type": "any"},
                    ],
                    "outputs": [
                        {"id": "result", "label": "result", "type": "any"},
                        {"id": "found", "label": "found", "type": "boolean"},
                    ],
                    "pinDefaults": {"array": [], "default": "x"},
                },
            },
            {
                "id": "node-4",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "picked", "label": "picked", "type": "any"},
                        {"id": "picked_found", "label": "picked_found", "type": "boolean"},
                        {"id": "empty", "label": "empty", "type": "any"},
                        {"id": "empty_found", "label": "empty_found", "type": "boolean"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e-exec", "source": "node-1", "target": "node-4", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e-a", "source": "node-2", "target": "node-4", "sourceHandle": "result", "targetHandle": "picked"},
            {"id": "e-af", "source": "node-2", "target": "node-4", "sourceHandle": "found", "targetHandle": "picked_found"},
            {"id": "e-b", "source": "node-3", "target": "node-4", "sourceHandle": "result", "targetHandle": "empty"},
            {"id": "e-bf", "source": "node-3", "target": "node-4", "sourceHandle": "found", "targetHandle": "empty_found"},
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
    assert state.output["picked_found"] is True
    assert state.output["picked"] in {"a", "b", "c"}

    assert state.output["empty_found"] is False
    assert state.output["empty"] == "x"
