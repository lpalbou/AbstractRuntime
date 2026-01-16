from __future__ import annotations

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def test_visualflow_random_int_and_float_nodes_return_values_in_range() -> None:
    flow = {
        "id": "random-nodes",
        "name": "random-nodes",
        "entryNode": "node-1",
        "nodes": [
            {
                "id": "node-1",
                "type": "on_flow_start",
                "data": {"inputs": [], "outputs": [{"id": "exec-out", "label": "", "type": "execution"}]},
            },
            {
                "id": "node-2",
                "type": "random_int",
                "data": {
                    "inputs": [
                        {"id": "min", "label": "min", "type": "number"},
                        {"id": "max", "label": "max", "type": "number"},
                    ],
                    "outputs": [{"id": "result", "label": "result", "type": "number"}],
                    "pinDefaults": {"min": 5, "max": 10},
                },
            },
            {
                "id": "node-3",
                "type": "random_float",
                "data": {
                    "inputs": [
                        {"id": "min", "label": "min", "type": "number"},
                        {"id": "max", "label": "max", "type": "number"},
                    ],
                    "outputs": [{"id": "result", "label": "result", "type": "number"}],
                    "pinDefaults": {"min": 0, "max": 1},
                },
            },
            {
                "id": "node-4",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "int_val", "label": "int_val", "type": "number"},
                        {"id": "float_val", "label": "float_val", "type": "number"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e-exec", "source": "node-1", "target": "node-4", "sourceHandle": "exec-out", "targetHandle": "exec-in"},
            {"id": "e-int", "source": "node-2", "target": "node-4", "sourceHandle": "result", "targetHandle": "int_val"},
            {"id": "e-float", "source": "node-3", "target": "node-4", "sourceHandle": "result", "targetHandle": "float_val"},
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

    int_val = state.output["int_val"]
    float_val = state.output["float_val"]

    assert isinstance(int_val, int)
    assert 5 <= int_val <= 10

    assert isinstance(float_val, float)
    assert 0.0 <= float_val <= 1.0
