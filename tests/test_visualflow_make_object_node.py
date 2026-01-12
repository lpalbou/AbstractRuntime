from __future__ import annotations

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def test_visualflow_make_object_node_builds_flat_object_from_pins() -> None:
    flow = {
        "id": "make-object",
        "name": "make-object",
        "entryNode": "node-1",
        "nodes": [
            {
                "id": "node-1",
                "type": "on_flow_start",
                "data": {
                    "inputs": [],
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                },
            },
            {
                "id": "node-2",
                "type": "make_object",
                "data": {
                    "inputs": [
                        {"id": "my_var1", "label": "my_var1", "type": "string"},
                        {"id": "my_var2", "label": "my_var2", "type": "number"},
                    ],
                    "outputs": [{"id": "result", "label": "result", "type": "object"}],
                    "pinDefaults": {"my_var1": "hello", "my_var2": 42},
                },
            },
            {
                "id": "node-3",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "obj", "label": "obj", "type": "object"},
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
                "id": "e-obj",
                "source": "node-2",
                "target": "node-3",
                "sourceHandle": "result",
                "targetHandle": "obj",
            },
        ],
    }

    spec = compile_visualflow(flow)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(workflow=spec, vars={})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output.get("success") is True
    assert state.output.get("result") == {"obj": {"my_var1": "hello", "my_var2": 42}}

