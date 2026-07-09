from __future__ import annotations

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def test_visualflow_get_var_uses_default_for_missing_path() -> None:
    flow = {
        "id": "get-var-default",
        "name": "get-var-default",
        "entryNode": "start",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "data": {
                    "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
                },
            },
            {
                "id": "get_state",
                "type": "get_var",
                "data": {
                    "inputs": [
                        {"id": "name", "label": "name", "type": "string"},
                        {"id": "default", "label": "default", "type": "any"},
                    ],
                    "outputs": [{"id": "value", "label": "value", "type": "any"}],
                    "pinDefaults": {
                        "name": "missing.path",
                        "default": {"rounds_completed": 0, "continue_research": True},
                    },
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "state", "label": "state", "type": "object"},
                    ],
                },
            },
        ],
        "edges": [
            {
                "id": "e-start-end",
                "source": "start",
                "sourceHandle": "exec-out",
                "target": "end",
                "targetHandle": "exec-in",
            },
            {
                "id": "e-get-state",
                "source": "get_state",
                "sourceHandle": "value",
                "target": "end",
                "targetHandle": "state",
            },
        ],
    }

    spec = compile_visualflow(flow)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(workflow=spec, vars={"existing": {"path": None}})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output["state"] == {"rounds_completed": 0, "continue_research": True}
