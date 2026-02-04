from __future__ import annotations

from abstractruntime import Runtime
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow


def test_tool_parameters_node_builds_single_tool_call_object() -> None:
    flow = {
        "id": "tool-params",
        "name": "tool-params",
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
                "type": "tool_parameters",
                "data": {
                    "toolParametersConfig": {"tool": "send_email"},
                    "inputs": [
                        {"id": "account", "label": "account", "type": "string"},
                        {"id": "to", "label": "to", "type": "string"},
                        {"id": "subject", "label": "subject", "type": "string"},
                        {"id": "body_text", "label": "body_text", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "tool_call", "label": "tool_call", "type": "object"},
                        {"id": "account", "label": "account", "type": "string"},
                        {"id": "to", "label": "to", "type": "string"},
                        {"id": "subject", "label": "subject", "type": "string"},
                        {"id": "body_text", "label": "body_text", "type": "string"},
                    ],
                    "pinDefaults": {
                        "account": "work",
                        "to": "you@example.com",
                        "subject": "Daily report",
                        "body_text": "Hello! Here is the report...",
                    },
                },
            },
            {
                "id": "node-3",
                "type": "on_flow_end",
                "data": {
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "tool_call", "label": "tool_call", "type": "object"},
                        {"id": "account", "label": "account", "type": "string"},
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
                "id": "e-tool_call",
                "source": "node-2",
                "target": "node-3",
                "sourceHandle": "tool_call",
                "targetHandle": "tool_call",
            },
            {
                "id": "e-smtp_host",
                "source": "node-2",
                "target": "node-3",
                "sourceHandle": "account",
                "targetHandle": "account",
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

    assert "result" not in state.output

    tool_call = state.output.get("tool_call")
    assert tool_call == {
        "name": "send_email",
        "arguments": {
            "account": "work",
            "to": "you@example.com",
            "subject": "Daily report",
            "body_text": "Hello! Here is the report...",
        },
    }

    assert state.output.get("account") == "work"
