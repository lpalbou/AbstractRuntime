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
                        {"id": "smtp_host", "label": "smtp_host", "type": "string"},
                        {"id": "smtp_port", "label": "smtp_port", "type": "number"},
                        {"id": "use_starttls", "label": "use_starttls", "type": "boolean"},
                        {"id": "username", "label": "username", "type": "string"},
                        {"id": "password_env_var", "label": "password_env_var", "type": "string"},
                        {"id": "to", "label": "to", "type": "string"},
                        {"id": "subject", "label": "subject", "type": "string"},
                        {"id": "body_text", "label": "body_text", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "tool_call", "label": "tool_call", "type": "object"},
                        {"id": "smtp_host", "label": "smtp_host", "type": "string"},
                        {"id": "smtp_port", "label": "smtp_port", "type": "number"},
                        {"id": "use_starttls", "label": "use_starttls", "type": "boolean"},
                        {"id": "username", "label": "username", "type": "string"},
                        {"id": "password_env_var", "label": "password_env_var", "type": "string"},
                        {"id": "to", "label": "to", "type": "string"},
                        {"id": "subject", "label": "subject", "type": "string"},
                        {"id": "body_text", "label": "body_text", "type": "string"},
                    ],
                    "pinDefaults": {
                        "smtp_host": "smtp.gmail.com",
                        "smtp_port": 587,
                        "use_starttls": True,
                        "username": "xxx@gmail.com",
                        "password_env_var": "EMAIL_PASSWORD",
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
                        {"id": "smtp_host", "label": "smtp_host", "type": "string"},
                        {"id": "smtp_port", "label": "smtp_port", "type": "number"},
                        {"id": "use_starttls", "label": "use_starttls", "type": "boolean"},
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
                "sourceHandle": "smtp_host",
                "targetHandle": "smtp_host",
            },
            {
                "id": "e-smtp_port",
                "source": "node-2",
                "target": "node-3",
                "sourceHandle": "smtp_port",
                "targetHandle": "smtp_port",
            },
            {
                "id": "e-use_starttls",
                "source": "node-2",
                "target": "node-3",
                "sourceHandle": "use_starttls",
                "targetHandle": "use_starttls",
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
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "use_starttls": True,
            "username": "xxx@gmail.com",
            "password_env_var": "EMAIL_PASSWORD",
            "to": "you@example.com",
            "subject": "Daily report",
            "body_text": "Hello! Here is the report...",
        },
    }

    assert state.output.get("smtp_host") == "smtp.gmail.com"
    assert state.output.get("smtp_port") == 587
    assert state.output.get("use_starttls") is True
