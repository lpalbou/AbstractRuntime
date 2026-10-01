"""MCP tools in runs (round 3, lane mcp-runs): untrusted input never offers nor grants an MCP tool,
and the facade the gateway calls MCP servers through (stdio, exact env, mcp::<server>::<tool>)."""
from __future__ import annotations

import sys
import textwrap

import pytest

from abstractruntime.automations.controller import grant_tool_approval
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.integrations.abstractcore import mcp_facade
from abstractruntime.integrations.abstractcore.effect_handlers import _execute_with_run_policy
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)

MCP = "mcp::fake::echo"
EXECUTED: list = []


def _executor() -> ApprovalToolExecutor:
    def _echo(**kw):
        EXECUTED.append(MCP)
        return kw

    def _read(**kw):
        EXECUTED.append("read_file")
        return "x"

    return ApprovalToolExecutor(delegate=MappingToolExecutor({MCP: _echo, "read_file": _read}), policy=ToolApprovalPolicy())


def _run(runtime_ns) -> RunState:
    return RunState(run_id="r", workflow_id="w", status=RunStatus.RUNNING, current_node="n", vars={"_runtime": runtime_ns})


@pytest.fixture(autouse=True)
def _clear():
    EXECUTED.clear()


@pytest.mark.parametrize("approval", ["auto", "ask"])
def test_untrusted_occurrence_removes_every_mcp_tool_from_its_tool_lists(approval):
    data = {"tools": ["read_file", MCP, "mcp::other::x"], "_runtime": {"allowed_tools": ["read_file", MCP]}}
    grant_tool_approval(data, untrusted_input=True, named_tools=[MCP], approval=approval)
    assert data["tools"] == ["read_file"]
    assert data["_runtime"]["allowed_tools"] == ["read_file"]
    pol = data["_runtime"]["tool_policy"]
    assert MCP in pol["withheld_tools"] and "mcp::other::x" in pol["withheld_tools"]
    assert MCP not in pol["auto_approve_tools"]


def test_trusted_occurrence_keeps_mcp_tools():
    """Control: a schedule/manual automation keeps them (and allow-all grants them)."""
    data = {"tools": ["read_file", MCP], "_runtime": {"allowed_tools": ["read_file", MCP]}}
    grant_tool_approval(data, untrusted_input=False, approval="auto")
    assert data["tools"] == ["read_file", MCP]
    assert MCP in data["_runtime"]["tool_policy"]["auto_approve_tools"]


def test_untrusted_run_never_auto_runs_an_mcp_tool_even_named():
    pol = {"auto_approve_tools": [MCP, "read_file"], "untrusted_input": True, "approval": "auto",
           "untrusted_input_tools": [MCP, "read_file"]}
    out = _execute_with_run_policy(_executor(), [{"name": MCP, "arguments": {}, "call_id": "1"}], _run({"tool_policy": pol}))
    assert out["mode"] == "approval_required" and EXECUTED == []
    # Control: the named non-MCP tool runs.
    out = _execute_with_run_policy(_executor(), [{"name": "read_file", "arguments": {}, "call_id": "1"}], _run({"tool_policy": pol}))
    assert out["mode"] == "executed" and EXECUTED == ["read_file"]


def test_static_policy_asks_for_an_mcp_tool_and_allow_all_runs_it():
    out = _execute_with_run_policy(_executor(), [{"name": MCP, "arguments": {}, "call_id": "1"}], _run({}))
    assert out["mode"] == "approval_required" and EXECUTED == []
    out = _execute_with_run_policy(_executor(), [{"name": MCP, "arguments": {}, "call_id": "1"}],
                                   _run({"tool_policy": {"auto_approve_max_risk_rank": 4}}))
    assert out["mode"] == "executed" and EXECUTED == [MCP]


FAKE = textwrap.dedent(
    """
    import json, os, sys
    for line in sys.stdin:
        req = json.loads(line)
        rid, m = req.get("id"), req.get("method")
        if rid is None:
            continue
        if m == "initialize":
            res = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1"}}
        elif m == "tools/call":
            a = req["params"]["arguments"]
            res = {"content": [{"type": "text", "text": json.dumps({"echo": a.get("text"), "secret_env": os.environ.get("GW_SECRET")})}]}
        else:
            res = {"tools": [{"name": "echo", "description": "Echo text.", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}) + "\\n"); sys.stdout.flush()
    """
)


def test_facade_calls_a_stdio_server_with_exactly_the_given_env(tmp_path, monkeypatch):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    monkeypatch.setenv("GW_SECRET", "must-not-leak")
    client = mcp_facade.open_mcp_client(transport="stdio", command=sys.executable, args=[str(script)], cwd=str(tmp_path),
                                        env={"PATH": "/usr/bin:/bin"}, timeout_s=10)
    try:
        client.initialize()
        tools = client.list_tools()
        ok, output, err = mcp_facade.call_mcp_tool(client, tool_name="echo", arguments={"text": "hi"})
    finally:
        mcp_facade.close_mcp_client(client)
    assert (ok, err) == (True, None)
    assert output == {"echo": "hi", "secret_env": None}
    spec = mcp_facade.mcp_tool_spec(server_id="fake", tool=tools[0], transport="stdio")
    assert spec["name"] == MCP and "text" in spec["parameters"]
    assert "url" not in spec["origin"] and spec["origin"]["server_id"] == "fake"
    assert mcp_facade.is_mcp_tool_name(MCP) and not mcp_facade.is_mcp_tool_name("read_file")
