import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime import Effect, EffectType, Runtime, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.tool_executor import DelegatingMcpToolExecutor, McpToolExecutor
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore


class _StubLLMClient:
    def generate(self, **kwargs):
        raise AssertionError("LLM should not be called in this test")


def _mcp_wsgi_app(environ: Dict[str, Any], start_response) -> List[bytes]:
    method = environ.get("REQUEST_METHOD")
    if method != "POST":
        start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
        return [b"method not allowed"]

    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except Exception:
        length = 0

    body = environ.get("wsgi.input").read(length) if environ.get("wsgi.input") else b""
    try:
        req = json.loads(body.decode("utf-8"))
    except Exception:
        start_response("400 Bad Request", [("Content-Type", "application/json")])
        return [json.dumps({"error": "invalid json"}).encode("utf-8")]

    req_id = req.get("id") if isinstance(req, dict) else None
    rpc_method = req.get("method") if isinstance(req, dict) else None
    params = req.get("params") if isinstance(req, dict) and isinstance(req.get("params"), dict) else {}

    resp: Dict[str, Any]
    if rpc_method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name == "add":
            a = int(arguments.get("a") or 0)
            b = int(arguments.get("b") or 0)
            value = a + b
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(value)}], "isError": False},
            }
        elif name == "soft_error":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": "Error: soft failure"}], "isError": False},
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": "Unknown tool"}], "isError": True},
            }
    elif rpc_method == "tools/list":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "add",
                        "description": "Add two integers.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                    {
                        "name": "soft_error",
                        "description": "Always returns an error string with isError=false.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            },
        }
    else:
        resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

    start_response("200 OK", [("Content-Type", "application/json")])
    return [json.dumps(resp).encode("utf-8")]


def test_mcp_tool_executor_executes_remote_calls() -> None:
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("abstractcore")

    from abstractcore.mcp import McpClient

    transport = httpx.WSGITransport(app=_mcp_wsgi_app)
    with httpx.Client(transport=transport, base_url="http://mcp.test") as http_client:
        mcp_client = McpClient(url="http://mcp.test/", client=http_client)
        executor = McpToolExecutor(server_id="srv", mcp_url="http://mcp.test/", mcp_client=mcp_client)
        result = executor.execute(tool_calls=[{"name": "mcp::srv::add", "arguments": {"a": 2, "b": 3}, "call_id": "c1"}])

    assert result["mode"] == "executed"
    assert result["results"][0]["call_id"] == "c1"
    assert result["results"][0]["success"] is True
    assert result["results"][0]["output"] == 5


def test_mcp_tool_executor_detects_soft_error_strings() -> None:
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("abstractcore")

    from abstractcore.mcp import McpClient

    transport = httpx.WSGITransport(app=_mcp_wsgi_app)
    with httpx.Client(transport=transport, base_url="http://mcp.test") as http_client:
        mcp_client = McpClient(url="http://mcp.test/", client=http_client)
        executor = McpToolExecutor(server_id="srv", mcp_url="http://mcp.test/", mcp_client=mcp_client)
        result = executor.execute(tool_calls=[{"name": "mcp::srv::soft_error", "arguments": {}, "call_id": "c2"}])

    assert result["mode"] == "executed"
    assert result["results"][0]["call_id"] == "c2"
    assert result["results"][0]["success"] is False
    assert "soft failure" in str(result["results"][0].get("error") or "")


def test_delegating_mcp_executor_waits_and_resumes_durably() -> None:
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("abstractcore")

    from abstractcore.mcp import McpClient

    transport = httpx.WSGITransport(app=_mcp_wsgi_app)
    with httpx.Client(transport=transport, base_url="http://mcp.test") as http_client:
        mcp_client = McpClient(url="http://mcp.test/", client=http_client)

        delegating = DelegatingMcpToolExecutor(
            server_id="srv",
            mcp_url="http://mcp.test/",
            wait_key_factory=lambda: "job:1",
        )
        remote_executor = McpToolExecutor(server_id="srv", mcp_url="http://mcp.test/", mcp_client=mcp_client)

        wf = WorkflowSpec(
            workflow_id="wf_mcp_delegate",
            entry_node="ACT",
            nodes={
                "ACT": lambda run, ctx: StepPlan(
                    node_id="ACT",
                    effect=Effect(
                        type=EffectType.TOOL_CALLS,
                        payload={
                            "tool_calls": [
                                {"name": "mcp::srv::add", "arguments": {"a": 2, "b": 3}, "call_id": "c1"}
                            ],
                            "allowed_tools": ["mcp::srv::add"],
                        },
                        result_key="tool_results",
                    ),
                    next_node="DONE",
                ),
                "DONE": lambda run, ctx: StepPlan(
                    node_id="DONE",
                    complete_output={"tool_results": run.vars.get("tool_results")},
                ),
            },
        )

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            rt1 = Runtime(
                run_store=JsonFileRunStore(base),
                ledger_store=JsonlLedgerStore(base),
                effect_handlers=build_effect_handlers(llm=_StubLLMClient(), tools=delegating),
            )

            run_id = rt1.start(workflow=wf, vars={})
            st1 = rt1.tick(workflow=wf, run_id=run_id, max_steps=1)
            assert st1.status == RunStatus.WAITING
            assert st1.waiting is not None
            assert st1.waiting.reason.value == "job"
            assert st1.waiting.wait_key == "job:1"
            assert isinstance(st1.waiting.details, dict)
            assert st1.waiting.details.get("mode") == "delegated"
            assert isinstance(st1.waiting.details.get("executor"), dict)
            assert st1.waiting.details["executor"]["protocol"] == "mcp"

            # Simulate process restart: load waiting run from disk.
            rt2 = Runtime(
                run_store=JsonFileRunStore(base),
                ledger_store=JsonlLedgerStore(base),
                effect_handlers=build_effect_handlers(llm=_StubLLMClient(), tools=delegating),
            )
            waiting = rt2.get_state(run_id)
            assert waiting.status == RunStatus.WAITING
            assert waiting.waiting is not None
            assert waiting.waiting.wait_key == "job:1"

            tool_calls = waiting.waiting.details.get("tool_calls")
            assert isinstance(tool_calls, list) and tool_calls
            executed = remote_executor.execute(tool_calls=tool_calls)

            final = rt2.resume(workflow=wf, run_id=run_id, wait_key="job:1", payload=executed)
            assert final.status == RunStatus.COMPLETED
            assert final.output is not None
            assert final.output["tool_results"]["results"][0]["output"] == 5
