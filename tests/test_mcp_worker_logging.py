from __future__ import annotations

import io
import json
import sys
from typing import Any, Dict

import httpx
import pytest


def test_worker_http_logs_successful_request(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("abstractcore")

    from abstractruntime.integrations.abstractcore.mcp_worker import McpWorkerState, build_worker_state, build_wsgi_app

    base = build_worker_state(toolsets=["files"])
    state = McpWorkerState(
        tools=base.tools,
        executor=base.executor,
        http_require_auth=False,
    )
    app = build_wsgi_app(state=state)

    transport = httpx.WSGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://worker.test") as client:
        resp = client.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert resp.status_code == 200
    payload: Dict[str, Any] = resp.json()
    assert isinstance(payload.get("result", {}).get("tools"), list)

    logs = capsys.readouterr().err
    assert logs.count("[RECEIVING_COMMANDS]") == 1
    assert "rpc method=tools/list" in logs
    assert logs.count("[RETURNING_RESULTS]") == 1
    assert "rpc_result tools_count=" in logs


def test_worker_http_logs_rejected_request_without_leaking_token(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("abstractcore")

    from abstractruntime.integrations.abstractcore.mcp_worker import McpWorkerState, build_worker_state, build_wsgi_app

    base = build_worker_state(toolsets=["files"])
    state = McpWorkerState(
        tools=base.tools,
        executor=base.executor,
        http_allowed_origins=("https://allowed.example",),
        http_require_auth=True,
        http_auth_token="secret",
    )
    app = build_wsgi_app(state=state)

    transport = httpx.WSGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://worker.test") as client:
        resp = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Origin": "https://evil.example", "Authorization": "Bearer secret"},
        )

    assert resp.status_code == 403
    logs = capsys.readouterr().err
    assert logs.count("[RECEIVING_COMMANDS]") == 1
    assert logs.count("[RETURNING_RESULTS]") == 1
    assert "origin=https://evil.example" in logs
    assert "http status=403" in logs
    assert "secret" not in logs


def test_worker_stdio_logs_request_and_response(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("abstractcore")

    from abstractruntime.integrations.abstractcore.mcp_worker import build_worker_state, serve_stdio

    state = build_worker_state(toolsets=["files"])

    req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, ensure_ascii=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO(req + "\n"))

    serve_stdio(state=state)

    captured = capsys.readouterr()
    payload: Dict[str, Any] = json.loads(captured.out.strip())
    assert isinstance(payload.get("result", {}).get("tools"), list)

    assert captured.err.count("[RECEIVING_COMMANDS]") == 1
    assert "stdio rpc method=tools/list" in captured.err
    assert captured.err.count("[RETURNING_RESULTS]") == 1
    assert "rpc_result tools_count=" in captured.err

