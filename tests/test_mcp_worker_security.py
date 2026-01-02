from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import pytest


def test_worker_http_rejects_unallowed_origin() -> None:
    pytest.importorskip("abstractcore")

    from abstractruntime.integrations.abstractcore.mcp_worker import McpWorkerState, build_worker_state, build_wsgi_app

    base = build_worker_state(toolsets=["files"])
    state = McpWorkerState(
        tools=base.tools,
        executor=base.executor,
        http_allowed_origins=("https://allowed.example",),
    )
    app = build_wsgi_app(state=state)

    transport = httpx.WSGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://worker.test") as client:
        resp = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Origin": "https://evil.example"},
        )

    assert resp.status_code == 403
    payload: Dict[str, Any] = resp.json()
    assert payload.get("error", {}).get("message")


def test_worker_http_requires_auth_when_enabled() -> None:
    pytest.importorskip("abstractcore")

    from abstractruntime.integrations.abstractcore.mcp_worker import McpWorkerState, build_worker_state, build_wsgi_app

    base = build_worker_state(toolsets=["files"])
    state = McpWorkerState(
        tools=base.tools,
        executor=base.executor,
        http_require_auth=True,
        http_auth_token="secret",
    )
    app = build_wsgi_app(state=state)

    transport = httpx.WSGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://worker.test") as client:
        resp = client.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"
    payload = json.loads(resp.text)
    assert payload.get("error", {}).get("message")


def test_worker_http_accepts_valid_origin_and_auth() -> None:
    pytest.importorskip("abstractcore")

    from abstractruntime.integrations.abstractcore.mcp_worker import McpWorkerState, build_worker_state, build_wsgi_app

    base = build_worker_state(toolsets=["files"])
    state = McpWorkerState(
        tools=base.tools,
        executor=base.executor,
        http_require_auth=True,
        http_auth_token="secret",
        http_allowed_origins=("https://allowed.example",),
    )
    app = build_wsgi_app(state=state)

    transport = httpx.WSGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://worker.test") as client:
        resp = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Origin": "https://allowed.example", "Authorization": "Bearer secret"},
        )

    assert resp.status_code == 200
    payload: Dict[str, Any] = resp.json()
    tools = payload.get("result", {}).get("tools")
    assert isinstance(tools, list) and tools
    names = {t.get("name") for t in tools if isinstance(t, dict)}
    assert "list_files" in names

