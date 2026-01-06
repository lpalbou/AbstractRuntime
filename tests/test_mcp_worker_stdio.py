from __future__ import annotations

from pathlib import Path

import pytest


def test_mcp_worker_list_and_call(tmp_path: Path) -> None:
    pytest.importorskip("abstractcore")

    from abstractruntime.integrations.abstractcore.mcp_worker import build_worker_state, handle_mcp_request

    state = build_worker_state(toolsets=["files"])

    init = handle_mcp_request(req={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}, state=state)
    assert init is not None
    assert init["result"]["protocolVersion"] == "2025-11-25"

    listed = handle_mcp_request(req={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, state=state)
    assert listed is not None
    tools = listed["result"]["tools"]
    assert isinstance(tools, list) and tools
    names = {t.get("name") for t in tools if isinstance(t, dict)}
    assert "list_files" in names

    (tmp_path / "hello.txt").write_text("hello")
    call = handle_mcp_request(
        req={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "list_files",
                "arguments": {"directory_path": str(tmp_path), "pattern": "*", "recursive": False, "include_hidden": False, "head_limit": 50},
            },
        },
        state=state,
    )
    assert call is not None
    assert call["result"]["isError"] is False
    content = call["result"]["content"]
    assert isinstance(content, list) and content and content[0]["type"] == "text"
    assert "hello.txt" in str(content[0].get("text") or "")

