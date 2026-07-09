"""fetch_url tier-1 tool: read-only GET internet access (maintainer ruling
2026-07-08: "internet access, GET not POST, so they can investigate,
explore, learn"). The verb is fixed in the runner; the entity supplies only
a URL, so the prompt can never reach a mutating request.
"""

from __future__ import annotations

from typing import Any, Dict

from abstractruntime.identity import tools as T


def test_fetch_url_is_tier1() -> None:
    assert "fetch_url" in T.TIER1_TOOL_NAMES
    # It is described to the entity as read-only.
    assert "read-only GET" in T.TOOLS_CONTRACT_PARAGRAPH


def test_fetch_url_forces_get_verb(monkeypatch) -> None:
    """The runner must call the underlying tool with method='GET' regardless
    of anything in the entity's body — no POST/PUT reachable from the prompt."""
    seen: Dict[str, Any] = {}

    def fake_fetch(**kwargs):
        seen.update(kwargs)
        return {"title": "Example", "content": "hello world", "success": True}

    import abstractcore.tools.common_tools as ct

    monkeypatch.setattr(ct, "fetch_url", fake_fetch)

    out = T._run_fetch_url("https://example.com")
    assert seen["method"] == "GET"
    assert seen["url"] == "https://example.com"
    assert "Example" in out and "hello world" in out


def test_fetch_url_rejects_non_http_and_empty() -> None:
    assert "needs a URL" in T._run_fetch_url("")
    assert "http(s)" in T._run_fetch_url("ftp://x") 
    assert "http(s)" in T._run_fetch_url("file:///etc/passwd")


def test_fetch_url_truncates_large_pages(monkeypatch) -> None:
    def fake_fetch(**kwargs):
        return {"title": "Big", "content": "x" * 20000, "success": True}

    import abstractcore.tools.common_tools as ct

    monkeypatch.setattr(ct, "fetch_url", fake_fetch)
    out = T._run_fetch_url("https://example.com/big")
    assert "#TRUNCATION" in out
    assert len(out) < 8000  # bounded, not the full 20k


def test_fetch_url_failure_is_honest_not_a_crash(monkeypatch) -> None:
    def fake_fetch(**kwargs):
        return {"error": "connection refused", "success": False}

    import abstractcore.tools.common_tools as ct

    monkeypatch.setattr(ct, "fetch_url", fake_fetch)
    out = T._run_fetch_url("https://down.example")
    assert "could not read" in out and "connection refused" in out


def test_fetch_url_elected_end_to_end() -> None:
    """A fenced fetch_url block parses, dispatches, and returns readable text
    through execute_tool_elections (the path the chat driver uses)."""
    reply = "Let me look this up.\n```tool name=fetch_url\nhttps://example.com/page\n```"
    marked, elections, notices = T.parse_tool_blocks(reply, allowed_names=T.TIER1_TOOL_NAMES)
    assert "[used tool: fetch_url]" in marked
    assert len(elections) == 1 and elections[0].name == "fetch_url"

    msg, exec_notices = T.execute_tool_elections(
        elections,
        diary_store=None,
        diary_read_effect=lambda x: {},
        web_search_fn=lambda q: "unused",
    )
    # No monkeypatch here: the real fetch may fail offline, but it must be an
    # HONEST section, never a crash, and never empty.
    assert "[fetch_url]" in msg
