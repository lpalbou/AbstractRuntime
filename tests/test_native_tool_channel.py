"""Native tool-channel fold (maintainer incident 2026-07-11): substrates
that call tools NATIVELY (structured `tool_calls` on the response) instead
of writing fenced ```tool blocks must have their tool intent HONORED, not
discarded.

Root cause pinned here (agent's live A/B on gpt-oss-120b: 0/9 fenced vs 5/5
native): ChatSession read only `resp.content`, so a native-channel model's
genuine tool calls were silently dropped — the model then "helpfully"
fabricated results in prose (Mnemosyne's transcript). Both mechanisms now
land in the SAME election executor; fenced behavior is byte-unchanged
(existing suites pin it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.tools import native_tool_elections  # noqa: E402


# ------------------------------------------------------------- conversion


def test_native_conversion_shapes() -> None:
    elections, markers, notices = native_tool_elections(
        [
            {"name": "web_search", "arguments": {"query": "beaver dams"}},
            # OpenAI convention: nested function + JSON-string arguments.
            {"function": {"name": "diary_read", "arguments": json.dumps({"entry": "diary_ab12cd34"})}},
        ]
    )
    assert [e.name for e in elections] == ["web_search", "diary_read"]
    assert elections[0].body == "beaver dams"
    assert elections[1].body == "diary_ab12cd34"
    assert markers == ["[used tool: web_search]", "[used tool: diary_read]"]
    assert notices == []


def test_native_unknown_tool_refused_loudly() -> None:
    elections, markers, notices = native_tool_elections(
        [{"name": "repo_browser.search_memory", "arguments": {"query": "x"}}]
    )
    assert elections == []
    assert markers == ["[tool call refused: repo_browser.search_memory is not available]"]
    assert any("#FALLBACK" in n for n in notices)


def test_native_cap_is_shared_currency() -> None:
    calls = [{"name": "web_search", "arguments": {"query": f"q{i}"}} for i in range(3)]
    elections, markers, notices = native_tool_elections(calls, max_elections=1)
    assert len(elections) == 1
    assert markers.count("[tool call ignored - too many this turn]") == 2
    assert sum("ignored (cap" in n for n in notices) == 2


def test_native_argument_tolerance() -> None:
    # Plain-string arguments become the body; unknown single-key dict too;
    # multi-key dicts without a content key dump honestly as JSON.
    elections, _markers, _notices = native_tool_elections(
        [
            {"name": "search_memory", "arguments": "the twelve bridges"},
            {"name": "fetch_url", "arguments": {"url": "https://example.org"}},
            {"name": "web_search", "arguments": {"a": "1", "b": "2"}},
        ],
        max_elections=3,  # this test pins argument tolerance, not the cap
    )
    assert elections[0].body == "the twelve bridges"
    assert elections[1].body == "https://example.org"
    assert json.loads(elections[2].body) == {"a": "1", "b": "2"}


# ------------------------------------------------------------- turn loop


class _NativeLLM:
    """Scripted model that answers via the NATIVE tool channel: replies are
    (content, tool_calls) pairs — the gpt-oss shape ChatSession discarded."""

    def __init__(self, replies: List[Any]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, messages: List[Dict[str, str]], system_prompt: str) -> Any:
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})
        item = self.replies.pop(0)

        class _R:
            content: Optional[str] = None
            tool_calls: Optional[List[Dict[str, Any]]] = None

        r = _R()
        if isinstance(item, tuple):
            r.content, r.tool_calls = item
        else:
            r.content = item
        return r


def _make_home(tmp_path: Path) -> Path:
    import copy

    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / "nativeling"
    home_dir.mkdir(parents=True)
    entity_id = "entity:nativeling@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Nativeling"
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store, journal = SQLiteTripleStore(db), SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


def test_native_tool_call_runs_instead_of_being_discarded(tmp_path: Path) -> None:
    """THE MNEMOSYNE CLASS: content pre-announces results while the REAL
    intent rides tool_calls. The lookup must run; the fabrication must not
    ship as the final word."""
    home = open_home(_make_home(tmp_path))
    searches: List[str] = []

    def fake_search(query: str) -> str:
        searches.append(query)
        return "RESULT: live search says NATIVE-MARKER-42."

    llm = _NativeLLM([
        # Turn reply: prose scaffolding + a NATIVE call (no fence anywhere).
        ("Here is what I found:", [{"name": "web_search", "arguments": {"query": "persistent AI report"}}]),
        # Continuation after TOOL RESULTS: the honest reply.
        "The search returned NATIVE-MARKER-42; here is what that means.",
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000,
            web_search_fn=fake_search, out=lambda s: None,
        )
        reply, report = session.turn("Run that internet search again?")

        assert searches == ["persistent AI report"]  # the intent RAN
        assert report.tools == ["web_search"]
        assert "NATIVE-MARKER-42" in reply
        # The continuation saw the results in-turn; the transcript carries
        # the door-authored marker, never a fabricated result as final.
        second_msgs = llm.calls[1]["messages"]
        assert any("TOOL RESULTS" in m["content"] for m in second_msgs)
        assert any("[used tool: web_search]" in m["content"] for m in second_msgs)
    finally:
        home.close()


def test_native_call_with_empty_content_still_completes_the_turn(tmp_path: Path) -> None:
    """gpt-oss often sends tool_calls with EMPTY content — that must not
    abort the turn as an empty reply."""
    home = open_home(_make_home(tmp_path))

    def fake_search(query: str) -> str:
        return "RESULT: found it."

    llm = _NativeLLM([
        ("", [{"name": "web_search", "arguments": {"query": "quiet call"}}]),
        "Found it - here are the words.",
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000,
            web_search_fn=fake_search, out=lambda s: None,
        )
        reply, report = session.turn("Look this up please.")
        assert report.tools == ["web_search"]
        assert reply == "Found it - here are the words."
    finally:
        home.close()


class _DeclaringLLM(_NativeLLM):
    """Test double whose generate() ACCEPTS tools= — records what each call
    declared (None when the kwarg was not passed)."""

    def __init__(self, replies: List[Any]) -> None:
        super().__init__(replies)
        self.declared: List[Optional[List[Dict[str, Any]]]] = []

    def generate(self, *, messages: List[Dict[str, str]], system_prompt: str,
                 tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        self.declared.append(tools)
        return super().generate(messages=messages, system_prompt=system_prompt)


def test_declare_half_grants_ride_the_payload(tmp_path: Path) -> None:
    """THE ARM-N FIX: a tools-capable client gets the GRANTED tools declared
    on the turn call and post-results continuations — and ONLY the granted
    ones (the grant is the single authority)."""
    home = open_home(_make_home(tmp_path))

    def fake_search(query: str) -> str:
        return "RESULT: declared and found."

    llm = _DeclaringLLM([
        ("", [{"name": "web_search", "arguments": {"query": "declared search"}}]),
        "Found it through the declared channel.",
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000,
            web_search_fn=fake_search, out=lambda s: None,
        )
        reply, report = session.turn("Search please.")
        assert report.tools == ["web_search"]
        assert "declared channel" in reply
        # Both reads-capable calls declared; the declared set == the grant.
        assert len(llm.declared) == 2
        for declared in llm.declared:
            assert declared is not None
            assert sorted(t["name"] for t in declared) == sorted(session.allowed_tools)
            assert all("parameters" in t for t in declared)
    finally:
        home.close()


def test_fence_convention_clients_are_called_unchanged(tmp_path: Path) -> None:
    """A client whose generate() takes no tools kwarg (the scripted-double /
    fence-substrate shape) is never handed the kwarg — byte-compatible."""
    home = open_home(_make_home(tmp_path))
    llm = _NativeLLM(["Just words, no tools."])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000,
            out=lambda s: None,
        )
        assert session._llm_accepts_tools is False
        reply, _report = session.turn("Say something.")
        assert reply == "Just words, no tools."
    finally:
        home.close()


def test_native_unknown_tool_is_refused_and_turn_stays_honest(tmp_path: Path) -> None:
    """A hallucinated function name (the repo_browser.* class) refuses
    loudly; no tool runs; the delivered words carry the refusal marker."""
    home = open_home(_make_home(tmp_path))
    llm = _NativeLLM([
        ("I will check my memory.", [{"name": "repo_browser.search", "arguments": {"query": "x"}}]),
    ])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000,
            out=lambda s: None,
        )
        reply, report = session.turn("Check something?")
        assert report.tools == []
        assert "[tool call refused: repo_browser.search is not available]" in reply
        assert any("#FALLBACK" in n and "repo_browser.search" in n for n in report.notices)
    finally:
        home.close()
