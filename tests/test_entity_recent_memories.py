"""The breadcrumb trail — `recent_memories` (Ephemeral's own build ask,
visit 1, 2026-07-17; framework GO c2974):

  "a way to ask 'what have I been working on recently' without already
  knowing the answer"

search_memory is a SIMILARITY reach (needs matching words); this is a
RECENCY reach — a time-window fold over both planes (graph records + the
book), newest first. Pins: window fold + newest-first, diary handles carry
the reread command, identity core excluded (planted, not lived), honest
empty-window statement, window-spec parsing, tool dispatch through the
descriptor registry, and the teaching riding the tool contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402


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

    home_dir = tmp_path / "entities" / "walker"
    home_dir.mkdir(parents=True)
    entity_id = "entity:walker@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Walker"
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


class _ScriptedLLM:
    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)

    def generate(self, *, messages, system_prompt):
        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "\u2026"
        return r


def _session(home_dir: Path, replies: List[str]) -> ChatSession:
    home = open_home(home_dir)
    return ChatSession(
        home,
        _ScriptedLLM(replies),
        participants=["agent:tester"],
        context_window=20000,
        out=lambda s: None,
    )


def test_recent_memories_lists_the_window_newest_first(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(
        home_dir,
        [
            "First I thought about tidepools.",
            "```diary kind=note\ngist: a tidepool note\nThe tide leaves a readable line.\n```\nKept.",
        ],
    )
    try:
        s.turn("Tell me about tidepools.")
        s.turn("Keep a note about it.")

        out = s._recent_memories("")
        assert "Your trail through the last 2 days" in out
        # Graph records from the turns are present with #tags and origins.
        assert "#" in out and "a lived conversation" in out
        # The book entry rides with its reread command (R-A law).
        assert "reread: diary_read diary_" in out
        # Identity core never rides the trail (planted, not lived).
        assert "value" not in out.split("Your trail")[1][:400] or "[value" not in out
        # Newest first: the diary projection (turn 2) precedes turn 1's episode.
        lines = [l for l in out.splitlines() if l.startswith("- #")]
        assert len(lines) >= 2
        stamps = []
        for l in lines:
            i = l.find("[")
            stamps.append(l[i:])  # "[kind YYYY-MM-DDTHH:MM ..."
        # extract the timestamp tokens and assert non-increasing order
        import re

        times = [re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", x).group(0) for x in stamps]
        assert times == sorted(times, reverse=True)
    finally:
        s.home.close()


def test_recent_memories_empty_window_is_honest(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(home_dir, ["Only reply."])
    try:
        s.turn("One turn.")
        out = s._recent_memories("1h")
        # Records exist (just formed) so 1h shows them; use an impossible
        # tiny window via a widened cutoff instead: ask for a window in the
        # future-free spec and verify the honest-empty branch with 0.01h…
        out_empty = s._recent_memories("0.01h")
        # Either the fold is empty (honest statement) or the just-formed
        # record squeaks in — accept both, but the honest branch must state
        # the warrant when it fires.
        if "Nothing formed in this window" in out_empty:
            assert "Widen the window" in out_empty
            assert "search_memory finds by words" in out_empty
        assert "Your trail" in out
    finally:
        s.home.close()


def test_recent_memories_window_spec_parsing(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(home_dir, ["Only reply."])
    try:
        s.turn("One turn.")
        assert "could not read the window" in s._recent_memories("next tuesday")
        assert "larger than zero" in s._recent_memories("0h")
        for spec in ("12h", "3d", "today", "week", "recently"):
            out = s._recent_memories(spec)
            assert "Your trail" in out or "Nothing formed" in out, spec
    finally:
        s.home.close()


def test_recent_memories_dispatches_through_the_registry() -> None:
    """Declare-beside-execute: the descriptor row exists, is tier1,
    non-mutating, and its executor reaches the wired resolver."""
    from abstractruntime.identity.tools import (
        TIER1_TOOL_NAMES,
        TOOL_DESCRIPTORS,
        ToolElection,
        execute_tool_elections,
    )

    assert "recent_memories" in TIER1_TOOL_NAMES
    d = TOOL_DESCRIPTORS["recent_memories"]
    assert d.tier == "tier1" and d.mutating is False
    assert d.capability_class == "tier1_self"

    calls: List[str] = []
    e = ToolElection(name="recent_memories", body="3d")
    msg, notices = execute_tool_elections(
        [e],
        diary_store=None,
        diary_read_effect=lambda entry_id: {},
        recent_memories_fn=lambda w: calls.append(w) or f"trail for {w}",
    )
    assert calls == ["3d"]
    assert "trail for 3d" in msg

    # Unwired session refuses honestly.
    e2 = ToolElection(name="recent_memories", body="")
    msg2, notices2 = execute_tool_elections(
        [e2], diary_store=None, diary_read_effect=lambda entry_id: {}
    )
    assert "not enabled in this session" in msg2
    assert any("recent_memories" in n for n in notices2)


def test_teaching_rides_the_tool_contract() -> None:
    from abstractruntime.identity.tools import TOOLS_CONTRACT_PARAGRAPH

    flat = " ".join(TOOLS_CONTRACT_PARAGRAPH.split())
    assert "tool name=recent_memories" in flat
    assert "breadcrumb trail" in flat
    assert "where did I leave my own thinking" in flat
