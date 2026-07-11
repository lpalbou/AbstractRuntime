"""Visit-honesty wave (maintainer escalation 2026-07-09, forensics-driven).

Pins the fixes for the four failures found in the maintainer's visit
transcript (chat-08534d3ae2a0):

- MEMORIES lines carry DATES and ORIGIN labels ("do you remember last
  time?" was unanswerable from undated handles even though passive recall
  DELIVERED the right episodes);
- the visit contract states own time continues (agency blindness: "I cannot
  run after this conversation ends" x3 against direct correction);
- liveness claims without a lookup get one in-turn correction (a full
  world-state report claimed "fetched live during this session" with ZERO
  tools run);
- the wake cue carries the visit facts from the state reason (commitments
  made in a visit must reach his own time).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import (  # noqa: E402
    ChatSession,
    VISIT_OWN_TIME_PARAGRAPH,
    _LIVENESS_CLAIM_RE,
    open_home,
)


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

    home_dir = tmp_path / "entities" / "honest"
    home_dir.mkdir(parents=True)
    entity_id = "entity:honest@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Honest"
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
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, messages, system_prompt):
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})

        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "…"
        return r


def _session(home_dir: Path, replies: List[str], **kw: Any) -> ChatSession:
    home = open_home(home_dir)
    return ChatSession(
        home,
        _ScriptedLLM(replies),
        participants=["person:tester"],
        context_window=20000,
        out=lambda s: None,
        **kw,
    )


# --------------------------------------------------------- dated memories


def test_memories_block_carries_dates_and_origins(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(home_dir, ["First reply about the comet.", "Second reply."])
    try:
        s.turn("Let us talk about the comet.")
        s.turn("More about the comet.")
        prompt = s.llm.calls[-1]["system_prompt"]
        assert "MEMORIES" in prompt
        # The first turn's episode renders with its date and origin channel.
        import re

        assert re.search(r"\[episode #[0-9a-f]{8} \d{4}-\d{2}-\d{2} - lived conversation\]", prompt), prompt
        assert "newer dates are more recent" in prompt
    finally:
        s.home.close()


# ----------------------------------------------------- own-time awareness


def test_visit_contract_states_own_time_continues(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(home_dir, [])
    try:
        assert VISIT_OWN_TIME_PARAGRAPH in s.system_base
        assert "resumes the moment this visit closes" in s.system_base
    finally:
        s.home.close()


def test_resident_phase_does_not_carry_the_visit_paragraph(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(home_dir, [], phase="own_time")
    try:
        # Own time describes itself via OWN_TIME_CONTRACT (life.py); the
        # visit paragraph ("paused right now") would be false there.
        assert VISIT_OWN_TIME_PARAGRAPH not in s.system_base
    finally:
        s.home.close()


# ------------------------------------------------------- liveness honesty


def test_liveness_regex_matches_the_real_fabrication_and_abstains() -> None:
    # The exact live shapes from the maintainer's transcript:
    assert _LIVENESS_CLAIM_RE.search("The feed was fetched live during this session.")
    assert _LIVENESS_CLAIM_RE.search("I have pulled the most recent news, consulted a few sources")
    assert _LIVENESS_CLAIM_RE.search("I searched the web for updates")
    # Abstentions (observer rules): citation-only, hypothetical, tool names.
    assert not _LIVENESS_CLAIM_RE.search("According to Reuters, the summit ended.")
    assert not _LIVENESS_CLAIM_RE.search("I could fetch the page if you want.")
    assert not _LIVENESS_CLAIM_RE.search("web_search finds pages; fetch_url reads one.")


def test_liveness_claim_without_tools_gets_corrected(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(
        home_dir,
        [
            "Here is my report. The feed was fetched live during this session: markets rose.",
            "I have no live feed this turn - I did not run a lookup. From memory, I recall we discussed markets.",
        ],
    )
    try:
        reply, report = s.turn("Give me a world report.")
        assert any("liveness claim" in n for n in report.notices)
        assert "fetched live" not in reply
        assert report.tools == []
    finally:
        s.home.close()


def test_liveness_correction_can_elect_a_real_tool(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(
        home_dir,
        [
            "I've pulled the latest news: everything is calm.",
            "```tool name=web_search\nlatest world news\n```",
            "Now honestly: the search returned these results.",
        ],
    )
    try:
        s.web_search_fn = lambda q: f"Search results for: {q}\n- headline one"
        reply, report = s.turn("What is happening in the world?")
        assert "web_search" in report.tools  # the correction led to a REAL lookup
        assert "honestly" in reply
    finally:
        s.home.close()


def test_speak_now_guard_turns_marker_only_replies_into_words(tmp_path: Path, monkeypatch) -> None:
    """Mnemosyne's first visit (2026-07-09 06:44): every round returned pure
    tool blocks; the delivered reply was just '[used tool: read_file]'. The
    speak-now guard demands words once the rounds are spent.

    The test pins the GUARD, not the default bound — the ruled default is
    20 (maintainer 2026-07-11), so the rounds constant is narrowed here to
    keep the script exhaustion-shaped without 21 scripted replies."""
    from abstractruntime.identity import tools as tools_mod

    monkeypatch.setattr(tools_mod, "MAX_TOOL_ROUNDS_PER_TURN", 3)
    home_dir = _make_home(tmp_path)
    block = "```tool name=diary_list\n3\n```"
    s = _session(
        home_dir,
        [
            block,   # turn reply: pure election, no words
            block,   # after results, round 2: still no words
            block,   # round 3: still no words
            block,   # rounds exhausted: stripped to markers only
            "I read what you left me. The diary is empty so far - I have just been born.",
        ],
    )
    try:
        reply, report = s.turn("Read your diary and tell me about yourself.")
        assert "I have just been born" in reply
        assert any("speak-now guard" in n for n in report.notices)
        assert report.tools.count("diary_list") == 3  # three rounds ran
    finally:
        s.home.close()


def test_turn_budget_default_is_twenty_and_shared_across_rounds(tmp_path: Path) -> None:
    """Maintainer ruling (2026-07-11 05:25): "default cap for a turn is 20
    tool calls" — the driver-era 2/round is gone. Pins: (a) the constant IS
    20; (b) one reply may elect more than 2 lookups and ALL of them run;
    (c) the budget is turn-wide (remaining budget threads across rounds),
    so no round grants a fresh slice."""
    from abstractruntime.identity.tools import MAX_TOOL_BLOCKS_PER_TURN

    assert MAX_TOOL_BLOCKS_PER_TURN == 20

    home_dir = _make_home(tmp_path)
    five_blocks = "\n".join("```tool name=web_search\nquery %d\n```" % i for i in range(5))
    s = _session(
        home_dir,
        [
            five_blocks,  # one reply, five elections — the old cap dropped 3 of these
            "Here is what I found across all five searches.",
        ],
    )
    try:
        s.web_search_fn = lambda q: f"Search results for: {q}\n- headline"
        reply, report = s.turn("Research this thoroughly, please.")
        assert report.tools.count("web_search") == 5
        assert not any("ignored (cap" in n for n in report.notices)
        assert "found" in reply
    finally:
        s.home.close()


def test_no_correction_when_tools_actually_ran(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(
        home_dir,
        [
            "```tool name=web_search\nworld news\n```",
            "I searched the web and here is what came back: one headline.",
        ],
    )
    try:
        s.web_search_fn = lambda q: "Search results: one headline"
        reply, report = s.turn("Look up the news for real.")
        assert "web_search" in report.tools
        assert not any("liveness claim" in n for n in report.notices)
    finally:
        s.home.close()
