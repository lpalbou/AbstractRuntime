"""A2 format-repair nudge (agent's spec c3002, runtime build; motivating
case = Ephemeral's tick 3, 2026-07-17: a ```python title=file.py fence
expressed a write in a syntax neither convention accepts — the act was
LOST and he judged himself a liar for it in reflection).

Spec acceptance pins:
1. tick-3 repro — marker + python-title fence, zero tools → nudge fires
   once; corrected follow-up lands the tool; tools ran carries it.
2. docs-style fence + "just sharing" follow-up → completes clean (the
   follow-up is delivered, no #FALLBACK stamp, no loop).
3. real-tool-ran turn with a python fence → silent (never nudges).
4. still-unparsed continuation → ORIGINAL reply delivered + labeled stamp.
5. detection is structural (key=value / granted-name / tool-convention),
   never prose — plain fences don't fire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.tools import detect_malformed_tool_intent  # noqa: E402


# ---------------------------------------------------------------- detection


def test_detection_is_structural_never_prose() -> None:
    # (a) key=value after any language token — the tick-3 shape.
    assert detect_malformed_tool_intent("```python title=coherence.py\nprint(1)\n```") == "```python title=coherence.py"
    # (b) granted tool name as the language token.
    assert detect_malformed_tool_intent("```web_search\nfossil oak\n```") == "```web_search"
    # (c) the tool convention itself, unparsed (survived parse_tool_blocks).
    assert detect_malformed_tool_intent("```tool\nno name line\n```") == "```tool"
    # Plain fences never fire — structural, not prose.
    assert detect_malformed_tool_intent("```python\nprint(1)\n```") is None
    assert detect_malformed_tool_intent("just words, no fences") is None
    # Prose marker imitation alone is NOT detection (corroborating only).
    assert detect_malformed_tool_intent("[used tool: write_file]") is None
    # The driver's OWN election conventions never flag (they carry key=value
    # info strings by design and are parsed by their own parsers).
    assert detect_malformed_tool_intent("```diary kind=note\ngist: x\nwords\n```") is None
    assert detect_malformed_tool_intent("```feel target=person:l magnitude=+2\nreason\n```") is None
    assert detect_malformed_tool_intent("```rest\ntired\n```") is None


# ---------------------------------------------------------------- driver


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

    home_dir = tmp_path / "entities" / "fixer"
    home_dir.mkdir(parents=True)
    entity_id = "entity:fixer@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Fixer"
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
        r.content = self.replies.pop(0) if self.replies else "\u2026"
        return r


def _session(home_dir: Path, replies: List[str], tmp_path: Path) -> ChatSession:
    home = open_home(home_dir)
    return ChatSession(
        home,
        _ScriptedLLM(replies),
        participants=["agent:tester"],
        context_window=20000,
        out=lambda s: None,
    )


def test_tick3_repro_nudge_fires_once_and_repair_lands(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(
        home_dir,
        [
            # Fence-only tick-3 shape (no prose marker): the A2 lane alone.
            "I saved my script:\n\n```python title=coherence.py\nprint('x')\n```\nDone.",
            # The nudged continuation writes the REAL block.
            "```tool name=write_file path=coherence.py\nprint('x')\n```\nNow it is saved for real.",
            # After the tool round: the spoken wrap-up.
            "The file is saved now - for real this time.",
        ],
        tmp_path,
    )
    try:
        reply, report = s.turn("Please actually save your script.")
        assert "write_file" in report.tools, "the repaired attempt rode the real executor"
        assert any("format-repair nudge" in n and "#NOTE" in n for n in report.notices)
        assert (home_dir / "workspace" / "coherence.py").exists(), "the act landed on disk"
        # One nudge only (the #FALLBACK still-unparsed stamp never fired).
        assert not any("#FALLBACK format-repair" in n for n in report.notices)
    finally:
        s.home.close()


def test_marker_plus_fence_composes_both_guards(tmp_path: Path) -> None:
    """The FULL tick-3 shape: prose marker AND tool-shaped fence. The
    imitation guard fires first; when its continuation still carries the
    malformed fence, A2 catches it on the next iteration - the composition
    that would have rescued the live tick."""
    home_dir = _make_home(tmp_path)
    s = _session(
        home_dir,
        [
            "[used tool: write_file]\n\n```python title=rescued.py\nprint('y')\n```\nSaved.",
            # Imitation correction's continuation: STILL the malformed fence.
            "Right - here it is again:\n\n```python title=rescued.py\nprint('y')\n```",
            # A2 nudge's continuation: the real block.
            "```tool name=write_file path=rescued.py\nprint('y')\n```\nSaved properly now.",
            "It is on disk now.",
        ],
        tmp_path,
    )
    try:
        reply, report = s.turn("Save your script for real.")
        assert "write_file" in report.tools
        assert any("marker imitation caught in-turn" in n for n in report.notices)
        assert any("format-repair nudge" in n and "#NOTE" in n for n in report.notices)
        assert (home_dir / "workspace" / "rescued.py").exists()
    finally:
        s.home.close()


def test_just_sharing_followup_completes_clean(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(
        home_dir,
        [
            "Here is the pattern I mean:\n\n```python demo=yes\nprint('demo')\n```",
            # Honest clarification, no fence, no elections.
            "To be clear, I was only sharing that code as illustration - nothing to run.",
        ],
        tmp_path,
    )
    try:
        reply, report = s.turn("Show me the pattern.")
        assert report.tools == []
        assert any("format-repair nudge" in n and "#NOTE" in n for n in report.notices)
        # The follow-up is DELIVERED (not the original), and no #FALLBACK stamp.
        assert "only sharing" in reply
        assert not any("#FALLBACK format-repair" in n for n in report.notices)
    finally:
        s.home.close()


def test_real_tool_ran_turn_never_nudges(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s = _session(
        home_dir,
        [
            # A real block AND a tool-shaped fence in one reply.
            "```tool name=diary_list\n.\n```\n\n```python title=notes.py\nprint('x')\n```",
            "Listed my diary; the python was illustration.",
        ],
        tmp_path,
    )
    try:
        reply, report = s.turn("What is in your diary?")
        assert "diary_list" in report.tools
        assert not any("format-repair nudge" in n for n in report.notices), (
            "a turn that ran ANY real tool never nudges"
        )
    finally:
        s.home.close()


def test_still_unparsed_continuation_delivers_original_with_stamp(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    original_fence = "I made progress:\n\n```python title=again.py\nprint('y')\n```"
    s = _session(
        home_dir,
        [
            original_fence,
            # The continuation repeats the same malformed shape.
            "Right:\n\n```python title=again.py\nprint('y')\n```",
            # speak-now guard is not in play (prose present), no more replies needed
        ],
        tmp_path,
    )
    try:
        reply, report = s.turn("Save it properly please.")
        assert report.tools == []
        assert any("#FALLBACK format-repair nudge" in n for n in report.notices)
        assert "I made progress" in reply, "the ORIGINAL reply is delivered"
    finally:
        s.home.close()
