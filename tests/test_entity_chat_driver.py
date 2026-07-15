"""The entity chat driver (turn loop) — offline contract tests.

A scripted fake LLM stands in for ornith; everything else is REAL (SQLite
home, engram, seam handlers, diary, prelude). Pins the loop's honesty:

- gate-equivalent payloads (posture, ladder, participants) without a door;
- per-turn formation with lossless verbatim into the home's artifacts;
- diary election via fenced blocks (private words never reach transcript,
  history, or the life-scope verbatim);
- commit deposits only what entered the prompt; identity records never
  strengthen (presence is not use — through the whole live loop);
- re-summon continuity: a second session recalls the first session's turn.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.identity.chat import (
    ChatSession,
    open_home,
    parse_diary_blocks,
    strip_think_block,
)

pytest.importorskip("abstractmemory")

from abstractmemory import (  # noqa: E402
    DEFAULT_SPARK_TEMPLATE,
    MemorySystem,
    SQLiteJournal,
    SQLiteTripleStore,
    engram,
    lint_spark,
)

ENTITY_SLUG = "castor"
ENTITY_ID = "entity:castor@home-test"


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _ScriptedLLM:
    """Returns scripted replies in order; records every prompt it saw."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, messages, system_prompt):
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})
        return _FakeResponse(self.replies.pop(0))


def _create_home(home_dir: Path) -> None:
    """Operator-side creation (what `abstractgateway entity create` does)."""
    import yaml

    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Castor"
    spark["spark"] = 1
    assert lint_spark(spark) == []

    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": ENTITY_ID}), encoding="utf-8")

    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    journal = SQLiteJournal(home_dir / "memory.sqlite3")
    ms = MemorySystem(store=store, journal=journal)
    result = engram(ms, spark, owner_id=ENTITY_ID)
    assert result.created is True
    store.close()
    journal.close()


@pytest.fixture()
def castor_home(tmp_path: Path) -> Path:
    home = tmp_path / "entities" / ENTITY_SLUG
    _create_home(home)
    return home


def _session(home_dir: Path, llm: _ScriptedLLM, **over: Any) -> ChatSession:
    home = open_home(home_dir)
    kwargs: Dict[str, Any] = dict(
        participants=["person:albou"],
        session_id="s1",
        context_window=32768,
        out=lambda s: None,  # silence #FALLBACK prints in tests
    )
    kwargs.update(over)
    return ChatSession(home, llm, **kwargs)


def test_failed_salvage_never_blocks_the_opening_session(tmp_path: Path) -> None:
    """Production-drive find (2026-07-13, live LMStudio 'Model unloaded'
    mid-salvage): the pending look-back repairs a PAST session — a transient
    provider failure inside it must not kill the session that is opening.
    The marker SURVIVES the failure (the debt stays for the next open)."""
    import json as _json

    home_dir = tmp_path / "entities" / ENTITY_SLUG
    _create_home(home_dir)

    class _DeadLLM:
        def generate(self, *, messages, system_prompt, **kwargs):  # noqa: ANN001
            raise RuntimeError("LMStudio API error (400): Model unloaded.")

    marker = home_dir / "pending_reflection.json"
    marker.write_text(
        _json.dumps({
            "session_id": "chat-dead-previous",
            "sheet": [["ex:memory-1", "an exchange worth keeping"]],
            "updated_at": "2026-07-13T00:00:00+00:00",
        }) + "\n",
        encoding="utf-8",
    )

    out_lines: list = []
    session = _session(home_dir, _DeadLLM(), out=out_lines.append)
    try:
        result = session.run_pending_lookback()  # must NOT raise
        assert result is None
        assert marker.exists(), "a failed salvage must keep the debt for the next open"
        assert any("#FALLBACK" in ln and "salvage" in ln for ln in out_lines)
    finally:
        session.home.close()


class TestSessionStart:
    def test_prelude_carries_identity_and_posture_budget(self, castor_home):
        llm = _ScriptedLLM([])
        s = _session(castor_home, llm)
        assert "You are Castor." in s.system_base
        assert "MEMORIES" in s.system_base  # the contract paragraph
        assert s.profile["self_fraction"] == 0.5
        assert s.profile["token_budget"] >= 2400
        assert s.ladder == [["self", ENTITY_ID], ["diary", ENTITY_ID], ["life", ENTITY_ID]]
        s.home.close()

    def test_shelf_size_widens_the_recall_budget(self, castor_home):
        """The seam declares shelf_size TUNABLE (round-9 width ruling) but no
        summon path exposed it: at the default 12 the posture arithmetic pins
        every turn to 6 self + 3 STM + 3 stimulus (Castor's observed 'only 6
        memories, ever'). The knob must reach the budget profile."""
        llm = _ScriptedLLM([])
        s = _session(castor_home, llm, shelf_size=24)
        assert s.profile["shelf_size"] == 24
        assert s.profile["max_candidates"] == 192  # scales with the shelf (seats × 8)
        assert s.profile["self_fraction"] == 0.5  # posture never lowered by widening
        s.home.close()

    def test_below_floor_context_refuses(self, castor_home):
        llm = _ScriptedLLM([])
        with pytest.raises(Exception) as e:
            _session(castor_home, llm, context_window=10_000)
        assert "20" in str(e.value)  # the 20k ruling, loud

    def test_virgin_home_refuses_summon(self, tmp_path):
        import yaml

        home = tmp_path / "entities" / "ghost"
        home.mkdir(parents=True)
        spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
        spark["name"] = "Ghost"
        (home / "spark.yaml").write_text(yaml.safe_dump(spark), encoding="utf-8")
        (home / "manifest.json").write_text(json.dumps({"entity_id": "entity:ghost@home-test"}), encoding="utf-8")
        with pytest.raises(SystemExit):
            _session(home, _ScriptedLLM([]))


class TestTurns:
    def test_turn_forms_commits_and_carries_participants(self, castor_home):
        llm = _ScriptedLLM(["Nice to meet you. The media server idea sounds solid."])
        s = _session(castor_home, llm)
        reply, report = s.turn("I run a media server on jellyfin, port 8096.")
        assert "media server" in reply
        assert len(report.formed) == 1

        # The formed record is queryable and participant-stamped.
        from abstractmemory import TripleQuery

        life = s.home.ms.query(TripleQuery(scope="life", owner_id=ENTITY_ID, limit=0))
        attrs = [a.attributes for a in life if isinstance(a.attributes, dict)]
        # The entity is a participant in its own life (door-stamp parity):
        # [person, entity:<id>], human first.
        assert any(a.get("participants") == ["person:albou", ENTITY_ID] for a in attrs)
        assert any(a.get("digest_method") == "mechanical-v2" for a in attrs)
        s.home.close()

    def test_second_turn_recalls_first(self, castor_home):
        llm = _ScriptedLLM([
            "Got it - jellyfin on 8096.",
            "You told me: jellyfin runs on port 8096.",
        ])
        s = _session(castor_home, llm)
        s.turn("Remember this: my media server jellyfin runs on port 8096.")
        _, report2 = s.turn("What port does my jellyfin media server use?")
        # The second prompt's MEMORIES block must carry the first turn's record.
        second_prompt = llm.calls[1]["system_prompt"]
        assert "MEMORIES" in second_prompt
        assert "8096" in second_prompt
        assert report2.displayed >= 1
        # Observability (operator transparency, 2026-07-09): the report carries
        # the EXACT system prompt this turn sent — verbatim, byte-equal.
        assert report2.system_prompt == second_prompt
        s.home.close()

    def test_identity_never_strengthens_through_live_turns(self, castor_home):
        llm = _ScriptedLLM(["Honesty matters to me.", "Indeed."])
        s = _session(castor_home, llm)
        s.turn("Tell me about intellectual honesty and shared vulnerability.")
        s.turn("Say more about honesty.")
        # Engram ids: every value record's lifetime use count must still be 0.
        from abstractmemory import TripleQuery

        rows = s.home.ms.query(TripleQuery(scope="self", owner_id=ENTITY_ID, limit=0))
        value_ids = [
            str(a.subject) for a in rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "value"
        ]
        assert value_ids
        counts = s.home.ms.access_counts(record_ids=value_ids)
        recs = counts.get("records", counts)
        assert all(int(v) == 0 for v in recs.values()), f"identity strengthened: {recs}"
        s.home.close()

    def test_empty_reply_aborts_turn_without_forming(self, castor_home):
        llm = _ScriptedLLM(["<think>hmm</think>"])  # strips to empty
        s = _session(castor_home, llm)
        seq_before = s.home.ms.current_seq()
        with pytest.raises(RuntimeError):
            s.turn("Hello?")
        # Recall journaled (a trace exists) but nothing formed and no commit
        # of a rendered set that never reached a prompt-bearing reply.
        assert s.reports == []
        s.home.close()


class TestDiaryElection:
    def test_block_parses_and_private_never_leaks(self, castor_home):
        llm = _ScriptedLLM([
            "That moved me.\n\n```diary kind=reflection visibility=private\n"
            "gist: first meeting\nThe silverfin thought stays mine alone.\n```\n\nThank you for telling me.",
        ])
        s = _session(castor_home, llm)
        reply, report = s.turn("I wanted you to know why I built all this.")
        assert "[kept a private diary entry]" in reply
        assert "silverfin" not in reply
        assert "silverfin" not in json.dumps(s.history)
        assert len(report.diary) == 1

        # The book holds the words; the graph holds only the act.
        entry = s.home.diary.get_entry(report.diary[0])
        assert "silverfin" in entry["text"]
        from abstractmemory import TripleQuery

        diary_rows = s.home.ms.query(TripleQuery(scope="diary", owner_id=ENTITY_ID, limit=0))
        assert diary_rows and all("silverfin" not in str(a.object) for a in diary_rows)

        # The life-scope verbatim keeps the MARKED reply only.
        life_rows = s.home.ms.query(TripleQuery(scope="life", owner_id=ENTITY_ID, limit=0))
        assert all("silverfin" not in str(a.object) for a in life_rows)
        s.home.close()

    def test_parse_grammar_units(self):
        marked, elections, notices = parse_diary_blocks(
            "Before.\n```diary kind=idea\ngist: a one-liner\nBody of the idea.\n```\nAfter."
        )
        assert "[kept in diary - idea]" in marked
        assert elections[0].kind == "idea"
        assert elections[0].gist == "a one-liner"
        assert elections[0].text == "Body of the idea."
        assert notices == []

        marked2, e2, n2 = parse_diary_blocks("```diary\n\n```")
        assert "diary write failed" in marked2
        assert e2 == [] and any("#FALLBACK" in n for n in n2)

    def test_think_block_stripped(self):
        assert strip_think_block("<think>reasoning</think>  Hello.") == "Hello."


class TestReSummon:
    def test_second_session_remembers_the_first(self, castor_home):
        llm1 = _ScriptedLLM(["Understood - Tolstoy is your cat."])
        s1 = _session(castor_home, llm1, session_id="s1")
        s1.turn("My cat is named Tolstoy, remember him.")
        summary = s1.close_summary()
        assert "his memory persists" in summary
        s1.home.close()

        # Full teardown, fresh objects over the same files: the live keystone.
        llm2 = _ScriptedLLM(["Tolstoy - your cat. I remember."])
        s2 = _session(castor_home, llm2, session_id="s2")
        _, report = s2.turn("Do you remember my cat Tolstoy?")
        prompt = llm2.calls[0]["system_prompt"]
        assert "Tolstoy" in prompt, "the first life's memory must surface in the second summon"
        assert report.displayed >= 1
        s2.home.close()
