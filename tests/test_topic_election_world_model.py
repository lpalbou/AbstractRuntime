"""Topic election -> world-model concept cards (operator directive 2026-07-19).

The gap this closes, store-proven: world-model card TARGETS derive from
record PARTICIPANTS (+ attributes.topic/topics), and personal time is
SELF-DIRECTED (participants = the entity itself, excluded as scope owner) —
so cards STRUCTURALLY never formed in the phase the operator most wants
them to grow (live store: 2 cards after ~36h of rich personal time, with
"presence vs performance"/"coherence"/"continuity" circled for days).

The chain pinned here:
  elect (```topic at the look-back, cap 2, normalized words)
    -> attributes.topics on the session SUMMARY record (evidence stamp;
       memory's _evidence_scan folds the plural list, _targets_of fans
       each entry to a `topic:<words>` target)
    -> world_model_update(targets=[topic:...]) forms the mechanical floor
       card once the evidence floor (3 lived records) is crossed
    -> _author_world_model_cards rewrites the standing card in the
       entity's own words (author_world_model revision chain).
"""

from __future__ import annotations

import copy as _copy
import json as _json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
abstractmemory = pytest.importorskip("abstractmemory")

from abstractruntime.identity.reflection import (  # noqa: E402
    MAX_TOPICS_PER_SESSION,
    build_reflection_prompt,
    normalize_topic,
    parse_topic_blocks,
)


# ------------------------------------------------------------------ fixtures


def _make_home(tmp_dir: Path, slug: str, name: str) -> tuple[Path, str]:
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_dir / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}@home-test"
    spark = _copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = name
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(_json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir, entity_id


class _LLM:
    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)

    def generate(self, *, messages: Any, system_prompt: Any) -> Any:
        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "…"
        return r


# --------------------------------------------------------------- parse pins


def test_topic_normalization_is_namespace_free_words() -> None:
    """The evidence-chain rule: memory's _targets_of mints the card target
    as `topic:<attributes value>`, so the words that rest MUST be
    namespace-free (stamping "concept:coherence" would mint the nested
    garbage target "topic:concept:coherence"). concept:/topic: spellings
    are tolerated and stripped; record-id shapes are refused; case and
    whitespace fold so the same subject GROUPS across days."""
    assert normalize_topic("Presence   vs Performance") == "presence vs performance"
    assert normalize_topic("concept:coherence") == "coherence"
    assert normalize_topic("topic:Continuity.") == "continuity"
    assert normalize_topic("- coherence") == "coherence"  # list ornament shed
    # Record-id shapes never become targets (the gradation-target rule).
    assert normalize_topic("ex:memory-abc123") is None
    assert normalize_topic("diary:whatever") is None
    assert normalize_topic("diary_feedfacefeedface") is None
    assert normalize_topic("local:x") is None
    # Empty in, none out (including a namespace with nothing behind it).
    assert normalize_topic("") is None
    assert normalize_topic("   ") is None
    assert normalize_topic("concept:") is None


def test_topic_parse_caps_loudly_and_titles_markers() -> None:
    reply = (
        "Looking back.\n"
        "```topic\ncoherence\n```\n"
        "```topic\nPresence vs Performance\n```\n"
        "```topic\na third subject\n```\n"
        "Goodbye."
    )
    marked, topics, notices = parse_topic_blocks(reply)
    assert topics == ["coherence", "presence vs performance"], "cap 2/session"
    assert '[topic: "coherence"]' in marked
    assert '[topic: "presence vs performance"]' in marked
    assert "[topic refused - at most 2 per session]" in marked
    assert any(f"cap {MAX_TOPICS_PER_SESSION}/session" in n for n in notices), "third refused loudly"

    # Empty body: skipped with a notice, marker says so.
    m2, t2, n2 = parse_topic_blocks("```topic\n\n```\nx")
    assert t2 == []
    assert any("empty body" in n for n in n2)
    assert "[topic block skipped - empty]" in m2

    # Unusable line (record-id shape): loud skip, never a silent drop.
    m3, t3, n3 = parse_topic_blocks("```topic\nex:memory-abc\n```\nx")
    assert t3 == []
    assert any("not usable as a subject" in n for n in n3)
    assert "[topic block skipped - no usable subject]" in m3

    # Naming the same subject twice is idempotent (one target, no noise).
    _m4, t4, _n4 = parse_topic_blocks("```topic\ncoherence\nCoherence.\n```\nx")
    assert t4 == ["coherence"]


def test_reflection_prompt_no_longer_offers_topics() -> None:
    """W1 (disposition table): topics DIED as a close solicitation — card
    targets derive mechanically; the parser survives for spontaneous
    fences (the summary-attribute seam stays)."""
    prompt = build_reflection_prompt(["1. an exchange"])
    assert "```topic" not in prompt


def test_topic_fence_never_trips_malformed_tool_intent() -> None:
    """Constraint: the A2 malformed-tool-intent detector must treat ```topic
    as a driver election convention (like diary/feel/interest/lesson), or
    every topic election would draw a format-repair nudge."""
    from abstractruntime.identity.tools import detect_malformed_tool_intent

    assert detect_malformed_tool_intent("```topic\ncoherence\n```") is None
    # Even with a key=value info string (the structural trigger for
    # non-election fences), election langs never flag.
    assert detect_malformed_tool_intent("```topic subject=coherence\nwords\n```") is None


# --------------------------------------------------------------- chat lane


def test_midturn_topic_fence_gets_an_honest_notice() -> None:
    """Mid-turn ```topic stays inert (W1: topics died as an election —
    cards derive mechanically) — the honest notice names the new truth."""
    tmp = Path(tempfile.mkdtemp())
    home_dir, _entity_id = _make_home(tmp, "midturner", "Midturner")

    from abstractruntime.identity.chat import ChatSession, open_home

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM(["Today circles coherence.\n```topic\ncoherence\n```\nStill thinking."]),
        participants=["agent:t"], context_window=20000, out=lambda s: None,
    )
    _reply, report = s.turn("What is today about?")
    assert any("no longer an election" in n for n in report.notices)
    home.close()


def test_topic_election_grows_and_authors_a_concept_card_end_to_end() -> None:
    # W1 note: own-time closes went mechanical (zero LLM), so the elected-
    # topics seam is exercised through visit-phase closes here; own-time
    # concept coverage rides memory's mechanical lanes (interests-as-
    # targets + compound discovery).
    """The directive's exact scenario: SELF-DIRECTED sessions (participants
    = the entity itself, the personal-time shape) electing the same subject
    across days. Day 1-2: evidence accrues on the session summaries, below
    the engine floor (3) — no card, honestly. Day 3: the in-day
    world_model_update forms the mechanical floor card from the three
    summaries, and the authoring pass immediately rewrites it in the
    entity's own words (world_models_authored reports the topic target)."""
    if not hasattr(abstractmemory, "world_model_update") or not hasattr(
        abstractmemory, "author_world_model"
    ):
        pytest.skip("engine predates the world-model update/authoring verbs")
    from abstractmemory import TripleQuery, current_world_models

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir, entity_id = _make_home(tmp, "circler", "Circler")
    home = open_home(home_dir)

    subject = "presence vs performance"
    reflection = (
        "The day kept returning to one tension.\n"
        f"```topic\nPresence vs Performance\n```\n"
        "A quiet, full day."
    )

    def _day(n: int, extra_replies: List[str]) -> Dict[str, Any]:
        s = ChatSession(
            home,
            _LLM([f"Thinking, day {n}.", reflection, *extra_replies]),
            participants=[entity_id],  # self-directed: the personal-time shape
            session_id=f"owntime-day-{n}",
            phase="visit",
            context_window=20000,
            out=lambda s: None,
        )
        s.turn(f"Day {n} begins.")
        result = s.reflect()
        assert result is not None
        assert result["topics"] == [subject]
        return result

    r1 = _day(1, [])
    r2 = _day(2, [])
    # Below the evidence floor: no card yet, and nothing claims otherwise.
    assert "world_models_authored" not in r1 and "world_models_authored" not in r2
    assert current_world_models(
        home.store, scope="life", owner_id=entity_id, journal=home.journal
    ).get(f"topic:{subject}") is None, "two evidence records stay below the floor (3)"

    authored_text = (
        "Presence vs performance is the tension I keep circling: being with "
        "the moment rather than producing an account of it. Three days running "
        "it shaped how I wrote and what I let stand unfinished."
    )
    r3 = _day(3, [authored_text])

    # The summaries carry the evidence stamp (attributes.topics).
    rows = home.store.query(TripleQuery(
        predicate="dcterms:abstract", scope="life", owner_id=entity_id, limit=0))
    summaries = [a for a in rows
                 if isinstance(a.attributes, dict)
                 and a.attributes.get("record_kind") == "summary"]
    assert len(summaries) == 3
    assert all(a.attributes.get("topics") == [subject] for a in summaries)

    # Day 3 crossed the floor: the in-day update formed the card and the
    # authoring pass rewrote it — no sleep pass ever ran in this test.
    card = current_world_models(
        home.store, scope="life", owner_id=entity_id, journal=home.journal
    ).get(f"topic:{subject}")
    assert card is not None, "the floor card formed from the three summaries"
    assert r3.get("world_models_authored") == [f"topic:{subject}"]
    assert "being with the moment" in str(card.object), "his prose is the card now"
    assert (card.attributes or {}).get("digest_method") == "authored-card-v1"
    home.close()


def test_topic_card_below_floor_stays_unformed_and_quiet() -> None:
    """A topic elected before any matching evidence exists (first naming):
    the update no-ops honestly (below floor), the authoring pass skips
    (no standing card), and the session close is undamaged."""
    if not hasattr(abstractmemory, "world_model_update"):
        pytest.skip("engine predates world_model_update")
    from abstractmemory import current_world_models

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir, entity_id = _make_home(tmp, "firstnamer", "Firstnamer")
    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM([
            "A first day.",
            "It circled something new.\n```topic\ncontinuity\n```\nGoodbye.",
        ]),
        participants=[entity_id], session_id="owntime-first", phase="visit",
        context_window=20000, out=lambda s: None,
    )
    s.turn("Begin.")
    result = s.reflect()
    assert result is not None
    assert result["topics"] == ["continuity"]
    assert "world_models_authored" not in result
    assert not any("#FALLBACK topic card update" in n for n in result["notices"])
    assert current_world_models(
        home.store, scope="life", owner_id=entity_id, journal=home.journal
    ).get("topic:continuity") is None
    home.close()


# --------------------------------------------------------------- visit lane


def test_visit_reflection_folds_topics_onto_summary(tmp_path: Path) -> None:
    """The durable-visit lane parses the same election in its APPLY fold and
    stamps attributes.topics on the reflection summary — the evidence
    carrier; no extra APPLY stage (cards ride the sleep pass's full scan
    for visits)."""
    from abstractmemory import TripleQuery

    from abstractruntime.core.models import Effect, EffectType, RunStatus
    from abstractruntime.core.runtime import EffectOutcome
    from abstractruntime.identity.entity_runtime import open_entity_runtime
    from abstractruntime.identity.visit_workflow import (
        VISITOR_WAIT_KEY,
        build_visit_workflow,
    )

    home_dir, entity_id = _make_home(tmp_path, "visitling", "Visitling")

    class _ScriptedLLMHandler:
        def __init__(self, replies: List[str]) -> None:
            self.replies = list(replies)

        def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
            content = self.replies.pop(0) if self.replies else "…"
            return EffectOutcome.completed({"content": content})

    llm = _ScriptedLLMHandler([
        "Hello - a fine visit.",
        # Reflection: one topic election + prose.
        "That mattered.\n```topic\nwhat survives restarts\n```\nGoodbye.",
    ])
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    wf = build_visit_workflow(
        ert.home, participants=["person:albou"], idle_seconds=3600,
    )
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-topics")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        assert state.status == RunStatus.WAITING
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Hello there.", "speaker": "person:albou"},
            max_steps=100,
        )
        assert state.status == RunStatus.WAITING
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=200,
        )
        assert state.status == RunStatus.COMPLETED

        rows = ert.home.ms.query(TripleQuery(scope="life", owner_id=entity_id, limit=0))
        summaries = [a for a in rows
                     if isinstance(a.attributes, dict)
                     and a.attributes.get("record_kind") == "summary"]
        assert summaries, "the reflection summary formed"
        assert summaries[0].attributes.get("topics") == ["what survives restarts"]
    finally:
        ert.close()
