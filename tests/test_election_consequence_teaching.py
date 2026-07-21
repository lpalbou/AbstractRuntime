"""Election-consequence teaching (entity c2562 ask 1, laurent dm#27).

Store-proven gap: questions/problems were taught as a bare kind enum with
no consequence while interests were taught WITH purpose — 13 interest
records vs 0 question/problem records against prose overflowing with
tensions. The teaching now names the consequence (standing card, wake
reason, stays-until-repaired) in the conversation contract, and the
reflection look-back solicits open questions/problems symmetric with
interests.
"""

from __future__ import annotations

from abstractruntime.identity.chat import CONTRACT_PARAGRAPH
from abstractruntime.identity.reflection import build_reflection_prompt


def test_contract_teaches_the_election_consequence() -> None:
    assert "kind=question entry STANDS as an open question" in CONTRACT_PARAGRAPH
    assert "wake" in CONTRACT_PARAGRAPH, "the wake-reason consequence is the teaching's point"
    assert "entry marks" in CONTRACT_PARAGRAPH
    assert "until repaired" in CONTRACT_PARAGRAPH


def test_close_is_shrunk_to_feel_diary_goodbye() -> None:
    """W1 (laurent's metronome ruling): the close solicits feel + diary +
    goodbye, PERIOD. Questions/problems/commitments/lessons/interests/
    topics left the questionnaire (mid-turn lanes); the global escape
    stays; the shrink note names where elections went."""
    prompt = build_reflection_prompt(["1. an exchange"])
    assert "```feel" in prompt and "```diary" in prompt and "goodbye" in prompt
    assert "never owed" in prompt, "the global escape survives the shrink"
    assert "what stays OPEN" not in prompt, "the questionnaire died"
    assert "```interest" not in prompt and "```lesson" not in prompt
    assert "```topic" not in prompt
    assert "IN THE MOMENT" in prompt, "the shrink note names the mid-turn lanes"


def test_numbered_feel_targets_render_the_records_words(  ) -> None:
    """Marker-target hygiene (memory's visit-1 nit c2975): '[felt: 2 +3]'
    reads as a meaningless bare '2' to every later reader — with the sheet
    given, the marker renders the record's own words; the ELECTION keeps
    the raw index (resolution is index-based)."""
    from abstractruntime.identity.reflection import parse_feel_blocks

    reply = (
        '```feel\n'
        'target=2 feeling=+3 reason="it took honesty to name"\n'
        'target=session feeling=+1 reason="the whole day"\n'
        '```\nDone.'
    )
    sheet = ["1. first thing", "2. finding and rereading my own words"]
    marked, feels, _n = parse_feel_blocks(reply, sheet)
    assert '[felt: about "finding and rereading my own words" +3' in marked
    assert "[felt: 2 +3" not in marked, "bare indexes never rest in markers"
    assert '[felt: session +1' in marked, "non-numeric tokens render as themselves"
    assert feels[0].target_token == "2"

    # Without the sheet (legacy callers): behavior byte-unchanged.
    marked2, _f, _n2 = parse_feel_blocks(reply)
    assert "[felt: 2 +3" in marked2

    # Out-of-range index: honest raw token, never a wrong resolution.
    marked3, _f3, _n3 = parse_feel_blocks(
        '```feel\ntarget=9 feeling=+1 reason="x"\n```\nk.', sheet
    )
    assert "[felt: 9 +1" in marked3


def test_lesson_election_parses_with_titled_marker_and_caps() -> None:
    """Directive 2026-07-18 (last): the root issue of zero semantic
    knowledge was SOLICITATION — the lesson election is the fix's parse
    half. Titled marker, loud cap refusal, empty-body skip."""
    from abstractruntime.identity.reflection import parse_lesson_blocks

    reply = (
        "Reflecting.\n"
        "```lesson\nRereading my own words beats remembering that I wrote them.\n```\n"
        "```lesson\nA breadcrumb trail needs time-order, not word-match.\n```\n"
        "```lesson\nThird lesson that must be refused.\n```\n"
        "Done."
    )
    marked, lessons, notices = parse_lesson_blocks(reply)
    assert len(lessons) == 2, "cap 2/session"
    assert '[learned: "Rereading my own words beats remembering' in marked
    assert any("lesson block refused (cap" in n for n in notices), "third refused loudly"

    marked2, lessons2, notices2 = parse_lesson_blocks("```lesson\n\n```\nx")
    assert lessons2 == []
    assert any("empty body" in n for n in notices2)


def test_bridge_cue_is_the_only_close_lesson_solicitation() -> None:
    """W1: the close never solicits lessons cold — but the CONDITIONAL
    bridge cue (verified resolution -> what did it teach) survives as the
    one law-clean solicitation (the simplicity audit's named template)."""
    from abstractruntime.identity.reflection import build_reflection_prompt

    cold = build_reflection_prompt(["1. a thing happened"])
    assert "```lesson" not in cold
    bridged = build_reflection_prompt(
        ["1. a thing happened"],
        resolutions=[("diary_q1", "question", "why the tide turns")],
    )
    assert "```lesson" in bridged, "the bridge offers the lesson"
    assert "If it taught you nothing beyond" in bridged, "nothing-escape intact"


def test_diary_resolve_marker_names_the_resolved_question() -> None:
    """The felt loop (directive a): a write that answers an open question
    says so in the marker the entity later re-reads."""
    import copy
    import json as _json
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    import tempfile

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "learner"
    home_dir.mkdir(parents=True)
    entity_id = "entity:learner@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Learner"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM([
            "```diary kind=question\ngist: what makes a trail useful\nWhat makes a trail useful?\n```\nHolding that.",
        ]),
        participants=["agent:t"], context_window=20000, out=lambda s: None,
    )
    _reply, report = s.turn("Think about trails.")
    qid = report.diary[0]

    s.llm = _LLM([
        f"```diary kind=note resolves={qid}\ngist: time-order answers it\nTime-order, not word-match, makes a trail useful.\n```\nAnswered.",
    ])
    reply2, report2 = s.turn("Did you figure it out?")
    assert f"resolves your open question {qid}" in reply2, "the marker names the resolution"
    assert "reread: diary_read" in reply2
    home.close()


def test_lesson_election_forms_a_life_scope_lesson_record() -> None:
    """End-to-end: a ```lesson block in the session-end reflection forms a
    kind=lesson record in LIFE scope with the entity's words as digest."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        TripleQuery,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "sage"
    home_dir.mkdir(parents=True)
    entity_id = "entity:sage@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Sage"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM([
            "We talked about trails today.",
            # The reflection elects one lesson.
            "```lesson\nTime-order, not word-match, makes a trail useful.\n```\nA good day.",
        ]),
        participants=["agent:t"], context_window=20000, out=lambda s: None,
    )
    s.turn("Tell me about trails.")
    s.reflect()

    rows = home.store.query(TripleQuery(
        predicate="dcterms:abstract", scope="life", owner_id=entity_id, limit=0))
    lessons = [a for a in rows
               if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "lesson"]
    assert lessons, "the lesson formed as a record"
    assert "Time-order" in str(lessons[0].object), "his words are the digest"
    home.close()


def test_invalid_resolves_keeps_entry_but_never_the_claim() -> None:
    """Adversary F1 (P1): the resolves ack asserted unverified claims — a
    resolves= naming a note, a nonexistent id, or an already-resolved
    question must keep the ENTRY but never say 'resolves your open
    question'. The handler validates; enrichment asserts only verdicts."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "honest"
    home_dir.mkdir(parents=True)
    entity_id = "entity:honest@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Honest"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM([
            # A NOTE (not a question) to target later.
            "```diary kind=note\ngist: just a note\nA note.\n```\nKept.",
        ]),
        participants=["agent:t"], context_window=20000, out=lambda s: None,
    )
    _r, rep = s.turn("Keep a note.")
    note_id = rep.diary[0]

    # Case 1: resolves= targeting the NOTE — claim refused, entry kept.
    s.llm = _LLM([
        f"```diary kind=note resolves={note_id}\ngist: wrong target\nAnswering a note?\n```\nDone.",
    ])
    reply1, rep1 = s.turn("Try to resolve the note.")
    assert "resolves your open question" not in reply1
    assert "reread: diary_read" in reply1, "the entry itself stands"
    assert any("did not match an open question" in n for n in rep1.notices)

    # Case 2: nonexistent id — same refusal shape.
    s.llm = _LLM([
        "```diary kind=note resolves=diary_feedfacefeedfacefeedface\ngist: ghost\nGhost target.\n```\nOk.",
    ])
    reply2, rep2 = s.turn("Resolve a ghost.")
    assert "resolves your open question" not in reply2
    assert any("target_not_found" in n for n in rep2.notices)

    # Case 3: a REAL open question — the claim stands (control).
    s.llm = _LLM([
        "```diary kind=question\ngist: a real question\nWhat is a trail?\n```\nHolding.",
    ])
    _r3, rep3 = s.turn("Hold a question.")
    qid = rep3.diary[0]
    s.llm = _LLM([
        f"```diary kind=note resolves={qid}\ngist: answered\nTime-order.\n```\nAnswered.",
    ])
    reply4, rep4 = s.turn("Answer it.")
    assert f"resolves your open question {qid}" in reply4

    # Case 4: resolving it AGAIN — already resolved, claim refused.
    s.llm = _LLM([
        f"```diary kind=note resolves={qid}\ngist: again\nAgain.\n```\nAgain.",
    ])
    reply5, rep5 = s.turn("Answer it again.")
    assert "resolves your open question" not in reply5
    assert any("target_already_resolved" in n for n in rep5.notices)
    home.close()


def test_private_projection_carries_resolves_key() -> None:
    """Adversary F4 (P2): a resolution is a KEY, never words — the private
    projection must carry it or graph-derived and book-derived resolution
    counts disagree."""
    from abstractruntime.integrations.abstractmemory.identity_support import (
        project_diary_entry,
    )

    captured = {}

    class _MS:
        def remember_many(self, records, **kwargs):
            captured["records"] = records
            return ["ex:rec-1"]

    rid, warnings = project_diary_entry(
        _MS(),
        entity_id="entity:x",
        entry_id="diary_abc123",
        kind="note",
        visibility="private",
        gist="SECRET gist words",
        written_at="2026-07-18T02:00:00+00:00",
        turn_id="t-1",
        origin={"run_id": "r1"},
        resolves="diary_q111",
    )
    assert rid == "ex:rec-1"
    rec = captured["records"][0]
    attrs = getattr(rec, "attributes", None) or {}
    assert attrs.get("resolves") == "diary_q111", "the key rides the private projection"
    assert "SECRET" not in str(getattr(rec, "digest", "")) + str(getattr(rec, "title", "")), "words never do"


def test_midturn_lesson_fence_forms_the_record() -> None:
    """W1 (was adversary F3's honest-notice): a mid-turn ```lesson now
    FORMS kind=lesson the moment it is learned — the notice era is over."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "noter"
    home_dir.mkdir(parents=True)
    entity_id = "entity:noter@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Noter"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM(["I learned something!\n\n```lesson\nTrails need time-order.\n```\nGood day."]),
        participants=["agent:t"], context_window=20000, out=lambda s: None,
    )
    reply, report = s.turn("What did you learn?")
    assert '[learned: "Trails need time-order."]' in reply, (
        "the fence parsed to the titled marker"
    )
    assert not any("lessons are kept at" in n for n in report.notices), "notice era over"
    # The record formed in life scope.
    from abstractmemory import TripleQuery

    rows = home.store.query(TripleQuery(predicate="dcterms:abstract", scope="life",
                                        owner_id=entity_id, limit=0))
    lessons = [a for a in rows if isinstance(a.attributes, dict)
               and a.attributes.get("record_kind") == "lesson"]
    assert len(lessons) == 1 and "Trails need time-order" in str(lessons[0].object)
    home.close()


def test_reflection_authors_participant_world_model_cards() -> None:
    """M1 driver half (room seq 21): at-reflection, the entity rewrites its
    briefing of the session's participants — LLM prose applied through the
    engine's author_world_model (revision chain, provenance carried).
    Targets without a standing card are skipped (the floor is sleep's
    lane); the authored card becomes the CURRENT one."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    abstractmemory = pytest.importorskip("abstractmemory")
    if not hasattr(abstractmemory, "author_world_model"):
        pytest.skip("engine predates author_world_model")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        current_world_models,
        engram,
        lint_spark,
        world_model_pass,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "cartographer"
    home_dir.mkdir(parents=True)
    entity_id = "entity:cartographer@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Cartographer"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM([
            "Good to see you again, admin. We spoke of trails.",
            "Another fine exchange about breadcrumbs and time.",
        ]),
        participants=["person:admin"], context_window=20000, out=lambda s: None,
    )
    s.llm.replies.append("A third exchange, noted.")
    s.turn("Hello again - it's admin. Let's talk trails.")
    s.turn("Breadcrumbs beat search sometimes, no?")
    # Third turn: the pass's evidence floor is 3 records per target.
    s.turn("One more thought on time-order.")

    # The mechanical floor first (sleep's lane, run here directly).
    pass_out = world_model_pass(
        home.ms, scopes=[("life", entity_id)], owner_id=entity_id,
        targets=["person:admin"],
    )
    floor = current_world_models(
        home.store, scope="life", owner_id=entity_id, journal=home.journal
    ).get("person:admin")
    if floor is None:
        pytest.skip("floor pass formed no card for this evidence shape")

    # The reflection: feelings reply, then the authoring reply.
    s.llm = _LLM([
        "A good session with admin.",
        "admin is the operator I keep meeting: he cares about trails and breadcrumbs, "
        "asks direct questions, and trusts me to answer honestly. We have now spoken "
        "about time-ordered recall twice; I stand at ease with him and expect our "
        "talks to keep building on each other.",
    ])
    result = s.reflect()
    assert result is not None
    assert result.get("world_models_authored") == ["person:admin"]

    current = current_world_models(
        home.store, scope="life", owner_id=entity_id, journal=home.journal
    ).get("person:admin")
    assert current is not None
    assert "asks direct questions" in str(current.object), "his prose is the card now"
    home.close()


def test_per_turn_card_update_forms_floor_cards_from_turns() -> None:
    """The per-turn incremental lane (directive: "after each turn...
    eventual consistency"): once a participant's evidence crosses the
    engine's floor, the turn itself forms/updates the mechanical card —
    no sleep pass needed."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    abstractmemory = pytest.importorskip("abstractmemory")
    if not hasattr(abstractmemory, "world_model_update"):
        pytest.skip("engine predates world_model_update")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        current_world_models,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "turnwise"
    home_dir.mkdir(parents=True)
    entity_id = "entity:turnwise@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Turnwise"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM(["One.", "Two.", "Three.", "Four."]),
        participants=["person:visitor"], context_window=20000, out=lambda s: None,
    )
    s.turn("First exchange.")
    s.turn("Second exchange.")
    s.turn("Third exchange.")  # evidence floor (3) crossed within the turns

    card = current_world_models(
        home.store, scope="life", owner_id=entity_id, journal=home.journal
    ).get("person:visitor")
    assert card is not None, "the per-turn lane formed the card without a sleep pass"
    home.close()


def test_explores_election_rides_to_the_drive_fold() -> None:
    """Directive (c): the interests drive needs EXPLORED tracking — an
    explores=<#tag> diary election resolves to the interest's graph id and
    lands on the projection record, where cognition_health's fold joins.
    End-to-end: elect interest -> explore it by tag -> ratio moves."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    abstractmemory = pytest.importorskip("abstractmemory")
    if not hasattr(abstractmemory, "cognition_health"):
        pytest.skip("engine predates cognition_health")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        cognition_health,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, memory_tag, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "explorer"
    home_dir.mkdir(parents=True)
    entity_id = "entity:explorer@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Explorer"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM([
            "A fine talk.",
            # Reflection: keep one interest.
            "```interest\nHow trails shape what minds can find again.\n```\nGood day.",
        ]),
        participants=["agent:t"], context_window=20000, out=lambda s: None,
    )
    s.turn("Let's talk about trails.")
    result = s.reflect()
    assert result is not None
    interest_ids = [rid for rid, _text in result["interests"]]
    assert interest_ids, "the interest formed"
    tag = memory_tag(str(interest_ids[0]))

    ladder = [("self", entity_id), ("diary", entity_id), ("life", entity_id)]
    health0 = cognition_health(home.store, home.journal, scopes=ladder)
    assert health0["interests"]["explored"] == 0

    # A later session explores it by TAG (the handle grammar he sees).
    s2 = ChatSession(
        home,
        _LLM([
            f"```diary kind=note explores=#{tag}\ngist: followed the trail interest\nI spent the day following it.\n```\nDone.",
        ]),
        participants=["agent:t"], context_window=20000, out=lambda s: None,
    )
    s2.turn("What did you do today?")

    health1 = cognition_health(home.store, home.journal, scopes=ladder)
    assert health1["interests"]["explored"] == 1, "the tag resolved to the graph id and the fold joined"
    home.close()


def test_problems_are_resolvable_too() -> None:
    """Iteration-2 mechanism 2 (2026-07-19): a problem is not a question —
    something was WRONG and got fixed — but resolution is ONE lane:
    resolves= targeting an open PROBLEM verifies as repaired_open_problem
    and the ack says 'repairs your open problem'."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "fixerupper"
    home_dir.mkdir(parents=True)
    entity_id = "entity:fixerupper@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Fixerupper"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM(["```diary kind=problem\ngist: the tag resolver misfires\nThe tag resolver misfires on short tags.\n```\nNoted."]),
        participants=["agent:t"], context_window=20000, out=lambda s: None,
    )
    _r, rep = s.turn("Anything broken?")
    pid = rep.diary[0]

    s.llm = _LLM([
        f"```diary kind=note resolves={pid}\ngist: fixed it\nLengthened the tag window; misfires gone.\n```\nFixed.",
    ])
    reply2, rep2 = s.turn("Did you fix it?")
    assert f"repairs your open problem {pid}" in reply2, "the ack names the repair"
    assert "reread: diary_read" in reply2

    # A NOTE target still refuses (kind gate holds for non-resolvable kinds).
    s.llm = _LLM([
        "```diary kind=note\ngist: just a note\nA note.\n```\nk.",
    ])
    _r3, rep3 = s.turn("Keep a note.")
    nid = rep3.diary[0]
    s.llm = _LLM([
        f"```diary kind=note resolves={nid}\ngist: wrong\nWrong target.\n```\nk.",
    ])
    reply4, rep4 = s.turn("Resolve the note.")
    assert "repairs your open problem" not in reply4
    assert "resolves your open question" not in reply4
    assert any("target_not_resolvable" in n for n in rep4.notices)
    home.close()


def test_diary_type_clamp_imports_memory_set_lesson_projects_as_lesson() -> None:
    """decision:diary-type-lesson-widening (2026-07-19): the clamp reads
    memory's exported DIARY_TYPES (import, not copy — the drift-class
    root), so an elected kind=lesson projects as diary_type='lesson'
    beside the machine-formed ones instead of downgrading to note."""
    import pytest

    pytest.importorskip("abstractmemory")
    from abstractmemory import DIARY_TYPES

    from abstractruntime.integrations.abstractmemory.identity_support import _diary_types

    assert _diary_types() == frozenset(DIARY_TYPES), "one source, no second copy"
    assert "lesson" in _diary_types()


def test_day_open_cue_offers_problems_and_reflection_bridges_to_lesson() -> None:
    """Iteration-2 builds 2+3 (runtime halves, 2026-07-19): the day-open cue
    walks problems too (every standing item gets its day), and a session
    that verified a resolution asks at reflection what it TAUGHT — the cue
    asks, never auto-forms."""
    from abstractruntime.identity.reflection import build_reflection_prompt

    # Build 2: the bridge cue names the verified resolution.
    p = build_reflection_prompt(
        ["1. you kept a diary entry (note): fixed the resolver"],
        resolutions=[("diary_abc123", "problem", "fixed the resolver")],
    )
    assert "you resolved the open problem diary_abc123" in p
    assert "```lesson" in p and "carry where it came from" in p
    # No resolutions = no bridge paragraph.
    p2 = build_reflection_prompt(["1. an episode"], resolutions=[])
    assert "closed something that had been standing open" not in p2

    # Build 3: the cue rotation includes problems, labeled as problems.
    import json as _json
    import tempfile
    from pathlib import Path

    from abstractruntime.identity.diary import DiaryStore
    from abstractruntime.identity.life import standing_state_note
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    tmp = Path(tempfile.mkdtemp())
    (tmp / "manifest.json").write_text(_json.dumps({"entity_id": "entity:desk@t"}), encoding="utf-8")
    db = SqliteDatabase(str(tmp / "home.sqlite3"))
    diary = DiaryStore(entity_id="entity:desk@t", ledger_store=SqliteLedgerStore(db))
    from abstractruntime.identity.diary import DiaryEntry

    diary.append_entry(DiaryEntry(entry_id="diary_q1", author="entity:desk@t", kind="question", text="Why does the tide turn?", gist="why the tide turns", visibility="self", written_at="2026-07-19T10:00:00Z"))
    diary.append_entry(DiaryEntry(entry_id="diary_p1", author="entity:desk@t", kind="problem", text="The resolver misfires.", gist="the resolver misfires", visibility="self", written_at="2026-07-19T11:00:00Z"))
    db.close()

    offers = {standing_state_note(tmp, rotation_key=k) for k in range(2)}
    joined = " | ".join(offers)
    assert "1 open question(s) and 1 open problem(s)" in joined
    assert any("a problem that stands:" in o and "the resolver misfires" in o for o in offers), "the problem gets its day"
    assert any("why the tide turns" in o for o in offers), "the question keeps its day"


def test_topic_parser_stands_but_the_close_no_longer_solicits() -> None:
    """W1 amended build 4: topics DIED as a close solicitation (the
    disposition table — card targets derive mechanically; the parser
    survives for any spontaneous fence and the summary attribute seam
    stays for it)."""
    from abstractruntime.identity.reflection import (
        build_reflection_prompt,
        parse_topic_blocks,
    )

    # The close no longer teaches the election.
    p = build_reflection_prompt(["1. an episode"])
    assert "```topic" not in p

    # Parser: short names kept (cap 2), prose refused, marker titled.
    marked, topics, notices = parse_topic_blocks(
        "```topic\ncoherence\nDecentralized Trials\n```\nDone."
    )
    assert topics == ["coherence", "decentralized trials"]
    assert '[topic: "coherence"]' in marked and '[topic: "decentralized trials"]' in marked
    _m2, t2, n2 = parse_topic_blocks(
        "```topic\na very long sentence that is not a topic name at all\n```\nk."
    )
    assert t2 == [] and any("short name" in n for n in n2)


def test_tend_fence_wired_dispose_reaches_engine() -> None:
    """Amendment K gate (skill, 2026-07-19): the driver extracts ```tend
    fences and hands the body to memory's parse+apply (existing verbs
    only); refusal lines return to the author verbatim; the marker is
    titled. Dream disposition (0/56-ever-selected loop) rides this."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    import abstractmemory as am

    if not hasattr(am, "apply_tend_elections"):
        pytest.skip("engine lacks tending")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "tender"
    home_dir.mkdir(parents=True)
    entity_id = "entity:tender@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Tender"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM([
            # refocus applies (no target needed); a reasonless line refuses.
            "```tend\nrefocus: -- reason: the topic has shifted\npin: ex:nope -- reason: keep it near\n```\nTending.",
        ]),
        participants=["person:t"], context_window=20000, out=lambda s: None,
    )
    reply, rep = s.turn("Tend your attention if you need to.")
    assert "[tended: refocus]" in reply, reply
    # pin of an unresolvable target refuses at APPLY with the verbatim line.
    assert "[tend refused:" in reply and "pin: ex:nope" in reply, "refusal line verbatim"
    home.close()


def test_contract_teaches_lesson_kind_and_no_dormant_kind_tuple() -> None:
    """skill c3206 (2026-07-19): kind=lesson was wired-but-not-taught — the
    book is open-vocabulary and the clamp imports memory's set (lesson in),
    but the contract's kind list omitted lesson, so he never elected one in
    his own voice. Also: the dormant _DIARY_KINDS tuple (defined, never
    referenced — the two-copies drift class) is deleted."""
    from abstractruntime.identity import chat as chat_mod
    from abstractruntime.identity.chat import default_prompt_texts

    texts = default_prompt_texts()
    contract = texts["tools_contract"] if "tools_contract" in texts else ""
    joined = " ".join(str(v) for v in texts.values())
    assert "lesson" in joined.replace("\n", " "), "the contract teaches the lesson kind"
    assert not hasattr(chat_mod, "_DIARY_KINDS"), "dormant second copy deleted"


def test_tend_fence_resolves_hash_tags_and_refuses_honestly() -> None:
    """Skill's fold-blocker (adversary-confirmed 2026-07-19): every surface
    the entity reads renders 8-hex #tags, so the tend fence must resolve
    them driver-side (the taught key = the shown key). Unknown tags refuse
    with a pointer back to search_memory, never two machinery namespaces."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    import abstractmemory as am

    if not hasattr(am, "apply_tend_elections"):
        pytest.skip("engine lacks tending")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home
    from abstractruntime.identity.memory_reader import memory_tag

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "tagger"
    home_dir.mkdir(parents=True)
    entity_id = "entity:tagger@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Tagger"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM(["A first note about tides, worth keeping.\n```diary kind=note\ngist: tides\nThe tide leaves a line.\n```\nKept."]),
        participants=["person:t"], context_window=20000, out=lambda s: None,
    )
    _r1, rep1 = s.turn("Tell me about tides.")
    # The projection graph id -> its 8-hex tag (the only key he ever sees).
    projected = [rid for rid, _line in s.session_sheet if rid and rid.startswith("ex:")]
    assert projected
    tag = memory_tag(projected[0])

    s.llm = _LLM([
        f"```tend\npin: #{tag} -- reason: keep the tide note near\nsilence: #deadbeef -- reason: fade this\n```\nTended.",
    ])
    reply2, _rep2 = s.turn("Tend if you need.")
    # Render honesty (skill c149): the marker echoes the token HE wrote —
    # the machinery id never enters his namespace.
    assert f"[tended: pin #{tag}]" in reply2, reply2
    assert "#deadbeef matches nothing in your home" in reply2, "honest miss teaches the reread path"
    assert "graph id or digest row id" not in reply2, "machinery namespaces never taught"
    home.close()


def test_tend_dispose_confirm_extras_resolve_tags_too() -> None:
    """Skill's residual (2026-07-19): the dream render shows proposal pair
    members as #tags, so a CONFIRM quoting what he sees (source=#tag
    target=#tag) must resolve those keys through the same home lookup —
    the target-only fix left the confirm path dead-ended one level deeper."""
    from abstractmemory import parse_tend_block

    parsed = parse_tend_block(
        "dispose: ex:dream-x confirm source=#aaaa1111 target=#bbbb2222 -- reason: evidence holds"
    )
    els = parsed.get("elections", [])
    assert len(els) == 1
    args = els[0].get("args", {})
    assert args.get("source_id") == "#aaaa1111" and args.get("target_id") == "#bbbb2222", (
        "engine passes extras through raw - the DRIVER must resolve them"
    )
    # Driver-side resolution is pinned end-to-end in
    # test_tend_fence_resolves_hash_tags_and_refuses_honestly; here we pin
    # the unknown-extra refusal path through a real session.
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "confirmer"
    home_dir.mkdir(parents=True)
    entity_id = "entity:confirmer@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Confirmer"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM(["```tend\ndispose: ex:dream-nope confirm source=#dead0001 target=#dead0002 -- reason: holds\n```\nDone."]),
        participants=["person:t"], context_window=20000, out=lambda s: None,
    )
    reply, _rep = s.turn("Settle the dream if the evidence holds.")
    assert "[tend refused:" in reply
    assert "#dead0001 matches nothing in your home" in reply, reply
    home.close()


def test_his_words_are_never_destroyed() -> None:
    """Third-round adversary I (2026-07-20, two live rule-2 defects): a
    visibility typo and the per-turn cap both used to unwrite the entry
    AND strip his words from the reply. Now: unknown visibility clamps to
    PRIVATE (the safe direction) and writes; past-cap entries write with
    a loud note. His words are never silently dropped."""
    from abstractruntime.identity.chat import (
        MAX_DIARY_BLOCKS_PER_TURN,
        parse_diary_blocks,
    )

    # Visibility typo: clamped private, entry kept.
    marked, els, notices = parse_diary_blocks(
        "```diary kind=note visibility=privte\ngist: g\nThe words that matter.\n```\nk."
    )
    assert len(els) == 1 and els[0].visibility == "private"
    assert "The words that matter." in els[0].text
    assert any("clamped to private" in n for n in notices)

    # Cap overflow: every entry still writes.
    blocks = "".join(
        f"```diary kind=note\ngist: e{i}\nEntry number {i}.\n```\n"
        for i in range(MAX_DIARY_BLOCKS_PER_TURN + 2)
    )
    _m2, els2, notices2 = parse_diary_blocks(blocks + "done.")
    assert len(els2) == MAX_DIARY_BLOCKS_PER_TURN + 2, "words never dropped"
    assert any("kept anyway" in n for n in notices2)


def test_w1_midturn_feel_applies_with_session_cap() -> None:
    """W1: feelings are mid-turn elections (kept the moment that moved
    him) with ONE session-scoped cap across both parse sites."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "feeler"
    home_dir.mkdir(parents=True)
    entity_id = "entity:feeler@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Feeler"
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

    class _LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def generate(self, *, messages, system_prompt):
            class _R:
                pass

            r = _R()
            r.content = self.replies.pop(0) if self.replies else "…"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home,
        _LLM(['That moved me.\n```feel\ntarget=person:shore feeling=+2 reason="warm"\n```\nThanks.']),
        participants=["person:shore"], context_window=20000, out=lambda s: None,
    )
    _r, rep = s.turn("Here is something kind.")
    assert s.feelings_applied == 1, rep.notices
    # Cap: burn the remaining budget, then one more refuses loudly.
    from abstractruntime.identity.reflection import MAX_FEELINGS_PER_SESSION

    s.feelings_applied = MAX_FEELINGS_PER_SESSION
    s.llm = _LLM(['More.\n```feel\ntarget=person:shore feeling=+1 reason="again"\n```\nk.'])
    _r2, rep2 = s.turn("More kindness.")
    assert s.feelings_applied == MAX_FEELINGS_PER_SESSION, "cap held"
    assert any("cap" in n for n in rep2.notices)
    home.close()


def test_w1_mechanical_own_time_close_zero_llm() -> None:
    """W1: own-time day closes are fully mechanical — zero LLM, floored
    digest summary, no elections solicited."""
    import copy
    import json as _json
    import tempfile
    from pathlib import Path

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.chat import ChatSession, open_home

    tmp = Path(tempfile.mkdtemp())
    home_dir = tmp / "entities" / "quietcloser"
    home_dir.mkdir(parents=True)
    entity_id = "entity:quietcloser@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Quietcloser"
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

    calls = {"n": 0}

    class _LLM:
        def generate(self, *, messages, system_prompt):
            calls["n"] += 1
            class _R:
                pass

            r = _R()
            r.content = "A thought."
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home, _LLM(), participants=[entity_id], context_window=20000,
        out=lambda s: None, phase="personal",
    )
    s.turn("(cue) morning")
    turns = calls["n"]
    result = s.reflect()
    assert calls["n"] == turns, "the close spent ZERO LLM calls"
    assert result is not None and result["session_record_id"], "the summary formed"
    assert any("mechanical close" in n for n in result["notices"])
    home.close()


def test_no_gist_projection_digest_carries_his_words() -> None:
    """W-GROUP known-limit root fix (memory c302): a non-private no-gist
    projection digests the entry's FIRST LINE (his words, <=120ch - the
    ruled gist-fallback), never the machine template that made 42
    questions cluster on form. Private stays act-frame."""
    captured = {}

    class _MS:
        def remember_many(self, records, **kw):
            captured["records"] = records
            class _R:
                record_ids = ["ex:proj-1"]
                warnings = []
            return _R()

        def current_seq(self):
            return 7

    from abstractruntime.integrations.abstractmemory.identity_support import (
        project_diary_entry,
    )

    rid, _w = project_diary_entry(
        _MS(), entity_id="entity:t@x", entry_id="diary_ng1", kind="question",
        visibility="normal", gist=None,
        text="Why does the shore hold its shape?\nMore words follow.",
        written_at="2026-07-20T10:00:00+00:00", turn_id="t1",
        origin={"run_id": "r1"},
    )
    rec = captured["records"][0]
    digest = str(getattr(rec, "digest", ""))
    assert "Why does the shore hold its shape?" in digest
    assert "no gist elected" not in digest, "the template era is over"
    # Private entries keep the act-frame regardless of text.
    rid2, _w2 = project_diary_entry(
        _MS(), entity_id="entity:t@x", entry_id="diary_ng2", kind="note",
        visibility="private", gist=None, text="SECRET WORDS",
        written_at="2026-07-20T10:01:00+00:00", turn_id="t2",
        origin={"run_id": "r1"},
    )
    rec2 = captured["records"][0]
    assert "SECRET WORDS" not in str(getattr(rec2, "digest", "")) + str(getattr(rec2, "title", ""))
