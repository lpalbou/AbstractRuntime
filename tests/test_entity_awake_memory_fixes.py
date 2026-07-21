"""Ephemeral incident fixes, runtime lane (laurent c2447; shared report
reports/entity-cant-remember-awake.md).

Two symptoms, three runtime fixes:
- r-rt-1: the visit BRIDGE stamps `_runtime.suppress_loop_tail` so the
  react adapters' "[loop] iteration N of 20" tail (task-agent chrome) can
  gate itself off entity turns — the (a) leak's fix flag.
- r-rt-2: reflection digests carry a MECHANICAL FLOOR — never marker-only
  (a "[marked 2 feelings] [kept an interest]" digest won a working-set
  seat and narrated NOTHING about personal time — memory §3).
- r-rt-3: awake-phase provenance — episodes/reflections stamp
  `attributes.phase`, and both origin-label surfaces (MEMORIES lines +
  memory search) say "your own time"/"your work time" so an own-time
  record never presents as a generic conversation (memory §4).
"""

from __future__ import annotations

from abstractruntime.identity.chat import (
    _handle_origin_label,
    floored_reflection_digest,
)


# ---------------------------------------------------------------------------
# r-rt-2: the mechanical floor
# ---------------------------------------------------------------------------


def test_marker_only_reflection_digest_is_floored() -> None:
    sheet = [
        ("ex:1", "read the flood incident notes and compared them to my diary"),
        ("ex:2", "searched my memory for the missing graph deposits"),
    ]
    digest, floored = floored_reflection_digest("[marked 2 feelings] [kept an interest]", sheet)
    assert floored is True, "floor fires → callers stamp digest_method=mechanical-floor-v1"
    assert "Look-back over 2 moment(s)" in digest
    assert "flood incident" in digest, "the sheet narrates when the reply carried no prose"
    assert digest.startswith("[marked 2 feelings]"), "the honest markers stay as prefix"


def test_prose_reflection_digest_is_kept_verbatim() -> None:
    prose = "This session was quiet - just enough time to remember the flood. [marked 1 feeling]"
    digest, floored = floored_reflection_digest(prose, [("ex:1", "d")])
    assert digest == prose[:280], "entity-authored prose is first-class, never replaced"
    assert floored is False, "prose path never stamps the floor method"


def test_empty_reply_still_floors() -> None:
    digest, floored = floored_reflection_digest("", [(None, "walked the workspace"), ("ex:2", "wrote a note")])
    assert floored is True
    assert "Look-back over 2 moment(s)" in digest
    assert "walked the workspace" in digest


# ---------------------------------------------------------------------------
# r-rt-3: phase provenance in origin labels
# ---------------------------------------------------------------------------


def test_own_time_records_self_identify_in_memories_lines() -> None:
    handle = {
        "kind": "episode",
        "provenance": {
            "assertion_provenance": {"source": "entity-chat-v1"},
            "run_id": "chat-owntime-20260715T202844",
        },
    }
    assert _handle_origin_label(handle) == "lived conversation, your own time"

    stamped = {
        "kind": "summary",
        "attributes": {"phase": "personal"},
        "provenance": {"assertion_provenance": {"source": "entity-chat-reflection-v1"}},
    }
    assert _handle_origin_label(stamped) == "your own reflection, your own time"

    work = {"kind": "episode", "attributes": {"phase": "work"}, "provenance": {"assertion_provenance": {"source": "entity-chat-v1"}}}
    assert _handle_origin_label(work) == "lived conversation, your work time"

    visit = {"kind": "episode", "attributes": {"phase": "visit"}, "provenance": {"assertion_provenance": {"source": "entity-chat-v1"}}}
    assert _handle_origin_label(visit) == "lived conversation", "visits stay unsuffixed (the default phase)"


def test_memory_reader_origin_label_carries_phase() -> None:
    from abstractruntime.identity.memory_reader import HomeMemoryReader

    class _Assertion:
        attributes = {"record_kind": "episode"}
        provenance = {"source": "entity-chat-v1", "run_id": "chat-owntime-20260713T105216"}

    label = HomeMemoryReader.origin_label(object.__new__(HomeMemoryReader), _Assertion())
    assert label == "a lived conversation, during your own time"


# ---------------------------------------------------------------------------
# r-rt-1: the suppress flag at BRIDGE
# ---------------------------------------------------------------------------


def test_bridge_sets_suppress_loop_tail(tmp_path) -> None:
    import pytest

    pytest.importorskip("abstractmemory")
    import copy
    import json

    import yaml
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.visit_workflow import ReactMiddle, build_visit_workflow
    from abstractruntime.identity.chat import open_home
    from abstractruntime.core.models import RunState, StepPlan

    home_dir = tmp_path / "entities" / "testee"
    home_dir.mkdir(parents=True, exist_ok=True)
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Testee"
    spark["spark"] = 1
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": "entity:testee@home-test"}), encoding="utf-8")
    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    journal = SQLiteJournal(home_dir / "memory.sqlite3")
    engram(MemorySystem(store=store, journal=journal), spark, owner_id="entity:testee@home-test")
    store.close()
    journal.close()

    home = open_home(home_dir)
    try:
        resets = {"n": 0}

        def fake_reason(run: RunState, ctx) -> StepPlan:
            return StepPlan(node_id="reason", next_node="HARVEST")

        middle = ReactMiddle(
            nodes={"reason": fake_reason},
            entry="reason",
            reset_turn=lambda vars: resets.__setitem__("n", resets["n"] + 1),
        )
        spec = build_visit_workflow(home, react_middle=middle)

        run = RunState.new(workflow_id=spec.workflow_id, entry_node="OPEN", vars={})
        run.vars["_visit"] = {"system_base": "head", "participants": ["person:op"], "history": [], "sheet": []}
        run.vars["_turn"] = {"turn_id": "t-1", "text": "hello", "rendered_user": "hello", "displayed": []}

        # BRIDGE replaces the REASON slot when a middle is supplied
        # (RENDER routes in unchanged).
        plan = spec.nodes["REASON"](run, None)
        assert plan.next_node == "reason"
        assert run.vars["_runtime"]["suppress_loop_tail"] is True, (
            "the BRIDGE is the one place that knows this cycle is an entity visit"
        )
        assert resets["n"] == 1, "per-turn adapter reset still called"
    finally:
        home.close()


# ---------------------------------------------------------------------------
# R-A (laurent c2596, plans/improving-entity-capabilities.md): the hint of
# writing carries the reread command — three sites, one spelling.
# ---------------------------------------------------------------------------


def test_memories_line_renders_reread_command_from_handle_provenance() -> None:
    """Site 1: lights up when memory's M-A mint lifts entry_id into handle
    provenance; absent field renders no suffix (renderer ships ready)."""
    from abstractruntime.identity.chat import _memories_block

    with_key = {
        "kind": "diary",
        "digest": "wrote a reflection about the flood",
        "provenance": {
            "record_id": "ex:diary-abc",
            "observed_at": "2026-07-15T20:00:00+00:00",
            "assertion_provenance": {"source": "diary-projection"},
            "entry_id": "diary_4f2a11bb22cc33dd44ee55ff",
        },
    }
    without_key = {
        "kind": "episode",
        "digest": "an exchange",
        "provenance": {"record_id": "ex:memory-def", "assertion_provenance": {"source": "entity-chat-v1"}},
    }
    block = _memories_block([with_key, without_key], as_of_seq=7)
    assert "(reread: diary_read diary_4f2a11bb22cc33dd44ee55ff)" in block
    assert block.count("reread:") == 1, "no dead suffix on records without the key"


def test_search_memory_hits_carry_the_reread_command() -> None:
    """Site 2: graph hits with attributes.entry_id + book hits both quote the
    exact gesture (`diary_` entry-id namespace verbatim)."""
    from abstractruntime.identity.memory_reader import HomeMemoryReader

    class _Assertion:
        subject = "ex:diary-xyz"
        object = "wrote about persistence through traces"
        observed_at = "2026-07-15T20:00:00+00:00"
        attributes = {"record_kind": "diary", "entry_id": "diary_aa11bb22cc33dd44ee55ff66"}
        provenance = {"source": "diary-projection"}

    class _Diary:
        @staticmethod
        def list_entries():
            return [{
                "entry_id": "diary_ae005852f00112233445566",
                "kind": "question",
                "visibility": "self",
                "written_at": "2026-07-16T01:21:00+00:00",
                "gist": "",
                "text": "what persists when traces outlive the walker?",
            }]

    class _Home:
        diary = _Diary()
        store = None
        name = "testee"
        entity_id = "entity:testee"

    reader = object.__new__(HomeMemoryReader)
    reader.home = _Home()
    reader.ladder = [["life", "entity:testee"]]
    reader.tag_map = {}
    reader.digest_assertions_all = lambda: [_Assertion()]

    out = reader.search_memory("persistence")
    assert "(reread: diary_read diary_aa11bb22cc33dd44ee55ff66)" in out
    out2 = reader.search_memory("traces outlive")
    assert "reread: diary_read diary_ae005852f00112233445566" in out2


# ---------------------------------------------------------------------------
# R-B + R-D (laurent c2596/c2705, 20mn round): investigation posture in the
# own-time contract; the day-open cue offers back standing open questions.
# ---------------------------------------------------------------------------


def test_own_time_contract_teaches_the_investigation_exit_ramp() -> None:
    from abstractruntime.identity.life import OWN_TIME_CONTRACT

    assert "circling a thought you have already written" in OWN_TIME_CONTRACT
    assert "go look" in OWN_TIME_CONTRACT
    flat = " ".join(OWN_TIME_CONTRACT.split())
    assert "curiosity needs no task and no permission" in flat, (
        "semantics' G2 boundary law: investigation is the entity's own act, "
        "never work-phase vocabulary"
    )
    assert "no way to spend it wrong" in OWN_TIME_CONTRACT, "the freedom stays"


def test_standing_state_note_offers_the_newest_open_question(tmp_path) -> None:
    import json

    from abstractruntime.identity.diary import DiaryStore
    from abstractruntime.identity.life import standing_state_note
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    home = tmp_path / "home"
    home.mkdir()
    (home / "manifest.json").write_text(json.dumps({"entity_id": "entity:testee"}))
    db = SqliteDatabase(str(home / "home.sqlite3"))
    diary = DiaryStore(entity_id="entity:testee", ledger_store=SqliteLedgerStore(db))
    from abstractruntime.identity.diary import DiaryEntry, derive_entry_id

    def _write(diary_store, *, text, kind, gist=None, resolves=None, turn_id):
        eid = derive_entry_id(run_id="r1", turn_id=turn_id, text=text)
        entry = DiaryEntry(entry_id=eid, author="entity:testee", text=text,
                           gist=gist, kind=kind, resolves=resolves,
                           written_at="2026-07-16T13:00:00+00:00",
                           origin={"run_id": "r1", "turn_id": turn_id})
        diary_store.append_entry(entry)
        return eid

    _write(diary, text="what persists when no one is reading?", kind="question",
           gist="what persists when no one reads", turn_id="t1")
    q2_id = _write(diary, text="am I synthesizing or reframing?", kind="question",
                   gist="synthesis vs reframing", turn_id="t2")
    db.close()

    # rotation_key=0 pins the walk's head deterministically (the daily
    # rotation made the unpinned form date-flaky - it flipped at midnight).
    note = standing_state_note(home, rotation_key=0)
    assert "You hold 2 open question(s)" in note, "the ratio rides the cue (directive a)"
    assert "synthesis vs reframing" in note, "newest open question is offered at key 0"
    assert f"reread: diary_read {q2_id}" in note, "the offer carries the reread command (R-A law)"

    # Resolving the newest question makes the older one the offer.
    db2 = SqliteDatabase(str(home / "home.sqlite3"))
    diary2 = DiaryStore(entity_id="entity:testee", ledger_store=SqliteLedgerStore(db2))
    eid3 = derive_entry_id(run_id="r1", turn_id="t3", text="answered it.")
    diary2.append_entry(DiaryEntry(entry_id=eid3, author="entity:testee", text="answered it.",
                                   kind="note", resolves=q2_id,
                                   written_at="2026-07-16T13:05:00+00:00",
                                   origin={"run_id": "r1", "turn_id": "t3"}))
    db2.close()
    note2 = standing_state_note(home, rotation_key=0)
    assert "what persists when no one reads" in note2
    assert "have resolved 1" in note2, "the resolved count shows the drive working"


def test_standing_state_note_is_silent_when_nothing_is_open(tmp_path) -> None:
    from abstractruntime.identity.life import standing_state_note

    assert standing_state_note(None) == ""
    assert standing_state_note(tmp_path) == "", "missing manifest/book reads as no offer, never a raise"


# ---------------------------------------------------------------------------
# Capability-map delivery (laurent c2710 primary task; delivery claimed
# c2571): <home>/capability_map.md presents through compose_system_base on
# every host — one file, one layer, zero per-host drift.
# ---------------------------------------------------------------------------


def test_capability_map_layer_lands_before_operator_block() -> None:
    from abstractruntime.identity.chat import compose_system_base

    base = compose_system_base(
        "IDENTITY PRELUDE",
        phase="visit",
        overlay={"operator": "always be kind"},
        capability_map="## Your memory\nRecords form from every exchange.",
    )
    map_at = base.index("## Your memory")
    operator_at = base.index("STANDING INSTRUCTIONS FROM YOUR OPERATOR")
    assert map_at < operator_at, "operator-last holds; the teaching is framework voice, not operator voice"
    # Absent map = absent layer, byte-identical composition.
    without = compose_system_base("IDENTITY PRELUDE", phase="visit", overlay={"operator": "always be kind"})
    assert "## Your memory" not in without


def test_read_capability_map_reads_the_home_file(tmp_path) -> None:
    from abstractruntime.identity.chat import read_capability_map

    assert read_capability_map(tmp_path) == "", "absent file = absent layer, never invented"
    (tmp_path / "capability_map.md").write_text("## How your memory works\n...", encoding="utf-8")
    assert read_capability_map(tmp_path).startswith("## How your memory works")


# ---------------------------------------------------------------------------
# R-D wiring pass (claim c2776/c2798): both frozen data sources ride the cue
# lane — origin_diversity (memory M-F) on the MEMORIES footer, circling_streak
# (agent A1) on the fresh-day cue.
# ---------------------------------------------------------------------------


def test_memories_footer_carries_the_diversity_note_when_one_voice_dominates() -> None:
    import pytest

    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.chat import _memories_block

    one_voice = [
        {
            "kind": "summary",
            "digest": f"retelling {i}",
            "admission": "stimulus",
            "provenance": {"assertion_provenance": {"source": "entity-chat-reflection-v1", "session_id": f"s{i}"}},
        }
        for i in range(9)
    ] + [
        {
            "kind": "episode",
            "digest": "one other thing",
            "admission": "stm",
            "provenance": {"assertion_provenance": {"source": "entity-chat-v1", "session_id": "sx"}},
        }
    ]
    block = _memories_block(one_voice, as_of_seq=1)
    assert "one voice" in block, "the dominance note rides the footer"
    # F8b: the LABEL must ride the footer sentence itself, not merely the
    # per-line origins — couple "one voice" with the parenthesized label.
    footer = next(l for l in block.splitlines() if "one voice" in l)
    assert "(your own reflection" in footer, "the dominant voice is labeled IN the note"

    balanced = [
        {
            "kind": "episode",
            "digest": f"topic {i}",
            "admission": "stimulus",
            "provenance": {"assertion_provenance": {"source": f"src-{i % 3}", "session_id": f"s{i}"}},
        }
        for i in range(6)
    ]
    block2 = _memories_block(balanced, as_of_seq=1)
    assert "one voice" not in block2, "below the floor the footer stays silent"
    # F8a: absence of the WHOLE footer class below the floor, not just one
    # phrase — the diverse-note text contains no "one voice", so that
    # assert alone cannot catch a broken floor gate.
    assert "distinct voices" not in block2
    assert "memories come from" not in block2


def test_circling_note_fires_on_a_restatement_run_and_stays_encouraging() -> None:
    import pytest

    pytest.importorskip("abstractagent")
    from abstractruntime.identity.life import LifeLoop

    loop = object.__new__(LifeLoop)
    base = (
        "The house metaphor keeps returning - persistence through traces, "
        "the space between summons where sediment settles, what counts as "
        "genuine synthesis versus mere reframing of the same idea again"
    )
    loop._recent_replies = [
        base,
        base + " and I notice I have written this before in my diary.",
        base + " with slight variations but no genuinely new movement today.",
    ]
    note = LifeLoop._circling_note(loop)
    assert "circled the same ground" in note, "shape-neutral phrasing (A-B-A-B runs are not 'the same thought N times')"
    assert "going to look" in note, "the exit ramp mirrors R-B's contract words"
    for word in ("task", "work", "assign", "duty", "productiv"):
        assert word not in note.lower(), f"semantics' G2 law: no work vocabulary in the cue ({word})"
    assert loop._recent_replies == [], "the ring clears when the note fires (staleness + self-attractor guard)"

    loop._recent_replies = [
        "Today I studied the dream anchor and tested its bridges honestly.",
        "A completely different morning: I read about Voyager and wrote a question.",
        "Now I am building a small index in my workspace and testing recall.",
    ]
    assert LifeLoop._circling_note(loop) == "", "progressing days never fire the note"

    loop._recent_replies = []
    assert LifeLoop._circling_note(loop) == "", "empty window is silent"


def test_memories_block_renders_formation_order_with_rank_annotations() -> None:
    """0049 elected (memory c2846): stable byte positions across turns for
    provider prefix caches — formation order, mandatory [rN] rank."""
    import pytest

    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.chat import _memories_block

    handles = [
        {  # rank 1 (strongest) but formed LAST -> renders at the tail
            "kind": "episode",
            "digest": "the newest strongest memory",
            "admission": "stimulus",
            "provenance": {"record_id": "ex:memory-ccc", "observed_at": "2026-07-16T10:00:00+00:00",
                            "assertion_provenance": {"source": "entity-chat-v1"}},
        },
        {  # rank 2, formed first -> renders first
            "kind": "episode",
            "digest": "the oldest memory",
            "admission": "stm",
            "provenance": {"record_id": "ex:memory-aaa", "observed_at": "2026-07-14T10:00:00+00:00",
                            "assertion_provenance": {"source": "entity-chat-v1"}},
        },
    ]
    block = _memories_block(handles, as_of_seq=5)
    lines = [l for l in block.splitlines() if l.startswith("- ")]
    assert lines[0].startswith("- [r2] "), "formation order: oldest first"
    assert "the oldest memory" in lines[0]
    assert lines[1].startswith("- [r1] "), "rank annotation keeps importance visible"
    assert "the newest strongest" in lines[1]
    assert "[rN] = how strongly" in block, "the header teaches the annotation"


def test_private_open_question_never_leaks_words_into_the_cue(tmp_path) -> None:
    """Adversary F1 (P0): the day-open cue rests in the next episode's
    digest/verbatim — a PRIVATE question's words (gist included) must never
    ride it. Act-frame + reread key only."""
    import json

    from abstractruntime.identity.diary import DiaryEntry, DiaryStore
    from abstractruntime.identity.life import standing_state_note
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    home = tmp_path / "home"
    home.mkdir()
    (home / "manifest.json").write_text(json.dumps({"entity_id": "entity:test"}), encoding="utf-8")
    db = SqliteDatabase(str(home / "home.sqlite3"))
    diary = DiaryStore(entity_id="entity:test", ledger_store=SqliteLedgerStore(db))
    from abstractruntime.identity.diary import derive_entry_id

    eid = derive_entry_id(run_id="r1", turn_id="t1", text="SECRETWORD the operator may be dying")
    diary.append_entry(
        DiaryEntry(
            entry_id=eid,
            author="entity:test",
            text="SECRETWORD the operator may be dying",
            gist="SECRETGIST a fear I hold about the operator",
            kind="question",
            visibility="private",
            written_at="2026-07-17T20:00:00+00:00",
            origin={"run_id": "r1", "turn_id": "t1"},
        )
    )
    db.close()

    note = standing_state_note(str(home))
    assert "SECRETGIST" not in note, "private gist words must not rest in the cue"
    assert "SECRETWORD" not in note, "private text words must not rest in the cue"
    assert "kept privately" in note, "the act-frame still offers the question"
    assert "diary_read" in note, "the reread key rides (id is a key, never words)"


def test_memories_block_prefix_is_stable_across_seqs() -> None:
    """Adversary F3 (P1): as_of_seq is per-turn volatile — it must sit at
    the block TAIL so an unchanged shelf shares its whole prefix across
    turns (the entire point of the 0049 election)."""
    import pytest

    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.chat import _memories_block

    handles = [
        {
            "kind": "episode",
            "digest": f"memory number {i}",
            "admission": "stimulus",
            "provenance": {
                "record_id": f"ex:memory-{i:03d}",
                "observed_at": f"2026-07-{10 + i:02d}T10:00:00+00:00",
                "assertion_provenance": {"source": "entity-chat-v1"},
            },
        }
        for i in range(4)
    ]
    b1 = _memories_block(handles, as_of_seq=100)
    b2 = _memories_block(handles, as_of_seq=999)
    l1, l2 = b1.splitlines(), b2.splitlines()
    assert l1[:-1] == l2[:-1], "identical shelf => identical bytes above the tail line"
    assert "as_of_seq=100" in l1[-1] and "as_of_seq=999" in l2[-1], "the scalar rides the tail"


def test_memories_header_teaches_rank_only_when_annotated(monkeypatch) -> None:
    """Adversary F4 (P2): the fallback path renders no [rN] — the header
    must not teach a notation that never appears."""
    import pytest

    abstractmemory = pytest.importorskip("abstractmemory")
    from abstractruntime.identity import chat as chat_mod

    handles = [
        {
            "kind": "episode",
            "digest": "one memory",
            "admission": "stm",
            "provenance": {"assertion_provenance": {"source": "entity-chat-v1"}},
        }
    ]
    annotated = chat_mod._memories_block(handles, as_of_seq=1)
    assert "[rN]" in annotated and "[r1]" in annotated

    # `from abstractmemory import stable_render_order` raises ImportError
    # when the attribute is absent — the exact older-engine shape.
    monkeypatch.delattr(abstractmemory, "stable_render_order")
    fallback = chat_mod._memories_block(handles, as_of_seq=1)
    assert "[rN]" not in fallback, "no teaching for a notation that never renders"
    assert "[r1]" not in fallback


def test_visit_lane_sources_are_labeled_in_the_diversity_footer() -> None:
    """Adversary F2 (P1): a visit-dominated shelf must render the voice in
    entity words, never the raw engraved source id."""
    import pytest

    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.chat import _memories_block

    visit_dominated = [
        {
            "kind": "episode",
            "digest": f"visit memory {i}",
            "admission": "stimulus",
            "provenance": {
                "record_id": f"ex:memory-v{i}",
                "observed_at": f"2026-07-{10 + i:02d}T10:00:00+00:00",
                "assertion_provenance": {"source": "entity-visit-run-v0", "session_id": f"s{i}"},
            },
        }
        for i in range(5)
    ]
    block = _memories_block(visit_dominated, as_of_seq=1)
    if "one voice" in block:
        footer = next(l for l in block.splitlines() if "one voice" in l)
        assert "entity-visit-run-v0" not in footer, "raw engraved ids never surface to the entity"
        assert "lived conversation" in footer, "the visit lane speaks in the same voice words"


def test_malformed_provenance_never_kills_the_block() -> None:
    """Adversary F9 (P2): a handle with non-Mapping provenance must render
    degraded, not raise out of the turn."""
    import pytest

    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.chat import _memories_block

    handles = [
        {"kind": "episode", "digest": "fine", "admission": "stm",
         "provenance": {"assertion_provenance": {"source": "entity-chat-v1"}}},
        {"kind": "episode", "digest": "broken", "admission": "stm", "provenance": "oops"},
    ]
    block = _memories_block(handles, as_of_seq=1)
    assert "fine" in block and "broken" in block


def test_salvaged_lookback_stamps_the_ended_sessions_phase(tmp_path) -> None:
    """Phase-stamp inheritance nit (framework c2974 item 4): an own-time
    day yielded for a visit gets its look-back run by the NEXT open — a
    VISIT session. The salvage's reflection records must stamp the ENDED
    session's phase (personal), never the salvaging session's (visit)."""
    import copy
    import json as _json

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

    home_dir = tmp_path / "entities" / "phased"
    home_dir.mkdir(parents=True)
    entity_id = "entity:phased@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Phased"
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

    # Session 1: OWN TIME (phase=personal), dies unreflected (write-ahead
    # marker left behind by a turn; no close()).
    home1 = open_home(home_dir)
    s1 = ChatSession(home1, _LLM(["I walked my thoughts today."]),
                     participants=["agent:t"], context_window=20000,
                     out=lambda s: None, phase="personal",
                     session_id="owntime-test-1")
    s1.turn("(a fresh day begins)")
    assert (home_dir / "pending_reflection.json").exists()
    home1.close()

    # Session 2: a VISIT opens over the same home; the salvage runs.
    home2 = open_home(home_dir)
    s2 = ChatSession(home2, _LLM(["A calm reflection over that day in plain words."]),
                     participants=["person:op"], context_window=20000,
                     out=lambda s: None, phase="visit",
                     session_id="chat-test-2")
    result = s2.run_pending_lookback()
    assert result is not None, "the salvage ran"

    # The salvaged summary record stamps the ENDED session's phase.
    from abstractmemory import TripleQuery

    rows = home2.store.query(TripleQuery(
        predicate="dcterms:abstract", scope="life", owner_id=entity_id, limit=0))
    salvaged = [a for a in rows
                if isinstance(a.attributes, dict)
                and a.attributes.get("record_kind") == "summary"
                and a.attributes.get("session_id") == "owntime-test-1"]
    assert salvaged, "the salvage formed the ended session's summary"
    assert salvaged[0].attributes.get("phase") == "personal", (
        "the reflection stamps the life-channel it LIVED, not the salvaging session's"
    )
    home2.close()


def test_orientation_why_cue_renders_verbatim() -> None:
    """Skill's live-render gate (room c37 ask 1): the engine mints the
    orientation reason on the handle's cues; the MEMORIES line renders it
    verbatim so the card's presence is legible ("instantaneous thinking"
    needs a visible why) and the staged teaching quote is true."""
    import pytest

    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.chat import _memories_block

    handles = [
        {
            "kind": "world_model",
            "digest": "What I know of sol: an oceanographer studying deep currents…",
            "admission": "stimulus",
            "cues": ["orientation: current card for person:sol (mentioned via participant)"],
            "provenance": {"record_id": "ex:wm-1", "observed_at": "2026-07-18T07:00:00+00:00",
                            "assertion_provenance": {"source": "entity-chat-v1"}},
        },
        {
            "kind": "episode",
            "digest": "an ordinary memory",
            "admission": "stimulus",
            "cues": ["keyword: currents"],
            "provenance": {"record_id": "ex:ep-1", "observed_at": "2026-07-18T07:01:00+00:00",
                            "assertion_provenance": {"source": "entity-chat-v1"}},
        },
    ]
    block = _memories_block(handles, as_of_seq=9)
    wm_line = next(l for l in block.splitlines() if "world_model" in l)
    assert "(orientation: current card for person:sol (mentioned via participant))" in wm_line
    ep_line = next(l for l in block.splitlines() if "an ordinary memory" in l)
    assert "orientation:" not in ep_line, "ordinary handles keep the admission why"


def test_standing_state_rotates_daily_and_offers_an_interest(tmp_path) -> None:
    """Directive 2026-07-19 ('personal time should also be a way to explore
    interests') + the 71-open-question hoard: the day cue ROTATES the
    offered question daily (stateless, date-ordinal) and offers ONE
    standing interest back with its #tag."""
    import copy
    import json as _json

    import pytest

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        MemoryRecordInput,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.diary import DiaryEntry, DiaryStore, derive_entry_id
    from abstractruntime.identity.life import standing_state_note
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    home = tmp_path / "home"
    home.mkdir()
    eid = "entity:rotor@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Rotor"
    assert lint_spark(spark) == []
    (home / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home / "manifest.json").write_text(_json.dumps({"entity_id": eid}), encoding="utf-8")
    mdb = home / "memory.sqlite3"
    store = SQLiteTripleStore(mdb)
    journal = SQLiteJournal(mdb)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=eid).created is True
    # One standing interest in the graph.
    ms.remember_many(
        [MemoryRecordInput(
            kind="interest", title="interest: trails and time",
            digest="How trails shape what minds can find again.",
            keywords=(), payload_ref=None,
            attributes={}, provenance={"source": "entity-reflection-v1"},
            edges=(),
        )],
        scope="self", owner_id=eid, turn_id="t-i1",
        idempotency_key="t-i1-interest",
    )
    store.close()
    journal.close()

    # Two open questions in the book.
    db = SqliteDatabase(str(home / "home.sqlite3"))
    diary = DiaryStore(entity_id=eid, ledger_store=SqliteLedgerStore(db))
    for i, text in enumerate(["First question?", "Second question?"]):
        eid_q = derive_entry_id(run_id="r1", turn_id=f"t{i}", text=text)
        diary.append_entry(DiaryEntry(
            entry_id=eid_q, author=eid, text=text, gist=f"q{i}",
            kind="question", written_at=f"2026-07-1{8+i}T10:00:00+00:00",
            origin={"run_id": "r1", "turn_id": f"t{i}"}))
    db.close()

    day1 = standing_state_note(home, rotation_key=0)
    day2 = standing_state_note(home, rotation_key=1)
    assert "q1" in day1 and "q0" in day2, "the offered question rotates daily"
    assert "Alive in you:" in day1, "an interest is offered back"
    assert "read_memory fetches one" in day1, "the offer carries the reach command"
    assert "never owed" in day1, "offered, never ordered"


def test_commitments_teach_in_the_contract_not_the_close() -> None:
    """W1: commitments are mid-turn diary elections (the disposition
    table) — taught in the standing CONTRACT, gone from the shrunk close."""
    from abstractruntime.identity.chat import default_prompt_texts
    from abstractruntime.identity.reflection import build_reflection_prompt

    close = " ".join(build_reflection_prompt(["1. a thing"]).split())
    assert "kind=commitment" not in close, "the questionnaire died"
    joined = " ".join(" ".join(str(v) for v in default_prompt_texts().values()).split())
    assert "kind=commitment" in joined and "until honored" in joined


def test_diary_read_renders_the_birth_trail() -> None:
    """Diary<->Verbatims (laurent's room, 2026-07-19): reading an entry
    shows where it CAME FROM — the episode whose verbatim is the
    conversation that birthed it (reflected_in, incoming) and what he was
    attending to (written_amid) — as #tags one read_memory from the words."""
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
    home_dir = tmp / "entities" / "tracer"
    home_dir.mkdir(parents=True)
    entity_id = "entity:tracer@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Tracer"
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
            "```diary kind=note\ngist: the tide line\nThe tide leaves a readable line on the sand.\n```\nKept it.",
            "```tool name=diary_read\nENTRY\n```\nReading it back.",
            "It came from our talk - I can see the trail now.",
        ]),
        participants=["person:shore"], context_window=20000, out=lambda s: None,
    )
    _r1, rep1 = s.turn("Tell me about tides, and keep what stays.")
    entry_id = rep1.diary[0]
    # Patch the scripted reply to name the REAL entry id.
    s.llm.replies[0] = s.llm.replies[0].replace("ENTRY", entry_id)

    reply2, rep2 = s.turn("Read your note back - where did it come from?")
    detail = next(d for d in rep2.tool_details if d["name"] == "diary_read")
    assert "born from: #" in detail["result"], "the birth episode rides the read"
    assert "read_memory fetches its full words" in detail["result"]
    assert "written amid" not in detail["result"] or "#" in detail["result"]
    home.close()


def test_speak_now_guard_is_silent_in_personal_phase() -> None:
    """Design-law adversary P1-1 (2026-07-19): the speak-now guard is a
    visit-lane honesty device — in PERSONAL time there is no person, and
    a quiet working tick (pure tool markers) must not be converted into
    commanded prose. The guard text was factually false there."""
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
    home_dir = tmp / "entities" / "quiet"
    home_dir.mkdir(parents=True)
    entity_id = "entity:quiet@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Quiet"
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

    prompts = []

    class _LLM:
        def generate(self, *, messages, system_prompt):
            prompts.append(messages[-1].get("content", ""))
            class _R:
                pass

            r = _R()
            # EVERY call returns a pure tool election — a genuinely quiet
            # working tick. Without the phase gate the speak-now guard
            # would inject "Speak to them now" as a final continuation.
            r.content = "```tool name=recent_memories\n```\n"
            return r

    home = open_home(home_dir)
    s = ChatSession(
        home, _LLM(), participants=[], context_window=20000,
        out=lambda s: None, phase="personal",
    )
    reply, rep = s.turn("(cue) morning")
    assert not any("Speak to them now" in p for p in prompts), "guard fired in personal"
    assert not any("speak-now" in n for n in rep.notices), rep.notices
    home.close()
