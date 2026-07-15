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
