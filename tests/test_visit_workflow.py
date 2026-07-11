"""The visit as one durable run (plan items 7-10; frozen seam spec §A).

Offline pins for the centerpiece: a full visit lifecycle through
Runtime.tick/resume over a per-entity runtime (recall → LLM → elections →
same-trace commit → form → answer → park), RESTART-MID-VISIT resume (the
phase's payoff: the process dies between turns and the visit continues in
a fresh process over the same home), idle-timeout close via the D3
deadline, reflection with feelings/interests through the staged APPLY
loop, refused-prelude honesty, and D2 (identity access counts stay 0
through a full workflow visit).
"""

from __future__ import annotations

import copy as _copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    VISITOR_WAIT_KEY,
    build_visit_workflow,
)


def _make_home(tmp_path: Path, slug: str = "visitling") -> Path:
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}@home-test"
    spark = _copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = slug.capitalize()
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


class _ScriptedLLMHandler:
    """LLM_CALL handler returning queued replies (the provider stub)."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        self.calls.append(dict(effect.payload or {}))
        content = self.replies.pop(0) if self.replies else "…"
        return EffectOutcome.completed({"content": content})


def _open(home_dir: Path, llm: _ScriptedLLMHandler):
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    wf = build_visit_workflow(
        ert.home,
        participants=["person:albou"],
        idle_seconds=3600,
        model_info={"provider": "test", "model": "scripted"},
        visit_id="visit-abc123",  # the door-stamped item-14 correlation key
    )
    return ert, wf


def test_full_visit_lifecycle_turns_elections_reflection(tmp_path: Path) -> None:
    from abstractmemory import TripleQuery

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLMHandler([
        "Hello Laurent - I remember beginnings.",
        "Noted.\n```diary kind=note\ngist: first visit note\nThe first durable visit happened.\n```",
        # Reflection: a feeling on sheet item 1, one interest, a goodbye.
        "Looking back, that mattered.\n```feel\ntarget=1 feeling=+2 reason=\"the first durable turn\"\n```\n"
        "```interest\nwhat survives a restart - durable conversation itself\n```\nGoodbye.",
    ])
    ert, wf = _open(home_dir, llm)
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        assert state.status == RunStatus.WAITING
        assert state.waiting.wait_key == VISITOR_WAIT_KEY
        assert (state.waiting.details or {}).get("kind") == "visitor_message"

        # Turn 1.
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Hello - do you remember beginnings?", "speaker": "person:albou"},
            max_steps=100,
        )
        assert state.status == RunStatus.WAITING  # parked again after answering
        # Turn 2 (with a diary election in the reply).
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Keep a note about this moment.", "speaker": "person:albou"},
            max_steps=100,
        )
        assert state.status == RunStatus.WAITING
        # Close -> reflection -> done.
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=200,
        )
        assert state.status == RunStatus.COMPLETED
        assert state.output["turns"] == 2
        assert state.output["close_reason"] == "closed"

        # The home's graph holds 2 episodes (participants stamped, both
        # sides), a summary, and an interest formed by the reflection.
        rows = ert.home.ms.query(TripleQuery(scope="life", owner_id=ert.entity_id, limit=0))
        kinds = [
            (a.attributes or {}).get("record_kind") for a in rows if isinstance(a.attributes, dict)
        ]
        assert kinds.count("episode") == 2
        assert kinds.count("summary") == 1
        episode_attrs = [
            a.attributes for a in rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "episode"
        ]
        assert all(
            a.get("participants") == ["person:albou", ert.entity_id] for a in episode_attrs
        )
        # Item-14 correlation key: every episode carries the door-stamped
        # visit_id as DATA (both legs of a cross-runtime visit pin one string).
        assert all(a.get("visit_id") == "visit-abc123" for a in episode_attrs)
        self_rows = ert.home.ms.query(TripleQuery(scope="self", owner_id=ert.entity_id, limit=0))
        self_kinds = [
            (a.attributes or {}).get("record_kind") for a in self_rows if isinstance(a.attributes, dict)
        ]
        assert "interest" in self_kinds

        # The book holds the elected entry (the words, in the diary plane).
        entries = ert.home.diary.list_entries()
        assert any("first durable visit happened" in (e.get("text") or "") for e in entries)

        # The visitor-facing replies rode ANSWER_USER records in the ledger.
        ledger = ert.runtime.get_ledger(run_id)
        answers = [
            ((r.get("effect") or {}).get("payload") or {}).get("message")
            for r in ledger
            if isinstance(r, dict) and ((r.get("effect") or {}).get("type")) == "answer_user"
        ]
        assert any("Hello Laurent" in (m or "") for m in answers)

        # D2 through the workflow: identity records untouched by presence.
        value_ids = [
            str(a.subject) for a in self_rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "value"
        ]
        assert value_ids
        counts = ert.home.ms.access_counts(record_ids=value_ids)
        recs = counts.get("records", counts)
        assert all(int(v) == 0 for v in recs.values()), f"identity strengthened: {recs}"
    finally:
        ert.close()


def test_refused_reflection_election_skips_loudly_never_kills_the_close(tmp_path: Path) -> None:
    """Gateway c709 interim (the fdf01e0 rule class, agency's step-10 bug):
    a door REFUSAL of one reflection election (live case: interest FORM
    into self refused under the visit's workplace stamp) must land as a
    loud #FALLBACK skip in the reflection output while the remaining
    stages (diary, feelings) still apply — never a terminal-FAILED close.
    """
    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLMHandler([
        "Hello - a reply.",
        # Reflection elects an interest (will be refused), a diary entry
        # AND a feeling (both must still apply behind the refusal).
        "Looking back.\n```interest\nwhat refusal teaches about doors\n```\n"
        "```diary kind=note\ngist: survived a refusal\nThe close survived a refused election.\n```\n"
        "```feel\ntarget=1 feeling=+1 reason=\"the turn happened\"\n```\nGoodbye.",
    ])
    ert, wf = _open(home_dir, llm)
    # Simulate the door's channel refusal AT THE HANDLER: interest FORMs
    # into self refuse (the exact live collision), everything else lands.
    inner_form = ert.runtime._handlers[EffectType.MEMORY_FORM]

    def _refusing_form(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        records = (effect.payload or {}).get("records") or []
        if any((r or {}).get("kind") == "interest" for r in records):
            return EffectOutcome.failed(
                "MEMORY_FORM into the 'self' scope is not a workplace act", retryable=False
            )
        return inner_form(run, effect, dnn)

    ert.runtime._handlers[EffectType.MEMORY_FORM] = _refusing_form
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-refused")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Hello.", "speaker": "person:albou"}, max_steps=100,
        )
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=300,
        )
        # The close COMPLETED despite the refused election...
        assert state.status == RunStatus.COMPLETED, state.error
        notices = list(state.output.get("reflection_notices") or [])
        assert any("#FALLBACK" in n and "interest" in n and "refused" in n for n in notices)
        # ...the refused interest did NOT land...
        from abstractmemory import TripleQuery

        self_rows = ert.home.ms.query(TripleQuery(scope="self", owner_id=ert.entity_id, limit=0))
        self_kinds = [
            (a.attributes or {}).get("record_kind")
            for a in self_rows if isinstance(a.attributes, dict)
        ]
        assert "interest" not in self_kinds
        # ...and the stages QUEUED BEHIND the refusal still applied: the
        # diary entry is in the book, and the summary formed in life scope.
        entries = ert.home.diary.list_entries()
        assert any("survived a refused election" in (e.get("text") or "") for e in entries)
        rows = ert.home.ms.query(TripleQuery(scope="life", owner_id=ert.entity_id, limit=0))
        kinds = [
            (a.attributes or {}).get("record_kind") for a in rows if isinstance(a.attributes, dict)
        ]
        assert kinds.count("summary") == 1
        # The ledger recorded the failure honestly (loud, never silent):
        # the refused effect's StepRecord carries status=failed + the
        # door's error — absorption changes the RUN's fate, not the record.
        ledger = ert.runtime.get_ledger(run_id)
        failed_steps = [
            r for r in ledger
            if isinstance(r, dict) and str(r.get("status") or "") == "failed"
        ]
        assert failed_steps, "the refused effect must appear as a failed step record"
        assert any("workplace act" in str(r.get("error") or "") for r in failed_steps)
    finally:
        ert.close()


def test_restart_mid_visit_resumes_in_a_fresh_process_context(tmp_path: Path) -> None:
    """THE PHASE'S PAYOFF (criterion 5 unit shadow): the hosting process
    dies between turns; a fresh EntityRuntime over the same home finds the
    parked run and the visit CONTINUES — prelude, history, and episode
    chain intact from run vars."""
    home_dir = _make_home(tmp_path, slug="reviveling")
    llm1 = _ScriptedLLMHandler(["First reply before the crash."])
    ert1, wf1 = _open(home_dir, llm1)
    run_id = ert1.runtime.start(workflow=wf1, vars={}, session_id="visit-1")
    ert1.runtime.tick(workflow=wf1, run_id=run_id, max_steps=50)
    state = ert1.runtime.resume(
        workflow=wf1, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
        payload={"text": "Hello before the crash.", "speaker": "person:albou"}, max_steps=100,
    )
    assert state.status == RunStatus.WAITING
    ert1.close()  # the "gateway restart": everything in-process is gone

    llm2 = _ScriptedLLMHandler([
        "Second reply after the restart.",
        "Reflection after restart.",
    ])
    ert2, wf2 = _open(home_dir, llm2)  # fresh home handles, fresh workflow closures
    try:
        parked = ert2.runtime.get_state(run_id)
        assert parked.status == RunStatus.WAITING
        assert parked.waiting.wait_key == VISITOR_WAIT_KEY
        state = ert2.runtime.resume(
            workflow=wf2, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Still there after the restart?", "speaker": "person:albou"},
            max_steps=100,
        )
        assert state.status == RunStatus.WAITING
        # The second turn's LLM call saw the FIRST turn's history (stable
        # head + transcript rebuilt from run vars, not from process memory).
        sent = llm2.calls[0]
        assert any("Hello before the crash." in m.get("content", "") for m in sent["messages"])
        assert "identity" in sent["system_prompt"].lower() or len(sent["system_prompt"]) > 200
        state = ert2.runtime.resume(
            workflow=wf2, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=200,
        )
        assert state.status == RunStatus.COMPLETED
        assert state.output["turns"] == 2  # one before the crash, one after
    finally:
        ert2.close()


def test_idle_deadline_closes_the_visit_with_reflection(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path, slug="idleling")
    llm = _ScriptedLLMHandler(["A reply.", "A quiet goodbye."])
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    wf = build_visit_workflow(ert.home, participants=["person:albou"], idle_seconds=0.01)
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "One turn, then silence.", "speaker": "person:albou"}, max_steps=100,
        )
        time.sleep(0.05)  # the visitor never comes back; the deadline passes
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=200)
        assert state.status == RunStatus.COMPLETED
        assert state.output["close_reason"] == "idle_timeout"
        assert state.output["turns"] == 1
    finally:
        ert.close()


def test_pause_close_skips_reflection_hard_freeze(tmp_path: Path) -> None:
    """closed_by=pause carries skip_reflection (gateway 0014/094354Z): a
    hard freeze completes the run WITHOUT the reflection LLM call — the
    look-back debt rides the door's pending-look-back at the next open."""
    home_dir = _make_home(tmp_path, slug="frozling")
    llm = _ScriptedLLMHandler(["One reply."])  # NO reflection reply queued
    ert, wf = _open(home_dir, llm)
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "One turn.", "speaker": "person:albou"}, max_steps=100,
        )
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close", "closed_by": "pause",
                     "reason": "operator pause", "skip_reflection": True},
            max_steps=200,
        )
        assert state.status == RunStatus.COMPLETED
        assert state.output["closed_by"] == "pause"
        assert llm.replies == []  # exactly one LLM call happened: the turn
        # The look-back DEBT is explicit: the door's next open consumes it.
        assert state.output["reflection_pending"] is True
        assert state.output["sheet"]  # word-free session sheet rides along
    finally:
        ert.close()


def _stub_middle():
    """A contract-faithful stand-in for the adapter cycle (two iterations:
    elect+tool-shaped reply, then the answer) — NO abstractagent import
    (the dependency points the other way; agent's real cycle is pinned in
    their tests/test_react_visit_merge.py against these same seams)."""
    from abstractruntime.identity.visit_workflow import HARVEST_NODE, ReactMiddle

    def reason(run: Any, ctx: Any) -> StepPlan:
        rn = run.vars.setdefault("_runtime", {})
        msgs = (run.vars.get("context") or {}).get("messages") or []
        extras = rn.get("llm_payload_extras") or {}
        return StepPlan(
            node_id="reason",
            effect=Effect(type=EffectType.LLM_CALL, payload={
                "messages": list(msgs),
                "system_prompt": rn.get("system_prompt"),
                "turn_id": rn.get("turn_id"),
                **extras,
            }, result_key="_temp.llm_response"),
            next_node="observe",
        )

    def observe(run: Any, ctx: Any) -> StepPlan:
        temp = run.vars.setdefault("_temp", {})
        resp = temp.get("llm_response") or {}
        caps = temp.setdefault("turn_captures", {"diary_entries": [], "act_only_warnings": []})
        caps["diary_entries"] = list(caps.get("diary_entries") or []) + list(resp.get("diary_entries") or [])
        caps["act_only_warnings"] = list(caps.get("act_only_warnings") or []) + list(resp.get("act_only_warnings") or [])
        content = str(resp.get("content") or "")
        run.vars.setdefault("context", {}).setdefault("messages", []).append(
            {"role": "assistant", "content": content}
        )
        i = int(temp.get("i") or 0) + 1
        temp["i"] = i
        if i >= 2:
            temp["final_answer"] = content
            return StepPlan(node_id="observe", next_node=HARVEST_NODE)
        return StepPlan(node_id="observe", next_node="reason")

    def reset(vars: Dict[str, Any]) -> None:
        temp = vars.setdefault("_temp", {})
        temp["i"] = 0
        temp.pop("final_answer", None)
        temp["turn_captures"] = {"diary_entries": [], "act_only_warnings": []}

    return ReactMiddle(nodes={"reason": reason, "observe": observe}, entry="reason", reset_turn=reset)


def test_react_middle_replaces_reason_and_downstream_runs_unchanged(tmp_path: Path) -> None:
    """The merge-ownership shape (a2a 0014 ask 1, ruled: ONE owner = this
    package; the middle arrives as DATA): a multi-iteration middle runs
    where v0's REASON was; a mid-loop diary election is captured at the
    result boundary INSIDE the middle; ELECT→FORM fold the harvested
    outcome byte-unchanged (words only in the book; episode stamped)."""
    from abstractmemory import TripleQuery
    from abstractruntime.identity.visit_workflow import build_visit_workflow

    private = "Mid-loop merged note - words for the book only."
    home_dir = _make_home(tmp_path, slug="middleling")
    llm = _ScriptedLLMHandler([
        f"Holding this.\n```diary kind=note\ngist: merged note\n{private}\n```",  # iteration 1: elects
        "It is good to be one loop now.",                                          # iteration 2: answers
        "Reflection: the merged turn held.",
    ])
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    wf = build_visit_workflow(
        ert.home, participants=["person:albou"], idle_seconds=3600,
        visit_id="visit-mid01", react_middle=_stub_middle(),
    )
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
        ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Keep a note mid-loop.", "speaker": "person:albou"}, max_steps=200,
        )
        assert state.status == RunStatus.WAITING  # parked after the 2-iteration turn
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=300,
        )
        assert state.status == RunStatus.COMPLETED
        assert state.output["turns"] == 1

        # The mid-loop election's words rest ONLY in the book.
        entries = ert.home.diary.list_entries()
        assert any(private in (e.get("text") or "") for e in entries)
        assert private not in json.dumps(ert.runtime.get_state(run_id).vars)
        assert private not in json.dumps(ert.runtime.get_ledger(run_id))

        # Downstream unchanged: episode formed with the stamp + the answer served.
        rows = ert.home.ms.query(TripleQuery(scope="life", owner_id=ert.entity_id, limit=0))
        ep = [a.attributes for a in rows
              if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "episode"]
        assert len(ep) == 1 and ep[0].get("visit_id") == "visit-mid01"
        answers = [
            ((r.get("effect") or {}).get("payload") or {}).get("message")
            for r in ert.runtime.get_ledger(run_id)
            if isinstance(r, dict) and ((r.get("effect") or {}).get("type")) == "answer_user"
        ]
        assert any("one loop now" in (m or "") for m in answers)
    finally:
        ert.close()


def test_react_middle_collision_refuses_loudly(tmp_path: Path) -> None:
    from abstractruntime.identity.visit_workflow import ReactMiddle, build_visit_workflow

    home_dir = _make_home(tmp_path, slug="collideling")
    ert = open_entity_runtime(home_dir)
    try:
        with pytest.raises(ValueError, match="collide"):
            build_visit_workflow(
                ert.home,
                react_middle=ReactMiddle(nodes={"REASON": lambda r, c: None}, entry="REASON"),
            )
        with pytest.raises(ValueError, match="not in its own node map"):
            build_visit_workflow(
                ert.home,
                react_middle=ReactMiddle(nodes={"reason": lambda r, c: None}, entry="ghost"),
            )
    finally:
        ert.close()


def test_refused_prelude_completes_with_reasons_never_truncates(tmp_path: Path) -> None:
    """A virgin home (no engram) refuses the summon on the durable record."""
    from abstractruntime.identity.chat import open_home

    home_dir = tmp_path / "entities" / "ghost"
    home_dir.mkdir(parents=True)
    spark = _copy.deepcopy(dict(__import__("abstractmemory").DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Ghost"
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark), encoding="utf-8")
    (home_dir / "manifest.json").write_text(
        json.dumps({"entity_id": "entity:ghost@home-test"}), encoding="utf-8"
    )
    llm = _ScriptedLLMHandler([])
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    wf = build_visit_workflow(ert.home, participants=["person:albou"])
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-1")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status == RunStatus.COMPLETED
        assert state.output["refused"] is True
        assert state.output["reasons"]  # the why, on the record
        assert llm.calls == []  # no model call for a refused summon
    finally:
        ert.close()
