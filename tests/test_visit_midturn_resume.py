"""Durable-visit resume after a mid-turn kill (was: criterion 7).

REWRITTEN under laurent's A ruling (2026-07-20): the act-only ref layer is
DELETED — diary tool results rest AS SERVED in the durable transcript (the
home is the privacy boundary). What this fixture still pins, because it
never depended on refs:

1. WRITE-BOUNDARY CAPTURE: a ```diary election's words fly to the book at
   the result boundary and never rest in run vars from the ELECTION turn
   (the marked reply rests; sole-authorship mechanics survive the ruling).
2. MID-TURN KILL + RESUME: the process dies between the tool round and the
   reply; a fresh process over the same home completes the turn.
3. NO RE-EXECUTION: the resumed cycle reuses the completed acts — exactly
   one tool round in the durable transcript, one book entry.
"""

from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.entity_runtime import (  # noqa: E402
    entity_run_store_path,
    open_entity_runtime,
)
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    HARVEST_NODE,
    VISITOR_WAIT_KEY,
    ReactMiddle,
    build_visit_workflow,
)

PRIVATE_TOKEN = "AURORA-criterion7-9f3a"


def _make_home(tmp_path: Path, slug: str = "resumeling") -> Path:
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
    store, journal = SQLiteTripleStore(db), SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


class _WireCapturingLLM:
    """Scripted LLM_CALL handler that records the WIRE payload it receives.

    `open_entity_runtime` composes the G1 act-only dereference AROUND the
    LLM handler, so this stub sees the RESOLVED (wire) copy while the run
    store/ledger keep the ref — capturing both sides of the boundary."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.wire_payloads: List[Dict[str, Any]] = []

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        self.wire_payloads.append(dict(effect.payload or {}))
        content = self.replies.pop(0) if self.replies else "…"
        return EffectOutcome.completed({"content": content})


def _act_only_middle(tool_ref_cell: Dict[str, Any]) -> ReactMiddle:
    """Contract-faithful stand-in for the adapter cycle (no abstractagent
    import — the dependency points the other way; agent's real cycle pins
    the same seams from their side). Turn shape: when `tool_ref_cell`
    carries an entry_id, iteration 1 runs an act-only diary_read round —
    the durable transcript gets the REF (the adapter's observe shape:
    words never rest) — and iteration 2 answers. Without an entry_id the
    first reply is the answer (turn 1 plants the entry via election)."""

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
        msgs = run.vars.setdefault("context", {}).setdefault("messages", [])
        msgs.append({"role": "assistant", "content": content})
        i = int(temp.get("i") or 0) + 1
        temp["i"] = i
        entry_id = str(tool_ref_cell.get("entry_id") or "")
        if entry_id and i == 1:
            # The act step ran diary_read: under the A ruling the durable
            # transcript carries the SERVED words (the home is the privacy
            # boundary; no refs, no send-time dereference).
            msgs.append({
                "role": "tool",
                "tool_call_id": "call-1",
                "content": f"[diary_read {entry_id}]\n" + str(tool_ref_cell.get("entry_text") or ""),
            })
            return StepPlan(node_id="observe", next_node="reason")
        temp["final_answer"] = content
        return StepPlan(node_id="observe", next_node=HARVEST_NODE)

    def reset(vars: Dict[str, Any]) -> None:
        temp = vars.setdefault("_temp", {})
        temp["i"] = 0
        temp.pop("final_answer", None)
        temp["turn_captures"] = {"diary_entries": [], "act_only_warnings": []}

    return ReactMiddle(nodes={"reason": reason, "observe": observe}, entry="reason", reset_turn=reset)


def _load_run_vars(home_dir: Path, run_id: str) -> Dict[str, Any]:
    """Read the run's DURABLE vars from the per-entity run store inside the
    home (`runtime_<slug>.sqlite3`) via a fresh connection — what a fresh
    process would see, never what this process holds in memory."""
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

    db = SqliteDatabase(entity_run_store_path(home_dir))
    try:
        run = SqliteRunStore(db).load(run_id)
        return dict(run.vars) if run is not None else {}
    finally:
        db.close()


def _book_entry_text(home: Any, token: str) -> str:
    """The planted private entry's verbatim words through the home's book
    surface (operator-view read; test-side)."""
    for e in home.diary.list_entries():
        text = str(e.get("text") or "")
        if token in text:
            return text
    raise AssertionError("the planted private entry is not in the book")


def _mid_wf(ert: Any, cell: Dict[str, Any]):
    return build_visit_workflow(
        ert.home,
        participants=["person:albou"],
        idle_seconds=3600,
        model_info={"provider": "test", "model": "scripted"},
        visit_id="visit-c7",
        react_middle=_act_only_middle(cell),
    )


def test_visit_resumes_after_midturn_kill_without_reexecution(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    cell: Dict[str, Any] = {}

    # ---- Session A: turn 1 plants one PRIVATE entry (election at the
    # result boundary: words fly to the book, the marked reply rests).
    llm_a = _WireCapturingLLM([
        "Kept.\n```diary kind=note visibility=private\ngist: a private beginning\n"
        f"The secret word is {PRIVATE_TOKEN}.\n```",
        "(thinking) I should re-read my note first.",  # turn 2, iteration 1
        # The reply that would follow the tool round never happens: killed.
    ])
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm_a})
    wf = _mid_wf(ert, cell)
    run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-c7")
    state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
    assert state.status == RunStatus.WAITING

    state = ert.runtime.resume(  # turn 1: plant the private entry
        workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
        payload={"text": "Please keep a private note with your secret word.",
                 "speaker": "person:albou"},
        max_steps=100,
    )
    assert state.status == RunStatus.WAITING
    # The words are in the book; the ELECTION turn's durable state carries
    # only the marked reply (write-boundary capture — sole authorship).
    entry_text = _book_entry_text(ert.home, PRIVATE_TOKEN)
    assert PRIVATE_TOKEN not in json.dumps(_load_run_vars(home_dir, run_id), default=str)
    entries = ert.home.diary.list_entries()
    cell["entry_id"] = next(
        str(e.get("entry_id") or "") for e in entries if PRIVATE_TOKEN in str(e.get("text") or "")
    )
    cell["entry_text"] = entry_text
    assert cell["entry_id"]

    # ---- Turn 2: drive by SINGLE STEPS until the tool round's SERVED
    # content first RESTS in the durable transcript, then "SIGKILL" (close
    # mid-turn). Deterministic against node-graph changes: the kill lands
    # at the exact step the tool message landed, before the reply.
    state = ert.runtime.resume(
        workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
        payload={"text": "What did you write? Read it back to yourself first.",
                 "speaker": "person:albou"},
        max_steps=1,
    )
    for _ in range(30):
        raw = json.dumps(_load_run_vars(home_dir, run_id), default=str)
        if "[diary_read " in raw:
            break
        assert state.status == RunStatus.RUNNING, "turn ended before the tool round"
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=1)
    else:
        raise AssertionError("the tool round never rested in durable vars")
    assert state.status == RunStatus.RUNNING  # mid-turn, not parked
    ert.close()  # the crash: process state gone, stores closed

    # (1) AT REST MID-TURN (A ruling): the SERVED words rest in the home's
    # own run store - the home is the privacy boundary.
    raw = json.dumps(_load_run_vars(home_dir, run_id), default=str)
    assert "[diary_read " in raw, "the served tool round is missing from the transcript"

    # ---- Session B: fresh process over the same home; drive to park.
    llm_b = _WireCapturingLLM([
        "You asked about my note - I re-read it privately; it stays mine.",
        "Looking back: continuity held.\nGoodbye.",  # reflection at close
    ])
    ert2 = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm_b})
    wf2 = _mid_wf(ert2, cell)
    try:
        state = ert2.runtime.tick(workflow=wf2, run_id=run_id, max_steps=100)
        assert state.status == RunStatus.WAITING, "turn did not complete after resume"

        # (2) The resumed WIRE payload carried the served words (they are
        # part of the transcript now - no dereference machinery).
        assert len(llm_b.wire_payloads) == 1, "expected exactly the resumed reply call"
        wire = json.dumps(llm_b.wire_payloads, default=str)
        assert PRIVATE_TOKEN in wire, "the transcript's tool round did not reach the wire"

        # (3) NO RE-EXECUTION: exactly one tool round in the durable
        # transcript — the resumed cycle reused the completed act, never
        # ran a second one.
        vars_after = _load_run_vars(home_dir, run_id)
        transcript = (vars_after.get("context") or {}).get("messages") or []
        tool_msgs = [m for m in transcript if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        # (4) ONE BOOK ENTRY: the kill+resume never double-wrote the book.
        matching = [e for e in ert2.home.diary.list_entries()
                    if PRIVATE_TOKEN in str(e.get("text") or "")]
        assert len(matching) == 1

        # Close completes cleanly.
        state = ert2.runtime.resume(
            workflow=wf2, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=300,
        )
        assert state.status == RunStatus.COMPLETED
    finally:
        ert2.close()
