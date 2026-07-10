"""EVENT wait with a DEADLINE (frozen seam spec D3, a2a thread 0013).

A visit parks on the visitor's next message; if nobody speaks before the
deadline, the wait resolves as a LABELED timeout into its resume path —
retiring the in-process idle-reaper class. Pins: the combined wait carries
wait_key AND until (UTC-normalized at the single write boundary); the event
resume wins before the deadline; past it, tick resumes with
{"timed_out": true} in result_key; deadline-carrying EVENT waits surface in
every store's due-scan so the scheduler wakes parked runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan, WaitReason
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _iso_in(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _park_workflow(until: str) -> WorkflowSpec:
    def park(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="PARK",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={
                    "wait_key": "visitor_input",
                    "until": until,
                    "details": {"kind": "visitor_message"},
                },
                result_key="_temp.resume",
            ),
            next_node="ROUTE",
        )

    def route(run: Any, ctx: Any) -> StepPlan:
        resume = (run.vars.get("_temp") or {}).get("resume") or {}
        return StepPlan(
            node_id="ROUTE",
            complete_output={
                "timed_out": bool(resume.get("timed_out")),
                "text": resume.get("text"),
            },
        )

    return WorkflowSpec(workflow_id="wf_park", entry_node="PARK", nodes={"PARK": park, "ROUTE": route})


def _runtime() -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())


def test_park_carries_key_and_deadline_normalized_to_utc() -> None:
    rt = _runtime()
    # Offset-form deadline (the WAIT_UNTIL invariant: normalize at the write
    # boundary — lexicographic string due-ness would silently mis-order).
    wf = _park_workflow("2999-01-01T12:00:00+02:00")
    run_id = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    assert state.status == RunStatus.WAITING
    assert state.waiting.reason == WaitReason.EVENT
    assert state.waiting.wait_key == "visitor_input"
    assert state.waiting.until == "2999-01-01T10:00:00+00:00"
    # The ledger's wait record carries BOTH (flow's render contract:
    # until beside wait_key + details.kind, zero new transport).
    last = rt.get_ledger(run_id)[-1]
    wait = (last.get("result") or {}).get("wait")
    assert wait["wait_key"] == "visitor_input"
    assert wait["until"] == "2999-01-01T10:00:00+00:00"
    assert (wait.get("details") or {}).get("kind") == "visitor_message"


def test_event_resume_wins_before_the_deadline() -> None:
    rt = _runtime()
    wf = _park_workflow(_iso_in(3600))
    run_id = rt.start(workflow=wf, vars={})
    rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    out = rt.resume(workflow=wf, run_id=run_id, wait_key="visitor_input",
                    payload={"text": "hello"}, max_steps=5)
    assert out.status == RunStatus.COMPLETED
    assert out.output["timed_out"] is False
    assert out.output["text"] == "hello"


def test_passed_deadline_resumes_as_labeled_timeout() -> None:
    rt = _runtime()
    wf = _park_workflow(_iso_in(3600))
    run_id = rt.start(workflow=wf, vars={})
    rt.tick(workflow=wf, run_id=run_id, max_steps=5)

    # The deadline passes while the run is parked (simulated by rewriting
    # the persisted wait — the run store is the durable truth tick reads).
    parked = rt.get_state(run_id)
    parked.waiting.until = _iso_in(-1)
    rt.run_store.save(parked)

    woke = rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    assert woke.status == RunStatus.COMPLETED
    assert woke.output["timed_out"] is True
    assert woke.output["text"] is None


def test_future_deadline_does_not_wake_the_park() -> None:
    rt = _runtime()
    wf = _park_workflow(_iso_in(3600))
    run_id = rt.start(workflow=wf, vars={})
    rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    still = rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    assert still.status == RunStatus.WAITING  # no premature wake


def test_invalid_deadline_fails_loudly() -> None:
    rt = _runtime()
    wf = _park_workflow("next tuesday")
    run_id = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    assert state.status == RunStatus.FAILED
    assert "not a valid ISO timestamp" in (state.error or "")


@pytest.mark.parametrize("backend", ["memory", "json", "sqlite"])
def test_due_scan_surfaces_deadline_event_waits(tmp_path, backend: str) -> None:
    """The scheduler's due-scan must wake parked visits: deadline-carrying
    EVENT waits join UNTIL waits in list_due_wait_until on every backend."""
    if backend == "memory":
        store = InMemoryRunStore()
    elif backend == "json":
        from abstractruntime.storage.json_files import JsonFileRunStore

        store = JsonFileRunStore(tmp_path / "runs")
    else:
        from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

        store = SqliteRunStore(SqliteDatabase(tmp_path / "rt.sqlite3"))

    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore())
    wf = _park_workflow(_iso_in(3600))
    run_id = rt.start(workflow=wf, vars={})
    rt.tick(workflow=wf, run_id=run_id, max_steps=5)

    now = datetime.now(timezone.utc).isoformat()
    assert store.list_due_wait_until(now_iso=now) == []  # future deadline: not due

    parked = rt.get_state(run_id)
    parked.waiting.until = _iso_in(-1)
    store.save(parked)
    due = store.list_due_wait_until(now_iso=datetime.now(timezone.utc).isoformat())
    assert [r.run_id for r in due] == [run_id]
    # And ticking the due run resolves it as the labeled timeout.
    woke = rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    assert woke.status == RunStatus.COMPLETED
    assert woke.output["timed_out"] is True
