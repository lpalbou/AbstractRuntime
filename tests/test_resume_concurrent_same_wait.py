"""A wait is resumed at most once, even when two callers race.

Incident 2026-09-26 (hermetic gateway, scheduled basic-agent flow): the
gateway resumes a parent's `subworkflow:<child>` wait from two places when a
child finishes — the tick thread (`_resume_subworkflow_parents`) and the loop
thread's repair pass (`_repair_terminal_subworkflow_waits`). Both read the
parent as WAITING before either saved, both passed `resume()`'s check, the
ledger got two `resume` records for the same wait key, and the second resume
re-pointed the parent at its Agent node after the first tick had consumed the
result — a second full agent loop (2x time and tokens).

`resume()` checked "is this run waiting on this key" and committed the resume
with no atomicity between the two. The loser of the race must be refused as
stale ("Run is not waiting"), exactly as a sequential second resume is.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from abstractruntime import Effect, EffectType, Runtime, StaleResumeError, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore
from abstractruntime.storage.json_files import JsonFileRunStore


class _RacingLedger(InMemoryLedgerStore):
    """Runs a second resume() while the first is between check and commit."""

    def __init__(self) -> None:
        super().__init__()
        self.on_first_resume_record = None
        self._fired = False

    def append(self, record: Any) -> None:
        eff = getattr(record, "effect", None)
        if isinstance(eff, dict) and eff.get("type") == "resume" and not self._fired and self.on_first_resume_record:
            self._fired = True
            self.on_first_resume_record()
        super().append(record)


def test_concurrent_resume_of_the_same_wait_runs_once(tmp_path) -> None:
    run_store = JsonFileRunStore(tmp_path)
    ledger = _RacingLedger()
    runtime = Runtime(run_store=run_store, ledger_store=ledger)

    executed: list[Any] = []

    def ask(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="ask",
            effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "go?"}, result_key="_temp.answer"),
            next_node="act",
        )

    def act(run, ctx) -> StepPlan:
        executed.append(run.vars.get("_temp", {}).get("answer"))
        return StepPlan(node_id="act", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="wf_resume_race", entry_node="ask", nodes={"ask": ask, "act": act})
    run_id = runtime.start(workflow=wf, vars={"_temp": {}, "_limits": {}, "_runtime": {}})
    st = runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert st.status == RunStatus.WAITING and st.waiting is not None
    wait_key = st.waiting.wait_key

    second: dict[str, Any] = {}

    def _second_resume() -> None:
        try:
            second["state"] = runtime.resume(
                workflow=wf, run_id=run_id, wait_key=wait_key, payload={"response": "second"}, max_steps=0
            )
        except Exception as e:  # noqa: BLE001 - the refusal is the expected outcome
            second["error"] = e

    racer = threading.Thread(target=_second_resume, daemon=True)

    def _race() -> None:
        # The first resume is past its check and has not committed yet: start
        # the second caller now and give it every chance to get through.
        racer.start()
        racer.join(timeout=1.0)

    ledger.on_first_resume_record = _race

    first = runtime.resume(workflow=wf, run_id=run_id, wait_key=wait_key, payload={"response": "first"}, max_steps=0)
    racer.join(timeout=10.0)
    assert not racer.is_alive()

    assert first.status == RunStatus.RUNNING
    assert "state" not in second, "a second resume of an already-resumed wait was accepted"
    assert isinstance(second.get("error"), StaleResumeError)
    assert isinstance(second.get("error"), ValueError)
    assert "not waiting" in str(second["error"])

    resumes = [
        r for r in runtime.get_ledger(run_id)
        if isinstance(r, dict) and (r.get("effect") or {}).get("type") == "resume"
    ]
    assert len(resumes) == 1

    final = runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert final.status == RunStatus.COMPLETED
    assert executed == [{"response": "first"}]


def _two_waits_workflow() -> WorkflowSpec:
    def ask1(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="ask1",
            effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "one?"}, result_key="_temp.one"),
            next_node="ask2",
        )

    def ask2(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="ask2",
            effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "two?"}, result_key="_temp.two"),
            next_node="end",
        )

    def end(run, ctx) -> StepPlan:
        return StepPlan(node_id="end", complete_output={"ok": True})

    return WorkflowSpec(workflow_id="wf_two_waits", entry_node="ask1", nodes={"ask1": ask1, "ask2": ask2, "end": end})


def test_stale_resumes_raise_typed_error_not_waiting_and_key_mismatch(tmp_path) -> None:
    """Both refusals of a stale resume are StaleResumeError (still ValueError).

    The loser of a resume race sees "not waiting" when the winner has not
    reached a new wait yet, and "wait_key mismatch" when the winner's tick
    already parked the run on its NEXT wait. Hosts treat both as a lost race.
    """
    runtime = Runtime(run_store=JsonFileRunStore(tmp_path), ledger_store=InMemoryLedgerStore())
    wf = _two_waits_workflow()
    run_id = runtime.start(workflow=wf, vars={"_temp": {}, "_limits": {}, "_runtime": {}})
    first_wait = runtime.tick(workflow=wf, run_id=run_id, max_steps=10).waiting.wait_key

    # Resumed without ticking: RUNNING, no wait -> "not waiting".
    runtime.resume(workflow=wf, run_id=run_id, wait_key=first_wait, payload={"response": "a"}, max_steps=0)
    with pytest.raises(StaleResumeError, match="Run is not waiting") as not_waiting:
        runtime.resume(workflow=wf, run_id=run_id, wait_key=first_wait, payload={"response": "b"}, max_steps=0)
    assert isinstance(not_waiting.value, ValueError)

    # The winner's tick moved on to the next wait -> "wait_key mismatch".
    st = runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert st.status == RunStatus.WAITING and st.waiting.wait_key != first_wait
    with pytest.raises(StaleResumeError, match="wait_key mismatch") as mismatch:
        runtime.resume(workflow=wf, run_id=run_id, wait_key=first_wait, payload={"response": "b"}, max_steps=0)
    assert isinstance(mismatch.value, ValueError)
