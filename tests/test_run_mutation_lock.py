"""`run_mutation_lock` serializes every in-process writer of one run (contract C16).

The tick keeps the RunState in memory across steps and saves it after each
one. A writer that load-modify-saves the run between two steps (an automation
command, a controller boundary) was overwritten by the tick's next save. The
tick now holds the run's mutation lock for its whole duration, so such a
writer waits for the tick to end and its change survives.

SQLite store on purpose: its `load()` parses a fresh RunState, so the tick's
in-memory object and the writer's object are distinct (the JSON/in-memory
stores alias loaded runs and would hide the overwrite).
"""

from __future__ import annotations

import threading
import time

import pytest

from abstractruntime import Runtime, StepPlan, WorkflowSpec, run_mutation_lock
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore


def _runtime(tmp_path):
    ledger = InMemoryLedgerStore()
    store = SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    return Runtime(run_store=store, ledger_store=ledger), store, ledger


def test_command_applied_between_two_tick_steps_is_not_overwritten(tmp_path) -> None:
    runtime, store, _ledger = _runtime(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def first(run, ctx) -> StepPlan:
        entered.set()
        assert release.wait(5.0)
        return StepPlan(node_id="first", next_node="second")

    def second(run, ctx) -> StepPlan:
        return StepPlan(node_id="second", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="wf_lock", entry_node="first", nodes={"first": first, "second": second})
    run_id = runtime.start(workflow=wf, vars={})

    ticker = threading.Thread(target=lambda: runtime.tick(workflow=wf, run_id=run_id, max_steps=10), daemon=True)
    ticker.start()
    assert entered.wait(5.0)

    applied = threading.Event()

    def apply_command() -> None:
        with run_mutation_lock(run_id):
            run = store.load(run_id)
            run.vars["command"] = "applied"
            store.save(run)
        applied.set()

    commander = threading.Thread(target=apply_command, daemon=True)
    commander.start()
    # With the lock the command waits for the tick; without it, it lands now
    # and the tick's next save erases it.
    time.sleep(0.2)
    release.set()
    ticker.join(5.0)
    commander.join(5.0)
    assert applied.is_set()

    final = store.load(run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.vars.get("command") == "applied"


def test_same_thread_reentry_does_not_deadlock(tmp_path) -> None:
    runtime, store, _ledger = _runtime(tmp_path)

    def node(run, ctx) -> StepPlan:
        with run_mutation_lock(run.run_id):  # a step touching its own run
            pass
        return StepPlan(node_id="node", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="wf_reentry", entry_node="node", nodes={"node": node})
    run_id = runtime.start(workflow=wf, vars={})
    done = threading.Event()
    t = threading.Thread(target=lambda: (runtime.tick(workflow=wf, run_id=run_id), done.set()), daemon=True)
    t.start()
    assert done.wait(5.0), "re-entrant hold deadlocked"
    assert store.load(run_id).status == RunStatus.COMPLETED


def test_lock_released_when_the_tick_raises(tmp_path) -> None:
    runtime, _store, _ledger = _runtime(tmp_path)

    def boom(run, ctx) -> StepPlan:
        raise RuntimeError("node exploded")

    wf = WorkflowSpec(workflow_id="wf_boom", entry_node="boom", nodes={"boom": boom})
    run_id = runtime.start(workflow=wf, vars={})
    try:
        runtime.tick(workflow=wf, run_id=run_id)
    except RuntimeError:
        pass

    acquired = threading.Event()

    def other_thread() -> None:
        with run_mutation_lock(run_id):
            acquired.set()

    t = threading.Thread(target=other_thread, daemon=True)
    t.start()
    assert acquired.wait(2.0), "tick left the run's mutation lock held"


def test_ledger_store_is_reachable_on_the_run_during_a_tick(tmp_path) -> None:
    runtime, _store, ledger = _runtime(tmp_path)
    seen: list = []

    def node(run, ctx) -> StepPlan:
        seen.append(getattr(run, "_runtime_ledger_store"))
        return StepPlan(node_id="node", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="wf_ledger", entry_node="node", nodes={"node": node})
    run_id = runtime.start(workflow=wf, vars={})
    runtime.tick(workflow=wf, run_id=run_id)
    assert seen == [ledger]
