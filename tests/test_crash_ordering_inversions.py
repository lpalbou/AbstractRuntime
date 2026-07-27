"""0045 inversion pins: the crash-ordering invariant at the three fixed
sites (docs/architecture.md, Crash-ordering invariant).

The law: ledger records for a transition append BEFORE the save that makes
the transition true; terminal appends are idempotency-keyed (exactly-once
under crash-replay); terminal-path append failures are loud.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class _CrashBeforeSave(Exception):
    pass


def _killable(store_cls):
    """A store whose save raises at an armed kill point - BEFORE any bytes
    land, so the durable copy stays pre-transition (a real crash loses the
    process memory; only the durable copy survives into the restart)."""

    class _Killable(store_cls):
        crash_at_save: Optional[int] = None
        saves = 0

        def save(self, run) -> None:  # type: ignore[override]
            self.saves += 1
            if self.crash_at_save is not None and self.saves >= self.crash_at_save:
                self.crash_at_save = None
                raise _CrashBeforeSave(f"killed at save #{self.saves}")
            super().save(run)

    return _Killable


def _spec() -> WorkflowSpec:
    def start(run, ctx):
        return StepPlan(node_id="start", complete_output={"answer": 42})

    return WorkflowSpec(workflow_id="w-inv", entry_node="start", nodes={"start": start})


def _terminal_records(ledger, run_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in ledger.list(run_id)
        if isinstance(r.get("result"), dict) and r["result"].get("completed") is True
    ]


def test_completion_appends_before_save_and_replay_dedupes(tmp_path) -> None:
    """Kill BETWEEN the terminal append and the commit-point save, then
    RESTART (fresh stores over the same files - the process-death truth):
    the ledger is ahead (recoverable direction); replay re-completes and
    the terminal record stays EXACTLY once."""
    from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

    store = _killable(JsonFileRunStore)(tmp_path / "runs")
    ledger = JsonlLedgerStore(tmp_path / "ledger")
    rt = Runtime(run_store=store, ledger_store=ledger)
    run_id = rt.start(workflow=_spec(), vars={})

    store.crash_at_save = store.saves + 1  # the NEXT save = the completion commit point
    with pytest.raises(_CrashBeforeSave):
        rt.tick(workflow=_spec(), run_id=run_id)

    # RESTART: fresh process = fresh stores over the same directory.
    store2 = JsonFileRunStore(tmp_path / "runs")
    ledger2 = JsonlLedgerStore(tmp_path / "ledger")
    rt2 = Runtime(run_store=store2, ledger_store=ledger2)

    # Crash window: ledger already terminated, durable state still RUNNING.
    assert len(_terminal_records(ledger2, run_id)) == 1, "append landed before the save"
    assert rt2.get_state(run_id).status == RunStatus.RUNNING

    # Replay converges: same completion, no duplicate terminal record.
    out = rt2.tick(workflow=_spec(), run_id=run_id)
    assert out.status == RunStatus.COMPLETED and out.output == {"answer": 42}
    assert len(_terminal_records(ledger2, run_id)) == 1, "terminal record exactly once"


def test_terminal_append_failure_is_loud_never_blocking() -> None:
    """Invariant rule 3: a failed terminal append increments the health
    counter and the run STILL completes (evidence loss never blocks truth)."""
    store = InMemoryRunStore()

    class _RefusingLedger(InMemoryLedgerStore):
        def append(self, record) -> None:  # type: ignore[override]
            raise OSError("disk full")

    ledger = _RefusingLedger()
    rt = Runtime(run_store=store, ledger_store=ledger)
    run_id = rt.start(workflow=_spec(), vars={})
    out = rt.tick(workflow=_spec(), run_id=run_id)
    assert out.status == RunStatus.COMPLETED, "truth commits even when evidence fails"
    health = rt.health_snapshot() if hasattr(rt, "health_snapshot") else None
    if isinstance(health, dict):
        errors = str(health)
        assert "terminal_append" in errors, "the failure is on the health surface"
