"""`JsonFileRunStore.change_token()` / `changed_run_ids()`: the store's own save signal.

The gateway runner used to learn "nothing changed" by globbing and stat-ing every run file
every 0.5 s (0.18 s at 11,500 runs: 13-36% of a core on the operator's idle gateway,
2026-10-26). The token is one stat of the creation log plus the active-index counter."""

from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.storage.json_files import JsonFileRunStore, _ChangeRing


def _run(rid: str, status: RunStatus = RunStatus.RUNNING) -> RunState:
    return RunState(run_id=rid, workflow_id="wf", status=status, current_node="n", vars={}, actor_id="gateway")


def test_token_moves_on_create_save_terminal_save_and_delete(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    t0 = store.change_token()
    store.save(_run("a"))  # create
    t1 = store.change_token()
    assert t1 != t0 and store.changed_run_ids(t0) == {"a"}
    store.save(_run("a"))  # the per-tick save of a run that stays RUNNING (no log line)
    t2 = store.change_token()
    assert t2 != t1 and store.changed_run_ids(t1) == {"a"}
    store.save(_run("a", RunStatus.COMPLETED))  # ends
    t3 = store.change_token()
    assert t3 != t2 and store.changed_run_ids(t2) == {"a"}
    store.create_if_absent(_run("b"))
    t4 = store.change_token()
    assert t4 != t3 and store.changed_run_ids(t3) == {"b"}
    store.delete("a")
    t5 = store.change_token()
    assert t5 != t4 and store.changed_run_ids(t4) == {"a"}
    assert store.changed_run_ids(t0) == {"a", "b"}


def test_an_untouched_store_keeps_its_token_through_listings(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    for i in range(30):
        store.save(_run(f"r{i}", RunStatus.COMPLETED if i % 3 else RunStatus.WAITING))
    store.list_runs(status=RunStatus.WAITING, limit=100)  # builds the active index once
    t0 = store.change_token()
    for _ in range(3):
        store.list_runs(status=RunStatus.RUNNING, limit=100)
        store.list_runs(status=RunStatus.WAITING, limit=100)
        store.list_run_index(limit=100)
        store.load("r1")
    assert store.change_token() == t0
    assert store.changed_run_ids(t0) == set()


def test_another_process_creating_a_run_moves_the_token_and_makes_the_set_unknown(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    store.save(_run("mine"))
    t0 = store.change_token()
    JsonFileRunStore(tmp_path).save(_run("theirs"))  # a second writer: its `+id` line
    assert store.change_token() != t0
    assert store.changed_run_ids(t0) is None  # not "nothing": the caller does a full pass


def test_token_is_one_stat_and_no_file_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonFileRunStore(tmp_path)
    for i in range(50):
        store.save(_run(f"r{i}", RunStatus.COMPLETED))
    calls = {"open": 0, "scandir": 0, "stat": 0}
    real_open, real_scandir, real_stat = builtins.open, os.scandir, os.stat
    monkeypatch.setattr(builtins, "open", lambda *a, **k: calls.__setitem__("open", calls["open"] + 1) or real_open(*a, **k))
    monkeypatch.setattr(os, "scandir", lambda *a, **k: calls.__setitem__("scandir", calls["scandir"] + 1) or real_scandir(*a, **k))
    monkeypatch.setattr(os, "stat", lambda *a, **k: calls.__setitem__("stat", calls["stat"] + 1) or real_stat(*a, **k))
    store.change_token()
    assert calls == {"open": 0, "scandir": 0, "stat": 1}


def test_change_ring_reports_unknown_once_entries_are_evicted() -> None:
    ring = _ChangeRing(maxlen=3)
    for seq, rid in enumerate(["a", "b", "c", "d", "e"], start=1):
        ring.note(seq, rid)
    assert ring.since(4) == {"e"}
    assert ring.since(2) == {"c", "d", "e"}
    assert ring.since(1) is None


def test_ledger_change_signal_names_the_ledgers_that_grew(tmp_path: Path) -> None:
    from abstractruntime.core.models import StepRecord, StepStatus
    from abstractruntime.storage.json_files import JsonlLedgerStore
    from abstractruntime.storage.observable import ObservableLedgerStore

    ledger = ObservableLedgerStore(JsonlLedgerStore(tmp_path))
    t0 = ledger.change_token()
    assert ledger.changed_run_ids(t0) == set()
    ledger.list("a")  # reads move nothing
    assert ledger.change_token() == t0
    ledger.append(StepRecord(run_id="a", step_id="s1", node_id="n", status=StepStatus.COMPLETED))
    ledger.append(StepRecord(run_id="b", step_id="s1", node_id="n", status=StepStatus.COMPLETED))
    t1 = ledger.change_token()
    assert t1 != t0 and ledger.changed_run_ids(t0) == {"a", "b"}
    ledger.append(StepRecord(run_id="b", step_id="s2", node_id="n", status=StepStatus.COMPLETED))
    assert ledger.changed_run_ids(t1) == {"b"}
