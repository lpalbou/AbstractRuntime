"""JsonFileRunStore scan memo (2026-07-15, entity's measured incident).

The gateway runner polls `list_runs(status=...)` ~3x per 0.25s; at 3,241 run
files the 512-entry RunState LRU could not cover the directory, so every
poll re-parsed ~2.7k multi-MB, mostly TERMINAL files — one worker thread
pegged ~98% CPU in json.loads, taxing every endpoint through the GIL
(commons c2394/c2431). The memo holds only the small index fields scans
filter on (never vars), covers the whole directory, and is mtime-validated
per use. These pins hold:

- warm scans parse NOTHING for unchanged files (the incident's fix);
- external writes (mtime change) are seen — cross-process freshness intact;
- unparseable files are tombstoned by mtime, not re-parsed every poll;
- delete purges the memo; filters behave identically through the memo;
- list_run_index rows keep their exact shape without full loads;
- the scheduler's due-scan stays correct through the memo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState
from abstractruntime.storage.json_files import JsonFileRunStore


def _mk_run(store: JsonFileRunStore, *, status: RunStatus = RunStatus.COMPLETED, waiting: Any = None, workflow_id: str = "wf", fat: bool = False) -> RunState:
    run = RunState.new(workflow_id=workflow_id, entry_node="n", vars={"blob": "x" * 50_000} if fat else {})
    run.status = status
    run.waiting = waiting
    store.save(run)
    return run


def _count_loads(store: JsonFileRunStore, monkeypatch) -> dict:
    counter = {"n": 0}
    original = store._load_from_path

    def counting(p: Path):
        counter["n"] += 1
        return original(p)

    monkeypatch.setattr(store, "_load_from_path", counting)
    return counter


def test_warm_scans_never_reparse_unchanged_files(tmp_path: Path, monkeypatch) -> None:
    store = JsonFileRunStore(tmp_path, run_cache_max=4)  # LRU far smaller than the dir — the incident shape
    for _ in range(20):
        _mk_run(store, status=RunStatus.COMPLETED, fat=True)
    running = _mk_run(store, status=RunStatus.RUNNING)

    # Cold store (fresh instance = empty memo, like a process restart).
    cold = JsonFileRunStore(tmp_path, run_cache_max=4)
    counter = _count_loads(cold, monkeypatch)
    first = cold.list_runs(status=RunStatus.RUNNING, limit=100)
    assert [r.run_id for r in first] == [running.run_id]
    warmup_loads = counter["n"]
    assert warmup_loads >= 21, "cold scan must parse to learn the fields"

    # Warm scans: the runner-poll shape. Only the MATCH loads (LRU-cached).
    counter["n"] = 0
    for _ in range(5):
        got = cold.list_runs(status=RunStatus.RUNNING, limit=100)
        assert [r.run_id for r in got] == [running.run_id]
    assert counter["n"] == 5, (
        "five warm polls load exactly the five matches — terminal files cost a stat, never a parse"
    )

    # A filter matching NOTHING parses NOTHING warm.
    counter["n"] = 0
    assert cold.list_runs(status=RunStatus.CANCELLED, limit=100) == []
    assert counter["n"] == 0


def test_external_write_is_seen_by_the_memo(tmp_path: Path) -> None:
    writer_a = JsonFileRunStore(tmp_path)
    run = _mk_run(writer_a, status=RunStatus.RUNNING)
    reader = JsonFileRunStore(tmp_path)
    assert [r.run_id for r in reader.list_runs(status=RunStatus.RUNNING)] == [run.run_id]

    # A SECOND instance (another process, in effect) completes the run.
    import os
    import time

    writer_b = JsonFileRunStore(tmp_path)
    loaded = writer_b.load(run.run_id)
    loaded.status = RunStatus.COMPLETED
    writer_b.save(loaded)
    # Guard against filesystems with coarse mtime granularity.
    os.utime(tmp_path / f"run_{run.run_id}.json")

    assert reader.list_runs(status=RunStatus.RUNNING) == [], "stale memo entries must re-validate by mtime"
    assert [r.run_id for r in reader.list_runs(status=RunStatus.COMPLETED)] == [run.run_id]


def test_unparseable_file_is_tombstoned_not_reparsed(tmp_path: Path, monkeypatch) -> None:
    store = JsonFileRunStore(tmp_path)
    _mk_run(store, status=RunStatus.RUNNING)
    torn = tmp_path / "run_torn.json"
    torn.write_text("{this is not json", encoding="utf-8")

    fresh = JsonFileRunStore(tmp_path)
    counter = _count_loads(fresh, monkeypatch)
    fresh.list_runs(status=RunStatus.RUNNING)
    loads_first = counter["n"]
    counter["n"] = 0
    for _ in range(3):
        fresh.list_runs(status=RunStatus.CANCELLED)
    assert counter["n"] == 0, "the torn file is tombstoned by mtime — never re-parsed per poll"
    assert loads_first >= 2


def test_delete_purges_the_memo(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    run = _mk_run(store, status=RunStatus.RUNNING)
    assert store.list_runs(status=RunStatus.RUNNING)
    store.delete(run.run_id)
    assert store.list_runs(status=RunStatus.RUNNING) == []
    assert run.run_id not in store._scan_memo


def test_list_run_index_rows_keep_their_shape_without_full_loads(tmp_path: Path, monkeypatch) -> None:
    store = JsonFileRunStore(tmp_path)
    wait = WaitState(reason=WaitReason.EVENT, wait_key="k", until="2026-08-01T00:00:00+00:00", resume_to_node="n", result_key="r")
    run = _mk_run(store, status=RunStatus.WAITING, waiting=wait, workflow_id="wf-idx")

    fresh = JsonFileRunStore(tmp_path)
    rows = fresh.list_run_index(status=RunStatus.WAITING, limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run.run_id
    assert row["workflow_id"] == "wf-idx"
    assert row["status"] == "waiting"
    assert row["wait_reason"] == "event"
    assert row["wait_until"] == "2026-08-01T00:00:00+00:00"
    assert "run_lifecycle" in row
    assert not any(k.startswith("__") for k in row)

    # Warm: an index page over unchanged files parses nothing at all.
    counter = _count_loads(fresh, monkeypatch)
    fresh.list_run_index(status=RunStatus.WAITING, limit=10)
    assert counter["n"] == 0


def test_due_scan_stays_correct_through_the_memo(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    due = _mk_run(
        store,
        status=RunStatus.WAITING,
        waiting=WaitState(reason=WaitReason.UNTIL, wait_key=None, until="2020-01-01T00:00:00+00:00", resume_to_node="n", result_key=None),
    )
    _mk_run(
        store,
        status=RunStatus.WAITING,
        waiting=WaitState(reason=WaitReason.UNTIL, wait_key=None, until="2099-01-01T00:00:00+00:00", resume_to_node="n", result_key=None),
    )
    _mk_run(store, status=RunStatus.RUNNING)

    fresh = JsonFileRunStore(tmp_path)
    hits = fresh.list_due_wait_until(now_iso="2026-07-15T00:00:00+00:00", limit=10)
    assert [r.run_id for r in hits] == [due.run_id]


# ---------------------------------------------------------------------------
# 2026-07-15 scan-memo adversary fold (P1-1, P2-1..P2-4)
# ---------------------------------------------------------------------------


def test_same_mtime_rewrite_is_seen_via_inode(tmp_path: Path) -> None:
    """P1-1: a rewrite that lands with an IDENTICAL mtime (1s-granularity
    filesystems, rsync -a restores) must still refresh the memo — the
    identity token includes the inode, and save() mints a new inode on
    every tmp.replace()."""
    import os

    store_a = JsonFileRunStore(tmp_path)
    run = _mk_run(store_a, status=RunStatus.COMPLETED)
    p = tmp_path / f"run_{run.run_id}.json"
    st_before = p.stat()

    # Warm A's memo, then rewrite through a SECOND instance with the mtime
    # pinned back to the original (coarse-fs simulation).
    assert store_a.list_runs(status=RunStatus.RUNNING, limit=10) == []
    store_b = JsonFileRunStore(tmp_path)
    flipped = store_b.load(run.run_id)
    assert flipped is not None
    flipped.status = RunStatus.RUNNING
    store_b.save(flipped)
    os.utime(p, ns=(st_before.st_atime_ns, st_before.st_mtime_ns))

    got = store_a.list_runs(status=RunStatus.RUNNING, limit=10)
    assert [r.run_id for r in got] == [run.run_id], "same-mtime rewrite must be visible (inode changed)"
    # Coherence: the filter's verdict and the returned object must agree.
    assert all(r.status == RunStatus.RUNNING for r in got)


def test_glob_matching_copy_is_memoized_not_reparsed(tmp_path: Path, monkeypatch) -> None:
    """P2-1: a stray operator backup (run_<id>_backup.json) memoizes under
    its FILENAME-derived id — warm scans parse neither file, and the copy
    never poisons the original's entry."""
    import shutil

    store = JsonFileRunStore(tmp_path)
    run = _mk_run(store, status=RunStatus.COMPLETED)
    p = tmp_path / f"run_{run.run_id}.json"
    shutil.copy2(p, tmp_path / f"run_{run.run_id}_backup.json")

    cold = JsonFileRunStore(tmp_path)
    counter = _count_loads(cold, monkeypatch)
    cold.list_runs(status=RunStatus.RUNNING, limit=10)
    warmup = counter["n"]
    for _ in range(5):
        cold.list_runs(status=RunStatus.RUNNING, limit=10)
    assert counter["n"] == warmup, "warm scans must not re-parse the copy (ping-pong class)"


def test_returned_index_rows_do_not_alias_the_memo(tmp_path: Path) -> None:
    """P2-2: mutating a returned row's nested dict must not poison what
    future list_run_index calls see."""
    store = JsonFileRunStore(tmp_path)
    run = RunState.new(workflow_id="wf", entry_node="n", vars={"_runtime": {"lifecycle": {"phase": "work"}}})
    run.status = RunStatus.COMPLETED
    store.save(run)

    rows = store.list_run_index(limit=10)
    assert rows
    for row in rows:
        for v in row.values():
            if isinstance(v, dict):
                v.clear()
        row["status"] = "poisoned"

    fresh = store.list_run_index(limit=10)
    assert fresh[0]["status"] == RunStatus.COMPLETED.value
    for v in fresh[0].values():
        assert v != {}, "nested dicts must be copies, not memo aliases"


def test_bogus_status_enum_is_tombstoned_for_scans(tmp_path: Path) -> None:
    """P2-3: a valid-JSON file with an invalid status enum must not kill
    every scan API — it tombstones for scans; direct load() stays loud."""
    import pytest

    store = JsonFileRunStore(tmp_path)
    good = _mk_run(store, status=RunStatus.RUNNING)
    bad = tmp_path / "run_badenum.json"
    payload = json.loads((tmp_path / f"run_{good.run_id}.json").read_text())
    payload["run_id"] = "badenum"
    payload["status"] = "runnning"
    bad.write_text(json.dumps(payload))

    fresh = JsonFileRunStore(tmp_path)
    got = fresh.list_runs(status=RunStatus.RUNNING, limit=10)
    assert [r.run_id for r in got] == [good.run_id], "scans survive the poison file"
    assert fresh.list_run_index(limit=10), "index survives too"
    fresh.list_due_wait_until(now_iso="2999-01-01T00:00:00+00:00")
    with pytest.raises(ValueError):
        fresh.load("badenum")  # repairs must see the real error


def test_externally_removed_runs_are_pruned_from_the_memo(tmp_path: Path) -> None:
    """P2-4: external pruning (the class's own recommended maintenance)
    must not leave memo entries behind for the process lifetime."""
    store = JsonFileRunStore(tmp_path)
    runs = [_mk_run(store, status=RunStatus.COMPLETED) for _ in range(5)]
    keep = _mk_run(store, status=RunStatus.RUNNING)
    store.list_runs(status=RunStatus.RUNNING, limit=10)  # memo warm
    for r in runs:
        (tmp_path / f"run_{r.run_id}.json").unlink()

    store.list_runs(status=RunStatus.RUNNING, limit=10)
    with store._scan_memo_lock:
        remaining = set(store._scan_memo)
    assert remaining == {keep.run_id}, "vanished files must leave the memo"
