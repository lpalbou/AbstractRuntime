"""The TRUSTED run index (2026-10-26, the gateway's idle core, backlog 1054).

Every listing that is not RUNNING/WAITING or session-scoped used to glob + stat the whole run
folder (11,500 files on the operator's store). Once one complete pass has filled the scan memo,
`JsonFileRunStore` (`trust_index=True`, the default) answers those listings from memory: no glob,
no stat of a run file. These tests pin:

1. the differential proof: 2,000 generated runs, random saves / finishes / re-saves / creates /
   deletes from two threads (one through the store itself, one through a second store object on
   the same folder, i.e. another writer), and the trusted listing equals the full-scan listing
   (a `use_active_index=False, trust_index=False` reference) after every step and after a
   reconciliation, which finds nothing to repair;
2. an idle minute stats no run file (os.stat counted), and the counter is live (verify mode,
   same calls: thousands);
3. the multi-process rule: a second PROCESS writing the folder through a store is seen at the
   next listing (the creation log), and one that bypasses the log is seen by the background
   reconciliation within ACTIVE_INDEX_RESCAN_S (scaled down here; the 60 s run is
   `ABSTRACTRUNTIME_SLOW_TESTS=1`), counted as repairs;
4. shadow mode (`trust_index=False`): the scan answers, the in-memory answer is compared, and
   a difference is counted in `run_index_stats()["shadow_mismatches"]`.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState
from abstractruntime.storage.json_files import JsonFileRunStore

ACTIVE = (RunStatus.RUNNING, RunStatus.WAITING)
TERMINAL = (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)
SRC = Path(__file__).resolve().parents[1] / "src"


def _new(*, status: RunStatus, workflow_id: str = "wf-a", automation: Any = None, parent: Any = None, session: Any = None) -> RunState:
    run = RunState.new(workflow_id=workflow_id, entry_node="n", vars={})
    _set_status(run, status)
    run.parent_run_id = parent
    if session:
        run.session_id = session
    if automation is not None:
        aid, role, idx = automation
        meta: Dict[str, Any] = {"automation_id": aid, "role": role}
        if idx is not None:
            meta["occurrence_index"] = idx
        run.vars["_meta"] = {"automation_run": meta}
    return run


def _set_status(run: RunState, status: RunStatus) -> None:
    run.status = status
    run.waiting = WaitState(reason=WaitReason.USER, wait_key="k") if status == RunStatus.WAITING else None


def _reference(base: Path) -> JsonFileRunStore:
    """The directory scan (disk truth): no active index, no trusted index."""
    return JsonFileRunStore(base, use_active_index=False, trust_index=False)


def _all_rows(store: JsonFileRunStore) -> List[Dict[str, Any]]:
    return store.list_run_index(limit=10**7)


QUERIES: List[Dict[str, Any]] = [
    dict(limit=20),
    dict(limit=200),
    dict(limit=50, oldest_first=True),
    dict(limit=200, root_only=True),
    dict(status=RunStatus.COMPLETED, limit=100),
    dict(status=RunStatus.FAILED, limit=7),
    dict(workflow_id="wf-b", limit=100),
    dict(role="controller", limit=10**6),
    dict(automation_id="auto-1", role="occurrence", limit=10**6),
    dict(session_kind="automation", limit=100),
]
RUN_QUERIES: List[Dict[str, Any]] = [
    dict(limit=20),
    dict(status=RunStatus.COMPLETED, limit=50),
    dict(status=RunStatus.CANCELLED, workflow_id="wf-a", limit=30),
]


def _assert_same(cand: JsonFileRunStore, ref: JsonFileRunStore, *, full: bool) -> None:
    assert _all_rows(cand) == _all_rows(ref)
    if full:
        for q in QUERIES:
            assert cand.list_run_index(**q) == ref.list_run_index(**q), q
        for q in RUN_QUERIES:
            assert [r.run_id for r in cand.list_runs(**q)] == [r.run_id for r in ref.list_runs(**q)], q
        assert cand.latest_occurrence_row("auto-1") == ref.latest_occurrence_row("auto-1")


def _seed(store: JsonFileRunStore, n: int, rng: random.Random) -> List[RunState]:
    runs: List[RunState] = []
    for i in range(n):
        r = rng.random()
        if r < 0.03:
            run = _new(status=rng.choice([RunStatus.WAITING, RunStatus.COMPLETED]), automation=(f"auto-{i % 4}", "controller", None))
        elif r < 0.25:
            run = _new(status=rng.choice(TERMINAL + ACTIVE), automation=(f"auto-{i % 4}", "occurrence", i))
        else:
            parent = rng.choice(runs).run_id if runs and rng.random() < 0.2 else None
            run = _new(status=rng.choice(TERMINAL + (RunStatus.RUNNING,)), workflow_id=rng.choice(["wf-a", "wf-b"]), parent=parent)
        store.save(run)
        runs.append(run)
    return runs


def _random_op(store: JsonFileRunStore, mine: List[RunState], rng: random.Random) -> None:
    """One random write on this thread's own runs (one writer per run: the store's contract)."""
    op = rng.random()
    if op < 0.30 and mine:  # re-save with a field change (a tick, a rename, a re-file)
        run = rng.choice(mine)
        run.workflow_id = rng.choice(["wf-a", "wf-b"])
        run.vars["tick"] = rng.random()
        store.save(run)
    elif op < 0.55 and mine:  # finish (or re-finish) a run
        run = rng.choice(mine)
        _set_status(run, rng.choice(TERMINAL))
        store.save(run)
    elif op < 0.70 and mine:  # (re)activate a run
        run = rng.choice(mine)
        _set_status(run, rng.choice(ACTIVE))
        store.save(run)
    elif op < 0.88:  # create
        run = _new(status=rng.choice(TERMINAL + ACTIVE), workflow_id=rng.choice(["wf-a", "wf-b"]),
                   automation=("auto-1", "occurrence", rng.randrange(10_000)) if rng.random() < 0.3 else None)
        if rng.random() < 0.5:
            store.save(run)
        else:
            store.create_if_absent(run)
        mine.append(run)
    elif mine:  # delete
        run = mine.pop(rng.randrange(len(mine)))
        store.delete(run.run_id)


def test_differential_two_threads_on_2000_runs(tmp_path: Path) -> None:
    rng = random.Random(20261026)
    cand = JsonFileRunStore(tmp_path)  # trusted (the default)
    other = JsonFileRunStore(tmp_path)  # a second writer on the same folder (its own memo)
    cand.RUN_INDEX_IN_FLIGHT_S = 0.0  # every reconciliation difference counts, however recent
    runs = _seed(cand, 2000, rng)
    rng.shuffle(runs)
    mine_a, mine_b = runs[:1000], runs[1000:]
    cand.warm_session_index()  # the boot pass: the index is complete from here on
    assert cand.run_index_stats()["complete"] is True
    ref = _reference(tmp_path)
    _assert_same(cand, ref, full=True)

    rng_a, rng_b = random.Random(1), random.Random(2)
    for step in range(240):
        ta = threading.Thread(target=_random_op, args=(cand, mine_a, rng_a))
        tb = threading.Thread(target=_random_op, args=(other, mine_b, rng_b))
        ta.start(); tb.start(); ta.join(); tb.join()
        _assert_same(cand, ref, full=step % 20 == 0)
        if step % 60 == 59:
            before = cand.run_index_stats()["reconcile_repairs"]
            cand.reconcile_run_index()
            assert cand.run_index_stats()["reconcile_repairs"] == before == 0, cand.run_index_stats()["last_reconcile"]
            _assert_same(cand, ref, full=True)

    stats = cand.run_index_stats()
    assert stats["mode"] == "trusted" and stats["scanned_listings"] == 0 and stats["trusted_listings"] > 400
    assert stats["reconciles"] >= 4 and stats["reconcile_repairs"] == 0
    assert stats["entries"] == len(list(tmp_path.glob("run_*.json")))


def test_a_reconciliation_running_beside_the_writers_changes_nothing(tmp_path: Path) -> None:
    """The background pass races the writers (the real deployment): rows still equal the scan."""
    rng = random.Random(5)
    cand = JsonFileRunStore(tmp_path)
    other = JsonFileRunStore(tmp_path)
    runs = _seed(cand, 600, rng)
    mine_a, mine_b = runs[:300], runs[300:]
    cand.warm_session_index()
    ref = _reference(tmp_path)
    stop = threading.Event()

    def reconcile_loop() -> None:
        while not stop.is_set():
            cand.rescan_active_index()

    t = threading.Thread(target=reconcile_loop)
    t.start()
    try:
        rng_a, rng_b = random.Random(3), random.Random(4)
        for _ in range(150):
            ta = threading.Thread(target=_random_op, args=(cand, mine_a, rng_a))
            tb = threading.Thread(target=_random_op, args=(other, mine_b, rng_b))
            ta.start(); tb.start(); ta.join(); tb.join()
            assert _all_rows(cand) == _all_rows(ref)
    finally:
        stop.set()
        t.join()
    cand.rescan_active_index()
    assert _all_rows(cand) == _all_rows(ref)


def _count_stats(monkeypatch) -> Dict[str, int]:
    counts = {"run_files": 0, "other": 0}
    real = os.stat

    def counting(path, *a, **kw):
        name = os.path.basename(os.fspath(path)) if not isinstance(path, int) else ""
        if name.startswith("run_") and name.endswith(".json"):
            counts["run_files"] += 1
        else:
            counts["other"] += 1
        return real(path, *a, **kw)

    monkeypatch.setattr(os, "stat", counting)
    return counts


def _idle_battery(store: JsonFileRunStore) -> None:
    """What the gateway's pollers ask of a store at rest, except RUNNING/WAITING (the active index
    keeps checking its members by stat, as before)."""
    store.list_run_index(limit=200)
    store.list_run_index(limit=200, root_only=True)
    store.list_run_index(role="controller", limit=10**6)
    store.list_run_index(automation_id="auto-1", role="occurrence", limit=10**6)
    store.latest_occurrence_row("auto-2")
    store.list_run_index(status=RunStatus.COMPLETED, limit=200)
    store.list_runs(status=RunStatus.COMPLETED, limit=20)
    store.list_runs(status=RunStatus.FAILED, limit=200)
    store.list_runs(limit=20)
    store.change_token()


def test_an_idle_minute_stats_no_run_file(tmp_path: Path, monkeypatch) -> None:
    rng = random.Random(9)
    _seed(JsonFileRunStore(tmp_path), 2000, rng)
    store = JsonFileRunStore(tmp_path)
    store.warm_session_index()
    _idle_battery(store)  # first calls after boot (the memo sidecar persists once)
    reconciles0 = store.run_index_stats()["reconciles"]
    counts = _count_stats(monkeypatch)
    threads0 = threading.active_count()
    for _ in range(60):  # one minute of 1 s polls; every one finds the reconciliation "due"
        store._active_built_at -= store.ACTIVE_INDEX_RESCAN_S + 1.0
        _idle_battery(store)
    assert counts["run_files"] == 0, counts
    assert counts["other"] <= 60 * 10 * 3  # the creation log and the folder: one stat each per call
    assert store.run_index_stats()["reconciles"] == reconciles0, "nothing moved: no reconciliation pass"
    assert threading.active_count() == threads0

    # The counter is live: the same minute in verify mode stats every run file on every call.
    verify = JsonFileRunStore(tmp_path, trust_index=False)
    verify.warm_session_index()
    counts["run_files"] = 0
    _idle_battery(verify)
    assert counts["run_files"] > 2000, counts


def test_a_finished_run_writes_one_log_line_and_ticks_write_none(tmp_path: Path) -> None:
    a = JsonFileRunStore(tmp_path)
    run = _new(status=RunStatus.RUNNING)
    a.save(run)
    log = tmp_path / ".runs_created.log"
    size = log.stat().st_size
    run.vars["tick"] = 1
    a.save(run)  # per-tick save of a run this store saved as active
    assert log.stat().st_size == size
    _set_status(run, RunStatus.COMPLETED)
    a.save(run)
    assert log.read_text().split()[-1] == f"={run.run_id}"
    run.vars["archived"] = True
    a.save(run)  # a finished run re-saved
    assert log.read_text().split()[-2:] == [f"={run.run_id}", f"={run.run_id}"]


def test_another_writer_finishing_or_retitling_a_run_is_seen_without_a_scan(tmp_path: Path) -> None:
    reader = JsonFileRunStore(tmp_path)
    writer = JsonFileRunStore(tmp_path)
    run = _new(status=RunStatus.RUNNING)
    writer.save(run)
    reader.warm_session_index()
    assert [r["status"] for r in reader.list_run_index(limit=10)] == ["running"]
    run.vars["tick"] = 2
    writer.save(run)  # a tick in the other writer: no log line, the folder moved
    assert reader.list_run_index(limit=10)[0]["updated_at"] == run.updated_at
    _set_status(run, RunStatus.COMPLETED)
    writer.save(run)  # `=id`
    assert [r["status"] for r in reader.list_run_index(limit=10)] == ["completed"]
    run.workflow_id = "wf-renamed"
    writer.save(run)  # `=id` again (a finished run re-saved)
    assert [r["workflow_id"] for r in reader.list_run_index(limit=10)] == ["wf-renamed"]
    writer.delete(run.run_id)  # `-id`
    assert reader.list_run_index(limit=10) == []
    assert reader.run_index_stats()["entries"] == 0
    assert reader.run_index_stats()["scanned_listings"] == 0


def _bypass_write(base: Path, run: RunState) -> None:
    """A writer that bypasses the creation log (an older store, a copy tool): tmp + replace."""
    from abstractruntime.storage.serialize import dumps_compact, runstate_to_dict

    p = base / f"run_{run.run_id}.json"
    tmp = base / f".bypass.{run.run_id}.tmp"
    tmp.write_text(dumps_compact(runstate_to_dict(run)), encoding="utf-8")
    tmp.replace(p)


def test_a_write_that_bypasses_the_log_is_repaired_by_the_reconciliation(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    store.RUN_INDEX_IN_FLIGHT_S = 0.0
    done = _new(status=RunStatus.COMPLETED)
    store.save(done)
    store.warm_session_index()
    ref = _reference(tmp_path)
    ghost = _new(status=RunStatus.FAILED, workflow_id="wf-ghost")
    _bypass_write(tmp_path, ghost)
    done.workflow_id = "wf-edited"
    _bypass_write(tmp_path, done)
    # Documented risk: a stale row until the reconciliation (never a lost run: load() reads disk).
    assert _all_rows(store) != _all_rows(ref)
    store.reconcile_run_index()
    assert _all_rows(store) == _all_rows(ref)
    assert store.load(ghost.run_id) is not None
    st = store.run_index_stats()
    assert st["reconcile_repairs"] == 2 and set(st["last_reconcile"]["sample"]) == {ghost.run_id, done.run_id}
    (tmp_path / f"run_{ghost.run_id}.json").unlink()  # a delete without a `-` line
    store.reconcile_run_index()
    assert _all_rows(store) == _all_rows(ref)
    assert store.run_index_stats()["reconcile_repairs"] == 3


def test_a_file_created_while_the_pass_ran_is_never_pruned(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    store.save(_new(status=RunStatus.COMPLETED))
    store.warm_session_index()
    late = _new(status=RunStatus.COMPLETED)
    store.save(late)
    # The glob of a pass that began before `late` existed does not list it: the prune keeps it.
    assert store._scan_memo_prune(set()) == [] or late.run_id not in store._scan_memo_prune(set())
    assert late.run_id in {r["run_id"] for r in store.list_run_index(limit=10)}


def test_a_rewritten_log_falls_back_to_the_scan_then_trusts_again(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    run = _new(status=RunStatus.COMPLETED)
    store.save(run)
    store.warm_session_index()
    (tmp_path / ".runs_created.log").write_text("")  # shrunk: every index rebuilds from disk
    ghost = _new(status=RunStatus.COMPLETED)
    _bypass_write(tmp_path, ghost)
    assert {r["run_id"] for r in store.list_run_index(limit=10)} == {run.run_id, ghost.run_id}
    assert store.run_index_stats()["scanned_listings"] == 1  # this call: the scan
    assert {r["run_id"] for r in store.list_run_index(limit=10)} == {run.run_id, ghost.run_id}
    st = store.run_index_stats()
    assert st["complete"] is True and st["scanned_listings"] == 1 and st["trusted_listings"] >= 1


def _subprocess(code: str, base: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run([sys.executable, "-c", textwrap.dedent(code), str(base)], check=True, env=env, timeout=120)


def test_a_second_process_writing_through_a_store_is_seen_at_the_next_listing(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path)
    keep = _new(status=RunStatus.COMPLETED)
    gone = _new(status=RunStatus.COMPLETED)
    store.save(keep)
    store.save(gone)
    store.warm_session_index()
    _subprocess(
        f"""
        import sys
        from abstractruntime.core.models import RunState, RunStatus
        from abstractruntime.storage.json_files import JsonFileRunStore
        s = JsonFileRunStore(sys.argv[1])
        r = RunState.new(workflow_id="wf-child", entry_node="n", vars={{}})
        r.status = RunStatus.FAILED
        s.save(r)
        k = s.load({keep.run_id!r})
        k.workflow_id = "wf-edited-elsewhere"
        s.save(k)
        s.delete({gone.run_id!r})
        """,
        tmp_path,
    )
    rows = {r["run_id"]: r for r in store.list_run_index(limit=10)}
    assert gone.run_id not in rows
    assert rows[keep.run_id]["workflow_id"] == "wf-edited-elsewhere"
    assert sorted(r["workflow_id"] for r in rows.values()) == ["wf-child", "wf-edited-elsewhere"]
    assert _all_rows(store) == _all_rows(_reference(tmp_path))
    assert store.run_index_stats()["reconcile_repairs"] == 0


def _second_process_bypassing_the_log(tmp_path: Path, interval_s: float) -> None:
    store = JsonFileRunStore(tmp_path)
    store.ACTIVE_INDEX_RESCAN_S = interval_s
    store.RUN_INDEX_IN_FLIGHT_S = 0.0
    edited = _new(status=RunStatus.COMPLETED)
    store.save(edited)
    store.warm_session_index()
    _subprocess(
        f"""
        import sys
        from pathlib import Path
        from abstractruntime.core.models import RunState, RunStatus
        from abstractruntime.storage.serialize import dumps_compact, runstate_to_dict
        base = Path(sys.argv[1])
        def write(r):
            tmp = base / (".x." + r.run_id + ".tmp")
            tmp.write_text(dumps_compact(runstate_to_dict(r)), encoding="utf-8")
            tmp.replace(base / ("run_" + r.run_id + ".json"))
        r = RunState.new(workflow_id="wf-foreign", entry_node="n", vars={{}})
        r.status = RunStatus.COMPLETED
        write(r)
        import json
        p = base / "run_{edited.run_id}.json"
        d = json.loads(p.read_text())
        d["workflow_id"] = "wf-foreign-edit"
        tmp = base / ".x.edit.tmp"
        tmp.write_text(json.dumps(d), encoding="utf-8")
        tmp.replace(p)
        """,
        tmp_path,
    )
    t0 = time.monotonic()
    want = sorted(r["workflow_id"] for r in _all_rows(_reference(tmp_path)))
    assert want == ["wf-foreign", "wf-foreign-edit"]
    seen_after = None
    while time.monotonic() - t0 < interval_s + 30.0:
        if sorted(r["workflow_id"] for r in store.list_run_index(limit=10)) == want:
            seen_after = time.monotonic() - t0
            break
        time.sleep(min(1.0, interval_s / 10))
    assert seen_after is not None and seen_after <= interval_s + 5.0, seen_after
    st = store.run_index_stats()
    assert st["reconcile_repairs"] == 2, st
    assert st["scanned_listings"] == 0


def test_a_second_process_bypassing_the_log_is_seen_by_the_reconciliation(tmp_path: Path) -> None:
    _second_process_bypassing_the_log(tmp_path, interval_s=1.0)


@pytest.mark.skipif(not os.environ.get("ABSTRACTRUNTIME_SLOW_TESTS"), reason="the 60 s run: ABSTRACTRUNTIME_SLOW_TESTS=1")
def test_a_second_process_bypassing_the_log_is_seen_within_60_s(tmp_path: Path) -> None:
    _second_process_bypassing_the_log(tmp_path, interval_s=JsonFileRunStore.ACTIVE_INDEX_RESCAN_S)


def test_verify_mode_answers_with_the_scan_and_counts_differences(tmp_path: Path) -> None:
    rng = random.Random(11)
    shadow = JsonFileRunStore(tmp_path, trust_index=False)
    other = JsonFileRunStore(tmp_path)
    runs = _seed(shadow, 300, rng)
    shadow.warm_session_index()
    ref = _reference(tmp_path)
    mine_a, mine_b = runs[:150], runs[150:]
    for step in range(80):
        _random_op(shadow, mine_a, rng)
        _random_op(other, mine_b, rng)
        _assert_same(shadow, ref, full=step % 20 == 0)
    st = shadow.run_index_stats()
    assert st["mode"] == "verify" and st["trusted_listings"] == 0
    assert st["shadow_compared"] > 80 and st["shadow_mismatches"] == 0, st

    victim = next(r for r in mine_a if r.status in TERMINAL)
    victim.workflow_id = "wf-bypassed"
    _bypass_write(tmp_path, victim)  # nothing announces it: the index is stale, the scan is not
    rows = shadow.list_run_index(limit=10**6)
    assert rows == _all_rows(ref)  # verify mode returns the scan's answer
    st = shadow.run_index_stats()
    assert st["shadow_mismatches"] == 1, st
    assert st["last_mismatch"]["listing"] == "list_run_index" and st["last_mismatch"]["same_ids"] is True


def test_the_wrapper_forwards_the_stats(tmp_path: Path) -> None:
    from abstractruntime.storage.artifacts import FileArtifactStore
    from abstractruntime.storage.offloading import OffloadingRunStore

    inner = JsonFileRunStore(tmp_path / "runs", trust_index=False)
    wrapped = OffloadingRunStore(inner, artifact_store=FileArtifactStore(tmp_path / "artifacts"))
    assert wrapped.run_index_stats()["mode"] == "verify"
    assert inner.trust_index is False and JsonFileRunStore(tmp_path / "runs").trust_index is True


def test_a_read_that_began_before_this_stores_save_never_overwrites_it(tmp_path: Path) -> None:
    """The race the guard closes: a reconciliation (or a load) reads a run file, this store
    saves the run, then the read's older entry lands: the save's entry must win."""
    store = JsonFileRunStore(tmp_path)
    run = _new(status=RunStatus.COMPLETED)
    store.save(run)
    rid = run.run_id
    with store._scan_memo_lock:
        seq0 = store._memo_version  # the read begins
        old_token, old_fields = store._scan_memo[rid]
    run.workflow_id = "wf-saved-after"
    store.save(run)
    store._scan_memo_put(rid, old_token, dict(old_fields), seq0=seq0)  # the read's entry lands late
    assert store._scan_memo[rid][1]["workflow_id"] == "wf-saved-after"
    # A read that began AFTER the save may write (it saw at least the saved file).
    with store._scan_memo_lock:
        seq1 = store._memo_version
    tok = store._scan_memo[rid][0]
    store._scan_memo_put(rid, tok, {**store._scan_memo[rid][1], "workflow_id": "wf-read"}, seq0=seq1)
    assert store._scan_memo[rid][1]["workflow_id"] == "wf-read"
