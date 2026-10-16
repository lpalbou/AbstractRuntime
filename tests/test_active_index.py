"""The ACTIVE index (2026-10-16): a RUNNING/WAITING listing takes its candidates from an
in-memory set instead of globbing + stat()ing every run file, and returns EXACTLY the rows the
scan returns — proven here on synthetic stores (the untracked/active-index harness replays it on a
copy of the live store under writers and a busy thread) — while the set is kept current by
save()/delete(), the creation journal (`+id`, `-id`, and the new `~id` for a run that BECAME
active through another store object), per-candidate verification against disk, and a periodic
background rescan for writers that bypass the journal."""

from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path
from typing import Any, List

from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState
from abstractruntime.storage.json_files import JsonFileRunStore

STATUSES = [RunStatus.RUNNING, RunStatus.WAITING, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]


def _mk(store: JsonFileRunStore, *, status: RunStatus, workflow_id: str = "wf", parent: str | None = None, waiting: Any = None, session: str | None = None) -> RunState:
    run = RunState.new(workflow_id=workflow_id, entry_node="n", vars={})
    run.status = status
    run.waiting = waiting if waiting is not None else (WaitState(reason=WaitReason.USER, wait_key="k") if status == RunStatus.WAITING else None)
    run.parent_run_id = parent
    if session:
        run.session_id = session
    store.save(run)
    return run


def _rows(store: JsonFileRunStore, **kw: Any) -> List[str]:
    return [r["run_id"] for r in store.list_run_index(**kw)]


def _runs(store: JsonFileRunStore, **kw: Any) -> List[str]:
    return [r.run_id for r in store.list_runs(**kw)]


QUERIES = [
    dict(status=RunStatus.RUNNING, limit=50),
    dict(status=RunStatus.WAITING, limit=50),
    dict(status=RunStatus.RUNNING, limit=7),
    dict(status=RunStatus.WAITING, limit=7, oldest_first=True),
    dict(status=RunStatus.RUNNING, limit=200, root_only=True),
    dict(status=RunStatus.WAITING, limit=200, root_only=True),
    dict(status=RunStatus.RUNNING, limit=50, workflow_id="wf-b"),
    dict(status=RunStatus.COMPLETED, limit=50),  # not indexed: the scan, unchanged
    dict(limit=50),  # no status: the scan, unchanged
]


def _seed(store: JsonFileRunStore, n: int, rng: random.Random) -> List[RunState]:
    runs: List[RunState] = []
    for i in range(n):
        st = rng.choice(STATUSES)
        parent = rng.choice(runs).run_id if runs and rng.random() < 0.3 else None
        runs.append(_mk(store, status=st, workflow_id=rng.choice(["wf-a", "wf-b"]), parent=parent))
    # Spread the mtimes so the ranking has real ties to break the same way.
    base = time.time() - n
    for i, r in enumerate(runs):
        p = store._path(r.run_id)
        os.utime(p, ns=(int((base + i) * 1e9), int((base + i) * 1e9)))
    return runs


def test_indexed_listings_equal_the_scan_on_a_mixed_store(tmp_path: Path) -> None:
    rng = random.Random(7)
    ref = JsonFileRunStore(tmp_path, use_active_index=False)
    runs = _seed(ref, 120, rng)
    cand = JsonFileRunStore(tmp_path)  # a second object: builds its index from disk
    for q in QUERIES:
        assert _rows(cand, **q) == _rows(ref, **q), q
    for q in QUERIES[:3]:
        assert _runs(cand, **q) == _runs(ref, **q), q
    assert cand._active_index == {r.run_id for r in runs if r.status in (RunStatus.RUNNING, RunStatus.WAITING)}


def test_the_index_follows_saves_and_deletes_through_the_same_store(tmp_path: Path) -> None:
    rng = random.Random(3)
    store = JsonFileRunStore(tmp_path)
    ref = JsonFileRunStore(tmp_path, use_active_index=False)
    runs = _seed(store, 40, rng)
    store.list_run_index(status=RunStatus.RUNNING, limit=10)  # builds
    for _ in range(60):
        r = rng.choice(runs)
        op = rng.random()
        if op < 0.5:
            r.status = rng.choice(STATUSES)
            r.waiting = WaitState(reason=WaitReason.USER, wait_key="k") if r.status == RunStatus.WAITING else None
            store.save(r)
        elif op < 0.7:
            runs.append(_mk(store, status=rng.choice(STATUSES)))
        elif op < 0.85 and len(runs) > 5:
            gone = runs.pop(rng.randrange(len(runs)))
            store.delete(gone.run_id)
        for q in QUERIES[:6]:
            assert _rows(store, **q) == _rows(ref, **q), q


def test_another_store_object_sees_creation_deletion_and_a_flip_to_active(tmp_path: Path) -> None:
    a = JsonFileRunStore(tmp_path)
    b = JsonFileRunStore(tmp_path)
    done = _mk(a, status=RunStatus.COMPLETED)
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == []  # a's index: empty
    # b creates an active run: `+id`.
    born = _mk(b, status=RunStatus.RUNNING)
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == [born.run_id]
    # b flips a terminal run to active: `~id` (nothing else announces it).
    flipped = b.load(done.run_id)
    assert flipped is not None
    flipped.status = RunStatus.RUNNING
    b.save(flipped)
    assert set(_rows(a, status=RunStatus.RUNNING, limit=10)) == {born.run_id, done.run_id}
    # The per-tick save of a run b already saved as active adds no journal line.
    size_before = a._journal_path.stat().st_size
    flipped.vars["tick"] = 1
    b.save(flipped)
    assert a._journal_path.stat().st_size == size_before
    # b deletes: `-id`.
    b.delete(born.run_id)
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == [done.run_id]
    assert born.run_id not in (a._active_index or set())


def test_a_stale_member_is_filtered_and_dropped_never_listed(tmp_path: Path) -> None:
    a = JsonFileRunStore(tmp_path)
    run = _mk(a, status=RunStatus.RUNNING)
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == [run.run_id]
    # A foreign writer ends the run on disk without telling anyone (a plain rewrite).
    p = a._path(run.run_id)
    text = p.read_text(encoding="utf-8").replace('"status": "running"', '"status": "completed"').replace('"status":"running"', '"status":"completed"')
    tmp = p.with_name(p.name + ".x")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    assert run.run_id in (a._active_index or set())
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == []  # verified against disk
    assert run.run_id not in (a._active_index or set())  # and dropped
    # A vanished file is dropped too.
    other = _mk(a, status=RunStatus.WAITING)
    assert _rows(a, status=RunStatus.WAITING, limit=10) == [other.run_id]
    a._path(other.run_id).unlink()
    assert _rows(a, status=RunStatus.WAITING, limit=10) == []
    assert other.run_id not in (a._active_index or set())


def test_the_background_rescan_catches_a_writer_that_bypasses_the_journal(tmp_path: Path, monkeypatch) -> None:
    a = JsonFileRunStore(tmp_path)
    monkeypatch.setattr(JsonFileRunStore, "ACTIVE_INDEX_RESCAN_S", 0.0)
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == []
    # A foreign process writes an active run file directly (no journal line).
    foreign = RunState.new(workflow_id="wf", entry_node="n", vars={})
    foreign.status = RunStatus.RUNNING
    from abstractruntime.storage.json_files import dumps_compact, runstate_to_dict

    (tmp_path / f"run_{foreign.run_id}.json").write_text(dumps_compact(runstate_to_dict(foreign)), encoding="utf-8")
    # The next listing kicks the rescan (a daemon thread); it lands within a moment.
    deadline = time.time() + 5
    while time.time() < deadline:
        if _rows(a, status=RunStatus.RUNNING, limit=10) == [foreign.run_id]:
            break
        time.sleep(0.02)
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == [foreign.run_id]


def test_listings_under_concurrent_saves_never_deadlock_or_diverge(tmp_path: Path) -> None:
    rng = random.Random(11)
    store = JsonFileRunStore(tmp_path)
    ref = JsonFileRunStore(tmp_path, use_active_index=False)
    runs = _seed(store, 30, rng)
    stop = threading.Event()
    errors: List[str] = []

    def writer() -> None:
        r2 = random.Random(5)
        while not stop.is_set():
            r = r2.choice(runs)
            r.status = r2.choice(STATUSES)
            r.waiting = WaitState(reason=WaitReason.USER, wait_key="k") if r.status == RunStatus.WAITING else None
            try:
                store.save(r)
            except Exception as e:  # noqa: BLE001
                errors.append(f"save: {e}")
            time.sleep(0.001)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    t_end = time.time() + 1.5
    while time.time() < t_end:
        for st in (RunStatus.RUNNING, RunStatus.WAITING):
            rows = store.list_run_index(status=st, limit=100)
            # Every indexed row IS the fields verified against disk at listing time: never a wrong status.
            assert all(r.get("status") == st.value for r in rows)
    stop.set()
    t.join(timeout=5)
    assert errors == []
    # Quiesced: the two paths agree exactly.
    for q in QUERIES[:6]:
        assert _rows(store, **q) == _rows(ref, **q), q


def test_the_flag_keeps_the_plain_scan(tmp_path: Path, monkeypatch) -> None:
    store = JsonFileRunStore(tmp_path, use_active_index=False)
    _mk(store, status=RunStatus.RUNNING)
    store.list_run_index(status=RunStatus.RUNNING, limit=10)
    assert store._active_index is None


def test_create_if_absent_and_same_instant_saves_keep_both_paths_identical(tmp_path: Path) -> None:
    """Harness smoke run 2026-10-16: a run born through create_if_absent was never indexed (this
    object's own journal line is skipped by design), and two files saved in the same nanosecond
    ranked differently on the two paths. Both are pinned here."""
    store = JsonFileRunStore(tmp_path)
    ref = JsonFileRunStore(tmp_path, use_active_index=False)
    assert _rows(store, status=RunStatus.WAITING, limit=10) == []  # builds the (empty) index
    born = RunState.new(workflow_id="wf", entry_node="n", vars={})
    born.status = RunStatus.WAITING
    born.waiting = WaitState(reason=WaitReason.USER, wait_key="k")
    got, created = store.create_if_absent(born)
    assert created is True
    assert _rows(store, status=RunStatus.WAITING, limit=10) == [born.run_id]
    # The id taken: the existing run's status is what counts.
    again, created2 = store.create_if_absent(born)
    assert created2 is False and again.run_id == born.run_id
    # Same-instant ties: pin every file to ONE mtime and compare orders on both paths.
    runs = [_mk(store, status=RunStatus.RUNNING) for _ in range(12)]
    stamp = int(time.time() * 1e9)
    for r in runs + [born]:
        os.utime(store._path(r.run_id), ns=(stamp, stamp))
    for q in (dict(status=RunStatus.RUNNING, limit=50), dict(status=RunStatus.RUNNING, limit=5), dict(status=RunStatus.RUNNING, limit=5, oldest_first=True)):
        assert _rows(store, **q) == _rows(ref, **q), q
    assert _runs(store, status=RunStatus.RUNNING, limit=5) == _runs(ref, status=RunStatus.RUNNING, limit=5)


def test_an_announced_flip_whose_content_has_not_landed_yet_is_re_checked_not_forgotten(tmp_path: Path) -> None:
    """Harness smoke run 2026-10-16 (origin "resurrect"): a reader that synced between another
    process's `~id` line and its file replace read the old, terminal content and dropped the
    announcement for good. The line is written before publication, like `+id`, and a reader
    keeps a `~` entry pending until the file reads active."""
    a = JsonFileRunStore(tmp_path)
    done = _mk(a, status=RunStatus.COMPLETED)
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == []
    # The announcement arrives BEFORE the content (a crashed or slow writer).
    a._journal_append(f"~{done.run_id}")
    a._journal_offset = a._journal_offset - len(f"\n~{done.run_id}\n".encode())  # as if another process wrote it
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == []
    assert done.run_id in a._journal_pending, "the flip stays pending while the file still reads completed"
    # The content lands (through another store object, which announces again after publication).
    b = JsonFileRunStore(tmp_path)
    flipped = b.load(done.run_id)
    assert flipped is not None
    flipped.status = RunStatus.RUNNING
    b.save(flipped)
    assert _rows(a, status=RunStatus.RUNNING, limit=10) == [done.run_id]
    assert done.run_id not in a._journal_pending


def test_a_listing_that_read_a_member_as_terminal_keeps_it_when_a_save_made_it_active_meanwhile(tmp_path: Path, monkeypatch) -> None:
    """Harness 2026-10-16 (200 rounds, origin "save"): the verify step read a member's file as
    terminal, a save flipped it to active and noted it, then the verify's drop erased the note —
    the run was missing until the next rescan. The drop is now conditional on the generation."""
    store = JsonFileRunStore(tmp_path)
    run = _mk(store, status=RunStatus.RUNNING)
    assert _rows(store, status=RunStatus.RUNNING, limit=10) == [run.run_id]
    run.status = RunStatus.COMPLETED
    run.waiting = None
    store.save(run)  # terminal on disk; the save noted it → dropped. Re-add for the race:
    store._note_active_status(run.run_id, "running")
    original = store._scan_fields

    def stale_read_then_flip(p):
        fields = original(p)  # reads the terminal content
        # … and while that read was happening, a save made the run active again.
        run.status = RunStatus.RUNNING
        store.save(run)
        return fields

    monkeypatch.setattr(store, "_scan_fields", stale_read_then_flip)
    assert _rows(store, status=RunStatus.RUNNING, limit=10) == []  # this listing saw the stale read
    monkeypatch.setattr(store, "_scan_fields", original)
    # The member survived the stale drop: the next listing shows the run without any rescan.
    assert run.run_id in (store._active_index or set())
    assert _rows(store, status=RunStatus.RUNNING, limit=10) == [run.run_id]


def test_a_listing_issued_while_this_store_publishes_a_new_active_run_lists_it(tmp_path: Path, monkeypatch) -> None:
    """Harness 2026-10-16 (200 rounds, origin "save"): between `tmp.replace` and the index note a
    listing missed a brand-new run for one call — the scan saw the file, the index did not yet. The
    id is a candidate from before the write (`_active_publishing`)."""
    store = JsonFileRunStore(tmp_path)
    assert _rows(store, status=RunStatus.RUNNING, limit=10) == []
    seen: List[List[str]] = []
    original = store._active_publish_end

    def listing_before_the_note(rid, *, status, token):
        # The file is published, the note has not landed: a listing right now must show the run.
        if token is not None:
            seen.append(_rows(store, status=RunStatus.RUNNING, limit=10))
        return original(rid, status=status, token=token)

    monkeypatch.setattr(store, "_active_publish_end", listing_before_the_note)
    born = _mk(store, status=RunStatus.RUNNING)
    assert seen and seen[0] == [born.run_id]
    assert all(e.publishing == 0 for e in store._active.values())
    monkeypatch.setattr(store, "_active_publish_end", original)
    assert _rows(store, status=RunStatus.RUNNING, limit=10) == [born.run_id]
    # rescan_active_index: synchronous, returns the size, and catches a journal-bypassing file now.
    from abstractruntime.storage.json_files import dumps_compact, runstate_to_dict

    foreign = RunState.new(workflow_id="wf", entry_node="n", vars={})
    foreign.status = RunStatus.RUNNING
    (tmp_path / f"run_{foreign.run_id}.json").write_text(dumps_compact(runstate_to_dict(foreign)), encoding="utf-8")
    store.rescan_active_index()
    assert set(_rows(store, status=RunStatus.RUNNING, limit=10)) == {born.run_id, foreign.run_id}


def test_a_run_ended_by_another_process_and_reopened_here_is_announced(tmp_path: Path) -> None:
    """Harness 2026-10-16 (two writers on one run): store A had saved X active, store B ended it
    (no journal line for that), A re-saved X active from its stale copy — A's own memory said
    "already active", so no `~` went out and B's index never learned the re-open. A now checks the
    file's identity token before trusting its memory."""
    a = JsonFileRunStore(tmp_path)
    b = JsonFileRunStore(tmp_path)
    x = _mk(a, status=RunStatus.RUNNING)  # A saved it active
    assert _rows(b, status=RunStatus.RUNNING, limit=10) == [x.run_id]  # B indexed it (+id)
    ended = b.load(x.run_id)
    assert ended is not None
    ended.status = RunStatus.COMPLETED
    ended.waiting = None
    b.save(ended)
    assert _rows(b, status=RunStatus.RUNNING, limit=10) == []  # B dropped it
    # A re-opens it from its own (stale) copy.
    x.status = RunStatus.RUNNING
    x.vars["again"] = 1
    a.save(x)
    assert _rows(b, status=RunStatus.RUNNING, limit=10) == [x.run_id], "B must learn the re-open from the journal"
    # The common per-tick save (nobody else wrote in between) still adds no line.
    size = a._journal_path.stat().st_size
    x.vars["tick"] = 2
    a.save(x)
    assert a._journal_path.stat().st_size == size


def test_load_mutate_save_after_another_process_ended_the_run_is_announced(tmp_path: Path) -> None:
    """Review 2 P1-1: `load()` refreshes this store's cache with the other writer's version, so a
    cache-token comparison said "unchanged". The announcement now compares against the token of THIS
    store's own last active save."""
    a = JsonFileRunStore(tmp_path)
    b = JsonFileRunStore(tmp_path)
    x = _mk(a, status=RunStatus.RUNNING)
    assert _rows(b, status=RunStatus.RUNNING, limit=10) == [x.run_id]
    ended = b.load(x.run_id)
    ended.status = RunStatus.COMPLETED
    ended.waiting = None
    b.save(ended)
    assert _rows(b, status=RunStatus.RUNNING, limit=10) == []
    fresh = a.load(x.run_id)  # the normal path: load, mutate, save
    fresh.status = RunStatus.RUNNING
    a.save(fresh)
    assert _rows(b, status=RunStatus.RUNNING, limit=10) == [x.run_id]


def test_a_save_during_the_first_build_is_neither_blocked_nor_lost(tmp_path: Path, monkeypatch) -> None:
    """Review 2 P1-2: the cold scan ran under the lock, so every save waited behind it; and a note
    made during a scan must survive the scan's verdict."""
    store = JsonFileRunStore(tmp_path)
    _mk(store, status=RunStatus.COMPLETED)
    original = store._scan_active_ids
    born: List[RunState] = []

    def scan_with_a_save_in_the_middle():
        found = original()
        born.append(_mk(store, status=RunStatus.RUNNING))  # lands while the scan is "running"
        return found

    monkeypatch.setattr(store, "_scan_active_ids", scan_with_a_save_in_the_middle)
    assert _rows(store, status=RunStatus.RUNNING, limit=10) == [born[0].run_id]
    monkeypatch.setattr(store, "_scan_active_ids", original)
    # And a rescan REMOVES a member the directory no longer has (review 2 P2).
    store._path(born[0].run_id).unlink()
    store.rescan_active_index()
    assert store._active_index == set()


def test_due_wait_listing_through_the_index_equals_the_scan(tmp_path: Path) -> None:
    """Review 2 P1-3: the runner's `list_due_wait_until` scanned the whole folder on every poll."""
    store = JsonFileRunStore(tmp_path)
    ref = JsonFileRunStore(tmp_path, use_active_index=False)
    now = "2026-10-16T12:00:00+00:00"
    due = [_mk(store, status=RunStatus.WAITING, waiting=WaitState(reason=WaitReason.UNTIL, wait_key=f"u{i}", until=f"2026-10-16T11:0{i}:00+00:00")) for i in range(3)]
    _mk(store, status=RunStatus.WAITING, waiting=WaitState(reason=WaitReason.UNTIL, wait_key="later", until="2026-10-16T13:00:00+00:00"))
    _mk(store, status=RunStatus.WAITING, waiting=WaitState(reason=WaitReason.USER, wait_key="k"))
    _mk(store, status=RunStatus.COMPLETED)
    got = [r.run_id for r in store.list_due_wait_until(now_iso=now, limit=10)]
    assert got == [r.run_id for r in ref.list_due_wait_until(now_iso=now, limit=10)]
    assert set(got) == {r.run_id for r in due}
    assert [r.run_id for r in store.list_due_wait_until(now_iso=now, limit=2)] == got[:2]
