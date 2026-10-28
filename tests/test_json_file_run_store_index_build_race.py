"""An index build never loses a run created by another store object while it builds (gateway backlog 0963).

`JsonFileRunStore._save` appends a new run's `+` journal line BEFORE it publishes the file. An index build
marks the journal, then scans the folder: a run whose line landed before the mark and whose file landed after
the scan was in neither, so `list_run_index(session_id=...)` missed it until the full reconciliation (600 s).
Seen as a flaky gateway test: archiving a session just written answered 404 `session_not_found` because the
gateway's boot thread warmed its store's index while the test wrote the runs.
"""
from __future__ import annotations

from pathlib import Path

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.storage.json_files import JsonFileRunStore


def _run(rid: str, sid: str) -> RunState:
    return RunState(run_id=rid, workflow_id="chat", status=RunStatus.COMPLETED, current_node="n", vars={}, session_id=sid)


def _foreign_with_build_in_the_window(base: Path, service: JsonFileRunStore) -> JsonFileRunStore:
    """Another writer whose new run's `+` line is followed by `service`'s index build BEFORE the file
    is published (the narrow window a busy machine hits by chance)."""
    foreign = JsonFileRunStore(base)
    real_append = foreign._journal_append

    def append_then_build(line: str) -> None:
        real_append(line)
        if line.startswith("+"):
            service.warm_session_index()

    foreign._journal_append = append_then_build  # type: ignore[method-assign]
    return foreign


def test_a_run_published_after_the_build_scanned_is_found(tmp_path: Path) -> None:
    service = JsonFileRunStore(tmp_path)
    foreign = _foreign_with_build_in_the_window(tmp_path, service)
    foreign.save(_run("root-s1", "s1"))
    assert (tmp_path / "run_root-s1.json").exists()
    assert [r["run_id"] if isinstance(r, dict) else getattr(r, "run_id", r) for r in service.list_run_index(session_id="s1", limit=5)] == ["root-s1"]
    assert "root-s1" in {r["run_id"] if isinstance(r, dict) else getattr(r, "run_id", r) for r in service.list_run_index(limit=50)}


def test_a_long_journal_replays_only_its_tail(tmp_path: Path) -> None:
    """The replay is bounded (one read of the tail, once per build) and still finds the run."""
    filler = JsonFileRunStore(tmp_path)
    for i in range(1200):  # ~50 KB of records before the race, more than the replayed tail
        filler._journal_append(f"+old-{i:05d}-0123456789abcdef0123456789abcdef")
    service = JsonFileRunStore(tmp_path)
    foreign = _foreign_with_build_in_the_window(tmp_path, service)
    foreign.save(_run("root-s2", "s2"))
    size = service._journal_path.stat().st_size
    assert len(service.list_run_index(session_id="s2", limit=5)) == 1
    start = service._journal_replay_start(size)
    assert 0 < size - start <= JsonFileRunStore.JOURNAL_REPLAY_TAIL_BYTES
    assert service._journal_path.read_bytes()[start - 1:start] == b"\n"  # a record boundary


def test_the_reconciliation_refiles_a_run_the_session_and_children_indexes_missed(tmp_path: Path) -> None:
    """Beyond the replayed tail (more than JOURNAL_REPLAY_TAIL_BYTES of records between the run's `+` line and
    the build), the reconciliation repairs the session and children indexes, not only the listing memo."""
    import os
    import time

    service = JsonFileRunStore(tmp_path)
    service.warm_session_index()  # the memo is complete: rescans are reconciliations
    foreign = JsonFileRunStore(tmp_path)
    real_append = foreign._journal_append

    def flood_then_rebuild(line: str) -> None:
        real_append(line)
        if line.startswith("+root-"):
            for i in range(600):  # ~25 KB of other records after this one
                real_append(f"+other-{i:05d}-0123456789abcdef0123456789abcdef")
            service._session_index = None  # the next query rebuilds (a fresh store object would)
            service._children_index = None
            service._journal_offset = None
            service.warm_session_index()

    foreign._journal_append = flood_then_rebuild  # type: ignore[method-assign]
    foreign.save(_run("root-s3", "s3"))
    child = RunState(run_id="child-s3", workflow_id="c", status=RunStatus.COMPLETED, current_node="n", vars={},
                     session_id="s3", parent_run_id="root-s3")
    JsonFileRunStore(tmp_path).save(child)
    service._session_index.get("s3", {}).pop("root-s3", None)  # what the build missed
    settled = time.time() - JsonFileRunStore.RUN_INDEX_IN_FLIGHT_S - 1  # changed long enough ago to be a miss
    p = tmp_path / "run_root-s3.json"
    os.utime(p, (settled, settled))
    with service._scan_memo_lock:
        service._scan_memo.pop("root-s3", None)
    service.rescan_active_index()
    assert "root-s3" in (service._session_index or {}).get("s3", {})


def test_the_replayed_tail_signals_no_change_for_runs_the_build_already_filed(tmp_path: Path) -> None:
    """0.15.2 review / gateway suite: the replay re-applied ~400 records the build's scan had just filed and
    bumped the change signal for each, so a host's change-driven pass (the gateway's mail notifications)
    re-loaded 400 finished runs once per build. Records whose file the scan filed as it is stay silent."""
    for own in (True, False):  # the runs saved by this store object (the gateway's case) or another one
        base = tmp_path / ("own" if own else "foreign")
        writer = JsonFileRunStore(base)
        for i in range(50):
            writer.save(_run(f"done-{i:03d}", f"s{i}"))
        service = writer if own else JsonFileRunStore(base)
        t0 = service.change_token()
        service.list_run_index(session_id="s7", limit=5)  # builds the session index (and replays the tail)
        service.list_children(parent_run_id="done-001")
        service.list_run_index(limit=100)
        assert not service.changed_run_ids(t0), own
