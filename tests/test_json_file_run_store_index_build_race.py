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
