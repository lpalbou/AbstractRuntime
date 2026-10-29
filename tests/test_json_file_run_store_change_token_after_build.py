"""An index build (the first `list_run_index` / `warm_session_index` of a store) replays the
journal tail before its mark (0.15.2, backlog 0863) WITHOUT announcing those records as changes:
`change_token` does not move for a read and `changed_run_ids` names nothing. In 0.15.2 the replay
noted every replayed `+id` on the change ring, so after each build a change-driven consumer (the
gateway's mail notification check) saw every finished run as changed and loaded them all
(AbstractGateway backlog 0965: `test_400_finished_runs_a_change_to_one_loads_exactly_one`)."""

from __future__ import annotations

from pathlib import Path

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.storage.json_files import JsonFileRunStore


def _run(rid: str, status: RunStatus, i: int) -> RunState:
    return RunState(run_id=rid, workflow_id="w", status=status, current_node="n", vars={}, actor_id="t",
                    updated_at=f"2020-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00")


def test_an_index_build_does_not_move_the_change_token(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path / "runs")
    for i in range(400):
        store.save(_run(f"done-{i:03d}", RunStatus.COMPLETED if i % 2 else RunStatus.FAILED, i))
    token = store.change_token()
    assert len(store.list_run_index(status=RunStatus.COMPLETED, limit=200)) == 200
    assert len(store.list_run_index(status=RunStatus.FAILED, limit=200)) == 200
    store.warm_session_index()
    assert store.change_token() == token, "a read moved the change token"
    assert store.changed_run_ids(token) == set(), "a read announced runs as changed"

    # A real change after the build is still announced (this store), or unknown (another process).
    store.save(_run("done-400", RunStatus.COMPLETED, 400))
    assert store.changed_run_ids(token) == {"done-400"}
    token2 = store.change_token()
    JsonFileRunStore(tmp_path / "runs").save(_run("foreign", RunStatus.COMPLETED, 401))
    assert store.changed_run_ids(token2) is None


def test_a_run_another_store_created_before_the_mark_is_still_filed_by_the_replay(tmp_path: Path) -> None:
    """The 0.15.2 guarantee (backlog 0863) survives: the replay files a run whose `+` line
    landed before the mark; it only stops announcing it."""
    writer = JsonFileRunStore(tmp_path / "runs")
    for i in range(20):
        writer.save(_run(f"r-{i:02d}", RunStatus.COMPLETED, i))
    reader = JsonFileRunStore(tmp_path / "runs")
    token = reader.change_token()
    writer.save(RunState(run_id="late", workflow_id="w", status=RunStatus.WAITING, current_node="n", vars={},
                         actor_id="t", session_id="s1"))
    reader.warm_session_index()
    rows = reader.list_run_index(session_id="s1", limit=10)
    assert [r["run_id"] for r in rows] == ["late"]
    assert "late" in {r["run_id"] for r in reader.list_run_index(status=RunStatus.WAITING, limit=10)}
    # Another process wrote the journal: the change set is unknown, never silently empty.
    assert reader.changed_run_ids(token) is None


def _publish_without_a_journal_line(src_dir: Path, dest_store: JsonFileRunStore, run: RunState) -> None:
    """A run file landing in the folder with no new journal record: its `+id` was appended earlier
    (J53-1: the line goes first, the file after), here by a store in another folder."""
    import shutil

    JsonFileRunStore(src_dir).save(run)
    shutil.copyfile(src_dir / f"run_{run.run_id}.json", dest_store._base / f"run_{run.run_id}.json.tmp")
    (dest_store._base / f"run_{run.run_id}.json.tmp").replace(dest_store._base / f"run_{run.run_id}.json")


def test_a_run_published_during_the_scan_is_announced_by_the_replay(tmp_path: Path) -> None:
    """Review (DESK1): the replay is quiet only for what the store knew before the build's mark. A
    `+X` appended before the mark whose file lands DURING the scan (the J53-1 window: line first,
    file after) must still be announced: a consumer whose token already covers that line would
    otherwise never learn of X. The file lands on the scan's first file read, so the scan's own
    sync replays `+X` with the file present (the pending re-check never sees it)."""
    writer = JsonFileRunStore(tmp_path / "runs")
    for i in range(20):
        writer.save(_run(f"r-{i:02d}", RunStatus.COMPLETED, i))
    writer._persist_scan_memo()  # the sidecar: what a restarted store knows at its mark
    reader = JsonFileRunStore(tmp_path / "runs")
    writer._journal_append("+late")  # the line, before the file (another process mid-save)
    token = reader.change_token()  # covers the `+late` line already
    real = reader._scan_fields
    landed = {"done": False}

    def scan_fields(p):
        if not landed["done"]:
            landed["done"] = True
            _publish_without_a_journal_line(tmp_path / "other", reader, _run("late", RunStatus.COMPLETED, 99))
        return real(p)

    reader._scan_fields = scan_fields  # type: ignore[method-assign]
    reader.warm_session_index()  # the build: mark, scan (late lands on the first read), sync
    assert landed["done"]
    assert "late" in {r["run_id"] for r in reader.list_run_index(status=RunStatus.COMPLETED, limit=50)}
    assert reader.changed_run_ids(token) == {"late"}, "a run the store did not know before the mark must be announced"


def test_a_store_without_a_sidecar_announces_its_first_replay(tmp_path: Path) -> None:
    """No sidecar, nothing saved by this store: at its mark it knows no file, so the replayed
    records are announced (a bounded one-off: the 16 KiB tail), never silently dropped."""
    writer = JsonFileRunStore(tmp_path / "runs")
    for i in range(20):
        writer.save(_run(f"r-{i:02d}", RunStatus.COMPLETED, i))
    reader = JsonFileRunStore(tmp_path / "runs")
    token = reader.change_token()
    reader.warm_session_index()
    assert reader.changed_run_ids(token) == {f"r-{i:02d}" for i in range(20)}


def test_a_run_unlinked_during_the_scan_is_announced_by_the_replay(tmp_path: Path) -> None:
    """A `-X` appended before the mark whose unlink lands right after the scan recorded X: the
    store knew X (its scan), so the replay announces that it is gone."""
    writer = JsonFileRunStore(tmp_path / "runs")
    for i in range(20):
        writer.save(_run(f"r-{i:02d}", RunStatus.COMPLETED, i))
    writer._persist_scan_memo()
    reader = JsonFileRunStore(tmp_path / "runs")
    writer._journal_append("-r-07")  # the line, before the unlink
    token = reader.change_token()
    real = reader._scan_fields

    def scan_fields(p):
        out = real(p)
        if p.name == "run_r-07.json" and p.exists():
            p.unlink()  # gone right after the scan recorded it
        return out

    reader._scan_fields = scan_fields  # type: ignore[method-assign]
    reader.warm_session_index()  # the build: mark, scan (r-07 recorded, then gone), sync
    assert not (tmp_path / "runs" / "run_r-07.json").exists()
    assert "r-07" not in {r["run_id"] for r in reader.list_run_index(status=RunStatus.COMPLETED, limit=50)}
    assert reader.changed_run_ids(token) == {"r-07"}, "a run gone since the store knew it must be announced"
