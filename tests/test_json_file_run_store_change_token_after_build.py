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
