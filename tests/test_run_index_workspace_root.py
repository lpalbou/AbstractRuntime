"""Run index rows carry `workspace_root` (the folder a run executes in).

Source: the run's inline top-level `vars["workspace_root"]`, only stripped;
None when absent or not a non-empty string. Every store returns it; SQLite
backfills existing rows once; the JSON scan sidecar version bump rebuilds old
memos; the offloading store never moves the key out of the run document.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.in_memory import InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.offloading import OffloadingRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore


def make_store(kind, tmp_path):
    if kind == "memory":
        return InMemoryRunStore()
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    if kind == "sqlite":
        return SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    return OffloadingRunStore(JsonFileRunStore(tmp_path / "runs"), artifact_store=FileArtifactStore(tmp_path / "a"))


def _run(run_id, vars_, status=RunStatus.RUNNING):
    return RunState(run_id=run_id, workflow_id="wf", status=status, current_node="n", vars=vars_, session_id="s",
                    created_at="2026-09-27T10:00:00+00:00", updated_at="2026-09-27T10:00:00+00:00")


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite", "offload"])
def test_rows_carry_the_runs_workspace_root(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    store.save(_run("with", {"workspace_root": "  /Users/me/work/discussion-1 "}))
    store.save(_run("without", {}))
    store.save(_run("blank", {"workspace_root": "   "}))
    store.save(_run("nested-only", {"_runtime": {"workspace_root": "/x"}}))
    rows = {r["run_id"]: r for r in store.list_run_index(limit=10)}
    assert rows["with"]["workspace_root"] == "/Users/me/work/discussion-1"  # stripped, not resolved
    for rid in ("without", "blank", "nested-only"):
        assert "workspace_root" in rows[rid] and rows[rid]["workspace_root"] is None


def test_sqlite_backfills_existing_rows_once(tmp_path) -> None:
    db_path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, status TEXT NOT NULL, "
        "wait_reason TEXT, wait_until TEXT, parent_run_id TEXT, actor_id TEXT, session_id TEXT, "
        "created_at TEXT, updated_at TEXT, run_json TEXT NOT NULL, paused INTEGER, run_lifecycle_json TEXT);"
    )
    for rid, vars_ in (("old-with", {"workspace_root": "/srv/ws"}), ("old-without", {})):
        doc = {"run_id": rid, "workflow_id": "wf", "status": "completed", "current_node": "n", "vars": vars_,
               "session_id": "s", "created_at": "2026-09-27T09:00:00+00:00", "updated_at": "2026-09-27T09:00:00+00:00"}
        conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?);",
                     (rid, "wf", "completed", None, None, None, None, "s", doc["created_at"], doc["updated_at"],
                      json.dumps(doc), 0, "null"))
    conn.commit()
    conn.close()

    store = SqliteRunStore(SqliteDatabase(db_path))
    rows = {r["run_id"]: r for r in store.list_run_index(limit=10)}
    assert rows["old-with"]["workspace_root"] == "/srv/ws"
    assert rows["old-without"]["workspace_root"] is None
    raw = sqlite3.connect(db_path)
    assert raw.execute("SELECT name FROM index_migrations").fetchall() == [("runs.workspace_root.v1",)]
    # Recorded as done: a later open does not re-run it (a row nulled by hand stays null).
    raw.execute("UPDATE runs SET workspace_root = NULL WHERE run_id = 'old-with';")
    raw.commit()
    SqliteRunStore(SqliteDatabase(db_path))
    assert raw.execute("SELECT workspace_root FROM runs WHERE run_id='old-with'").fetchone() == (None,)


def test_json_sidecar_without_the_field_is_rebuilt(tmp_path) -> None:
    store = JsonFileRunStore(tmp_path / "runs")
    store.save(_run("r", {"workspace_root": "/srv/ws"}))
    store.list_run_index()
    store._persist_scan_memo()
    sidecar = tmp_path / "runs" / ".runs_scan_cache.json"
    data = json.loads(sidecar.read_text())
    for _rid, (_token, fields) in data["entries"].items():
        fields.pop("workspace_root", None)
    data["version"] = 2  # a sidecar from before the field existed
    sidecar.write_text(json.dumps(data))
    (row,) = JsonFileRunStore(tmp_path / "runs").list_run_index()
    assert row["workspace_root"] == "/srv/ws"


def test_the_offloader_keeps_workspace_root_inline(tmp_path) -> None:
    inner = JsonFileRunStore(tmp_path / "runs")
    store = OffloadingRunStore(inner, artifact_store=FileArtifactStore(tmp_path / "a"), max_inline_bytes=16)
    long_root = "/srv/" + "w" * 200  # far above the inline cap
    store.save(_run("r", {"workspace_root": long_root, "_temp": {"blob": "x" * 200}}, status=RunStatus.COMPLETED))
    stored = inner.load("r")
    assert stored.vars["workspace_root"] == long_root
    assert stored.vars["_temp"]["blob"] != "x" * 200  # private namespaces still offload
    (row,) = store.list_run_index()
    assert row["workspace_root"] == long_root
