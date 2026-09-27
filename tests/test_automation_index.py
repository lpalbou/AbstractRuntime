"""Run index attribution columns and automation queries (automations C11 / E).

Every store indexes `automation_id`, `role`, `occurrence_index` and
`session_kind` from the run's inline `_meta`; `list_run_index` filters on them
and `root_only=True` returns TURN ROOTS (parent-less non-controller runs plus
occurrences). SQLite backfills existing rows (guarded); the JSON scan sidecar
bumps its version so rows written before the columns existed are re-read.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from abstractruntime.automation_queries import (
    ChangedSinceUnsupported,
    InvalidCursor,
    latest_occurrence,
    list_automations,
)
from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.in_memory import InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.offloading import OffloadingRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

STORES = ["memory", "json", "sqlite", "offload_sqlite"]
AUTO = "auto-a"


def make_store(kind, tmp_path):
    if kind == "memory":
        return InMemoryRunStore()
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    sqlite_store = SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    if kind == "sqlite":
        return sqlite_store
    return OffloadingRunStore(sqlite_store, artifact_store=FileArtifactStore(tmp_path / "a"))


def _save(store, run_id, *, session_id, parent=None, meta=None, created_at, status=RunStatus.COMPLETED, workflow_id="wf", runtime=None, waiting=None):
    vars_obj = {"_meta": dict(meta or {})}
    if runtime is not None:
        vars_obj["_runtime"] = runtime
    run = RunState(run_id=run_id, workflow_id=workflow_id, status=status, current_node="n", vars=vars_obj,
                   session_id=session_id, parent_run_id=parent, created_at=created_at, updated_at=created_at,
                   waiting=waiting)
    store.save(run)
    return run


def _occ(index, *, role="occurrence", kind="automation", attempt=1):
    return {"occurrence": {"automation_id": AUTO, "occurrence_index": index, "attempt": attempt, "role": role, "session_kind": kind}}


def populate(store):
    _save(store, "chat-1", session_id="s-chat", created_at="2026-09-27T10:00:00+00:00")
    _save(store, "chat-1-child", session_id="s-chat", parent="chat-1", created_at="2026-09-27T10:00:01+00:00")
    _save(store, AUTO, session_id="s-auto", created_at="2026-09-27T10:01:00+00:00", status=RunStatus.WAITING,
          meta={"automation": {"title": "Memory watch", "revision": 1, "context": {"mode": "growing"},
                               "trigger": {"source_id": "schedule", "source_version": 1}}},
          runtime={"automation": {"next_index": 3, "pending_occurrence": None, "paused": False}},
          waiting=WaitState(reason=WaitReason.EVENT, wait_key=f"automation:{AUTO}:wake", until="2026-09-27T10:10:00+00:00"))
    _save(store, "occ-1", session_id="s-auto", parent=AUTO, meta=_occ(1), created_at="2026-09-27T10:02:00+00:00")
    _save(store, "occ-2a1", session_id="s-auto", parent=AUTO, meta=_occ(2), created_at="2026-09-27T10:04:00+00:00", status=RunStatus.FAILED)
    _save(store, "occ-2a2", session_id="s-auto", parent=AUTO, meta=_occ(2, attempt=2), created_at="2026-09-27T10:05:00+00:00")
    _save(store, "occ-1-desc", session_id="s-auto", parent="occ-1", meta=_occ(1, role="descendant"), created_at="2026-09-27T10:02:01+00:00")
    _save(store, "disc-1", session_id="s-disc", meta={"discussion": {"automation_id": AUTO, "occurrence_index": 1}}, created_at="2026-09-27T10:06:00+00:00")
    _save(store, "legacy-1", session_id="s-legacy", workflow_id="scheduled:x", meta={"schedule": {"kind": "scheduled_run"}}, created_at="2026-09-27T10:07:00+00:00")


@pytest.mark.parametrize("kind", STORES)
def test_rows_carry_attribution_and_filters_apply(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    rows = {r["run_id"]: r for r in store.list_run_index(limit=100)}
    assert {k: rows[AUTO][k] for k in ("automation_id", "role", "occurrence_index", "session_kind")} == {
        "automation_id": AUTO, "role": "controller", "occurrence_index": None, "session_kind": "automation"}
    assert (rows["occ-1"]["role"], rows["occ-1"]["occurrence_index"], rows["occ-1"]["session_kind"]) == ("occurrence", 1, "automation")
    assert rows["occ-1-desc"]["role"] == "descendant"
    assert (rows["disc-1"]["role"], rows["disc-1"]["session_kind"]) == ("discussion", "discussion")
    assert (rows["legacy-1"]["role"], rows["legacy-1"]["session_kind"]) == ("legacy_schedule", "automation")
    assert (rows["chat-1"]["role"], rows["chat-1"]["session_kind"]) == (None, "chat")

    ids = lambda **f: {r["run_id"] for r in store.list_run_index(limit=100, **f)}
    assert ids(automation_id=AUTO) == {AUTO, "occ-1", "occ-2a1", "occ-2a2", "occ-1-desc", "disc-1"}
    assert ids(automation_id=AUTO, role="occurrence") == {"occ-1", "occ-2a1", "occ-2a2"}
    assert ids(session_kind="chat,discussion") == {"chat-1", "chat-1-child", "disc-1"}
    assert ids(session_kind=["discussion"]) == {"disc-1"}


@pytest.mark.parametrize("kind", STORES)
def test_root_only_returns_turn_roots(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    roots = {r["run_id"] for r in store.list_run_index(root_only=True, limit=100)}
    # Controllers and descendants are never turn roots; occurrences are.
    assert roots == {"chat-1", "occ-1", "occ-2a1", "occ-2a2", "disc-1", "legacy-1"}
    session = {r["run_id"] for r in store.list_run_index(root_only=True, session_id="s-auto", limit=100)}
    assert session == {"occ-1", "occ-2a1", "occ-2a2"}
    chats = {r["run_id"] for r in store.list_run_index(root_only=True, session_kind="chat,discussion", limit=100)}
    assert chats == {"chat-1", "disc-1"}


@pytest.mark.parametrize("kind", STORES)
def test_latest_occurrence_and_list_automations(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    populate(store)
    assert latest_occurrence(store, AUTO)["run_id"] == "occ-2a2"  # highest index, newest attempt
    assert latest_occurrence(store, "nobody") is None

    page = list_automations(store)
    assert page.next_cursor is None
    (summary,) = page.items
    assert summary["automation_id"] == AUTO
    assert summary["title"] == "Memory watch"
    assert summary["status"] == "active"
    assert summary["context_mode"] == "growing"
    assert summary["trigger"] == {"source_id": "schedule", "source_version": 1}
    assert summary["next_fire_at"] == "2026-09-27T10:10:00+00:00"
    assert summary["occurrence_count"] == 2
    assert list_automations(store, status="paused").items == []
    with pytest.raises(ChangedSinceUnsupported) as info:
        list_automations(store, changed_since="x")
    assert info.value.reason_code == "unsupported_feature"


def test_list_automations_pages_with_a_stable_cursor(tmp_path) -> None:
    store = make_store("sqlite", tmp_path)
    for i in range(5):
        _save(store, f"auto-{i}", session_id=f"s{i}", created_at=f"2026-09-27T10:0{i}:00+00:00",
              meta={"automation": {"title": f"A{i}", "archived_at": "2026-09-27T11:00:00+00:00" if i == 0 else None}})
    first = list_automations(store, limit=2)
    assert [s["automation_id"] for s in first.items] == ["auto-4", "auto-3"]
    # "Restart": a fresh store object resolves the same cursor.
    store2 = SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    second = list_automations(store2, limit=2, cursor=first.next_cursor)
    assert [s["automation_id"] for s in second.items] == ["auto-2", "auto-1"]
    third = list_automations(store2, limit=2, cursor=second.next_cursor)
    assert [(s["automation_id"], s["status"]) for s in third.items] == [("auto-0", "archived")]
    assert third.next_cursor is None
    with pytest.raises(InvalidCursor):
        list_automations(store2, cursor="garbage")


def test_sqlite_backfills_rows_written_before_the_columns(tmp_path) -> None:
    db_path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, status TEXT NOT NULL, "
        "wait_reason TEXT, wait_until TEXT, parent_run_id TEXT, actor_id TEXT, session_id TEXT, "
        "created_at TEXT, updated_at TEXT, run_json TEXT NOT NULL, paused INTEGER, run_lifecycle_json TEXT);"
    )
    doc = {"run_id": "occ-old", "workflow_id": "wf", "status": "completed", "current_node": "n",
           "vars": {"_meta": _occ(4, kind="occurrence")}, "session_id": "s", "parent_run_id": AUTO,
           "created_at": "2026-09-27T09:00:00+00:00", "updated_at": "2026-09-27T09:00:00+00:00"}
    conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?);",
                 ("occ-old", "wf", "completed", None, None, AUTO, None, "s", doc["created_at"], doc["updated_at"],
                  json.dumps(doc), 0, "null"))
    conn.commit()
    conn.close()

    store = SqliteRunStore(SqliteDatabase(db_path))
    raw = sqlite3.connect(db_path).execute(
        "SELECT automation_id, role, occurrence_index, session_kind FROM runs WHERE run_id='occ-old'").fetchone()
    assert raw == (AUTO, "occurrence", 4, "occurrence")
    assert [r["run_id"] for r in store.list_run_index(automation_id=AUTO, role="occurrence")] == ["occ-old"]


def test_sqlite_backfill_never_overwrites_a_concurrent_save(tmp_path, monkeypatch) -> None:
    """A row selected as unbackfilled and saved by another writer before the
    backfill's UPDATE keeps the saved values (the UPDATE is guarded)."""
    import abstractruntime.storage.sqlite as sqlite_mod

    db_path = tmp_path / "runs.sqlite"
    store = SqliteRunStore(SqliteDatabase(db_path))
    _save(store, "occ-1", session_id="s", parent=AUTO, meta=_occ(1), created_at="2026-09-27T10:00:00+00:00")
    raw = sqlite3.connect(db_path)
    raw.execute("UPDATE runs SET automation_id=NULL, role=NULL, occurrence_index=NULL, session_kind=NULL;")
    raw.commit()

    real = sqlite_mod.automation_index_fields

    def racing(vars_obj, *, run_id=None):
        # Another writer saves the run between the backfill's SELECT and UPDATE.
        other = sqlite3.connect(db_path)
        other.execute("UPDATE runs SET session_kind='discussion', role='discussion' WHERE run_id='occ-1';")
        other.commit()
        other.close()
        return real(vars_obj, run_id=run_id)

    monkeypatch.setattr(sqlite_mod, "automation_index_fields", racing)
    SqliteRunStore(SqliteDatabase(db_path))  # reopen: the backfill runs
    assert raw.execute("SELECT session_kind, role FROM runs WHERE run_id='occ-1'").fetchone() == ("discussion", "discussion")


def test_json_sidecar_from_before_the_columns_is_reread(tmp_path) -> None:
    store = make_store("json", tmp_path)
    _save(store, "occ-1", session_id="s", parent=AUTO, meta=_occ(1), created_at="2026-09-27T10:00:00+00:00")
    store.list_run_index()
    store._persist_scan_memo()
    sidecar = tmp_path / "runs" / ".runs_scan_cache.json"
    data = json.loads(sidecar.read_text())
    for _rid, (_token, fields) in data["entries"].items():
        for key in ("automation_id", "role", "occurrence_index", "session_kind"):
            fields.pop(key, None)
    data["version"] = 1  # a sidecar written by the previous release
    sidecar.write_text(json.dumps(data))

    fresh = JsonFileRunStore(tmp_path / "runs")
    (row,) = fresh.list_run_index()
    assert (row["role"], row["automation_id"], row["occurrence_index"]) == ("occurrence", AUTO, 1)


def test_next_fire_at_during_retry_backoff_is_the_retry_deadline(tmp_path) -> None:
    store = make_store("sqlite", tmp_path)
    _save(store, AUTO, session_id="s", created_at="2026-09-27T10:00:00+00:00", status=RunStatus.WAITING,
          meta={"automation": {"title": "t"}},
          runtime={"automation": {"next_index": 2, "pending_occurrence": {"phase": "backoff", "index": 1}}},
          waiting=WaitState(reason=WaitReason.EVENT, wait_key=f"automation:{AUTO}:wake", until="2026-09-27T10:00:30+00:00"))
    (summary,) = list_automations(store).items
    assert summary["next_fire_at"] == summary["retry_at"] == "2026-09-27T10:00:30+00:00"
    # A wait on any other key (a running occurrence) is not a fire time.
    run = store.load(AUTO)
    run.waiting = WaitState(reason=WaitReason.SUBWORKFLOW, wait_key="subworkflow:occ-1")
    store.save(run)
    (summary,) = list_automations(store).items
    assert summary["next_fire_at"] is None


def test_sqlite_latest_occurrence_is_one_index_seek(tmp_path) -> None:
    store = make_store("sqlite", tmp_path)
    for i in range(1, 41):
        _save(store, f"occ-{i}", session_id="s", parent=AUTO, meta=_occ(i), created_at=f"2026-09-27T10:{i:02d}:00+00:00")
    assert latest_occurrence(store, AUTO)["run_id"] == "occ-40"
    conn = sqlite3.connect(tmp_path / "runs.sqlite")
    plan = " ".join(str(r[-1]) for r in conn.execute(
        f"EXPLAIN QUERY PLAN SELECT run_id FROM runs {SqliteRunStore._LATEST_OCCURRENCE_WHERE} "
        f"ORDER BY {SqliteRunStore._LATEST_OCCURRENCE_ORDER} LIMIT 1", (AUTO,)))
    assert "idx_runs_automation" in plan and "SCAN runs" not in plan, plan
    # The call path issues that one indexed statement (no scan of every occurrence row).
    fetched: list = []
    raw = store._db.connection()
    raw.set_trace_callback(fetched.append)
    try:
        latest_occurrence(store, AUTO)
    finally:
        raw.set_trace_callback(None)
    (query,) = [q for q in fetched if "FROM runs" in q]  # exactly one statement
    assert SqliteRunStore._LATEST_OCCURRENCE_ORDER in query and "run_json" not in query


def test_json_latest_occurrence_parses_no_run_file_when_warm(tmp_path, monkeypatch) -> None:
    store = make_store("json", tmp_path)
    for i in range(1, 21):
        _save(store, f"occ-{i}", session_id="s", parent=AUTO, meta=_occ(i), created_at=f"2026-09-27T10:{i:02d}:00+00:00")
    store.list_run_index()  # warm the scan memo
    loads: list = []
    real = store._load_from_path
    monkeypatch.setattr(store, "_load_from_path", lambda p: loads.append(p) or real(p))
    assert latest_occurrence(store, AUTO)["run_id"] == "occ-20"
    assert loads == []


def test_latest_occurrence_requires_the_store_primitive() -> None:
    class NoPrimitive:
        def list_run_index(self, **kw):
            return []

    with pytest.raises(AttributeError):
        latest_occurrence(NoPrimitive(), AUTO)
