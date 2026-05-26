from __future__ import annotations

from pathlib import Path

import pytest

from abstractruntime.core.models import RunState, RunStatus, StepRecord, StepStatus
from abstractruntime.storage.commands import CommandRecord, InMemoryCommandStore, JsonlCommandStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.storage.ledger_chain import HashChainedLedgerStore
from abstractruntime.storage.observable import ObservableLedgerStore
from abstractruntime.storage.offloading import OffloadingLedgerStore, OffloadingRunStore
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.sqlite import SqliteCommandStore, SqliteDatabase, SqliteLedgerStore, SqliteRunStore


def _run(run_id: str, *, parent_run_id: str | None = None) -> RunState:
    return RunState(
        run_id=run_id,
        workflow_id="wf",
        status=RunStatus.COMPLETED,
        current_node="done",
        vars={},
        parent_run_id=parent_run_id,
    )


def _record(run_id: str, step_id: str) -> StepRecord:
    return StepRecord(run_id=run_id, step_id=step_id, node_id="n", status=StepStatus.COMPLETED, result={"ok": True})


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_run_store_delete_removes_checkpoint_index_and_children(kind: str, tmp_path: Path) -> None:
    if kind == "memory":
        store = InMemoryRunStore()
    elif kind == "json":
        store = JsonFileRunStore(tmp_path / "runs")
    else:
        store = SqliteRunStore(SqliteDatabase(tmp_path / "gateway.sqlite3"))

    root = _run("root")
    child = _run("child", parent_run_id="root")
    other = _run("other")
    store.save(root)
    store.save(child)
    store.save(other)

    assert store.load("child") is not None
    assert [r.run_id for r in store.list_children(parent_run_id="root")] == ["child"]

    assert store.delete("child") is True
    assert store.delete("child") is False
    assert store.load("child") is None
    assert store.list_children(parent_run_id="root") == []
    assert {r["run_id"] for r in store.list_run_index(limit=10)} == {"root", "other"}


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_ledger_store_delete_removes_records_and_is_idempotent(kind: str, tmp_path: Path) -> None:
    if kind == "memory":
        store = InMemoryLedgerStore()
    elif kind == "json":
        store = JsonlLedgerStore(tmp_path / "ledger")
    else:
        store = SqliteLedgerStore(SqliteDatabase(tmp_path / "gateway.sqlite3"))

    store.append(_record("run_1", "s1"))
    store.append(_record("run_1", "s2"))
    store.append(_record("run_2", "s3"))

    assert len(store.list("run_1")) == 2
    assert store.delete("run_1") == 2
    assert store.delete("run_1") == 0
    assert store.list("run_1") == []
    assert len(store.list("run_2")) == 1


def test_wrapped_stores_delegate_delete_and_leave_artifacts_for_gateway_purge(tmp_path: Path) -> None:
    artifacts = InMemoryArtifactStore()
    run_store = OffloadingRunStore(JsonFileRunStore(tmp_path / "runs"), artifact_store=artifacts, max_inline_bytes=1)
    ledger_store = OffloadingLedgerStore(
        ObservableLedgerStore(HashChainedLedgerStore(JsonlLedgerStore(tmp_path / "ledger"))),
        artifact_store=artifacts,
        max_inline_bytes=1,
    )

    run = _run("run_1")
    run.vars["_temp"] = {"large": "x" * 64}
    run_store.save(run)
    ledger_store.append(_record("run_1", "s1"))

    assert artifacts.list_by_run("run_1")
    assert run_store.delete("run_1") is True
    assert ledger_store.delete("run_1") == 1
    assert artifacts.list_by_run("run_1")
    assert artifacts.delete_by_run("run_1") >= 1
    assert artifacts.list_by_run("run_1") == []


@pytest.mark.parametrize("kind", ["memory", "json", "sqlite"])
def test_command_store_delete_by_run_preserves_other_runs(kind: str, tmp_path: Path) -> None:
    if kind == "memory":
        store = InMemoryCommandStore()
    elif kind == "json":
        store = JsonlCommandStore(tmp_path / "commands")
    else:
        store = SqliteCommandStore(SqliteDatabase(tmp_path / "gateway.sqlite3"))

    store.append(CommandRecord(command_id="c1", run_id="run_1", type="pause", payload={}, ts="t"))
    store.append(CommandRecord(command_id="c2", run_id="run_1", type="resume", payload={}, ts="t"))
    store.append(CommandRecord(command_id="c3", run_id="run_2", type="cancel", payload={}, ts="t"))

    assert store.delete_by_run("run_1") == 2
    assert store.delete_by_run("run_1") == 0

    items, cursor = store.list_after(after=0, limit=10)
    assert [r.command_id for r in items] == ["c3"]
    assert cursor == items[-1].seq
