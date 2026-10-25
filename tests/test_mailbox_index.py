"""MailboxQueryableRunStore (backlog 1046, design round-1 F3): durable fan-out
and the scheduler's wake horizon without a recency window, identical across the
in-memory, JSON-file and SQLite stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState
from abstractruntime.storage.base import MailboxQueryableRunStore, mailbox_names_of
from abstractruntime.storage.in_memory import InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore


def _store(kind: str, tmp_path: Path):
    if kind == "memory":
        return InMemoryRunStore()
    if kind == "file":
        return JsonFileRunStore(tmp_path / "runs")
    return SqliteRunStore(SqliteDatabase(tmp_path / "db.sqlite3"))


def _run(rid: str, *, status=RunStatus.WAITING, mailbox=None, until=None, reason=WaitReason.EVENT) -> RunState:
    vars_ = {} if mailbox is None else {"events_mailbox": mailbox}
    waiting = WaitState(reason=reason, wait_key=f"k:{rid}", until=until) if status == RunStatus.WAITING else None
    return RunState(run_id=rid, workflow_id="wf", status=status, current_node="n", vars=vars_, waiting=waiting)


def test_mailbox_names_of_shapes() -> None:
    assert mailbox_names_of({"events_mailbox": " a "}) == frozenset({"a"})
    assert mailbox_names_of({"events_mailbox": ["a", "", 3, "b"]}) == frozenset({"a", "b"})
    assert mailbox_names_of({}) == frozenset() and mailbox_names_of(None) == frozenset()


@pytest.mark.parametrize("kind", ["memory", "file", "sqlite"])
def test_list_mailbox_runs_follows_status_and_declaration(kind: str, tmp_path: Path) -> None:
    store = _store(kind, tmp_path)
    assert isinstance(store, MailboxQueryableRunStore)
    store.save(_run("w", mailbox="box"))
    store.save(_run("r", status=RunStatus.RUNNING, mailbox=["x", "box"]))
    store.save(_run("other", mailbox="nope"))
    store.save(_run("done", status=RunStatus.COMPLETED, mailbox="box"))
    for i in range(250):  # more than any recency window
        store.save(_run(f"p{i}", until="2099-01-01T00:00:00+00:00"))
    assert sorted(r.run_id for r in store.list_mailbox_runs(names=["box"])) == ["r", "w"]

    # Dropping the declaration or finishing removes the run; a new declarer appears.
    w = store.load("w")
    w.vars.pop("events_mailbox")
    store.save(w)
    r = store.load("r")
    r.status, r.waiting = RunStatus.COMPLETED, None
    store.save(r)
    store.save(_run("late", mailbox="box"))
    assert [x.run_id for x in store.list_mailbox_runs(names=["box"])] == ["late"]
    assert store.list_mailbox_runs(names=[]) == []


def test_file_mailbox_index_rebuilds_from_disk_and_survives_delete(tmp_path: Path) -> None:
    first = JsonFileRunStore(tmp_path / "runs")
    first.save(_run("a", mailbox="box"))
    first.save(_run("b", mailbox="box"))
    assert len(first.list_mailbox_runs(names=["box"])) == 2  # index built
    first.delete("a")
    assert [r.run_id for r in first.list_mailbox_runs(names=["box"])] == ["b"]
    fresh = JsonFileRunStore(tmp_path / "runs")  # restart: lazy rebuild from disk
    assert [r.run_id for r in fresh.list_mailbox_runs(names=["box"])] == ["b"]


@pytest.mark.parametrize("kind", ["memory", "file", "sqlite"])
def test_next_wait_deadline_is_the_minimum_over_every_waiting_run(kind: str, tmp_path: Path) -> None:
    store = _store(kind, tmp_path)
    assert store.next_wait_deadline() is None
    store.save(_run("no_deadline"))
    assert store.next_wait_deadline() is None
    for i in range(250):
        store.save(_run(f"p{i:03d}", until=f"2099-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00"))
    store.save(_run("early", until="2030-01-01T00:00:00+00:00", reason=WaitReason.UNTIL))
    assert store.next_wait_deadline() == "2030-01-01T00:00:00+00:00"
    early = store.load("early")
    early.status, early.waiting = RunStatus.COMPLETED, None
    store.save(early)
    assert store.next_wait_deadline() == "2099-01-01T00:00:00+00:00"
