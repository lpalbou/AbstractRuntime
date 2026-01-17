from __future__ import annotations

import tempfile
from pathlib import Path

from abstractruntime.storage.commands import (
    CommandRecord,
    InMemoryCommandCursorStore,
    InMemoryCommandStore,
    JsonFileCommandCursorStore,
    JsonlCommandStore,
)
from abstractruntime.storage.sqlite import SqliteCommandCursorStore, SqliteCommandStore, SqliteDatabase


def test_in_memory_command_store_idempotency_and_cursor() -> None:
    store = InMemoryCommandStore()

    r1 = store.append(CommandRecord(command_id="c1", run_id="run_1", type="pause", payload={}, ts="t", seq=0))
    assert r1.accepted is True
    assert r1.duplicate is False
    assert r1.seq == 1

    r2 = store.append(CommandRecord(command_id="c1", run_id="run_1", type="pause", payload={}, ts="t", seq=0))
    assert r2.accepted is False
    assert r2.duplicate is True
    assert r2.seq == 1

    items, cur = store.list_after(after=0, limit=10)
    assert [x.command_id for x in items] == ["c1"]
    assert cur == 1

    items2, cur2 = store.list_after(after=1, limit=10)
    assert items2 == []
    assert cur2 == 1


def test_jsonl_command_store_idempotency_and_restart_cursor() -> None:
    with tempfile.TemporaryDirectory(prefix="abstractruntime-cmdstore-") as td:
        base = Path(td)
        store = JsonlCommandStore(base)

        r1 = store.append(CommandRecord(command_id="c1", run_id="run_1", type="pause", payload={}, ts="t", seq=0))
        r2 = store.append(CommandRecord(command_id="c2", run_id="run_1", type="cancel", payload={}, ts="t", seq=0))
        assert r1.seq == 1
        assert r2.seq == 2

        r1b = store.append(CommandRecord(command_id="c1", run_id="run_1", type="pause", payload={}, ts="t", seq=0))
        assert r1b.duplicate is True
        assert store.get_last_seq() == 2

        items, cur = store.list_after(after=0, limit=10)
        assert [x.command_id for x in items] == ["c1", "c2"]
        assert cur == 2

        # Restart: re-open from disk and confirm idempotency holds.
        store2 = JsonlCommandStore(base)
        r2b = store2.append(CommandRecord(command_id="c2", run_id="run_1", type="cancel", payload={}, ts="t", seq=0))
        assert r2b.duplicate is True
        assert store2.get_last_seq() == 2

        items2, cur2 = store2.list_after(after=1, limit=10)
        assert [x.command_id for x in items2] == ["c2"]
        assert cur2 == 2


def test_command_cursor_store_roundtrip() -> None:
    mem = InMemoryCommandCursorStore()
    assert mem.load() == 0
    mem.save(12)
    assert mem.load() == 12

    with tempfile.TemporaryDirectory(prefix="abstractruntime-cursor-") as td:
        p = Path(td) / "cursor.json"
        fs = JsonFileCommandCursorStore(p)
        assert fs.load() == 0
        fs.save(7)
        assert fs.load() == 7


def test_sqlite_command_store_idempotency_and_restart_cursor(tmp_path: Path) -> None:
    db_path = tmp_path / "gateway.sqlite3"
    db = SqliteDatabase(db_path)
    store = SqliteCommandStore(db)
    cursor_store = SqliteCommandCursorStore(db, consumer_id="t")

    r1 = store.append(CommandRecord(command_id="c1", run_id="run_1", type="pause", payload={}, ts="t", seq=0))
    assert r1.accepted is True
    assert r1.duplicate is False
    assert r1.seq >= 1

    r1b = store.append(CommandRecord(command_id="c1", run_id="run_1", type="pause", payload={}, ts="t", seq=0))
    assert r1b.accepted is False
    assert r1b.duplicate is True
    assert r1b.seq == r1.seq

    cursor_store.save(r1.seq)
    assert cursor_store.load() == r1.seq

    # Restart: re-open and verify both command idempotency and cursor persist.
    db2 = SqliteDatabase(db_path)
    store2 = SqliteCommandStore(db2)
    cursor2 = SqliteCommandCursorStore(db2, consumer_id="t")
    assert cursor2.load() == r1.seq

    r2 = store2.append(CommandRecord(command_id="c2", run_id="run_1", type="cancel", payload={}, ts="t", seq=0))
    assert r2.accepted is True
    assert store2.get_last_seq() >= r2.seq

    items, cur = store2.list_after(after=0, limit=10)
    assert [x.command_id for x in items] == ["c1", "c2"]
    assert cur == items[-1].seq

