from __future__ import annotations

from pathlib import Path

from abstractruntime.core.models import StepRecord, StepStatus
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore


def test_sqlite_ledger_store_append_list_count_and_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "gateway.sqlite3"
    db = SqliteDatabase(db_path)
    store = SqliteLedgerStore(db)

    run_id = "run_1"
    store.append(StepRecord(run_id=run_id, step_id="s1", node_id="n1", status=StepStatus.STARTED))
    store.append(StepRecord(run_id=run_id, step_id="s2", node_id="n2", status=StepStatus.COMPLETED))

    assert store.count(run_id) == 2
    items = store.list(run_id)
    assert [x.get("step_id") for x in items] == ["s1", "s2"]

    page1, cur1 = store.list_after(run_id=run_id, after=0, limit=1)
    assert [x.get("step_id") for x in page1] == ["s1"]
    assert cur1 == 1

    page2, cur2 = store.list_after(run_id=run_id, after=cur1, limit=10)
    assert [x.get("step_id") for x in page2] == ["s2"]
    assert cur2 == 2

    # Restart: re-open and verify count/list still work.
    db2 = SqliteDatabase(db_path)
    store2 = SqliteLedgerStore(db2)
    assert store2.count(run_id) == 2
    assert [x.get("step_id") for x in store2.list(run_id)] == ["s1", "s2"]

