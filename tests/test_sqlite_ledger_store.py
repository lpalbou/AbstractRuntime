from __future__ import annotations

import threading
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


def test_sqlite_ledger_store_count_many_and_metrics_many(tmp_path: Path) -> None:
    db_path = tmp_path / "gateway.sqlite3"
    db = SqliteDatabase(db_path)
    store = SqliteLedgerStore(db)

    run_id = "run_1"
    # Started record shouldn't count toward metrics_many (completed-only).
    store.append(
        StepRecord(
            run_id=run_id,
            step_id="s0",
            node_id="n0",
            status=StepStatus.STARTED,
            effect={"type": "llm_call", "payload": {}, "result_key": "_tmp"},
        )
    )
    # Completed LLM call with usage.
    store.append(
        StepRecord(
            run_id=run_id,
            step_id="s1",
            node_id="n1",
            status=StepStatus.COMPLETED,
            effect={"type": "llm_call", "payload": {}, "result_key": "_tmp"},
            result={"usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}},
        )
    )
    # Completed tool_calls with 2 tool calls.
    store.append(
        StepRecord(
            run_id=run_id,
            step_id="s2",
            node_id="n2",
            status=StepStatus.COMPLETED,
            effect={"type": "tool_calls", "payload": {"tool_calls": [{"name": "a"}, {"name": "b"}]}, "result_key": "_tmp"},
            result={},
        )
    )

    assert store.count_many([run_id, "missing"]) == {run_id: 3}

    metrics = store.metrics_many([run_id, "missing"])
    assert metrics.get(run_id) == {"steps": 2, "llm_calls": 1, "tool_calls": 2, "tokens_total": 12}


def test_sqlite_ledger_store_append_is_concurrency_safe(tmp_path: Path) -> None:
    db_path = tmp_path / "gateway.sqlite3"
    db = SqliteDatabase(db_path)
    store = SqliteLedgerStore(db)

    run_id = "run_1"

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def _worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for j in range(50):
                store.append(StepRecord(run_id=run_id, step_id=f"{worker_id}:{j}", node_id="n", status=StepStatus.COMPLETED))
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors
    assert store.count(run_id) == 400
