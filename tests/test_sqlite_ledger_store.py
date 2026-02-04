from __future__ import annotations

import json
import sqlite3
import threading
import multiprocessing
from pathlib import Path

from abstractruntime.core.models import StepRecord, StepStatus
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore


def _mp_sqlite_ledger_store_append_worker(
    *,
    db_path: str,
    run_id: str,
    worker_id: int,
    start_evt: multiprocessing.synchronize.Event,
    errors_q: multiprocessing.queues.Queue[str],
    count: int,
) -> None:
    try:
        if not start_evt.wait(timeout=5.0):
            raise RuntimeError("start timeout")
        db = SqliteDatabase(Path(db_path))
        store = SqliteLedgerStore(db)
        for j in range(int(count)):
            store.append(StepRecord(run_id=run_id, step_id=f"{worker_id}:{j}", node_id="n", status=StepStatus.COMPLETED))
    except BaseException as e:
        try:
            errors_q.put_nowait(repr(e))
        except Exception:
            pass


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


def test_sqlite_ledger_store_append_is_multiprocess_concurrency_safe(tmp_path: Path) -> None:
    db_path = tmp_path / "gateway.sqlite3"
    run_id = "run_1"

    # Use a multiprocess test to approximate real gateway deployments where
    # API + runner may append concurrently from different processes.
    ctx = multiprocessing.get_context("spawn")
    start_evt = ctx.Event()
    errors_q: multiprocessing.queues.Queue[str] = ctx.Queue()
    procs = [
        ctx.Process(
            target=_mp_sqlite_ledger_store_append_worker,
            kwargs={
                "db_path": str(db_path),
                "run_id": run_id,
                "worker_id": i,
                "start_evt": start_evt,
                "errors_q": errors_q,
                "count": 40,
            },
            daemon=True,
        )
        for i in range(4)
    ]
    for p in procs:
        p.start()
    start_evt.set()
    for p in procs:
        p.join(timeout=20.0)

    errors: list[str] = []
    try:
        while True:
            errors.append(errors_q.get_nowait())
    except Exception:
        pass

    for p in procs:
        if p.is_alive():
            p.terminate()
        assert p.exitcode == 0

    assert not errors
    db2 = SqliteDatabase(db_path)
    store2 = SqliteLedgerStore(db2)
    assert store2.count(run_id) == 160


def test_sqlite_ledger_store_backfills_heads_from_existing_ledger(tmp_path: Path) -> None:
    db_path = tmp_path / "gateway.sqlite3"

    # Simulate an older DB: ledger exists and contains records, but ledger_heads does not.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE ledger (
              run_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              record_json TEXT NOT NULL,
              PRIMARY KEY (run_id, seq)
            );
            """
        )
        conn.execute("INSERT INTO ledger (run_id, seq, record_json) VALUES (?, ?, ?);", ("run_1", 1, json.dumps({"step_id": "s1"})))
        conn.execute("INSERT INTO ledger (run_id, seq, record_json) VALUES (?, ?, ?);", ("run_1", 2, json.dumps({"step_id": "s2"})))
        conn.commit()
    finally:
        conn.close()

    # New store must continue from seq=2 (so next append is seq=3).
    db = SqliteDatabase(db_path)
    store = SqliteLedgerStore(db)
    store.append(StepRecord(run_id="run_1", step_id="s3", node_id="n3", status=StepStatus.COMPLETED))

    assert store.count("run_1") == 3

    row = db.connection().execute("SELECT last_seq FROM ledger_heads WHERE run_id = ?;", ("run_1",)).fetchone()
    assert row is not None
    assert int(row["last_seq"] or 0) == 3
