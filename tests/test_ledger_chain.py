from abstractruntime.core.models import RunState, StepRecord
from abstractruntime.storage.in_memory import InMemoryLedgerStore
from abstractruntime.storage.ledger_chain import HashChainedLedgerStore, verify_ledger_chain


def test_hash_chained_ledger_store_and_verify_ok():
    inner = InMemoryLedgerStore()
    store = HashChainedLedgerStore(inner)

    run = RunState.new(workflow_id="wf", entry_node="n1")

    r1 = StepRecord.start(run=run, node_id="n1", effect=None).finish_success({"a": 1})
    r2 = StepRecord.start(run=run, node_id="n2", effect=None).finish_success({"b": 2})

    store.append(r1)
    store.append(r2)

    records = store.list(run.run_id)
    report = verify_ledger_chain(records)

    assert report["ok"] is True
    assert report["count"] == 2
    assert report["head_hash"] == records[-1]["record_hash"]


def test_verify_detects_tampering():
    inner = InMemoryLedgerStore()
    store = HashChainedLedgerStore(inner)

    run = RunState.new(workflow_id="wf", entry_node="n1")

    r1 = StepRecord.start(run=run, node_id="n1", effect=None).finish_success({"a": 1})
    r2 = StepRecord.start(run=run, node_id="n2", effect=None).finish_success({"b": 2})

    store.append(r1)
    store.append(r2)

    records = store.list(run.run_id)
    records[1]["result"] = {"b": 999}

    report = verify_ledger_chain(records)
    assert report["ok"] is False
    assert report["first_bad_index"] in (0, 1)


def test_verify_reports_missing_hashes():
    # Ledger without chain decorator
    inner = InMemoryLedgerStore()

    run = RunState.new(workflow_id="wf", entry_node="n1")
    r1 = StepRecord.start(run=run, node_id="n1", effect=None).finish_success({"a": 1})
    inner.append(r1)

    report = verify_ledger_chain(inner.list(run.run_id))
    assert report["ok"] is False
    assert any(e["type"] == "missing_record_hash" for e in report["errors"])


def test_two_handles_over_one_persisted_ledger_never_fork(tmp_path):
    """THE FORK CLASS (2026-07-10 adversarial finding): two store handles
    over ONE persisted ledger — the shape of two processes appending to one
    home — used to each cache their own head, so both linked to the same
    prev_hash and the chain forked permanently (verify: prev_hash_mismatch;
    on the never-purge diary book, unrepairable). The head must come from
    the PERSISTED tail — for SQLite inside one BEGIN IMMEDIATE transaction
    (append_chained) — never from process-local state."""
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    path = tmp_path / "book.sqlite3"
    # Two independent handles: separate SqliteDatabase objects = separate
    # connections, exactly what two processes would hold.
    a = HashChainedLedgerStore(SqliteLedgerStore(SqliteDatabase(path)))
    b = HashChainedLedgerStore(SqliteLedgerStore(SqliteDatabase(path)))

    run = RunState.new(workflow_id="wf", entry_node="n1")
    writers = (a, b, a, b, a, b)  # interleaved appends, worst case
    for i, store in enumerate(writers):
        rec = StepRecord.start(run=run, node_id=f"n{i}", effect=None).finish_success({"i": i})
        store.append(rec)

    records = a.list(run.run_id)
    report = verify_ledger_chain(records)
    assert report["ok"] is True, report["errors"]
    assert report["count"] == len(writers)


def test_fallback_path_rereads_persisted_head_across_handles():
    """Inner stores WITHOUT append_chained (in-memory here) still get the
    stale-cache fix: the head is re-read from the persisted tail on every
    append, so two decorator instances over one inner store chain
    correctly instead of forking."""
    inner = InMemoryLedgerStore()
    a = HashChainedLedgerStore(inner)
    b = HashChainedLedgerStore(inner)

    run = RunState.new(workflow_id="wf", entry_node="n1")
    for i, store in enumerate((a, b, a, b)):
        rec = StepRecord.start(run=run, node_id=f"n{i}", effect=None).finish_success({"i": i})
        store.append(rec)

    report = verify_ledger_chain(inner.list(run.run_id))
    assert report["ok"] is True, report["errors"]
    assert report["count"] == 4

