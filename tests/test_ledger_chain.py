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

