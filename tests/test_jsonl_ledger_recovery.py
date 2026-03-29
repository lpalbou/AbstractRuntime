import json
import logging

from abstractruntime.storage.json_files import JsonlLedgerStore


def test_jsonl_ledger_recovers_concatenated_records(tmp_path, caplog):
    store = JsonlLedgerStore(tmp_path)
    run_id = "r1"
    ledger_path = tmp_path / f"ledger_{run_id}.jsonl"

    rec1 = {"run_id": run_id, "node_id": "n1", "status": "started"}
    rec2 = {"run_id": run_id, "node_id": "n1", "status": "completed"}
    ledger_path.write_text(json.dumps(rec1) + json.dumps(rec2) + "\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        records = store.list(run_id)

    assert rec1 in records
    assert rec2 in records
    assert any("#FALLBACK" in r.message for r in caplog.records)
