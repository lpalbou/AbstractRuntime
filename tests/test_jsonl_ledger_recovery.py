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

    # Records written without a seq are numbered by position (ADR-0010, the durable cursor).
    assert records == [{**rec1, "seq": 1}, {**rec2, "seq": 2}]
    assert any("#FALLBACK" in r.message for r in caplog.records)

    # list_after reads the recovered line the same way (I4: recovery preserved, observable).
    after, last = store.list_after(run_id=run_id, after=1)
    assert after == [{**rec2, "seq": 2}] and last == 2
