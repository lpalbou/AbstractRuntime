"""`LedgerStore.get_record(run_id, seq)`: the ledger row address (ADR 0011).

A step is a STARTED row and a terminal row sharing one `step_id`; the terminal
row is slimmed (`$slim` markers point at bytes the STARTED row holds) and,
under `OffloadingLedgerStore`, oversized leaves become `{"$artifact": id}`
refs. `get_record` must hand back the terminal row READY TO READ: markers
resolved against the STARTED row of the same step, offloaded leaves
rehydrated. `seq` is added on read and never persisted (the hash chain
excludes it).

Mutants this file kills (each checked by hand when it was written):
- `get_record` returns the slimmed row unresolved -> the payload assertions fail;
- `OffloadingLedgerStore` resolves markers before rehydrating its own refs ->
  the sha check refuses the marker and the offloading case fails;
- `compute_record_hash` hashes `seq` -> the chain verification cases fail;
- JSONL `append` returns a cached count that ignores another writer -> the
  two-instances case fails.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan, StepRecord
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonlLedgerStore
from abstractruntime.storage.ledger_chain import HashChainedLedgerStore, verify_ledger_chain
from abstractruntime.storage.ledger_slim import SLIM_FIELD_THRESHOLD_BYTES, contains_slim_marker, is_slim_marker
from abstractruntime.storage.offloading import OffloadingLedgerStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

pytestmark = pytest.mark.basic

BIG = "verbatim " * (SLIM_FIELD_THRESHOLD_BYTES // 4)
BIG_CALLS = [{"name": "write_file", "arguments": {"path": "a.txt", "content": BIG}}]


def _backend(kind: str, tmp_path) -> Any:
    if kind == "sqlite":
        return SqliteLedgerStore(SqliteDatabase(tmp_path / "ledger.sqlite3"))
    if kind == "jsonl":
        return JsonlLedgerStore(tmp_path / "ledger")
    if kind == "memory":
        return InMemoryLedgerStore()
    if kind == "offloading-sqlite":
        return OffloadingLedgerStore(
            SqliteLedgerStore(SqliteDatabase(tmp_path / "ledger.sqlite3")),
            artifact_store=FileArtifactStore(tmp_path / "artifacts"),
            max_inline_bytes=256,
        )
    if kind == "offloading-jsonl":
        return OffloadingLedgerStore(
            JsonlLedgerStore(tmp_path / "ledger"),
            artifact_store=FileArtifactStore(tmp_path / "artifacts"),
            max_inline_bytes=256,
        )
    raise AssertionError(kind)


def _run_one_tool_step(ledger: Any) -> str:
    """A real Runtime executes one TOOL_CALLS effect whose payload is far above
    the slimming threshold, so its terminal row is slimmed."""

    def node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("out") is not None:
            return StepPlan(node_id="n1", complete_output={"ok": True})
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": BIG_CALLS, "mode": "x"}, result_key="out"),
            next_node="n1",
        )

    def handler(run: RunState, effect: Effect, default_next_node: Optional[str] = None) -> EffectOutcome:
        return EffectOutcome.completed({"results": [{"name": "write_file", "success": True}]})

    wf = WorkflowSpec(workflow_id="wf-get-record", entry_node="n1", nodes={"n1": node})
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=ledger, effect_handlers={EffectType.TOOL_CALLS: handler})
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)
    return run_id


def _status(record: Dict[str, Any]) -> str:
    status = record.get("status")
    return str(getattr(status, "value", status))  # in-memory rows keep the enum


def _tool_rows(records: list) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for r in records:
        if (r.get("effect") or {}).get("type") == "tool_calls":
            rows[_status(r)] = r
    return rows


@pytest.mark.parametrize("kind", ["sqlite", "jsonl", "memory", "offloading-sqlite", "offloading-jsonl"])
def test_get_record_returns_the_terminal_row_with_its_payload_rebuilt(kind: str, tmp_path) -> None:
    ledger = _backend(kind, tmp_path)
    run_id = _run_one_tool_step(ledger)

    listed = ledger.list(run_id)
    assert [r["seq"] for r in listed] == list(range(1, len(listed) + 1)), "list() records carry their row address"
    rows = _tool_rows(listed)
    started, completed = rows["started"], rows["completed"]
    assert started["step_id"] == completed["step_id"]
    assert is_slim_marker(completed["effect"]["payload"]["tool_calls"]), "precondition: the terminal row is slimmed"

    record = ledger.get_record(run_id, completed["seq"])
    assert record is not None
    assert record["seq"] == completed["seq"]
    assert _status(record) == "completed"
    assert record["step_id"] == completed["step_id"]
    calls = record["effect"]["payload"]["tool_calls"]
    assert not is_slim_marker(calls), "the slimmed payload comes back"
    # The runtime stamps call ids on the calls: the STARTED row is the byte truth.
    assert [(c["name"], c["arguments"]) for c in calls] == [(c["name"], c["arguments"]) for c in BIG_CALLS]
    assert record["effect"]["payload"]["mode"] == "x"
    assert record["result"] == {"results": [{"name": "write_file", "success": True}]}
    assert not contains_slim_marker(record)

    started_record = ledger.get_record(run_id, started["seq"])
    assert started_record is not None and _status(started_record) == "started"
    assert started_record["effect"]["payload"]["tool_calls"] == calls, "terminal payload == STARTED payload"
    if not kind.startswith("offloading"):
        assert calls == started["effect"]["payload"]["tool_calls"]

    assert ledger.get_record(run_id, len(listed) + 1) is None
    assert ledger.get_record(run_id, 0) is None
    assert ledger.get_record(run_id, True) is None
    assert ledger.get_record("no-such-run", 1) is None


def test_offloading_rehydrates_before_it_resolves(tmp_path) -> None:
    """Under the offloader the STARTED row holds `$artifact` refs, so the marker's
    sha only verifies against the REHYDRATED bytes: the inner store alone keeps
    the marker (loud), the decorator rebuilds the payload."""
    ledger = _backend("offloading-sqlite", tmp_path)
    run_id = _run_one_tool_step(ledger)
    completed = _tool_rows(ledger.list(run_id))["completed"]

    inner_view = ledger.inner.get_record(run_id, completed["seq"])
    assert is_slim_marker(inner_view["effect"]["payload"]["tool_calls"]), "unverifiable marker stays in place"
    calls = ledger.get_record(run_id, completed["seq"])["effect"]["payload"]["tool_calls"]
    assert not is_slim_marker(calls)
    assert calls[0]["arguments"]["content"] == BIG


@pytest.mark.parametrize("kind", ["sqlite", "jsonl", "memory"])
def test_append_returns_the_seq_it_wrote(kind: str, tmp_path) -> None:
    ledger = _backend(kind, tmp_path)
    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    seqs = []
    for i in range(3):
        rec = StepRecord.start(run=run, node_id=f"n{i}", effect=None, idempotency_key=f"k{i}")
        seqs.append(ledger.append(rec))
    assert seqs == [1, 2, 3]
    assert [r["seq"] for r in ledger.list(run.run_id)] == [1, 2, 3]
    for s in seqs:
        assert ledger.get_record(run.run_id, s)["node_id"] == f"n{s - 1}"


def test_jsonl_seq_stays_true_when_another_writer_appends(tmp_path) -> None:
    a = JsonlLedgerStore(tmp_path)
    b = JsonlLedgerStore(tmp_path)
    run = RunState.new(workflow_id="wf", entry_node="n", vars={})

    def rec(node: str) -> StepRecord:
        return StepRecord.start(run=run, node_id=node, effect=None, idempotency_key=node)

    assert a.append(rec("a1")) == 1
    assert b.append(rec("b1")) == 2
    assert a.append(rec("a2")) == 3, "a's cache must notice b's append"
    assert [r["node_id"] for r in a.list(run.run_id)] == ["a1", "b1", "a2"]
    assert a.get_record(run.run_id, 3)["node_id"] == "a2"
    # A fresh instance (cold process) recounts from the file.
    assert JsonlLedgerStore(tmp_path).append(rec("c1")) == 4
    raw = (tmp_path / f"ledger_{run.run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    # The seq is persisted on each line (ADR 0010, the durable cursor): the row address survives a rewrite.
    assert [json.loads(line)["seq"] for line in raw] == [1, 2, 3, 4]


@pytest.mark.parametrize("kind", ["sqlite", "jsonl"])
def test_hash_chain_verifies_records_that_carry_seq(kind: str, tmp_path) -> None:
    chained = HashChainedLedgerStore(_backend(kind, tmp_path))
    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    seqs = [
        chained.append(StepRecord.start(run=run, node_id=f"n{i}", effect=None, idempotency_key=f"k{i}"))
        for i in range(3)
    ]
    assert seqs == [1, 2, 3]
    records = chained.list(run.run_id)
    assert all("seq" in r for r in records)
    report = verify_ledger_chain(records)
    assert report["ok"], report
    assert chained.get_record(run.run_id, 2)["record_hash"] == records[1]["record_hash"]


def test_sqlite_seq_is_not_persisted_inside_record_json(tmp_path) -> None:
    db = SqliteDatabase(tmp_path / "ledger.sqlite3")
    ledger = SqliteLedgerStore(db)
    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    ledger.append(StepRecord.start(run=run, node_id="n", effect=None, idempotency_key="k"))
    row = db.connection().execute("SELECT record_json FROM ledger WHERE run_id = ?;", (run.run_id,)).fetchone()
    assert "seq" not in json.loads(row["record_json"])
