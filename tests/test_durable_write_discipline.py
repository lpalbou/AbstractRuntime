"""Durable-write discipline (backlog 0067, operator-signed 2026-07-13).

The run/ledger hot path serialized with `dataclasses.asdict` (a full deep
copy of the vars tree per save/append) and wrote run files with `indent=2`
(+72% serializer CPU, ~35% larger files — 2026-07-13 performance adversary,
~5 saves per agent cycle). The fix serializes BY REFERENCE with compact
separators (storage/serialize.py) under the documented single-writer
ownership contract.

Pins here:
- round-trip FIDELITY: what the old path persisted, the new path persists
  (values, not whitespace) for both run stores and both ledger stores;
- run files are compact (no indent) and single-line;
- `runstate_to_dict` passes `vars` BY REFERENCE (the perf contract: no deep
  copy on the hot path);
- nested dataclasses inside vars/results still serialize exactly as
  `asdict` produced them (lazy `default=` hook — behavior, not crash);
- the hash chain still verifies over both backends (canonical hashing is
  whitespace-independent and hook-aware).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from typing import Any, Dict

from abstractruntime.core.models import RunState, RunStatus, StepRecord, WaitReason, WaitState
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.storage.ledger_chain import HashChainedLedgerStore, verify_ledger_chain
from abstractruntime.storage.serialize import (
    dumps_compact,
    runstate_to_dict,
    steprecord_to_dict,
)
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore


def _rich_run() -> RunState:
    run = RunState.new(workflow_id="wf-discipline", entry_node="N1", session_id="s-1")
    run.vars = {
        "_runtime": {"effect_seq": 3, "inbox": []},
        "context": {"messages": [{"role": "user", "content": "héllo — unicode"}]},
        "numbers": [1, 2.5, None, True],
        "nested": {"deep": {"deeper": ["a", "b"]}},
    }
    run.waiting = WaitState(reason=WaitReason.UNTIL, until="2026-07-14T00:00:00+00:00")
    return run


def test_runstate_round_trip_fidelity_json_files(tmp_path) -> None:
    store = JsonFileRunStore(tmp_path)
    run = _rich_run()
    store.save(run)

    # Bypass the aliasing cache: a FRESH store re-reads the disk bytes.
    loaded = JsonFileRunStore(tmp_path).load(run.run_id)
    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.status == run.status
    assert loaded.vars == run.vars
    assert loaded.waiting is not None
    assert loaded.waiting.reason == WaitReason.UNTIL
    assert loaded.waiting.until == run.waiting.until
    assert loaded.session_id == "s-1"


def test_run_file_is_compact_single_line(tmp_path) -> None:
    store = JsonFileRunStore(tmp_path)
    run = _rich_run()
    store.save(run)
    raw = (tmp_path / f"run_{run.run_id}.json").read_text(encoding="utf-8")
    assert "\n" not in raw.strip(), "compact writes are single-line"
    # Compact separators, pinned on a KNOWN key (2026-07-14 audit: a broad
    # '\": ' scan would false-fail on vars strings that legitimately
    # contain that sequence — assert the writer, not the fixture).
    assert '"run_id":"' in raw
    # And it is still plain JSON carrying the exact values.
    data = json.loads(raw)
    assert data["vars"] == run.vars


def test_runstate_round_trip_fidelity_sqlite(tmp_path) -> None:
    db = SqliteDatabase(tmp_path / "runs.sqlite3")
    store = SqliteRunStore(db)
    run = _rich_run()
    store.save(run)
    loaded = store.load(run.run_id)
    assert loaded is not None
    assert loaded.vars == run.vars
    assert loaded.status == run.status
    assert loaded.waiting is not None and loaded.waiting.until == run.waiting.until


def test_runstate_to_dict_passes_vars_by_reference() -> None:
    """The perf contract: no deep copy of the vars tree on the hot path."""
    run = _rich_run()
    d = runstate_to_dict(run)
    assert d["vars"] is run.vars
    assert d["output"] is run.output  # None or by-reference, never copied
    # waiting IS converted (tiny nested dataclass).
    assert isinstance(d["waiting"], dict)
    # Field-complete: nothing silently dropped as RunState evolves.
    assert set(d.keys()) == {f.name for f in dataclasses.fields(run)}


def test_ledger_record_round_trip_matches_asdict(tmp_path) -> None:
    """The durable bytes parse back to exactly what asdict() persisted."""
    run = _rich_run()
    rec = StepRecord.start(
        run=run,
        node_id="N1",
        effect=None,
        idempotency_key="k-parity",
    ).finish_success({"content": "résultat", "items": [1, 2]})

    jsonl = JsonlLedgerStore(tmp_path)
    jsonl.append(rec)
    listed = jsonl.list(run.run_id)
    assert len(listed) == 1
    assert listed[0] == json.loads(json.dumps(asdict(rec), ensure_ascii=False))

    db = SqliteDatabase(tmp_path / "ledger.sqlite3")
    sq = SqliteLedgerStore(db)
    sq.append(rec)
    listed_sq = sq.list(run.run_id)
    assert len(listed_sq) == 1
    assert listed_sq[0] == listed[0]


def test_nested_dataclass_in_result_serializes_like_asdict(tmp_path) -> None:
    """`asdict` used to convert dataclasses nested inside results; the lazy
    `default=` hook must preserve that behavior instead of crashing."""

    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    run = _rich_run()
    rec = StepRecord.start(run=run, node_id="N1", effect=None).finish_success(
        {"point": Point(1, 2)}
    )
    store = JsonlLedgerStore(tmp_path)
    store.append(rec)
    listed = store.list(run.run_id)
    assert listed[0]["result"] == {"point": {"x": 1, "y": 2}}

    run2 = _rich_run()
    run2.vars["marker"] = Point(3, 4)
    fs = JsonFileRunStore(tmp_path)
    fs.save(run2)
    loaded = JsonFileRunStore(tmp_path).load(run2.run_id)
    assert loaded is not None
    assert loaded.vars["marker"] == {"x": 3, "y": 4}


def test_hash_chain_verifies_over_both_backends(tmp_path) -> None:
    run = RunState.new(workflow_id="wf-chain", entry_node="n1")

    chained_jsonl = HashChainedLedgerStore(JsonlLedgerStore(tmp_path))
    chained_sqlite = HashChainedLedgerStore(
        SqliteLedgerStore(SqliteDatabase(tmp_path / "chain.sqlite3"))
    )
    for store in (chained_jsonl, chained_sqlite):
        for i in range(3):
            rec = StepRecord.start(run=run, node_id=f"n{i}", effect=None).finish_success(
                {"i": i, "text": "café"}
            )
            store.append(rec)
        report = verify_ledger_chain(store.list(run.run_id))
        assert report["ok"] is True, report
        assert report["count"] == 3


def test_mixed_version_ledger_chain_verifies(tmp_path) -> None:
    """Old-writer records (spacey separators, asdict-hashed at append time)
    continued by the NEW writer must verify as ONE chain — the never-purge
    diary book's exact upgrade path (2026-07-14 durability audit P1: this
    was proven in a throwaway script and then discarded; the repo is
    growing line/byte-oriented fast paths, which is precisely the future
    this pin guards)."""
    from abstractruntime.storage.ledger_chain import compute_record_hash

    run = RunState.new(workflow_id="wf-chain-mixed", entry_node="n1")
    prev = None
    with (tmp_path / f"ledger_{run.run_id}.jsonl").open("a", encoding="utf-8") as f:
        for i in range(2):  # the pre-0067 writer, verbatim
            rec = StepRecord.start(run=run, node_id=f"old{i}", effect=None).finish_success(
                {"i": i, "text": "café — unicode"}
            )
            rec.prev_hash = prev
            rec.record_hash = compute_record_hash(record=asdict(rec), prev_hash=prev)
            prev = rec.record_hash
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    chained = HashChainedLedgerStore(JsonlLedgerStore(tmp_path))
    for i in range(2):  # the 0067 writer continues the same chain
        chained.append(
            StepRecord.start(
                run=run, node_id=f"new{i}", effect=None, idempotency_key=f"k{i}"
            ).finish_success({"i": i})
        )
    report = verify_ledger_chain(chained.list(run.run_id))
    assert report["ok"] is True and report["count"] == 4, report

    # And the cold tail read answers from an OLD-format completed line.
    with (tmp_path / f"ledger_{run.run_id}.jsonl").open("a", encoding="utf-8") as f:
        old = StepRecord.start(
            run=run, node_id="oldk", effect=None, idempotency_key="k-old"
        ).finish_success({"v": "old-format hit"})
        f.write(json.dumps(asdict(old), ensure_ascii=False) + "\n")
    assert JsonlLedgerStore(tmp_path).find_completed_result(run.run_id, "k-old") == {
        "v": "old-format hit"
    }


def test_mixed_version_sqlite_chain_verifies(tmp_path) -> None:
    """SQLite twin: old-format raw rows continued via append_chained."""
    import sqlite3

    from abstractruntime.storage.ledger_chain import compute_record_hash

    run = RunState.new(workflow_id="wf-chain-sq", entry_node="n1")
    path = tmp_path / "mixed.sqlite3"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE ledger (run_id TEXT NOT NULL, seq INTEGER NOT NULL, "
        "record_json TEXT NOT NULL, PRIMARY KEY (run_id, seq));"
    )
    conn.execute("CREATE TABLE ledger_heads (run_id TEXT PRIMARY KEY, last_seq INTEGER NOT NULL);")
    prev = None
    for i in range(2):
        rec = StepRecord.start(run=run, node_id=f"old{i}", effect=None).finish_success({"i": i})
        rec.prev_hash = prev
        rec.record_hash = compute_record_hash(record=asdict(rec), prev_hash=prev)
        prev = rec.record_hash
        conn.execute(
            "INSERT INTO ledger (run_id, seq, record_json) VALUES (?,?,?)",
            (run.run_id, i + 1, json.dumps(asdict(rec), ensure_ascii=False)),
        )
    conn.execute("INSERT INTO ledger_heads (run_id, last_seq) VALUES (?,?)", (run.run_id, 2))
    conn.commit()
    conn.close()

    chained = HashChainedLedgerStore(SqliteLedgerStore(SqliteDatabase(path)))
    for i in range(2):
        chained.append(
            StepRecord.start(run=run, node_id=f"new{i}", effect=None).finish_success({"i": i})
        )
    report = verify_ledger_chain(chained.list(run.run_id))
    assert report["ok"] is True and report["count"] == 4, report


def test_old_indented_run_file_loads_and_resaves(tmp_path) -> None:
    """A pre-0067 run file (asdict + indent=2) must load through the new
    reader, survive a mutate + re-save by the new writer, and re-load with
    value fidelity — the mid-flight upgrade path for run checkpoints."""
    run = _rich_run()
    p = tmp_path / f"run_{run.run_id}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(run), f, ensure_ascii=False, indent=2)  # the old writer, verbatim

    store = JsonFileRunStore(tmp_path)
    loaded = store.load(run.run_id)
    assert loaded is not None
    assert loaded.vars == run.vars

    loaded.vars["upgraded"] = True
    store.save(loaded)
    reloaded = JsonFileRunStore(tmp_path).load(run.run_id)
    assert reloaded is not None
    assert reloaded.vars["upgraded"] is True
    assert reloaded.vars["context"] == run.vars["context"]


def test_dumps_compact_is_whitespace_free_and_unicode_preserving() -> None:
    text = dumps_compact({"a": [1, 2], "é": "ü"})
    assert text == '{"a":[1,2],"é":"ü"}'


def test_steprecord_to_dict_is_field_complete() -> None:
    run = RunState.new(workflow_id="wf", entry_node="n1")
    rec = StepRecord.start(run=run, node_id="n1", effect=None)
    d = steprecord_to_dict(rec)
    assert set(d.keys()) == {f.name for f in dataclasses.fields(rec)}
