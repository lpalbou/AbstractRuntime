"""Indexed idempotency lookup (backlog 0047, operator-signed 2026-07-13).

The tick loop probes "prior COMPLETED result for this key?" before EVERY
effect step; the lookup was a full-ledger parse — O(ledger) per step,
measured at ~44ms/step on a 32MB ledger (quadratic per run). The fix:

- SQLite: `idempotency_key`/`step_status` columns + partial index; the
  probe is a point query. Pre-0047 databases are column-backfilled at open.
- JSONL: write-through cache of COMPLETED results + bounded backward tail
  read (O(tail bytes), never the file).
- Correctness envelope: keys are issuance-scoped (`_runtime.effect_seq`),
  so a genuine hit can only live at the ledger tail (crash-replay = effect
  completed, the save after it did not land). Beyond the window the runtime
  re-executes — the documented at-least-once default.

Pins here: point-query correctness + oldest-wins parity, legacy-database
migration/backfill, the version-skew guard, JSONL cache + cold tail read +
the documented window bound, decorator delegation (chained/observable/
offloading), and the runtime seam (routed stores + duck-typed fallback).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from abstractruntime.core.models import RunState, StepRecord
from abstractruntime.storage.base import IDEMPOTENCY_TAIL_WINDOW
from abstractruntime.storage.in_memory import InMemoryLedgerStore
from abstractruntime.storage.json_files import JsonlLedgerStore
from abstractruntime.storage.ledger_chain import HashChainedLedgerStore
from abstractruntime.storage.observable import ObservableLedgerStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore


def _completed(run: RunState, *, node: str, key: str, result: Dict[str, Any]) -> StepRecord:
    return StepRecord.start(
        run=run, node_id=node, effect=None, idempotency_key=key
    ).finish_success(result)


def _failed(run: RunState, *, node: str, key: str) -> StepRecord:
    return StepRecord.start(
        run=run, node_id=node, effect=None, idempotency_key=key
    ).finish_failure("boom")


# ---------------------------------------------------------------------------
# SQLite: indexed point query
# ---------------------------------------------------------------------------


def test_sqlite_point_query_finds_completed_result(tmp_path) -> None:
    store = SqliteLedgerStore(SqliteDatabase(tmp_path / "l.sqlite3"))
    run = RunState.new(workflow_id="wf", entry_node="n1")

    store.append(_failed(run, node="n1", key="k1"))
    store.append(_completed(run, node="n1", key="k1", result={"content": "first"}))
    store.append(_completed(run, node="n2", key="k2", result={"content": "other"}))

    assert store.find_completed_result(run.run_id, "k1") == {"content": "first"}
    assert store.find_completed_result(run.run_id, "k2") == {"content": "other"}
    assert store.find_completed_result(run.run_id, "absent") is None
    assert store.find_completed_result("other-run", "k1") is None


def test_sqlite_oldest_completed_wins_parity(tmp_path) -> None:
    """Legacy duplicate keys (pre-issuance-counter ledgers): the historical
    full scan returned the FIRST completed match; the point query must too."""
    store = SqliteLedgerStore(SqliteDatabase(tmp_path / "l.sqlite3"))
    run = RunState.new(workflow_id="wf", entry_node="n1")
    store.append(_completed(run, node="n1", key="dup", result={"v": "oldest"}))
    store.append(_completed(run, node="n1", key="dup", result={"v": "newest"}))
    assert store.find_completed_result(run.run_id, "dup") == {"v": "oldest"}


def test_sqlite_legacy_database_is_backfilled_at_open(tmp_path) -> None:
    """A pre-0047 database (no idempotency columns) opened by the new code
    gains the columns AND the backfill, so pre-migration records stay
    findable by the indexed query."""
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE ledger (run_id TEXT NOT NULL, seq INTEGER NOT NULL, "
        "record_json TEXT NOT NULL, PRIMARY KEY (run_id, seq));"
    )
    conn.execute("CREATE TABLE ledger_heads (run_id TEXT PRIMARY KEY, last_seq INTEGER NOT NULL);")
    legacy_record = {
        "run_id": "r-legacy",
        "step_id": "s1",
        "node_id": "n1",
        "status": "completed",
        "idempotency_key": "k-legacy",
        "result": {"content": "from the old world"},
    }
    conn.execute(
        "INSERT INTO ledger (run_id, seq, record_json) VALUES (?, ?, ?);",
        ("r-legacy", 1, json.dumps(legacy_record)),
    )
    conn.execute("INSERT INTO ledger_heads (run_id, last_seq) VALUES (?, ?);", ("r-legacy", 1))
    conn.commit()
    conn.close()

    store = SqliteLedgerStore(SqliteDatabase(path))
    assert store.find_completed_result("r-legacy", "k-legacy") == {
        "content": "from the old world"
    }
    # The backfill actually landed in the columns (not just the scan path).
    check = sqlite3.connect(str(path))
    row = check.execute(
        "SELECT idempotency_key, step_status FROM ledger WHERE run_id='r-legacy';"
    ).fetchone()
    check.close()
    assert row == ("k-legacy", "completed")


def test_sqlite_migration_race_duplicate_column_is_tolerated(tmp_path) -> None:
    """The schema-init lock is per-SqliteDatabase instance, so two PROCESSES
    opening one legacy file can both pass the column check before either
    ALTERs — the loser sees OperationalError('duplicate column') and must
    treat it as already-migrated, never crash the open. Simulated by racing
    the ALTER: a competing writer adds the column between this instance's
    PRAGMA check and its ALTER."""
    path = tmp_path / "race.sqlite3"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE ledger (run_id TEXT NOT NULL, seq INTEGER NOT NULL, "
        "record_json TEXT NOT NULL, PRIMARY KEY (run_id, seq));"
    )
    conn.execute("CREATE TABLE ledger_heads (run_id TEXT PRIMARY KEY, last_seq INTEGER NOT NULL);")
    conn.commit()
    conn.close()

    real_connect = sqlite3.connect
    raced = {"done": False}

    class _RacingConnection:
        """Delegates everything; injects the competing ALTER just before
        this instance's own ALTER runs."""

        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def execute(self, sql, *args):
            if (
                not raced["done"]
                and isinstance(sql, str)
                and sql.strip().startswith("ALTER TABLE ledger ADD COLUMN idempotency_key")
            ):
                raced["done"] = True
                rival = real_connect(str(path))
                rival.execute("ALTER TABLE ledger ADD COLUMN idempotency_key TEXT;")
                rival.commit()
                rival.close()
            return self._inner.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __setattr__(self, name, value):
            # Attribute sets (row_factory!) must reach the REAL connection.
            setattr(self._inner, name, value)

    import unittest.mock as mock

    def _wrapping_connect(*args, **kwargs):
        return _RacingConnection(real_connect(*args, **kwargs))

    with mock.patch.object(sqlite3, "connect", _wrapping_connect):
        db = SqliteDatabase(path)  # must not raise despite losing the race

    assert raced["done"] is True
    store = SqliteLedgerStore(db)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    store.append(_completed(run, node="n1", key="k1", result={"v": 1}))
    assert store.find_completed_result(run.run_id, "k1") == {"v": 1}


def test_sqlite_version_skew_rows_degrade_to_scan(tmp_path) -> None:
    """Rows appended by a pre-0047 writer AFTER migration carry NULL columns
    and are invisible to the point query — the near-empty partial index
    probe must detect them and degrade to the bounded scan, never miss a
    genuinely completed effect."""
    path = tmp_path / "skew.sqlite3"
    db = SqliteDatabase(path)
    store = SqliteLedgerStore(db)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    store.append(_completed(run, node="n1", key="k-new", result={"v": 1}))

    # Simulate an old-version writer: raw insert with NULL columns.
    skew_record = {
        "run_id": run.run_id,
        "step_id": "s-skew",
        "node_id": "n2",
        "status": "completed",
        "idempotency_key": "k-skew",
        "result": {"v": "written by old code"},
    }
    conn = db.connection()
    with conn:
        conn.execute(
            "UPDATE ledger_heads SET last_seq = last_seq + 1 WHERE run_id = ?;", (run.run_id,)
        )
        conn.execute(
            "INSERT INTO ledger (run_id, seq, record_json) VALUES (?, ?, ?);",
            (run.run_id, 2, json.dumps(skew_record)),
        )

    assert store.find_completed_result(run.run_id, "k-skew") == {"v": "written by old code"}
    # Indexed rows still answer via the point query.
    assert store.find_completed_result(run.run_id, "k-new") == {"v": 1}


def test_sqlite_probe_uses_the_covering_index(tmp_path) -> None:
    """Perf adversary N1 (2026-07-13): with a 2-column (run_id, key) partial
    index the planner preferred idx_ledger_run_seq (it satisfies ORDER BY
    seq) and row-fetched EVERY record of the run per probe — O(per-run rows)
    per step, the cliff re-grown. Pin the PLAN: the probe must search via
    idx_ledger_idem (covering 3-column shape), no run-seq scan."""
    db = SqliteDatabase(tmp_path / "plan.sqlite3")
    store = SqliteLedgerStore(db)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    store.append(_completed(run, node="n1", key="k1", result={"v": 1}))
    conn = db.connection()
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT record_json FROM ledger "
        "WHERE run_id = ? AND idempotency_key = ? AND step_status = ? "
        "ORDER BY seq ASC LIMIT 1;",
        (run.run_id, "k1", "completed"),
    ).fetchall()
    details = " | ".join(str(row["detail"]) for row in plan)
    assert "idx_ledger_idem" in details, details


def test_sqlite_two_column_index_shape_is_upgraded_at_open(tmp_path) -> None:
    """`CREATE INDEX IF NOT EXISTS` never rewrites an existing index, so a
    database carrying the first (2-column) shape must be detected by column
    mismatch and rebuilt as the covering 3-column shape at next open."""
    path = tmp_path / "upgrade.sqlite3"
    SqliteLedgerStore(SqliteDatabase(path))  # creates the 3-column shape

    conn = sqlite3.connect(str(path))
    conn.execute("DROP INDEX IF EXISTS idx_ledger_idem;")
    conn.execute(
        "CREATE INDEX idx_ledger_idem ON ledger(run_id, idempotency_key) "
        "WHERE idempotency_key IS NOT NULL;"
    )
    conn.commit()
    conn.close()

    db = SqliteDatabase(path)  # fresh instance: schema pass runs again
    check = db.connection()
    cols = [str(r["name"]) for r in check.execute("PRAGMA index_info(idx_ledger_idem);").fetchall()]
    assert cols == ["run_id", "idempotency_key", "seq"]


# ---------------------------------------------------------------------------
# JSONL: write-through cache + bounded backward tail read
# ---------------------------------------------------------------------------


def test_jsonl_write_through_cache_hit(tmp_path) -> None:
    """The CACHE serves the same-process hit — isolated by poisoning the
    tail read (replay adversary P2-e: the earlier list()-poison version was
    vacuous, the lookup path never used list(); the tail read served the
    hit even with the cache neutered)."""
    store = JsonlLedgerStore(tmp_path)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    store.append(_completed(run, node="n1", key="k1", result={"content": "cached"}))

    def _no_disk(p, n):  # noqa: ANN001 - test double
        raise AssertionError("tail read must not run on a cache hit")

    store._tail_lines = _no_disk  # type: ignore[method-assign]
    assert store.find_completed_result(run.run_id, "k1") == {"content": "cached"}


def test_jsonl_tail_read_survives_unicode_line_separators(tmp_path) -> None:
    """Replay adversary P0 (2026-07-14): JSON leaves U+2028/U+2029/U+0085
    RAW under ensure_ascii=False, and str.splitlines() splits on all three
    — a completed record carrying U+2028 in scraped/LLM text fragmented
    into unparseable pieces, the COLD crash-replay probe missed it, and the
    completed effect re-executed. The tail read must split on the writer's
    actual line discipline ('\\n' only)."""
    run = RunState.new(workflow_id="wf", entry_node="n1")
    writer = JsonlLedgerStore(tmp_path)
    result = {
        "content": "line one\u2028line two\u2029para\u0085next",
        "ok": True,
    }
    writer.append(_completed(run, node="n1", key="k-sep", result=result))
    writer.append(_completed(run, node="n2", key="k-after", result={"v": 2}))

    cold = JsonlLedgerStore(tmp_path)  # crash-replay = new process = cold cache
    found = cold.find_completed_result(run.run_id, "k-sep")
    assert found == result
    # And the record parses identically through the full reader.
    listed = cold.list(run.run_id)
    assert listed[0]["result"] == result


def test_jsonl_prefilter_handles_keys_with_json_escaped_chars(tmp_path) -> None:
    """Replay adversary P2-a: host-pluggable EffectPolicy keys are arbitrary
    strings; a key containing '"' or '\\' appears ESCAPED on disk, so the
    quoted-key prefilter would false-negative a genuine hit. Such keys must
    skip the prefilter and still answer from the cold tail read."""
    run = RunState.new(workflow_id="wf", entry_node="n1")
    writer = JsonlLedgerStore(tmp_path)
    weird_key = 'tool:{"path": "a\\b.txt"}'
    writer.append(_completed(run, node="n1", key=weird_key, result={"v": "hit"}))

    cold = JsonlLedgerStore(tmp_path)
    assert cold.find_completed_result(run.run_id, weird_key) == {"v": "hit"}


def test_jsonl_cache_never_outlives_a_deleted_ledger(tmp_path) -> None:
    """Replay adversary P2-b: a SECOND store instance over one directory
    kept serving a deleted run's results from its OWN write-through cache
    after the other instance deleted the ledger. Disk truth wins: no file,
    no answer — from ANY instance, warm cache or not."""
    run = RunState.new(workflow_id="wf", entry_node="n1")
    a = JsonlLedgerStore(tmp_path)
    b = JsonlLedgerStore(tmp_path)
    # B is the APPENDER, so B's write-through cache holds the entry.
    b.append(_completed(run, node="n1", key="k1", result={"v": "old"}))
    assert b.find_completed_result(run.run_id, "k1") == {"v": "old"}

    a.delete(run.run_id)  # A deletes; B's cache is now stale vs disk
    assert a.find_completed_result(run.run_id, "k1") is None
    assert b.find_completed_result(run.run_id, "k1") is None


def test_jsonl_cache_returns_disk_truth_not_live_objects(tmp_path) -> None:
    """Workflow reducers may mutate result dicts in place after the append;
    the cache must answer with DISK truth (JSON round-trip), never the
    live object."""
    store = JsonlLedgerStore(tmp_path)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    result: Dict[str, Any] = {"content": "original", "coords": (1, 2)}
    store.append(_completed(run, node="n1", key="k1", result=result))
    result["content"] = "MUTATED AFTER APPEND"

    found = store.find_completed_result(run.run_id, "k1")
    assert found is not None
    assert found["content"] == "original"
    assert found["coords"] == [1, 2], "JSON round-trip coerces tuples to lists (disk parity)"


def test_jsonl_cold_process_tail_read(tmp_path) -> None:
    """A fresh store instance (new process) has an empty cache and must find
    the completed record via the backward tail read."""
    run = RunState.new(workflow_id="wf", entry_node="n1")
    writer = JsonlLedgerStore(tmp_path)
    writer.append(_failed(run, node="n1", key="k1"))
    writer.append(_completed(run, node="n1", key="k1", result={"content": "durable"}))

    reader = JsonlLedgerStore(tmp_path)
    assert reader.find_completed_result(run.run_id, "k1") == {"content": "durable"}
    assert reader.find_completed_result(run.run_id, "missing") is None


def test_jsonl_tail_window_bound_is_the_documented_semantic(tmp_path) -> None:
    """Beyond IDEMPOTENCY_TAIL_WINDOW records, a cold lookup answers None —
    the documented at-least-once bound (issuance-scoped keys make genuine
    hits at that depth impossible; this pins the bound as a DECISION)."""
    run = RunState.new(workflow_id="wf", entry_node="n1")
    writer = JsonlLedgerStore(tmp_path)
    writer.append(_completed(run, node="n1", key="k-deep", result={"v": "buried"}))
    for i in range(IDEMPOTENCY_TAIL_WINDOW + 5):
        writer.append(_completed(run, node="n1", key=f"k-{i}", result={"i": i}))

    cold = JsonlLedgerStore(tmp_path)
    assert cold.find_completed_result(run.run_id, "k-deep") is None
    # Same-process lookups still answer from the write-through cache.
    assert writer.find_completed_result(run.run_id, "k-deep") == {"v": "buried"}


def test_jsonl_tail_byte_cap_stops_honestly(tmp_path) -> None:
    """The 32MB byte ceiling (perf adversary N3) exercised at a small test
    scale: keys within the cap answer; keys beyond it answer an honest
    miss; the possibly-truncated oldest line never produces a bogus hit."""
    run = RunState.new(workflow_id="wf", entry_node="n1")
    writer = JsonlLedgerStore(tmp_path)
    writer.append(_completed(run, node="n1", key="k-deep", result={"pad": "y" * 300}))
    writer.append(_completed(run, node="n2", key="k-mid", result={"pad": "z" * 300}))
    writer.append(_completed(run, node="n3", key="k-near", result={"v": "near"}))

    cold = JsonlLedgerStore(tmp_path)
    # Instance overrides: small blocks so the cap can land mid-file (the
    # default 64KB first read would swallow this whole small fixture).
    cold._TAIL_MAX_BYTES = 512
    cold._TAIL_BLOCK_BYTES = 128
    assert cold.find_completed_result(run.run_id, "k-near") == {"v": "near"}
    assert cold.find_completed_result(run.run_id, "k-deep") is None, "beyond the byte cap: honest miss"


def test_jsonl_oldest_wins_within_window(tmp_path) -> None:
    run = RunState.new(workflow_id="wf", entry_node="n1")
    writer = JsonlLedgerStore(tmp_path)
    writer.append(_completed(run, node="n1", key="dup", result={"v": "oldest"}))
    writer.append(_completed(run, node="n1", key="dup", result={"v": "newest"}))
    cold = JsonlLedgerStore(tmp_path)
    assert cold.find_completed_result(run.run_id, "dup") == {"v": "oldest"}


# ---------------------------------------------------------------------------
# Decorators + ABC default
# ---------------------------------------------------------------------------


def test_decorators_delegate_to_inner_lookup(tmp_path) -> None:
    """Delegation, pinned with a SPY on the inner store's method (2026-07-14
    audit: without it, deleting the delegation would still pass via the ABC
    default's list() scan — the exact O(n) fallback delegation avoids)."""
    inner_calls: List[str] = []

    class _SpySqlite(SqliteLedgerStore):
        def find_completed_result(self, run_id, idempotency_key):  # noqa: ANN001
            inner_calls.append(idempotency_key)
            return super().find_completed_result(run_id, idempotency_key)

    run = RunState.new(workflow_id="wf", entry_node="n1")
    sq = _SpySqlite(SqliteDatabase(tmp_path / "d.sqlite3"))
    chained = HashChainedLedgerStore(sq)
    observable = ObservableLedgerStore(chained)

    observable.append(_completed(run, node="n1", key="k1", result={"content": "chained"}))
    assert observable.find_completed_result(run.run_id, "k1") == {"content": "chained"}
    assert chained.find_completed_result(run.run_id, "k1") == {"content": "chained"}
    assert inner_calls == ["k1", "k1"], "both wrappers must route to the INNER lookup"


def test_in_memory_store_uses_abc_default(tmp_path) -> None:
    """The base default is a BOUNDED tail scan — exercised at its boundary
    (2026-07-14 audit: a single-record fixture never touched the window)."""
    store = InMemoryLedgerStore()
    run = RunState.new(workflow_id="wf", entry_node="n1")
    store.append(_completed(run, node="n1", key="k-first", result={"v": "first"}))
    for i in range(IDEMPOTENCY_TAIL_WINDOW + 5):
        store.append(_completed(run, node="n1", key=f"k-{i}", result={"i": i}))
    # Within the window: found. The first record is now beyond it: honest miss.
    assert store.find_completed_result(run.run_id, f"k-{IDEMPOTENCY_TAIL_WINDOW}") is not None
    assert store.find_completed_result(run.run_id, "k-first") is None
    assert store.find_completed_result(run.run_id, "nope") is None


def test_offloading_store_delegates(tmp_path) -> None:
    """An OFFLOADED result (over the inline cap) is a ref on the read
    surface (`list()` delegates — rehydrating every read measured 113x
    time/593x bytes on offload-heavy ledgers, 2026-07-14 adversary P1-2),
    while `find_completed_result` — the crash-replay path where
    byte-identity with the live handler result is a correctness
    requirement — rehydrates the offloader's own refs."""
    from abstractruntime.storage.artifacts import FileArtifactStore
    from abstractruntime.storage.offloading import OffloadingLedgerStore

    run = RunState.new(workflow_id="wf", entry_node="n1")
    inner = SqliteLedgerStore(SqliteDatabase(tmp_path / "o.sqlite3"))
    store = OffloadingLedgerStore(
        inner,
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
        max_inline_bytes=64,
    )
    big = {"content": "x" * 4096}
    store.append(_completed(run, node="n1", key="k1", result=big))

    listed = json.dumps(store.list(run.run_id)[0]["result"])
    assert listed.find("x" * 4096) == -1, "oversized content stays offloaded on the read surface"
    assert "$artifact" in listed, "the ref is intact, not silently dropped"

    found = store.find_completed_result(run.run_id, "k1")
    assert found == big, "crash-replay reuse rehydrates to the live handler bytes"


# ---------------------------------------------------------------------------
# Runtime seam
# ---------------------------------------------------------------------------


class _DuckLedger:
    """A duck-typed host ledger WITHOUT find_completed_result: the runtime
    must keep the historical inline scan for it."""

    def __init__(self) -> None:
        self.records: Dict[str, List[Dict[str, Any]]] = {}
        self.list_calls = 0

    def append(self, record: StepRecord) -> None:
        from dataclasses import asdict

        self.records.setdefault(record.run_id, []).append(asdict(record))

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        self.list_calls += 1
        return list(self.records.get(run_id, []))


def test_runtime_routes_through_store_lookup(tmp_path) -> None:
    from abstractruntime import Runtime

    calls: List[str] = []

    class _SpyLedger(InMemoryLedgerStore):
        def find_completed_result(
            self, run_id: str, idempotency_key: str
        ) -> Optional[Dict[str, Any]]:
            calls.append(idempotency_key)
            return super().find_completed_result(run_id, idempotency_key)

    rt = Runtime(run_store=__import__("abstractruntime").InMemoryRunStore(), ledger_store=_SpyLedger())
    assert rt._find_prior_completed_result("r1", "k-spy") is None
    assert calls == ["k-spy"]


def test_runtime_falls_back_to_inline_scan_for_duck_stores() -> None:
    from abstractruntime import InMemoryRunStore, Runtime

    duck = _DuckLedger()
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=duck)  # type: ignore[arg-type]
    run = RunState.new(workflow_id="wf", entry_node="n1")
    duck.append(_completed(run, node="n1", key="k1", result={"v": 42}))
    assert rt._find_prior_completed_result(run.run_id, "k1") == {"v": 42}
    assert duck.list_calls == 1
