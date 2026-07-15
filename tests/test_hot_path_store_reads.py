"""Hot-path store reads (backlog 0068).

The production SQLite backend used to re-parse multi-MB run documents for
one or two fields on the hottest paths: the external-control probe (loop top
+ before every save), the run index page (per row), the progress-event key
(full ledger parse per callback), and the steer sidecar (fresh connection
per tick iteration). These tests pin the column twins and fast paths:

- `paused`/`run_lifecycle_json` columns written in the SAME upsert as
  run_json (one truth, two read speeds), ALTER-migrated + backfilled on
  legacy databases;
- `probe_control` answers (status, paused) without a document parse and
  answers None on pre-migration rows (the runtime then full-loads);
- the runtime's control probe uses the fast path and still honors
  cancel/pause exactly as before;
- `list_run_index` reads the column and falls back to run_json per NULL row;
- the steer sidecar reuses a per-thread connection and still heals a
  data-root purge (the F3 property the fresh-connect design paid for).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus, StepPlan
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore
from abstractruntime.storage.steer_sidecar import SqliteSteerSidecar


def _mark_paused(run: RunState, paused: bool) -> None:
    runtime_ns = run.vars.setdefault("_runtime", {})
    control = runtime_ns.setdefault("control", {})
    control["paused"] = bool(paused)


# ---------------------------------------------------------------------------
# probe_control
# ---------------------------------------------------------------------------


def test_probe_control_reads_columns_and_matches_document_truth(tmp_path: Path) -> None:
    db = SqliteDatabase(tmp_path / "runs.sqlite3")
    store = SqliteRunStore(db)

    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    store.save(run)
    assert store.probe_control(run.run_id) == ("running", False)

    _mark_paused(run, True)
    store.save(run)
    assert store.probe_control(run.run_id) == ("running", True)

    _mark_paused(run, False)
    run.status = RunStatus.CANCELLED
    store.save(run)
    assert store.probe_control(run.run_id) == ("cancelled", False)

    assert store.probe_control("nope") is None
    assert store.probe_control("") is None


def test_probe_control_answers_none_on_pre_migration_rows(tmp_path: Path) -> None:
    """A row saved by a pre-0068 writer has SQL NULL in `paused` — the probe
    must answer None (unknown), never a guessed False."""
    path = tmp_path / "legacy.sqlite3"
    db = SqliteDatabase(path)
    store = SqliteRunStore(db)
    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    store.save(run)

    conn = db.connection()
    with conn:
        conn.execute("UPDATE runs SET paused = NULL WHERE run_id = ?;", (run.run_id,))
    assert store.probe_control(run.run_id) is None


def test_legacy_database_backfills_hot_columns_at_open(tmp_path: Path) -> None:
    """A database created WITHOUT the 0068 columns migrates + backfills at
    open: paused/lifecycle derive from run_json exactly as save() would."""
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE runs (
          run_id TEXT PRIMARY KEY,
          workflow_id TEXT NOT NULL,
          status TEXT NOT NULL,
          wait_reason TEXT,
          wait_until TEXT,
          parent_run_id TEXT,
          actor_id TEXT,
          session_id TEXT,
          created_at TEXT,
          updated_at TEXT,
          run_json TEXT NOT NULL
        );
        """
    )
    paused_doc = {
        "run_id": "r-paused",
        "workflow_id": "wf",
        "status": "running",
        "current_node": "n",
        "vars": {
            "_runtime": {"control": {"paused": True}},
            "_run_lifecycle": {"source": "editor", "purpose": "draft_test"},
        },
        "output": None,
        "error": None,
        "waiting": None,
        "created_at": "2026-07-14T00:00:00+00:00",
        "updated_at": "2026-07-14T00:00:00+00:00",
    }
    plain_doc = dict(paused_doc, run_id="r-plain", vars={})
    torn = "{this is not json"
    conn.execute(
        "INSERT INTO runs (run_id, workflow_id, status, run_json) VALUES (?, ?, ?, ?);",
        ("r-paused", "wf", "running", json.dumps(paused_doc)),
    )
    conn.execute(
        "INSERT INTO runs (run_id, workflow_id, status, run_json) VALUES (?, ?, ?, ?);",
        ("r-plain", "wf", "running", json.dumps(plain_doc)),
    )
    conn.execute(
        "INSERT INTO runs (run_id, workflow_id, status, run_json) VALUES (?, ?, ?, ?);",
        ("r-torn", "wf", "running", torn),
    )
    conn.commit()
    conn.close()

    db = SqliteDatabase(path)
    store = SqliteRunStore(db)
    assert store.probe_control("r-paused") == ("running", True)
    assert store.probe_control("r-plain") == ("running", False)
    # Torn run_json backfills to the readers' {} fallback: not paused, no lifecycle.
    assert store.probe_control("r-torn") == ("running", False)

    rows = {r["run_id"]: r for r in store.list_run_index(limit=10)}
    assert rows["r-paused"]["run_lifecycle"] == {"source": "editor", "purpose": "draft_test"}
    assert rows["r-plain"]["run_lifecycle"] is None
    assert rows["r-torn"]["run_lifecycle"] is None


def test_runtime_control_probe_uses_fast_path_and_still_honors_pause(tmp_path: Path) -> None:
    """The tick loop's external-control check must (a) skip full loads when
    the probe answers 'not controlled' and (b) still stop on pause/cancel."""
    db = SqliteDatabase(tmp_path / "rt.sqlite3")
    run_store = SqliteRunStore(db)
    ledger = SqliteLedgerStore(db)

    probes = {"n": 0}
    original_probe = run_store.probe_control

    class _CountingStore:
        def __getattr__(self, name: str) -> Any:
            return getattr(run_store, name)

        def probe_control(self, run_id: str) -> Optional[tuple]:
            probes["n"] += 1
            return original_probe(run_id)

    def node(run: RunState, ctx: Any) -> StepPlan:
        n = int(run.vars.get("n") or 0)
        if n >= 3:
            return StepPlan(node_id="n1", complete_output={"n": n})
        run.vars["n"] = n + 1
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "x"}, result_key="_temp.e"),
            next_node="n1",
        )

    from abstractruntime.core.runtime import EffectOutcome

    def stub_llm(run: RunState, effect: Effect, default_next_node: Optional[str] = None) -> EffectOutcome:
        return EffectOutcome.completed({"content": "ok"})

    wf = WorkflowSpec(workflow_id="wf-probe", entry_node="n1", nodes={"n1": node})
    rt = Runtime(
        run_store=_CountingStore(),
        ledger_store=ledger,
        effect_handlers={EffectType.LLM_CALL: stub_llm},
    )

    run_id = rt.start(workflow=wf)
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.COMPLETED
    assert probes["n"] > 0, "the fast probe must be consulted"

    # Pause honored through the fast path: a paused run refuses to advance.
    run2_id = rt.start(workflow=wf)
    latest = run_store.load(run2_id)
    _mark_paused(latest, True)
    run_store.save(latest)
    state2 = rt.tick(workflow=wf, run_id=run2_id)
    assert int(state2.vars.get("n") or 0) == 0, "a paused run must not execute nodes"


# ---------------------------------------------------------------------------
# list_run_index column read
# ---------------------------------------------------------------------------


def test_list_run_index_reads_lifecycle_from_the_column(tmp_path: Path) -> None:
    db = SqliteDatabase(tmp_path / "idx.sqlite3")
    store = SqliteRunStore(db)

    run = RunState.new(
        workflow_id="wf",
        entry_node="n",
        vars={"_run_lifecycle": {"source": "editor", "purpose": "draft_test", "bogus": "dropped"}},
    )
    store.save(run)

    # Poison run_json: if the index still parsed the document, this would
    # either crash or change the answer — the column must serve the read.
    conn = db.connection()
    with conn:
        conn.execute("UPDATE runs SET run_json = '{broken' WHERE run_id = ?;", (run.run_id,))

    rows = store.list_run_index(limit=5)
    assert rows and rows[0]["run_id"] == run.run_id
    assert rows[0]["run_lifecycle"] == {"source": "editor", "purpose": "draft_test"}


# ---------------------------------------------------------------------------
# Steer sidecar connection reuse + purge healing
# ---------------------------------------------------------------------------


def test_sidecar_reuses_one_connection_per_thread(tmp_path: Path) -> None:
    sidecar = SqliteSteerSidecar(str(tmp_path / "steer.sqlite3"))
    sidecar.append("r1", {"role": "system", "content": "a"})
    conn_after_first = getattr(sidecar._local, "conn", None)
    assert conn_after_first is not None
    sidecar.pending("r1")
    sidecar.watermark("r1")
    assert getattr(sidecar._local, "conn", None) is conn_after_first, "calls must reuse the thread connection"

    # A different thread gets its own connection (sqlite3 thread affinity).
    seen: dict[str, Any] = {}

    def _other() -> None:
        sidecar.append("r1", {"role": "system", "content": "b"})
        seen["conn"] = getattr(sidecar._local, "conn", None)

    t = threading.Thread(target=_other)
    t.start()
    t.join()
    assert seen["conn"] is not None and seen["conn"] is not conn_after_first


def test_sidecar_survives_a_data_root_purge_with_a_cached_connection(tmp_path: Path) -> None:
    """The F3 healing property must survive connection caching: deleting the
    database file under a LIVE sidecar reopens + recreates on the next call
    — a cached connection writing the unlinked inode would silently lose
    steers until a process restart."""
    path = tmp_path / "steer.sqlite3"
    sidecar = SqliteSteerSidecar(str(path))
    sidecar.append("r1", {"role": "system", "content": "before purge"})

    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
    assert not path.exists()

    seq = sidecar.append("r1", {"role": "system", "content": "after purge"})
    assert seq == 1, "a purged store restarts its seq space"
    assert path.exists(), "the file must be recreated on disk"
    items = sidecar.pending("r1")
    assert [i["message"]["content"] for i in items] == ["after purge"]


# ---------------------------------------------------------------------------
# Adversary folds (2026-07-14)
# ---------------------------------------------------------------------------


def test_stale_backfill_update_never_clobbers_a_live_pause(tmp_path: Path) -> None:
    """Adversary P1-1: runs rows are MUTABLE (unlike the ledger rows the
    backfill was modeled on). A stale backfill UPDATE racing a concurrent
    save() used to stamp paused=0 over a freshly-paused row — the probe
    then said "not controlled" and the tick's next save DESTROYED the
    pause. The guarded statement must be a no-op on any stamped row."""
    db = SqliteDatabase(tmp_path / "race.sqlite3")
    store = SqliteRunStore(db)
    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    _mark_paused(run, True)
    store.save(run)
    assert store.probe_control(run.run_id) == ("running", True)

    # Replay the exact stale write the TOCTOU produced (snapshot said
    # not-paused) — the `AND paused IS NULL` guard must reject it.
    conn = db.connection()
    with conn:
        conn.execute(
            "UPDATE runs SET paused = ?, run_lifecycle_json = ? WHERE run_id = ? AND paused IS NULL;",
            (0, "null", run.run_id),
        )
    assert store.probe_control(run.run_id) == ("running", True), "a concurrent save always wins"


def test_backfill_streams_in_batches_and_survives_restart(tmp_path: Path) -> None:
    """Adversary P1-3: fetchall() over every unbackfilled row materialized
    the whole table in RAM (~200GB at the 100k-run design target) and the
    single end-of-open commit made an OOM kill an open-crash-loop. The
    backfill must page with a bounded batch and commit per batch."""
    path = tmp_path / "big-legacy.sqlite3"
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE runs (
          run_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, status TEXT NOT NULL,
          wait_reason TEXT, wait_until TEXT, parent_run_id TEXT, actor_id TEXT,
          session_id TEXT, created_at TEXT, updated_at TEXT, run_json TEXT NOT NULL
        );
        """
    )
    docs = []
    for i in range(21):
        doc = {
            "run_id": f"r-{i:03d}", "workflow_id": "wf", "status": "running",
            "current_node": "n", "vars": {"_runtime": {"control": {"paused": bool(i % 2)}}},
            "output": None, "error": None, "waiting": None,
            "created_at": "2026-07-14T00:00:00+00:00", "updated_at": "2026-07-14T00:00:00+00:00",
        }
        docs.append(doc)
        conn.execute(
            "INSERT INTO runs (run_id, workflow_id, status, run_json) VALUES (?, ?, ?, ?);",
            (doc["run_id"], "wf", "running", json.dumps(doc)),
        )
    conn.commit()
    conn.close()

    original_batch = SqliteDatabase._BACKFILL_BATCH_ROWS
    SqliteDatabase._BACKFILL_BATCH_ROWS = 4  # force many pages
    try:
        db = SqliteDatabase(path)
        store = SqliteRunStore(db)
        for i, doc in enumerate(docs):
            expected_paused = bool(i % 2)
            assert store.probe_control(doc["run_id"]) == ("running", expected_paused)
    finally:
        SqliteDatabase._BACKFILL_BATCH_ROWS = original_batch


def test_sidecar_purge_heals_every_thread_not_just_the_first(tmp_path: Path) -> None:
    """Adversary P1-2: existence-checking heals only the FIRST thread to
    touch after a purge — it recreates the file, and every other thread's
    existence check then passes while its cached connection still points at
    the unlinked inode: appends 'succeed' into the orphaned file and are
    silently lost. Identity (st_dev, st_ino) must be the freshness test."""
    path = tmp_path / "steer.sqlite3"
    sidecar = SqliteSteerSidecar(str(path))

    # A worker thread caches its own connection.
    ready = threading.Event()
    go_after_heal = threading.Event()
    results: dict[str, Any] = {}

    def worker() -> None:
        sidecar.append("r1", {"role": "system", "content": "pre-purge"})
        ready.set()
        go_after_heal.wait(timeout=10)
        results["seq"] = sidecar.append("r1", {"role": "system", "content": "AFTER PURGE"})

    t = threading.Thread(target=worker)
    t.start()
    ready.wait(timeout=10)

    # Purge the data root, then MAIN thread heals first (recreates the file).
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
    sidecar.append("r1", {"role": "system", "content": "healer"})
    assert path.exists()

    # Now the worker appends: with existence-checking its cached connection
    # would write the DEAD inode; identity-checking must reopen.
    go_after_heal.set()
    t.join(timeout=10)
    assert results.get("seq") is not None

    on_disk = sqlite3.connect(str(path))
    rows = [r[0] for r in on_disk.execute("SELECT payload FROM steer_messages ORDER BY seq").fetchall()]
    on_disk.close()
    assert any("AFTER PURGE" in r for r in rows), "the second thread's steer must land in the REAL file"


def test_offloading_run_store_forwards_probe_control(tmp_path: Path) -> None:
    """Adversary P1-4: the gateway's production wiring is
    OffloadingRunStore(SqliteRunStore), and the wrapper forwards methods
    explicitly — without a passthrough the fast path silently never fired
    on the exact deployment whose measurements justified it."""
    from abstractruntime.storage.artifacts import InMemoryArtifactStore
    from abstractruntime.storage.in_memory import InMemoryRunStore
    from abstractruntime.storage.offloading import OffloadingRunStore

    db = SqliteDatabase(tmp_path / "gw.sqlite3")
    inner = SqliteRunStore(db)
    wrapped = OffloadingRunStore(inner, artifact_store=InMemoryArtifactStore())

    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    _mark_paused(run, True)
    wrapped.save(run)
    assert wrapped.probe_control(run.run_id) == ("running", True)

    # Inner stores without the probe answer None (runtime full-load fallback).
    plain = OffloadingRunStore(InMemoryRunStore(), artifact_store=InMemoryArtifactStore())
    assert plain.probe_control("whatever") is None


# ---------------------------------------------------------------------------
# Progress event key
# ---------------------------------------------------------------------------


def test_progress_event_key_never_lists_the_ledger(tmp_path: Path) -> None:
    """The progress-event idempotency key is a pure uniquifier; building it
    must not parse the ledger (generated-media runs emit many per effect)."""
    db = SqliteDatabase(tmp_path / "prog.sqlite3")
    run_store = SqliteRunStore(db)
    ledger = SqliteLedgerStore(db)

    lists = {"n": 0}
    original_list = ledger.list

    class _CountingLedger:
        def __getattr__(self, name: str) -> Any:
            return getattr(ledger, name)

        def list(self, run_id: str):
            lists["n"] += 1
            return original_list(run_id)

    counting = _CountingLedger()
    rt = Runtime(run_store=run_store, ledger_store=counting)
    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    run_store.save(run)

    lists["n"] = 0
    keys = set()
    for i in range(5):
        rt._append_progress_event(
            run=run,
            node_id="n",
            step_id="step-1",
            idempotency_key="k",
            attempt=1,
            event={"pct": i},
        )
    assert lists["n"] == 0, "progress events must not parse the ledger"

    records = ledger.list(run.run_id)
    keys = {r.get("idempotency_key") for r in records}
    assert len(keys) == 5, "each progress event carries a unique key"
    assert all(str(k).startswith("system:progress:step-1:") for k in keys)
