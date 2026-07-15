"""abstractruntime.storage.sqlite

SQLite-backed durability stores for production-oriented single-host deployments.

Design goals:
- Keep the durable execution substrate dependency-light (stdlib `sqlite3`).
- Provide restart-safe storage with real indexing (avoid directory scans + JSON parsing loops).
- Preserve the existing store interfaces (RunStore/LedgerStore/CommandStore) so hosts can
  switch backends without rewriting runtime logic.

Scope (backlog 446):
- RunStore (checkpointed RunState JSON)
- LedgerStore (append-only StepRecord JSON with per-run seq)
- CommandStore + CommandCursorStore (durable inbox + consumer cursor)
- WAIT_UNTIL index (wait_index table) so runners can query due runs efficiently.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import RunState, RunStatus, StepRecord, StepStatus, WaitReason, WaitState
from ..core.run_lifecycle import run_lifecycle_index_fields
from ..core.vars import is_paused_vars
from .base import LedgerStore, RunStore
from .commands import CommandAppendResult, CommandCursorStore, CommandRecord, CommandStore
from .serialize import dumps_compact, runstate_to_dict, steprecord_to_dict

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_value(v) for k, v in value.items())
    return False


def _runstate_from_dict(data: Dict[str, Any]) -> RunState:
    raw_status = data.get("status")
    status = raw_status if isinstance(raw_status, RunStatus) else RunStatus(str(raw_status))

    waiting: Optional[WaitState] = None
    raw_waiting = data.get("waiting")
    if isinstance(raw_waiting, dict):
        raw_reason = raw_waiting.get("reason")
        if raw_reason is None:
            raise ValueError("Persisted waiting state missing 'reason'")
        reason = raw_reason if isinstance(raw_reason, WaitReason) else WaitReason(str(raw_reason))
        waiting = WaitState(
            reason=reason,
            wait_key=raw_waiting.get("wait_key"),
            until=raw_waiting.get("until"),
            resume_to_node=raw_waiting.get("resume_to_node"),
            result_key=raw_waiting.get("result_key"),
            prompt=raw_waiting.get("prompt"),
            choices=raw_waiting.get("choices"),
            allow_free_text=bool(raw_waiting.get("allow_free_text", True)),
            details=raw_waiting.get("details"),
        )

    return RunState(
        run_id=str(data.get("run_id") or ""),
        workflow_id=str(data.get("workflow_id") or ""),
        status=status,
        current_node=str(data.get("current_node") or ""),
        vars=dict(data.get("vars") or {}),
        waiting=waiting,
        output=data.get("output"),
        error=data.get("error"),
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
        actor_id=data.get("actor_id"),
        session_id=data.get("session_id"),
        parent_run_id=data.get("parent_run_id"),
    )


class SqliteDatabase:
    """Small helper around a SQLite file with per-thread connections."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            self._apply_pragmas(conn)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the CALLING thread's connection (checkpoints WAL so the file
        is copy-clean). Per-thread connections mean each thread closes its
        own; other threads' connections close on their next close()/GC.
        Needed by per-entity runtimes (plan item 8): copying a home must
        carry a checkpointed run store, not a dangling -wal sidecar."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            return
        self._local.conn = None
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        # WAL improves writer/reader concurrency for the API+runner split.
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA foreign_keys=ON;")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            conn = sqlite3.connect(str(self._path), timeout=30.0)
            try:
                conn.row_factory = sqlite3.Row
                self._apply_pragmas(conn)

                # --- Runs ---
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
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
                conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status_updated ON runs(status, updated_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_workflow_updated ON runs(workflow_id, updated_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_waiting ON runs(status, wait_reason, wait_until);")
                # Unfiltered index pages ORDER BY updated_at with a LIMIT; a
                # bare index lets the 100k-run design target serve a page
                # without scanning every row (backlog 0068).
                conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at DESC);")

                # Hot-path read columns (backlog 0068): the external-control
                # probe runs at loop top AND before every save, and the run
                # index page is polled by hosts/UIs — both used to json.loads
                # the ENTIRE multi-MB run document for one or two fields
                # (~2.4ms/load, ~5-7ms/step of pure waste at 2MB states).
                # `paused` mirrors `vars._runtime.control.paused`;
                # `run_lifecycle_json` mirrors the sanitized `_run_lifecycle`
                # namespace ('null' when absent). SQL NULL = pre-0068 row not
                # yet backfilled (readers fall back to run_json). Same source
                # of truth, faster read — save() derives both from run.vars
                # in the same upsert that writes run_json.
                runs_cols = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA table_info(runs);").fetchall()
                }
                for column, decl in (("paused", "INTEGER"), ("run_lifecycle_json", "TEXT")):
                    if column in runs_cols:
                        continue
                    try:
                        conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {decl};")
                    except sqlite3.OperationalError as e:
                        # Two processes can race the migration (init lock is
                        # per-instance): duplicate column = already migrated.
                        if "duplicate column" not in str(e).lower():
                            raise
                # Near-empty partial index: the every-boot backfill probe and
                # the batch pagination stay O(1) once the table is converted
                # (same discipline as idx_ledger_unbackfilled).
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_unbackfilled "
                    "ON runs(run_id) WHERE paused IS NULL;"
                )
                self._backfill_run_hot_columns(conn)

                # --- WAIT_UNTIL index (scheduler) ---
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS wait_index (
                      run_id TEXT PRIMARY KEY,
                      next_due_iso TEXT NOT NULL,
                      updated_at_iso TEXT NOT NULL,
                      status TEXT
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_wait_index_next_due ON wait_index(next_due_iso);")

                # --- Ledger ---
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ledger (
                      run_id TEXT NOT NULL,
                      seq INTEGER NOT NULL,
                      record_json TEXT NOT NULL,
                      idempotency_key TEXT,
                      step_status TEXT,
                      PRIMARY KEY (run_id, seq)
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_run_seq ON ledger(run_id, seq);")

                # Idempotency columns (backlog 0047): the tick loop probes
                # "prior COMPLETED result for this key?" before EVERY effect
                # step; without an index that was a full-ledger parse per
                # step (the measured scale cliff: ~44ms/step at 32MB).
                # Upgrade path for pre-0047 databases: add the columns, then
                # backfill from record_json so pre-migration records stay
                # findable by the indexed query.
                ledger_cols = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA table_info(ledger);").fetchall()
                }
                # Duplicate-column tolerance: the init lock is per-instance,
                # so two PROCESSES opening one legacy file can both pass the
                # column check before either ALTERs — the loser must treat
                # "duplicate column" as already-migrated, not crash the open.
                for column in ("idempotency_key", "step_status"):
                    if column in ledger_cols:
                        continue
                    try:
                        conn.execute(f"ALTER TABLE ledger ADD COLUMN {column} TEXT;")
                    except sqlite3.OperationalError as e:
                        if "duplicate column" not in str(e).lower():
                            raise
                # Near-empty partial index: makes the backfill probe (every
                # boot) and the read path's version-skew guard O(1) instead
                # of a table scan. Only unbackfilled rows live in it.
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ledger_unbackfilled "
                    "ON ledger(run_id) "
                    "WHERE step_status IS NULL;"
                )
                self._backfill_ledger_idempotency(conn)
                # THREE-column index, deliberately (2026-07-13 perf adversary
                # N1): with a 2-column (run_id, idempotency_key) index the
                # planner preferred idx_ledger_run_seq to satisfy the probe's
                # ORDER BY seq and row-fetched EVERY record of the run — the
                # probe stayed O(per-run rows) and the cliff re-grew on
                # long-lived residents (measured 69ms/miss at 50k records).
                # Covering (run_id, idempotency_key, seq) serves equality +
                # order in one seek: 0.006ms at 10k, four orders better.
                # Upgrade guard: IF NOT EXISTS never rewrites an existing
                # 2-column shape, so drop-by-name when the columns mismatch.
                idx_cols = [
                    str(row["name"])
                    for row in conn.execute("PRAGMA index_info(idx_ledger_idem);").fetchall()
                ]
                if idx_cols and idx_cols != ["run_id", "idempotency_key", "seq"]:
                    conn.execute("DROP INDEX IF EXISTS idx_ledger_idem;")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ledger_idem "
                    "ON ledger(run_id, idempotency_key, seq) "
                    "WHERE idempotency_key IS NOT NULL;"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ledger_heads (
                      run_id TEXT PRIMARY KEY,
                      last_seq INTEGER NOT NULL
                    );
                    """
                )
                # Backfill ledger_heads for upgraded DBs.
                #
                # When `ledger_heads` is introduced after `ledger` already has rows, future appends must
                # continue from the existing MAX(seq) for each run_id (otherwise we would re-allocate
                # seq starting at 1 and hit `UNIQUE constraint failed: ledger.run_id, ledger.seq`).
                #
                # This is idempotent and safe to run on every startup.
                conn.execute(
                    """
                    INSERT INTO ledger_heads (run_id, last_seq)
                    SELECT run_id, MAX(seq) AS last_seq
                    FROM ledger
                    GROUP BY run_id
                    ON CONFLICT(run_id) DO UPDATE SET last_seq =
                      CASE
                        WHEN excluded.last_seq > ledger_heads.last_seq THEN excluded.last_seq
                        ELSE ledger_heads.last_seq
                      END;
                    """
                )

                # --- Commands (durable inbox) ---
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commands (
                      seq INTEGER PRIMARY KEY AUTOINCREMENT,
                      command_id TEXT NOT NULL UNIQUE,
                      run_id TEXT NOT NULL,
                      type TEXT NOT NULL,
                      payload_json TEXT NOT NULL,
                      ts TEXT NOT NULL,
                      client_id TEXT
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_commands_run_id ON commands(run_id);")

                # --- Command consumer cursors ---
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS command_cursors (
                      consumer_id TEXT PRIMARY KEY,
                      cursor INTEGER NOT NULL,
                      updated_at TEXT NOT NULL
                    );
                    """
                )

                conn.commit()
                self._initialized = True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    # Backfill batch size (class attr so tests can shrink it): bounds peak
    # memory to BATCH x state-size regardless of table size — a fetchall()
    # over the 100k-run design target at ~2MB states was a ~200GB
    # allocation and an open-crash-loop (2026-07-14 adversary P1-3).
    _BACKFILL_BATCH_ROWS = 32

    @classmethod
    def _backfill_run_hot_columns(cls, conn: sqlite3.Connection) -> None:
        """One-time backfill of `paused` / `run_lifecycle_json` (backlog 0068).

        Idempotent: matches nothing once every row carries values. Active
        runs self-heal on their next save() regardless; this converts
        dormant/terminal rows so the read-side run_json fallback goes quiet.
        Sanitization must match the write path, so this is a Python loop
        (json_extract cannot apply `sanitize_run_lifecycle`); failures
        leave rows NULL — readers degrade to the run_json parse, never a
        wrong answer.

        Adversary-hardened (2026-07-14): (P1-1) every UPDATE is guarded
        `AND paused IS NULL` — runs rows are MUTABLE (unlike the ledger
        backfill this was modeled on), and an unguarded stale UPDATE racing
        a concurrent `save()` from another process stamped `paused=0` over
        a row whose document said paused — the probe then answered "not
        controlled" and the tick's next save DESTROYED the pause. With the
        guard, a concurrent save always wins. (P1-3) rows stream in
        cursor-paged batches with a commit per batch — bounded memory at
        any table size, and a crash resumes where it stopped instead of
        re-fetching everything.
        """
        try:
            probe = conn.execute(
                "SELECT 1 FROM runs WHERE paused IS NULL LIMIT 1;"
            ).fetchone()
            if probe is None:
                return
        except Exception:  # noqa: BLE001 - probe failure: leave backfill to a later boot
            return
        batch = max(1, int(cls._BACKFILL_BATCH_ROWS))
        cursor = ""
        while True:
            try:
                rows = conn.execute(
                    "SELECT run_id, run_json FROM runs "
                    "WHERE paused IS NULL AND run_id > ? "
                    "ORDER BY run_id LIMIT ?;",
                    (cursor, batch),
                ).fetchall()
            except Exception:  # noqa: BLE001 - best-effort; NULL rows keep the fallback path
                return
            if not rows:
                return
            for row in rows:
                try:
                    data = json.loads(str(row["run_json"] or "{}"))
                    vars_obj = data.get("vars") if isinstance(data, dict) else None
                except Exception:  # noqa: BLE001 - torn row: matches the readers' {} fallback
                    vars_obj = None
                paused = 1 if is_paused_vars(vars_obj) else 0
                lifecycle = run_lifecycle_index_fields(vars_obj).get("run_lifecycle")
                try:
                    conn.execute(
                        "UPDATE runs SET paused = ?, run_lifecycle_json = ? "
                        "WHERE run_id = ? AND paused IS NULL;",
                        (paused, dumps_compact(lifecycle), str(row["run_id"])),
                    )
                except Exception:  # noqa: BLE001 - best-effort; NULL rows keep the fallback path
                    logger.warning(
                        "SqliteDatabase run hot-column backfill failed for a row #FALLBACK "
                        "(readers keep the run_json parse for it)",
                        exc_info=True,
                    )
            cursor = str(rows[-1]["run_id"])
            try:
                conn.commit()
            except Exception:  # noqa: BLE001 - commit failure: rows retry at the next boot
                return

    @staticmethod
    def _backfill_ledger_idempotency(conn: sqlite3.Connection) -> None:
        """One-time column backfill for pre-0047 rows (idempotent: the WHERE
        clause matches nothing once every row carries a step_status).

        Prefers SQLite's json1 (one UPDATE); falls back to a Python-side
        batched loop on builds without json_extract. Failures leave rows
        unbackfilled — the read path detects that and degrades to the
        bounded scan, so this can never brick a database open.
        """
        try:
            has_rows = conn.execute(
                "SELECT 1 FROM ledger WHERE step_status IS NULL LIMIT 1;"
            ).fetchone()
            if has_rows is None:
                return
        except Exception:  # noqa: BLE001 - probe failure: leave backfill to a later boot
            return
        try:
            conn.execute(
                """
                UPDATE ledger SET
                  idempotency_key = json_extract(record_json, '$.idempotency_key'),
                  step_status = json_extract(record_json, '$.status')
                WHERE step_status IS NULL;
                """
            )
            return
        except sqlite3.OperationalError:
            # json1 unavailable — OR the one-shot UPDATE died on a malformed
            # record_json row (json_extract raises on torn JSON, failing the
            # whole statement). Both land here; the Python fallback below
            # handles each row individually, so behavior is correct either
            # way (replay adversary P2-c note: the misattribution is benign
            # because the fallback is row-tolerant).
            pass
        try:
            rows = conn.execute(
                "SELECT rowid, record_json FROM ledger WHERE step_status IS NULL;"
            ).fetchall()
            for row in rows:
                try:
                    rec = json.loads(str(row["record_json"] or "{}"))
                except Exception:  # noqa: BLE001 - torn row: stamp it out of the NULL index
                    rec = None
                if not isinstance(rec, dict):
                    # Stamp torn/non-dict rows OUT of the unbackfilled index
                    # (replay adversary P2-c): a permanently-NULL row made
                    # the skew probe true for its run FOREVER, silently
                    # degrading every probe miss back to the full scan the
                    # index exists to kill. 'unparseable' can never equal a
                    # real step status, so lookups are unaffected.
                    conn.execute(
                        "UPDATE ledger SET step_status = 'unparseable' WHERE rowid = ?;",
                        (row["rowid"],),
                    )
                    continue
                conn.execute(
                    "UPDATE ledger SET idempotency_key = ?, step_status = ? WHERE rowid = ?;",
                    (rec.get("idempotency_key"), rec.get("status"), row["rowid"]),
                )
        except Exception:  # noqa: BLE001 - best-effort; unbackfilled rows degrade to scan
            logger.warning(
                "SqliteDatabase ledger idempotency backfill failed #FALLBACK "
                "(indexed lookup degrades to bounded scan for pre-migration rows)",
                exc_info=True,
            )


class SqliteRunStore(RunStore):
    """SQLite-backed RunStore with QueryableRunStore methods and a WAIT_UNTIL index."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def save(self, run: RunState) -> None:
        wait_reason: Optional[str] = None
        wait_until: Optional[str] = None
        if run.waiting is not None:
            try:
                wait_reason = str(getattr(run.waiting.reason, "value", run.waiting.reason))
            except Exception:
                wait_reason = None
            # Deadline-carrying waits enter the due index: UNTIL always;
            # EVENT when it carries `until` (the D3 idle-timeout shape) —
            # the scheduler's due-scan must wake a parked visit whose
            # deadline passed, not just pure timers.
            if run.waiting.reason in (WaitReason.UNTIL, WaitReason.EVENT):
                wait_until = str(run.waiting.until) if run.waiting.until else None

        # Compact + by-reference serialization (backlog 0067): save() runs
        # for every step of every run; `asdict` deep-copied the whole vars
        # tree per save under a single-writer contract that makes the copy
        # pure waste (see storage/serialize.py).
        payload = dumps_compact(runstate_to_dict(run))

        # Hot-path read columns (backlog 0068): derived from run.vars in the
        # SAME upsert as run_json — one write, one truth, two read speeds.
        paused = 1 if is_paused_vars(run.vars) else 0
        lifecycle = run_lifecycle_index_fields(run.vars).get("run_lifecycle")
        lifecycle_json = dumps_compact(lifecycle)

        conn = self._db.connection()
        with conn:
            conn.execute(
                """
                INSERT INTO runs (
                  run_id, workflow_id, status, wait_reason, wait_until,
                  parent_run_id, actor_id, session_id,
                  created_at, updated_at,
                  run_json, paused, run_lifecycle_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  workflow_id=excluded.workflow_id,
                  status=excluded.status,
                  wait_reason=excluded.wait_reason,
                  wait_until=excluded.wait_until,
                  parent_run_id=excluded.parent_run_id,
                  actor_id=excluded.actor_id,
                  session_id=excluded.session_id,
                  updated_at=excluded.updated_at,
                  run_json=excluded.run_json,
                  paused=excluded.paused,
                  run_lifecycle_json=excluded.run_lifecycle_json;
                """,
                (
                    str(run.run_id),
                    str(run.workflow_id),
                    str(getattr(run.status, "value", run.status)),
                    wait_reason,
                    wait_until,
                    str(run.parent_run_id) if run.parent_run_id else None,
                    str(run.actor_id) if run.actor_id else None,
                    str(run.session_id) if run.session_id else None,
                    str(run.created_at),
                    str(run.updated_at),
                    payload,
                    paused,
                    lifecycle_json,
                ),
            )

            # Maintain the due index (WAITING runs with a deadline: UNTIL
            # always; EVENT when it carries an idle deadline — D3).
            if run.status == RunStatus.WAITING and wait_until:
                index_status = (
                    "waiting_until" if wait_reason == WaitReason.UNTIL.value else "waiting_event_deadline"
                )
                conn.execute(
                    """
                    INSERT INTO wait_index (run_id, next_due_iso, updated_at_iso, status)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                      next_due_iso=excluded.next_due_iso,
                      updated_at_iso=excluded.updated_at_iso,
                      status=excluded.status;
                    """,
                    (str(run.run_id), str(wait_until), str(run.updated_at), index_status),
                )
            else:
                conn.execute("DELETE FROM wait_index WHERE run_id = ?;", (str(run.run_id),))

    def load(self, run_id: str) -> Optional[RunState]:
        rid = str(run_id or "").strip()
        if not rid:
            return None
        conn = self._db.connection()
        row = conn.execute("SELECT run_json FROM runs WHERE run_id = ?;", (rid,)).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(str(row["run_json"] or "{}"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        try:
            return _runstate_from_dict(data)
        except Exception:
            return None

    def probe_control(self, run_id: str) -> Optional[tuple[str, bool]]:
        """Cheap `(status, paused)` read for the external-control probe
        (backlog 0068): the tick loop asks "cancelled or paused?" at loop
        top AND before every save, and the full-document load answered a
        two-field question at ~2.4ms per probe on multi-MB states.

        Returns None when the run is unknown OR the row predates the
        `paused` column (SQL NULL = not yet backfilled) — the runtime falls
        back to the full load, never guesses. The column is written in the
        same transaction as run_json, so this is the same truth at the same
        freshness, one column-read cheaper.
        """
        rid = str(run_id or "").strip()
        if not rid:
            return None
        conn = self._db.connection()
        row = conn.execute("SELECT status, paused FROM runs WHERE run_id = ?;", (rid,)).fetchone()
        if row is None:
            return None
        paused_raw = row["paused"]
        if paused_raw is None:
            return None  # pre-0068 row: unknown, let the caller full-load
        return (str(row["status"] or ""), bool(int(paused_raw)))

    def delete(self, run_id: str) -> bool:
        rid = str(run_id or "").strip()
        if not rid:
            return False
        conn = self._db.connection()
        with conn:
            cur = conn.execute("DELETE FROM runs WHERE run_id = ?;", (rid,))
            conn.execute("DELETE FROM wait_index WHERE run_id = ?;", (rid,))
        return int(getattr(cur, "rowcount", 0) or 0) > 0

    # --- QueryableRunStore methods ---

    def list_runs(
        self,
        *,
        status: Optional[RunStatus] = None,
        wait_reason: Optional[WaitReason] = None,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RunState]:
        clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(str(getattr(status, "value", status)))
        if workflow_id is not None:
            clauses.append("workflow_id = ?")
            params.append(str(workflow_id))
        if wait_reason is not None:
            clauses.append("wait_reason = ?")
            params.append(str(getattr(wait_reason, "value", wait_reason)))

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        lim = max(1, int(limit or 100))

        conn = self._db.connection()
        rows = conn.execute(
            f"SELECT run_json FROM runs {where} ORDER BY updated_at DESC LIMIT ?;",
            (*params, lim),
        ).fetchall()

        out: List[RunState] = []
        for row in rows or []:
            try:
                data = json.loads(str(row["run_json"] or "{}"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            try:
                out.append(_runstate_from_dict(data))
            except Exception:
                continue
        return out

    def list_run_index(
        self,
        *,
        status: Optional[RunStatus] = None,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
        root_only: bool = False,
        limit: int = 100,
        oldest_first: bool = False,
    ) -> List[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(str(getattr(status, "value", status)))
        if workflow_id is not None:
            clauses.append("workflow_id = ?")
            params.append(str(workflow_id))
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(str(session_id))
        if bool(root_only):
            clauses.append("(parent_run_id IS NULL OR parent_run_id = '')")

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        lim = max(1, int(limit or 100))

        conn = self._db.connection()
        # Column-first (backlog 0068), and deliberately WITHOUT run_json in
        # the page query: fetching the multi-MB document column dominates the
        # old cost even before json.loads (a 100-row page over ~2MB states
        # measured ~1.1s either way when run_json rode the SELECT). Honest
        # bound (adversary P2-1): `run_lifecycle_json` was ALTERed AFTER
        # run_json, so reading it still walks each row's overflow chain
        # (~30ms/100 rows at 2MB states) — the eliminated cost is the
        # 100x json.loads, not all I/O; a narrow side table could recover
        # the rest and was judged not worth the migration. Pre-0068 rows
        # (SQL NULL lifecycle) fetch their document individually below.
        direction = "ASC" if oldest_first else "DESC"
        rows = conn.execute(
            f"""
            SELECT
              run_id, workflow_id, status,
              wait_reason, wait_until,
              parent_run_id, actor_id, session_id,
              created_at, updated_at,
              run_lifecycle_json
            FROM runs
            {where}
            ORDER BY updated_at {direction}
            LIMIT ?;
            """,
            (*params, lim),
        ).fetchall()

        out: List[Dict[str, Any]] = []
        for row in rows or []:
            lifecycle: Any = None
            lifecycle_raw = row["run_lifecycle_json"]
            if lifecycle_raw is not None:
                try:
                    lifecycle = json.loads(str(lifecycle_raw))
                except Exception:
                    lifecycle = None
                lifecycle_fields = {"run_lifecycle": lifecycle if isinstance(lifecycle, dict) else None}
            else:
                # Pre-migration row: the honest fallback is the document.
                doc_row = conn.execute(
                    "SELECT run_json FROM runs WHERE run_id = ?;",
                    (str(row["run_id"] or ""),),
                ).fetchone()
                try:
                    run_payload = json.loads(str(doc_row["run_json"] or "{}")) if doc_row is not None else {}
                except Exception:
                    run_payload = {}
                vars_obj = run_payload.get("vars") if isinstance(run_payload, dict) else None
                lifecycle_fields = run_lifecycle_index_fields(vars_obj)
            out.append(
                {
                    "run_id": str(row["run_id"] or ""),
                    "workflow_id": str(row["workflow_id"] or ""),
                    "status": str(row["status"] or ""),
                    "wait_reason": str(row["wait_reason"] or "") or None,
                    "wait_until": str(row["wait_until"] or "") or None,
                    "parent_run_id": str(row["parent_run_id"] or "") or None,
                    "actor_id": str(row["actor_id"] or "") or None,
                    "session_id": str(row["session_id"] or "") or None,
                    "created_at": str(row["created_at"] or "") or None,
                    "updated_at": str(row["updated_at"] or "") or None,
                    **lifecycle_fields,
                }
            )
        return out

    def list_due_wait_until(
        self,
        *,
        now_iso: str,
        limit: int = 100,
    ) -> List[RunState]:
        now = str(now_iso or "").strip()
        lim = max(1, int(limit or 100))
        conn = self._db.connection()
        rows = conn.execute(
            """
            SELECT r.run_json
            FROM wait_index w
            JOIN runs r ON r.run_id = w.run_id
            WHERE w.next_due_iso <= ?
            ORDER BY w.next_due_iso ASC
            LIMIT ?;
            """,
            (now, lim),
        ).fetchall()

        out: List[RunState] = []
        for row in rows or []:
            try:
                data = json.loads(str(row["run_json"] or "{}"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            try:
                out.append(_runstate_from_dict(data))
            except Exception:
                continue
        return out

    def list_children(
        self,
        *,
        parent_run_id: str,
        status: Optional[RunStatus] = None,
    ) -> List[RunState]:
        parent = str(parent_run_id or "").strip()
        if not parent:
            return []
        clauses = ["parent_run_id = ?"]
        params: list[Any] = [parent]

        if status is not None:
            clauses.append("status = ?")
            params.append(str(getattr(status, "value", status)))

        where = "WHERE " + " AND ".join(clauses)
        conn = self._db.connection()
        rows = conn.execute(
            f"SELECT run_json FROM runs {where} ORDER BY created_at ASC, run_id ASC;",
            tuple(params),
        ).fetchall()

        out: List[RunState] = []
        for row in rows or []:
            try:
                data = json.loads(str(row["run_json"] or "{}"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            try:
                out.append(_runstate_from_dict(data))
            except Exception:
                continue
        return out


class SqliteLedgerStore(LedgerStore):
    """SQLite-backed append-only ledger store with per-run seq."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def append(self, record: StepRecord) -> None:
        run_id = str(record.run_id or "").strip()
        if not run_id:
            raise ValueError("StepRecord.run_id must be non-empty")

        payload = dumps_compact(steprecord_to_dict(record))
        conn = self._db.connection()
        with conn:
            self._insert_in_txn(conn, run_id, payload, record=record)

    def last_record(self, run_id: str) -> Optional[Dict[str, Any]]:
        """The run's LAST persisted record (cheap indexed tail read) — the
        authoritative chain head for `HashChainedLedgerStore`, so appenders
        never trust a per-process cache across processes."""
        rid = str(run_id or "").strip()
        if not rid:
            return None
        conn = self._db.connection()
        row = conn.execute(
            "SELECT record_json FROM ledger WHERE run_id = ? ORDER BY seq DESC LIMIT 1;",
            (rid,),
        ).fetchone()
        if row is None:
            return None
        try:
            obj = json.loads(str(row["record_json"] or "{}"))
        except Exception:  # noqa: BLE001 - a torn row reads as no head; verify reports it
            return None
        return obj if isinstance(obj, dict) else None

    def find_completed_result(
        self, run_id: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]:
        """Indexed idempotency lookup (backlog 0047): a point query on
        `idx_ledger_idem` replaces the per-step full-ledger parse (the
        measured scale cliff). Oldest-completed-wins (`ORDER BY seq ASC`)
        for exact parity with the historical scan on legacy duplicate keys.

        Version-skew guard: rows appended by a pre-0047 writer AFTER the
        backfill ran carry NULL columns and are invisible to the point
        query — if any such row exists for this run (O(1) probe on the
        near-empty partial index), degrade to the bounded base-class scan.
        HONEST BOUND (replay adversary P2-c): the degrade is the base
        class's IDEMPOTENCY_TAIL_WINDOW scan, so a NULL-column completed
        row deeper than the window is still missed — the documented
        at-least-once bound, not a no-miss guarantee. Materially defused by
        issuance-scoped keys: pre-0047 writers used the old key format,
        which post-0047 probes can never match anyway.
        """
        key = str(idempotency_key or "")
        rid = str(run_id or "").strip()
        if not key or not rid:
            return None
        conn = self._db.connection()
        row = conn.execute(
            "SELECT record_json FROM ledger "
            "WHERE run_id = ? AND idempotency_key = ? AND step_status = ? "
            "ORDER BY seq ASC LIMIT 1;",
            (rid, key, StepStatus.COMPLETED.value),
        ).fetchone()
        if row is not None:
            try:
                rec = json.loads(str(row["record_json"] or "{}"))
            except Exception:  # noqa: BLE001 - torn row reads as no result
                return None
            return rec.get("result") if isinstance(rec, dict) else None
        skew = conn.execute(
            "SELECT 1 FROM ledger WHERE run_id = ? AND step_status IS NULL LIMIT 1;",
            (rid,),
        ).fetchone()
        if skew is not None:
            return super().find_completed_result(rid, key)
        return None

    def append_chained(self, record: StepRecord, compute_hash: Any) -> None:
        """Append with the hash chain derived INSIDE one write transaction
        (the fork fix, 2026-07-10 adversarial finding): `BEGIN IMMEDIATE`
        takes the database write lock BEFORE the head is read, so two
        processes appending to one run can never both link to the same
        head — safe by construction, not by lock discipline. The per-home
        writer lease keeps this unreachable in normal topology; this is
        the data layer refusing to fork even if every belt above fails.

        `compute_hash(record_dict, prev_hash) -> record_hash` stays the
        chain policy's callable (`ledger_chain.compute_record_hash`); the
        store owns atomicity, never the hash policy."""
        run_id = str(record.run_id or "").strip()
        if not run_id:
            raise ValueError("StepRecord.run_id must be non-empty")
        conn = self._db.connection()
        # Explicit IMMEDIATE: the default deferred txn would read the head
        # under a read lock and race the upgrade; IMMEDIATE serializes
        # writers at the head read (busy_timeout absorbs the wait).
        conn.execute("BEGIN IMMEDIATE;")
        try:
            row = conn.execute(
                "SELECT record_json FROM ledger WHERE run_id = ? ORDER BY seq DESC LIMIT 1;",
                (run_id,),
            ).fetchone()
            prev: Optional[str] = None
            if row is not None:
                try:
                    tail = json.loads(str(row["record_json"] or "{}"))
                    if isinstance(tail, dict):
                        prev = tail.get("record_hash")
                except Exception:  # noqa: BLE001 - torn tail: chain from None; verify reports it
                    prev = None
            record.prev_hash = prev
            record_dict = steprecord_to_dict(record)
            record.record_hash = compute_hash(record_dict, prev)
            record_dict["record_hash"] = record.record_hash
            payload = dumps_compact(record_dict)
            self._insert_in_txn(conn, run_id, payload, record=record)
            conn.execute("COMMIT;")
        except BaseException:
            try:
                conn.execute("ROLLBACK;")
            except Exception:  # noqa: BLE001 - rollback of a failed txn best-effort
                pass
            raise

    def _insert_in_txn(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        payload: str,
        *,
        record: Optional[StepRecord] = None,
    ) -> None:
        idem_key: Optional[str] = None
        step_status: Optional[str] = None
        if record is not None:
            k = getattr(record, "idempotency_key", None)
            idem_key = k if isinstance(k, str) and k else None
            s = getattr(record, "status", None)
            step_status = str(getattr(s, "value", s)) if s is not None else None
        conn.execute(
            """
            INSERT INTO ledger_heads (run_id, last_seq)
            VALUES (?, 0)
            ON CONFLICT(run_id) DO NOTHING;
            """,
            (run_id,),
        )
        conn.execute("UPDATE ledger_heads SET last_seq = last_seq + 1 WHERE run_id = ?;", (run_id,))
        row = conn.execute("SELECT last_seq FROM ledger_heads WHERE run_id = ?;", (run_id,)).fetchone()
        if row is None:
            raise RuntimeError("Failed to allocate ledger seq")
        seq = int(row["last_seq"] or 0)
        try:
            conn.execute(
                "INSERT INTO ledger (run_id, seq, record_json, idempotency_key, step_status) "
                "VALUES (?, ?, ?, ?, ?);",
                (run_id, int(seq), payload, idem_key, step_status),
            )
        except sqlite3.IntegrityError as e:
            msg = str(e)
            if "UNIQUE constraint failed: ledger.run_id, ledger.seq" in msg:
                logger.error(
                    "SqliteLedgerStore.append failed due to duplicate seq allocation "
                    "(run_id=%s seq=%s db=%s). This usually indicates concurrent writers running "
                    "a non-atomic seq allocator or a corrupted ledger_heads table.",
                    run_id,
                    seq,
                    self._db.path,
                )
            raise sqlite3.IntegrityError(f"{msg} (run_id={run_id!r}, seq={seq}, db={self._db.path})") from e

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        rid = str(run_id or "").strip()
        if not rid:
            return []
        conn = self._db.connection()
        rows = conn.execute(
            "SELECT record_json FROM ledger WHERE run_id = ? ORDER BY seq ASC;",
            (rid,),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows or []:
            try:
                obj = json.loads(str(row["record_json"] or "{}"))
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def delete(self, run_id: str) -> int:
        rid = str(run_id or "").strip()
        if not rid:
            return 0
        conn = self._db.connection()
        with conn:
            cur = conn.execute("DELETE FROM ledger WHERE run_id = ?;", (rid,))
            conn.execute("DELETE FROM ledger_heads WHERE run_id = ?;", (rid,))
        return int(getattr(cur, "rowcount", 0) or 0)

    def count(self, run_id: str) -> int:
        rid = str(run_id or "").strip()
        if not rid:
            return 0
        conn = self._db.connection()
        row = conn.execute("SELECT last_seq FROM ledger_heads WHERE run_id = ?;", (rid,)).fetchone()
        if row is None:
            return 0
        try:
            return int(row["last_seq"] or 0)
        except Exception:
            return 0

    def count_many(self, run_ids: List[str]) -> Dict[str, int]:
        ids = [str(r or "").strip() for r in (run_ids or []) if str(r or "").strip()]
        if not ids:
            return {}
        # SQLite parameter limit is high enough for typical UI pages; chunk defensively.
        out: Dict[str, int] = {}
        conn = self._db.connection()
        for i in range(0, len(ids), 900):
            chunk = ids[i : i + 900]
            q = ",".join(["?"] * len(chunk))
            rows = conn.execute(f"SELECT run_id, last_seq FROM ledger_heads WHERE run_id IN ({q});", tuple(chunk)).fetchall()
            for row in rows or []:
                rid = str(row["run_id"] or "").strip()
                if not rid:
                    continue
                try:
                    out[rid] = int(row["last_seq"] or 0)
                except Exception:
                    out[rid] = 0
        return out

    def metrics_many(self, run_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """Return best-effort per-run metrics derived from completed ledger records."""
        ids = [str(r or "").strip() for r in (run_ids or []) if str(r or "").strip()]
        if not ids:
            return {}
        out: Dict[str, Dict[str, int]] = {}
        conn = self._db.connection()
        for i in range(0, len(ids), 300):
            chunk = ids[i : i + 300]
            q = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"""
                WITH completed AS (
                  SELECT run_id, record_json
                  FROM ledger
                  WHERE run_id IN ({q})
                    AND json_extract(record_json, '$.status') = 'completed'
                )
                SELECT
                  run_id AS run_id,
                  COUNT(*) AS steps,
                  SUM(CASE WHEN json_extract(record_json, '$.effect.type') = 'llm_call' THEN 1 ELSE 0 END) AS llm_calls,
                  SUM(
                    CASE
                      WHEN json_extract(record_json, '$.effect.type') = 'tool_calls'
                        THEN COALESCE(json_array_length(json_extract(record_json, '$.effect.payload.tool_calls')), 0)
                      ELSE 0
                    END
                  ) AS tool_calls,
                  SUM(
                    CASE
                      WHEN json_extract(record_json, '$.effect.type') = 'llm_call'
                        THEN COALESCE(json_extract(record_json, '$.result.usage.total_tokens'), 0)
                      ELSE 0
                    END
                  ) AS tokens_total
                FROM completed
                GROUP BY run_id;
                """,
                tuple(chunk),
            ).fetchall()
            for row in rows or []:
                rid = str(row["run_id"] or "").strip()
                if not rid:
                    continue
                def _i(v: Any) -> int:
                    try:
                        return int(v or 0)
                    except Exception:
                        return 0
                out[rid] = {
                    "steps": _i(row["steps"]),
                    "llm_calls": _i(row["llm_calls"]),
                    "tool_calls": _i(row["tool_calls"]),
                    "tokens_total": _i(row["tokens_total"]),
                }
        return out

    def list_after(self, *, run_id: str, after: int, limit: int = 1000) -> Tuple[List[Dict[str, Any]], int]:
        """Optional cursor API (not part of LedgerStore ABC).

        Cursor semantics match the existing gateway API: `after` is the last consumed seq.
        """
        rid = str(run_id or "").strip()
        a = int(after or 0)
        lim = max(1, int(limit or 1000))
        if not rid:
            return ([], a)
        conn = self._db.connection()
        rows = conn.execute(
            "SELECT seq, record_json FROM ledger WHERE run_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?;",
            (rid, a, lim),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        next_cursor = a
        for row in rows or []:
            try:
                obj = json.loads(str(row["record_json"] or "{}"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            out.append(obj)
            try:
                next_cursor = max(next_cursor, int(row["seq"] or next_cursor))
            except Exception:
                pass
        return (out, next_cursor)


class SqliteCommandStore(CommandStore):
    """SQLite-backed CommandStore (append-only, idempotent by command_id)."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def append(self, record: CommandRecord) -> CommandAppendResult:
        cid = str(record.command_id or "").strip() or uuid.uuid4().hex
        run_id = str(record.run_id or "").strip()
        typ = str(record.type or "").strip()
        payload = dict(record.payload or {})
        ts = str(record.ts or "").strip() or _utc_now_iso()
        client_id = str(record.client_id).strip() if isinstance(record.client_id, str) and record.client_id else None

        if not run_id:
            raise ValueError("CommandRecord.run_id must be non-empty")
        if not typ:
            raise ValueError("CommandRecord.type must be non-empty")
        if not isinstance(payload, dict) or not _is_json_value(payload):
            raise ValueError("CommandRecord.payload must be a JSON-serializable dict")

        conn = self._db.connection()
        with conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO commands (command_id, run_id, type, payload_json, ts, client_id)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (cid, run_id, typ, json.dumps(payload, ensure_ascii=False), ts, client_id),
            )
            if int(cur.rowcount or 0) == 1:
                seq = int(cur.lastrowid or 0)
                return CommandAppendResult(accepted=True, duplicate=False, seq=seq)

            row = conn.execute("SELECT seq FROM commands WHERE command_id = ?;", (cid,)).fetchone()
            seq2 = int(row["seq"]) if row is not None else 0
            return CommandAppendResult(accepted=False, duplicate=True, seq=seq2)

    def list_after(self, *, after: int, limit: int = 1000) -> Tuple[List[CommandRecord], int]:
        a = int(after or 0)
        lim = max(1, int(limit or 1000))
        conn = self._db.connection()
        rows = conn.execute(
            """
            SELECT seq, command_id, run_id, type, payload_json, ts, client_id
            FROM commands
            WHERE seq > ?
            ORDER BY seq ASC
            LIMIT ?;
            """,
            (a, lim),
        ).fetchall()

        out: List[CommandRecord] = []
        next_cursor = a
        for row in rows or []:
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            seq = int(row["seq"] or 0)
            out.append(
                CommandRecord(
                    command_id=str(row["command_id"] or ""),
                    run_id=str(row["run_id"] or ""),
                    type=str(row["type"] or ""),
                    payload=payload,
                    ts=str(row["ts"] or ""),
                    client_id=str(row["client_id"] or "") or None,
                    seq=seq,
                )
            )
            next_cursor = max(next_cursor, seq)
        return (out, next_cursor)

    def get_last_seq(self) -> int:
        conn = self._db.connection()
        row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS max_seq FROM commands;").fetchone()
        if row is None:
            return 0
        try:
            return int(row["max_seq"] or 0)
        except Exception:
            return 0

    def delete_by_run(self, run_id: str) -> int:
        rid = str(run_id or "").strip()
        if not rid:
            return 0
        conn = self._db.connection()
        with conn:
            cur = conn.execute("DELETE FROM commands WHERE run_id = ?;", (rid,))
        return int(getattr(cur, "rowcount", 0) or 0)


class SqliteCommandCursorStore(CommandCursorStore):
    """SQLite-backed durable cursor store for CommandStore replay."""

    def __init__(self, db: SqliteDatabase, *, consumer_id: str = "gateway_runner") -> None:
        self._db = db
        self._consumer_id = str(consumer_id or "gateway_runner").strip() or "gateway_runner"

    def load(self) -> int:
        conn = self._db.connection()
        row = conn.execute(
            "SELECT cursor FROM command_cursors WHERE consumer_id = ?;",
            (self._consumer_id,),
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row["cursor"] or 0)
        except Exception:
            return 0

    def save(self, cursor: int) -> None:
        cur = int(cursor or 0)
        conn = self._db.connection()
        with conn:
            conn.execute(
                """
                INSERT INTO command_cursors (consumer_id, cursor, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(consumer_id) DO UPDATE SET
                  cursor=excluded.cursor,
                  updated_at=excluded.updated_at;
                """,
                (self._consumer_id, cur, _utc_now_iso()),
            )
