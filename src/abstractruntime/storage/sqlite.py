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
import sqlite3
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import RunState, RunStatus, StepRecord, WaitReason, WaitState
from .base import LedgerStore, RunStore
from .commands import CommandAppendResult, CommandCursorStore, CommandRecord, CommandStore


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
                      PRIMARY KEY (run_id, seq)
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_run_seq ON ledger(run_id, seq);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ledger_heads (
                      run_id TEXT PRIMARY KEY,
                      last_seq INTEGER NOT NULL
                    );
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
            if run.waiting.reason == WaitReason.UNTIL:
                wait_until = str(run.waiting.until) if run.waiting.until else None

        payload = json.dumps(asdict(run), ensure_ascii=False)

        conn = self._db.connection()
        with conn:
            conn.execute(
                """
                INSERT INTO runs (
                  run_id, workflow_id, status, wait_reason, wait_until,
                  parent_run_id, actor_id, session_id,
                  created_at, updated_at,
                  run_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  workflow_id=excluded.workflow_id,
                  status=excluded.status,
                  wait_reason=excluded.wait_reason,
                  wait_until=excluded.wait_until,
                  parent_run_id=excluded.parent_run_id,
                  actor_id=excluded.actor_id,
                  session_id=excluded.session_id,
                  updated_at=excluded.updated_at,
                  run_json=excluded.run_json;
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
                ),
            )

            # Maintain WAIT_UNTIL index (only applies to WAITING runs).
            if run.status == RunStatus.WAITING and wait_reason == WaitReason.UNTIL.value and wait_until:
                conn.execute(
                    """
                    INSERT INTO wait_index (run_id, next_due_iso, updated_at_iso, status)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                      next_due_iso=excluded.next_due_iso,
                      updated_at_iso=excluded.updated_at_iso,
                      status=excluded.status;
                    """,
                    (str(run.run_id), str(wait_until), str(run.updated_at), "waiting_until"),
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
        rows = conn.execute(
            f"""
            SELECT
              run_id, workflow_id, status,
              wait_reason, wait_until,
              parent_run_id, actor_id, session_id,
              created_at, updated_at
            FROM runs
            {where}
            ORDER BY updated_at DESC
            LIMIT ?;
            """,
            (*params, lim),
        ).fetchall()

        out: List[Dict[str, Any]] = []
        for row in rows or []:
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

        payload = json.dumps(asdict(record), ensure_ascii=False)
        conn = self._db.connection()
        with conn:
            row = conn.execute("SELECT last_seq FROM ledger_heads WHERE run_id = ?;", (run_id,)).fetchone()
            last = int(row["last_seq"]) if row is not None else 0
            seq = last + 1
            conn.execute(
                "INSERT INTO ledger (run_id, seq, record_json) VALUES (?, ?, ?);",
                (run_id, int(seq), payload),
            )
            conn.execute(
                """
                INSERT INTO ledger_heads (run_id, last_seq)
                VALUES (?, ?)
                ON CONFLICT(run_id) DO UPDATE SET last_seq=excluded.last_seq;
                """,
                (run_id, int(seq)),
            )

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
