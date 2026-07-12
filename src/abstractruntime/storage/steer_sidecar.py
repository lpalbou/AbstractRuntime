"""Steer sidecar store (hooks plan H4): single-writer guidance delivery.

THE PROBLEM THIS SOLVES: steering a live run by mutating `_runtime.inbox`
from a host thread races the tick thread's own run-state saves — the append
can be lost for a whole tick (stale save landing last) or, worse, a stale
RUNNING snapshot can clobber a terminal state. The gateway's inject_guidance
documents this exact loss window and defers the fix to "route the mutation
through the single tick-writer" — this module is that route.

SHAPE: an append-only per-run message queue OUTSIDE run vars. Hosts (gateway
command lane, tests, in-process callers) APPEND; only the TICK THREAD drains
— at iteration boundaries it moves pending messages into `_runtime.inbox`
(where the ReAct reason node already looks), advances the consumed watermark,
and writes a `steer_seen` ledger record (the delivery ack that H4 requires:
"seen at iteration N" is a fact in the run's own record, not a client guess).

The store is deliberately dumb: no status checks (the Runtime's steer() verb
owns refusal semantics), no delivery guarantees beyond append-order, and
message payloads are small JSON dicts (the inbox item shape, typically
{"role": "system", "content": ...}).
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Protocol

__all__ = [
    "SteerSidecarStore",
    "InMemorySteerSidecar",
    "SqliteSteerSidecar",
]

# A run with this many UNDELIVERED steers is not being steered, it is being
# flooded — refuse loudly instead of growing without bound (dropping steers
# silently would be worse: a steer is an operator's word).
MAX_PENDING_STEERS_PER_RUN = 500


class SteerSidecarStore(Protocol):
    """Append-only steer queue with a per-run consumed watermark."""

    def append(self, run_id: str, message: Dict[str, Any]) -> int:
        """Queue a message for the run; returns its per-run monotonic seq (1-based)."""
        ...

    def pending(self, run_id: str) -> List[Dict[str, Any]]:
        """Unconsumed messages in append order: [{"seq": int, "message": dict}, ...]."""
        ...

    def ack(self, run_id: str, up_to_seq: int) -> None:
        """Advance the consumed watermark (idempotent; never moves backwards)."""
        ...

    def watermark(self, run_id: str) -> int:
        """Highest consumed seq (0 = nothing consumed)."""
        ...


class InMemorySteerSidecar:
    """Thread-safe in-process sidecar (tests, single-process hosts).

    Consumed messages are COMPACTED away on ack (a long-lived host steering
    many runs must not leak every message ever sent); `_next_seq` keeps the
    per-run seq monotonic across compactions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        self._watermarks: Dict[str, int] = {}
        self._next_seq: Dict[str, int] = {}

    def append(self, run_id: str, message: Dict[str, Any]) -> int:
        rid = str(run_id)
        with self._lock:
            queue = self._messages.setdefault(rid, [])
            if len(queue) >= MAX_PENDING_STEERS_PER_RUN:
                raise RuntimeError(
                    f"run '{rid}' already has {len(queue)} undelivered steers "
                    f"(cap {MAX_PENDING_STEERS_PER_RUN}); refusing the append — "
                    "is the run actually ticking?"
                )
            seq = self._next_seq.get(rid, 0) + 1
            self._next_seq[rid] = seq
            queue.append({"seq": seq, "message": copy.deepcopy(message)})
            return seq

    def pending(self, run_id: str) -> List[Dict[str, Any]]:
        rid = str(run_id)
        with self._lock:
            mark = self._watermarks.get(rid, 0)
            return [
                {"seq": item["seq"], "message": copy.deepcopy(item["message"])}
                for item in self._messages.get(rid, [])
                if item["seq"] > mark
            ]

    def ack(self, run_id: str, up_to_seq: int) -> None:
        rid = str(run_id)
        with self._lock:
            current = self._watermarks.get(rid, 0)
            mark = max(current, int(up_to_seq))
            self._watermarks[rid] = mark
            queue = self._messages.get(rid)
            if queue:
                self._messages[rid] = [item for item in queue if item["seq"] > mark]

    def watermark(self, run_id: str) -> int:
        with self._lock:
            return self._watermarks.get(str(run_id), 0)


class SqliteSteerSidecar:
    """Durable sidecar sharing SQLite semantics with the run stores.

    One table, WAL mode, per-instance lock around write transactions. A steer
    accepted here survives a host restart and is delivered at the run's next
    iteration boundary — the durable half of the H4 promise.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        directory = os.path.dirname(self._db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Touch once at construction so misconfiguration fails HERE, loudly.
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: autocommit + fully manual transactions, so the
        # explicit BEGIN IMMEDIATE in append() never fights the driver's
        # implicit transaction management.
        conn = sqlite3.connect(self._db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Schema per-connection, not per-instance (gateway adversary F3): a
        # data-root purge deletes the file under a LIVE instance; the next
        # connect silently recreates an EMPTY db, and appends would die on
        # "no such table" until a process restart. IF NOT EXISTS is a cheap
        # no-op on the hot path and closes the class everywhere.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steer_messages (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                payload TEXT NOT NULL,
                consumed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, seq)
            )
            """
        )
        return conn

    def append(self, run_id: str, message: Dict[str, Any]) -> int:
        rid = str(run_id)
        payload = json.dumps(dict(message), ensure_ascii=False, default=str)
        with self._lock:
            conn = self._connect()
            try:
                # BEGIN IMMEDIATE takes the write lock BEFORE reading MAX(seq),
                # so two processes can never both compute the same next seq
                # (the ledger-chain append_chained precedent — a plain deferred
                # transaction upgraded the lock only at INSERT time, letting a
                # concurrent writer die on the primary key).
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0), COUNT(*) FILTER (WHERE consumed = 0) "
                    "FROM steer_messages WHERE run_id = ?",
                    (rid,),
                ).fetchone()
                pending_count = int(row[1])
                if pending_count >= MAX_PENDING_STEERS_PER_RUN:
                    conn.execute("ROLLBACK")
                    raise RuntimeError(
                        f"run '{rid}' already has {pending_count} undelivered steers "
                        f"(cap {MAX_PENDING_STEERS_PER_RUN}); refusing the append — "
                        "is the run actually ticking?"
                    )
                seq = int(row[0]) + 1
                conn.execute(
                    "INSERT INTO steer_messages (run_id, seq, payload, consumed) VALUES (?, ?, ?, 0)",
                    (rid, seq, payload),
                )
                conn.execute("COMMIT")
                return seq
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def pending(self, run_id: str) -> List[Dict[str, Any]]:
        rid = str(run_id)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT seq, payload FROM steer_messages WHERE run_id = ? AND consumed = 0 ORDER BY seq",
                (rid,),
            ).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for seq, payload in rows:
            try:
                message = json.loads(payload)
            except Exception:
                message = {"role": "system", "content": str(payload)}
            out.append({"seq": int(seq), "message": message})
        return out

    def ack(self, run_id: str, up_to_seq: int) -> None:
        rid = str(run_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "UPDATE steer_messages SET consumed = 1 WHERE run_id = ? AND seq <= ? AND consumed = 0",
                    (rid, int(up_to_seq)),
                )
                # Compact: consumed rows are spent — keep only the highest one
                # (it anchors watermark()) so a long-lived run never accretes
                # every steer it was ever sent.
                conn.execute(
                    "DELETE FROM steer_messages WHERE run_id = ? AND consumed = 1 AND seq < "
                    "(SELECT MAX(seq) FROM steer_messages WHERE run_id = ? AND consumed = 1)",
                    (rid, rid),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def watermark(self, run_id: str) -> int:
        rid = str(run_id)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM steer_messages WHERE run_id = ? AND consumed = 1",
                (rid,),
            ).fetchone()
        finally:
            conn.close()
        return int(row[0])
