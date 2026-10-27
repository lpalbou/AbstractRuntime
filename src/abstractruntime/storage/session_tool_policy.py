"""Session tool policy store: a person's standing tool decision for one conversation.

THE PROBLEM THIS SOLVES: "approve all" / "approve this kind" used to live in each client. The
runtime knew only the run's start-time `_runtime.tool_policy`, so it emitted an approval wait for
every gated call, the ledger recorded it, every client showed the gate, and the granting client
answered it a moment later (the gate flashed in every app and in the replayable ledger).

SHAPE: one row per `session_id`, outside run vars (a host writing run vars races the tick
thread's saves: the loss window `steer_sidecar.py` removed). Hosts (the gateway's
`/sessions/{id}/tool-policy` routes) WRITE; the TOOL_CALLS handler READS the row of the run's
session before it decides whether a batch must wait, so every run of the session (each turn's
root run, every delegated child, a run resumed after a restart) sees the same decision with no
copy step. The stored dict is the gateway's normalized, clamped policy (contract v1):

    {"version": 1, "session_id": str, "revision": int,
     "approve": {tool_name: matched_selector}, "deny": {tool_name: matched_selector},
     "ask": [tool_name, ...], "set_by": {...}, "set_at": iso, ...}

The runtime reads only `approve`, `deny`, `ask`, `revision`, `set_by`, `set_at`; other keys are the
host's provenance and are returned unchanged. Every write and reset is also appended to a history
table (who, when, what), so the policy's own changes are auditable.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol, Set, Tuple

__all__ = [
    "SESSION_TOOL_POLICY_VERSION",
    "SessionToolPolicyConflict",
    "SessionToolPolicyStore",
    "InMemorySessionToolPolicyStore",
    "SqliteSessionToolPolicyStore",
    "session_policy_names",
]

SESSION_TOOL_POLICY_VERSION = 1


class SessionToolPolicyConflict(RuntimeError):
    """`if_revision` did not match the stored revision (a concurrent writer won)."""

    def __init__(self, *, expected: int, actual: int) -> None:
        super().__init__(f"session tool policy revision is {actual}, not {expected}")
        self.expected = int(expected)
        self.actual = int(actual)


class SessionToolPolicyStore(Protocol):
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """The current policy of the session, or None (no policy / reset)."""
        ...

    def put(self, session_id: str, policy: Mapping[str, Any], *, if_revision: Optional[int] = None) -> Dict[str, Any]:
        """Replace the session's policy; returns it with the new `revision` and `session_id`."""
        ...

    def delete(self, session_id: str, *, if_revision: Optional[int] = None, actor: Optional[Mapping[str, Any]] = None) -> int:
        """Reset the session (no policy); returns the new revision."""
        ...

    def history(self, session_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
        """The session's writes and resets, oldest first: [{revision, at, action, policy}]."""
        ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_sid(session_id: Any) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    return sid


def _stamp(policy: Mapping[str, Any], *, session_id: str, revision: int) -> Dict[str, Any]:
    out = copy.deepcopy(dict(policy))
    out["version"] = SESSION_TOOL_POLICY_VERSION
    out["session_id"] = session_id
    out["revision"] = int(revision)
    out.setdefault("set_at", _now())
    return out


def session_policy_names(policy: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(approved, denied): tool name -> matched selector, from a stored policy.

    Deny wins over approve; `ask` names are removed from approve. A malformed policy yields
    nothing (fail toward asking): only a mapping of non-empty names counts.
    """
    if not isinstance(policy, Mapping):
        return {}, {}

    def _names(key: str) -> Dict[str, str]:
        raw = policy.get(key)
        out: Dict[str, str] = {}
        if isinstance(raw, Mapping):
            for name, matched in raw.items():
                n = str(name or "").strip()
                if n:
                    out[n] = str(matched or "tool")
        return out

    denied = _names("deny")
    approved = _names("approve")
    ask_raw = policy.get("ask")
    ask: Set[str] = {str(n).strip() for n in ask_raw if str(n).strip()} if isinstance(ask_raw, list) else set()
    for name in list(approved):
        if name in denied or name in ask:
            approved.pop(name, None)
    return approved, denied


class InMemorySessionToolPolicyStore:
    """Thread-safe in-process store (tests, single-process hosts)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: Dict[str, Optional[Dict[str, Any]]] = {}
        self._revisions: Dict[str, int] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}

    def _check(self, sid: str, if_revision: Optional[int]) -> int:
        current = self._revisions.get(sid, 0)
        if if_revision is not None and int(if_revision) != current:
            raise SessionToolPolicyConflict(expected=int(if_revision), actual=current)
        return current

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with self._lock:
            row = self._rows.get(sid)
            return copy.deepcopy(row) if row is not None else None

    def put(self, session_id: str, policy: Mapping[str, Any], *, if_revision: Optional[int] = None) -> Dict[str, Any]:
        sid = _clean_sid(session_id)
        with self._lock:
            revision = self._check(sid, if_revision) + 1
            row = _stamp(policy, session_id=sid, revision=revision)
            self._rows[sid] = row
            self._revisions[sid] = revision
            self._history.setdefault(sid, []).append(
                {"revision": revision, "at": row["set_at"], "action": "put", "policy": copy.deepcopy(row)}
            )
            return copy.deepcopy(row)

    def delete(self, session_id: str, *, if_revision: Optional[int] = None, actor: Optional[Mapping[str, Any]] = None) -> int:
        sid = _clean_sid(session_id)
        with self._lock:
            revision = self._check(sid, if_revision) + 1
            self._rows[sid] = None
            self._revisions[sid] = revision
            self._history.setdefault(sid, []).append(
                {"revision": revision, "at": _now(), "action": "delete", "policy": None,
                 "actor": dict(actor) if isinstance(actor, Mapping) else None}
            )
            return revision

    def revision(self, session_id: str) -> int:
        with self._lock:
            return self._revisions.get(str(session_id or "").strip(), 0)

    def history(self, session_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._history.get(str(session_id or "").strip(), [])[-max(1, int(limit)):])


class SqliteSessionToolPolicyStore:
    """Durable store beside the run stores (`<data_root>/session_tool_policy.sqlite3`).

    One connection per operation: reads happen once per tool batch (not per tick), and a fresh
    connection survives a data-root purge (the schema is re-created per connection, like the
    steer sidecar's). Writes run in `BEGIN IMMEDIATE` so the revision check and the write are
    one transaction across processes.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        directory = os.path.dirname(self._db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._connect().close()  # misconfiguration fails HERE, loudly

    @property
    def db_path(self) -> str:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_tool_policy (
                session_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,
                policy TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_tool_policy_history (
                session_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                at TEXT NOT NULL,
                action TEXT NOT NULL,
                policy TEXT,
                actor TEXT,
                PRIMARY KEY (session_id, revision)
            )
            """
        )
        return conn

    @staticmethod
    def _current(conn: sqlite3.Connection, sid: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        row = conn.execute("SELECT revision, policy FROM session_tool_policy WHERE session_id = ?", (sid,)).fetchone()
        if row is None:
            return 0, None
        policy = json.loads(row[1]) if row[1] else None
        return int(row[0]), policy if isinstance(policy, dict) else None

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        sid = str(session_id or "").strip()
        if not sid:
            return None
        conn = self._connect()
        try:
            return self._current(conn, sid)[1]
        finally:
            conn.close()

    def revision(self, session_id: str) -> int:
        conn = self._connect()
        try:
            return self._current(conn, str(session_id or "").strip())[0]
        finally:
            conn.close()

    def _write(self, sid: str, policy: Optional[Mapping[str, Any]], *, if_revision: Optional[int], actor: Optional[Mapping[str, Any]]) -> Tuple[int, Optional[Dict[str, Any]]]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                current, _old = self._current(conn, sid)
                if if_revision is not None and int(if_revision) != current:
                    raise SessionToolPolicyConflict(expected=int(if_revision), actual=current)
                revision = current + 1
                row = _stamp(policy, session_id=sid, revision=revision) if policy is not None else None
                text = json.dumps(row, sort_keys=True) if row is not None else None
                conn.execute(
                    "INSERT INTO session_tool_policy (session_id, revision, policy) VALUES (?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET revision = excluded.revision, policy = excluded.policy",
                    (sid, revision, text),
                )
                conn.execute(
                    "INSERT INTO session_tool_policy_history (session_id, revision, at, action, policy, actor) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        sid,
                        revision,
                        row["set_at"] if row is not None else _now(),
                        "put" if row is not None else "delete",
                        text,
                        json.dumps(dict(actor), sort_keys=True) if isinstance(actor, Mapping) else None,
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return revision, row
        finally:
            conn.close()

    def put(self, session_id: str, policy: Mapping[str, Any], *, if_revision: Optional[int] = None) -> Dict[str, Any]:
        sid = _clean_sid(session_id)
        _revision, row = self._write(sid, policy, if_revision=if_revision, actor=policy.get("set_by") if isinstance(policy, Mapping) else None)
        assert row is not None
        return row

    def delete(self, session_id: str, *, if_revision: Optional[int] = None, actor: Optional[Mapping[str, Any]] = None) -> int:
        sid = _clean_sid(session_id)
        revision, _row = self._write(sid, None, if_revision=if_revision, actor=actor)
        return revision

    def history(self, session_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
        sid = str(session_id or "").strip()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT revision, at, action, policy, actor FROM session_tool_policy_history "
                "WHERE session_id = ? ORDER BY revision DESC LIMIT ?",
                (sid, max(1, int(limit))),
            ).fetchall()
        finally:
            conn.close()
        out = []
        for revision, at, action, policy, actor in reversed(rows):
            out.append(
                {
                    "revision": int(revision),
                    "at": at,
                    "action": action,
                    "policy": json.loads(policy) if policy else None,
                    "actor": json.loads(actor) if actor else None,
                }
            )
        return out


# --- the runtime side: which policy a run uses, and its audit record ------------------------

TOOL_POLICY_DECISION_EVENT = "abstract.tool_policy_decision"
SESSION_POLICY_DENIED_ERROR = "Denied by this conversation's tool policy"
_AUTOMATION_GRANT_SOURCE = "automation-policy"  # automations.controller.AUTOMATION_GRANT_SOURCE


def _root_session_id(run: Any, run_store: Any) -> str:
    """The session of the run tree's ROOT (the conversation), not a child's own: a flow that
    starts a child with another `session_id` must not borrow that session's grant. Falls back to
    the run's own session when the ancestry cannot be read (no store, a missing parent)."""
    own = str(getattr(run, "session_id", "") or "").strip()
    if run_store is None or not callable(getattr(run_store, "load", None)):
        return own
    current, seen = run, set()
    for _ in range(256):
        pid = str(getattr(current, "parent_run_id", "") or "").strip()
        if not pid or pid in seen:
            break
        seen.add(pid)
        try:
            parent = run_store.load(pid)
        except Exception:  # noqa: BLE001
            parent = None
        if parent is None:
            break
        current = parent
    return str(getattr(current, "session_id", "") or "").strip() or own


def resolve_session_tool_policy(run: Any, *, store: Any, run_store: Any = None) -> Tuple[Optional[Dict[str, Any]], bool]:
    """(policy, approve_applies) for `run`.

    The policy is the row of the run tree's root session (None without a store, a session or a
    row). `deny` applies to every run of the session (narrowing is always safe); `approve`
    (approve_applies) never to an untrusted-input run (it reads text other people wrote) or an
    automation occurrence and its descendants (unattended; the automation's grant is the only
    grant). A store read failure is logged by the caller's logger and yields (None, False): the
    normal decision (ask where the defaults ask), never a wider one.
    """
    if store is None:
        return None, False
    sid = _root_session_id(run, run_store)
    if not sid:
        return None, False
    policy = store.get(sid)
    if not isinstance(policy, dict):
        return None, False
    vars_ = run.vars if isinstance(getattr(run, "vars", None), dict) else {}
    rt = vars_.get("_runtime") if isinstance(vars_.get("_runtime"), dict) else {}
    pol = rt.get("tool_policy") if isinstance(rt.get("tool_policy"), dict) else {}
    meta = vars_.get("_meta") if isinstance(vars_.get("_meta"), dict) else {}
    out_of_scope = (
        rt.get("untrusted_input") is True
        or pol.get("untrusted_input") is True
        or isinstance(meta.get("occurrence"), dict)
        or isinstance(meta.get("automation"), dict)
        or pol.get("source") == _AUTOMATION_GRANT_SOURCE
    )
    return policy, not out_of_scope


def append_tool_policy_record(
    ledger: Any,
    run: Any,
    *,
    decision: str,
    calls: List[Dict[str, Any]],
    matched: Mapping[str, str],
    policy: Mapping[str, Any],
    idempotency_suffix: str,
) -> None:
    """Append one `abstract.tool_policy_decision` record (an EMIT_EVENT record with a namespaced
    name, status completed: the `abstract.steer_seen` shape every ledger reader classifies as an
    activity). Raises on failure: the caller decides what a missing audit means."""
    from ..core.models import Effect, EffectType, StepRecord, StepStatus

    set_by = policy.get("set_by") if isinstance(policy.get("set_by"), Mapping) else None
    rows = []
    for c in calls:
        name = str((c or {}).get("name") or "").strip()
        rows.append({"call_id": str((c or {}).get("call_id") or ""), "name": name, "matched": matched.get(name, "tool")})
    body = {
        "decision": decision,
        "source": "session_policy",
        "session_id": str(policy.get("session_id") or getattr(run, "session_id", "") or ""),
        "node_id": getattr(run, "current_node", None),
        "tool_calls": rows,
        "policy": {
            "revision": policy.get("revision"),
            "set_by": dict(set_by) if set_by else None,
            "set_at": policy.get("set_at"),
        },
    }
    eff = Effect(type=EffectType.EMIT_EVENT, payload={"name": TOOL_POLICY_DECISION_EVENT, "scope": "run", "payload": body})
    rec = StepRecord.start(
        run=run,
        node_id=getattr(run, "current_node", None),
        effect=eff,
        idempotency_key=f"system:tool_policy_decision:{run.run_id}:{idempotency_suffix}:{decision}",
    )
    rec.status = StepStatus.COMPLETED
    rec.result = {"emitted": True, "tool_policy_decision": {"decision": decision, "count": len(rows)}}
    rec.ended_at = _now()
    ledger.append(rec)
