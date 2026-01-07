"""abstractruntime.storage.commands

Durable command inbox primitives (append-only, idempotent).

Why this exists:
- A remote Run Gateway (ADR-0018 / backlog 307) needs a control plane that is safe under
  retries and intermittent networks.
- The key SQS/Temporal insight is to decouple *command acceptance* from *fulfillment*:
  clients submit commands with idempotency keys, and a worker processes them asynchronously.

Design constraints:
- JSON-safe records only (persisted).
- Append-only storage (audit-friendly, replayable).
- Idempotency by `command_id` (duplicate submissions are ignored).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


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


@dataclass(frozen=True)
class CommandRecord:
    """A durable command record.

    Notes:
    - `seq` is assigned by the store and provides cursor semantics for consumers.
    - `payload` must be JSON-serializable (dict of JSON values).
    """

    command_id: str
    run_id: str
    type: str
    payload: Dict[str, Any]
    ts: str
    client_id: Optional[str] = None
    seq: int = 0

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommandAppendResult:
    """Result of appending a command to a CommandStore."""

    accepted: bool
    duplicate: bool
    seq: int


@runtime_checkable
class CommandStore(Protocol):
    """Append-only inbox of commands with cursor replay semantics."""

    def append(self, record: CommandRecord) -> CommandAppendResult:
        """Append a command if its command_id is new (idempotent)."""

    def list_after(self, *, after: int, limit: int = 1000) -> Tuple[List[CommandRecord], int]:
        """Return commands with seq > after, up to limit, and the next cursor."""

    def get_last_seq(self) -> int:
        """Return the greatest assigned sequence number (0 if empty)."""


@runtime_checkable
class CommandCursorStore(Protocol):
    """Durable consumer cursor for CommandStore replay."""

    def load(self) -> int: ...

    def save(self, cursor: int) -> None: ...


class InMemoryCommandStore(CommandStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._by_id: Dict[str, CommandRecord] = {}
        self._ordered: List[CommandRecord] = []

    def append(self, record: CommandRecord) -> CommandAppendResult:
        cid = str(record.command_id or "").strip()
        if not cid:
            cid = uuid.uuid4().hex
        with self._lock:
            existing = self._by_id.get(cid)
            if existing is not None:
                return CommandAppendResult(accepted=False, duplicate=True, seq=int(existing.seq or 0))
            self._seq += 1
            rec = CommandRecord(
                command_id=cid,
                run_id=str(record.run_id or ""),
                type=str(record.type or ""),
                payload=dict(record.payload or {}),
                ts=str(record.ts or _utc_now_iso()),
                client_id=str(record.client_id) if isinstance(record.client_id, str) and record.client_id else None,
                seq=self._seq,
            )
            self._by_id[cid] = rec
            self._ordered.append(rec)
            return CommandAppendResult(accepted=True, duplicate=False, seq=rec.seq)

    def list_after(self, *, after: int, limit: int = 1000) -> Tuple[List[CommandRecord], int]:
        after2 = int(after or 0)
        limit2 = int(limit or 1000)
        if limit2 <= 0:
            limit2 = 1000
        with self._lock:
            items = [r for r in self._ordered if int(r.seq or 0) > after2]
            out = items[:limit2]
            next_cursor = after2
            if out:
                next_cursor = int(out[-1].seq or after2)
            return (list(out), next_cursor)

    def get_last_seq(self) -> int:
        with self._lock:
            return int(self._seq or 0)


class InMemoryCommandCursorStore(CommandCursorStore):
    def __init__(self, initial: int = 0) -> None:
        self._cursor = int(initial or 0)
        self._lock = threading.Lock()

    def load(self) -> int:
        with self._lock:
            return int(self._cursor or 0)

    def save(self, cursor: int) -> None:
        with self._lock:
            self._cursor = int(cursor or 0)


class JsonFileCommandCursorStore(CommandCursorStore):
    """JSON file-backed cursor store.

    Atomic write semantics are important because this file is updated frequently.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self) -> int:
        with self._lock:
            if not self._path.exists():
                return 0
            try:
                data = json.loads(self._path.read_text(encoding="utf-8") or "{}")
            except Exception:
                return 0
            cur = data.get("cursor")
            try:
                return int(cur or 0)
            except Exception:
                return 0

    def save(self, cursor: int) -> None:
        cur = int(cursor or 0)
        tmp = self._path.with_name(f"{self._path.name}.{uuid.uuid4().hex}.tmp")
        payload = {"cursor": cur, "updated_at": _utc_now_iso()}
        with self._lock:
            try:
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(self._path)
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass


class JsonlCommandStore(CommandStore):
    """Append-only JSONL command store.

    File format: one JSON object per line. Each record includes a store-assigned `seq`.
    """

    def __init__(self, base_dir: str | Path, *, filename: str = "commands.jsonl") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._path = self._base / str(filename or "commands.jsonl")
        self._lock = threading.Lock()

        # Idempotency index (rebuilt on init from the log).
        self._seq = 0
        self._by_id: Dict[str, int] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self._path.exists():
            self._seq = 0
            self._by_id = {}
            return
        seq = 0
        by_id: Dict[str, int] = {}
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    s = obj.get("seq")
                    cid = obj.get("command_id")
                    try:
                        s_int = int(s or 0)
                    except Exception:
                        s_int = 0
                    if s_int <= 0:
                        continue
                    if s_int > seq:
                        seq = s_int
                    if isinstance(cid, str) and cid and cid not in by_id:
                        by_id[cid] = s_int
        except Exception:
            seq = 0
            by_id = {}
        self._seq = seq
        self._by_id = by_id

    def append(self, record: CommandRecord) -> CommandAppendResult:
        cid = str(record.command_id or "").strip()
        if not cid:
            cid = uuid.uuid4().hex
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

        with self._lock:
            existing_seq = self._by_id.get(cid)
            if existing_seq is not None:
                return CommandAppendResult(accepted=False, duplicate=True, seq=int(existing_seq))

            self._seq += 1
            seq = int(self._seq)
            rec = CommandRecord(
                command_id=cid,
                run_id=run_id,
                type=typ,
                payload=payload,
                ts=ts,
                client_id=client_id,
                seq=seq,
            )
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_json(), ensure_ascii=False))
                f.write("\n")
            self._by_id[cid] = seq
            return CommandAppendResult(accepted=True, duplicate=False, seq=seq)

    def list_after(self, *, after: int, limit: int = 1000) -> Tuple[List[CommandRecord], int]:
        after2 = int(after or 0)
        limit2 = int(limit or 1000)
        if limit2 <= 0:
            limit2 = 1000

        if not self._path.exists():
            return ([], after2)

        out: List[CommandRecord] = []
        next_cursor = after2
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    try:
                        seq = int(obj.get("seq") or 0)
                    except Exception:
                        continue
                    if seq <= after2:
                        continue
                    try:
                        rec = CommandRecord(
                            command_id=str(obj.get("command_id") or ""),
                            run_id=str(obj.get("run_id") or ""),
                            type=str(obj.get("type") or ""),
                            payload=dict(obj.get("payload") or {}),
                            ts=str(obj.get("ts") or ""),
                            client_id=str(obj.get("client_id") or "") or None,
                            seq=seq,
                        )
                    except Exception:
                        continue
                    out.append(rec)
                    next_cursor = seq
                    if len(out) >= limit2:
                        break
        except Exception:
            return ([], after2)
        return (out, next_cursor)

    def get_last_seq(self) -> int:
        with self._lock:
            return int(self._seq or 0)


