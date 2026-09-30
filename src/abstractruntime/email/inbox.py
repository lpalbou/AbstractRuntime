"""Durable external-event inbox (framework backlog 0929, the minimal part 0992 needs).

Events that happen outside the runtime (today: a new email) are appended here by a watcher
BEFORE any automation sees them, so a busy or restarting controller never misses one:

- append-only, ordered by a per-inbox sequence number (`seq`, 1, 2, ...);
- unique on `event_id`: appending an id that is already present is a no-op that returns the
  existing record (`duplicate=True`); the receipts are kept after consumption;
- an optional per-stream `dedupe_key` (the Message-ID of an email) lets a watcher skip a message
  it already appended under another id (after a server rebuilt a folder);
- per-stream state (a watcher's mail cursor and health) is stored next to the events;
- consumers keep their OWN cursor (the highest `seq` they consumed) in their own durable state
  (an automation keeps it in `_runtime.automation.source_state`).

Two stores: `InMemoryEventInbox` (tests, ephemeral hosts) and `JsonFileEventInbox(base_dir)`
(one JSON file per event, an index written atomically after the event file; a crash between
the two is repaired on the next open by re-indexing the event files past the index).

Retention (`EventInboxRetention`, typed; default: keep 90 days AND at most 10,000 events):
`prune(retention=..., protect_after_seq=...)` removes the stored events (their payloads, the
mail bodies) that are older than `keep_days` or beyond the newest `keep_events`, never one a
consumer has not read yet (`seq > protect_after_seq`). The id/dedupe receipts stay in the
index, so a pruned message is still never appended twice. Hosts call
`abstractruntime.email.prune_email_inbox(runtime)`, which computes `protect_after_seq` from the
active email automations' cursors.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

try:  # POSIX: serialize writers across processes too
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - Windows
    fcntl = None  # type: ignore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AppendResult:
    seq: int
    event_id: str
    duplicate: bool
    record: Dict[str, Any]


DEFAULT_KEEP_DAYS = 90
DEFAULT_KEEP_EVENTS = 10_000
MAX_KEEP_DAYS = 3650
MAX_KEEP_EVENTS = 1_000_000


@dataclass(frozen=True)
class EventInboxRetention:
    """How long received events are kept: at most `keep_days` days AND `keep_events` events."""

    keep_days: int = DEFAULT_KEEP_DAYS
    keep_events: int = DEFAULT_KEEP_EVENTS

    def __post_init__(self) -> None:
        for name, value, top in (("keep_days", self.keep_days, MAX_KEEP_DAYS), ("keep_events", self.keep_events, MAX_KEEP_EVENTS)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= top:
                raise ValueError(f"EventInboxRetention.{name} must be an integer 1..{top}, got {value!r}")

    def to_dict(self) -> Dict[str, int]:
        return {"keep_days": self.keep_days, "keep_events": self.keep_events}

    @classmethod
    def from_value(cls, value: Any) -> "EventInboxRetention":
        """From an `EventInboxRetention`, a `{keep_days?, keep_events?}` dict, or None (default)."""
        if value is None:
            return cls()
        if isinstance(value, EventInboxRetention):
            return value
        if not isinstance(value, dict):
            raise TypeError("retention must be an EventInboxRetention, a {keep_days, keep_events} dict, or None")
        unknown = sorted(k for k in value if k not in ("keep_days", "keep_events"))
        if unknown:
            raise ValueError(f"retention has unknown field(s): {unknown}")
        return cls(**value)


def _prunable(records: List[Dict[str, Any]], *, head: int, retention: EventInboxRetention,
              protect_after_seq: Optional[int], now: Optional[datetime]) -> List[int]:
    """The seqs to remove among `records` (each {seq, appended_at})."""
    from datetime import timedelta

    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=retention.keep_days)
    oldest_kept_by_count = head - retention.keep_events  # seq <= this is beyond the newest N
    out: List[int] = []
    for rec in records:
        seq = int(rec.get("seq") or 0)
        if protect_after_seq is not None and seq > int(protect_after_seq):
            continue  # a consumer has not read it yet
        too_many = seq <= oldest_kept_by_count
        too_old = False
        try:
            at = datetime.fromisoformat(str(rec.get("appended_at")))
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
            too_old = at < cutoff
        except ValueError:
            too_old = False  # an unreadable date is kept (the count limit still applies)
        if too_many or too_old:
            out.append(seq)
    return out


@runtime_checkable
class EventInbox(Protocol):
    def append(
        self,
        *,
        stream: str,
        event_id: str,
        payload: Dict[str, Any],
        dedupe_key: Optional[str] = None,
        appended_at: Optional[str] = None,
    ) -> AppendResult: ...

    def get(self, event_id: str) -> Optional[Dict[str, Any]]: ...

    def read(self, *, after_seq: int = 0, stream: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

    def head_seq(self) -> int: ...

    def has_dedupe_key(self, stream: str, key: str) -> bool: ...

    def stream_state(self, stream: str) -> Dict[str, Any]: ...

    def set_stream_state(self, stream: str, state: Dict[str, Any]) -> None: ...


def _check_event(stream: str, event_id: str, payload: Any) -> None:
    if not isinstance(stream, str) or not stream.strip():
        raise ValueError("stream must be a non-empty string")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    json.dumps(payload)  # JSON-safe or raise


class InMemoryEventInbox:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: List[Dict[str, Any]] = []
        self._by_id: Dict[str, int] = {}
        self._dedupe: Dict[str, Dict[str, int]] = {}
        self._streams: Dict[str, Dict[str, Any]] = {}

    def append(self, *, stream, event_id, payload, dedupe_key=None, appended_at=None) -> AppendResult:
        _check_event(stream, event_id, payload)
        with self._lock:
            seq = self._by_id.get(event_id)
            if seq is not None:
                rec = self._events[seq - 1] or {"seq": seq, "event_id": event_id, "stream": stream, "pruned": True}
                return AppendResult(seq=seq, event_id=event_id, duplicate=True, record=copy.deepcopy(rec))
            seq = len(self._events) + 1
            rec = {
                "seq": seq,
                "event_id": event_id,
                "stream": stream,
                "appended_at": appended_at or _now_iso(),
                "payload": copy.deepcopy(payload),
            }
            if dedupe_key:
                rec["dedupe_key"] = dedupe_key
                self._dedupe.setdefault(stream, {})[dedupe_key] = seq
            self._events.append(rec)
            self._by_id[event_id] = seq
            return AppendResult(seq=seq, event_id=event_id, duplicate=False, record=copy.deepcopy(rec))

    def get(self, event_id):
        with self._lock:
            seq = self._by_id.get(str(event_id))
            return copy.deepcopy(self._events[seq - 1]) if seq and self._events[seq - 1] else None

    def read(self, *, after_seq=0, stream=None, limit=None):
        with self._lock:
            out = [
                copy.deepcopy(r) for r in self._events[max(0, int(after_seq)):]
                if r is not None and (stream is None or r["stream"] == stream)
            ]
        return out[: int(limit)] if limit else out

    def prune(self, *, retention: Any = None, protect_after_seq: Optional[int] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Remove stored events per `retention` (see the module doc); returns a report."""
        policy = EventInboxRetention.from_value(retention)
        with self._lock:
            live = [r for r in self._events if r is not None]
            gone = _prunable(live, head=len(self._events), retention=policy, protect_after_seq=protect_after_seq, now=now)
            for seq in gone:
                self._events[seq - 1] = None  # the id/dedupe receipts stay
            return {"removed": len(gone), "kept": len(live) - len(gone), "head_seq": len(self._events), "retention": policy.to_dict()}

    def head_seq(self) -> int:
        with self._lock:
            return len(self._events)

    def has_dedupe_key(self, stream, key) -> bool:
        with self._lock:
            return bool(key) and key in self._dedupe.get(stream, {})

    def stream_state(self, stream):
        with self._lock:
            return copy.deepcopy(self._streams.get(stream) or {})

    def set_stream_state(self, stream, state):
        json.dumps(state)
        with self._lock:
            self._streams[stream] = copy.deepcopy(dict(state))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:  # pragma: no cover - platforms without directory fsync
        pass


class JsonFileEventInbox:
    """File-backed inbox under `base_dir` (created 0700; files 0600)."""

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.base_dir, 0o700)
        except OSError:  # pragma: no cover
            pass
        self._events_dir = self.base_dir / "events"
        self._streams_dir = self.base_dir / "streams"
        self._events_dir.mkdir(exist_ok=True)
        self._streams_dir.mkdir(exist_ok=True)
        self._index_path = self.base_dir / "index.json"
        self._lock_path = self.base_dir / ".lock"
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"JsonFileEventInbox({str(self.base_dir)!r})"

    # -- locking / index -------------------------------------------------------------------

    class _FileLock:
        def __init__(self, inbox: "JsonFileEventInbox") -> None:
            self.inbox = inbox
            self.fh = None

        def __enter__(self):
            self.inbox._lock.acquire()
            if fcntl is not None:
                self.fh = open(self.inbox._lock_path, "a+")
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, *exc):
            try:
                if self.fh is not None:
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
                    self.fh.close()
            finally:
                self.inbox._lock.release()

    def _event_path(self, seq: int) -> Path:
        return self._events_dir / f"{int(seq):012d}.json"

    def _load_index(self) -> Dict[str, Any]:
        try:
            idx = json.loads(self._index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            idx = {}
        idx.setdefault("next_seq", 1)
        idx.setdefault("event_ids", {})
        idx.setdefault("dedupe", {})
        # Repair: event files written after the last index save (a crash in between).
        repaired = False
        while True:
            path = self._event_path(idx["next_seq"])
            if not path.is_file():
                break
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                break  # a torn write never happens (atomic replace); stop defensively
            idx["event_ids"][rec["event_id"]] = rec["seq"]
            if rec.get("dedupe_key"):
                idx["dedupe"].setdefault(rec["stream"], {})[rec["dedupe_key"]] = rec["seq"]
            idx["next_seq"] = int(rec["seq"]) + 1
            repaired = True
        if repaired:
            _atomic_write_json(self._index_path, idx)
        return idx

    # -- API -------------------------------------------------------------------------------

    def append(self, *, stream, event_id, payload, dedupe_key=None, appended_at=None) -> AppendResult:
        _check_event(stream, event_id, payload)
        with self._FileLock(self):
            idx = self._load_index()
            existing = idx["event_ids"].get(event_id)
            if existing is not None:
                path = self._event_path(existing)
                if path.is_file():
                    rec = json.loads(path.read_text(encoding="utf-8"))
                else:  # pruned by retention: the receipt proves it was received
                    rec = {"seq": int(existing), "event_id": event_id, "stream": stream, "pruned": True}
                return AppendResult(seq=int(existing), event_id=event_id, duplicate=True, record=rec)
            seq = int(idx["next_seq"])
            rec = {
                "seq": seq,
                "event_id": event_id,
                "stream": stream,
                "appended_at": appended_at or _now_iso(),
                "payload": payload,
            }
            if dedupe_key:
                rec["dedupe_key"] = dedupe_key
            # The event file first, then the index: the event is durable once its file is.
            _atomic_write_json(self._event_path(seq), rec)
            idx["event_ids"][event_id] = seq
            if dedupe_key:
                idx["dedupe"].setdefault(stream, {})[dedupe_key] = seq
            idx["next_seq"] = seq + 1
            _atomic_write_json(self._index_path, idx)
            return AppendResult(seq=seq, event_id=event_id, duplicate=False, record=copy.deepcopy(rec))

    def get(self, event_id):
        with self._FileLock(self):
            seq = self._load_index()["event_ids"].get(str(event_id))
            if seq is None or not self._event_path(seq).is_file():
                return None
            return json.loads(self._event_path(seq).read_text(encoding="utf-8"))

    def read(self, *, after_seq=0, stream=None, limit=None):
        with self._FileLock(self):
            head = int(self._load_index()["next_seq"]) - 1
            out: List[Dict[str, Any]] = []
            for seq in range(max(0, int(after_seq)) + 1, head + 1):
                path = self._event_path(seq)
                if not path.is_file():
                    continue
                rec = json.loads(path.read_text(encoding="utf-8"))
                if stream is not None and rec.get("stream") != stream:
                    continue
                out.append(rec)
                if limit and len(out) >= int(limit):
                    break
            return out

    def head_seq(self) -> int:
        with self._FileLock(self):
            return int(self._load_index()["next_seq"]) - 1

    def prune(self, *, retention: Any = None, protect_after_seq: Optional[int] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Delete stored event files per `retention` (see the module doc); returns a report.

        The index keeps `next_seq` and the id/dedupe receipts, so sequence numbers never repeat
        and a pruned message is never appended again."""
        policy = EventInboxRetention.from_value(retention)
        with self._FileLock(self):
            head = int(self._load_index()["next_seq"]) - 1
            records: List[Dict[str, Any]] = []
            for path in sorted(self._events_dir.glob("*.json")):
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                records.append({"seq": rec.get("seq"), "appended_at": rec.get("appended_at")})
            gone = _prunable(records, head=head, retention=policy, protect_after_seq=protect_after_seq, now=now)
            for seq in gone:
                try:
                    self._event_path(seq).unlink()
                except FileNotFoundError:
                    pass
            return {"removed": len(gone), "kept": len(records) - len(gone), "head_seq": head, "retention": policy.to_dict()}

    def has_dedupe_key(self, stream, key) -> bool:
        if not key:
            return False
        with self._FileLock(self):
            return key in (self._load_index()["dedupe"].get(stream) or {})

    def _stream_path(self, stream: str) -> Path:
        return self._streams_dir / (hashlib.sha256(stream.encode("utf-8")).hexdigest()[:32] + ".json")

    def stream_state(self, stream):
        with self._FileLock(self):
            try:
                raw = json.loads(self._stream_path(stream).read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {}
            return dict(raw.get("state") or {})

    def set_stream_state(self, stream, state):
        with self._FileLock(self):
            _atomic_write_json(self._stream_path(stream), {"stream": stream, "state": dict(state)})


__all__ = [
    "AppendResult",
    "DEFAULT_KEEP_DAYS",
    "DEFAULT_KEEP_EVENTS",
    "EventInbox",
    "EventInboxRetention",
    "InMemoryEventInbox",
    "JsonFileEventInbox",
]
