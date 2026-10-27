"""abstractruntime.storage.observable

In-process pub/sub for ledger append events.

Design intent:
- Keep AbstractRuntime kernel dependency-light (stdlib only).
- Treat the ledger as the durable source of truth (replay via `list(run_id)`).
- Provide a small, optional subscription surface for live UX (WS/CLI) without
  introducing a second global event system.

This is intentionally *process-local* (no cross-process guarantees).
"""

from __future__ import annotations

from dataclasses import asdict
import threading
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .base import LedgerStore
from .change_ring import ChangeRing
from ..core.models import StepRecord


LedgerRecordDict = Dict[str, Any]
LedgerSubscriber = Callable[[LedgerRecordDict], None]


@runtime_checkable
class ObservableLedgerStoreProtocol(Protocol):
    """Optional LedgerStore extension for in-process subscriptions."""

    def subscribe(
        self,
        callback: LedgerSubscriber,
        *,
        run_id: Optional[str] = None,
    ) -> Callable[[], None]:
        """Subscribe to appended ledger records.

        Args:
            callback: Called after each append with a JSON-safe record dict.
            run_id: If set, only receive records for that run.

        Returns:
            An unsubscribe callback.
        """
        ...


class ObservableLedgerStore(LedgerStore):
    """LedgerStore decorator that notifies subscribers on append()."""

    def __init__(self, inner: LedgerStore):
        self._inner = inner
        self._lock = threading.Lock()
        self._subscribers: list[tuple[Optional[str], LedgerSubscriber]] = []
        # CHANGE SIGNAL (2026-10-26): a counter every append through this store bumps, and the
        # run ids behind its last bumps, so a periodic reader (the gateway's mail notification
        # check) learns which ledgers grew without re-reading them. Process-local, like the
        # subscriptions.
        self._change_lock = threading.Lock()
        self._change_seq = 0
        self._change_ring = ChangeRing()

    def change_token(self) -> int:
        """A counter every append through this store moves (process-local)."""
        with self._change_lock:
            return self._change_seq

    def changed_run_ids(self, since: Any) -> Optional[set]:
        """The run ids whose ledger this store appended to after token `since`; None when the
        record no longer reaches back that far."""
        try:
            seq0 = int(since)
        except Exception:
            return None
        with self._change_lock:
            return self._change_ring.since(seq0)

    def append(self, record: StepRecord) -> Optional[int]:
        seq = self._inner.append(record)
        with self._change_lock:
            self._change_seq += 1
            self._change_ring.note(self._change_seq, str(record.run_id))

        payload: LedgerRecordDict = asdict(record)
        if isinstance(seq, int) and not isinstance(seq, bool):
            payload["seq"] = seq  # subscribers fold by the durable cursor (ADR-0009)
        with self._lock:
            subscribers = list(self._subscribers)

        for run_id, callback in subscribers:
            if run_id is not None and run_id != record.run_id:
                continue
            try:
                callback(payload)
            except Exception:
                # Observability must never compromise durability/execution.
                continue
        return seq

    def list(self, run_id: str) -> List[LedgerRecordDict]:
        return self._inner.list(run_id)

    def list_after(self, *, run_id: str, after: int, limit: int = 1000) -> Tuple[List[LedgerRecordDict], int]:
        return self._inner.list_after(run_id=run_id, after=after, limit=limit)

    def find_completed_result(
        self, run_id: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]:
        """Delegate to the inner store's (possibly indexed) lookup (0047)."""
        fn = getattr(self._inner, "find_completed_result", None)
        if callable(fn):
            return fn(run_id, idempotency_key)
        return super().find_completed_result(run_id, idempotency_key)

    def delete(self, run_id: str) -> int:
        fn = getattr(self._inner, "delete", None)
        if not callable(fn):
            raise NotImplementedError("Inner LedgerStore does not support delete")
        return int(fn(run_id))

    def count(self, run_id: str) -> int:
        """Best-effort record count for run_id.

        When the inner LedgerStore implements a faster `count()` method (e.g. JsonlLedgerStore),
        delegate to it. Otherwise, fall back to `len(list(run_id))`.
        """
        inner = self._inner
        try:
            count_fn = getattr(inner, "count", None)
            if callable(count_fn):
                return int(count_fn(run_id))
        except Exception:
            pass
        try:
            return len(inner.list(run_id))
        except Exception:
            return 0

    def count_many(self, run_ids: List[str]) -> Dict[str, int]:  # type: ignore[override]
        fn = getattr(self._inner, "count_many", None)
        if callable(fn):
            try:
                out = fn(run_ids)
                return out if isinstance(out, dict) else {}
            except Exception:
                return {}
        return {str(r or "").strip(): self.count(str(r or "").strip()) for r in (run_ids or []) if str(r or "").strip()}

    def metrics_many(self, run_ids: List[str]) -> Dict[str, Dict[str, int]]:  # type: ignore[override]
        fn = getattr(self._inner, "metrics_many", None)
        if callable(fn):
            try:
                out = fn(run_ids)
                return out if isinstance(out, dict) else {}
            except Exception:
                return {}
        return {}

    def subscribe(
        self,
        callback: LedgerSubscriber,
        *,
        run_id: Optional[str] = None,
    ) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append((run_id, callback))

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove((run_id, callback))
                except ValueError:
                    return

        return _unsubscribe

    def clear_subscribers(self) -> None:
        """Clear all subscribers (test utility)."""
        with self._lock:
            self._subscribers.clear()
