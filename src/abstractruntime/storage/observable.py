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
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .base import LedgerStore
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

    def append(self, record: StepRecord) -> None:
        self._inner.append(record)

        payload: LedgerRecordDict = asdict(record)
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

    def list(self, run_id: str) -> List[LedgerRecordDict]:
        return self._inner.list(run_id)

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

