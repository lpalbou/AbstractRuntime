"""abstractruntime.storage.in_memory

In-memory durability backends (testing/dev).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .base import LedgerStore, RunStore
from ..core.models import RunState, RunStatus, StepRecord, WaitReason


class InMemoryRunStore(RunStore):
    """In-memory run store with query support.

    Implements both RunStore (ABC) and QueryableRunStore (Protocol).
    """

    def __init__(self):
        self._runs: Dict[str, RunState] = {}

    def save(self, run: RunState) -> None:
        # store a shallow copy to avoid accidental mutation surprises
        self._runs[run.run_id] = run

    def load(self, run_id: str) -> Optional[RunState]:
        return self._runs.get(run_id)

    # --- QueryableRunStore methods ---

    def list_runs(
        self,
        *,
        status: Optional[RunStatus] = None,
        wait_reason: Optional[WaitReason] = None,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RunState]:
        """List runs matching the given filters."""
        results: List[RunState] = []

        for run in self._runs.values():
            # Apply filters
            if status is not None and run.status != status:
                continue
            if workflow_id is not None and run.workflow_id != workflow_id:
                continue
            if wait_reason is not None:
                if run.waiting is None or run.waiting.reason != wait_reason:
                    continue

            results.append(run)

        # Sort by updated_at descending (most recent first)
        results.sort(key=lambda r: r.updated_at or "", reverse=True)

        return results[:limit]

    def list_due_wait_until(
        self,
        *,
        now_iso: str,
        limit: int = 100,
    ) -> List[RunState]:
        """List runs waiting for a time threshold that has passed."""
        results: List[RunState] = []

        for run in self._runs.values():
            # Must be WAITING with reason UNTIL
            if run.status != RunStatus.WAITING:
                continue
            if run.waiting is None:
                continue
            if run.waiting.reason != WaitReason.UNTIL:
                continue
            if run.waiting.until is None:
                continue

            # Check if the wait time has passed (ISO string comparison works for UTC)
            if run.waiting.until <= now_iso:
                results.append(run)

        # Sort by waiting.until ascending (earliest due first)
        results.sort(key=lambda r: r.waiting.until if r.waiting else "")

        return results[:limit]

    def list_children(
        self,
        *,
        parent_run_id: str,
        status: Optional[RunStatus] = None,
    ) -> List[RunState]:
        """List child runs of a parent."""
        results: List[RunState] = []

        for run in self._runs.values():
            if run.parent_run_id != parent_run_id:
                continue
            if status is not None and run.status != status:
                continue
            results.append(run)

        return results


class InMemoryLedgerStore(LedgerStore):
    def __init__(self):
        self._records: Dict[str, List[Dict[str, Any]]] = {}

    def append(self, record: StepRecord) -> None:
        self._records.setdefault(record.run_id, []).append(asdict(record))

    def list(self, run_id: str) -> List[Dict[str, Any]]:
        return list(self._records.get(run_id, []))


