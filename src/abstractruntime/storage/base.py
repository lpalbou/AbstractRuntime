"""abstractruntime.storage.base

Storage interfaces (durability backends).

These are intentionally minimal for v0.1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..core.models import RunState, RunStatus, StepRecord, StepStatus, WaitReason


class RunStore(ABC):
    @abstractmethod
    def save(self, run: RunState) -> None: ...

    @abstractmethod
    def load(self, run_id: str) -> Optional[RunState]: ...


@runtime_checkable
class QueryableRunStore(Protocol):
    """Extended interface for querying runs.

    This is a Protocol (structural typing) so existing RunStore implementations
    can add these methods without changing their inheritance.

    Used by:
    - Scheduler/driver loops (find due wait_until runs)
    - Operational tooling (list waiting runs)
    - UI backoffice views (runs by status)
    """

    def list_runs(
        self,
        *,
        status: Optional[RunStatus] = None,
        wait_reason: Optional[WaitReason] = None,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RunState]:
        """List runs matching the given filters.

        Args:
            status: Filter by run status (RUNNING, WAITING, COMPLETED, FAILED)
            wait_reason: Filter by wait reason (only applies to WAITING runs)
            workflow_id: Filter by workflow ID
            limit: Maximum number of runs to return

        Returns:
            List of matching RunState objects, ordered by updated_at descending
        """
        ...

    def list_due_wait_until(
        self,
        *,
        now_iso: str,
        limit: int = 100,
    ) -> List[RunState]:
        """List runs whose wait DEADLINE has passed.

        This finds runs where:
        - status == WAITING
        - waiting.reason == UNTIL, or waiting.reason == EVENT with a
          deadline (`waiting.until` set — the D3 idle-timeout shape:
          an event wait that times out into its resume path)
        - waiting.until <= now_iso

        Args:
            now_iso: Current time as ISO 8601 string
            limit: Maximum number of runs to return

        Returns:
            List of due RunState objects, ordered by waiting.until ascending
        """
        ...

    def list_children(
        self,
        *,
        parent_run_id: str,
        status: Optional[RunStatus] = None,
    ) -> List[RunState]:
        """List child runs of a parent.

        Args:
            parent_run_id: The parent run ID
            status: Optional filter by status

        Returns:
            List of child RunState objects
        """
        ...


@runtime_checkable
class QueryableRunIndexStore(Protocol):
    """Optional fast-path for listing run summaries without loading full RunState payloads."""

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
        """List lightweight run index rows (most recent first by default;
        `oldest_first=True` inverts the order — required for stall queries,
        where a newest-first window silently hides the oldest waits)."""
        ...



@runtime_checkable
class DeletableRunStore(Protocol):
    """Optional RunStore extension for deleting one run checkpoint."""

    def delete(self, run_id: str) -> bool:
        """Delete a run checkpoint.

        Returns True when a run existed and was removed, False when it was absent.
        """
        ...


# Bounded lookup window for idempotency dedup on stores without an index
# (backlog 0047). A prior COMPLETED record for the CURRENT effect issuance
# can only exist within the current step's own records — near the ledger
# tail by construction (crash-replay = "effect completed, the save after it
# did not land"). Issuance-scoped keys (`_runtime.effect_seq`) make hits
# outside the tail impossible going forward; the window is generous padding
# for progress events and legacy mid-flight runs.
IDEMPOTENCY_TAIL_WINDOW = 256


class LedgerStore(ABC):
    """Append-only journal store."""

    @abstractmethod
    def append(self, record: StepRecord) -> None: ...

    @abstractmethod
    def list(self, run_id: str) -> List[Dict[str, Any]]: ...

    def find_completed_result(
        self, run_id: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]:
        """Prior COMPLETED result for an idempotency key, or None.

        Default: bounded scan over the tail of `list()` (correct for any
        conforming store; oldest-first within the window for parity with
        the historical full scan). Backends override with indexed lookups
        (SQLite point query) or bounded tail reads (JSONL) so the per-step
        dedup probe stops scaling with ledger length (backlog 0047).
        """
        key = str(idempotency_key or "")
        if not key:
            return None
        records = self.list(run_id)
        window = records[-IDEMPOTENCY_TAIL_WINDOW:]
        for record in window:
            if not isinstance(record, dict):
                continue
            if record.get("idempotency_key") != key:
                continue
            if record.get("status") == StepStatus.COMPLETED.value:
                return record.get("result")
        return None


@runtime_checkable
class DeletableLedgerStore(Protocol):
    """Optional LedgerStore extension for deleting one run ledger."""

    def delete(self, run_id: str) -> int:
        """Delete ledger records for one run and return the number removed."""
        ...
