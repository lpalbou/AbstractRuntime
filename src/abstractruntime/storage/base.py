"""abstractruntime.storage.base

Storage interfaces (durability backends).

These are intentionally minimal for v0.1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ..core.models import RunState, RunStatus, StepRecord, StepStatus, WaitReason


class RunStore(ABC):
    @abstractmethod
    def save(self, run: RunState) -> None: ...

    @abstractmethod
    def load(self, run_id: str) -> Optional[RunState]: ...

    def create_if_absent(self, run: RunState) -> Tuple[RunState, bool]:
        """Atomically create `run` unless a run with its id exists.

        Returns `(run, True)` when this call created it, `(existing, False)`
        when the id was taken by a run with the same creation identity
        (`core.run_identity.verify_run_identity`), and raises
        `RunIdentityConflict` otherwise. Never overwrites, never reseeds.

        Mandatory for explicit-id starts (`Runtime.start(run_id=...)`,
        `START_SUBWORKFLOW.payload.run_id`). The base class does NOT emulate
        it with load-then-save (that is a race, not a primitive): stores
        without a real implementation raise, and callers preflight with
        `store_supports_create_if_absent` / `require_create_if_absent`.

        Durability claim: process-crash recovery (the run is either fully
        present or absent). Power-loss durability would additionally need
        file and directory fsync and is not claimed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement create_if_absent "
            "(required for runs started with an explicit run_id)"
        )

    def supports_create_if_absent(self) -> bool:
        """True when this store (through any wrapped store) implements `create_if_absent`."""
        return type(self).create_if_absent is not RunStore.create_if_absent


def store_supports_create_if_absent(store: Any) -> bool:
    """Capability preflight for `create_if_absent` (sees through wrappers)."""
    probe = getattr(store, "supports_create_if_absent", None)
    return bool(probe()) if callable(probe) else False


def require_create_if_absent(store: Any) -> None:
    """Raise `NotImplementedError` unless `store` supports `create_if_absent`."""
    if not store_supports_create_if_absent(store):
        raise NotImplementedError(
            f"run store {type(store).__name__} does not support create_if_absent; "
            "explicit-id starts (automations, discussions) need a store that does"
        )


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
class EventWaiterQueryableRunStore(Protocol):
    """Optional fast-path for `emit_event`'s listener lookup.

    Without it, delivering an event means `list_runs(status=WAITING,
    wait_reason=EVENT, limit=big)` — a whole-store scan per emit (57ms at
    9,326 run files, twice per agent chat turn, on a store whose event-waiter
    count is usually zero). A store implementing this answers in O(waiters).

    CONTRACT: `list_event_waiters(wait_keys=K)` must return exactly the runs
    `list_runs(status=WAITING, wait_reason=EVENT, limit=big)` would return
    whose `waiting.wait_key` is in K, in the same `updated_at`-descending
    order. Callers keep their own run-level filters (paused, workflow
    resolvable, ...); this is a lookup, not a policy.
    """

    def list_event_waiters(self, *, wait_keys: List[str], limit: int = 100) -> List[RunState]:
        """Runs parked in WAITING(EVENT) on any of `wait_keys`."""
        ...

    def list_event_waiters_by_prefix(self, *, prefix: str, limit: int = 100) -> List[RunState]:
        """Runs parked in WAITING(EVENT) on a wait_key starting with `prefix`."""
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
