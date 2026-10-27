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


def mailbox_names_of(vars_obj: Any) -> frozenset:
    """Mailbox names a run declares in `vars.events_mailbox` (str or list of str)."""
    if not isinstance(vars_obj, dict):
        return frozenset()
    declared = vars_obj.get("events_mailbox")
    if isinstance(declared, str):
        declared = [declared]
    if not isinstance(declared, list):
        return frozenset()
    return frozenset(m.strip() for m in declared if isinstance(m, str) and m.strip())


@runtime_checkable
class MailboxQueryableRunStore(Protocol):
    """Durable `emit_event` fan-out and the scheduler's sleep horizon without
    a recency window (backlog 1046, design round-1 F3).

    `list_mailbox_runs(names=N)` returns EVERY active (RUNNING/WAITING) run
    whose `events_mailbox` names one of N, `updated_at` descending.
    `next_wait_deadline()` returns the earliest `waiting.until` among WAITING
    runs (UNTIL, or EVENT with a deadline), or None.
    """

    def list_mailbox_runs(self, *, names: List[str], limit: int = 10_000) -> List[RunState]:
        ...

    def next_wait_deadline(self) -> Optional[str]:
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
        automation_id: Any = None,
        role: Any = None,
        session_kind: Any = None,
    ) -> List[Dict[str, Any]]:
        """List lightweight run index rows (most recent first by default;
        `oldest_first=True` inverts the order — required for stall queries,
        where a newest-first window silently hides the oldest waits).

        Rows carry `automation_id`, `role`, `occurrence_index` and
        `session_kind` (`core.run_attribution`). `automation_id` / `role` /
        `session_kind` filter on them (a value, a comma-separated string or a
        list). `root_only=True` returns TURN ROOTS: parent-less runs that are
        not automation controllers, plus automation occurrence runs — so a
        client folding root rows by session shows an automation's session as
        a chat whose turns are its occurrences.
        """
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
    def append(self, record: StepRecord) -> Optional[int]:
        """Append one record. Stores that number records return its per-run `seq`."""

    @abstractmethod
    def list(self, run_id: str) -> List[Dict[str, Any]]:
        """The run's records in append order. Each carries `seq` (1, 2, ...: the durable cursor,
        ADR-0010) on the stores that number records (JSONL, SQLite, in-memory)."""

    def list_after(self, *, run_id: str, after: int, limit: int = 1000) -> Tuple[List[Dict[str, Any]], int]:
        """Records with `seq` > `after` (at most `limit`) and the last seq returned (or `after`).

        `after` is the last seq a reader consumed (0 = from the start). Default: `list()` then
        filter; JSONL and SQLite override it with incremental reads."""
        a = max(0, int(after or 0))
        lim = max(1, int(limit or 1000))
        out: List[Dict[str, Any]] = []
        prev = 0
        for rec in self.list(run_id):
            if not isinstance(rec, dict):
                continue
            seq = rec.get("seq")
            seq = int(seq) if isinstance(seq, int) and not isinstance(seq, bool) and seq > prev else prev + 1
            prev = seq
            if seq <= a:
                continue
            if len(out) >= lim:
                break
            out.append(rec if rec.get("seq") == seq else {**rec, "seq": seq})
        return (out, int(out[-1]["seq"]) if out else a)

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
