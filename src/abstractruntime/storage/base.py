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


def coerce_ledger_seq(seq: Any) -> Optional[int]:
    """A ledger row address component: a positive int, else None (bools refused)."""
    if isinstance(seq, bool):
        return None
    try:
        value = int(seq)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


class LedgerStore(ABC):
    """Append-only journal store.

    LEDGER ROW ADDRESS (ADR 0009): one record is addressed by `(run_id, seq)`,
    `seq` being its 1-based position in the run's ledger. `step_id` is not an
    address: a step's STARTED row and its terminal row share it. Backends with
    a stored sequence (SQLite) use it; the JSON-file backend's `seq` is the
    record's position in its file, i.e. in the order `list()` returns. `seq`
    is read-side metadata: `list()` adds it to each record it returns, it is
    never persisted inside the record and the hash chain excludes it.
    """

    @abstractmethod
    def append(self, record: StepRecord) -> Optional[int]:
        """Append one record. Returns the `seq` of the row written when the
        backend knows it (SQLite, JSON files, in-memory), else None — the
        runtime hands that seq to the effect handler as its step context and
        never guesses one."""
        ...

    @abstractmethod
    def list(self, run_id: str) -> List[Dict[str, Any]]: ...

    def get_record_unresolved(self, run_id: str, seq: int) -> Optional[Dict[str, Any]]:
        """The row at `(run_id, seq)` as persisted (`$slim` markers unresolved),
        carrying `"seq"`; None when there is no such row.

        Default: a scan of `list()` matching each record's `seq` (or, for a
        backend whose records carry none, its 1-based position). Backends
        override with a point read."""
        s = coerce_ledger_seq(seq)
        if s is None:
            return None
        for position, rec in enumerate(self.list(run_id), start=1):
            if not isinstance(rec, dict):
                continue
            rec_seq = coerce_ledger_seq(rec.get("seq")) or position
            if rec_seq == s:
                out = dict(rec)
                out["seq"] = s
                return out
        return None

    def find_started_record(
        self, run_id: str, step_id: str, *, before_seq: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """The STARTED row of `step_id` (the first one, below `before_seq` when
        given), carrying `"seq"`; None when absent. Default: a scan of `list()`."""
        sid = str(step_id or "")
        if not sid:
            return None
        limit = coerce_ledger_seq(before_seq)
        for position, rec in enumerate(self.list(run_id), start=1):
            if not isinstance(rec, dict):
                continue
            rec_seq = coerce_ledger_seq(rec.get("seq")) or position
            if limit is not None and rec_seq >= limit:
                break
            status = rec.get("status")
            status = getattr(status, "value", status)
            if str(rec.get("step_id") or "") == sid and status == StepStatus.STARTED.value:
                out = dict(rec)
                out["seq"] = rec_seq
                return out
        return None

    def get_record(self, run_id: str, seq: int) -> Optional[Dict[str, Any]]:
        """One ledger record by its row address, ready to read (ADR 0009).

        The record carries `"seq"`; a terminal row's `$slim` markers are
        resolved (verified) against the STARTED row of the same `step_id`
        (`ledger_slim.resolve_record_slim_markers`), so the payload bytes the
        terminal row dropped come back. A marker that cannot be verified stays
        in place, never a silently-wrong payload. `OffloadingLedgerStore`
        also rehydrates its offloaded leaves. None when the row does not exist
        (deleted ledger, wrong seq).
        """
        from .ledger_slim import record_has_slim_markers, resolve_record_slim_markers

        record = self.get_record_unresolved(run_id, seq)
        if record is None:
            return None
        if not record_has_slim_markers(record):
            return record
        started = self.find_started_record(
            run_id, str(record.get("step_id") or ""), before_seq=record.get("seq")
        )
        resolved = resolve_record_slim_markers(record, started)
        return resolved

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
