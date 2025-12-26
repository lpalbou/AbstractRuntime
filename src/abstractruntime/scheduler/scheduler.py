"""abstractruntime.scheduler.scheduler

Built-in scheduler for automatic run resumption.

The scheduler:
- Polls for due wait_until runs and resumes them automatically
- Provides an API to resume wait_event/ask_user runs
- Runs in a background thread for zero-config operation
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..core.models import RunState, RunStatus, WaitReason, WaitState
from ..core.runtime import Runtime
from ..core.event_keys import build_event_wait_key
from ..storage.base import QueryableRunStore
from .registry import WorkflowRegistry


logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_paused(vars: Any) -> bool:
    if not isinstance(vars, dict):
        return False
    runtime_ns = vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        return False
    control = runtime_ns.get("control")
    if not isinstance(control, dict):
        return False
    return bool(control.get("paused") is True)


@dataclass
class SchedulerStats:
    """Statistics about scheduler operation."""

    runs_resumed: int = 0
    runs_failed: int = 0
    poll_cycles: int = 0
    last_poll_at: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class Scheduler:
    """Built-in scheduler for automatic run resumption.

    The scheduler runs a background polling loop that:
    1. Finds runs waiting for wait_until whose time has passed
    2. Calls runtime.tick() to resume them

    It also provides an API to resume wait_event/ask_user runs.

    Example:
        # Create runtime with queryable store
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        # Create registry and register workflows
        registry = WorkflowRegistry()
        registry.register(my_workflow)

        # Create and start scheduler
        scheduler = Scheduler(runtime=runtime, registry=registry)
        scheduler.start()

        # ... runs with wait_until will be resumed automatically ...

        # Resume a wait_event run manually
        state = scheduler.resume_event(
            run_id="...",
            wait_key="my_event",
            payload={"data": "value"},
        )

        # Stop scheduler gracefully
        scheduler.stop()
    """

    def __init__(
        self,
        *,
        runtime: Runtime,
        registry: WorkflowRegistry,
        poll_interval_s: float = 1.0,
        max_errors_kept: int = 100,
        on_run_resumed: Optional[Callable[[RunState], None]] = None,
        on_run_failed: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            runtime: The Runtime instance to use for tick/resume.
            registry: WorkflowRegistry mapping workflow_id -> WorkflowSpec.
            poll_interval_s: Seconds between poll cycles (default: 1.0).
            max_errors_kept: Maximum number of errors to keep in stats (default: 100).
            on_run_resumed: Optional callback when a run is resumed successfully.
            on_run_failed: Optional callback when a run fails to resume.

        Raises:
            TypeError: If the runtime's run_store doesn't implement QueryableRunStore.
        """
        self._runtime = runtime
        self._registry = registry
        self._poll_interval = poll_interval_s
        self._max_errors = max_errors_kept
        self._on_run_resumed = on_run_resumed
        self._on_run_failed = on_run_failed

        # Verify run_store is queryable
        run_store = runtime.run_store
        if not isinstance(run_store, QueryableRunStore):
            raise TypeError(
                f"Scheduler requires a QueryableRunStore, but got {type(run_store).__name__}. "
                "Use InMemoryRunStore or JsonFileRunStore which implement QueryableRunStore."
            )
        self._run_store: QueryableRunStore = run_store

        # Threading state
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Stats
        self._stats = SchedulerStats()

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is currently running."""
        return self._running

    @property
    def stats(self) -> SchedulerStats:
        """Get scheduler statistics."""
        return self._stats

    def start(self) -> None:
        """Start the scheduler background thread.

        Raises:
            RuntimeError: If the scheduler is already running.
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Scheduler is already running")

            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._poll_loop,
                name="abstractruntime-scheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info("Scheduler started (poll_interval=%.1fs)", self._poll_interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the scheduler gracefully.

        Args:
            timeout: Maximum seconds to wait for the thread to stop.
        """
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Scheduler thread did not stop within timeout")
            self._thread = None

        logger.info("Scheduler stopped")

    def resume_event(
        self,
        *,
        run_id: str,
        wait_key: str,
        payload: Dict[str, Any],
    ) -> RunState:
        """Resume a run waiting for an event.

        This is used to resume runs waiting for:
        - wait_event (external events)
        - ask_user (user input)

        Args:
            run_id: The run ID to resume.
            wait_key: The wait key to match.
            payload: The payload to inject.

        Returns:
            The updated RunState after resumption.

        Raises:
            KeyError: If the run or workflow is not found.
            ValueError: If the run is not waiting or wait_key doesn't match.
        """
        run = self._runtime.get_state(run_id)

        if run.status != RunStatus.WAITING:
            raise ValueError(f"Run '{run_id}' is not waiting (status={run.status.value})")

        if run.waiting is None:
            raise ValueError(f"Run '{run_id}' has no waiting state")

        workflow = self._registry.get_or_raise(run.workflow_id)

        return self._runtime.resume(
            workflow=workflow,
            run_id=run_id,
            wait_key=wait_key,
            payload=payload,
        )

    def emit_event(
        self,
        *,
        name: str,
        payload: Dict[str, Any],
        scope: str = "session",
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        run_id: Optional[str] = None,
        max_steps: int = 100,
        limit: int = 10_000,
    ) -> List[RunState]:
        """Emit an event and resume all matching WAIT_EVENT runs.

        This is the host-facing API for external signals (Temporal-style).

        Default scope is "session" (workflow instance). For session scope, you must
        provide `session_id` (typically the root run_id for that instance).
        """
        name2 = str(name or "").strip()
        if not name2:
            raise ValueError("emit_event requires a non-empty name")

        scope2 = str(scope or "session").strip().lower() or "session"

        wait_key = build_event_wait_key(
            scope=scope2,
            name=name2,
            session_id=session_id,
            workflow_id=workflow_id,
            run_id=run_id,
        )

        # Find runs waiting for this event key.
        waiting_runs = self._run_store.list_runs(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.EVENT,
            limit=limit,
        )

        resumed: List[RunState] = []
        envelope: Dict[str, Any] = {
            "event_id": None,
            "name": name2,
            "scope": scope2,
            "session_id": session_id,
            "payload": dict(payload) if isinstance(payload, dict) else {"value": payload},
            "emitted_at": utc_now_iso(),
            "emitter": {"source": "external"},
        }

        for r in waiting_runs:
            if _is_paused(getattr(r, "vars", None)):
                continue
            if r.waiting is None:
                continue
            if r.waiting.wait_key != wait_key:
                continue
            wf = self._registry.get_or_raise(r.workflow_id)
            new_state = self._runtime.resume(
                workflow=wf,
                run_id=r.run_id,
                wait_key=wait_key,
                payload=envelope,
                max_steps=max_steps,
            )
            resumed.append(new_state)

        return resumed

    def find_waiting_runs(
        self,
        *,
        wait_reason: Optional[WaitReason] = None,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RunState]:
        """Find runs that are currently waiting.

        Useful for building UIs that show pending user prompts.

        Args:
            wait_reason: Filter by wait reason (USER, EVENT, UNTIL).
            workflow_id: Filter by workflow ID.
            limit: Maximum number of runs to return.

        Returns:
            List of waiting RunState objects.
        """
        return self._run_store.list_runs(
            status=RunStatus.WAITING,
            wait_reason=wait_reason,
            workflow_id=workflow_id,
            limit=limit,
        )

    def poll_once(self) -> int:
        """Run a single poll cycle manually.

        This is useful for testing or for manual control.

        Returns:
            Number of runs resumed in this cycle.
        """
        return self._do_poll_cycle()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background polling loop."""
        logger.debug("Scheduler poll loop started")

        while not self._stop_event.is_set():
            try:
                self._do_poll_cycle()
            except Exception as e:
                logger.exception("Error in scheduler poll cycle: %s", e)
                self._record_error(f"Poll cycle error: {e}")

            # Wait for next cycle or stop signal
            self._stop_event.wait(timeout=self._poll_interval)

        logger.debug("Scheduler poll loop exited")

    def _do_poll_cycle(self) -> int:
        """Execute one poll cycle.

        Returns:
            Number of runs resumed.
        """
        self._stats.poll_cycles += 1
        self._stats.last_poll_at = utc_now_iso()

        # Find due wait_until runs
        now = utc_now_iso()
        due_runs = self._run_store.list_due_wait_until(now_iso=now)

        resumed_count = 0
        for run in due_runs:
            if _is_paused(getattr(run, "vars", None)):
                continue
            # Record resumption immediately to avoid a race where the run completes
            # (via runtime.tick) before the main thread observes updated stats.
            self._stats.runs_resumed += 1
            resumed_count += 1
            try:
                self._resume_wait_until(run)
            except Exception as e:
                self._stats.runs_resumed -= 1
                resumed_count -= 1
                logger.error("Failed to resume run %s: %s", run.run_id, e)
                self._stats.runs_failed += 1
                self._record_error(f"Run {run.run_id}: {e}")
                if self._on_run_failed:
                    try:
                        self._on_run_failed(run.run_id, e)
                    except Exception:
                        pass

        return resumed_count

    def _resume_wait_until(self, run: RunState) -> RunState:
        """Resume a wait_until run.

        Args:
            run: The run to resume.

        Returns:
            The updated RunState.

        Raises:
            KeyError: If the workflow is not registered.
        """
        workflow = self._registry.get_or_raise(run.workflow_id)

        # For wait_until, we just call tick() - it will auto-unblock
        new_state = self._runtime.tick(workflow=workflow, run_id=run.run_id)

        logger.debug(
            "Resumed wait_until run %s (workflow=%s, new_status=%s)",
            run.run_id,
            run.workflow_id,
            new_state.status.value,
        )

        if self._on_run_resumed:
            try:
                self._on_run_resumed(new_state)
            except Exception:
                pass

        return new_state

    def _record_error(self, error: str) -> None:
        """Record an error in stats, keeping only the most recent."""
        self._stats.errors.append(f"{utc_now_iso()}: {error}")
        if len(self._stats.errors) > self._max_errors:
            self._stats.errors = self._stats.errors[-self._max_errors :]

    def resume_subworkflow_parent(
        self,
        *,
        child_run_id: str,
        child_output: Dict[str, Any],
    ) -> Optional[RunState]:
        """Resume a parent workflow when its child subworkflow completes.

        This finds the parent waiting on the given child and resumes it
        with the child's output.

        Args:
            child_run_id: The completed child run ID.
            child_output: The child's output to pass to parent.

        Returns:
            The updated parent RunState, or None if no parent was waiting.
        """
        # Find runs waiting for this subworkflow
        waiting_runs = self._run_store.list_runs(
            status=RunStatus.WAITING,
            limit=1000,
        )

        for run in waiting_runs:
            if _is_paused(getattr(run, "vars", None)):
                continue
            if run.waiting is None:
                continue
            if run.waiting.reason != WaitReason.SUBWORKFLOW:
                continue
            if run.waiting.details is None:
                continue
            if run.waiting.details.get("sub_run_id") != child_run_id:
                continue

            # Found the parent - resume it
            wait_key = run.waiting.wait_key
            workflow = self._registry.get_or_raise(run.workflow_id)

            return self._runtime.resume(
                workflow=workflow,
                run_id=run.run_id,
                wait_key=wait_key,
                payload={"sub_run_id": child_run_id, "output": child_output},
            )

        return None

    def cancel_with_children(
        self,
        run_id: str,
        *,
        reason: Optional[str] = None,
    ) -> List[RunState]:
        """Cancel a run and all its descendant subworkflows.

        Traverses the parent-child tree and cancels all runs that are
        still active (RUNNING or WAITING).

        Args:
            run_id: The root run to cancel.
            reason: Optional cancellation reason.

        Returns:
            List of all cancelled RunState objects (including the root).
        """
        cancelled: List[RunState] = []
        to_cancel = [run_id]

        while to_cancel:
            current_id = to_cancel.pop(0)

            try:
                state = self._runtime.cancel_run(current_id, reason=reason)
                if state.status == RunStatus.CANCELLED:
                    cancelled.append(state)
            except KeyError:
                continue

            # Find children using list_children if available
            if hasattr(self._run_store, "list_children"):
                children = self._run_store.list_children(parent_run_id=current_id)
                for child in children:
                    if child.status in (RunStatus.RUNNING, RunStatus.WAITING):
                        to_cancel.append(child.run_id)

        return cancelled
