"""abstractruntime.scheduler.convenience

Convenience functions for zero-config scheduler setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..core.models import RunState, WaitReason
from ..core.runtime import Runtime
from ..core.spec import WorkflowSpec
from ..storage.base import LedgerStore, RunStore
from .registry import WorkflowRegistry
from .scheduler import Scheduler, SchedulerStats


@dataclass
class ScheduledRuntime:
    """A Runtime bundled with a Scheduler for zero-config operation.

    This is a convenience wrapper that provides a simpler API for common use cases.

    Example:
        # Create with convenience function
        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[my_workflow, another_workflow],
            auto_start=True,
        )

        # Use like a normal runtime
        run_id = sr.start(workflow=my_workflow)
        state = sr.tick(workflow=my_workflow, run_id=run_id)

        # Resume events through the scheduler
        state = sr.resume_event(run_id=run_id, wait_key="...", payload={...})

        # Stop when done
        sr.stop()
    """

    runtime: Runtime
    scheduler: Scheduler
    registry: WorkflowRegistry

    def start(
        self,
        *,
        workflow: WorkflowSpec,
        vars: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
    ) -> str:
        """Start a new run. Delegates to runtime.start()."""
        # Auto-register workflow if not already registered
        if workflow.workflow_id not in self.registry:
            self.registry.register(workflow)
        return self.runtime.start(workflow=workflow, vars=vars, actor_id=actor_id)

    def tick(
        self,
        *,
        workflow: WorkflowSpec,
        run_id: str,
        max_steps: int = 100,
    ) -> RunState:
        """Progress a run. Delegates to runtime.tick()."""
        return self.runtime.tick(workflow=workflow, run_id=run_id, max_steps=max_steps)

    def resume(
        self,
        *,
        workflow: WorkflowSpec,
        run_id: str,
        wait_key: Optional[str],
        payload: Dict[str, Any],
    ) -> RunState:
        """Resume a waiting run. Delegates to runtime.resume()."""
        return self.runtime.resume(
            workflow=workflow,
            run_id=run_id,
            wait_key=wait_key,
            payload=payload,
        )

    def resume_event(
        self,
        *,
        run_id: str,
        wait_key: str,
        payload: Dict[str, Any],
    ) -> RunState:
        """Resume a run waiting for an event. Delegates to scheduler.resume_event()."""
        return self.scheduler.resume_event(
            run_id=run_id,
            wait_key=wait_key,
            payload=payload,
        )

    def get_state(self, run_id: str) -> RunState:
        """Get run state. Delegates to runtime.get_state()."""
        return self.runtime.get_state(run_id)

    def get_ledger(self, run_id: str) -> list[dict[str, Any]]:
        """Get run ledger. Delegates to runtime.get_ledger()."""
        return self.runtime.get_ledger(run_id)

    def find_waiting_runs(
        self,
        *,
        wait_reason: Optional[WaitReason] = None,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RunState]:
        """Find waiting runs. Delegates to scheduler.find_waiting_runs()."""
        return self.scheduler.find_waiting_runs(
            wait_reason=wait_reason,
            workflow_id=workflow_id,
            limit=limit,
        )

    @property
    def stats(self) -> SchedulerStats:
        """Get scheduler statistics."""
        return self.scheduler.stats

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self.scheduler.is_running

    def start_scheduler(self) -> None:
        """Start the scheduler if not already running."""
        if not self.scheduler.is_running:
            self.scheduler.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the scheduler."""
        self.scheduler.stop(timeout=timeout)


def create_scheduled_runtime(
    *,
    run_store: RunStore,
    ledger_store: LedgerStore,
    workflows: Optional[List[WorkflowSpec]] = None,
    effect_handlers: Optional[Dict] = None,
    poll_interval_s: float = 1.0,
    auto_start: bool = False,
    on_run_resumed: Optional[Callable[[RunState], None]] = None,
    on_run_failed: Optional[Callable[[str, Exception], None]] = None,
) -> ScheduledRuntime:
    """Create a Runtime with an integrated Scheduler.

    This is the recommended way to set up AbstractRuntime for production use.

    Args:
        run_store: Storage backend for run state (must be QueryableRunStore).
        ledger_store: Storage backend for the execution ledger.
        workflows: Optional list of workflows to pre-register.
        effect_handlers: Optional custom effect handlers.
        poll_interval_s: Scheduler poll interval in seconds (default: 1.0).
        auto_start: If True, start the scheduler immediately.
        on_run_resumed: Optional callback when a run is resumed.
        on_run_failed: Optional callback when a run fails to resume.

    Returns:
        A ScheduledRuntime instance.

    Example:
        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[my_workflow],
            auto_start=True,
        )

        run_id = sr.start(workflow=my_workflow)
        # ... wait_until runs will be resumed automatically ...
        sr.stop()
    """
    # Create runtime
    runtime = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=effect_handlers,
    )

    # Create registry and register workflows
    registry = WorkflowRegistry()
    if workflows:
        for wf in workflows:
            registry.register(wf)

    # Create scheduler
    scheduler = Scheduler(
        runtime=runtime,
        registry=registry,
        poll_interval_s=poll_interval_s,
        on_run_resumed=on_run_resumed,
        on_run_failed=on_run_failed,
    )

    # Optionally start
    if auto_start:
        scheduler.start()

    return ScheduledRuntime(
        runtime=runtime,
        scheduler=scheduler,
        registry=registry,
    )
