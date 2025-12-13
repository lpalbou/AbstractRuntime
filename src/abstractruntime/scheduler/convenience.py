"""abstractruntime.scheduler.convenience

Convenience functions for zero-config scheduler setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..core.models import RunState, RunStatus, WaitReason
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
            workflows=[my_workflow],
            auto_start=True,
        )

        # Simplest usage: run() does start + tick in one call
        run_id, state = sr.run(my_workflow)

        # If waiting for user input, respond
        if state.status == RunStatus.WAITING:
            state = sr.respond(run_id, {"answer": "yes"})

        # Stop when done
        sr.stop()
    """

    runtime: Runtime
    scheduler: Scheduler
    registry: WorkflowRegistry

    def run(
        self,
        workflow: WorkflowSpec,
        *,
        vars: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        max_steps: int = 100,
    ) -> tuple[str, RunState]:
        """Start and run a workflow until it blocks or completes.

        This is the simplest way to execute a workflow. It:
        1. Registers the workflow (if not already registered)
        2. Starts a new run
        3. Ticks until WAITING, COMPLETED, or FAILED

        Args:
            workflow: The workflow to run.
            vars: Optional initial variables.
            actor_id: Optional actor ID for provenance.
            max_steps: Maximum steps before stopping (default: 100).

        Returns:
            Tuple of (run_id, final_state).

        Example:
            run_id, state = sr.run(my_workflow)
            if state.status == RunStatus.COMPLETED:
                print(state.output)
        """
        # Auto-register
        if workflow.workflow_id not in self.registry:
            self.registry.register(workflow)

        run_id = self.runtime.start(workflow=workflow, vars=vars, actor_id=actor_id)
        state = self.runtime.tick(workflow=workflow, run_id=run_id, max_steps=max_steps)
        return run_id, state

    def respond(
        self,
        run_id: str,
        payload: Dict[str, Any],
    ) -> RunState:
        """Respond to a waiting run (user input or event).

        This is the simplest way to resume a waiting run. It automatically
        finds the wait_key from the run's waiting state.

        Args:
            run_id: The run to respond to.
            payload: The response payload.

        Returns:
            The updated RunState.

        Raises:
            ValueError: If the run is not waiting.

        Example:
            state = sr.respond(run_id, {"answer": "yes"})
        """
        state = self.runtime.get_state(run_id)

        if state.status != RunStatus.WAITING:
            raise ValueError(f"Run '{run_id}' is not waiting (status={state.status.value})")

        if state.waiting is None:
            raise ValueError(f"Run '{run_id}' has no waiting state")

        wait_key = state.waiting.wait_key
        if wait_key is None:
            raise ValueError(f"Run '{run_id}' has no wait_key")

        return self.scheduler.resume_event(
            run_id=run_id,
            wait_key=wait_key,
            payload=payload,
        )

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
        run_id: str,
        *,
        max_steps: int = 100,
    ) -> RunState:
        """Progress a run. Looks up workflow from registry."""
        state = self.runtime.get_state(run_id)
        workflow = self.registry.get_or_raise(state.workflow_id)
        return self.runtime.tick(workflow=workflow, run_id=run_id, max_steps=max_steps)

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

    def resume_subworkflow_parent(
        self,
        *,
        child_run_id: str,
        child_output: Dict[str, Any],
    ) -> Optional[RunState]:
        """Resume a parent workflow when its child subworkflow completes.

        Delegates to scheduler.resume_subworkflow_parent().
        """
        return self.scheduler.resume_subworkflow_parent(
            child_run_id=child_run_id,
            child_output=child_output,
        )

    def cancel_run(self, run_id: str, *, reason: Optional[str] = None) -> RunState:
        """Cancel a run. Delegates to runtime.cancel_run()."""
        return self.runtime.cancel_run(run_id, reason=reason)

    def cancel_with_children(
        self,
        run_id: str,
        *,
        reason: Optional[str] = None,
    ) -> List[RunState]:
        """Cancel a run and all its descendant subworkflows.

        Delegates to scheduler.cancel_with_children().
        """
        return self.scheduler.cancel_with_children(run_id, reason=reason)

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
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    artifact_store: Optional[Any] = None,
    effect_policy: Optional[Any] = None,
    workflows: Optional[List[WorkflowSpec]] = None,
    effect_handlers: Optional[Dict] = None,
    poll_interval_s: float = 1.0,
    auto_start: bool = True,
    on_run_resumed: Optional[Callable[[RunState], None]] = None,
    on_run_failed: Optional[Callable[[str, Exception], None]] = None,
) -> ScheduledRuntime:
    """Create a Runtime with an integrated Scheduler.

    This is the recommended way to set up AbstractRuntime. Defaults to
    in-memory storage and auto-starting the scheduler for zero-config operation.

    Args:
        run_store: Storage backend for run state. Defaults to InMemoryRunStore.
        ledger_store: Storage backend for the ledger. Defaults to InMemoryLedgerStore.
        artifact_store: Optional artifact store for large payloads.
        workflows: Optional list of workflows to pre-register.
        effect_handlers: Optional custom effect handlers.
        poll_interval_s: Scheduler poll interval in seconds (default: 1.0).
        auto_start: If True (default), start the scheduler immediately.
        on_run_resumed: Optional callback when a run is resumed.
        on_run_failed: Optional callback when a run fails to resume.

    Returns:
        A ScheduledRuntime instance.

    Example:
        # Zero-config: just works
        sr = create_scheduled_runtime()
        run_id, state = sr.run(my_workflow)
        sr.stop()

        # With artifact store
        sr = create_scheduled_runtime(
            artifact_store=InMemoryArtifactStore(),
            workflows=[my_workflow],
        )
    """
    # Import here to avoid circular imports
    from ..storage.in_memory import InMemoryRunStore, InMemoryLedgerStore

    # Use defaults if not provided
    if run_store is None:
        run_store = InMemoryRunStore()
    if ledger_store is None:
        ledger_store = InMemoryLedgerStore()

    # Create registry and register workflows
    registry = WorkflowRegistry()
    if workflows:
        for wf in workflows:
            registry.register(wf)

    # Create runtime with registry for subworkflow support
    runtime = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=effect_handlers,
        workflow_registry=registry,
        artifact_store=artifact_store,
        effect_policy=effect_policy,
    )

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
