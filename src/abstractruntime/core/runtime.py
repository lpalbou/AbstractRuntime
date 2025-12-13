"""abstractruntime.core.runtime

Minimal durable graph runner (v0.1).

Key semantics:
- `tick()` progresses a run until it blocks (WAITING) or completes.
- Blocking is represented by a persisted WaitState in RunState.
- `resume()` injects an external payload to unblock a waiting run.

Durability note:
This MVP persists checkpoints + a ledger, but does NOT attempt to implement
full Temporal-like replay/determinism guarantees.

We keep the design explicitly modular:
- stores: RunStore + LedgerStore
- effect handlers: pluggable registry
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
import inspect

from .models import (
    Effect,
    EffectType,
    RunState,
    RunStatus,
    StepPlan,
    StepRecord,
    StepStatus,
    WaitReason,
    WaitState,
)
from .spec import WorkflowSpec
from .policy import DefaultEffectPolicy, EffectPolicy
from ..storage.base import LedgerStore, RunStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DefaultRunContext:
    def now_iso(self) -> str:
        return utc_now_iso()


# NOTE:
# Effect handlers are given the node's `next_node` as `default_next_node` so that
# waiting effects (ask_user / wait_until / tool passthrough) can safely resume
# into the next node without forcing every node to duplicate `resume_to_node`
# into the effect payload.
EffectHandler = Callable[[RunState, Effect, Optional[str]], "EffectOutcome"]


@dataclass(frozen=True)
class EffectOutcome:
    """Result of executing an effect."""

    status: str  # "completed" | "waiting" | "failed"
    result: Optional[Dict[str, Any]] = None
    wait: Optional[WaitState] = None
    error: Optional[str] = None

    @classmethod
    def completed(cls, result: Optional[Dict[str, Any]] = None) -> "EffectOutcome":
        return cls(status="completed", result=result)

    @classmethod
    def waiting(cls, wait: WaitState) -> "EffectOutcome":
        return cls(status="waiting", wait=wait)

    @classmethod
    def failed(cls, error: str) -> "EffectOutcome":
        return cls(status="failed", error=error)


class Runtime:
    """Durable graph runner."""

    def __init__(
        self,
        *,
        run_store: RunStore,
        ledger_store: LedgerStore,
        effect_handlers: Optional[Dict[EffectType, EffectHandler]] = None,
        context: Optional[Any] = None,
        workflow_registry: Optional[Any] = None,
        artifact_store: Optional[Any] = None,
        effect_policy: Optional[EffectPolicy] = None,
    ):
        self._run_store = run_store
        self._ledger_store = ledger_store
        self._ctx = context or DefaultRunContext()
        self._workflow_registry = workflow_registry
        self._artifact_store = artifact_store
        self._effect_policy: EffectPolicy = effect_policy or DefaultEffectPolicy()

        self._handlers: Dict[EffectType, EffectHandler] = {}
        self._register_builtin_handlers()
        if effect_handlers:
            self._handlers.update(effect_handlers)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @property
    def run_store(self) -> RunStore:
        """Access the run store."""
        return self._run_store

    @property
    def ledger_store(self) -> LedgerStore:
        """Access the ledger store."""
        return self._ledger_store

    @property
    def workflow_registry(self) -> Optional[Any]:
        """Access the workflow registry (if set)."""
        return self._workflow_registry

    def set_workflow_registry(self, registry: Any) -> None:
        """Set the workflow registry for subworkflow support."""
        self._workflow_registry = registry

    @property
    def artifact_store(self) -> Optional[Any]:
        """Access the artifact store (if set)."""
        return self._artifact_store

    def set_artifact_store(self, store: Any) -> None:
        """Set the artifact store for large payload support."""
        self._artifact_store = store

    @property
    def effect_policy(self) -> EffectPolicy:
        """Access the effect policy."""
        return self._effect_policy

    def set_effect_policy(self, policy: EffectPolicy) -> None:
        """Set the effect policy for retry and idempotency."""
        self._effect_policy = policy

    def start(self, *, workflow: WorkflowSpec, vars: Optional[Dict[str, Any]] = None, actor_id: Optional[str] = None, parent_run_id: Optional[str] = None) -> str:
        run = RunState.new(workflow_id=workflow.workflow_id, entry_node=workflow.entry_node, vars=vars, actor_id=actor_id, parent_run_id=parent_run_id)
        self._run_store.save(run)
        return run.run_id

    def cancel_run(self, run_id: str, *, reason: Optional[str] = None) -> RunState:
        """Cancel a run.

        Sets the run status to CANCELLED. Only RUNNING or WAITING runs can be cancelled.
        COMPLETED, FAILED, or already CANCELLED runs are returned unchanged.

        Args:
            run_id: The run to cancel.
            reason: Optional cancellation reason (stored in error field).

        Returns:
            The updated RunState.

        Raises:
            KeyError: If run_id not found.
        """
        run = self.get_state(run_id)

        # Terminal states cannot be cancelled
        if run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            return run

        run.status = RunStatus.CANCELLED
        run.error = reason or "Cancelled"
        run.waiting = None
        run.updated_at = utc_now_iso()
        self._run_store.save(run)
        return run

    def get_state(self, run_id: str) -> RunState:
        run = self._run_store.load(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return run

    def get_ledger(self, run_id: str) -> list[dict[str, Any]]:
        return self._ledger_store.list(run_id)

    def tick(self, *, workflow: WorkflowSpec, run_id: str, max_steps: int = 100) -> RunState:
        run = self.get_state(run_id)
        if run.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            return run
        if run.status == RunStatus.WAITING:
            # For WAIT_UNTIL we can auto-unblock if time passed
            if run.waiting and run.waiting.reason == WaitReason.UNTIL and run.waiting.until:
                if utc_now_iso() >= run.waiting.until:
                    self._apply_resume_payload(run, payload={}, override_node=run.waiting.resume_to_node)
                else:
                    return run
            else:
                return run

        steps = 0
        while steps < max_steps:
            steps += 1

            handler = workflow.get_node(run.current_node)
            plan = handler(run, self._ctx)

            # Completion
            if plan.complete_output is not None:
                run.status = RunStatus.COMPLETED
                run.output = plan.complete_output
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                # ledger: completion record (no effect)
                rec = StepRecord.start(run=run, node_id=plan.node_id, effect=None)
                rec.status = StepStatus.COMPLETED
                rec.result = {"completed": True}
                rec.ended_at = utc_now_iso()
                self._ledger_store.append(rec)
                return run

            # Pure transition
            if plan.effect is None:
                if not plan.next_node:
                    raise ValueError(f"Node '{plan.node_id}' returned no effect and no next_node")
                run.current_node = plan.next_node
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                continue

            # Effectful step - check for prior completed result (idempotency)
            idempotency_key = self._effect_policy.idempotency_key(
                run=run, node_id=plan.node_id, effect=plan.effect
            )
            prior_result = self._find_prior_completed_result(run.run_id, idempotency_key)

            if prior_result is not None:
                # Reuse prior result - skip re-execution
                outcome = EffectOutcome.completed(prior_result)
            else:
                # Execute with retry logic
                outcome = self._execute_effect_with_retry(
                    run=run,
                    node_id=plan.node_id,
                    effect=plan.effect,
                    idempotency_key=idempotency_key,
                    default_next_node=plan.next_node,
                )

            if outcome.status == "failed":
                run.status = RunStatus.FAILED
                run.error = outcome.error or "unknown error"
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                return run

            if outcome.status == "waiting":
                assert outcome.wait is not None
                run.status = RunStatus.WAITING
                run.waiting = outcome.wait
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                return run

            # completed
            if plan.effect.result_key and outcome.result is not None:
                _set_nested(run.vars, plan.effect.result_key, outcome.result)

            if not plan.next_node:
                raise ValueError(f"Node '{plan.node_id}' executed effect but did not specify next_node")
            run.current_node = plan.next_node
            run.updated_at = utc_now_iso()
            self._run_store.save(run)

        return run

    def resume(self, *, workflow: WorkflowSpec, run_id: str, wait_key: Optional[str], payload: Dict[str, Any]) -> RunState:
        run = self.get_state(run_id)
        if run.status != RunStatus.WAITING or run.waiting is None:
            raise ValueError("Run is not waiting")

        # Validate wait_key if provided
        if wait_key is not None and run.waiting.wait_key is not None and wait_key != run.waiting.wait_key:
            raise ValueError(f"wait_key mismatch: expected '{run.waiting.wait_key}', got '{wait_key}'")

        resume_to = run.waiting.resume_to_node
        result_key = run.waiting.result_key

        if result_key:
            _set_nested(run.vars, result_key, payload)

        self._apply_resume_payload(run, payload=payload, override_node=resume_to)
        run.updated_at = utc_now_iso()
        self._run_store.save(run)

        return self.tick(workflow=workflow, run_id=run_id)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _register_builtin_handlers(self) -> None:
        self._handlers[EffectType.WAIT_EVENT] = self._handle_wait_event
        self._handlers[EffectType.WAIT_UNTIL] = self._handle_wait_until
        self._handlers[EffectType.ASK_USER] = self._handle_ask_user
        self._handlers[EffectType.START_SUBWORKFLOW] = self._handle_start_subworkflow

    def _find_prior_completed_result(
        self, run_id: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]:
        """Find a prior completed result for an idempotency key.
        
        Scans the ledger for a completed step with the same idempotency key.
        Returns the result if found, None otherwise.
        """
        records = self._ledger_store.list(run_id)
        for record in records:
            if record.get("idempotency_key") == idempotency_key:
                if record.get("status") == StepStatus.COMPLETED.value:
                    return record.get("result")
        return None

    def _execute_effect_with_retry(
        self,
        *,
        run: RunState,
        node_id: str,
        effect: Effect,
        idempotency_key: str,
        default_next_node: Optional[str],
    ) -> EffectOutcome:
        """Execute an effect with retry logic.
        
        Retries according to the effect policy. Records each attempt
        in the ledger with attempt number and idempotency key.
        """
        import time

        max_attempts = self._effect_policy.max_attempts(effect)
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            # Record attempt start
            rec = StepRecord.start(
                run=run,
                node_id=node_id,
                effect=effect,
                attempt=attempt,
                idempotency_key=idempotency_key,
            )
            self._ledger_store.append(rec)

            # Execute the effect (catch exceptions as failures)
            try:
                outcome = self._execute_effect(run, effect, default_next_node)
            except Exception as e:
                outcome = EffectOutcome.failed(f"Effect handler raised exception: {e}")

            if outcome.status == "completed":
                rec.finish_success(outcome.result)
                self._ledger_store.append(rec)
                return outcome

            if outcome.status == "waiting":
                rec.finish_waiting(outcome.wait)
                self._ledger_store.append(rec)
                return outcome

            # Failed - record and maybe retry
            last_error = outcome.error or "unknown error"
            rec.finish_failure(last_error)
            self._ledger_store.append(rec)

            if attempt < max_attempts:
                # Wait before retry
                backoff = self._effect_policy.backoff_seconds(
                    effect=effect, attempt=attempt
                )
                if backoff > 0:
                    time.sleep(backoff)

        # All attempts exhausted
        return EffectOutcome.failed(
            f"Effect failed after {max_attempts} attempts: {last_error}"
        )

    def _execute_effect(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        if effect.type not in self._handlers:
            return EffectOutcome.failed(f"No effect handler registered for {effect.type.value}")
        handler = self._handlers[effect.type]

        # Backward compatibility: allow older handlers with signature (run, effect).
        # New handlers can accept (run, effect, default_next_node) to implement
        # correct resume semantics for waiting effects without duplicating payload fields.
        try:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            if has_varargs or len(params) >= 3:
                return handler(run, effect, default_next_node)
            return handler(run, effect)
        except Exception:
            # If signature inspection fails, fall back to attempting the new call form,
            # then the legacy form.
            try:
                return handler(run, effect, default_next_node)
            except TypeError:
                return handler(run, effect)

    def _apply_resume_payload(self, run: RunState, *, payload: Dict[str, Any], override_node: Optional[str]) -> None:
        run.status = RunStatus.RUNNING
        run.waiting = None
        if override_node:
            run.current_node = override_node

    # Built-in wait handlers ------------------------------------------------

    def _handle_wait_event(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        wait_key = effect.payload.get("wait_key")
        if not wait_key:
            return EffectOutcome.failed("wait_event requires payload.wait_key")
        resume_to = effect.payload.get("resume_to_node") or default_next_node
        wait = WaitState(
            reason=WaitReason.EVENT,
            wait_key=str(wait_key),
            resume_to_node=resume_to,
            result_key=effect.result_key,
        )
        return EffectOutcome.waiting(wait)

    def _handle_wait_until(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        until = effect.payload.get("until")
        if not until:
            return EffectOutcome.failed("wait_until requires payload.until (ISO timestamp)")

        resume_to = effect.payload.get("resume_to_node") or default_next_node
        if utc_now_iso() >= str(until):
            # immediate
            return EffectOutcome.completed({"until": str(until), "ready": True})

        wait = WaitState(
            reason=WaitReason.UNTIL,
            until=str(until),
            resume_to_node=resume_to,
            result_key=effect.result_key,
        )
        return EffectOutcome.waiting(wait)

    def _handle_ask_user(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        prompt = effect.payload.get("prompt")
        if not prompt:
            return EffectOutcome.failed("ask_user requires payload.prompt")

        resume_to = effect.payload.get("resume_to_node") or default_next_node
        wait_key = effect.payload.get("wait_key") or f"user:{run.run_id}:{run.current_node}"
        choices = effect.payload.get("choices")
        allow_free_text = bool(effect.payload.get("allow_free_text", True))

        wait = WaitState(
            reason=WaitReason.USER,
            wait_key=str(wait_key),
            resume_to_node=resume_to,
            result_key=effect.result_key,
            prompt=str(prompt),
            choices=list(choices) if isinstance(choices, list) else None,
            allow_free_text=allow_free_text,
        )
        return EffectOutcome.waiting(wait)

    def _handle_start_subworkflow(
        self, run: RunState, effect: Effect, default_next_node: Optional[str]
    ) -> EffectOutcome:
        """Handle START_SUBWORKFLOW effect.

        Payload:
            workflow_id: str - ID of the subworkflow to start (required)
            vars: dict - Initial variables for the subworkflow (optional)
            async: bool - If True, don't wait for completion (optional, default False)

        Sync mode (async=False):
            - Starts the subworkflow and runs it until completion or waiting
            - If subworkflow completes: returns its output
            - If subworkflow waits: parent also waits (WaitReason.SUBWORKFLOW)

        Async mode (async=True):
            - Starts the subworkflow and returns immediately
            - Returns {"sub_run_id": "..."} so parent can track it
        """
        workflow_id = effect.payload.get("workflow_id")
        if not workflow_id:
            return EffectOutcome.failed("start_subworkflow requires payload.workflow_id")

        if self._workflow_registry is None:
            return EffectOutcome.failed(
                "start_subworkflow requires a workflow_registry. "
                "Set it via Runtime(workflow_registry=...) or runtime.set_workflow_registry(...)"
            )

        # Look up the subworkflow
        sub_workflow = self._workflow_registry.get(workflow_id)
        if sub_workflow is None:
            return EffectOutcome.failed(f"Workflow '{workflow_id}' not found in registry")

        sub_vars = effect.payload.get("vars") or {}
        is_async = bool(effect.payload.get("async", False))
        resume_to = effect.payload.get("resume_to_node") or default_next_node

        # Start the subworkflow with parent tracking
        sub_run_id = self.start(
            workflow=sub_workflow,
            vars=sub_vars,
            actor_id=run.actor_id,  # Inherit actor from parent
            parent_run_id=run.run_id,  # Track parent for hierarchy
        )

        if is_async:
            # Async mode: return immediately with sub_run_id
            # The child is started but not ticked - caller is responsible for driving it
            return EffectOutcome.completed({"sub_run_id": sub_run_id, "async": True})

        # Sync mode: run the subworkflow until completion or waiting
        try:
            sub_state = self.tick(workflow=sub_workflow, run_id=sub_run_id)
        except Exception as e:
            # Child raised an exception - propagate as failure
            return EffectOutcome.failed(f"Subworkflow '{workflow_id}' failed: {e}")

        if sub_state.status == RunStatus.COMPLETED:
            # Subworkflow completed - return its output
            return EffectOutcome.completed({
                "sub_run_id": sub_run_id,
                "output": sub_state.output,
            })

        if sub_state.status == RunStatus.FAILED:
            # Subworkflow failed - propagate error
            return EffectOutcome.failed(
                f"Subworkflow '{workflow_id}' failed: {sub_state.error}"
            )

        if sub_state.status == RunStatus.WAITING:
            # Subworkflow is waiting - parent must also wait
            wait = WaitState(
                reason=WaitReason.SUBWORKFLOW,
                wait_key=f"subworkflow:{sub_run_id}",
                resume_to_node=resume_to,
                result_key=effect.result_key,
                details={
                    "sub_run_id": sub_run_id,
                    "sub_workflow_id": workflow_id,
                    "sub_waiting": {
                        "reason": sub_state.waiting.reason.value if sub_state.waiting else None,
                        "wait_key": sub_state.waiting.wait_key if sub_state.waiting else None,
                    },
                },
            )
            return EffectOutcome.waiting(wait)

        # Unexpected status
        return EffectOutcome.failed(f"Unexpected subworkflow status: {sub_state.status.value}")


def _set_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set nested dict value using dot notation."""

    parts = dotted_key.split(".")
    cur: Dict[str, Any] = target
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


