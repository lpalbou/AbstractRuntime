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

from .config import RuntimeConfig
from .models import (
    Effect,
    EffectType,
    LimitWarning,
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
        config: Optional[RuntimeConfig] = None,
    ):
        self._run_store = run_store
        self._ledger_store = ledger_store
        self._ctx = context or DefaultRunContext()
        self._workflow_registry = workflow_registry
        self._artifact_store = artifact_store
        self._effect_policy: EffectPolicy = effect_policy or DefaultEffectPolicy()
        self._config: RuntimeConfig = config or RuntimeConfig()

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

    @property
    def config(self) -> RuntimeConfig:
        """Access the runtime configuration."""
        return self._config

    def start(
        self,
        *,
        workflow: WorkflowSpec,
        vars: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> str:
        # Initialize vars with _limits from config if not already set
        vars = dict(vars or {})
        if "_limits" not in vars:
            vars["_limits"] = self._config.to_limits_dict()

        run = RunState.new(
            workflow_id=workflow.workflow_id,
            entry_node=workflow.entry_node,
            vars=vars,
            actor_id=actor_id,
            session_id=session_id,
            parent_run_id=parent_run_id,
        )
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

    # ---------------------------------------------------------------------
    # Limit Management
    # ---------------------------------------------------------------------

    def get_limit_status(self, run_id: str) -> Dict[str, Any]:
        """Get current limit status for a run.

        Returns a structured dict with information about iterations, tokens,
        and history limits, including whether warning thresholds are reached.

        Args:
            run_id: The run to check

        Returns:
            Dict with "iterations", "tokens", and "history" status info

        Raises:
            KeyError: If run_id not found
        """
        run = self.get_state(run_id)
        limits = run.vars.get("_limits", {})

        def pct(current: int, maximum: int) -> float:
            return round(current / maximum * 100, 1) if maximum > 0 else 0

        current_iter = int(limits.get("current_iteration", 0) or 0)
        max_iter = int(limits.get("max_iterations", 25) or 25)
        tokens_used = int(limits.get("estimated_tokens_used", 0) or 0)
        max_tokens = int(limits.get("max_tokens", 32768) or 32768)

        return {
            "iterations": {
                "current": current_iter,
                "max": max_iter,
                "pct": pct(current_iter, max_iter),
                "warning": pct(current_iter, max_iter) >= limits.get("warn_iterations_pct", 80),
            },
            "tokens": {
                "estimated_used": tokens_used,
                "max": max_tokens,
                "pct": pct(tokens_used, max_tokens),
                "warning": pct(tokens_used, max_tokens) >= limits.get("warn_tokens_pct", 80),
            },
            "history": {
                "max_messages": limits.get("max_history_messages", -1),
            },
        }

    def check_limits(self, run: RunState) -> list[LimitWarning]:
        """Check if any limits are approaching or exceeded.

        This is the hybrid enforcement model: the runtime provides warnings,
        workflow nodes are responsible for enforcement decisions.

        Args:
            run: The RunState to check

        Returns:
            List of LimitWarning objects for any limits at warning threshold or exceeded
        """
        warnings: list[LimitWarning] = []
        limits = run.vars.get("_limits", {})

        # Check iterations
        current = int(limits.get("current_iteration", 0) or 0)
        max_iter = int(limits.get("max_iterations", 25) or 25)
        warn_pct = int(limits.get("warn_iterations_pct", 80) or 80)

        if max_iter > 0:
            if current >= max_iter:
                warnings.append(LimitWarning("iterations", "exceeded", current, max_iter))
            elif (current / max_iter * 100) >= warn_pct:
                warnings.append(LimitWarning("iterations", "warning", current, max_iter))

        # Check tokens
        tokens_used = int(limits.get("estimated_tokens_used", 0) or 0)
        max_tokens = int(limits.get("max_tokens", 32768) or 32768)
        warn_tokens_pct = int(limits.get("warn_tokens_pct", 80) or 80)

        if max_tokens > 0 and tokens_used > 0:
            if tokens_used >= max_tokens:
                warnings.append(LimitWarning("tokens", "exceeded", tokens_used, max_tokens))
            elif (tokens_used / max_tokens * 100) >= warn_tokens_pct:
                warnings.append(LimitWarning("tokens", "warning", tokens_used, max_tokens))

        return warnings

    def update_limits(self, run_id: str, updates: Dict[str, Any]) -> None:
        """Update limits for a running workflow.

        This allows mid-session updates (e.g., from /max-tokens command).
        Only allowed limit keys are updated; unknown keys are ignored.

        Args:
            run_id: The run to update
            updates: Dict of limit updates (e.g., {"max_tokens": 65536})

        Raises:
            KeyError: If run_id not found
        """
        run = self.get_state(run_id)
        limits = run.vars.setdefault("_limits", {})

        allowed_keys = {
            "max_iterations",
            "max_tokens",
            "max_output_tokens",
            "max_history_messages",
            "warn_iterations_pct",
            "warn_tokens_pct",
            "estimated_tokens_used",
            "current_iteration",
        }

        for key, value in updates.items():
            if key in allowed_keys:
                limits[key] = value

        self._run_store.save(run)

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
        self._handlers[EffectType.MEMORY_QUERY] = self._handle_memory_query
        self._handlers[EffectType.MEMORY_TAG] = self._handle_memory_tag
        self._handlers[EffectType.MEMORY_COMPACT] = self._handle_memory_compact
        self._handlers[EffectType.MEMORY_NOTE] = self._handle_memory_note
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
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            params = list(sig.parameters.values())
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            if has_varargs or len(params) >= 3:
                return handler(run, effect, default_next_node)
            return handler(run, effect)

        # If signature inspection fails, fall back to attempting the new call form,
        # then the legacy form (only for arity-mismatch TypeError).
        try:
            return handler(run, effect, default_next_node)
        except TypeError as e:
            msg = str(e)
            if "positional" in msg and "argument" in msg and ("given" in msg or "required" in msg):
                return handler(run, effect)
            raise

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
            session_id=getattr(run, "session_id", None),  # Inherit session from parent
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

    # Built-in memory handlers ---------------------------------------------

    def _handle_memory_query(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Handle MEMORY_QUERY.

        This effect supports provenance-first recall over archived memory spans stored in ArtifactStore.
        It is intentionally metadata-first and embedding-free (semantic retrieval belongs in AbstractMemory).

        Payload (all optional unless otherwise stated):
          - span_id: str | int | list[str|int]  (artifact_id or 1-based index into _runtime.memory_spans)
          - query: str                          (keyword substring match)
          - since: str                          (ISO8601, span intersection filter)
          - until: str                          (ISO8601, span intersection filter)
          - tags: dict[str,str]                 (span tag filter)
          - limit_spans: int                    (default 5)
          - deep: bool                          (default True when query is set; scans archived messages)
          - deep_limit_spans: int               (default 50)
          - deep_limit_messages_per_span: int   (default 400)
          - connected: bool                     (include connected spans via time adjacency + shared tags)
          - neighbor_hops: int                  (default 1 when connected=True)
          - connect_by: list[str]               (default ["topic","person"])
          - max_messages: int                   (default 80; total messages rendered across all spans)
          - tool_name: str                      (default "recall_memory"; for formatting)
          - call_id: str                        (tool-call id passthrough)
        """
        from .vars import ensure_namespaces

        ensure_namespaces(run.vars)
        runtime_ns = run.vars.get("_runtime")
        if not isinstance(runtime_ns, dict):
            runtime_ns = {}
            run.vars["_runtime"] = runtime_ns

        artifact_store = self._artifact_store
        if artifact_store is None:
            return EffectOutcome.failed(
                "MEMORY_QUERY requires an ArtifactStore; configure runtime.set_artifact_store(...)"
            )

        payload = dict(effect.payload or {})
        tool_name = str(payload.get("tool_name") or "recall_memory")
        call_id = str(payload.get("call_id") or "memory")

        query = payload.get("query")
        query_text = str(query or "").strip()
        since = payload.get("since")
        until = payload.get("until")
        tags = payload.get("tags")
        tags_dict = dict(tags) if isinstance(tags, dict) else None

        try:
            limit_spans = int(payload.get("limit_spans", 5) or 5)
        except Exception:
            limit_spans = 5
        if limit_spans < 1:
            limit_spans = 1

        deep = payload.get("deep")
        if deep is None:
            deep_enabled = bool(query_text)
        else:
            deep_enabled = bool(deep)

        try:
            deep_limit_spans = int(payload.get("deep_limit_spans", 50) or 50)
        except Exception:
            deep_limit_spans = 50
        if deep_limit_spans < 1:
            deep_limit_spans = 1

        try:
            deep_limit_messages_per_span = int(payload.get("deep_limit_messages_per_span", 400) or 400)
        except Exception:
            deep_limit_messages_per_span = 400
        if deep_limit_messages_per_span < 1:
            deep_limit_messages_per_span = 1

        connected = bool(payload.get("connected", False))
        try:
            neighbor_hops = int(payload.get("neighbor_hops", 1) or 1)
        except Exception:
            neighbor_hops = 1
        if neighbor_hops < 0:
            neighbor_hops = 0

        connect_by = payload.get("connect_by")
        if isinstance(connect_by, list):
            connect_keys = [str(x) for x in connect_by if isinstance(x, (str, int, float)) and str(x).strip()]
        else:
            connect_keys = ["topic", "person"]

        try:
            max_messages = int(payload.get("max_messages", 80) or 80)
        except Exception:
            max_messages = 80
        if max_messages < 1:
            max_messages = 1

        from ..memory.active_context import ActiveContextPolicy, TimeRange

        spans = ActiveContextPolicy.list_memory_spans_from_run(run)

        # Resolve explicit span ids if provided.
        span_id_payload = payload.get("span_id")
        span_ids_payload = payload.get("span_ids")
        explicit_ids = span_ids_payload if isinstance(span_ids_payload, list) else span_id_payload
        explicit_resolved: list[str] = []
        if explicit_ids is not None:
            if isinstance(explicit_ids, list):
                explicit_resolved = ActiveContextPolicy.resolve_span_ids_from_spans(explicit_ids, spans)
            else:
                explicit_resolved = ActiveContextPolicy.resolve_span_ids_from_spans([explicit_ids], spans)

        selected: list[str] = []
        if explicit_resolved:
            selected = list(explicit_resolved)
        else:
            time_range = None
            if since or until:
                time_range = TimeRange(
                    start=str(since) if since else None,
                    end=str(until) if until else None,
                )
            matches = ActiveContextPolicy.filter_spans_from_run(
                run,
                artifact_store=artifact_store,
                time_range=time_range,
                tags=tags_dict,
                query=query_text or None,
                limit=limit_spans,
            )
            selected = [str(s.get("artifact_id") or "") for s in matches if isinstance(s, dict) and s.get("artifact_id")]

            if deep_enabled and query_text:
                selected = _dedup_preserve_order(selected + _deep_scan_span_ids(
                    spans=spans,
                    artifact_store=artifact_store,
                    query=query_text,
                    limit_spans=deep_limit_spans,
                    limit_messages_per_span=deep_limit_messages_per_span,
                ))

        if connected and selected:
            selected = _dedup_preserve_order(
                _expand_connected_span_ids(
                    spans=spans,
                    seed_artifact_ids=selected,
                    connect_keys=connect_keys,
                    neighbor_hops=neighbor_hops,
                    limit=max(limit_spans, len(selected)),
                )
            )

        # Render output (provenance + messages).
        summary_by_artifact = ActiveContextPolicy.summary_text_by_artifact_id_from_run(run)
        text = _render_memory_query_output(
            spans=spans,
            artifact_store=artifact_store,
            selected_artifact_ids=selected,
            summary_by_artifact=summary_by_artifact,
            max_messages=max_messages,
        )

        result = {
            "mode": "executed",
            "results": [
                {
                    "call_id": call_id,
                    "name": tool_name,
                    "success": True,
                    "output": text,
                    "error": None,
                }
            ],
        }
        return EffectOutcome.completed(result=result)

    def _handle_memory_tag(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Handle MEMORY_TAG.

        Payload (required unless stated):
          - span_id: str | int   (artifact_id or 1-based index into `_runtime.memory_spans`)
          - tags: dict[str,str]  (merged into span["tags"] by default)
          - merge: bool          (optional, default True; when False, replaces span["tags"])
          - tool_name: str       (optional; for tool-style output, default "remember")
          - call_id: str         (optional; passthrough for tool-style output)

        Notes:
        - This mutates the in-run span index (`_runtime.memory_spans`) only; it does not change artifacts.
        - Tagging is intentionally JSON-safe (string->string).
        """
        import json

        from .vars import ensure_namespaces

        ensure_namespaces(run.vars)
        runtime_ns = run.vars.get("_runtime")
        if not isinstance(runtime_ns, dict):
            runtime_ns = {}
            run.vars["_runtime"] = runtime_ns

        spans = runtime_ns.get("memory_spans")
        if not isinstance(spans, list):
            return EffectOutcome.failed("MEMORY_TAG requires _runtime.memory_spans to be a list")

        payload = dict(effect.payload or {})
        tool_name = str(payload.get("tool_name") or "remember")
        call_id = str(payload.get("call_id") or "memory")

        span_id = payload.get("span_id")
        tags = payload.get("tags")
        if span_id is None:
            return EffectOutcome.failed("MEMORY_TAG requires payload.span_id")
        if not isinstance(tags, dict) or not tags:
            return EffectOutcome.failed("MEMORY_TAG requires payload.tags as a non-empty dict[str,str]")

        merge = bool(payload.get("merge", True))

        clean_tags: Dict[str, str] = {}
        for k, v in tags.items():
            if isinstance(k, str) and isinstance(v, str) and k and v:
                clean_tags[k] = v
        if not clean_tags:
            return EffectOutcome.failed("MEMORY_TAG requires at least one non-empty string tag")

        artifact_id: Optional[str] = None
        target_index: Optional[int] = None

        if isinstance(span_id, int):
            idx = span_id - 1
            if idx < 0 or idx >= len(spans):
                return EffectOutcome.failed(f"Unknown span index: {span_id}")
            span = spans[idx]
            if not isinstance(span, dict):
                return EffectOutcome.failed(f"Invalid span record at index {span_id}")
            artifact_id = str(span.get("artifact_id") or "").strip() or None
            target_index = idx
        elif isinstance(span_id, str):
            s = span_id.strip()
            if not s:
                return EffectOutcome.failed("MEMORY_TAG requires a non-empty span_id")
            if s.isdigit():
                idx = int(s) - 1
                if idx < 0 or idx >= len(spans):
                    return EffectOutcome.failed(f"Unknown span index: {s}")
                span = spans[idx]
                if not isinstance(span, dict):
                    return EffectOutcome.failed(f"Invalid span record at index {s}")
                artifact_id = str(span.get("artifact_id") or "").strip() or None
                target_index = idx
            else:
                artifact_id = s
        else:
            return EffectOutcome.failed("MEMORY_TAG requires span_id as str or int")

        if not artifact_id:
            return EffectOutcome.failed("Could not resolve span_id to an artifact_id")

        if target_index is None:
            for i, span in enumerate(spans):
                if not isinstance(span, dict):
                    continue
                if str(span.get("artifact_id") or "") == artifact_id:
                    target_index = i
                    break

        if target_index is None:
            return EffectOutcome.failed(f"Unknown span_id: {artifact_id}")

        target = spans[target_index]
        if not isinstance(target, dict):
            return EffectOutcome.failed(f"Invalid span record at index {target_index + 1}")

        existing_tags = target.get("tags")
        if not isinstance(existing_tags, dict):
            existing_tags = {}

        if merge:
            merged_tags = dict(existing_tags)
            merged_tags.update(clean_tags)
        else:
            merged_tags = dict(clean_tags)

        target["tags"] = merged_tags
        target["tagged_at"] = utc_now_iso()
        if run.actor_id:
            target["tagged_by"] = str(run.actor_id)

        rendered_tags = json.dumps(merged_tags, ensure_ascii=False, sort_keys=True)
        text = f"Tagged span_id={artifact_id} tags={rendered_tags}"

        result = {
            "mode": "executed",
            "results": [
                {
                    "call_id": call_id,
                    "name": tool_name,
                    "success": True,
                    "output": text,
                    "error": None,
                }
            ],
        }
        return EffectOutcome.completed(result=result)

    def _handle_memory_compact(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Handle MEMORY_COMPACT.

        This is a runtime-owned compaction of a run's active context:
        - archives the compacted messages to ArtifactStore (provenance preserved)
        - inserts a system summary message that includes `span_id=...` (LLM-visible handle)
        - updates `_runtime.memory_spans` index with metadata/tags

        Payload (optional unless stated):
          - preserve_recent: int        (default 6; preserves N most recent non-system messages)
          - compression_mode: str       ("light"|"standard"|"heavy", default "standard")
          - focus: str                  (optional; topic to prioritize)
          - target_run_id: str          (optional; defaults to current run)
          - tool_name: str              (optional; for tool-style output, default "compact_memory")
          - call_id: str                (optional)
        """
        import json
        from uuid import uuid4

        from .vars import ensure_namespaces
        from ..memory.compaction import normalize_messages, split_for_compaction, span_metadata_from_messages

        ensure_namespaces(run.vars)

        artifact_store = self._artifact_store
        if artifact_store is None:
            return EffectOutcome.failed(
                "MEMORY_COMPACT requires an ArtifactStore; configure runtime.set_artifact_store(...)"
            )

        payload = dict(effect.payload or {})
        tool_name = str(payload.get("tool_name") or "compact_memory")
        call_id = str(payload.get("call_id") or "memory")

        target_run_id = payload.get("target_run_id")
        if target_run_id is not None:
            target_run_id = str(target_run_id).strip() or None

        try:
            preserve_recent = int(payload.get("preserve_recent", 6) or 6)
        except Exception:
            preserve_recent = 6
        if preserve_recent < 0:
            preserve_recent = 0

        compression_mode = str(payload.get("compression_mode") or "standard").strip().lower()
        if compression_mode not in ("light", "standard", "heavy"):
            compression_mode = "standard"

        focus = payload.get("focus")
        focus_text = str(focus).strip() if isinstance(focus, str) else ""
        focus_text = focus_text or None

        # Resolve which run is being compacted.
        target_run = run
        if target_run_id and target_run_id != run.run_id:
            loaded = self._run_store.load(target_run_id)
            if loaded is None:
                return EffectOutcome.failed(f"Unknown target_run_id: {target_run_id}")
            target_run = loaded
        ensure_namespaces(target_run.vars)

        ctx = target_run.vars.get("context")
        if not isinstance(ctx, dict):
            return EffectOutcome.failed("MEMORY_COMPACT requires vars.context to be a dict")
        messages_raw = ctx.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            return EffectOutcome.completed(
                result={
                    "mode": "executed",
                    "results": [
                        {
                            "call_id": call_id,
                            "name": tool_name,
                            "success": True,
                            "output": "No messages to compact.",
                            "error": None,
                        }
                    ],
                }
            )

        now_iso = utc_now_iso
        messages = normalize_messages(messages_raw, now_iso=now_iso)
        split = split_for_compaction(messages, preserve_recent=preserve_recent)

        if not split.older_messages:
            return EffectOutcome.completed(
                result={
                    "mode": "executed",
                    "results": [
                        {
                            "call_id": call_id,
                            "name": tool_name,
                            "success": True,
                            "output": f"Nothing to compact (non-system messages <= preserve_recent={preserve_recent}).",
                            "error": None,
                        }
                    ],
                }
            )

        # ------------------------------------------------------------------
        # 1) LLM summary (ledgered via a child run with an LLM_CALL effect)
        # ------------------------------------------------------------------

        older_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in split.older_messages])
        focus_line = f"Focus: {focus_text}\n" if focus_text else ""
        mode_line = f"Compression mode: {compression_mode}\n"

        prompt = (
            "You are compressing older conversation context for an agent runtime.\n"
            "Write a faithful, compact summary that preserves decisions, constraints, names, file paths, commands, and open questions.\n"
            "Do NOT invent details. If something is unknown, say so.\n"
            f"{mode_line}"
            f"{focus_line}"
            "Return STRICT JSON with keys: summary (string), key_points (array of strings), confidence (number 0..1).\n\n"
            "OLDER MESSAGES (to be archived):\n"
            f"{older_text}\n"
        )

        # Best-effort output budget for the summary itself.
        limits = target_run.vars.get("_limits") if isinstance(target_run.vars.get("_limits"), dict) else {}
        max_out = limits.get("max_output_tokens")
        try:
            max_out_tokens = int(max_out) if max_out is not None else None
        except Exception:
            max_out_tokens = None

        llm_payload: Dict[str, Any] = {"prompt": prompt}
        if max_out_tokens is not None:
            llm_payload["params"] = {"max_tokens": max_out_tokens}

        def llm_node(sub_run: RunState, sub_ctx) -> StepPlan:
            return StepPlan(
                node_id="llm",
                effect=Effect(type=EffectType.LLM_CALL, payload=llm_payload, result_key="_temp.llm"),
                next_node="done",
            )

        def done_node(sub_run: RunState, sub_ctx) -> StepPlan:
            temp = sub_run.vars.get("_temp") if isinstance(sub_run.vars.get("_temp"), dict) else {}
            return StepPlan(node_id="done", complete_output={"response": temp.get("llm")})

        wf = WorkflowSpec(workflow_id="wf_memory_compact_llm", entry_node="llm", nodes={"llm": llm_node, "done": done_node})

        sub_run_id = self.start(
            workflow=wf,
            vars={"context": {"prompt": prompt}, "scratchpad": {}, "_runtime": {}, "_temp": {}, "_limits": dict(limits)},
            actor_id=run.actor_id,
            session_id=getattr(run, "session_id", None),
            parent_run_id=run.run_id,
        )

        sub_state = self.tick(workflow=wf, run_id=sub_run_id)
        if sub_state.status == RunStatus.WAITING:
            return EffectOutcome.failed("MEMORY_COMPACT does not support waiting subworkflows yet")
        if sub_state.status == RunStatus.FAILED:
            return EffectOutcome.failed(sub_state.error or "Compaction LLM subworkflow failed")
        response = (sub_state.output or {}).get("response")
        if not isinstance(response, dict):
            response = {}

        content = response.get("content")
        content_text = "" if content is None else str(content).strip()
        lowered = content_text.lower()
        if any(
            keyword in lowered
            for keyword in (
                "operation not permitted",
                "failed to connect",
                "connection refused",
                "timed out",
                "timeout",
                "not running",
                "model not found",
            )
        ):
            return EffectOutcome.failed(f"Compaction LLM unavailable: {content_text}")

        summary_text_out = content_text
        key_points: list[str] = []
        confidence: Optional[float] = None

        # Parse JSON if present (support fenced output).
        if content_text:
            candidate = content_text
            if "```" in candidate:
                # extract first JSON-ish block
                start = candidate.find("{")
                end = candidate.rfind("}")
                if 0 <= start < end:
                    candidate = candidate[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    if parsed.get("summary") is not None:
                        summary_text_out = str(parsed.get("summary") or "").strip() or summary_text_out
                    kp = parsed.get("key_points")
                    if isinstance(kp, list):
                        key_points = [str(x) for x in kp if isinstance(x, (str, int, float))][:20]
                    conf = parsed.get("confidence")
                    if isinstance(conf, (int, float)):
                        confidence = float(conf)
            except Exception:
                pass

        summary_text_out = summary_text_out.strip()
        if not summary_text_out:
            summary_text_out = "(summary unavailable)"

        # ------------------------------------------------------------------
        # 2) Archive older messages + update run state with summary
        # ------------------------------------------------------------------

        span_meta = span_metadata_from_messages(split.older_messages)
        artifact_payload = {
            "messages": split.older_messages,
            "span": span_meta,
            "created_at": now_iso(),
        }
        artifact_tags: Dict[str, str] = {
            "kind": "conversation_span",
            "compression_mode": compression_mode,
            "preserve_recent": str(preserve_recent),
        }
        if focus_text:
            artifact_tags["focus"] = focus_text

        meta = artifact_store.store_json(artifact_payload, run_id=target_run.run_id, tags=artifact_tags)
        archived_ref = meta.artifact_id

        summary_message_id = f"msg_{uuid4().hex}"
        summary_prefix = f"[CONVERSATION HISTORY SUMMARY span_id={archived_ref}]"
        summary_metadata: Dict[str, Any] = {
            "message_id": summary_message_id,
            "kind": "memory_summary",
            "compression_mode": compression_mode,
            "preserve_recent": preserve_recent,
            "source_artifact_id": archived_ref,
            "source_message_count": int(span_meta.get("message_count") or 0),
            "source_from_timestamp": span_meta.get("from_timestamp"),
            "source_to_timestamp": span_meta.get("to_timestamp"),
            "source_from_message_id": span_meta.get("from_message_id"),
            "source_to_message_id": span_meta.get("to_message_id"),
        }
        if focus_text:
            summary_metadata["focus"] = focus_text

        summary_message = {
            "role": "system",
            "content": f"{summary_prefix}: {summary_text_out}",
            "timestamp": now_iso(),
            "metadata": summary_metadata,
        }

        new_messages = list(split.system_messages) + [summary_message] + list(split.recent_messages)
        ctx["messages"] = new_messages
        if isinstance(getattr(target_run, "output", None), dict):
            target_run.output["messages"] = new_messages

        runtime_ns = target_run.vars.get("_runtime")
        if not isinstance(runtime_ns, dict):
            runtime_ns = {}
            target_run.vars["_runtime"] = runtime_ns
        spans = runtime_ns.get("memory_spans")
        if not isinstance(spans, list):
            spans = []
            runtime_ns["memory_spans"] = spans
        spans.append(
            {
                "kind": "conversation_span",
                "artifact_id": archived_ref,
                "created_at": now_iso(),
                "summary_message_id": summary_message_id,
                "from_timestamp": span_meta.get("from_timestamp"),
                "to_timestamp": span_meta.get("to_timestamp"),
                "from_message_id": span_meta.get("from_message_id"),
                "to_message_id": span_meta.get("to_message_id"),
                "message_count": int(span_meta.get("message_count") or 0),
                "compression_mode": compression_mode,
                "focus": focus_text,
            }
        )

        if target_run is not run:
            target_run.updated_at = now_iso()
            self._run_store.save(target_run)

        out = {
            "llm_run_id": sub_run_id,
            "span_id": archived_ref,
            "summary_message_id": summary_message_id,
            "preserve_recent": preserve_recent,
            "compression_mode": compression_mode,
            "focus": focus_text,
            "key_points": key_points,
            "confidence": confidence,
        }
        text = f"Compacted {len(split.older_messages)} messages into span_id={archived_ref}."
        result = {
            "mode": "executed",
            "results": [
                {
                    "call_id": call_id,
                    "name": tool_name,
                    "success": True,
                    "output": text,
                    "error": None,
                    "meta": out,
                }
            ],
        }
        return EffectOutcome.completed(result=result)

    def _handle_memory_note(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Handle MEMORY_NOTE.

        Store a small, durable memory note (key insight/decision) with tags and provenance sources.

        Payload:
          - note: str                (required)
          - tags: dict[str,str]      (optional)
          - sources: dict            (optional)
              - run_id: str          (optional; defaults to current run_id)
              - span_ids: list[str]  (optional; referenced span ids)
              - message_ids: list[str] (optional; referenced message ids)
          - tool_name: str           (optional; default "remember_note")
          - call_id: str             (optional; passthrough)
        """
        import json

        from .vars import ensure_namespaces

        ensure_namespaces(run.vars)
        runtime_ns = run.vars.get("_runtime")
        if not isinstance(runtime_ns, dict):
            runtime_ns = {}
            run.vars["_runtime"] = runtime_ns

        spans = runtime_ns.get("memory_spans")
        if not isinstance(spans, list):
            spans = []
            runtime_ns["memory_spans"] = spans

        artifact_store = self._artifact_store
        if artifact_store is None:
            return EffectOutcome.failed(
                "MEMORY_NOTE requires an ArtifactStore; configure runtime.set_artifact_store(...)"
            )

        payload = dict(effect.payload or {})
        tool_name = str(payload.get("tool_name") or "remember_note")
        call_id = str(payload.get("call_id") or "memory")

        note = payload.get("note")
        note_text = str(note or "").strip()
        if not note_text:
            return EffectOutcome.failed("MEMORY_NOTE requires payload.note (non-empty string)")

        tags = payload.get("tags")
        clean_tags: Dict[str, str] = {}
        if isinstance(tags, dict):
            for k, v in tags.items():
                if isinstance(k, str) and isinstance(v, str) and k and v:
                    if k == "kind":
                        continue
                    clean_tags[k] = v

        sources = payload.get("sources")
        sources_dict = dict(sources) if isinstance(sources, dict) else {}

        def _norm_list(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            out: list[str] = []
            for item in value:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append(s)
                elif isinstance(item, int):
                    out.append(str(item))
            # preserve order but dedup
            seen: set[str] = set()
            deduped: list[str] = []
            for s in out:
                if s in seen:
                    continue
                seen.add(s)
                deduped.append(s)
            return deduped

        source_run_id = str(sources_dict.get("run_id") or run.run_id).strip() or run.run_id
        span_ids = _norm_list(sources_dict.get("span_ids"))
        message_ids = _norm_list(sources_dict.get("message_ids"))

        created_at = utc_now_iso()
        artifact_payload: Dict[str, Any] = {
            "note": note_text,
            "sources": {"run_id": source_run_id, "span_ids": span_ids, "message_ids": message_ids},
            "created_at": created_at,
        }
        if run.actor_id:
            artifact_payload["actor_id"] = str(run.actor_id)
        session_id = getattr(run, "session_id", None)
        if session_id:
            artifact_payload["session_id"] = str(session_id)

        artifact_tags: Dict[str, str] = {"kind": "memory_note"}
        artifact_tags.update(clean_tags)
        meta = artifact_store.store_json(artifact_payload, run_id=run.run_id, tags=artifact_tags)
        artifact_id = meta.artifact_id

        preview = note_text
        if len(preview) > 160:
            preview = preview[:157] + "…"

        span_record: Dict[str, Any] = {
            "kind": "memory_note",
            "artifact_id": artifact_id,
            "created_at": created_at,
            # Treat notes as point-in-time spans for time-range filtering.
            "from_timestamp": created_at,
            "to_timestamp": created_at,
            "message_count": 0,
            "note_preview": preview,
        }
        if clean_tags:
            span_record["tags"] = dict(clean_tags)
        if span_ids or message_ids:
            span_record["sources"] = {"run_id": source_run_id, "span_ids": span_ids, "message_ids": message_ids}
        if run.actor_id:
            span_record["created_by"] = str(run.actor_id)

        spans.append(span_record)

        rendered_tags = json.dumps(clean_tags, ensure_ascii=False, sort_keys=True) if clean_tags else "{}"
        text = f"Stored memory_note span_id={artifact_id} tags={rendered_tags}"
        result = {
            "mode": "executed",
            "results": [
                {
                    "call_id": call_id,
                    "name": tool_name,
                    "success": True,
                    "output": text,
                    "error": None,
                    "meta": {"span_id": artifact_id, "created_at": created_at},
                }
            ],
        }
        return EffectOutcome.completed(result=result)


def _dedup_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _span_sort_key(span: dict) -> tuple[str, str]:
    """Sort key for span adjacency. Prefer from_timestamp, then created_at."""
    from_ts = str(span.get("from_timestamp") or "")
    created = str(span.get("created_at") or "")
    return (from_ts or created, created)


def _expand_connected_span_ids(
    *,
    spans: list[dict[str, Any]],
    seed_artifact_ids: list[str],
    connect_keys: list[str],
    neighbor_hops: int,
    limit: int,
) -> list[str]:
    """Expand seed spans to include deterministic neighbors (time + shared tags)."""
    if not spans or not seed_artifact_ids:
        return list(seed_artifact_ids)

    ordered = [s for s in spans if isinstance(s, dict) and s.get("artifact_id")]
    ordered.sort(key=_span_sort_key)
    idx_by_artifact: dict[str, int] = {str(s["artifact_id"]): i for i, s in enumerate(ordered)}

    # Build tag index for requested keys.
    tag_index: dict[tuple[str, str], list[str]] = {}
    for s in ordered:
        tags = s.get("tags") if isinstance(s.get("tags"), dict) else {}
        for k in connect_keys:
            v = tags.get(k)
            if isinstance(v, str) and v:
                tag_index.setdefault((k, v), []).append(str(s["artifact_id"]))

    out: list[str] = []
    for aid in seed_artifact_ids:
        if len(out) >= limit:
            break
        out.append(aid)

        idx = idx_by_artifact.get(aid)
        if idx is not None and neighbor_hops > 0:
            for delta in range(1, neighbor_hops + 1):
                for j in (idx - delta, idx + delta):
                    if 0 <= j < len(ordered):
                        out.append(str(ordered[j]["artifact_id"]))

        if connect_keys:
            s = ordered[idx] if idx is not None and 0 <= idx < len(ordered) else None
            if isinstance(s, dict):
                tags = s.get("tags") if isinstance(s.get("tags"), dict) else {}
                for k in connect_keys:
                    v = tags.get(k)
                    if isinstance(v, str) and v:
                        out.extend(tag_index.get((k, v), []))

    return _dedup_preserve_order(out)[:limit]


def _deep_scan_span_ids(
    *,
    spans: list[dict[str, Any]],
    artifact_store: Any,
    query: str,
    limit_spans: int,
    limit_messages_per_span: int,
) -> list[str]:
    """Fallback keyword scan over archived messages when metadata/summary is insufficient."""
    q = str(query or "").strip().lower()
    if not q:
        return []

    scanned = 0
    matches: list[str] = []
    for s in spans:
        if scanned >= limit_spans:
            break
        if not isinstance(s, dict):
            continue
        artifact_id = s.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            continue
        scanned += 1

        payload = artifact_store.load_json(artifact_id)
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            continue

        for m in messages[:limit_messages_per_span]:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if not content:
                continue
            if q in str(content).lower():
                matches.append(artifact_id)
                break

    return _dedup_preserve_order(matches)


def _render_memory_query_output(
    *,
    spans: list[dict[str, Any]],
    artifact_store: Any,
    selected_artifact_ids: list[str],
    summary_by_artifact: dict[str, str],
    max_messages: int,
) -> str:
    if not selected_artifact_ids:
        return "No matching memory spans."

    span_by_id: dict[str, dict[str, Any]] = {
        str(s.get("artifact_id")): s for s in spans if isinstance(s, dict) and s.get("artifact_id")
    }

    lines: list[str] = []
    lines.append("Recalled memory spans (provenance-preserving):")

    remaining = int(max_messages)
    for i, aid in enumerate(selected_artifact_ids, start=1):
        span = span_by_id.get(aid, {})
        kind = span.get("kind") or "span"
        created = span.get("created_at") or ""
        from_ts = span.get("from_timestamp") or ""
        to_ts = span.get("to_timestamp") or ""
        count = span.get("message_count") or ""
        tags = span.get("tags") if isinstance(span.get("tags"), dict) else {}
        tags_txt = ", ".join([f"{k}={v}" for k, v in sorted(tags.items()) if isinstance(v, str) and v])

        lines.append("")
        lines.append(f"[{i}] span_id={aid} kind={kind} msgs={count} created_at={created}")
        if from_ts or to_ts:
            lines.append(f"    time_range: {from_ts} .. {to_ts}")
        if tags_txt:
            lines.append(f"    tags: {tags_txt}")

        summary = summary_by_artifact.get(aid)
        if summary:
            summary_clean = str(summary).strip()
            if len(summary_clean) > 800:
                summary_clean = summary_clean[:800] + "…"
            lines.append(f"    summary: {summary_clean}")

        if remaining <= 0:
            continue

        payload = artifact_store.load_json(aid)
        if not isinstance(payload, dict):
            lines.append("    (artifact payload unavailable)")
            continue
        if kind == "memory_note" or "note" in payload:
            note = str(payload.get("note") or "").strip()
            if note:
                if len(note) > 2400:
                    note = note[:2400] + "…"
                lines.append("    note: " + note.replace("\n", "\\n"))
            else:
                lines.append("    (note payload missing note text)")

            sources = payload.get("sources")
            if isinstance(sources, dict):
                src_run = sources.get("run_id")
                span_ids = sources.get("span_ids")
                msg_ids = sources.get("message_ids")
                if isinstance(src_run, str) and src_run:
                    lines.append(f"    sources.run_id: {src_run}")
                if isinstance(span_ids, list) and span_ids:
                    cleaned = [str(x) for x in span_ids if isinstance(x, (str, int))]
                    if cleaned:
                        lines.append(f"    sources.span_ids: {', '.join(cleaned[:12])}")
                if isinstance(msg_ids, list) and msg_ids:
                    cleaned = [str(x) for x in msg_ids if isinstance(x, (str, int))]
                    if cleaned:
                        lines.append(f"    sources.message_ids: {', '.join(cleaned[:12])}")
            continue

        messages = payload.get("messages")
        if not isinstance(messages, list):
            lines.append("    (artifact missing messages)")
            continue

        # Render messages with a global cap.
        rendered = 0
        for m in messages:
            if remaining <= 0:
                break
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "unknown")
            content = str(m.get("content") or "")
            ts = str(m.get("timestamp") or "")
            # Keep per-message lines bounded.
            if len(content) > 1200:
                content = content[:1200] + "…"
            prefix = f"    - {role}: "
            if ts:
                prefix = f"    - {ts} {role}: "
            lines.append(prefix + content.replace("\n", "\\n"))
            rendered += 1
            remaining -= 1

        total = sum(1 for m in messages if isinstance(m, dict))
        if rendered < total:
            lines.append(f"    … ({total - rendered} more messages not shown)")

    if remaining <= 0:
        lines.append("")
        lines.append(f"(Output truncated: max_messages={int(max_messages)})")

    return "\n".join(lines)


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
