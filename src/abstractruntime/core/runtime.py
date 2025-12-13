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
    ):
        self._run_store = run_store
        self._ledger_store = ledger_store
        self._ctx = context or DefaultRunContext()

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

    def start(self, *, workflow: WorkflowSpec, vars: Optional[Dict[str, Any]] = None, actor_id: Optional[str] = None) -> str:
        run = RunState.new(workflow_id=workflow.workflow_id, entry_node=workflow.entry_node, vars=vars, actor_id=actor_id)
        self._run_store.save(run)
        return run.run_id

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

            # Effectful step
            rec = StepRecord.start(run=run, node_id=plan.node_id, effect=plan.effect)
            self._ledger_store.append(rec)

            outcome = self._execute_effect(run, plan.effect, default_next_node=plan.next_node)

            if outcome.status == "failed":
                rec.finish_failure(outcome.error or "unknown error")
                self._ledger_store.append(rec)
                run.status = RunStatus.FAILED
                run.error = outcome.error or "unknown error"
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                return run

            if outcome.status == "waiting":
                assert outcome.wait is not None
                rec.finish_waiting(outcome.wait)
                self._ledger_store.append(rec)
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


