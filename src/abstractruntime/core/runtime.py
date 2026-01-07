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
from typing import Any, Callable, Dict, Optional, List
import copy
import inspect
import json
import os
import re

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
from ..storage.base import LedgerStore, RunStore, QueryableRunStore
from .event_keys import build_event_wait_key


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_DEFAULT_GLOBAL_MEMORY_RUN_ID = "global_memory"
_SAFE_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _ensure_runtime_namespace(vars: Dict[str, Any]) -> Dict[str, Any]:
    runtime_ns = vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        runtime_ns = {}
        vars["_runtime"] = runtime_ns
    return runtime_ns


def _ensure_control_namespace(vars: Dict[str, Any]) -> Dict[str, Any]:
    runtime_ns = _ensure_runtime_namespace(vars)
    control = runtime_ns.get("control")
    if not isinstance(control, dict):
        control = {}
        runtime_ns["control"] = control
    return control


def _is_paused_run_vars(vars: Any) -> bool:
    if not isinstance(vars, dict):
        return False
    runtime_ns = vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        return False
    control = runtime_ns.get("control")
    if not isinstance(control, dict):
        return False
    return bool(control.get("paused") is True)


def _is_pause_wait(waiting: Any, *, run_id: str) -> bool:
    if waiting is None:
        return False
    try:
        reason = getattr(waiting, "reason", None)
        reason_value = reason.value if hasattr(reason, "value") else str(reason) if reason else None
    except Exception:
        reason_value = None
    if reason_value != WaitReason.USER.value:
        return False
    try:
        wait_key = getattr(waiting, "wait_key", None)
        if isinstance(wait_key, str) and wait_key == f"pause:{run_id}":
            return True
    except Exception:
        pass
    try:
        details = getattr(waiting, "details", None)
        if isinstance(details, dict) and details.get("kind") == "pause":
            return True
    except Exception:
        pass
    return False


def _record_node_trace(
    *,
    run: RunState,
    node_id: str,
    effect: Effect,
    outcome: "EffectOutcome",
    idempotency_key: Optional[str],
    reused_prior_result: bool,
    duration_ms: Optional[float] = None,
    max_entries_per_node: int = 100,
) -> None:
    """Record a JSON-safe per-node execution trace in run.vars["_runtime"].

    This trace is runtime-owned and durable (stored in RunStore checkpoints).
    It exists to support higher-level hosts (AbstractFlow, AbstractCode, etc.)
    that need structured "scratchpad"/debug information without inventing
    host-specific persistence formats.
    """

    runtime_ns = _ensure_runtime_namespace(run.vars)
    traces = runtime_ns.get("node_traces")
    if not isinstance(traces, dict):
        traces = {}
        runtime_ns["node_traces"] = traces

    node_trace = traces.get(node_id)
    if not isinstance(node_trace, dict):
        node_trace = {"node_id": node_id, "steps": []}
        traces[node_id] = node_trace

    steps = node_trace.get("steps")
    if not isinstance(steps, list):
        steps = []
        node_trace["steps"] = steps

    wait_dict: Optional[Dict[str, Any]] = None
    if outcome.status == "waiting" and outcome.wait is not None:
        w = outcome.wait
        wait_dict = {
            "reason": w.reason.value if hasattr(w.reason, "value") else str(w.reason),
            "wait_key": w.wait_key,
            "until": w.until,
            "resume_to_node": w.resume_to_node,
            "result_key": w.result_key,
            "prompt": w.prompt,
            "choices": w.choices,
            "allow_free_text": w.allow_free_text,
            "details": w.details,
        }

    entry: Dict[str, Any] = {
        "ts": utc_now_iso(),
        "node_id": node_id,
        "status": outcome.status,
        "idempotency_key": idempotency_key,
        "reused_prior_result": reused_prior_result,
        "effect": {
            "type": effect.type.value,
            "payload": effect.payload,
            "result_key": effect.result_key,
        },
    }
    if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
        # UI/UX consumers use this for per-step timing badges (kept JSON-safe).
        entry["duration_ms"] = float(duration_ms)
    if outcome.status == "completed":
        entry["result"] = outcome.result
    elif outcome.status == "failed":
        entry["error"] = outcome.error
    elif wait_dict is not None:
        entry["wait"] = wait_dict

    # Ensure the trace remains JSON-safe even if a handler violates the contract.
    try:
        json.dumps(entry)
    except TypeError:
        entry = {
            "ts": entry.get("ts"),
            "node_id": node_id,
            "status": outcome.status,
            "idempotency_key": idempotency_key,
            "reused_prior_result": reused_prior_result,
            "effect": {"type": effect.type.value, "result_key": effect.result_key},
            "error": "non_json_safe_trace_entry",
        }

    steps.append(entry)
    if max_entries_per_node > 0 and len(steps) > max_entries_per_node:
        del steps[: max(0, len(steps) - max_entries_per_node)]
    node_trace["updated_at"] = utc_now_iso()


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
        chat_summarizer: Optional[Any] = None,
    ):
        self._run_store = run_store
        self._ledger_store = ledger_store
        self._ctx = context or DefaultRunContext()
        self._workflow_registry = workflow_registry
        self._artifact_store = artifact_store
        self._effect_policy: EffectPolicy = effect_policy or DefaultEffectPolicy()
        self._config: RuntimeConfig = config or RuntimeConfig()
        self._chat_summarizer = chat_summarizer

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

        # Ensure a durable `_runtime` namespace exists and seed default provider/model metadata
        # from the Runtime config (best-effort).
        #
        # Rationale:
        # - The Runtime is the orchestration authority (ADR-0001/0014), and `start()` is the
        #   choke point where durable run state is initialized.
        # - Agents/workflows should not have to guess/duplicate routing metadata to make prompt
        #   composition decisions (e.g. native-tools => omit Tools(session) prompt catalogs).
        runtime_ns = vars.get("_runtime")
        if not isinstance(runtime_ns, dict):
            runtime_ns = {}
            vars["_runtime"] = runtime_ns
        try:
            provider_id = getattr(self._config, "provider", None)
            model_id = getattr(self._config, "model", None)
            if isinstance(provider_id, str) and provider_id.strip():
                runtime_ns.setdefault("provider", provider_id.strip())
            if isinstance(model_id, str) and model_id.strip():
                runtime_ns.setdefault("model", model_id.strip())
        except Exception:
            pass

        # Seed tool-support metadata from model capabilities (best-effort).
        #
        # This makes the native-vs-prompted tools decision explicit and durable in run state,
        # so adapters/UI helpers don't have to guess or re-run AbstractCore detection logic.
        try:
            caps = getattr(self._config, "model_capabilities", None)
            if isinstance(caps, dict):
                tool_support = caps.get("tool_support")
                if isinstance(tool_support, str) and tool_support.strip():
                    ts = tool_support.strip()
                    runtime_ns.setdefault("tool_support", ts)
                    runtime_ns.setdefault("supports_native_tools", ts == "native")
        except Exception:
            pass

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
        try:
            control = _ensure_control_namespace(run.vars)
            control.pop("paused", None)
        except Exception:
            pass
        run.updated_at = utc_now_iso()
        self._run_store.save(run)
        return run

    def pause_run(self, run_id: str, *, reason: Optional[str] = None) -> RunState:
        """Pause a run (durably) until it is explicitly resumed.

        Semantics:
        - Pausing a RUNNING run transitions it to WAITING with a synthetic USER wait.
        - Pausing a WAITING run (non-USER waits such as UNTIL/EVENT/SUBWORKFLOW) sets a
          runtime-owned `paused` flag so schedulers/event emitters can skip it.
        - Pausing an ASK_USER wait is a no-op (already blocked by user input).
        """
        run = self.get_state(run_id)

        if run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            return run

        # If already paused, keep as-is.
        if _is_paused_run_vars(run.vars):
            return run

        # Don't interfere with real user prompts (ASK_USER).
        if run.status == RunStatus.WAITING and run.waiting is not None:
            if getattr(run.waiting, "reason", None) == WaitReason.USER and not _is_pause_wait(run.waiting, run_id=run_id):
                return run

        control = _ensure_control_namespace(run.vars)
        control["paused"] = True
        control["paused_at"] = utc_now_iso()
        if isinstance(reason, str) and reason.strip():
            control["pause_reason"] = reason.strip()

        if run.status == RunStatus.RUNNING:
            run.status = RunStatus.WAITING
            run.waiting = WaitState(
                reason=WaitReason.USER,
                wait_key=f"pause:{run.run_id}",
                resume_to_node=run.current_node,
                prompt="Paused",
                choices=None,
                allow_free_text=False,
                details={"kind": "pause"},
            )

        run.updated_at = utc_now_iso()
        self._run_store.save(run)
        return run

    def resume_run(self, run_id: str) -> RunState:
        """Resume a previously paused run (durably).

        If the run was paused while RUNNING, this clears the synthetic pause wait
        and returns the run to RUNNING. If the run was paused while WAITING
        (UNTIL/EVENT/SUBWORKFLOW), this only clears the paused flag.
        """
        run = self.get_state(run_id)

        if run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            return run

        if not _is_paused_run_vars(run.vars):
            return run

        try:
            control = _ensure_control_namespace(run.vars)
            control.pop("paused", None)
            control.pop("pause_reason", None)
            control["resumed_at"] = utc_now_iso()
        except Exception:
            pass

        if run.status == RunStatus.WAITING and _is_pause_wait(run.waiting, run_id=run_id):
            resume_to = getattr(run.waiting, "resume_to_node", None)
            self._apply_resume_payload(run, payload={}, override_node=resume_to)

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

    def subscribe_ledger(
        self,
        callback: Callable[[Dict[str, Any]], None],
        *,
        run_id: Optional[str] = None,
    ) -> Callable[[], None]:
        """Subscribe to ledger append events (in-process only).

        This is an optional capability: not all LedgerStore implementations
        support subscriptions. When unavailable, wrap the configured store with
        `abstractruntime.storage.observable.ObservableLedgerStore`.
        """
        subscribe = getattr(self._ledger_store, "subscribe", None)
        if not callable(subscribe):
            raise RuntimeError(
                "Configured LedgerStore does not support subscriptions. "
                "Wrap it with ObservableLedgerStore to enable `subscribe_ledger()`."
            )
        return subscribe(callback, run_id=run_id)

    # ---------------------------------------------------------------------
    # Trace Helpers (Runtime-Owned)
    # ---------------------------------------------------------------------

    def get_node_traces(self, run_id: str) -> Dict[str, Any]:
        """Return runtime-owned per-node traces for a run.

        Traces are stored in `RunState.vars["_runtime"]["node_traces"]`.
        Returns a deep copy so callers can safely inspect without mutating the run.
        """
        run = self.get_state(run_id)
        runtime_ns = run.vars.get("_runtime")
        traces = runtime_ns.get("node_traces") if isinstance(runtime_ns, dict) else None
        return copy.deepcopy(traces) if isinstance(traces, dict) else {}

    def get_node_trace(self, run_id: str, node_id: str) -> Dict[str, Any]:
        """Return a single node trace object for a run.

        Returns an empty `{node_id, steps: []}` object when missing.
        """
        traces = self.get_node_traces(run_id)
        trace = traces.get(node_id)
        if isinstance(trace, dict):
            return trace
        return {"node_id": node_id, "steps": []}

    # ---------------------------------------------------------------------
    # Evidence Helpers (Runtime-Owned)
    # ---------------------------------------------------------------------

    def list_evidence(self, run_id: str) -> list[dict[str, Any]]:
        """List evidence records for a run (index entries only).

        Evidence is indexed as `kind="evidence"` items inside `vars["_runtime"]["memory_spans"]`.
        """
        run = self.get_state(run_id)
        runtime_ns = run.vars.get("_runtime")
        spans = runtime_ns.get("memory_spans") if isinstance(runtime_ns, dict) else None
        if not isinstance(spans, list):
            return []
        out: list[dict[str, Any]] = []
        for s in spans:
            if not isinstance(s, dict):
                continue
            if s.get("kind") != "evidence":
                continue
            out.append(copy.deepcopy(s))
        return out

    def load_evidence(self, evidence_id: str) -> Optional[dict[str, Any]]:
        """Load an evidence record payload from ArtifactStore by id."""
        artifact_store = self._artifact_store
        if artifact_store is None:
            raise RuntimeError("Evidence requires an ArtifactStore; configure runtime.set_artifact_store(...)")
        payload = artifact_store.load_json(str(evidence_id))
        return payload if isinstance(payload, dict) else None

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
        # Terminal runs never progress.
        if run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            return run
        if _is_paused_run_vars(run.vars):
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

        # IMPORTANT (Web hosts / concurrency):
        # A run may be paused/cancelled by an external control plane (e.g. AbstractFlow Web UI)
        # while we're blocked inside a long-running effect (LLM/tool execution).
        #
        # We make `tick()` resilient to that by re-loading the persisted RunState before
        # committing any updates. If an external pause/cancel is observed, we stop without
        # overwriting it.
        def _abort_if_externally_controlled() -> Optional[RunState]:
            try:
                latest = self.get_state(run_id)
            except Exception:
                return None
            if latest.status == RunStatus.CANCELLED:
                return latest
            if _is_paused_run_vars(latest.vars):
                return latest
            return None

        steps = 0
        while steps < max_steps:
            steps += 1

            controlled = _abort_if_externally_controlled()
            if controlled is not None:
                return controlled

            handler = workflow.get_node(run.current_node)
            plan = handler(run, self._ctx)

            # Completion
            if plan.complete_output is not None:
                controlled = _abort_if_externally_controlled()
                if controlled is not None:
                    return controlled
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
                controlled = _abort_if_externally_controlled()
                if controlled is not None:
                    return controlled
                run.current_node = plan.next_node
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                continue

            # Effectful step - check for prior completed result (idempotency)
            idempotency_key = self._effect_policy.idempotency_key(
                run=run, node_id=plan.node_id, effect=plan.effect
            )
            prior_result = self._find_prior_completed_result(run.run_id, idempotency_key)
            reused_prior_result = prior_result is not None

            # Measure effect execution duration (wall-clock). This is used for
            # host-side UX (badges, throughput estimates) and is stored in the
            # runtime-owned node trace (JSON-safe).
            import time
            t0 = time.perf_counter()

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

            duration_ms = float((time.perf_counter() - t0) * 1000.0)

            # Evidence capture (runtime-owned, durable):
            # After tool execution completes, record provenance-first evidence for a small set of
            # external-boundary tools (web_search/fetch_url/execute_command). This must happen
            # BEFORE we persist node traces / result_key outputs so run state remains bounded.
            try:
                if (
                    not reused_prior_result
                    and plan.effect.type == EffectType.TOOL_CALLS
                    and outcome.status == "completed"
                ):
                    self._maybe_record_tool_evidence(
                        run=run,
                        node_id=plan.node_id,
                        effect=plan.effect,
                        tool_results=outcome.result,
                    )
            except Exception:
                # Evidence capture should never crash the run; failures are recorded in run vars.
                pass

            _record_node_trace(
                run=run,
                node_id=plan.node_id,
                effect=plan.effect,
                outcome=outcome,
                idempotency_key=idempotency_key,
                reused_prior_result=reused_prior_result,
                duration_ms=duration_ms,
            )

            if outcome.status == "failed":
                controlled = _abort_if_externally_controlled()
                if controlled is not None:
                    return controlled
                run.status = RunStatus.FAILED
                run.error = outcome.error or "unknown error"
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                return run

            if outcome.status == "waiting":
                assert outcome.wait is not None
                controlled = _abort_if_externally_controlled()
                if controlled is not None:
                    return controlled
                run.status = RunStatus.WAITING
                run.waiting = outcome.wait
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                return run

            # completed
            if plan.effect.result_key and outcome.result is not None:
                _set_nested(run.vars, plan.effect.result_key, outcome.result)

            # Terminal effect node: treat missing next_node as completion.
            #
            # Rationale: StepPlan.complete_output is evaluated *before* effects
            # execute, so an effectful node cannot both execute an effect and
            # complete the run in a single StepPlan. Allowing next_node=None
            # makes "end on an effect node" valid (Blueprint-style UX).
            if not plan.next_node:
                controlled = _abort_if_externally_controlled()
                if controlled is not None:
                    return controlled
                run.status = RunStatus.COMPLETED
                run.output = {"success": True, "result": outcome.result}
                run.updated_at = utc_now_iso()
                self._run_store.save(run)
                return run
            controlled = _abort_if_externally_controlled()
            if controlled is not None:
                return controlled
            run.current_node = plan.next_node
            run.updated_at = utc_now_iso()
            self._run_store.save(run)

        return run

    def _maybe_record_tool_evidence(
        self,
        *,
        run: RunState,
        node_id: str,
        effect: Effect,
        tool_results: Optional[Dict[str, Any]],
    ) -> None:
        """Best-effort evidence capture for TOOL_CALLS.

        This is intentionally non-fatal: evidence capture must not crash the run,
        but failures should be visible in durable run state for debugging.
        """
        if effect.type != EffectType.TOOL_CALLS:
            return
        if not isinstance(tool_results, dict):
            return
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return

        artifact_store = self._artifact_store
        if artifact_store is None:
            return

        try:
            from ..evidence import EvidenceRecorder

            EvidenceRecorder(artifact_store=artifact_store).record_tool_calls(
                run=run,
                node_id=str(node_id or ""),
                tool_calls=list(tool_calls),
                tool_results=tool_results,
            )
        except Exception as e:
            runtime_ns = _ensure_runtime_namespace(run.vars)
            warnings = runtime_ns.get("evidence_warnings")
            if not isinstance(warnings, list):
                warnings = []
                runtime_ns["evidence_warnings"] = warnings
            warnings.append({"ts": utc_now_iso(), "node_id": str(node_id or ""), "error": str(e)})

    def resume(
        self,
        *,
        workflow: WorkflowSpec,
        run_id: str,
        wait_key: Optional[str],
        payload: Dict[str, Any],
        max_steps: int = 100,
    ) -> RunState:
        run = self.get_state(run_id)
        if _is_paused_run_vars(run.vars):
            raise ValueError("Run is paused")
        if run.status != RunStatus.WAITING or run.waiting is None:
            raise ValueError("Run is not waiting")

        # Validate wait_key if provided
        if wait_key is not None and run.waiting.wait_key is not None and wait_key != run.waiting.wait_key:
            raise ValueError(f"wait_key mismatch: expected '{run.waiting.wait_key}', got '{wait_key}'")

        resume_to = run.waiting.resume_to_node
        result_key = run.waiting.result_key

        # Keep track of what we actually persisted for this resume (tool resumes may
        # merge blocked-by-allowlist entries back into the payload).
        stored_payload: Dict[str, Any] = payload

        if result_key:
            # Tool waits may carry blocked-by-allowlist metadata. External hosts typically only execute
            # the filtered subset of tool calls and resume with results for those calls. To keep agent
            # semantics correct (and evidence indices aligned), merge blocked entries back into the
            # resumed payload deterministically.
            merged_payload: Dict[str, Any] = payload
            try:
                details = run.waiting.details if run.waiting is not None else None
                if isinstance(details, dict):
                    blocked = details.get("blocked_by_index")
                    original_count = details.get("original_call_count")
                    results = payload.get("results") if isinstance(payload, dict) else None
                    if (
                        isinstance(blocked, dict)
                        and isinstance(original_count, int)
                        and original_count > 0
                        and isinstance(results, list)
                        and len(results) != original_count
                    ):
                        merged_results: list[Any] = []
                        executed_iter = iter(results)

                        for idx in range(original_count):
                            blocked_entry = blocked.get(str(idx))
                            if isinstance(blocked_entry, dict):
                                merged_results.append(blocked_entry)
                                continue
                            try:
                                merged_results.append(next(executed_iter))
                            except StopIteration:
                                merged_results.append(
                                    {
                                        "call_id": "",
                                        "name": "",
                                        "success": False,
                                        "output": None,
                                        "error": "Missing tool result",
                                    }
                                )

                        merged_payload = dict(payload)
                        merged_payload["results"] = merged_results
                        merged_payload.setdefault("mode", "executed")
            except Exception:
                merged_payload = payload

            _set_nested(run.vars, result_key, merged_payload)
            stored_payload = merged_payload
            # Passthrough tool execution: the host resumes with tool results. We still want
            # evidence capture and payload-bounding (store large parts as artifacts) before
            # the run continues.
            try:
                details = run.waiting.details if run.waiting is not None else None
                tool_calls_for_evidence = None
                if isinstance(details, dict):
                    tool_calls_for_evidence = details.get("tool_calls_for_evidence")
                    if not isinstance(tool_calls_for_evidence, list):
                        tool_calls_for_evidence = details.get("tool_calls")

                if isinstance(tool_calls_for_evidence, list):
                    from ..evidence import EvidenceRecorder

                    artifact_store = self._artifact_store
                    if artifact_store is not None and isinstance(payload, dict):
                        EvidenceRecorder(artifact_store=artifact_store).record_tool_calls(
                            run=run,
                            node_id=str(run.current_node or ""),
                            tool_calls=list(tool_calls_for_evidence or []),
                            tool_results=merged_payload,
                        )
            except Exception:
                pass

        # Terminal waiting node: if there is no resume target, treat the resume payload as
        # the final output instead of re-executing the waiting node again (which would
        # otherwise create an infinite wait/resume loop).
        if resume_to is None:
            run.status = RunStatus.COMPLETED
            run.waiting = None
            run.output = {"success": True, "result": stored_payload}
            run.updated_at = utc_now_iso()
            self._run_store.save(run)
            return run

        self._apply_resume_payload(run, payload=payload, override_node=resume_to)
        run.updated_at = utc_now_iso()
        self._run_store.save(run)

        if max_steps <= 0:
            return run
        return self.tick(workflow=workflow, run_id=run_id, max_steps=max_steps)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _register_builtin_handlers(self) -> None:
        self._handlers[EffectType.WAIT_EVENT] = self._handle_wait_event
        self._handlers[EffectType.WAIT_UNTIL] = self._handle_wait_until
        self._handlers[EffectType.ASK_USER] = self._handle_ask_user
        self._handlers[EffectType.ANSWER_USER] = self._handle_answer_user
        self._handlers[EffectType.EMIT_EVENT] = self._handle_emit_event
        self._handlers[EffectType.MEMORY_QUERY] = self._handle_memory_query
        self._handlers[EffectType.MEMORY_TAG] = self._handle_memory_tag
        self._handlers[EffectType.MEMORY_COMPACT] = self._handle_memory_compact
        self._handlers[EffectType.MEMORY_NOTE] = self._handle_memory_note
        self._handlers[EffectType.MEMORY_REHYDRATE] = self._handle_memory_rehydrate
        self._handlers[EffectType.VARS_QUERY] = self._handle_vars_query
        self._handlers[EffectType.START_SUBWORKFLOW] = self._handle_start_subworkflow

    # Built-in memory helpers ------------------------------------------------

    def _global_memory_run_id(self) -> str:
        """Return the global memory run id (stable).

        Hosts can override via `ABSTRACTRUNTIME_GLOBAL_MEMORY_RUN_ID`.
        """
        rid = os.environ.get("ABSTRACTRUNTIME_GLOBAL_MEMORY_RUN_ID")
        rid = str(rid or "").strip()
        if rid and _SAFE_RUN_ID_PATTERN.match(rid):
            return rid
        return _DEFAULT_GLOBAL_MEMORY_RUN_ID

    def _ensure_global_memory_run(self) -> RunState:
        """Load or create the global memory run used as the owner for `scope="global"` spans."""
        rid = self._global_memory_run_id()
        existing = self._run_store.load(rid)
        if existing is not None:
            return existing

        run = RunState(
            run_id=rid,
            workflow_id="__global_memory__",
            status=RunStatus.COMPLETED,
            current_node="done",
            vars={
                "context": {"task": "", "messages": []},
                "scratchpad": {},
                "_runtime": {"memory_spans": []},
                "_temp": {},
                "_limits": {},
            },
            waiting=None,
            output={"messages": []},
            error=None,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            actor_id=None,
            session_id=None,
            parent_run_id=None,
        )
        self._run_store.save(run)
        return run

    def _resolve_session_root_run(self, run: RunState) -> RunState:
        """Resolve the root run of the current run-tree (walk `parent_run_id`)."""
        cur = run
        seen: set[str] = set()
        while True:
            parent_id = getattr(cur, "parent_run_id", None)
            if not isinstance(parent_id, str) or not parent_id.strip():
                return cur
            pid = parent_id.strip()
            if pid in seen:
                # Defensive: break cycles.
                return cur
            seen.add(pid)
            parent = self._run_store.load(pid)
            if parent is None:
                return cur
            cur = parent

    def _resolve_scope_owner_run(self, base_run: RunState, *, scope: str) -> RunState:
        s = str(scope or "").strip().lower() or "run"
        if s == "run":
            return base_run
        if s == "session":
            return self._resolve_session_root_run(base_run)
        if s == "global":
            return self._ensure_global_memory_run()
        raise ValueError(f"Unknown memory scope: {scope}")

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
            # Allow structured payloads (scope/name) so hosts can compute stable keys.
            scope = effect.payload.get("scope", "session")
            name = effect.payload.get("name") or effect.payload.get("event_name")
            if not isinstance(name, str) or not name.strip():
                return EffectOutcome.failed("wait_event requires payload.wait_key or payload.name")

            session_id = effect.payload.get("session_id") or run.session_id or run.run_id
            try:
                wait_key = build_event_wait_key(
                    scope=str(scope or "session"),
                    name=str(name),
                    session_id=str(session_id) if session_id is not None else None,
                    workflow_id=run.workflow_id,
                    run_id=run.run_id,
                )
            except Exception as e:
                return EffectOutcome.failed(f"wait_event invalid payload: {e}")
        resume_to = effect.payload.get("resume_to_node") or default_next_node
        # Optional UX metadata for hosts:
        # - "prompt"/"choices"/"allow_free_text" enable durable human-in-the-loop
        #   waits using EVENT as the wakeup mechanism (useful for thin clients).
        prompt: Optional[str] = None
        try:
            p = effect.payload.get("prompt")
            if isinstance(p, str) and p.strip():
                prompt = p
        except Exception:
            prompt = None

        choices: Optional[List[str]] = None
        try:
            raw_choices = effect.payload.get("choices")
            if isinstance(raw_choices, list):
                normalized: List[str] = []
                for c in raw_choices:
                    if isinstance(c, str) and c.strip():
                        normalized.append(c.strip())
                choices = normalized
        except Exception:
            choices = None

        allow_free_text = True
        try:
            aft = effect.payload.get("allow_free_text")
            if aft is None:
                aft = effect.payload.get("allowFreeText")
            if aft is not None:
                allow_free_text = bool(aft)
        except Exception:
            allow_free_text = True

        details = None
        try:
            d = effect.payload.get("details")
            if isinstance(d, dict):
                details = dict(d)
        except Exception:
            details = None
        wait = WaitState(
            reason=WaitReason.EVENT,
            wait_key=str(wait_key),
            resume_to_node=resume_to,
            result_key=effect.result_key,
            prompt=prompt,
            choices=choices,
            allow_free_text=allow_free_text,
            details=details,
        )
        return EffectOutcome.waiting(wait)

    def _handle_emit_event(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Emit a durable event and resume matching WAIT_EVENT runs.

        Payload:
          - name: str (required)  event name
          - scope: str (optional, default "session")  "session" | "workflow" | "run" | "global"
          - session_id: str (optional)  target session id (for cross-workflow targeted delivery)
          - payload: dict (optional)  event payload delivered to listeners
          - max_steps: int (optional, default 100)  tick budget per resumed run

        Notes:
        - This is durable because it resumes WAIT_EVENT runs via Runtime.resume(), which
          checkpoints run state and appends ledger records for subsequent steps.
        - Delivery is best-effort and at-least-once; listeners should be idempotent if needed.
        """
        name = effect.payload.get("name") or effect.payload.get("event_name")
        if not isinstance(name, str) or not name.strip():
            return EffectOutcome.failed("emit_event requires payload.name")

        scope = effect.payload.get("scope", "session")
        target_session_id = effect.payload.get("session_id")
        payload = effect.payload.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {"value": payload}

        # NOTE: we intentionally resume listeners with max_steps=0 (no execution).
        # Hosts (web backend, workers, schedulers) should drive RUNNING runs and
        # stream their StepRecords deterministically (better observability and UX).
        try:
            max_steps = int(effect.payload.get("max_steps", 0) or 0)
        except Exception:
            max_steps = 0
        if max_steps < 0:
            max_steps = 0

        # Determine target scope id (default: current session/run).
        session_id = target_session_id
        if session_id is None and str(scope or "session").strip().lower() == "session":
            session_id = run.session_id or run.run_id

        try:
            wait_key = build_event_wait_key(
                scope=str(scope or "session"),
                name=str(name),
                session_id=str(session_id) if session_id is not None else None,
                workflow_id=run.workflow_id,
                run_id=run.run_id,
            )
        except Exception as e:
            return EffectOutcome.failed(f"emit_event invalid payload: {e}")

        # Wildcard listeners ("*") receive all events within the same scope_id.
        wildcard_wait_key: Optional[str] = None
        try:
            wildcard_wait_key = build_event_wait_key(
                scope=str(scope or "session"),
                name="*",
                session_id=str(session_id) if session_id is not None else None,
                workflow_id=run.workflow_id,
                run_id=run.run_id,
            )
        except Exception:
            wildcard_wait_key = None

        if self._workflow_registry is None:
            return EffectOutcome.failed(
                "emit_event requires a workflow_registry to resume target runs. "
                "Set it via Runtime(workflow_registry=...) or runtime.set_workflow_registry(...)."
            )

        if not isinstance(self._run_store, QueryableRunStore):
            return EffectOutcome.failed(
                "emit_event requires a QueryableRunStore to find waiting runs. "
                "Use InMemoryRunStore/JsonFileRunStore or provide a queryable store."
            )

        # Find all runs waiting for this event key.
        candidates = self._run_store.list_runs(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.EVENT,
            limit=10_000,
        )

        delivered_to: list[str] = []
        resumed: list[Dict[str, Any]] = []
        envelope = {
            "event_id": effect.payload.get("event_id") or None,
            "name": str(name),
            "scope": str(scope or "session"),
            "session_id": str(session_id) if session_id is not None else None,
            "payload": dict(payload),
            "emitted_at": utc_now_iso(),
            "emitter": {
                "run_id": run.run_id,
                "workflow_id": run.workflow_id,
                "node_id": run.current_node,
            },
        }

        available_in_session: list[str] = []
        prefix = f"evt:session:{session_id}:"

        for r in candidates:
            if _is_paused_run_vars(getattr(r, "vars", None)):
                continue
            w = getattr(r, "waiting", None)
            if w is None:
                continue
            wk = getattr(w, "wait_key", None)
            if isinstance(wk, str) and wk.startswith(prefix):
                # Help users debug name mismatches (best-effort).
                suffix = wk[len(prefix) :]
                if suffix and suffix not in available_in_session and len(available_in_session) < 15:
                    available_in_session.append(suffix)
            if wk != wait_key and (wildcard_wait_key is None or wk != wildcard_wait_key):
                continue

            wf = self._workflow_registry.get(r.workflow_id)
            if wf is None:
                # Can't resume without the spec; skip but include diagnostic in result.
                resumed.append({"run_id": r.run_id, "status": "skipped", "error": "workflow_not_registered"})
                continue

            try:
                # Resume using the run's own wait_key (supports wildcard listeners).
                resume_key = wk if isinstance(wk, str) and wk else None
                new_state = self.resume(
                    workflow=wf,
                    run_id=r.run_id,
                    wait_key=resume_key,
                    payload=envelope,
                    max_steps=max_steps,
                )
                delivered_to.append(r.run_id)
                resumed.append({"run_id": r.run_id, "status": new_state.status.value})
            except Exception as e:
                resumed.append({"run_id": r.run_id, "status": "error", "error": str(e)})

        out: Dict[str, Any] = {
            "wait_key": wait_key,
            "name": str(name),
            "scope": str(scope or "session"),
            "delivered": len(delivered_to),
            "delivered_to": delivered_to,
            "resumed": resumed,
        }
        if not delivered_to and available_in_session:
            out["available_listeners_in_session"] = available_in_session
        return EffectOutcome.completed(out)

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

    def _handle_answer_user(
        self, run: RunState, effect: Effect, default_next_node: Optional[str]
    ) -> EffectOutcome:
        """Handle ANSWER_USER effect.

        This effect is intentionally non-blocking: it completes immediately and
        returns the message payload so the host UI can render it.
        """
        message = effect.payload.get("message")
        if message is None:
            # Backward/compat convenience aliases.
            message = effect.payload.get("text") or effect.payload.get("content")
        if message is None:
            return EffectOutcome.failed("answer_user requires payload.message")
        return EffectOutcome.completed({"message": str(message)})

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
        wait_for_completion = bool(effect.payload.get("wait", False))
        include_traces = bool(effect.payload.get("include_traces", False))
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
            # Async mode: start the child and return immediately.
            #
            # If `wait=True`, we *also* transition the parent into a durable WAITING state
            # so a host (e.g. AbstractFlow WebSocket runner loop) can:
            # - tick the child run incrementally (and stream node traces in real time)
            # - resume the parent once the child completes (by calling runtime.resume(...))
            #
            # Without `wait=True`, this remains fire-and-forget.
            if wait_for_completion:
                wait = WaitState(
                    reason=WaitReason.SUBWORKFLOW,
                    wait_key=f"subworkflow:{sub_run_id}",
                    resume_to_node=resume_to,
                    result_key=effect.result_key,
                    details={
                        "sub_run_id": sub_run_id,
                        "sub_workflow_id": workflow_id,
                        "async": True,
                    },
                )
                return EffectOutcome.waiting(wait)

            # Fire-and-forget: caller is responsible for driving/observing the child.
            return EffectOutcome.completed({"sub_run_id": sub_run_id, "async": True})

        # Sync mode: run the subworkflow until completion or waiting
        try:
            sub_state = self.tick(workflow=sub_workflow, run_id=sub_run_id)
        except Exception as e:
            # Child raised an exception - propagate as failure
            return EffectOutcome.failed(f"Subworkflow '{workflow_id}' failed: {e}")

        if sub_state.status == RunStatus.COMPLETED:
            # Subworkflow completed - return its output
            result: Dict[str, Any] = {"sub_run_id": sub_run_id, "output": sub_state.output}
            if include_traces:
                result["node_traces"] = self.get_node_traces(sub_run_id)
            return EffectOutcome.completed(result)

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
          - tags: dict[str, str|list[str]]      (span tag filter; values can be multi-valued)
          - tags_mode: "all"|"any"              (default "all"; AND/OR across tag keys)
          - authors: list[str]                  (alias: usernames; matches span.created_by case-insensitively)
          - locations: list[str]                (matches span.location case-insensitively)
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
        from .vars import ensure_namespaces, parse_vars_path, resolve_vars_path

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

        # Scope routing (run-tree/global). Scope affects which run owns the span index queried.
        scope = str(payload.get("scope") or "run").strip().lower() or "run"
        if scope not in {"run", "session", "global", "all"}:
            return EffectOutcome.failed(f"Unknown memory_query scope: {scope}")

        # Return mode controls whether we include structured meta in the tool result.
        return_mode = str(payload.get("return") or payload.get("return_mode") or "rendered").strip().lower() or "rendered"
        if return_mode not in {"rendered", "meta", "both"}:
            return EffectOutcome.failed(f"Unknown memory_query return mode: {return_mode}")

        query = payload.get("query")
        query_text = str(query or "").strip()
        since = payload.get("since")
        until = payload.get("until")
        tags = payload.get("tags")
        tags_dict: Optional[Dict[str, Any]] = None
        if isinstance(tags, dict):
            # Accept str or list[str] values. Ignore reserved key "kind".
            out_tags: Dict[str, Any] = {}
            for k, v in tags.items():
                if not isinstance(k, str) or not k.strip():
                    continue
                if k == "kind":
                    continue
                if isinstance(v, str) and v.strip():
                    out_tags[k.strip()] = v.strip()
                elif isinstance(v, (list, tuple)):
                    vals = [str(x).strip() for x in v if isinstance(x, str) and str(x).strip()]
                    if vals:
                        out_tags[k.strip()] = vals
            tags_dict = out_tags or None

        tags_mode_raw = payload.get("tags_mode")
        if tags_mode_raw is None:
            tags_mode_raw = payload.get("tagsMode")
        if tags_mode_raw is None:
            tags_mode_raw = payload.get("tag_mode")
        tags_mode = str(tags_mode_raw or "all").strip().lower() or "all"
        if tags_mode in {"and"}:
            tags_mode = "all"
        if tags_mode in {"or"}:
            tags_mode = "any"
        if tags_mode not in {"all", "any"}:
            tags_mode = "all"

        def _norm_str_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                v = value.strip()
                return [v] if v else []
            if not isinstance(value, list):
                return []
            out: list[str] = []
            for x in value:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            # preserve order but dedup (case-insensitive)
            seen: set[str] = set()
            deduped: list[str] = []
            for s in out:
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(s)
            return deduped

        authors = _norm_str_list(payload.get("authors") if "authors" in payload else payload.get("usernames"))
        if not authors:
            authors = _norm_str_list(payload.get("users"))
        locations = _norm_str_list(payload.get("locations") if "locations" in payload else payload.get("location"))

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
            max_messages = int(payload.get("max_messages", -1) or -1)
        except Exception:
            max_messages = -1
        # `-1` means "no truncation" for rendered messages.
        if max_messages < -1:
            max_messages = -1
        if max_messages != -1 and max_messages < 1:
            max_messages = 1

        from ..memory.active_context import ActiveContextPolicy, TimeRange

        # Select run(s) to query.
        runs_to_query: list[RunState] = []
        if scope == "run":
            runs_to_query = [run]
        elif scope == "session":
            runs_to_query = [self._resolve_scope_owner_run(run, scope="session")]
        elif scope == "global":
            runs_to_query = [self._resolve_scope_owner_run(run, scope="global")]
        else:  # all
            # Deterministic order; dedup by run_id.
            root = self._resolve_scope_owner_run(run, scope="session")
            global_run = self._resolve_scope_owner_run(run, scope="global")
            seen_ids: set[str] = set()
            for r in (run, root, global_run):
                if r.run_id in seen_ids:
                    continue
                seen_ids.add(r.run_id)
                runs_to_query.append(r)

        # Collect per-run span indexes (metadata) and summary maps for rendering.
        spans_by_run_id: dict[str, list[dict[str, Any]]] = {}
        all_spans: list[dict[str, Any]] = []
        all_summary_by_artifact: dict[str, str] = {}
        for target in runs_to_query:
            spans = ActiveContextPolicy.list_memory_spans_from_run(target)
            # `memory_spans` is a general span-like index (conversation spans, notes, evidence, etc).
            # MEMORY_QUERY is specifically for provenance-first *memory recall*, not evidence retrieval.
            spans = [s for s in spans if not (isinstance(s, dict) and str(s.get("kind") or "") == "evidence")]
            spans_by_run_id[target.run_id] = spans
            all_spans.extend([dict(s) for s in spans if isinstance(s, dict)])
            all_summary_by_artifact.update(ActiveContextPolicy.summary_text_by_artifact_id_from_run(target))

        # Resolve explicit span ids if provided.
        span_id_payload = payload.get("span_id")
        span_ids_payload = payload.get("span_ids")
        explicit_ids = span_ids_payload if isinstance(span_ids_payload, list) else span_id_payload

        all_selected: list[str] = []

        if explicit_ids is not None:
            explicit_list = list(explicit_ids) if isinstance(explicit_ids, list) else [explicit_ids]

            # Indices are inherently scoped to a single run's span list; for `scope="all"`,
            # require stable artifact ids to avoid ambiguity.
            if scope == "all":
                for x in explicit_list:
                    if isinstance(x, int):
                        return EffectOutcome.failed("memory_query scope='all' requires explicit span_ids as artifact ids (no indices)")
                    if isinstance(x, str) and x.strip().isdigit():
                        return EffectOutcome.failed("memory_query scope='all' requires explicit span_ids as artifact ids (no indices)")
                # Treat as artifact ids.
                all_selected = _dedup_preserve_order([str(x).strip() for x in explicit_list if str(x).strip()])
            else:
                # Single-run resolution for indices.
                target = runs_to_query[0]
                spans = spans_by_run_id.get(target.run_id, [])
                all_selected = ActiveContextPolicy.resolve_span_ids_from_spans(explicit_list, spans)
        else:
            # Filter spans per target and union.
            time_range = None
            if since or until:
                time_range = TimeRange(
                    start=str(since) if since else None,
                    end=str(until) if until else None,
                )

            for target in runs_to_query:
                spans = spans_by_run_id.get(target.run_id, [])
                matches = ActiveContextPolicy.filter_spans_from_run(
                    target,
                    artifact_store=artifact_store,
                    time_range=time_range,
                    tags=tags_dict,
                    tags_mode=tags_mode,
                    authors=authors or None,
                    locations=locations or None,
                    query=query_text or None,
                    limit=limit_spans,
                )
                selected = [str(s.get("artifact_id") or "") for s in matches if isinstance(s, dict) and s.get("artifact_id")]

                if deep_enabled and query_text:
                    # Deep scan is bounded and should respect metadata filters (tags/authors/locations/time).
                    deep_candidates = ActiveContextPolicy.filter_spans_from_run(
                        target,
                        artifact_store=artifact_store,
                        time_range=time_range,
                        tags=tags_dict,
                        tags_mode=tags_mode,
                        authors=authors or None,
                        locations=locations or None,
                        query=None,
                        limit=deep_limit_spans,
                    )
                    selected = _dedup_preserve_order(
                        selected
                        + _deep_scan_span_ids(
                            spans=deep_candidates,
                            artifact_store=artifact_store,
                            query=query_text,
                            limit_spans=deep_limit_spans,
                            limit_messages_per_span=deep_limit_messages_per_span,
                        )
                    )

                if connected and selected:
                    connect_candidates = ActiveContextPolicy.filter_spans_from_run(
                        target,
                        artifact_store=artifact_store,
                        time_range=time_range,
                        tags=tags_dict,
                        tags_mode=tags_mode,
                        authors=authors or None,
                        locations=locations or None,
                        query=None,
                        limit=max(1000, len(spans)),
                    )
                    selected = _dedup_preserve_order(
                        _expand_connected_span_ids(
                            spans=connect_candidates,
                            seed_artifact_ids=selected,
                            connect_keys=connect_keys,
                            neighbor_hops=neighbor_hops,
                            limit=max(limit_spans, len(selected)),
                        )
                    )

                all_selected = _dedup_preserve_order(all_selected + selected)

        rendered_text = ""
        if return_mode in {"rendered", "both"}:
            # Render output (provenance + messages). Note: this may load artifacts.
            rendered_text = _render_memory_query_output(
                spans=all_spans,
                artifact_store=artifact_store,
                selected_artifact_ids=all_selected,
                summary_by_artifact=all_summary_by_artifact,
                max_messages=max_messages,
            )

        # Structured meta output (for deterministic workflows).
        meta: dict[str, Any] = {}
        if return_mode in {"meta", "both"}:
            # Index span record by artifact id (first match wins deterministically).
            by_artifact: dict[str, dict[str, Any]] = {}
            for s in all_spans:
                try:
                    aid = str(s.get("artifact_id") or "").strip()
                except Exception:
                    aid = ""
                if not aid or aid in by_artifact:
                    continue
                by_artifact[aid] = s

            matches: list[dict[str, Any]] = []
            for aid in all_selected:
                span = by_artifact.get(aid)
                if not isinstance(span, dict):
                    continue
                m: dict[str, Any] = {
                    "span_id": aid,
                    "kind": span.get("kind"),
                    "created_at": span.get("created_at"),
                    "from_timestamp": span.get("from_timestamp"),
                    "to_timestamp": span.get("to_timestamp"),
                    "tags": span.get("tags") if isinstance(span.get("tags"), dict) else {},
                }
                for k in ("created_by", "location"):
                    if k in span:
                        m[k] = span.get(k)
                # Include known preview fields without enforcing a global schema.
                for k in ("note_preview", "message_count", "summary_message_id"):
                    if k in span:
                        m[k] = span.get(k)
                matches.append(m)

            meta = {"matches": matches, "span_ids": list(all_selected)}

        result = {
            "mode": "executed",
            "results": [
                {
                    "call_id": call_id,
                    "name": tool_name,
                    "success": True,
                    "output": rendered_text if return_mode in {"rendered", "both"} else "",
                    "error": None,
                    "meta": meta if meta else None,
                }
            ],
        }
        return EffectOutcome.completed(result=result)

    def _handle_vars_query(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Handle VARS_QUERY.

        This is a JSON-safe, runtime-owned introspection primitive intended for:
        - progressive recall/debugging (e.g., inspect `scratchpad`)
        - host tooling parity (schema-only tools that map to runtime effects)

        Payload (all optional unless stated):
          - path: str                 (default "scratchpad"; supports dot path or JSON pointer "/a/b/0")
          - keys_only: bool           (default False; when True, return keys/length instead of full value)
          - target_run_id: str        (optional; inspect another run state)
          - tool_name: str            (default "inspect_vars"; for tool-style output)
          - call_id: str              (tool-call id passthrough)
        """
        import json

        from .vars import ensure_namespaces, parse_vars_path, resolve_vars_path

        payload = dict(effect.payload or {})
        tool_name = str(payload.get("tool_name") or "inspect_vars")
        call_id = str(payload.get("call_id") or "vars")

        target_run_id = payload.get("target_run_id")
        if target_run_id is not None:
            target_run_id = str(target_run_id).strip() or None

        path = payload.get("path")
        if path is None:
            path = payload.get("var_path")
        path_text = str(path or "").strip() or "scratchpad"

        keys_only = bool(payload.get("keys_only", False))

        target_run = run
        if target_run_id and target_run_id != run.run_id:
            loaded = self._run_store.load(target_run_id)
            if loaded is None:
                return EffectOutcome.completed(
                    result={
                        "mode": "executed",
                        "results": [
                            {
                                "call_id": call_id,
                                "name": tool_name,
                                "success": False,
                                "output": None,
                                "error": f"Unknown target_run_id: {target_run_id}",
                            }
                        ],
                    }
                )
            target_run = loaded

        ensure_namespaces(target_run.vars)

        try:
            tokens = parse_vars_path(path_text)
            value = resolve_vars_path(target_run.vars, tokens)
        except Exception as e:
            return EffectOutcome.completed(
                result={
                    "mode": "executed",
                    "results": [
                        {
                            "call_id": call_id,
                            "name": tool_name,
                            "success": False,
                            "output": None,
                            "error": str(e),
                        }
                    ],
                }
            )

        out: Dict[str, Any] = {"path": path_text, "type": type(value).__name__}
        if keys_only:
            if isinstance(value, dict):
                out["keys"] = sorted([str(k) for k in value.keys()])
            elif isinstance(value, list):
                out["length"] = len(value)
            else:
                out["value"] = value
        else:
            out["value"] = value

        text = json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, default=str)

        return EffectOutcome.completed(
            result={
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
        )

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
        # 1) LLM summary - use integration layer summarizer if available
        # ------------------------------------------------------------------
        #
        # When chat_summarizer is injected (from AbstractCore integration layer),
        # use it for adaptive chunking based on max_tokens. This handles cases
        # where the environment can't use the model's full context window
        # (e.g., GPU memory constraints).
        #
        # When max_tokens == -1 (AUTO): Uses model's full capability
        # When max_tokens > 0: Chunks messages if they exceed the limit

        sub_run_id: Optional[str] = None  # Track for provenance if using fallback

        if self._chat_summarizer is not None:
            # Use AbstractCore's BasicSummarizer with adaptive chunking
            try:
                summarizer_result = self._chat_summarizer.summarize_chat_history(
                    messages=split.older_messages,
                    preserve_recent=0,  # Already split; don't preserve again
                    focus=focus_text,
                    compression_mode=compression_mode,
                )
                summary_text_out = summarizer_result.get("summary", "(summary unavailable)")
                key_points = list(summarizer_result.get("key_points") or [])
                confidence = summarizer_result.get("confidence")
            except Exception as e:
                return EffectOutcome.failed(f"Summarizer failed: {e}")
        else:
            # Fallback: Original prompt-based approach (for non-AbstractCore runtimes)
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
        span_record: Dict[str, Any] = {
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
        if run.actor_id:
            span_record["created_by"] = str(run.actor_id)
        spans.append(span_record)

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
          - target_run_id: str       (optional; defaults to current run_id)
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

        artifact_store = self._artifact_store
        if artifact_store is None:
            return EffectOutcome.failed(
                "MEMORY_NOTE requires an ArtifactStore; configure runtime.set_artifact_store(...)"
            )

        payload = dict(effect.payload or {})
        tool_name = str(payload.get("tool_name") or "remember_note")
        call_id = str(payload.get("call_id") or "memory")

        base_run_id = str(payload.get("target_run_id") or run.run_id).strip() or run.run_id
        base_run = run
        if base_run_id != run.run_id:
            loaded = self._run_store.load(base_run_id)
            if loaded is None:
                return EffectOutcome.failed(f"Unknown target_run_id: {base_run_id}")
            base_run = loaded
            ensure_namespaces(base_run.vars)

        scope = str(payload.get("scope") or "run").strip().lower() or "run"
        try:
            target_run = self._resolve_scope_owner_run(base_run, scope=scope)
        except Exception as e:
            return EffectOutcome.failed(str(e))
        ensure_namespaces(target_run.vars)

        target_runtime_ns = target_run.vars.get("_runtime")
        if not isinstance(target_runtime_ns, dict):
            target_runtime_ns = {}
            target_run.vars["_runtime"] = target_runtime_ns
        spans = target_runtime_ns.get("memory_spans")
        if not isinstance(spans, list):
            spans = []
            target_runtime_ns["memory_spans"] = spans

        note = payload.get("note")
        note_text = str(note or "").strip()
        if not note_text:
            return EffectOutcome.failed("MEMORY_NOTE requires payload.note (non-empty string)")

        location_raw = payload.get("location")
        location = str(location_raw).strip() if isinstance(location_raw, str) else ""

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

        # Provenance default: the run that emitted this effect (not the scope owner).
        source_run_id = str(sources_dict.get("run_id") or run.run_id).strip() or run.run_id
        span_ids = _norm_list(sources_dict.get("span_ids"))
        message_ids = _norm_list(sources_dict.get("message_ids"))

        created_at = utc_now_iso()
        artifact_payload: Dict[str, Any] = {
            "note": note_text,
            "sources": {"run_id": source_run_id, "span_ids": span_ids, "message_ids": message_ids},
            "created_at": created_at,
        }
        if location:
            artifact_payload["location"] = location
        if run.actor_id:
            artifact_payload["actor_id"] = str(run.actor_id)
        session_id = getattr(target_run, "session_id", None) or getattr(run, "session_id", None)
        if session_id:
            artifact_payload["session_id"] = str(session_id)

        artifact_tags: Dict[str, str] = {"kind": "memory_note"}
        artifact_tags.update(clean_tags)
        meta = artifact_store.store_json(artifact_payload, run_id=target_run.run_id, tags=artifact_tags)
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
        if location:
            span_record["location"] = location
        if clean_tags:
            span_record["tags"] = dict(clean_tags)
        if span_ids or message_ids:
            span_record["sources"] = {"run_id": source_run_id, "span_ids": span_ids, "message_ids": message_ids}
        if run.actor_id:
            span_record["created_by"] = str(run.actor_id)

        spans.append(span_record)

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    return float(value) != 0.0
                except Exception:
                    return False
            if isinstance(value, str):
                s = value.strip().lower()
                if not s:
                    return False
                if s in {"false", "0", "no", "off"}:
                    return False
                if s in {"true", "1", "yes", "on"}:
                    return True
            return False

        # Optional UX convenience: keep the stored note immediately visible to downstream LLM calls by
        # rehydrating it into `base_run.context.messages` as a synthetic system message.
        keep_raw = payload.get("keep_in_context")
        if keep_raw is None:
            keep_raw = payload.get("keepInContext")
        keep_in_context = _coerce_bool(keep_raw)
        kept: Optional[Dict[str, Any]] = None
        if keep_in_context:
            try:
                from ..memory.active_context import ActiveContextPolicy

                policy = ActiveContextPolicy(run_store=self._run_store, artifact_store=artifact_store)
                out = policy.rehydrate_into_context_from_run(
                    base_run,
                    span_ids=[artifact_id],
                    placement="end",
                    dedup_by="message_id",
                    max_messages=1,
                )
                kept = {"inserted": out.get("inserted", 0), "skipped": out.get("skipped", 0)}

                # Persist when mutating a different run than the currently executing one.
                if base_run is not run:
                    base_run.updated_at = utc_now_iso()
                    self._run_store.save(base_run)
            except Exception as e:
                kept = {"inserted": 0, "skipped": 0, "error": str(e)}

        if target_run is not run:
            target_run.updated_at = utc_now_iso()
            self._run_store.save(target_run)

        rendered_tags = json.dumps(clean_tags, ensure_ascii=False, sort_keys=True) if clean_tags else "{}"
        text = f"Stored memory_note span_id={artifact_id} tags={rendered_tags}"
        meta_out: Dict[str, Any] = {"span_id": artifact_id, "created_at": created_at, "note_preview": preview}
        if isinstance(kept, dict):
            meta_out["kept_in_context"] = kept

        result = {
            "mode": "executed",
            "results": [
                {
                    "call_id": call_id,
                    "name": tool_name,
                    "success": True,
                    "output": text,
                    "error": None,
                    "meta": meta_out,
                }
            ],
        }
        return EffectOutcome.completed(result=result)

    def _handle_memory_rehydrate(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Handle MEMORY_REHYDRATE.

        This is a runtime-owned, deterministic mutation of `context.messages`:
        - loads archived conversation span artifacts from ArtifactStore
        - inserts them into `context.messages` with dedup
        - persists the mutated run (RunStore checkpoint)

        Payload (required unless stated):
          - span_ids: list[str|int]  (required; artifact ids preferred; indices allowed)
          - placement: str          ("after_summary"|"after_system"|"end", default "after_summary")
          - dedup_by: str           (default "message_id")
          - max_messages: int       (optional; max inserted messages)
          - target_run_id: str      (optional; defaults to current run)
        """
        from .vars import ensure_namespaces

        ensure_namespaces(run.vars)
        artifact_store = self._artifact_store
        if artifact_store is None:
            return EffectOutcome.failed(
                "MEMORY_REHYDRATE requires an ArtifactStore; configure runtime.set_artifact_store(...)"
            )

        payload = dict(effect.payload or {})
        target_run_id = str(payload.get("target_run_id") or run.run_id).strip() or run.run_id

        # Normalize span_ids (accept legacy `span_id` too).
        raw_span_ids = payload.get("span_ids")
        if raw_span_ids is None:
            raw_span_ids = payload.get("span_id")
        span_ids: list[Any] = []
        if isinstance(raw_span_ids, list):
            span_ids = list(raw_span_ids)
        elif raw_span_ids is not None:
            span_ids = [raw_span_ids]
        if not span_ids:
            return EffectOutcome.failed("MEMORY_REHYDRATE requires payload.span_ids (non-empty list)")

        placement = str(payload.get("placement") or "after_summary").strip() or "after_summary"
        dedup_by = str(payload.get("dedup_by") or "message_id").strip() or "message_id"
        max_messages = payload.get("max_messages")

        # Load the target run (may be different from current).
        target_run = run
        if target_run_id != run.run_id:
            loaded = self._run_store.load(target_run_id)
            if loaded is None:
                return EffectOutcome.failed(f"Unknown target_run_id: {target_run_id}")
            target_run = loaded
            ensure_namespaces(target_run.vars)

        # Best-effort: rehydrate only span kinds that are meaningful to inject into
        # `context.messages` for downstream LLM calls.
        #
        # Rationale:
        # - conversation_span: archived chat messages
        # - memory_note: durable notes (rehydrated as a synthetic message by ActiveContextPolicy)
        #
        # Evidence and other span kinds are intentionally skipped by default.
        from ..memory.active_context import ActiveContextPolicy

        spans = ActiveContextPolicy.list_memory_spans_from_run(target_run)
        resolved = ActiveContextPolicy.resolve_span_ids_from_spans(span_ids, spans)
        if not resolved:
            return EffectOutcome.completed(result={"inserted": 0, "skipped": 0, "artifacts": []})

        kind_by_artifact: dict[str, str] = {}
        for s in spans:
            if not isinstance(s, dict):
                continue
            aid = str(s.get("artifact_id") or "").strip()
            if not aid or aid in kind_by_artifact:
                continue
            kind_by_artifact[aid] = str(s.get("kind") or "").strip()

        to_rehydrate: list[str] = []
        skipped_artifacts: list[dict[str, Any]] = []
        allowed_kinds = {"conversation_span", "memory_note"}
        for aid in resolved:
            kind = kind_by_artifact.get(aid, "")
            if kind and kind not in allowed_kinds:
                skipped_artifacts.append(
                    {"span_id": aid, "inserted": 0, "skipped": 0, "error": None, "kind": kind}
                )
                continue
            to_rehydrate.append(aid)

        # Reuse the canonical policy implementation (no duplicated logic).
        # Mutate the in-memory RunState to keep runtime tick semantics consistent.
        policy = ActiveContextPolicy(run_store=self._run_store, artifact_store=artifact_store)
        out = policy.rehydrate_into_context_from_run(
            target_run,
            span_ids=to_rehydrate,
            placement=placement,
            dedup_by=dedup_by,
            max_messages=max_messages,
        )

        # Persist when mutating a different run than the currently executing one.
        if target_run is not run:
            target_run.updated_at = utc_now_iso()
            self._run_store.save(target_run)

        # Normalize output shape to match backlog expectations (`span_id` field, optional kind).
        artifacts_out: list[dict[str, Any]] = []
        artifacts = out.get("artifacts")
        if isinstance(artifacts, list):
            for a in artifacts:
                if not isinstance(a, dict):
                    continue
                aid = str(a.get("artifact_id") or "").strip()
                artifacts_out.append(
                    {
                        "span_id": aid,
                        "inserted": a.get("inserted"),
                        "skipped": a.get("skipped"),
                        "error": a.get("error"),
                        "kind": kind_by_artifact.get(aid) or None,
                        "preview": a.get("preview"),
                    }
                )
        artifacts_out.extend(skipped_artifacts)

        return EffectOutcome.completed(
            result={
                "inserted": out.get("inserted", 0),
                "skipped": out.get("skipped", 0),
                "artifacts": artifacts_out,
            }
        )


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

    remaining: Optional[int] = None if int(max_messages) == -1 else int(max_messages)
    for i, aid in enumerate(selected_artifact_ids, start=1):
        span = span_by_id.get(aid, {})
        kind = span.get("kind") or "span"
        created = span.get("created_at") or ""
        from_ts = span.get("from_timestamp") or ""
        to_ts = span.get("to_timestamp") or ""
        count = span.get("message_count") or ""
        created_by = span.get("created_by") or ""
        location = span.get("location") or ""
        tags = span.get("tags") if isinstance(span.get("tags"), dict) else {}
        tags_txt = ", ".join([f"{k}={v}" for k, v in sorted(tags.items()) if isinstance(v, str) and v])

        lines.append("")
        lines.append(f"[{i}] span_id={aid} kind={kind} msgs={count} created_at={created}")
        if from_ts or to_ts:
            lines.append(f"    time_range: {from_ts} .. {to_ts}")
        if isinstance(created_by, str) and str(created_by).strip():
            lines.append(f"    created_by: {str(created_by).strip()}")
        if isinstance(location, str) and str(location).strip():
            lines.append(f"    location: {str(location).strip()}")
        if tags_txt:
            lines.append(f"    tags: {tags_txt}")

        summary = summary_by_artifact.get(aid)
        if summary:
            lines.append(f"    summary: {str(summary).strip()}")

        if remaining is not None and remaining <= 0:
            continue

        payload = artifact_store.load_json(aid)
        if not isinstance(payload, dict):
            lines.append("    (artifact payload unavailable)")
            continue
        if kind == "memory_note" or "note" in payload:
            note = str(payload.get("note") or "").strip()
            if note:
                lines.append("    note: " + note)
            else:
                lines.append("    (note payload missing note text)")

            if not (isinstance(location, str) and location.strip()):
                loc = payload.get("location")
                if isinstance(loc, str) and loc.strip():
                    lines.append(f"    location: {loc.strip()}")

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
            if remaining is not None and remaining <= 0:
                break
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "unknown")
            content = str(m.get("content") or "")
            ts = str(m.get("timestamp") or "")
            prefix = f"    - {role}: "
            if ts:
                prefix = f"    - {ts} {role}: "
            lines.append(prefix + content)
            rendered += 1
            if remaining is not None:
                remaining -= 1

        total = sum(1 for m in messages if isinstance(m, dict))
        if remaining is not None and rendered < total:
            lines.append(f"    (remaining {total - rendered} messages omitted by max_messages={int(max_messages)})")

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
