"""abstractruntime.core.models

Core data model for AbstractRuntime (v0.1).

Design intent:
- Keep everything JSON-serializable (durable execution)
- Separate *what to do* (Effect) from *how to do it* (EffectHandler)
- Represent long pauses explicitly (WaitState), never by keeping Python stacks alive

We intentionally keep this module dependency-light (stdlib only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WaitReason(str, Enum):
    EVENT = "event"         # arbitrary external signal
    UNTIL = "until"         # time-based
    USER = "user"           # human-in-the-loop
    JOB = "job"             # external job completion
    SUBWORKFLOW = "subworkflow"  # waiting for child workflow


class EffectType(str, Enum):
    """Side-effects a node can request."""

    # Pure waiting primitives
    WAIT_EVENT = "wait_event"
    WAIT_UNTIL = "wait_until"
    ASK_USER = "ask_user"
    ANSWER_USER = "answer_user"

    # Eventing
    EMIT_EVENT = "emit_event"

    # Integrations (implemented via pluggable handlers)
    LLM_CALL = "llm_call"
    TOOL_CALLS = "tool_calls"
    MEMORY_QUERY = "memory_query"
    MEMORY_TAG = "memory_tag"
    MEMORY_COMPACT = "memory_compact"
    MEMORY_NOTE = "memory_note"
    MEMORY_REHYDRATE = "memory_rehydrate"

    # Debug / inspection (schema-only tools -> runtime effects)
    VARS_QUERY = "vars_query"

    # Composition
    START_SUBWORKFLOW = "start_subworkflow"


@dataclass(frozen=True)
class Effect:
    """A request for an external side-effect.

    Notes:
    - Effects must be serializable (payload is JSON-like).
    - `result_key` specifies where the effect result is stored in run state variables.
    """

    type: EffectType
    payload: Dict[str, Any] = field(default_factory=dict)
    result_key: Optional[str] = None


@dataclass(frozen=True)
class StepPlan:
    """What the runtime should do next for a node."""

    node_id: str
    effect: Optional[Effect] = None
    next_node: Optional[str] = None

    # If set, the runtime completes the run immediately.
    complete_output: Optional[Dict[str, Any]] = None


@dataclass
class WaitState:
    """Represents a durable pause.

    The run can be resumed by calling `resume(run_id, event)`.

    - For EVENT/USER/JOB: `wait_key` identifies which event unblocks the run.
    - For UNTIL: `until` specifies when the run can continue.

    `resume_to_node` defines where execution continues after resume.
    `result_key` tells where to store the resume payload.
    """

    reason: WaitReason
    wait_key: Optional[str] = None
    until: Optional[str] = None  # ISO timestamp

    resume_to_node: Optional[str] = None
    result_key: Optional[str] = None

    prompt: Optional[str] = None
    choices: Optional[List[str]] = None
    allow_free_text: bool = True

    # Optional structured details for non-user waits (e.g. tool passthrough).
    # Must be JSON-serializable.
    details: Optional[Dict[str, Any]] = None


@dataclass
class RunState:
    """Durable state for a workflow run."""

    run_id: str
    workflow_id: str
    status: RunStatus
    current_node: str

    vars: Dict[str, Any] = field(default_factory=dict)

    waiting: Optional[WaitState] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    updated_at: str = field(default_factory=lambda: utc_now().isoformat())

    # Optional provenance fields
    actor_id: Optional[str] = None
    session_id: Optional[str] = None
    parent_run_id: Optional[str] = None  # For subworkflow tracking

    @classmethod
    def new(
        cls,
        *,
        workflow_id: str,
        entry_node: str,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        vars: Optional[Dict[str, Any]] = None,
        parent_run_id: Optional[str] = None,
    ) -> "RunState":
        return cls(
            run_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            status=RunStatus.RUNNING,
            current_node=entry_node,
            vars=vars or {},
            actor_id=actor_id,
            session_id=session_id,
            parent_run_id=parent_run_id,
        )


class StepStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    WAITING = "waiting"
    FAILED = "failed"


@dataclass
class StepRecord:
    """One append-only ledger entry (journal d'exécution)."""

    run_id: str
    step_id: str
    node_id: str
    status: StepStatus

    effect: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    started_at: str = field(default_factory=lambda: utc_now().isoformat())
    ended_at: Optional[str] = None

    # Optional provenance/integrity
    actor_id: Optional[str] = None
    session_id: Optional[str] = None

    # Retry and idempotency fields
    attempt: int = 1  # Current attempt number (1-indexed)
    idempotency_key: Optional[str] = None  # For deduplication on restart

    # Tamper-evident chain fields (optional in v0.1; filled by a chained LedgerStore).
    prev_hash: Optional[str] = None
    record_hash: Optional[str] = None
    signature: Optional[str] = None

    @classmethod
    def start(
        cls,
        *,
        run: RunState,
        node_id: str,
        effect: Optional[Effect],
        attempt: int = 1,
        idempotency_key: Optional[str] = None,
    ) -> "StepRecord":
        return cls(
            run_id=run.run_id,
            step_id=str(uuid.uuid4()),
            node_id=node_id,
            status=StepStatus.STARTED,
            effect={
                "type": effect.type.value,
                "payload": effect.payload,
                "result_key": effect.result_key,
            } if effect else None,
            actor_id=run.actor_id,
            session_id=getattr(run, "session_id", None),
            attempt=attempt,
            idempotency_key=idempotency_key,
        )

    def finish_success(self, result: Optional[Dict[str, Any]] = None) -> "StepRecord":
        self.status = StepStatus.COMPLETED
        self.result = result
        self.ended_at = utc_now().isoformat()
        return self

    def finish_waiting(self, wait_state: WaitState) -> "StepRecord":
        self.status = StepStatus.WAITING
        self.result = {
            "wait": {
                "reason": wait_state.reason.value,
                "wait_key": wait_state.wait_key,
                "until": wait_state.until,
                "resume_to_node": wait_state.resume_to_node,
                "result_key": wait_state.result_key,
                # Optional fields for richer audit/debugging
                "prompt": wait_state.prompt,
                "choices": wait_state.choices,
                "allow_free_text": wait_state.allow_free_text,
                "details": wait_state.details,
            }
        }
        self.ended_at = utc_now().isoformat()
        return self

    def finish_failure(self, error: str) -> "StepRecord":
        self.status = StepStatus.FAILED
        self.error = error
        self.ended_at = utc_now().isoformat()
        return self


@dataclass
class LimitWarning:
    """Warning about approaching or exceeding a runtime limit.

    Generated by Runtime.check_limits() to proactively notify about
    resource constraints before they cause failures.

    Attributes:
        limit_type: Type of limit ("iterations", "tokens", "history")
        status: Warning status ("warning" at threshold, "exceeded" at limit)
        current: Current value of the resource
        maximum: Maximum allowed value
        pct: Percentage of limit used (computed in __post_init__)

    Example:
        >>> warning = LimitWarning("iterations", "warning", 20, 25)
        >>> warning.pct
        80.0
    """

    limit_type: str  # "iterations", "tokens", "history"
    status: str  # "warning", "exceeded"
    current: int
    maximum: int
    pct: float = 0.0

    def __post_init__(self) -> None:
        if self.maximum > 0:
            self.pct = round(self.current / self.maximum * 100, 1)
