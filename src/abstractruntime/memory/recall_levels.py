from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class RecallLevel(str, Enum):
    """Framework-wide recall effort policy (binding).

    See `docs/adr/memory-recall-levels.md`.
    """

    URGENT = "urgent"
    STANDARD = "standard"
    DEEP = "deep"


def parse_recall_level(raw: Any) -> Optional[RecallLevel]:
    """Parse a recall_level value.

    Returns:
      - None when the field is absent/blank (caller did not opt into policy).
      - RecallLevel when valid.

    Raises:
      - ValueError when a non-blank value is present but invalid (no silent fallback).
    """

    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("recall_level must be a string")
    s = str(raw).strip().lower()
    if not s:
        return None
    if s in ("urgent", "standard", "deep"):
        return RecallLevel(s)
    raise ValueError(f"Unknown recall_level: {s}")


@dataclass(frozen=True)
class SpanRecallPolicy:
    """Budgets for span-index recall (memory_query)."""

    limit_spans_default: int
    limit_spans_max: int

    deep_allowed: bool
    deep_limit_spans_max: int
    deep_limit_messages_per_span_max: int

    connected_allowed: bool
    neighbor_hops_max: int

    max_messages_default: int
    max_messages_max: int


@dataclass(frozen=True)
class RehydratePolicy:
    """Budgets for rehydration into context (memory_rehydrate)."""

    max_messages_default: int
    max_messages_max: int


@dataclass(frozen=True)
class KgQueryPolicy:
    """Budgets for KG recall (memory_kg_query)."""

    min_score_default: float
    min_score_floor: float
    limit_default: int
    limit_max: int
    max_input_tokens_default: int
    max_input_tokens_max: int


@dataclass(frozen=True)
class RecallPolicy:
    level: RecallLevel
    span: SpanRecallPolicy
    rehydrate: RehydratePolicy
    kg: KgQueryPolicy


_POLICIES: dict[RecallLevel, RecallPolicy] = {
    RecallLevel.URGENT: RecallPolicy(
        level=RecallLevel.URGENT,
        span=SpanRecallPolicy(
            limit_spans_default=2,
            limit_spans_max=3,
            deep_allowed=False,
            deep_limit_spans_max=0,
            deep_limit_messages_per_span_max=0,
            connected_allowed=False,
            neighbor_hops_max=0,
            max_messages_default=30,
            max_messages_max=60,
        ),
        rehydrate=RehydratePolicy(max_messages_default=30, max_messages_max=60),
        kg=KgQueryPolicy(
            min_score_default=0.55,
            min_score_floor=0.5,
            limit_default=20,
            limit_max=40,
            max_input_tokens_default=600,
            max_input_tokens_max=1000,
        ),
    ),
    RecallLevel.STANDARD: RecallPolicy(
        level=RecallLevel.STANDARD,
        span=SpanRecallPolicy(
            limit_spans_default=5,
            limit_spans_max=8,
            deep_allowed=True,
            deep_limit_spans_max=25,
            deep_limit_messages_per_span_max=200,
            connected_allowed=True,
            neighbor_hops_max=1,
            max_messages_default=80,
            max_messages_max=150,
        ),
        rehydrate=RehydratePolicy(max_messages_default=80, max_messages_max=200),
        kg=KgQueryPolicy(
            min_score_default=0.4,
            min_score_floor=0.25,
            limit_default=80,
            limit_max=200,
            max_input_tokens_default=1200,
            max_input_tokens_max=3000,
        ),
    ),
    RecallLevel.DEEP: RecallPolicy(
        level=RecallLevel.DEEP,
        span=SpanRecallPolicy(
            limit_spans_default=8,
            limit_spans_max=20,
            deep_allowed=True,
            deep_limit_spans_max=100,
            deep_limit_messages_per_span_max=400,
            connected_allowed=True,
            neighbor_hops_max=2,
            max_messages_default=200,
            max_messages_max=600,
        ),
        rehydrate=RehydratePolicy(max_messages_default=200, max_messages_max=800),
        kg=KgQueryPolicy(
            min_score_default=0.25,
            min_score_floor=0.0,
            limit_default=200,
            limit_max=1000,
            max_input_tokens_default=2400,
            max_input_tokens_max=6000,
        ),
    ),
}


def policy_for(level: RecallLevel) -> RecallPolicy:
    return _POLICIES[level]

