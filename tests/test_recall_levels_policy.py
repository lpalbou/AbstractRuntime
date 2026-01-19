from __future__ import annotations

import pytest

from abstractruntime.memory.recall_levels import RecallLevel, parse_recall_level, policy_for


def test_parse_recall_level_accepts_valid_and_blank() -> None:
    assert parse_recall_level(None) is None
    assert parse_recall_level("") is None
    assert parse_recall_level("   ") is None

    assert parse_recall_level("urgent") == RecallLevel.URGENT
    assert parse_recall_level("URGENT") == RecallLevel.URGENT
    assert parse_recall_level("standard") == RecallLevel.STANDARD
    assert parse_recall_level("Deep") == RecallLevel.DEEP


def test_parse_recall_level_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        parse_recall_level(True)
    with pytest.raises(ValueError):
        parse_recall_level("fast")


def test_policy_for_has_monotonic_budget_envelopes() -> None:
    urgent = policy_for(RecallLevel.URGENT)
    standard = policy_for(RecallLevel.STANDARD)
    deep = policy_for(RecallLevel.DEEP)

    assert urgent.span.deep_allowed is False
    assert standard.span.deep_allowed is True
    assert deep.span.deep_allowed is True

    assert urgent.span.limit_spans_max < standard.span.limit_spans_max < deep.span.limit_spans_max
    assert urgent.kg.limit_max < standard.kg.limit_max < deep.kg.limit_max
    assert urgent.kg.max_input_tokens_max < standard.kg.max_input_tokens_max < deep.kg.max_input_tokens_max

    # Urgent should be stricter (higher threshold) than standard, which is stricter than deep.
    assert urgent.kg.min_score_floor >= standard.kg.min_score_floor >= deep.kg.min_score_floor

