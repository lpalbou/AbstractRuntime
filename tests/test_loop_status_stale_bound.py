"""The loop-corpse bound must outlast a healthy tick (2026-11-07: entity LLM
attempts may now last 1800 s, and one retry may start before the budget ends)."""
from abstractruntime.identity.chat import ENTITY_LLM_TIMEOUT_FLOOR_S
from abstractruntime.identity.life import LOOP_STATUS_STALE_SECONDS


def test_stale_bound_outlasts_two_full_llm_attempts():
    assert ENTITY_LLM_TIMEOUT_FLOOR_S >= 1800.0
    assert LOOP_STATUS_STALE_SECONDS > 2 * ENTITY_LLM_TIMEOUT_FLOOR_S
