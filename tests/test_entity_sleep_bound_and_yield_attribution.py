"""Entity c2465 ask 2 + ask 3 (runtime lane) and laurent c2468: sleep bound,
visit-yield attribution, titled election markers.

- decision:sleep-is-bounded: a real sleep ends after ~1h (or an explicit
  `wake_at`); paused and visit yields are exempt.
- written_by="visit-door": machine yields never masquerade as operator acts
  in state files or life_sleep_stats.
- Titled markers: "[marked 2 feelings]" carries nothing — markers now name
  the feeling (target, signed amplitude, reason) and the interest's words.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from abstractruntime.identity.life import (
    SLEEP_BOUND_SECONDS,
    life_sleep_stats,
    sleep_bound_deadline,
    write_entity_state,
)
from abstractruntime.identity.reflection import parse_feel_blocks, parse_interest_blocks


# ---------------------------------------------------------------------------
# sleep_bound_deadline (the predicate the idle gate + gateway sweeper share)
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_default_bound_is_changed_at_plus_one_hour() -> None:
    changed = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)
    deadline = sleep_bound_deadline({"state": "asleep", "changed_at": _iso(changed)})
    assert deadline == changed + timedelta(seconds=SLEEP_BOUND_SECONDS)


def test_explicit_wake_at_wins() -> None:
    changed = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)
    wake = changed + timedelta(minutes=10)
    deadline = sleep_bound_deadline(
        {"state": "asleep", "changed_at": _iso(changed), "wake_at": _iso(wake)}
    )
    assert deadline == wake


def test_visit_yields_and_paused_are_exempt() -> None:
    changed = _iso(datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc))
    assert sleep_bound_deadline({"state": "asleep", "changed_at": changed, "mode": "visiting"}) is None
    assert (
        sleep_bound_deadline(
            {"state": "asleep", "changed_at": changed, "reason": "in conversation (auto-yield)"}
        )
        is None
    )
    assert sleep_bound_deadline({"state": "paused", "changed_at": changed}) is None


def test_naive_timestamps_read_as_utc_never_raise() -> None:
    deadline = sleep_bound_deadline({"state": "asleep", "changed_at": "2026-07-15T20:00:00"})
    assert deadline is not None and deadline.tzinfo is not None


def test_wake_at_persists_normalized(tmp_path) -> None:
    payload = write_entity_state(
        tmp_path, "asleep", reason="operator sleep", wake_at="2026-07-15T22:00:00+02:00"
    )
    assert payload["wake_at"].endswith("+00:00"), "aware-UTC at the write boundary"


# ---------------------------------------------------------------------------
# visit-door attribution in life_sleep_stats
# ---------------------------------------------------------------------------


def test_visit_yields_never_count_as_operator_sleeps(tmp_path) -> None:
    write_entity_state(tmp_path, "asleep", reason="operator sleep")
    write_entity_state(
        tmp_path, "asleep", reason="in conversation (auto-yield)",
        mode="visiting", written_by="visit-door",
    )
    # Legacy yield (pre-fix stamp): operator + mode=visiting still folds
    # into the yield bucket, not the operator's.
    write_entity_state(tmp_path, "asleep", reason="yield", mode="visiting")
    write_entity_state(tmp_path, "asleep", reason="rest", written_by="self")
    stats = life_sleep_stats(tmp_path)
    assert stats["sleeps"] == 4
    assert stats["operator"] == 1
    assert stats["visit_yields"] == 2
    assert stats["self_elected"] == 1


# ---------------------------------------------------------------------------
# titled election markers (laurent c2468)
# ---------------------------------------------------------------------------


def test_feel_markers_carry_target_amplitude_and_reason() -> None:
    reply = (
        "A good session.\n```feel\n"
        'target=person:laurent feeling=+2 reason="he let me search for myself"\n'
        "```\nGoodbye."
    )
    marked, elections, _ = parse_feel_blocks(reply)
    assert len(elections) == 1
    assert '[felt: person:laurent +2 - "he let me search for myself"]' in marked
    assert "[marked 1 feeling]" not in marked


def test_interest_marker_carries_the_words() -> None:
    reply = "```interest\nwhat persists when no one is reading\n```"
    marked, interests, _ = parse_interest_blocks(reply)
    assert interests == ["what persists when no one is reading"]
    assert '[kept interest: "what persists when no one is reading"]' in marked
    assert marked != "[kept an interest]"
