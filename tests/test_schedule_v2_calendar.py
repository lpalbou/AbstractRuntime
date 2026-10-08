"""schedule@2 calendar rules (R16.1 A1/A3): daily/weekly/monthly on wall time in an IANA zone.

Every expected instant is hand-computed; nothing reads the wall clock.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from abstractruntime.triggers import ScheduleTriggerAdapter, ScheduleV2TriggerAdapter, TriggerConfigError, validate_time_zone
from abstractruntime.triggers.registry import get_trigger_adapter

V2 = ScheduleV2TriggerAdapter()
V1 = ScheduleTriggerAdapter()


def _binding(config, *, now, binding_id="b-2"):
    cfg = V2.validate(config, now=now)
    return {"binding_id": binding_id, "source_id": "schedule", "source_version": 2, "config": cfg}


def _ticks(binding, n):
    """The first n tick instants (UTC ISO) by walking prepare/admit like the controller does."""
    st = V2.initial_state(binding["config"])
    out = []
    for _ in range(n):
        wait = V2.prepare(binding, state=st, now=binding["config"]["start_at"])
        if wait["kind"] == "exhausted":
            break
        adm = V2.admit(binding, state=st, now=wait["until"])
        assert adm is not None and adm["fired_at"] == wait["until"]
        out.append(wait["until"])
        st = adm["state"]
    return out


def _local(iso, tz):
    return datetime.fromisoformat(iso).astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M %Z")


def test_registry_serves_v2_and_v1_side_by_side():
    assert isinstance(get_trigger_adapter("schedule", 2), ScheduleV2TriggerAdapter)
    assert type(get_trigger_adapter("schedule", 1)) is ScheduleTriggerAdapter
    assert V2.descriptor["version"] == 2 and V1.descriptor["version"] == 1


# ------------------------------------------------------------------ DST (A3)


@pytest.mark.parametrize(
    "tz, start, at, expected_local",
    [
        # Europe/Paris spring forward 2026-03-29 02:00 -> 03:00: 02:30 runs at 03:30 ONCE, then 02:30.
        ("Europe/Paris", "2026-03-27T00:00:00Z", "02:30",
         ["2026-03-27 02:30 CET", "2026-03-28 02:30 CET", "2026-03-29 03:30 CEST", "2026-03-30 02:30 CEST", "2026-03-31 02:30 CEST"]),
        # Europe/Paris fall back 2026-10-25 03:00 -> 02:00: 02:30 happens twice; the rule runs once (the first).
        ("Europe/Paris", "2026-10-23T12:00:00Z", "02:30",
         ["2026-10-24 02:30 CEST", "2026-10-25 02:30 CEST", "2026-10-26 02:30 CET", "2026-10-27 02:30 CET"]),
        # America/Los_Angeles spring forward 2026-03-08 02:00 -> 03:00.
        ("America/Los_Angeles", "2026-03-06T20:00:00Z", "02:30",
         ["2026-03-07 02:30 PST", "2026-03-08 03:30 PDT", "2026-03-09 02:30 PDT"]),
        # America/Los_Angeles fall back 2026-11-01 02:00 -> 01:00: 01:30 happens twice; runs once.
        ("America/Los_Angeles", "2026-10-30T20:00:00Z", "01:30",
         ["2026-10-31 01:30 PDT", "2026-11-01 01:30 PDT", "2026-11-02 01:30 PST"]),
        # A daily 08:00 keeps 08:00 local across both transitions (no 24 h drift).
        ("Europe/Paris", "2026-03-28T00:00:00Z", "08:00",
         ["2026-03-28 08:00 CET", "2026-03-29 08:00 CEST", "2026-03-30 08:00 CEST"]),
        ("Europe/Paris", "2026-10-24T00:00:00Z", "08:00",
         ["2026-10-24 08:00 CEST", "2026-10-25 08:00 CET", "2026-10-26 08:00 CET"]),
    ],
)
def test_daily_rule_is_dst_correct(tz, start, at, expected_local):
    b = _binding({"kind": "daily", "at": at, "time_zone": tz, "start_at": start}, now=start)
    got = [_local(t, tz) for t in _ticks(b, len(expected_local))]
    assert got == expected_local


def test_spring_forward_instants_in_utc():
    b = _binding({"kind": "daily", "at": "02:30", "time_zone": "Europe/Paris", "start_at": "2026-03-28T12:00:00Z"}, now="2026-03-28T12:00:00Z")
    assert _ticks(b, 2) == ["2026-03-29T01:30:00+00:00", "2026-03-30T00:30:00+00:00"]


def test_fall_back_second_wall_occurrence_never_fires_again():
    tz = "Europe/Paris"
    b = _binding({"kind": "daily", "at": "02:30", "time_zone": tz, "start_at": "2026-10-24T12:00:00Z"}, now="2026-10-24T12:00:00Z")
    st = V2.initial_state(b["config"])
    adm = V2.admit(b, state=st, now="2026-10-25T00:30:00Z")  # 02:30 CEST
    assert adm is not None and adm["fired_at"] == "2026-10-25T00:30:00+00:00"
    st = adm["state"]
    # 02:30 CET (the repeated wall time) is 01:30Z: nothing is due.
    assert V2.admit(b, state=st, now="2026-10-25T01:30:00Z") is None
    assert V2.prepare(b, state=st, now="2026-10-25T01:30:00Z") == {"kind": "until", "until": "2026-10-26T01:30:00+00:00"}


# ------------------------------------------------------------- monthly clamp


def test_monthly_day_31_clamps_to_the_last_day():
    tz = "Europe/Paris"
    b = _binding({"kind": "monthly", "day": 31, "at": "09:00", "time_zone": tz, "start_at": "2026-01-01T00:00:00Z"}, now="2026-01-01T00:00:00Z")
    got = [_local(t, tz)[:10] for t in _ticks(b, 6)]
    assert got == ["2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"]


@pytest.mark.parametrize(
    "day, start, expected",
    [
        (31, "2028-01-15T00:00:00Z", ["2028-01-31", "2028-02-29", "2028-03-31"]),  # leap year
        (29, "2027-01-15T00:00:00Z", ["2027-01-29", "2027-02-28", "2027-03-29"]),  # not a leap year
        (29, "2028-02-01T00:00:00Z", ["2028-02-29", "2028-03-29"]),               # leap day itself
        ("last", "2028-01-15T00:00:00Z", ["2028-01-31", "2028-02-29", "2028-03-31", "2028-04-30"]),
        (1, "2026-12-15T00:00:00Z", ["2027-01-01", "2027-02-01"]),
    ],
)
def test_monthly_leap_and_last(day, start, expected):
    tz = "UTC"
    b = _binding({"kind": "monthly", "day": day, "at": "06:00", "time_zone": tz, "start_at": start}, now=start)
    assert [t[:10] for t in _ticks(b, len(expected))] == expected


# ------------------------------------------------------------------- weekly


def test_weekly_across_a_year_boundary():
    tz = "Europe/Paris"
    # 2026-12-28 is a Monday.
    b = _binding({"kind": "weekly", "days": ["fri", "mon"], "at": "07:30", "time_zone": tz, "start_at": "2026-12-28T00:00:00Z"}, now="2026-12-28T00:00:00Z")
    assert b["config"]["days"] == ["mon", "fri"]  # canonical order
    got = [_local(t, tz) for t in _ticks(b, 5)]
    assert got == ["2026-12-28 07:30 CET", "2027-01-01 07:30 CET", "2027-01-04 07:30 CET", "2027-01-08 07:30 CET", "2027-01-11 07:30 CET"]


def test_weekly_first_tick_is_the_next_matching_day_at_or_after_start():
    # Start Wednesday 2026-10-07 10:00 Paris; Mon/Wed at 09:00 -> Wednesday has passed -> next Monday.
    b = _binding({"kind": "weekly", "days": ["mon", "wed"], "at": "09:00", "time_zone": "Europe/Paris", "start_at": "2026-10-07T08:00:00Z"}, now="2026-10-07T08:00:00Z")
    assert _ticks(b, 2) == ["2026-10-12T07:00:00+00:00", "2026-10-14T07:00:00+00:00"]


def test_daily_first_tick_today_when_not_yet_passed():
    b = _binding({"kind": "daily", "at": "08:00", "time_zone": "Europe/Paris", "start_at": "2026-10-08T05:00:00Z"}, now="2026-10-08T05:00:00Z")
    assert _ticks(b, 1) == ["2026-10-08T06:00:00+00:00"]
    b = _binding({"kind": "daily", "at": "08:00", "time_zone": "Europe/Paris", "start_at": "2026-10-08T06:00:00Z"}, now="2026-10-08T06:00:00Z")
    assert _ticks(b, 1) == ["2026-10-08T06:00:00+00:00"]  # exactly at start: included


# ------------------------------------------------------- catch-up policy (A3)


def test_missed_calendar_ticks_coalesce_into_one_admission():
    b = _binding({"kind": "daily", "at": "08:00", "time_zone": "Europe/Paris", "start_at": "2026-10-01T00:00:00Z"}, now="2026-10-01T00:00:00Z")
    st = V2.initial_state(b["config"])
    adm = V2.admit(b, state=st, now="2026-10-01T06:00:00Z")
    st = adm["state"]
    assert st["tick"] == 1
    # The gateway was down from Oct 1 to Oct 4 09:00 Paris: Oct 2, 3, 4 were missed -> ONE admission (Oct 4).
    adm = V2.admit(b, state=st, now="2026-10-04T07:00:00Z")
    assert adm["fired_at"] == "2026-10-04T06:00:00+00:00"
    assert adm["coalesced"] == {"first_tick": 1, "last_tick": 3, "missed_count": 2}
    assert adm["event_id"] == "schedule@2:b-2:3"
    st = adm["state"]
    assert V2.admit(b, state=st, now="2026-10-04T07:00:00Z") is None
    assert V2.prepare(b, state=st, now="2026-10-04T07:00:00Z") == {"kind": "until", "until": "2026-10-05T06:00:00+00:00"}


def test_resume_after_pause_skips_passed_calendar_ticks():
    b = _binding({"kind": "weekly", "days": ["mon"], "at": "08:00", "time_zone": "Europe/Paris", "start_at": "2026-10-05T00:00:00Z"}, now="2026-10-05T00:00:00Z")
    st = V2.initial_state(b["config"])
    st = V2.rearm(b, state=st, now="2026-10-20T12:00:00Z")  # Oct 5, 12, 19 passed while paused
    assert st["tick"] == 3 and st["scheduled_count"] == 0
    assert V2.admit(b, state=st, now="2026-10-20T12:00:00Z") is None
    assert V2.prepare(b, state=st, now="2026-10-20T12:00:00Z") == {"kind": "until", "until": "2026-10-26T07:00:00+00:00"}


def test_catch_up_policy_is_the_same_for_v1_and_v2():
    """v2 `every` reproduces v1 tick for tick (same grid, same coalescing); only the event id prefix differs."""
    cfg = {"start_at": "2026-10-01T00:00:00Z", "every": "8h", "count": 50}
    b1 = {"binding_id": "b", "source_id": "schedule", "source_version": 1, "config": V1.validate(cfg, now="2026-10-01T00:00:00Z")}
    b2 = {"binding_id": "b", "source_id": "schedule", "source_version": 2, "config": V2.validate({"kind": "every", **cfg}, now="2026-10-01T00:00:00Z")}
    s1, s2 = V1.initial_state(b1["config"]), V2.initial_state(b2["config"])
    for now in ["2026-10-01T00:00:00Z", "2026-10-02T01:00:00Z", "2026-10-02T09:00:00Z", "2026-10-05T00:00:00Z"]:
        a1, a2 = V1.admit(b1, state=s1, now=now), V2.admit(b2, state=s2, now=now)
        assert a2["event_id"] == a1["event_id"].replace("schedule@1", "schedule@2")
        a2 = {**a2, "event_id": a1["event_id"]}
        assert a1 == a2
        s1, s2 = a1["state"], a2["state"]
        assert V1.rearm(b1, state=s1, now=now) == V2.rearm(b2, state=s2, now=now)


# --------------------------------------------------------- count / until / far


def test_calendar_count_and_until():
    b = _binding({"kind": "daily", "at": "08:00", "time_zone": "UTC", "start_at": "2026-10-01T00:00:00Z", "count": 2}, now="2026-10-01T00:00:00Z")
    assert _ticks(b, 5) == ["2026-10-01T08:00:00+00:00", "2026-10-02T08:00:00+00:00"]
    b = _binding({"kind": "daily", "at": "08:00", "time_zone": "UTC", "start_at": "2026-10-01T00:00:00Z", "until": "2026-10-03T08:00:00Z"}, now="2026-10-01T00:00:00Z")
    assert _ticks(b, 5) == ["2026-10-01T08:00:00+00:00", "2026-10-02T08:00:00+00:00"]  # until is exclusive


def test_far_future_index_is_constant_time():
    b = _binding({"kind": "weekly", "days": ["tue", "sat"], "at": "23:59", "time_zone": "Pacific/Auckland", "start_at": "2026-01-01T00:00:00Z"}, now="2026-01-01T00:00:00Z")
    st = V2.initial_state(b["config"])
    t0 = _time.perf_counter()
    adm = V2.admit(b, state=st, now="2066-01-01T00:00:00Z")
    assert _time.perf_counter() - t0 < 0.05
    assert adm["coalesced"]["missed_count"] > 4000
    local = datetime.fromisoformat(adm["fired_at"]).astimezone(ZoneInfo("Pacific/Auckland"))
    assert local.strftime("%a %H:%M") in ("Tue 23:59", "Sat 23:59")


# ------------------------------------------------------------------ once/at


def test_once_wall_time_in_zone_becomes_start_at():
    cfg = V2.validate({"kind": "once", "at": "2026-10-09T08:00", "time_zone": "Europe/Paris"}, now="2026-10-08T00:00:00Z")
    assert cfg == {"kind": "once", "at": "2026-10-09T08:00", "time_zone": "Europe/Paris",
                   "start_at": "2026-10-09T06:00:00+00:00", "anchor": "2026-10-09T06:00:00+00:00"}
    # Re-validating the normalized config (an echo) is stable; a new zone moves the instant.
    assert V2.validate(cfg, now="2030-01-01T00:00:00Z") == cfg
    moved = V2.validate({**cfg, "time_zone": "America/New_York"}, now="2026-10-08T00:00:00Z")
    assert moved["start_at"] == "2026-10-09T12:00:00+00:00"


def test_every_and_once_without_kind_normalize_like_v1_plus_kind():
    assert V2.validate({"every": "5m"}, now="2026-01-01T00:00:00Z") == {
        "kind": "every", "start_at": "2026-01-01T00:00:00+00:00", "anchor": "2026-01-01T00:00:00+00:00", "every": "5m"}
    assert V2.validate({}, now="2026-01-01T00:00:00Z")["kind"] == "once"


def test_normalize_carries_source_version_2():
    b = _binding({"kind": "daily", "at": "08:00", "time_zone": "UTC"}, now="2026-01-01T00:00:00Z")
    env = V2.normalize(b, event_id="e", fired_at="2026-01-01T08:00:00+00:00", payload={"tick": 0, "scheduled_at": "2026-01-01T08:00:00+00:00"})
    assert env["source_version"] == 2 and env["source_id"] == "schedule"


# --------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "config, field",
    [
        ({"kind": "daily", "at": "08:00"}, "config.time_zone"),
        ({"kind": "daily", "at": "8:00", "time_zone": "UTC"}, "config.at"),
        ({"kind": "daily", "at": "24:00", "time_zone": "UTC"}, "config.at"),
        ({"kind": "daily", "at": "08:60", "time_zone": "UTC"}, "config.at"),
        ({"kind": "daily", "time_zone": "UTC"}, "config.at"),
        ({"kind": "daily", "at": "08:00", "time_zone": "Mars/Olympus"}, "config.time_zone"),
        ({"kind": "daily", "at": "08:00", "time_zone": "/etc/localtime"}, "config.time_zone"),
        ({"kind": "daily", "at": "08:00", "time_zone": "UTC", "every": "1d"}, "config.every"),
        ({"kind": "daily", "at": "08:00", "time_zone": "UTC", "days": ["mon"]}, "config.days"),
        ({"kind": "weekly", "at": "08:00", "time_zone": "UTC"}, "config.days"),
        ({"kind": "weekly", "at": "08:00", "time_zone": "UTC", "days": []}, "config.days"),
        ({"kind": "weekly", "at": "08:00", "time_zone": "UTC", "days": ["monday"]}, "config.days"),
        ({"kind": "monthly", "at": "08:00", "time_zone": "UTC", "day": 0}, "config.day"),
        ({"kind": "monthly", "at": "08:00", "time_zone": "UTC", "day": 32}, "config.day"),
        ({"kind": "monthly", "at": "08:00", "time_zone": "UTC", "day": "first"}, "config.day"),
        ({"kind": "monthly", "at": "08:00", "time_zone": "UTC", "day": True}, "config.day"),
        ({"kind": "hourly", "at": "08:00", "time_zone": "UTC"}, "config.kind"),
        ({"kind": "every"}, "config.every"),
        ({"kind": "every", "every": "1h", "at": "08:00"}, "config.at"),
        ({"kind": "once", "every": "1h"}, "config.every"),
        ({"kind": "once", "at": "2026-10-09T08:00"}, "config.time_zone"),
        ({"kind": "once", "at": "2026-10-09 08:00", "time_zone": "UTC"}, "config.at"),
        ({"kind": "daily", "at": "08:00", "time_zone": "UTC", "cron": "* * *"}, "config.cron"),
    ],
)
def test_validate_rejects(config, field):
    with pytest.raises(TriggerConfigError) as exc:
        V2.validate(config, now="2026-01-01T00:00:00Z")
    assert exc.value.field == field


def test_schedule_v1_still_refuses_calendar_fields():
    with pytest.raises(TriggerConfigError):
        V1.validate({"kind": "daily", "at": "08:00", "time_zone": "UTC"}, now="2026-01-01T00:00:00Z")


def test_validate_time_zone():
    assert validate_time_zone("Europe/Paris") == "Europe/Paris"
    assert validate_time_zone(" UTC ") == "UTC"
    for bad in ("", None, "Europe/Nowhere", "../../etc/passwd", 3):
        with pytest.raises(TriggerConfigError):
            validate_time_zone(bad)
