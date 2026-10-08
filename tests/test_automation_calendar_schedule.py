"""schedule@2 through the real controller (R16.1 A3): DST-correct admissions, the projected
`next_fire_at`, downtime coalescing and pause/resume — the same policy as schedule@1."""

from __future__ import annotations

import pytest

from automation_harness import Clock, at, automation_state, children, create, drive, make_runtime, make_stores
from abstractruntime.automations import apply_automation_command, get_automation
from abstractruntime.automations.ledger import automation_records

START = "2026-03-27T00:00:00+00:00"
DAILY_0230_PARIS = {"source_id": "schedule", "source_version": 2,
                    "config": {"kind": "daily", "at": "02:30", "time_zone": "Europe/Paris", "start_at": START}}


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    run_store, ledger_store = make_stores(request.param, tmp_path)
    return make_runtime(run_store, ledger_store), Clock(monkeypatch, now=START)


def _fired(runtime, aid):
    return [c.vars["_meta"]["occurrence"]["fired_at"] for c in children(runtime, aid)]


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_daily_rule_across_spring_forward(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger=DAILY_0230_PARIS)
    state = drive(runtime, aid)
    assert state.waiting.until == "2026-03-27T01:30:00+00:00"  # 02:30 CET
    at(runtime, clock, aid, "2026-03-27T01:30:00+00:00")
    at(runtime, clock, aid, "2026-03-28T01:30:00+00:00")
    assert get_automation(runtime.run_store, aid)["next_fire_at"] == "2026-03-29T01:30:00+00:00"  # 03:30 CEST (02:30 does not exist)
    at(runtime, clock, aid, "2026-03-29T01:30:00+00:00")
    assert get_automation(runtime.run_store, aid)["next_fire_at"] == "2026-03-30T00:30:00+00:00"  # 02:30 CEST
    at(runtime, clock, aid, "2026-03-30T00:30:00+00:00")
    assert _fired(runtime, aid) == [
        "2026-03-27T01:30:00+00:00", "2026-03-28T01:30:00+00:00", "2026-03-29T01:30:00+00:00", "2026-03-30T00:30:00+00:00",
    ]
    occ = children(runtime, aid)[0].vars["_meta"]["occurrence"]
    assert occ["event_id"].startswith("schedule@2:")


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_downtime_coalesces_one_catch_up(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger=DAILY_0230_PARIS)
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-03-27T01:30:00+00:00")
    # Down from Mar 27 to Mar 31 12:00: Mar 28, 29, 30, 31 were due -> ONE occurrence.
    at(runtime, clock, aid, "2026-03-31T10:00:00+00:00")
    assert len(children(runtime, aid)) == 2
    coalesced = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.coalesced")]
    assert len(coalesced) == 1 and coalesced[0]["missed_count"] == 3
    assert _fired(runtime, aid)[-1] == "2026-03-31T00:30:00+00:00"
    assert get_automation(runtime.run_store, aid)["next_fire_at"] == "2026-04-01T00:30:00+00:00"


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_pause_resume_never_catches_up(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger=DAILY_0230_PARIS)
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-03-27T01:30:00+00:00")
    apply_automation_command(runtime, automation_id=aid, command_id="p", type="automation.pause", payload=None, now="2026-03-27T02:00:00+00:00")
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-03-30T12:00:00+00:00")
    apply_automation_command(runtime, automation_id=aid, command_id="r", type="automation.resume", payload=None, now="2026-03-30T12:00:00+00:00")
    clock.set("2026-03-30T12:00:00+00:00")
    drive(runtime, aid)
    assert len(children(runtime, aid)) == 1
    resumed = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.resumed")]
    assert resumed[0]["next_fire_at"] == "2026-03-31T00:30:00+00:00"
    assert automation_state(runtime, aid)["paused"] is False
