"""`next_fire_at` / `current_occurrence` in automation summaries (no schedule arithmetic in clients).

While an occurrence runs, `next_fire_at` is the tick the controller will admit
next, computed by the trigger adapter on the persisted cursor; the tests drive
the controller and compare with the occurrence it actually admits.
"""

from __future__ import annotations

import pytest

from automation_harness import TARGETS, Clock, children, create, drive, make_runtime, make_stores
from abstractruntime.automation_queries import list_automations
from abstractruntime.automations import apply_automation_command, get_automation

HOURLY = {"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z", "every": "1h"}}
MANUAL = {"source_id": "manual", "source_version": 1, "config": {}}


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    return make_runtime(*make_stores(request.param, tmp_path)), Clock(monkeypatch)


def _both(runtime, aid):
    """(get_automation, list summary) — they must agree."""
    got = get_automation(runtime.run_store, aid)
    listed = next(i for i in list_automations(runtime.run_store, limit=50).items if i["automation_id"] == aid)
    assert (got["next_fire_at"], got["current_occurrence"]) == (listed["next_fire_at"], listed["current_occurrence"])
    return got


def _answer(runtime, aid):
    child = children(runtime, aid)[-1]
    runtime.resume(workflow=TARGETS["ask"], run_id=child.run_id, wait_key=child.waiting.wait_key,
                   payload={"response": "yes"})


def _fired_at(runtime, aid, index):
    return children(runtime, aid)[index - 1].vars["_meta"]["occurrence"]["fired_at"]


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_running_occurrence_reports_the_next_tick_the_controller_admits(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    drive(runtime, aid)  # occurrence 1 admitted at 00:00, waiting on a person
    clock.set("2026-01-01T00:10:00+00:00")
    summary = _both(runtime, aid)
    assert summary["current_occurrence"] == {"index": 1, "run_id": children(runtime, aid)[0].run_id,
                                             "attempt": 1, "status": "running"}
    projected = summary["next_fire_at"]
    assert projected == "2026-01-01T01:00:00+00:00"

    _answer(runtime, aid)
    drive(runtime, aid)  # occurrence 1 ends; the controller parks until its next tick
    assert _both(runtime, aid)["current_occurrence"] is None
    assert _both(runtime, aid)["next_fire_at"] == projected
    clock.set(projected)
    from automation_harness import wake

    wake(runtime, aid)
    drive(runtime, aid)
    assert _fired_at(runtime, aid, 2) == projected


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_a_long_occurrence_reports_the_coalesced_tick(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    drive(runtime, aid)
    clock.set("2026-01-01T03:30:00+00:00")  # ticks 1, 2 and 3 passed while occurrence 1 ran
    projected = _both(runtime, aid)["next_fire_at"]
    assert projected == "2026-01-01T03:00:00+00:00"  # due: fires as soon as occurrence 1 ends
    _answer(runtime, aid)
    drive(runtime, aid)
    assert _fired_at(runtime, aid, 2) == projected


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_paused_and_manual_have_no_next_fire(env):
    runtime, clock = env
    paused = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    drive(runtime, paused)
    apply_automation_command(runtime, automation_id=paused, command_id="p", type="automation.pause", now=clock.now)
    summary = _both(runtime, paused)
    assert summary["next_fire_at"] is None and summary["current_occurrence"]["status"] == "running"

    manual = create(runtime, clock, workflow_id="ask", trigger=MANUAL)
    drive(runtime, manual)
    assert _both(runtime, manual) == {**_both(runtime, manual), "next_fire_at": None, "current_occurrence": None}
    apply_automation_command(runtime, automation_id=manual, command_id="go", type="automation.run_now", now=clock.now)
    drive(runtime, manual)
    summary = _both(runtime, manual)
    assert summary["next_fire_at"] is None and summary["current_occurrence"]["status"] == "running"

    archived = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    drive(runtime, archived)
    apply_automation_command(runtime, automation_id=archived, command_id="a", type="automation.archive", now=clock.now)
    assert _both(runtime, archived)["next_fire_at"] is None
