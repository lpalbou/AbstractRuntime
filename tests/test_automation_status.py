"""One status rule for an automation: `get_automation` and `list_automations` agree (contract A)."""

from __future__ import annotations

import pytest

from automation_harness import Clock, create, drive, make_runtime, make_stores
from abstractruntime import RunStatus
from abstractruntime.automation_queries import automation_status as listed_status
from abstractruntime.automation_queries import list_automations
from abstractruntime.automations import apply_automation_command, get_automation
from abstractruntime.automations.models import automation_status
from abstractruntime.session_history import SessionHistoryError
from abstractruntime.session_turns import select_session_turns
from abstractruntime.storage.base import RunStore

ONE_SHOT = {"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z"}}
HOURLY = {"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z", "every": "1h"}}


def test_there_is_one_implementation():
    assert listed_status is automation_status


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_list_and_get_agree_for_every_terminal_shape(tmp_path, monkeypatch, kind):
    clock = Clock(monkeypatch)
    runtime = make_runtime(*make_stores(kind, tmp_path))
    expected = {}

    active = create(runtime, clock, trigger=HOURLY)
    drive(runtime, active)
    expected[active] = "active"

    paused = create(runtime, clock, trigger=HOURLY)
    drive(runtime, paused)
    apply_automation_command(runtime, automation_id=paused, command_id="p", type="automation.pause", now=clock.now)
    expected[paused] = "paused"

    completed = create(runtime, clock, trigger=ONE_SHOT)
    assert drive(runtime, completed).status == RunStatus.COMPLETED
    expected[completed] = "completed"

    archived = create(runtime, clock, trigger=HOURLY)
    drive(runtime, archived)
    apply_automation_command(runtime, automation_id=archived, command_id="a", type="automation.archive", now=clock.now)
    drive(runtime, archived)
    assert runtime.get_state(archived).status == RunStatus.COMPLETED
    expected[archived] = "archived"

    cancelled_archived = create(runtime, clock, trigger=HOURLY)
    drive(runtime, cancelled_archived)
    apply_automation_command(runtime, automation_id=cancelled_archived, command_id="a", type="automation.archive", now=clock.now)
    runtime.cancel_run(cancelled_archived, reason="host stop")
    expected[cancelled_archived] = "archived"

    cancelled = create(runtime, clock, trigger=HOURLY)
    drive(runtime, cancelled)
    runtime.cancel_run(cancelled, reason="operator")
    expected[cancelled] = "failed"

    failed = create(runtime, clock, trigger=HOURLY)
    drive(runtime, failed)
    run = runtime.get_state(failed)
    run.status, run.waiting, run.error = RunStatus.FAILED, None, "controller crashed"
    runtime.run_store.save(run)
    expected[failed] = "failed"

    listed = {item["automation_id"]: item["status"] for item in list_automations(runtime.run_store, limit=50).items}
    for aid, status in expected.items():
        assert get_automation(runtime.run_store, aid)["status"] == status, aid
        assert listed[aid] == status, aid


class _NoIndexStore(RunStore):
    def __init__(self):
        self._runs = {}

    def save(self, run):
        self._runs[run.run_id] = run

    def load(self, run_id):
        return self._runs.get(run_id)


def test_through_occurrence_without_a_run_index_is_a_typed_error():
    with pytest.raises(SessionHistoryError, match="needs a run index"):
        select_session_turns(_NoIndexStore(), "s-1", through_occurrence=1)
