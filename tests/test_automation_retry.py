"""Retry policy (contract C15): per-attempt deterministic children, frozen inputs, attention only on final failure."""

from __future__ import annotations

import pytest

from automation_harness import Clock, at, automation_state, children, create, drive, make_runtime, make_stores
from abstractruntime.automations import apply_automation_command, list_attention, list_occurrences, occurrence_run_id
from abstractruntime.automations.ledger import automation_records
from abstractruntime.core.models import RunStatus

STORES = ["json", "sqlite"]


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    run_store, ledger_store = make_stores(request.param, tmp_path)
    return make_runtime(run_store, ledger_store), Clock(monkeypatch)


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_failure_then_success_is_quiet_with_two_attempts(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="flaky", input_data={"prompt": "p", "fail_until": 2})
    state = drive(runtime, aid)
    pending = automation_state(runtime, aid)["pending_occurrence"]
    assert pending["phase"] == "backoff" and pending["attempt"] == 2
    assert pending["retry_at"] == "2026-01-01T00:00:30+00:00"  # 30s after the failure
    assert state.waiting.until == "2026-01-01T00:00:30+00:00"

    at(runtime, clock, aid, "2026-01-01T00:00:31+00:00")
    kids = children(runtime, aid)
    assert [k.run_id for k in kids] == [
        occurrence_run_id(aid, revision=1, index=1),
        occurrence_run_id(aid, revision=1, index=1, attempt=2),
    ]
    assert [k.vars["_meta"]["occurrence"]["attempt"] for k in kids] == [1, 2]
    done = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")]
    assert [(d["status"], d["attempts"], d["attention"]) for d in done] == [("completed", 2, None)]
    assert list_attention(runtime.ledger_store, aid)["items"] == []
    row = list_occurrences(runtime, aid)["items"][0]
    assert row["attempts"] == 2 and row["run_ids"] == [k.run_id for k in kids]


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_always_failing_allocates_one_failure_after_three_attempts(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="flaky", input_data={"prompt": "p", "fail_until": 99})
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:00:30+00:00")
    pending = automation_state(runtime, aid)["pending_occurrence"]
    assert pending["attempt"] == 3 and pending["retry_at"] == "2026-01-01T00:01:30+00:00"  # 30s x 2
    at(runtime, clock, aid, "2026-01-01T00:01:30+00:00")
    assert len(children(runtime, aid)) == 3
    items = list_attention(runtime.ledger_store, aid)["items"]
    assert len(items) == 1 and items[0]["kind"] == "failure" and "boom on attempt 3" in items[0]["body"]
    done = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")]
    assert [(d["status"], d["attempts"]) for d in done] == [("failed", 3)]
    assert len(list(automation_records(runtime.ledger_store, aid, "automation.retry_scheduled"))) == 2
    # failure: continue — the next scheduled tick still runs.
    st = automation_state(runtime, aid)
    assert st["pending_occurrence"] is None and st["last_outcome"]["status"] == "failed"


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_backoff_is_capped(env):
    runtime, clock = env
    aid = create(
        runtime, clock, workflow_id="flaky", input_data={"prompt": "p", "fail_until": 99},
        retry={"max_attempts": 4, "backoff": {"initial": "4m", "factor": 3, "max": "10m"}},
    )
    drive(runtime, aid)
    assert automation_state(runtime, aid)["pending_occurrence"]["retry_at"] == "2026-01-01T00:04:00+00:00"
    at(runtime, clock, aid, "2026-01-01T00:04:00+00:00")
    assert automation_state(runtime, aid)["pending_occurrence"]["retry_at"] == "2026-01-01T00:14:00+00:00"  # 12m -> 10m


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_revise_during_backoff_does_not_change_the_next_attempt(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="flaky", input_data={"prompt": "original", "fail_until": 2})
    drive(runtime, aid)
    receipt = apply_automation_command(
        runtime, automation_id=aid, command_id="c-rev", type="automation.revise",
        payload={"changes": {"target": {"workflow_id": "echo", "bundle_ref": "fixtures@1.0.0", "flow_id": "echo",
                                        "input_data": {"prompt": "REVISED"}}}},
        now="2026-01-01T00:00:10+00:00",
    )
    assert receipt["status"] == "applied"
    at(runtime, clock, aid, "2026-01-01T00:00:30+00:00")
    attempt2 = children(runtime, aid)[1]
    assert attempt2.workflow_id == "flaky"
    assert attempt2.vars["prompt"].endswith("\noriginal")
    assert attempt2.vars["_meta"]["occurrence"]["revision"] == 1
    assert attempt2.status == RunStatus.COMPLETED
    # The next occurrence uses revision 2.
    at(runtime, clock, aid, "2026-01-01T00:02:00+00:00")
    third = children(runtime, aid)[2]
    assert third.workflow_id == "echo" and third.vars["prompt"].endswith("\nREVISED")
    assert third.run_id == occurrence_run_id(aid, revision=2, index=2)
