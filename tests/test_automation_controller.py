"""The automation controller end to end (contracts A-D), on both persistent stores.

Real runtime, real stores, the packaged controller bundle, deterministic
targets (see automation_harness). The controller's clock is moved explicitly.
"""

from __future__ import annotations

import pytest

from automation_harness import (
    Clock,
    at,
    automation_state,
    children,
    create,
    drive,
    make_runtime,
    make_stores,
)
from abstractruntime.automations import (
    CONTROLLER_WORKFLOW_ID,
    get_automation,
    is_interactive_wait,
    list_attention,
    list_occurrences,
    occurrence_run_id,
    pending_waits,
)
from abstractruntime.automations.ledger import automation_records
from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState

STORES = ["json", "sqlite"]


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    kind = getattr(request, "param", "json")
    run_store, ledger_store = make_stores(kind, tmp_path)
    runtime = make_runtime(run_store, ledger_store)
    return runtime, Clock(monkeypatch)


def _names(runtime, aid):
    return [r["name"] for r in automation_records(runtime.ledger_store, aid)]


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_scheduled_occurrences_are_deterministic_serial_children(env):
    runtime, clock = env
    aid = create(runtime, clock)
    controller = runtime.get_state(aid)
    assert controller.workflow_id == CONTROLLER_WORKFLOW_ID
    assert controller.session_id == f"automation:{aid}"

    state = drive(runtime, aid)
    assert state.status == RunStatus.WAITING
    assert state.waiting.reason == WaitReason.EVENT
    assert state.waiting.wait_key == f"automation:{aid}:wake"
    assert state.waiting.until == "2026-01-01T00:02:00+00:00"

    at(runtime, clock, aid, "2026-01-01T00:02:30+00:00")
    kids = children(runtime, aid)
    assert [k.run_id for k in kids] == [
        occurrence_run_id(aid, revision=1, index=1),
        occurrence_run_id(aid, revision=1, index=2),
    ]
    first = kids[0]
    assert first.status == RunStatus.COMPLETED
    assert first.session_id == first.run_id  # independent: a fresh session per occurrence
    occ = first.vars["_meta"]["occurrence"]
    assert occ["role"] == "occurrence" and occ["session_kind"] == "occurrence"
    assert occ["occurrence_index"] == 1 and occ["attempt"] == 1 and occ["revision"] == 1
    assert occ["event_id"].startswith("schedule@1:")
    assert first.vars["workspace_root"] == "/tmp/automation-ws"
    assert first.vars["prompt"] == "[Trigger schedule@1 · occurrence 1 · fired 2026-01-01T00:00:00+00:00]\ncheck memory"

    st = automation_state(runtime, aid)
    assert st["next_index"] == 3 and st["scheduled_count"] == 2 and st["pending_occurrence"] is None
    assert _names(runtime, aid) == [
        "automation.created",
        "automation.admitted", "automation.dispatched", "automation.completed",
        "automation.admitted", "automation.dispatched", "automation.completed",
    ]
    summary = get_automation(runtime.run_store, aid)
    assert summary["status"] == "active" and summary["next_fire_at"] == "2026-01-01T00:04:00+00:00"
    rows = list_occurrences(runtime, aid)["items"]
    assert [(r["index"], r["status"], r["attempts"]) for r in rows] == [(2, "completed", 1), (1, "completed", 1)]


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_downtime_coalesces_into_one_occurrence(env):
    runtime, clock = env
    aid = create(runtime, clock)
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:11:00+00:00")  # ticks 1..5 were due
    assert len(children(runtime, aid)) == 2
    coalesced = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.coalesced")]
    assert len(coalesced) == 1
    assert coalesced[0]["first_tick"] == 1 and coalesced[0]["last_tick"] == 5 and coalesced[0]["missed_count"] == 4
    assert automation_state(runtime, aid)["scheduled_count"] == 2
    envelope = children(runtime, aid)[1].vars["_meta"]["occurrence"]["trigger_envelope"]
    assert envelope["payload"] == {"tick": 5, "scheduled_at": "2026-01-01T00:10:00+00:00",
                                   "coalesced": {"first_tick": 1, "last_tick": 5, "missed_count": 4}}


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_one_shot_exhausts_and_the_controller_completes(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger={"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T01:00:00Z"}})
    state = drive(runtime, aid)
    assert state.waiting.until == "2026-01-01T01:00:00+00:00"
    state = at(runtime, clock, aid, "2026-01-01T01:00:05+00:00")
    assert state.status == RunStatus.COMPLETED
    assert len(children(runtime, aid)) == 1
    assert get_automation(runtime.run_store, aid)["status"] == "completed"


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_stale_wake_never_admits(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger={"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T01:00:00Z", "every": "1h"}})
    drive(runtime, aid)
    for now in ("2026-01-01T00:10:00+00:00", "2026-01-01T00:59:59+00:00"):
        state = at(runtime, clock, aid, now)
        assert state.status == RunStatus.WAITING and state.waiting.until == "2026-01-01T01:00:00+00:00"
    assert children(runtime, aid) == []


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_manual_trigger_parks_idle(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger={"source_id": "manual", "source_version": 1, "config": {}})
    state = drive(runtime, aid)
    assert state.status == RunStatus.WAITING and state.waiting.until is None
    assert at(runtime, clock, aid, "2026-02-01T00:00:00+00:00").waiting.until is None
    assert children(runtime, aid) == []


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_quiet_by_default_and_notify_allocates_attention(env):
    runtime, clock = env
    quiet = create(runtime, clock, request_id="quiet")
    drive(runtime, quiet)
    assert list_attention(runtime.ledger_store, quiet)["items"] == []

    loud = create(runtime, clock, request_id="loud", input_data={"prompt": "price?", "notify": True})
    drive(runtime, loud)
    custom = create(runtime, clock, request_id="custom", input_data={"prompt": "p", "notify": {"title": "Spike", "body": "CPU 97%"}})
    drive(runtime, custom)
    empty = create(runtime, clock, request_id="empty", input_data={"prompt": "p", "notify": {"title": "", "body": " "}})
    drive(runtime, empty)

    items = list_attention(runtime.ledger_store, loud)["items"]
    assert len(items) == 1 and items[0]["kind"] == "notify" and items[0]["title"] == "Memory watch"
    assert items[0]["body"].startswith("echo:[Trigger")
    assert items[0]["cursor"] == "att1:1"
    custom_items = list_attention(runtime.ledger_store, custom)["items"]
    assert [(i["title"], i["body"]) for i in custom_items] == [("Spike", "CPU 97%")]
    assert list_attention(runtime.ledger_store, empty)["items"] == []
    done = [r["payload"] for r in automation_records(runtime.ledger_store, loud, "automation.completed")]
    assert done[0]["notify"]["title"] == "Memory watch" and done[0]["attempts"] == 1


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_attention_pages_oldest_first_and_acknowledges_only_what_was_shown(env):
    runtime, clock = env
    aid = create(runtime, clock, input_data={"prompt": "p", "notify": True})
    drive(runtime, aid)
    for minute in (2, 4, 6):
        at(runtime, clock, aid, f"2026-01-01T00:0{minute}:00+00:00")
    page = list_attention(runtime.ledger_store, aid, limit=2)
    assert [i["seq"] for i in page["items"]] == [1, 2] and page["next_cursor"] == "att1:2"
    rest = list_attention(runtime.ledger_store, aid, cursor=page["next_cursor"], limit=2)
    assert [i["seq"] for i in rest["items"]] == [3, 4] and rest["next_cursor"] is None
    assert [i["seq"] for i in list_attention(runtime.ledger_store, aid, after_seq=3)["items"]] == [4]


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_human_waits_are_counted_but_pause_and_controller_waits_are_not(env):
    runtime, clock = env
    user_wait = create(runtime, clock, request_id="u", workflow_id="ask")
    event_wait = create(runtime, clock, request_id="e", workflow_id="ask_event")
    plain = create(runtime, clock, request_id="p")
    for aid in (user_wait, event_wait, plain):
        drive(runtime, aid)
    waits = pending_waits(runtime.run_store, user_wait)
    assert [(w["reason"], w["index"], w.get("prompt")) for w in waits] == [("user", 1, "Proceed?")]
    waits = pending_waits(runtime.run_store, event_wait)
    assert [(w["reason"], w.get("prompt")) for w in waits] == [("event", "Approve?")]
    assert pending_waits(runtime.run_store, plain) == []  # its controller parks on its wake wait
    # A paused occurrence is not waiting on a human (runtime pause flag) ...
    runtime.pause_run(children(runtime, event_wait)[0].run_id, reason="operator")
    assert pending_waits(runtime.run_store, event_wait) == []
    # ... nor is the synthetic USER wait a pause of a RUNNING run creates.
    paused = RunState(run_id="r-paused", workflow_id="echo", status=RunStatus.WAITING, current_node="answer",
                      waiting=WaitState(reason=WaitReason.USER, wait_key="pause:r-paused", prompt="Paused",
                                        details={"kind": "pause"}))
    assert is_interactive_wait(paused) is False
    paused.waiting = WaitState(reason=WaitReason.USER, wait_key="ask:1", prompt="Proceed?")
    assert is_interactive_wait(paused) is True
