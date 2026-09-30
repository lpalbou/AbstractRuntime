"""Automation commands (contracts A/D, C8, C16): semantics, idempotent replay, races, crash matrix."""

from __future__ import annotations

import threading
import time

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
    apply_automation_command,
    get_automation,
    occurrence_run_id,
    record_automation_command_result,
)
from abstractruntime.automations import controller as controller_mod
from abstractruntime.automations.ledger import automation_records
from abstractruntime.core.models import RunStatus
from abstractruntime.core.runtime import run_mutation_lock

STORES = ["json", "sqlite"]


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    kind = getattr(request, "param", "json")
    run_store, ledger_store = make_stores(kind, tmp_path)
    return make_runtime(run_store, ledger_store), Clock(monkeypatch)


def cmd(runtime, aid, command_id, type_, now, payload=None, **kw):
    return apply_automation_command(
        runtime, automation_id=aid, command_id=command_id, type=f"automation.{type_}", payload=payload, now=now, **kw
    )


def names(runtime, aid, *only):
    return [r["name"] for r in automation_records(runtime.ledger_store, aid, *only)]


HOURLY = {"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z", "every": "1h"}}


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_pause_blocks_scheduled_admission_and_resume_does_not_fire(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger=HOURLY)
    drive(runtime, aid)  # occurrence 1 at 00:00
    assert cmd(runtime, aid, "c1", "pause", "2026-01-01T00:10:00+00:00") == {"status": "applied", "duplicate": False}
    state = drive(runtime, aid)
    assert state.status == RunStatus.WAITING and state.waiting.until is None  # parked: no deadline while paused
    assert get_automation(runtime.run_store, aid)["status"] == "paused"
    at(runtime, clock, aid, "2026-01-01T03:30:00+00:00")  # ticks 1..3 passed while paused
    assert len(children(runtime, aid)) == 1

    clock.set("2026-01-01T03:30:00+00:00")
    assert cmd(runtime, aid, "c2", "resume", "2026-01-01T03:30:00+00:00")["status"] == "applied"
    state = drive(runtime, aid)
    assert len(children(runtime, aid)) == 1  # resume never fires, never catches up
    assert state.waiting.until == "2026-01-01T04:00:00+00:00"
    resumed = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.resumed")]
    assert resumed[0]["next_fire_at"] == "2026-01-01T04:00:00+00:00"
    at(runtime, clock, aid, "2026-01-01T04:00:00+00:00")
    assert len(children(runtime, aid)) == 2


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_run_now_while_paused_runs_once_and_stays_paused(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger=HOURLY)
    drive(runtime, aid)
    cmd(runtime, aid, "p", "pause", "2026-01-01T00:05:00+00:00")
    drive(runtime, aid)
    clock.set("2026-01-01T00:06:00+00:00")
    assert cmd(runtime, aid, "run-1", "run_now", "2026-01-01T00:06:00+00:00")["status"] == "applied"
    drive(runtime, aid)
    kids = children(runtime, aid)
    assert len(kids) == 2
    assert kids[1].run_id == occurrence_run_id(aid, revision=1, index=2, command_id="run-1")
    assert kids[1].vars["_meta"]["occurrence"]["trigger_envelope"]["source_id"] == "manual"
    assert kids[1].vars["_meta"]["occurrence"]["event_id"] == "manual:run-1"
    st = automation_state(runtime, aid)
    assert st["paused"] is True and st["manual_pending"] is None and st["scheduled_count"] == 1  # manual never counts


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_run_now_is_rejected_while_busy_and_on_a_finished_automation(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    drive(runtime, aid)  # occurrence 1 waits on a human
    r = cmd(runtime, aid, "rn", "run_now", "2026-01-01T00:01:00+00:00")
    assert r["status"] == "rejected" and r["error"]["reason_code"] == "automation_busy"
    one_shot = create(runtime, clock, request_id="once",
                      trigger={"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z"}})
    assert drive(runtime, one_shot).status == RunStatus.COMPLETED
    r = cmd(runtime, one_shot, "rn2", "run_now", "2026-01-01T00:01:00+00:00")
    assert r["error"]["reason_code"] == "invalid_state"
    # A second run_now before the first was admitted: busy, no queue.
    manual = create(runtime, clock, request_id="m", trigger={"source_id": "manual", "source_version": 1, "config": {}})
    drive(runtime, manual)
    with run_mutation_lock(manual):  # keep the controller from admitting in between
        assert cmd(runtime, manual, "a", "run_now", "2026-01-01T00:01:00+00:00")["status"] == "applied"
        assert cmd(runtime, manual, "b", "run_now", "2026-01-01T00:01:00+00:00")["error"]["reason_code"] == "automation_busy"


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_replayed_command_ids_are_idempotent(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger=HOURLY)
    drive(runtime, aid)
    first = cmd(runtime, aid, "same", "pause", "2026-01-01T00:05:00+00:00")
    again = cmd(runtime, aid, "same", "pause", "2026-01-01T00:06:00+00:00")
    assert first == {"status": "applied", "duplicate": False}
    assert again == {"status": "applied", "duplicate": True}
    rejected = cmd(runtime, aid, "bad", "revise", "2026-01-01T00:06:00+00:00", payload={"changes": {"title": ""}})
    replay = cmd(runtime, aid, "bad", "revise", "2026-01-01T00:07:00+00:00", payload={"changes": {"title": ""}})
    assert rejected["status"] == "rejected" and replay["status"] == "rejected" and replay["duplicate"] is True
    assert replay["error"] == rejected["error"]
    assert names(runtime, aid, "automation.command_result", "automation.paused") == [
        "automation.command_result", "automation.paused", "automation.command_result",
    ]
    assert automation_state(runtime, aid)["state_version"] == 5  # 3 controller decisions + 2 command results


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_a_reused_command_id_for_a_different_command_is_refused_and_not_recorded(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger=HOURLY)
    drive(runtime, aid)
    assert cmd(runtime, aid, "p2", "pause", "2026-01-01T00:05:00+00:00")["status"] == "applied"
    before = names(runtime, aid)
    version = automation_state(runtime, aid)["state_version"]
    for type_, payload in (("archive", None), ("pause", {"reason": "other"}), ("revise", {"changes": {"title": "x"}})):
        r = cmd(runtime, aid, "p2", type_, "2026-01-01T00:06:00+00:00", payload=payload)
        assert r["status"] == "rejected" and r["duplicate"] is False, type_
        assert (r["error"]["reason_code"], r["error"]["field"]) == ("identity_conflict", "command_id")
    r = cmd(runtime, aid, "p2", "pause", "2026-01-01T00:06:00+00:00", expected_revision=1)
    assert r["error"]["reason_code"] == "identity_conflict"
    assert names(runtime, aid) == before  # no false automation.archived (or any) record
    assert automation_state(runtime, aid)["state_version"] == version
    assert get_automation(runtime.run_store, aid)["definition"]["archived_at"] is None
    assert cmd(runtime, aid, "p2", "pause", "2026-01-01T00:07:00+00:00") == {"status": "applied", "duplicate": True}


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_revising_policy_keeps_the_fields_not_sent(env):
    runtime, clock = env
    req_policy = {"tool_approval": "ask", "retry": {"max_attempts": 5}}
    from automation_harness import request
    from abstractruntime.automations import create_automation

    req = request(trigger=HOURLY)
    req["policy"] = req_policy
    aid = create_automation(runtime, req, now=clock.now)[0]
    r = cmd(runtime, aid, "r", "revise", "2026-01-01T00:01:00+00:00",
            payload={"changes": {"policy": {"retry": {"max_attempts": 2}}}})
    assert r["status"] == "applied"
    policy = get_automation(runtime.run_store, aid)["definition"]["policy"]
    assert policy["tool_approval"] == "ask" and policy["retry"]["max_attempts"] == 2
    r = cmd(runtime, aid, "r2", "revise", "2026-01-01T00:02:00+00:00",
            payload={"changes": {"policy": {"tool_approval": "auto"}}})
    policy = get_automation(runtime.run_store, aid)["definition"]["policy"]
    assert policy["tool_approval"] == "auto" and policy["retry"]["max_attempts"] == 2


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_revise_checks_expected_revision_at_application_and_rebinds_the_trigger(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger=HOURLY)
    drive(runtime, aid)
    old_binding = get_automation(runtime.run_store, aid)["definition"]["trigger"]["binding_id"]
    ok = cmd(runtime, aid, "r1", "revise", "2026-01-01T00:20:00+00:00",
             payload={"changes": {"title": "Renamed"}}, expected_revision=1)
    assert ok["status"] == "applied"
    stale = cmd(runtime, aid, "r2", "revise", "2026-01-01T00:21:00+00:00",
                payload={"changes": {"title": "Lost update"}}, expected_revision=1)
    assert stale["status"] == "rejected" and stale["error"]["reason_code"] == "revision_conflict"
    definition = get_automation(runtime.run_store, aid)["definition"]
    assert definition["revision"] == 2 and definition["title"] == "Renamed"
    assert definition["trigger"]["binding_id"] == old_binding  # trigger unchanged: same binding

    clock.set("2026-01-01T00:30:00+00:00")
    changed = cmd(runtime, aid, "r3", "revise", "2026-01-01T00:30:00+00:00", payload={"changes": {"trigger": {
        "source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z", "every": "15m"}}}})
    assert changed["status"] == "applied"
    definition = get_automation(runtime.run_store, aid)["definition"]
    assert definition["revision"] == 3 and definition["trigger"]["binding_id"] != old_binding
    state = drive(runtime, aid)
    assert state.waiting.until == "2026-01-01T00:45:00+00:00"  # re-armed after now: 00:15 and 00:30 do not fire
    assert len(children(runtime, aid)) == 1
    revised = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.revised")]
    assert [(p["previous_revision"], p["definition"]["revision"]) for p in revised] == [(1, 2), (2, 3)]
    bad = cmd(runtime, aid, "r4", "revise", "2026-01-01T00:31:00+00:00",
              payload={"changes": {"context": {"mode": "growing", "growing": {"summary": {"enabled": True, "every_n": 3, "max_tokens": 100}}}}})
    assert bad["error"]["reason_code"] == "unsupported_feature"


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_stop_current_cancels_the_occurrence_tree_quietly(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    drive(runtime, aid)
    child = children(runtime, aid)[0]
    assert child.status == RunStatus.WAITING
    assert cmd(runtime, aid, "stop", "stop_current", "2026-01-01T00:01:00+00:00")["status"] == "applied"
    assert runtime.get_state(child.run_id).status == RunStatus.CANCELLED
    drive(runtime, aid)
    done = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")]
    assert [(d["status"], d["attention"]) for d in done] == [("cancelled", None)]
    assert automation_state(runtime, aid)["pending_occurrence"] is None
    assert cmd(runtime, aid, "stop2", "stop_current", "2026-01-01T00:02:00+00:00")["error"]["reason_code"] == "invalid_state"


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_stop_current_during_backoff_cancels_the_retry(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="flaky", input_data={"prompt": "p", "fail_until": 99}, trigger=HOURLY)
    drive(runtime, aid)
    assert automation_state(runtime, aid)["pending_occurrence"]["phase"] == "backoff"
    cmd(runtime, aid, "stop", "stop_current", "2026-01-01T00:00:05+00:00")
    drive(runtime, aid)
    assert len(children(runtime, aid)) == 1  # attempt 2 never started
    done = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")]
    assert [d["status"] for d in done] == ["cancelled"]


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_pause_during_backoff_cancels_the_retry(env):
    """N3: a pause lands while the occurrence only waits for its retry: no attempt 2."""
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="flaky", input_data={"prompt": "p", "fail_until": 99}, trigger=HOURLY)
    drive(runtime, aid)
    assert automation_state(runtime, aid)["pending_occurrence"]["phase"] == "backoff"
    clock.set("2026-01-01T00:00:05+00:00")
    assert cmd(runtime, aid, "p", "pause", "2026-01-01T00:00:05+00:00")["status"] == "applied"
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:00:31+00:00")  # past the retry time
    assert len(children(runtime, aid)) == 1  # attempt 2 never started
    done = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")]
    assert [(d["status"], d["attention"]) for d in done] == [("cancelled", None)]
    st = automation_state(runtime, aid)
    assert st["paused"] is True and st["pending_occurrence"] is None
    assert get_automation(runtime.run_store, aid)["status"] == "paused"


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_an_attempt_failing_while_paused_is_not_retried(env):
    """N3: the attempt running when the pause lands finishes; its failure schedules no retry."""
    from abstractruntime.automations.bundle import controller_workflow_spec

    runtime, clock = env
    aid = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    drive(runtime, aid)
    child = children(runtime, aid)[0]
    assert child.status == RunStatus.WAITING
    assert cmd(runtime, aid, "p", "pause", "2026-01-01T00:01:00+00:00")["status"] == "applied"
    assert runtime.get_state(child.run_id).status == RunStatus.WAITING  # a running attempt is not stopped
    failed = runtime.run_store.load(child.run_id)
    failed.status = RunStatus.FAILED
    failed.error = "boom while paused"
    failed.waiting = None
    runtime.run_store.save(failed)
    ctl = runtime.get_state(aid)
    runtime.resume(workflow=controller_workflow_spec(), run_id=aid, wait_key=ctl.waiting.wait_key,
                   payload={"sub_run_id": child.run_id, "output": {"success": False, "error": "boom while paused"}}, max_steps=0)
    drive(runtime, aid)
    at(runtime, clock, aid, "2026-01-01T00:10:00+00:00")  # well past a 30 s backoff
    assert len(children(runtime, aid)) == 1
    assert list(automation_records(runtime.ledger_store, aid, "automation.retry_scheduled")) == []
    done = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")]
    assert [(d["status"], d["attempts"]) for d in done] == [("failed", 1)]
    assert automation_state(runtime, aid)["paused"] is True


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_a_manual_run_while_paused_keeps_its_retries(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="flaky", input_data={"prompt": "p", "fail_until": 2}, trigger=HOURLY)
    drive(runtime, aid)  # the 00:00 occurrence: attempt 1 fails, backoff
    at(runtime, clock, aid, "2026-01-01T00:00:30+00:00")  # attempt 2 succeeds
    cmd(runtime, aid, "p", "pause", "2026-01-01T00:05:00+00:00")
    drive(runtime, aid)
    clock.set("2026-01-01T00:06:00+00:00")
    assert cmd(runtime, aid, "run-1", "run_now", "2026-01-01T00:06:00+00:00")["status"] == "applied"
    drive(runtime, aid)
    pending = automation_state(runtime, aid)["pending_occurrence"]
    assert pending["phase"] == "backoff" and pending["attempt"] == 2 and pending["command_id"] == "run-1"
    at(runtime, clock, aid, "2026-01-01T00:06:30+00:00")
    done = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")]
    assert [(d["status"], d["attempts"]) for d in done] == [("completed", 2), ("completed", 2)]


@pytest.mark.parametrize("env", STORES, indirect=True)
def test_archive_lets_the_current_occurrence_finish_then_ends(env):
    runtime, clock = env
    aid = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    drive(runtime, aid)
    assert cmd(runtime, aid, "arch", "archive", "2026-01-01T00:01:00+00:00")["status"] == "applied"
    assert get_automation(runtime.run_store, aid)["status"] == "archived"
    child = children(runtime, aid)[0]
    runtime.resume(workflow=runtime.workflow_registry.get("ask"), run_id=child.run_id,
                   wait_key=child.waiting.wait_key, payload={"text": "yes"})
    state = drive(runtime, aid)
    assert state.status == RunStatus.COMPLETED
    assert [r["payload"]["status"] for r in automation_records(runtime.ledger_store, aid, "automation.completed")] == ["completed"]
    archived = [r["payload"] for r in automation_records(runtime.ledger_store, aid, "automation.archived")]
    assert archived[0]["active_occurrence_run_id"] == child.run_id
    for type_ in ("pause", "resume", "run_now", "revise"):
        r = cmd(runtime, aid, f"after-{type_}", type_, "2026-01-01T00:02:00+00:00", payload={"changes": {"title": "x"}})
        assert r["error"]["reason_code"] == "invalid_state", type_
    assert cmd(runtime, aid, "arch-again", "archive", "2026-01-01T00:02:00+00:00")["status"] == "applied"


def test_unknown_automation_and_bad_requests_are_rejected_not_raised(env):
    runtime, clock = env
    assert cmd(runtime, "no-such", "c", "pause", "2026-01-01T00:00:00+00:00")["error"]["reason_code"] == "automation_not_found"
    aid = create(runtime, clock)
    r = apply_automation_command(runtime, automation_id=aid, command_id="c", type="automation.explode", now="2026-01-01T00:00:00+00:00")
    assert r["error"]["reason_code"] == "invalid_request"
    host = record_automation_command_result(runtime, automation_id=aid, command_id="h1", type="automation.pause",
                                            error={"reason_code": "forbidden", "message": "no"}, now="2026-01-01T00:00:00+00:00")
    assert host == {"status": "rejected", "error": {"reason_code": "forbidden", "message": "no"}}
    assert cmd(runtime, aid, "h1", "pause", "2026-01-01T00:00:00+00:00")["duplicate"] is True


# --- races ----------------------------------------------------------------------


def test_a_command_never_interleaves_with_a_controller_tick(env, monkeypatch):
    """The tick holds the run's mutation lock end to end: a pause applied while a
    tick is admitting is applied after the tick's saves, never overwritten."""
    runtime, clock = env
    aid = create(runtime, clock, trigger=HOURLY)
    entered, release = threading.Event(), threading.Event()
    real = controller_mod.build_prepared

    def slow_prepare(*args, **kwargs):
        entered.set()
        release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(controller_mod, "build_prepared", slow_prepare)
    ticker = threading.Thread(target=lambda: drive(runtime, aid))
    ticker.start()
    assert entered.wait(5)
    results = {}
    pauser = threading.Thread(target=lambda: results.update(r=cmd(runtime, aid, "p", "pause", "2026-01-01T00:00:01+00:00")))
    pauser.start()
    time.sleep(0.2)
    assert pauser.is_alive()  # blocked on the lock the tick holds
    release.set()
    ticker.join(5)
    pauser.join(5)
    assert results["r"]["status"] == "applied"
    st = automation_state(runtime, aid)
    assert st["paused"] is True  # not clobbered by the tick's in-memory state
    assert st["next_index"] == 2  # and the admission the tick decided stands


def test_concurrent_commands_serialize(env):
    runtime, clock = env
    aid = create(runtime, clock, trigger={"source_id": "manual", "source_version": 1, "config": {}})
    drive(runtime, aid)
    out = []

    def go(cid, type_):
        out.append((type_, cmd(runtime, aid, cid, type_, "2026-01-01T00:00:01+00:00", payload={"changes": {"title": cid}})))

    threads = [threading.Thread(target=go, args=(f"c{i}", t)) for i, t in enumerate(["pause", "run_now", "revise", "revise", "run_now"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    by_type = {}
    for type_, r in out:
        by_type.setdefault(type_, []).append(r["status"])
    assert sorted(by_type["run_now"]) == ["applied", "rejected"]  # exactly one manual run pending
    assert by_type["revise"] == ["applied", "applied"]
    st = automation_state(runtime, aid)
    assert st["state_version"] == 5 and st["paused"] is True
    assert get_automation(runtime.run_store, aid)["definition"]["revision"] == 3


# --- crash matrix -----------------------------------------------------------------


class Crash(Exception):
    pass


class CrashingStores:
    """Delegating run/ledger stores that raise once at a chosen point of a decision commit."""

    def __init__(self, run_store, ledger_store, point):
        self.point = point
        self.fired = False
        self.saves = 0
        outer = self

        class Runs:
            def __getattr__(self, name):
                return getattr(run_store, name)

            def save(self, run):
                intent = ((run.vars.get("_runtime") or {}).get("automation") or {}).get("intent")
                if not outer.fired:
                    if outer.point == "intent_save" and intent:
                        outer.fired = True
                        raise Crash("before the intent is saved")
                    if outer.point == "state_save" and not intent and outer.appended:
                        outer.fired = True
                        raise Crash("after the decision append, before the state save")
                return run_store.save(run)

        class Ledger:
            def __getattr__(self, name):
                return getattr(ledger_store, name)

            def append(self, record):
                if not outer.fired and outer.point == "append" and record.idempotency_key.startswith("automation:command_result:"):
                    outer.fired = True
                    raise Crash("before the decision append")
                ledger_store.append(record)
                if record.idempotency_key.startswith("automation:command_result:"):
                    outer.appended = True

        self.appended = False
        self.run_store = Runs()
        self.ledger_store = Ledger()


@pytest.mark.parametrize("kind", STORES)
@pytest.mark.parametrize("point", ["intent_save", "append", "state_save", "after_commit"])
@pytest.mark.parametrize("type_", ["revise", "pause", "run_now"])
def test_crash_around_a_command_commit_applies_it_exactly_once(tmp_path, monkeypatch, kind, point, type_):
    clock = Clock(monkeypatch)
    run_store, ledger_store = make_stores(kind, tmp_path)
    runtime = make_runtime(run_store, ledger_store)
    aid = create(runtime, clock, trigger={"source_id": "manual", "source_version": 1, "config": {}})
    drive(runtime, aid)
    before = automation_state(runtime, aid)["state_version"]
    payload = {"changes": {"title": "After crash"}}

    crashing = CrashingStores(*make_stores(kind, tmp_path), point)
    crashed_runtime = make_runtime(crashing.run_store, crashing.ledger_store)
    if point == "after_commit":
        assert cmd(crashed_runtime, aid, "c1", type_, "2026-01-01T00:00:01+00:00", payload=payload)["status"] == "applied"
    else:
        with pytest.raises(Crash):
            cmd(crashed_runtime, aid, "c1", type_, "2026-01-01T00:00:01+00:00", payload=payload)

    # Restart: fresh store objects over the same files; the host replays the command.
    runtime = make_runtime(*make_stores(kind, tmp_path))
    replay = cmd(runtime, aid, "c1", type_, "2026-01-01T00:00:02+00:00", payload=payload)
    assert replay["status"] == "applied"
    assert replay["duplicate"] is (point in ("state_save", "after_commit"))
    results = [r for r in automation_records(runtime.ledger_store, aid, "automation.command_result")]
    assert len(results) == 1
    st = automation_state(runtime, aid)
    assert st["state_version"] == before + 1 and st["intent"] is None
    definition = get_automation(runtime.run_store, aid)["definition"]
    if type_ == "revise":
        assert definition["revision"] == 2 and definition["title"] == "After crash"
        assert len(names(runtime, aid, "automation.revised")) == 1
    elif type_ == "pause":
        assert st["paused"] is True and len(names(runtime, aid, "automation.paused")) == 1
    else:
        assert st["manual_pending"] == {"command_id": "c1"}
        drive(runtime, aid)
        assert len(children(runtime, aid)) == 1
