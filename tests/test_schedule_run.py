"""SCHEDULE_RUN (framework backlog 1045): a run schedules work for later, at most with its own tools.

The host side is a fake here (`FakeHost`): it implements the documented `schedule_run_handler` seam with
`create_scheduled_run`, and keeps a list standing in for the user's event stream."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from automation_harness import TARGETS, Clock, children, drive, wake
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.automations import apply_automation_command, register_controller_bundle
from abstractruntime.automations.ledger import definition_of
from abstractruntime.automations.wakes import (
    ScheduleRunError,
    create_scheduled_run,
    plan_schedule_run,
    wake_cancelled_event,
    wake_scheduled_event,
)
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.core.run_attribution import session_attribution
from abstractruntime.core.run_origin import origin_of
from abstractruntime.email.inbox import InMemoryEventInbox
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

CEILING = ["read_file", "schedule_run", "web_search"]


def _caller_node(run, ctx):
    return StepPlan(node_id="go", effect=Effect(type=EffectType.SCHEDULE_RUN, payload=dict(run.vars["schedule"]), result_key="wake"),
                    next_node="done")


def _done_node(run, ctx):
    return StepPlan(node_id="done", complete_output={"response": "scheduled", "wake": run.vars.get("wake")})


CALLER = WorkflowSpec(workflow_id="caller", entry_node="go", nodes={"go": _caller_node, "done": _done_node})


class FakeHost:
    """The gateway's role: create the planned automation, append the user-stream events."""

    def __init__(self, runtime: Runtime, workspace: str) -> None:
        self.runtime, self.workspace, self.stream, self.plans = runtime, workspace, [], []

    def __call__(self, run, plan):
        self.plans.append(plan)
        created = create_scheduled_run(
            self.runtime, plan,
            target={"workflow_id": "echo", "bundle_ref": "fixtures@1.0.0", "flow_id": "echo", "input_data": {}},
            tenant="t", user="alice", workspace_root=self.workspace,
        )
        self.stream.append(wake_scheduled_event(plan, created["automation_id"]))
        return created


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    registry = WorkflowRegistry()
    for spec in [*TARGETS.values(), CALLER]:
        registry.register(spec)
    register_controller_bundle(registry)
    runtime = Runtime(run_store=JsonFileRunStore(tmp_path / "runs"), ledger_store=JsonlLedgerStore(tmp_path / "ledger"),
                      workflow_registry=registry, artifact_store=FileArtifactStore(tmp_path / "artifacts"))
    runtime.set_event_inbox(InMemoryEventInbox())
    host = FakeHost(runtime, str(tmp_path / "ws"))
    runtime.set_schedule_run_handler(host)
    return runtime, host, Clock(monkeypatch)


def _call(runtime, schedule, *, allowed_tools=CEILING, session_id="chat-S", extra_runtime=None):
    rt_ns = {"allowed_tools": list(allowed_tools)} if allowed_tools is not None else {}
    rt_ns.update(extra_runtime or {})
    run_id = runtime.start(workflow=CALLER, vars={"schedule": schedule, "_runtime": rt_ns}, session_id=session_id)
    state = runtime.tick(workflow=CALLER, run_id=run_id)
    return run_id, state


def _fire(runtime, clock, automation_id, at_iso):
    clock.set((datetime.fromisoformat(at_iso) + timedelta(seconds=1)).isoformat())
    wake(runtime, automation_id)
    return drive(runtime, automation_id)


def test_a_note_wake_runs_later_in_the_same_session(env):
    runtime, host, clock = env
    drive_now = datetime.now(timezone.utc)
    run_id, state = _call(runtime, {"after_s": 300, "note": "check the oven"})
    assert state.status == RunStatus.COMPLETED
    wake_out = state.output["wake"]
    automation_id = wake_out["automation_id"]
    at = wake_out["at"]
    assert abs((datetime.fromisoformat(at) - drive_now).total_seconds() - 300) < 30
    assert (wake_out["session_id"], wake_out["mode"], wake_out["kind"], wake_out["allowed_tools"]) == ("chat-S", "same_session", "note", [])
    assert host.stream == [{"type": "wake.scheduled", "data": {"automation_id": automation_id, "session_id": "chat-S",
                                                               "origin_run_id": run_id, "at": at}}]
    definition = definition_of(runtime.run_store.load(automation_id))
    assert (definition["session_id"], definition["origin_run_id"], definition["one_shot"]) == ("chat-S", run_id, True)
    assert runtime.run_store.load(automation_id).session_id == f"automation:{automation_id}"  # the controller keeps its own
    drive(runtime, automation_id)
    assert children(runtime, automation_id) == []  # nothing before the time
    _fire(runtime, clock, automation_id, at)
    (kid,) = children(runtime, automation_id)
    assert kid.session_id == "chat-S"
    assert kid.vars["prompt"].startswith("[Trigger schedule@2 · occurrence 1")
    assert "check the oven" in kid.vars["prompt"]
    assert kid.vars["_runtime"]["allowed_tools"] == []
    origin = origin_of(kid.vars)
    assert origin["kind"] == "agent" and origin["ref"]["origin_run_id"] == run_id and origin["ref"]["session_id"] == "chat-S"
    assert session_attribution(runtime.run_store, "chat-S")["kind"] == "chat"  # still a chat
    assert runtime.run_store.load(automation_id).status == RunStatus.COMPLETED  # one-shot: done


def test_a_task_wake_gets_at_most_the_callers_tools_never_schedule_run(env):
    runtime, host, clock = env
    _, state = _call(runtime, {"after_s": 60, "task": "summarize the logs", "tools": ["read_file", "execute_command", "schedule_run"]})
    out = state.output["wake"]
    assert out["allowed_tools"] == ["read_file"]
    assert sorted(out["dropped_tools"]) == ["execute_command", "schedule_run"]
    # Omitted tools: the whole ceiling minus schedule_run.
    _, state2 = _call(runtime, {"after_s": 60, "task": "again", "mode": "independent"})
    out2 = state2.output["wake"]
    assert out2["allowed_tools"] == ["read_file", "web_search"] and out2["session_id"] is None
    _fire(runtime, clock, out["automation_id"], out["at"])
    (kid,) = children(runtime, out["automation_id"])
    assert kid.vars["_runtime"]["allowed_tools"] == ["read_file"]
    assert kid.vars["prompt"].endswith("summarize the logs")
    # The child cannot schedule runs: its ceiling has no schedule_run.
    with pytest.raises(ScheduleRunError) as exc:
        plan_schedule_run(kid, {"after_s": 5, "note": "x"}, now=datetime.now(timezone.utc).isoformat(), run_store=runtime.run_store)
    assert exc.value.reason_code == "not_permitted"


def test_refusals(env):
    runtime, host, _ = env
    now = datetime.now(timezone.utc).isoformat()
    unrestricted = RunState.new(workflow_id="w", entry_node="n", vars={}, session_id="s")
    with pytest.raises(ScheduleRunError) as exc:
        plan_schedule_run(unrestricted, {"after_s": 5, "task": "t"}, now=now)
    assert exc.value.reason_code == "tools_required"  # no ceiling to inherit: name the tools
    assert plan_schedule_run(unrestricted, {"after_s": 5, "task": "t", "tools": ["read_file", "schedule_run"]}, now=now)["allowed_tools"] == ["read_file"]
    for bad in ({"note": "x"}, {"after_s": 5, "at": now, "note": "x"}, {"after_s": 5}, {"after_s": 5, "note": "x", "task": "y"},
                {"at": "2020-01-01T00:00:00Z", "note": "x"}, {"after_s": 5, "note": "x", "mode": "later"},
                {"after_s": 5, "note": "x", "session_id": "someone-else"}):
        with pytest.raises(ScheduleRunError):
            plan_schedule_run(unrestricted, bad, now=now)
    # Through the effect, an agent call gets a tool-style error, never a crash.
    _, state = _call(runtime, {"after_s": 5, "note": "x", "tool_name": "schedule_run", "call_id": "c1"}, allowed_tools=["read_file"])
    (result,) = state.output["wake"]["results"]
    assert result["success"] is False and "may not schedule runs" in result["error"]
    runtime.set_schedule_run_handler(None)
    _, state = _call(runtime, {"after_s": 5, "note": "x"})
    assert state.status == RunStatus.FAILED and "no schedule_run handler" in str(state.error)


def test_untrusted_caller_makes_an_untrusted_wake(env):
    runtime, host, clock = env
    _, state = _call(runtime, {"after_s": 60, "task": "follow up", "tools": ["read_file", "web_search"]},
                     extra_runtime={"untrusted_input": True})
    out = state.output["wake"]
    assert definition_of(runtime.run_store.load(out["automation_id"]))["untrusted_origin"] is True
    _fire(runtime, clock, out["automation_id"], out["at"])
    (kid,) = children(runtime, out["automation_id"])
    assert kid.vars["_runtime"]["untrusted_input"] is True
    assert kid.vars["_runtime"]["tool_policy"]["untrusted_input"] is True


def test_deleting_the_wake_before_it_fires_runs_nothing(env):
    runtime, host, clock = env
    _, state = _call(runtime, {"after_s": 120, "note": "never"})
    out = state.output["wake"]
    automation_id = out["automation_id"]
    drive(runtime, automation_id)
    res = apply_automation_command(runtime, automation_id=automation_id, command_id="del-1", type="automation.archive")
    assert res["status"] == "applied"
    host.stream.append(wake_cancelled_event(automation_id=automation_id, session_id="chat-S", origin_run_id=host.plans[0]["origin_run_id"]))
    drive(runtime, automation_id)
    _fire(runtime, clock, automation_id, out["at"])
    assert children(runtime, automation_id) == []
    assert [e["type"] for e in host.stream] == ["wake.scheduled", "wake.cancelled"]


def test_a_replayed_effect_refinds_the_same_wake(env):
    runtime, host, _ = env
    run_id = runtime.start(workflow=CALLER, vars={"schedule": {"after_s": 30, "note": "once"}, "_runtime": {"allowed_tools": CEILING}},
                           session_id="chat-S")
    run = runtime.run_store.load(run_id)
    effect = Effect(type=EffectType.SCHEDULE_RUN, payload={"after_s": 30, "note": "once"}, result_key="wake")
    first = runtime._handle_schedule_run(run, effect, "done")
    second = runtime._handle_schedule_run(run, effect, "done")
    assert first.result["automation_id"] == second.result["automation_id"]
    assert host.plans[0]["request_id"] == host.plans[1]["request_id"]


def test_an_on_wake_fires_once_on_the_first_event(env, tmp_path):
    from abstractruntime.triggers import TriggerConnectionStore, wake_connection_automations
    from abstractruntime.triggers.webhook import append_webhook_event, sign_standard_webhook

    runtime, host, clock = env
    store = TriggerConnectionStore(tmp_path / "plane", sealing_key_file=tmp_path / "data" / "secrets" / "sealing.key")
    runtime.set_trigger_connection_store(store)
    conn = store.create(principal="t:alice", source_id="webhook.received", source_version=1, config={})
    secret = store.secrets(conn.connection_id, principal="t:alice")["secret"]
    _, state = _call(runtime, {"on": {"source_id": "webhook.received", "connection_id": conn.connection_id}, "task": "handle it",
                               "tools": ["read_file"]})
    automation_id = state.output["wake"]["automation_id"]
    assert state.output["wake"]["on"]["config"] == {"connection_id": conn.connection_id}
    drive(runtime, automation_id)
    clock.set(datetime.now(timezone.utc).isoformat())
    for wid in ("w1", "w2"):
        ts = int(datetime.now(timezone.utc).timestamp())
        body = b'{"n": 1}'
        headers = {"webhook-id": wid, "webhook-timestamp": str(ts), "content-type": "application/json",
                   "webhook-signature": sign_standard_webhook(secret, webhook_id=wid, timestamp=ts, body=body)}
        assert append_webhook_event(runtime.event_inbox, conn, body, headers, store=store)["status"] == "appended"
        wake_connection_automations(runtime, conn.connection_id)
        drive(runtime, automation_id)
    assert len(children(runtime, automation_id)) == 1  # one-shot: the second event runs nothing
    assert runtime.run_store.load(automation_id).status == RunStatus.COMPLETED


def test_a_note_wake_reaches_the_person_a_task_wake_stays_quiet(env):
    """Backlog 1045 follow-up: a note wake's definition carries notify (always, titled with the note), so its
    answer becomes an attention item; a task wake does not unless the caller asks."""
    from abstractruntime.automations import list_attention

    runtime, host, clock = env
    _, note_state = _call(runtime, {"after_s": 60, "note": "call the dentist"})
    note_out = note_state.output["wake"]
    assert note_out["notify"] is True
    note_def = definition_of(runtime.run_store.load(note_out["automation_id"]))
    assert note_def["notify"] == {"channels": ["console"], "always": True, "title": "call the dentist", "body": "call the dentist"}
    _, task_state = _call(runtime, {"after_s": 60, "task": "tidy the notes", "tools": ["read_file"]})
    task_def = definition_of(runtime.run_store.load(task_state.output["wake"]["automation_id"]))
    assert task_state.output["wake"]["notify"] is False and "always" not in task_def["notify"]
    _, asked = _call(runtime, {"after_s": 60, "task": "tidy", "tools": ["read_file"], "notify": True})
    assert definition_of(runtime.run_store.load(asked.output["wake"]["automation_id"]))["notify"]["always"] is True
    _, quiet = _call(runtime, {"after_s": 60, "note": "shh", "notify": False})
    assert "always" not in definition_of(runtime.run_store.load(quiet.output["wake"]["automation_id"]))["notify"]
    # Fired: the note wake raises one attention item (kind notify, title = the note, body = the answer).
    _fire(runtime, clock, note_out["automation_id"], note_out["at"])
    (item,) = list_attention(runtime.ledger_store, note_out["automation_id"])["items"]
    (kid,) = children(runtime, note_out["automation_id"])
    assert item["kind"] == "notify" and item["title"] == "call the dentist"
    assert item["body"].startswith("echo:") and "call the dentist" in item["body"]
    _fire(runtime, clock, task_state.output["wake"]["automation_id"], task_state.output["wake"]["at"])
    assert list_attention(runtime.ledger_store, task_state.output["wake"]["automation_id"])["items"] == []
