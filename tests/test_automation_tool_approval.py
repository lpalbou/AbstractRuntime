"""Unattended tool approval and typed waits (MISSIONS decision D1).

`policy.tool_approval: "auto"` (the default) freezes the runtime's existing
per-run grant `_runtime.tool_policy.auto_approve_tools` into each occurrence at
admission; `"ask"` leaves tools behind the normal approval wait. Discussions
never inherit the grant. `pending_waits` types every human wait by structure.
"""

from __future__ import annotations

import pytest

from automation_harness import TARGETS, Clock, children, create, drive, make_stores, request
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.automations import (
    apply_automation_command,
    create_automation,
    pending_waits,
    register_controller_bundle,
    start_discussion,
)
from abstractruntime.core.models import RunStatus, WaitReason
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_effects import TOOL_EFFECT_CLASSES
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)
from abstractruntime.scheduler.registry import WorkflowRegistry

HOURLY = {"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z", "every": "1h"}}
EXECUTED = []


def _run_command(command: str) -> str:
    EXECUTED.append(command)
    return f"ran {command}"


def _call_tool(run, ctx):
    return StepPlan(
        node_id="call",
        effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [
            {"name": "execute_command", "arguments": {"command": "vm_stat"}, "call_id": "c1"}]}, result_key="tool"),
        next_node="done",
    )


def _done(run, ctx):
    results = (run.vars.get("tool") or {}).get("results") or [{}]
    return StepPlan(node_id="done", complete_output={"response": str(results[0].get("output")), "success": True})


TOOL_TARGET = WorkflowSpec(workflow_id="uses_tool", entry_node="call", nodes={"call": _call_tool, "done": _done})


def make_tool_runtime(run_store, ledger_store) -> Runtime:
    registry = WorkflowRegistry()
    for spec in (*TARGETS.values(), TOOL_TARGET):
        registry.register(spec)
    register_controller_bundle(registry)
    tools = ApprovalToolExecutor(
        delegate=MappingToolExecutor({"execute_command": _run_command}),
        policy=ToolApprovalPolicy(auto_approve_tools=set(), require_approval_tools=set()),  # every tool asks
    )
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry,
                      effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    runtime.set_tool_executor_for_resume(tools)
    return runtime


@pytest.fixture
def env(tmp_path, monkeypatch, request):
    EXECUTED.clear()
    return make_tool_runtime(*make_stores(request.param, tmp_path)), Clock(monkeypatch)


def _create(runtime, clock, *, tool_approval=None):
    req = request(workflow_id="uses_tool", trigger=HOURLY, input_data={"prompt": "memory?"})
    if tool_approval is not None:
        req["policy"] = {"tool_approval": tool_approval}
    return create_automation(runtime, req, now=clock.now)[0]


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_auto_runs_tools_without_a_wait(env):
    runtime, clock = env
    aid = _create(runtime, clock)  # default policy: auto
    drive(runtime, aid)
    child = children(runtime, aid)[0]
    assert child.status == RunStatus.COMPLETED and child.output["response"] == "ran vm_stat"
    assert EXECUTED == ["vm_stat"]
    policy = child.vars["_runtime"]["tool_policy"]
    assert policy["source"] == "automation-policy"
    assert policy["auto_approve_tools"] == sorted(TOOL_EFFECT_CLASSES)
    assert pending_waits(runtime.run_store, aid) == []


@pytest.mark.parametrize("env", ["json", "sqlite"], indirect=True)
def test_ask_parks_on_a_typed_tool_approval_wait(env):
    runtime, clock = env
    aid = _create(runtime, clock, tool_approval="ask")
    drive(runtime, aid)
    child = children(runtime, aid)[0]
    assert child.status == RunStatus.WAITING and "tool_policy" not in (child.vars.get("_runtime") or {})
    assert EXECUTED == []
    [wait] = pending_waits(runtime.run_store, aid)
    assert wait["kind"] == "tool_approval" and wait["reason"] == "user" and wait["index"] == 1
    [call] = wait["details"]
    assert (call["name"], call["arguments"]["command"], call["call_id"]) == ("execute_command", "vm_stat", "c1")
    # The documented answer for this kind approves and runs the calls.
    runtime.resume(workflow=TOOL_TARGET, run_id=child.run_id, wait_key=wait["wait_key"], payload={"approved": True})
    assert runtime.get_state(child.run_id).status == RunStatus.COMPLETED and EXECUTED == ["vm_stat"]


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_the_grant_is_frozen_at_admission(env):
    runtime, clock = env
    aid = _create(runtime, clock, tool_approval="ask")
    drive(runtime, aid)
    first = children(runtime, aid)[0]  # admitted under "ask": parked
    r = apply_automation_command(runtime, automation_id=aid, command_id="to-auto", type="automation.revise",
                                 payload={"changes": {"policy": {"tool_approval": "auto"}}}, now="2026-01-01T00:00:05+00:00")
    assert r["status"] == "applied"
    assert "tool_policy" not in (runtime.get_state(first.run_id).vars.get("_runtime") or {})
    assert pending_waits(runtime.run_store, aid)[0]["kind"] == "tool_approval"  # still asks

    # And the other way: an occurrence admitted under "auto" keeps its grant after a revise to "ask".
    from abstractruntime.automations import controller_workflow_spec

    aid2 = _create(runtime, clock)
    for _ in range(20):  # step the controller until the occurrence is admitted, not yet dispatched
        runtime.tick(workflow=controller_workflow_spec(), run_id=aid2, max_steps=1)
        pending = runtime.get_state(aid2).vars["_runtime"]["automation"]["pending_occurrence"]
        if pending is not None:
            break
    assert pending["phase"] == "admitted" and children(runtime, aid2) == []
    r = apply_automation_command(runtime, automation_id=aid2, command_id="to-ask", type="automation.revise",
                                 payload={"changes": {"policy": {"tool_approval": "ask"}}}, now="2026-01-01T00:00:05+00:00")
    assert r["status"] == "applied"
    drive(runtime, aid2)
    child = children(runtime, aid2)[0]
    assert child.status == RunStatus.COMPLETED and child.vars["_runtime"]["tool_policy"]["source"] == "automation-policy"


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_discussions_never_inherit_the_grant(env, tmp_path):
    runtime, clock = env
    ws = tmp_path / "ws"
    ws.mkdir()
    req = request(workflow_id="uses_tool", trigger=HOURLY, input_data={"prompt": "memory?"}, workspace_root=str(ws))
    aid = create_automation(runtime, req, now=clock.now)[0]
    drive(runtime, aid)
    own = tmp_path / "discussion-ws"
    own.mkdir()
    started = start_discussion(runtime, automation_id=aid, occurrence_index=1, request_id="d", prompt="why?",
                               workspace_root=str(own))
    disc = runtime.get_state(started["run_id"])
    assert "tool_policy" not in disc.vars["_runtime"]


@pytest.mark.parametrize("env", ["json"], indirect=True)
def test_each_wait_kind_is_typed_by_structure(env):
    runtime, clock = env
    ask = create(runtime, clock, workflow_id="ask", trigger=HOURLY)
    event = create(runtime, clock, workflow_id="ask_event", trigger=HOURLY)
    drive(runtime, ask)
    drive(runtime, event)
    [w_ask] = pending_waits(runtime.run_store, ask)
    assert (w_ask["kind"], w_ask["reason"], w_ask["prompt"], w_ask["choices"]) == ("ask_user", "user", "Proceed?", ["yes", "no"])
    assert "details" not in w_ask
    [w_event] = pending_waits(runtime.run_store, event)
    assert (w_event["kind"], w_event["reason"], w_event["prompt"]) == ("event", "event", "Approve?")
    assert "details" not in w_event  # a raw wait key: scope/name unknown
    named = create(runtime, clock, workflow_id="ask_named_event", trigger=HOURLY)
    drive(runtime, named)
    [w_named] = pending_waits(runtime.run_store, named)
    assert w_named["kind"] == "event" and w_named["details"] == {"scope": "session", "name": "approval.requested"}
    # ask_user under the default "auto" policy still waits for a person.
    assert runtime.get_state(w_ask["run_id"]).waiting.reason == WaitReason.USER
