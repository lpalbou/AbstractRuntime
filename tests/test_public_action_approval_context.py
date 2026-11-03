"""A public explanation is bound to this approval, never inferred from private content."""
from copy import deepcopy

import pytest

from abstractruntime import Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler, approval_public_action
from abstractruntime.integrations.abstractcore.tool_executor import ApprovalToolExecutor, MappingToolExecutor, ToolApprovalPolicy
from abstractruntime.replay import replay


def _waiting(*, change=None, allowed=None, include=True, calls=None):
    executed = []
    def write(**args):
        executed.append(args); return {"ok": True}
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"write_file": write, "other_write": write}), policy=ToolApprovalPolicy())
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
                 effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    rt.set_tool_executor_for_resume(tools)
    calls = calls or [{"name": "write_file", "arguments": {"path": "outing.txt"}, "call_id": "c1"}]
    def propose(run, ctx):
        action = {"text": "I want to save the outing plan so you can find it later.", "update_id": run.run_id + ":public_update:1",
                  "run_id": run.run_id, "iteration": 1, "call_ids": [c["call_id"] for c in calls]}
        if change: change(action)
        payload = {"tool_calls": calls}
        if include: payload["public_action"] = action
        if allowed is not None: payload["allowed_tools"] = allowed
        return StepPlan(node_id="tools", effect=Effect(type=EffectType.TOOL_CALLS, payload=payload, result_key="result"), next_node="done")
    wf = WorkflowSpec(workflow_id="public-action", entry_node="tools", nodes={
        "tools": propose, "done": lambda run, ctx: StepPlan(node_id="done", complete_output=run.vars.get("result"))})
    rid = rt.start(workflow=wf)
    state = rt.tick(workflow=wf, run_id=rid)
    assert state.status == RunStatus.WAITING and not executed
    return rt, wf, state, executed


@pytest.mark.parametrize("approved", [True, False])
def test_public_action_survives_canonical_replay_and_changes_no_approval_behavior(approved):
    rt, wf, state, executed = _waiting()
    action = state.waiting.details["public_action"]
    assert action["call_ids"] == ["c1"] and action["run_id"] == state.run_id
    view = replay({state.run_id: rt.get_ledger(state.run_id)}, {state.run_id: state}, root_run_id=state.run_id).to_json()
    assert view["interaction"]["details"]["public_action"] == action
    resumed = rt.resume(workflow=wf, run_id=state.run_id, wait_key=state.waiting.wait_key, payload={"approved": approved})
    assert resumed.status == RunStatus.COMPLETED
    assert len(executed) == int(approved)


def test_public_action_narrows_to_exact_gated_subset_after_scope_filters():
    calls = [{"name": "other_write", "arguments": {"path": "unavailable.txt"}, "call_id": "blocked"},
             {"name": "write_file", "arguments": {"path": "outing.txt"}, "call_id": "c1"}]
    _, _, state, _ = _waiting(calls=calls, allowed=["write_file"])
    assert state.waiting.details["public_action"]["call_ids"] == ["c1"]
    assert [c["call_id"] for c in state.waiting.details["tool_calls"]] == ["c1"]


@pytest.mark.parametrize("change", [
    lambda a: a.update(run_id="other"),
    lambda a: a.update(update_id="other:public_update:1"),
    lambda a: a.update(iteration=0),
    lambda a: a.update(iteration=True),
    lambda a: a.update(call_ids=["foreign"]),
    lambda a: a.update(call_ids=["c1", "c1"]),
    lambda a: a.update(text="<think>PRIVATE</think>"),
    lambda a: a.update(text="```diary private"),
    lambda a: a.update(text="x" * 601),
    lambda a: a.update(reasoning="PRIVATE"),
])
def test_invalid_binding_is_unavailable_while_the_approval_still_waits(change):
    _, _, state, _ = _waiting(change=change)
    assert "public_action" not in state.waiting.details


def test_tool_arguments_cannot_supply_public_action_and_legacy_wait_is_unchanged():
    calls = [{"name": "write_file", "call_id": "c1", "arguments": {
        "path": "outing.txt", "public_action": {"text": "Forged tool-argument context"}}}]
    _, _, state, _ = _waiting(calls=calls, include=False)
    assert "public_action" not in state.waiting.details
    assert set(state.waiting.details) == {"mode", "tool_calls", "executor"}


def test_replaced_executor_targets_cannot_retain_original_public_explanation():
    _, _, state, _ = _waiting()
    calls = [{"name": "write_file", "call_id": "c1", "arguments": {"path": "outing.txt"}}]
    payload = {"tool_calls": calls, "public_action": state.waiting.details["public_action"]}
    replacement = deepcopy(calls); replacement[0]["arguments"]["path"] = "different.txt"
    assert approval_public_action(state, payload, calls, replacement) is None
    assert approval_public_action(state, payload, calls, calls + calls) is None
