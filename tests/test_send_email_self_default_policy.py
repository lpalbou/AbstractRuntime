"""A chat run's send to the user's own address runs unasked under the gateway's DEFAULT
executor policy (AbstractRuntime 0.8.1; 0.7.0 Linux end-to-end F1b).

The gateway builds `ApprovalToolExecutor(policy=ToolApprovalPolicy())`: the default policy lists
`send_email` in `require_approval_tools`. A run without `_runtime.tool_policy` took the static
branch, which passed that list to the refiner pass as "require", so `send_email_recipient@v2`
(self / pre-authorised recipients -> auto) never applied and every self-send parked.

Pinned here with that exact executor: self -> sent; a stranger -> approval; a self-send beside a
stranger -> approval; a per-run require list (the user's explicit choice) still wins; an
email-triggered (untrusted-input) run under "ask" still asks.
"""

from __future__ import annotations

import pytest

from email_wp2_support import (  # noqa: F401 - fixtures
    ALICE,
    STRANGER,
    _reset_core_resolver,
    ca,
    make_context,
    smtp,
)
from automation_harness import make_stores
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.email import bind_email_account
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)
from abstractruntime.scheduler.registry import WorkflowRegistry

pytestmark = pytest.mark.basic

SELF = ALICE


def _send_node(run, ctx):
    calls = [{"name": "send_email", "arguments": {"to": to, "subject": "Status", "body_text": "ok"}, "call_id": f"c{i}"}
             for i, to in enumerate(run.vars["recipients"])]
    return StepPlan(node_id="send", effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": calls}, result_key="sent"),
                    next_node="done")


def _done_node(run, ctx):
    return StepPlan(node_id="done", complete_output={"success": True, "sent": run.vars.get("sent")})


SENDER = WorkflowSpec(workflow_id="sender", entry_node="send", nodes={"send": _send_node, "done": _done_node})


def _runtime(tmp_path, ctx):
    from abstractcore.tools.comms_tools import send_email

    run_store, ledger_store = make_stores("json", tmp_path / "rt")
    registry = WorkflowRegistry()
    registry.register(SENDER)
    # EXACTLY the gateway's executor (abstractgateway hosts/bundle_host.py): the default policy.
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"send_email": send_email}), policy=ToolApprovalPolicy())
    assert "send_email" in tools.policy.require_approval_tools  # the premise of the finding
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry,
                 effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    rt.set_tool_executor_for_resume(tools)
    rt.set_email_context_resolver(lambda binding: ctx)
    return rt


def _start(rt, recipients, **runtime_ns):
    vars0 = bind_email_account({"recipients": recipients, "_runtime": {"operator_email": SELF, **runtime_ns}},
                               account_ref="t:alice:1", address=ALICE)
    rid = rt.start(workflow=SENDER, vars=vars0)
    return rt.tick(workflow=SENDER, run_id=rid)


def test_a_chat_run_mails_its_owner_without_asking(tmp_path, ca, smtp):
    rt = _runtime(tmp_path, make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF, STRANGER]))
    state = _start(rt, [SELF])
    assert state.status == RunStatus.COMPLETED, state.waiting
    assert [m["rcpt_tos"] for m in smtp.messages] == [[SELF]]


def test_a_stranger_or_a_mixed_batch_still_asks(tmp_path, ca, smtp):
    rt = _runtime(tmp_path, make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF, STRANGER]))
    for recipients in ([STRANGER], [SELF, STRANGER]):
        state = _start(rt, recipients)
        assert state.status == RunStatus.WAITING and state.waiting.details["mode"] == "approval_required"
    assert smtp.messages == []


def test_a_per_run_require_list_still_wins(tmp_path, ca, smtp):
    rt = _runtime(tmp_path, make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF]))
    state = _start(rt, [SELF], tool_policy={"auto_approve_tools": ["read_file"], "require_approval_tools": ["send_email"]})
    assert state.status == RunStatus.WAITING
    assert smtp.messages == []


def test_an_untrusted_input_run_under_ask_still_asks(tmp_path, ca, smtp):
    rt = _runtime(tmp_path, make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF]))
    state = _start(rt, [SELF], untrusted_input=True,
                   tool_policy={"approval": "ask", "auto_approve_tools": ["send_email"], "untrusted_input": True})
    assert state.status == RunStatus.WAITING
    assert smtp.messages == []
