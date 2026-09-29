"""Run-scoped email accounts and the `send_email_recipient@v2` refiner (framework backlog 0992 WP2).

- A run acts for ONE account: its `_runtime.email_account` binding, resolved at the moment of
  use through its own Runtime's resolver; credentials never enter vars, the ledger or results.
- A child run inherits the binding and the pre-authorised recipients (the parent wins).
- `send_email` runs unattended only to "self" (the registered address) or to recipients the
  user pre-authorised in the automation definition; anything else waits for a person.
- An automation with the default policy cannot mail an unlisted address without an approval
  wait, even when an inbound email asks for it.
"""

from __future__ import annotations

import pytest

from automation_harness import TARGETS, Clock, children, drive, make_stores, request
from email_wp2_support import (  # noqa: F401 - fixtures
    ALICE,
    ALICE_PASSWORD,
    BOB,
    BOB_PASSWORD,
    STRANGER,
    _reset_core_resolver,
    ca,
    imap,
    make_context,
    smtp,
    tree_text,
)
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.automations import create_automation, pending_waits, register_controller_bundle
from abstractruntime.core.models import RunStatus
from abstractruntime.email import EmailBinding, bind_email_account, binding_of, strip_client_email_keys
from abstractruntime.integrations.abstractcore.effect_handlers import (
    _TOOL_REFINERS,
    _send_email_recipient_refiner_v2,
    make_tool_calls_handler,
)
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)
from abstractruntime.scheduler.registry import WorkflowRegistry

pytestmark = pytest.mark.basic

SELF = "owner@example.test"
BOSS = "boss@example.test"


class _Run:
    def __init__(self, rt):
        self.vars = {"_runtime": rt}


def _call(to=None, cc=None, bcc=None, name="send_email", **extra):
    args = {"subject": "s", "body_text": "b", **extra}
    if to is not None:
        args["to"] = to
    if cc is not None:
        args["cc"] = cc
    if bcc is not None:
        args["bcc"] = bcc
    return {"name": name, "arguments": args}


def _rt(allowed=None, operator=SELF):
    rt = {}
    if operator:
        rt["operator_email"] = operator
    if allowed is not None:
        rt["email_allowed_recipients"] = allowed
    return rt


# --- the refiner ------------------------------------------------------------------------


def test_v2_is_registered_for_the_core_row():
    from abstractruntime.integrations.abstractcore.effect_handlers import _risk_row_for_tool

    assert _TOOL_REFINERS["send_email_recipient@v2"] is _send_email_recipient_refiner_v2
    assert _risk_row_for_tool("send_email")["risk_refiner"] == "send_email_recipient@v2"
    assert _risk_row_for_tool("reply_email")["risk_refiner"] == "send_email_recipient@v2"


@pytest.mark.parametrize(
    "call, rt, expected",
    [
        (_call(to=SELF), _rt(), "auto"),  # self (registered address)
        (_call(to=" Owner@Example.TEST "), _rt(), "auto"),  # strip + lowercase only
        (_call(to=BOSS), _rt(allowed=["self", BOSS]), "auto"),  # pre-authorised
        (_call(to=[SELF, BOSS], cc=BOSS), _rt(allowed=["self", BOSS]), "auto"),
        (_call(to=BOSS), _rt(allowed=["self"]), "ask"),  # not listed
        (_call(to=SELF, cc=STRANGER), _rt(allowed=["self"]), "ask"),  # one stranger in cc
        (_call(to=SELF, bcc=STRANGER), _rt(allowed=["self"]), "ask"),  # one stranger in bcc
        (_call(to=f"Owner <{SELF}>"), _rt(), "ask"),  # display-name form never matches
        (_call(to=SELF.replace("o", "о", 1)), _rt(), "ask"),  # homoglyph (Cyrillic o)
        (_call(to=[]), _rt(), "ask"),  # no recipient: never a vacuous auto
        (_call(to=SELF), _rt(operator=None), "ask"),  # no self value, nothing listed
        (_call(to=SELF), _rt(allowed=["self"], operator=None), "ask"),  # "self" without a registered address
        (_call(to=SELF), _rt(allowed=[]), "auto"),  # the registered address is always self
        (_call(to=BOSS), _rt(allowed=[f"Boss <{BOSS}>", "*@example.test", "example.test"]), "ask"),  # not plain addresses
        (_call(to=SELF, name="reply_email"), _rt(), "ask"),  # recipients come from the original
        ({"name": "reply_email", "arguments": {"uid": "3", "body_text": "x"}}, _rt(), "ask"),
        (_call(to=SELF, recipients=STRANGER), _rt(), "ask"),  # unknown argument key
        ({"name": "send_email", "arguments": {"arguments": {"to": SELF}}}, _rt(), "ask"),  # wrapper shape
        ({"name": "send_email", "arguments": "{not json"}, _rt(), "ask"),
        ({"name": "send_email", "arguments": '{"to": "owner@example.test", "subject": "s", "body_text": "b"}'}, _rt(), "auto"),
    ],
)
def test_refiner_v2_decisions(call, rt, expected):
    assert _send_email_recipient_refiner_v2(call, _Run(rt)) == expected


def test_refiner_v2_fails_toward_asking():
    class Broken:
        @property
        def vars(self):
            raise RuntimeError("boom")

    assert _send_email_recipient_refiner_v2(_call(to=SELF), Broken()) == "ask"
    assert _send_email_recipient_refiner_v2(None, _Run(_rt())) == "ask"


# --- binding helpers ---------------------------------------------------------------------


def test_binding_helpers_set_and_strip():
    v = {"_runtime": {"email_account": {"account_ref": "t:mallory:x"}, "email_allowed_recipients": [STRANGER], "keep": 1}}
    strip_client_email_keys(v)
    assert v["_runtime"] == {"keep": 1}
    bind_email_account(v, account_ref="t:alice:1", address=ALICE)
    assert binding_of(v) == EmailBinding("t:alice:1", ALICE)
    bind_email_account(v, binding=None)
    assert binding_of(v) is None
    with pytest.raises(ValueError):
        EmailBinding("")


# --- execution: each run resolves ITS account -----------------------------------------------


def _send_node(run, ctx):
    to = run.vars.get("to") or SELF
    return StepPlan(
        node_id="send",
        effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [
            {"name": "send_email", "arguments": {"to": to, "subject": "Status", "body_text": "all good"}, "call_id": "s1"}]},
            result_key="sent"),
        next_node="done",
    )


def _spawn_node(run, ctx):
    return StepPlan(
        node_id="spawn",
        effect=Effect(type=EffectType.START_SUBWORKFLOW, payload={
            "workflow_id": "sender",
            # A child that tries to pick another account and widen its recipients.
            "vars": {"to": run.vars.get("to"), "_runtime": {"email_account": {"account_ref": "t:mallory:9"},
                                                               "email_allowed_recipients": [STRANGER]}},
        }, result_key="child"),
        next_node="done",
    )


def _done_node(run, ctx):
    results = ((run.vars.get("sent") or {}).get("results")) or [{}]
    return StepPlan(node_id="done", complete_output={"success": True, "result": results[0].get("output")})


SENDER = WorkflowSpec(workflow_id="sender", entry_node="send", nodes={"send": _send_node, "done": _done_node})
SPAWNER = WorkflowSpec(workflow_id="spawner", entry_node="spawn", nodes={"spawn": _spawn_node, "done": _done_node})


def _runtime(tmp_path, name, *, auto=("send_email",)):
    from abstractcore.tools.comms_tools import send_email

    run_store, ledger_store = make_stores("json", tmp_path / name)
    registry = WorkflowRegistry()
    for spec in (*TARGETS.values(), SENDER, SPAWNER):
        registry.register(spec)
    register_controller_bundle(registry)
    tools = ApprovalToolExecutor(
        delegate=MappingToolExecutor({"send_email": send_email}),
        policy=ToolApprovalPolicy(auto_approve_tools=set(auto), require_approval_tools=set()),
    )
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry,
                 effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    rt.set_tool_executor_for_resume(tools)
    return rt


def _result(runtime, run_id):
    return runtime.get_state(run_id).output["result"]


def test_each_runtime_resolves_its_own_users_account(tmp_path, ca, smtp):
    """Two users in one process: each run sends from its own account; no crossing."""
    alice_ctx = make_context(tmp_path, ca, smtp=smtp, address=ALICE, password=ALICE_PASSWORD, policy_entries=[SELF])
    bob_ctx = make_context(tmp_path, ca, smtp=smtp, address=BOB, password=BOB_PASSWORD, policy_entries=[SELF])
    seen = []

    def resolver_for(ref, ctx):
        def resolve(binding):
            seen.append(binding.account_ref)
            return ctx if binding.account_ref == ref else None  # a host checks the ref is its user's

        return resolve

    rt_a = _runtime(tmp_path, "alice")
    rt_b = _runtime(tmp_path, "bob")
    rt_a.set_email_context_resolver(resolver_for("t:alice:1", alice_ctx))
    rt_b.set_email_context_resolver(resolver_for("t:bob:1", bob_ctx))
    va = bind_email_account({"to": SELF}, account_ref="t:alice:1", address=ALICE)
    vb = bind_email_account({"to": SELF}, account_ref="t:bob:1", address=BOB)
    ra = rt_a.start(workflow=SENDER, vars=va)
    rb = rt_b.start(workflow=SENDER, vars=vb)
    rt_a.tick(workflow=SENDER, run_id=ra)
    rt_b.tick(workflow=SENDER, run_id=rb)
    assert _result(rt_a, ra)["from"] == ALICE and _result(rt_b, rb)["from"] == BOB
    senders = sorted(m["mail_from"] for m in smtp.messages)
    assert senders == [ALICE, BOB]
    assert seen == ["t:alice:1", "t:bob:1"]

    # A run carrying ANOTHER user's ref on Bob's runtime gets nothing from Bob's resolver.
    forged = bind_email_account({"to": SELF}, account_ref="t:alice:1", address=ALICE)
    rf = rt_b.start(workflow=SENDER, vars=forged)
    rt_b.tick(workflow=SENDER, run_id=rf)
    assert _result(rt_b, rf)["error_code"] == "email_not_configured"
    assert len(smtp.messages) == 2

    # No credential anywhere in either runtime's run store or ledger.
    text = tree_text(tmp_path / "alice", tmp_path / "bob")
    assert ALICE_PASSWORD not in text and BOB_PASSWORD not in text
    assert "localhost" not in str(_result(rt_a, ra))  # results carry no host


def test_an_unbound_run_never_falls_back_to_a_local_account(tmp_path, ca, smtp):
    ctx = make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF])
    rt = _runtime(tmp_path, "rt")
    rt.set_email_context_resolver(lambda binding: ctx)
    rid = rt.start(workflow=SENDER, vars={"to": SELF})  # no binding
    rt.tick(workflow=SENDER, run_id=rid)
    out = _result(rt, rid)
    assert out["success"] is False and out["error_code"] == "email_not_configured"
    assert "Settings -> Email" in out["fix"]
    assert smtp.messages == []


def test_child_runs_inherit_the_binding_and_the_parent_wins(tmp_path, ca, smtp):
    ctx = make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF, STRANGER])
    refs = []
    rt = _runtime(tmp_path, "rt", auto=())  # every send goes through the refiner
    rt.set_email_context_resolver(lambda binding: (refs.append(binding.account_ref), ctx)[1])
    parent_vars = bind_email_account({"to": STRANGER, "_runtime": {"operator_email": SELF, "email_allowed_recipients": ["self"]}},
                                     account_ref="t:alice:1", address=ALICE)
    rid = rt.start(workflow=SPAWNER, vars=parent_vars)
    rt.tick(workflow=SPAWNER, run_id=rid)
    [child] = rt.run_store.list_children(parent_run_id=rid)
    assert child.vars["_runtime"]["email_account"] == {"account_ref": "t:alice:1", "address": ALICE}
    assert child.vars["_runtime"]["email_allowed_recipients"] == ["self"]
    # The child's own attempt to widen recipients lost: mailing the stranger waits for a person.
    assert child.status == RunStatus.WAITING and child.waiting.details.get("mode") == "approval_required"
    assert smtp.messages == [] and refs == []
    # Approving on resume runs the send AS the child (its binding, this runtime's resolver).
    rt.resume(workflow=SENDER, run_id=child.run_id, wait_key=child.waiting.wait_key, payload={"approved": True})
    assert refs == ["t:alice:1"] and [m["rcpt_tos"] for m in smtp.messages] == [[STRANGER]]


# --- automations: the default policy cannot mail an unlisted address ----------------------------


def _obey_inbound(run, ctx):
    """An agent that obeys a prompt injection: it mails whatever address the input names."""
    to = (run.vars.get("inbound") or {}).get("reply_to") or SELF
    return StepPlan(
        node_id="send",
        effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [
            {"name": "send_email", "arguments": {"to": to, "subject": "data", "body_text": "private"}, "call_id": "m1"}]},
            result_key="sent"),
        next_node="done",
    )


OBEY = WorkflowSpec(workflow_id="obey", entry_node="send", nodes={"send": _obey_inbound, "done": _done_node})


def _automation_runtime(tmp_path, ctx):
    from abstractcore.tools.comms_tools import send_email

    run_store, ledger_store = make_stores("json", tmp_path / "plane")
    registry = WorkflowRegistry()
    for spec in (*TARGETS.values(), OBEY):
        registry.register(spec)
    register_controller_bundle(registry)
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"send_email": send_email}),
                                 policy=ToolApprovalPolicy(auto_approve_tools=set(), require_approval_tools=set()))
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry,
                 effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    rt.set_tool_executor_for_resume(tools)
    rt.set_email_context_resolver(lambda binding: ctx)
    rt.set_email_binding(EmailBinding("t:alice:1", ALICE))
    return rt


HOURLY = {"source_id": "schedule", "source_version": 1, "config": {"start_at": "2026-01-01T00:00:00Z", "every": "1h"}}


def _mail_automation(rt, clock, *, reply_to, policy=None):
    input_data = {"prompt": "handle mail", "inbound": {"reply_to": reply_to},
                  # A client/target trying to widen what the occurrence may mail: replaced by the definition's list.
                  "_runtime": {"operator_email": SELF, "email_allowed_recipients": [STRANGER]}}
    req = request(workflow_id="obey", trigger=HOURLY, input_data=input_data)
    if policy is not None:
        req["policy"] = policy
    return create_automation(rt, req, now=clock.now)[0]


def test_default_automation_cannot_mail_an_unlisted_address(tmp_path, ca, smtp, monkeypatch):
    ctx = make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF, STRANGER, BOSS])  # the policy would allow it
    rt = _automation_runtime(tmp_path, ctx)
    clock = Clock(monkeypatch)
    aid = _mail_automation(rt, clock, reply_to=STRANGER)
    drive(rt, aid)
    child = children(rt, aid)[0]
    assert child.vars["_runtime"]["email_allowed_recipients"] == ["self"]  # SET from the definition
    assert child.vars["_runtime"]["email_account"] == {"account_ref": "t:alice:1", "address": ALICE}
    assert child.status == RunStatus.WAITING
    [wait] = pending_waits(rt.run_store, aid)
    assert wait["kind"] == "tool_approval" and wait["details"][0]["arguments"]["to"] == STRANGER
    assert smtp.messages == []


def test_pre_authorised_recipients_and_self_run_unattended(tmp_path, ca, smtp, monkeypatch):
    ctx = make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF, BOSS])
    rt = _automation_runtime(tmp_path, ctx)
    clock = Clock(monkeypatch)
    aid = _mail_automation(rt, clock, reply_to=BOSS, policy={"email_allowed_recipients": ["self", BOSS]})
    drive(rt, aid)
    assert children(rt, aid)[0].status == RunStatus.COMPLETED
    aid2 = _mail_automation(rt, clock, reply_to=SELF)
    drive(rt, aid2)
    assert children(rt, aid2)[0].status == RunStatus.COMPLETED
    assert sorted(m["rcpt_tos"][0] for m in smtp.messages) == [BOSS, SELF]
    assert ALICE_PASSWORD not in tree_text(tmp_path / "plane")


def test_the_recipient_policy_still_refuses_after_approval(tmp_path, ca, smtp, monkeypatch):
    """Approval decides whether a send runs unattended; the account's policy decides who can
    receive mail at all (AbstractCore guarded_send), even for a pre-authorised recipient."""
    ctx = make_context(tmp_path, ca, smtp=smtp, policy_entries=[SELF])  # BOSS not in the allowlist
    rt = _automation_runtime(tmp_path, ctx)
    clock = Clock(monkeypatch)
    aid = _mail_automation(rt, clock, reply_to=BOSS, policy={"email_allowed_recipients": ["self", BOSS]})
    drive(rt, aid)
    child = children(rt, aid)[0]
    assert child.status == RunStatus.COMPLETED
    assert child.output["result"]["error_code"] == "email_policy_refused"
    assert smtp.messages == []
