"""LIVE runtime email tests against a real test mailbox (opt-in; framework backlog 0992 WP2).

Skipped unless every AF_TEST_EMAIL_* variable is set (the harness input; the runtime gets the
account the way a host hands it over: an in-memory `EmailContext` behind
`Runtime.set_email_context_resolver`). Not marked `basic`, so CI never selects them. Run:

    set -a; . <your mailbox env file>; set +a; \
    python -m pytest tests/live_email -m live_email -s --durations=0

What is proved against the real server:

- an automation with the `email.received@1` trigger admits exactly one uniquely-tagged
  self-sent message, once: not again on a second poll, a later wake, or a restarted runtime;
  the feeder's reads leave the message's \\Seen flag as it was;
- a `send_email` tool call to the account's own address runs unattended (and arrives), while
  one to another address parks on a `tool_approval` wait and is never sent — even once approved,
  the account's recipient policy refuses it before any SMTP connection.

Mail only ever goes to the test account's own address (a guard refuses anything else before
MAIL FROM). Messages are tagged "[af-live-test]" and stay small.
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from automation_harness import TARGETS, Clock, children, drive, make_stores, request  # noqa: E402
from live_email_support import ARRIVAL_TIMEOUT_S, FOREIGN, fact, new_nonce, tagged_subject, wait_for  # noqa: E402

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec  # noqa: E402
from abstractruntime.automations import create_automation, pending_waits, register_controller_bundle  # noqa: E402
from abstractruntime.core.models import RunStatus  # noqa: E402
from abstractruntime.email import (  # noqa: E402
    EmailBinding,
    EmailInboxFeeder,
    JsonFileEventInbox,
    bind_email_account,
    email_trigger_consumers,
    wake_email_automations,
)
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler  # noqa: E402
from abstractruntime.integrations.abstractcore.tool_executor import (  # noqa: E402
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)
from abstractruntime.scheduler.registry import WorkflowRegistry  # noqa: E402

pytestmark = pytest.mark.live_email

REF = "live:self:1"
# A per-run tool policy as a gateway chat run carries it (a rank ceiling below outreach). The
# send_email_recipient@v2 refiner runs per call with or without it: inside the per-run policy
# pass here, and over the executor's static policy for a run WITHOUT any `_runtime.tool_policy`
# (hermetic: test_send_to_self_runs_unattended_without_a_run_policy), so a send to self runs
# unattended either way; a send to any other address still asks.
CHAT_POLICY = {"auto_approve_max_risk_rank": 1}


def tree_text(*roots: Path) -> str:
    """Every byte under the given directories, decoded leniently (for secret greps)."""
    return "\n".join(p.read_bytes().decode("utf-8", errors="replace")
                     for root in roots if root.exists() for p in sorted(root.rglob("*")) if p.is_file())


def _now(offset_s: float = 0.0) -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=offset_s)).isoformat()


# --- workflows -------------------------------------------------------------------------------

SEEN = []


def _record_node(run, ctx):
    SEEN.append({"trigger": run.vars.get("trigger"), "tool_policy": (run.vars.get("_runtime") or {}).get("tool_policy")})
    return StepPlan(node_id="answer", complete_output={"response": "handled", "success": True})


def _send_node(run, ctx):
    return StepPlan(
        node_id="send",
        effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [
            {"name": "send_email", "call_id": "s1",
             "arguments": {"to": run.vars.get("to"), "subject": run.vars.get("subject"), "body_text": run.vars.get("body")}}]},
            result_key="sent"),
        next_node="done",
    )


def _done_node(run, ctx):
    results = ((run.vars.get("sent") or {}).get("results")) or [{}]
    return StepPlan(node_id="done", complete_output={"success": True, "result": results[0].get("output")})


RECORDER = WorkflowSpec(workflow_id="live_recorder", entry_node="answer", nodes={"answer": _record_node})
SENDER = WorkflowSpec(workflow_id="live_sender", entry_node="send", nodes={"send": _send_node, "done": _done_node})


def _plane(tmp_path: Path, mailbox, *, root: str = "plane"):
    """One user's runtime: json stores, an event inbox, the account behind the host resolver.

    Tool approval: nothing is pre-approved by name; `send_email` goes through the
    `send_email_recipient@v2` refiner (self = `_runtime.operator_email`)."""
    from abstractcore.tools.comms_tools import send_email

    run_store, ledger_store = make_stores("json", tmp_path / root)
    registry = WorkflowRegistry()
    for spec in (*TARGETS.values(), RECORDER, SENDER):
        registry.register(spec)
    register_controller_bundle(registry)
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"send_email": send_email}),
                                 policy=ToolApprovalPolicy(auto_approve_tools=set(), require_approval_tools=set()))
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry,
                 effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    rt.set_tool_executor_for_resume(tools)
    ctx = mailbox.context(tmp_path / f"{root}-limits")
    resolved = []
    rt.set_email_context_resolver(lambda binding: (resolved.append(binding.account_ref), ctx if binding.account_ref == REF else None)[1])
    rt.set_email_binding(EmailBinding(REF, mailbox.address))
    inbox = JsonFileEventInbox(tmp_path / root / "inbox")
    rt.set_event_inbox(inbox)
    return rt, EmailInboxFeeder(inbox, account_ref=REF), ctx, inbox, resolved


def _count_smtp_connections(monkeypatch):
    from abstractcore.comms.email import EmailClient

    opened = []
    real = EmailClient._smtp
    monkeypatch.setattr(EmailClient, "_smtp", lambda self: (opened.append(1), real(self))[1])
    return opened


# --- email.received@1 ------------------------------------------------------------------------


def test_live_email_trigger_admits_one_tagged_message_exactly_once(live_mailbox, smtp_self_only, tmp_path, monkeypatch):
    from abstractcore.comms.email import OutgoingMessage, SearchCriteria

    SEEN.clear()
    rt, feeder, ctx, inbox, _ = _plane(tmp_path, live_mailbox)
    clock = Clock(monkeypatch, now=_now())
    t0 = time.monotonic()
    base = feeder.poll(ctx, now=_now(), force=True)
    assert base.ok and base.baseline, f"baseline poll failed: {base.error and base.error.get('code')}"
    nonce = new_nonce()
    subject = tagged_subject("runtime-trigger", nonce)
    req = request(workflow_id=RECORDER.workflow_id,
                  trigger={"source_id": "email.received", "source_version": 1,
                           "config": {"uses_model": False, "filter": {"subject_contains": nonce}}},
                  input_data={"prompt": "Handle my new mail.", "_runtime": {"operator_email": live_mailbox.address}})
    aid = create_automation(rt, req, now=clock.now)[0]
    assert email_trigger_consumers(rt) == [aid]
    state = drive(rt, aid)
    assert state.status == RunStatus.WAITING and state.waiting.until is None  # idle until mail arrives

    # Mail after the automation was created (appended_at must be later than its start_at).
    time.sleep(1.0)
    ctx.send(OutgoingMessage(to=(live_mailbox.address,), subject=subject,
                             text=f"AbstractFramework live email test (runtime trigger). Nonce: {nonce}\n"))
    assert smtp_self_only.count == 1
    client = ctx.client()
    crit = SearchCriteria.build(subject_contains=nonce)
    found, arrival_s, polls = wait_for(lambda: client.search(crit, limit=5)["messages"])
    assert found, f"the tagged message did not arrive within {ARRIVAL_TIMEOUT_S:.0f} s"
    flags_before = set(found[0].flags) - {"\\Recent"}

    # Poll 1: the feeder appends it (whole) to the durable inbox; the automation admits it.
    report1 = feeder.poll(ctx, now=_now(), force=True)
    mine = [e for e in inbox.read() if nonce in str((e.get("payload") or {}).get("subject") or "")]
    assert report1.ok and len(mine) == 1
    clock.set(_now())
    woken = wake_email_automations(rt)
    drive(rt, aid)
    kids = children(rt, aid)
    assert woken == [aid] and len(kids) == 1 and kids[0].status == RunStatus.COMPLETED
    [seen] = SEEN
    trig = seen["trigger"]
    assert trig["source"] == "email.received@1" and trig["content_trust"] == "untrusted" and trig["count"] == 1
    assert trig["emails"][0]["subject"] == subject and nonce in trig["emails"][0]["body_text"]
    assert "send_email" in seen["tool_policy"]["withheld_tools"]

    # Poll 2 + a later wake: nothing is appended or admitted twice.
    report2 = feeder.poll(ctx, now=_now(), force=True)
    mine_after = [e for e in inbox.read() if nonce in str((e.get("payload") or {}).get("subject") or "")]
    assert report2.ok and len(mine_after) == 1
    clock.set(_now(300))  # past the 60 s batch interval
    wake_email_automations(rt)
    drive(rt, aid)
    assert len(children(rt, aid)) == 1 and len(SEEN) == 1

    # A restarted runtime over the same stores admits nothing new either.
    rt2, feeder2, ctx2, inbox2, _ = _plane(tmp_path, live_mailbox)
    report3 = feeder2.poll(ctx2, now=_now(301), force=True)
    clock.set(_now(600))
    wake_email_automations(rt2)
    drive(rt2, aid)
    assert report3.ok and len(children(rt2, aid)) == 1 and len(SEEN) == 1

    # Read-only: the feeder's reads (BODY.PEEK) left \Seen exactly as it was.
    flags_after = set(next(m.flags for m in client.search(crit, limit=5)["messages"] if m.uid == found[0].uid)) - {"\\Recent"}
    src = rt2.get_state(aid).vars["_runtime"]["automation"]["source_state"]
    fact(live_mailbox, "runtime_trigger", {
        "arrival_s": round(arrival_s, 1), "polls_to_arrival": polls, "total_s": round(time.monotonic() - t0, 1),
        "poll1_appended": len(report1.appended), "poll2_appended": len(report2.appended), "poll3_appended": len(report3.appended),
        "occurrences": len(children(rt2, aid)), "flags_before": sorted(flags_before), "flags_after": sorted(flags_after),
        "mail_cursor_uidvalidity": (src.get("mail_cursor") or {}).get("uidvalidity"),
    })
    assert flags_before == flags_after, "the feeder changed the message's flags"
    assert not live_mailbox.contains_secret(tree_text(tmp_path / "plane")), "the password appears in the run/ledger/inbox store"


# --- send_email: self unattended, others park -------------------------------------------------


def test_live_send_email_to_self_runs_unattended(live_mailbox, smtp_self_only, tmp_path):
    from abstractcore.comms.email import SearchCriteria

    rt, _feeder, ctx, _inbox, resolved = _plane(tmp_path, live_mailbox)
    nonce = new_nonce()
    subject = tagged_subject("runtime-tool", nonce)
    run_vars = bind_email_account({"to": live_mailbox.address, "subject": subject,
                                   "body": f"AbstractFramework live email test (runtime send_email). Nonce: {nonce}\n",
                                   "_runtime": {"operator_email": live_mailbox.address, "tool_policy": CHAT_POLICY}},
                                  account_ref=REF, address=live_mailbox.address)
    t0 = time.monotonic()
    rid = rt.start(workflow=SENDER, vars=run_vars)
    rt.tick(workflow=SENDER, run_id=rid)
    run_s = time.monotonic() - t0
    state = rt.get_state(rid)
    out = (state.output or {}).get("result") or {}
    assert state.status == RunStatus.COMPLETED, f"the run did not complete unattended: {state.status}"
    assert out.get("success") is True, f"send_email failed: {out.get('error_code')}"
    assert smtp_self_only.count == 1 and smtp_self_only.transactions[0]["all_self"]
    assert resolved == [REF]
    client = ctx.client()
    found, arrival_s, polls = wait_for(lambda: client.search(SearchCriteria.build(subject_contains=nonce), limit=5)["messages"])
    fact(live_mailbox, "runtime_send_self", {"run_s": round(run_s, 2), "arrival_s": round(arrival_s, 1), "polls": polls,
                                             "arrived": bool(found), "result_has_host": live_mailbox.smtp_host in str(out)})
    assert found and len(found) == 1, f"the self-sent message did not arrive within {ARRIVAL_TIMEOUT_S:.0f} s"
    assert live_mailbox.smtp_host not in str(out)  # results never carry the server
    assert not live_mailbox.contains_secret(tree_text(tmp_path / "plane")), "the password appears in the run/ledger store"


def test_live_send_email_to_another_address_parks_and_is_never_sent(live_mailbox, smtp_self_only, tmp_path, monkeypatch):
    opened = _count_smtp_connections(monkeypatch)
    rt, _feeder, _ctx, _inbox, _resolved = _plane(tmp_path, live_mailbox)
    run_vars = bind_email_account({"to": FOREIGN, "subject": tagged_subject("runtime-foreign", new_nonce()), "body": "never sent",
                                   "_runtime": {"operator_email": live_mailbox.address, "tool_policy": CHAT_POLICY}},
                                  account_ref=REF, address=live_mailbox.address)
    rid = rt.start(workflow=SENDER, vars=run_vars)
    rt.tick(workflow=SENDER, run_id=rid)
    state = rt.get_state(rid)
    assert state.status == RunStatus.WAITING and state.waiting.details.get("mode") == "approval_required"
    assert opened == [] and smtp_self_only.count == 0
    # Approval does not widen who can receive mail: the account's policy (allowlist = own
    # address) refuses before any SMTP connection.
    rt.resume(workflow=SENDER, run_id=rid, wait_key=state.waiting.wait_key, payload={"approved": True})
    state = rt.get_state(rid)
    out = (state.output or {}).get("result") or {}
    fact(live_mailbox, "runtime_send_foreign", {"parked": True, "after_approval": out.get("error_code"),
                                                "smtp_connections": len(opened), "smtp_transactions": smtp_self_only.count})
    assert state.status == RunStatus.COMPLETED and out.get("error_code") == "email_policy_refused"
    assert opened == [] and smtp_self_only.count == 0


def test_live_default_automation_parks_a_send_to_another_address(live_mailbox, smtp_self_only, tmp_path, monkeypatch):
    """The load-bearing control: an automation with the default grant cannot mail an unlisted
    address without a person (the tool_approval wait); nothing reaches SMTP."""
    opened = _count_smtp_connections(monkeypatch)
    rt, _feeder, _ctx, _inbox, _resolved = _plane(tmp_path, live_mailbox)
    clock = Clock(monkeypatch, now=_now())
    req = request(workflow_id=SENDER.workflow_id,
                  trigger={"source_id": "schedule", "source_version": 1, "config": {"start_at": clock.now, "every": "1h"}},
                  input_data={"to": FOREIGN, "subject": tagged_subject("runtime-automation", new_nonce()), "body": "never sent",
                              "_runtime": {"operator_email": live_mailbox.address}})
    aid = create_automation(rt, req, now=clock.now)[0]
    drive(rt, aid)
    [child] = children(rt, aid)
    [wait] = pending_waits(rt.run_store, aid)
    fact(live_mailbox, "runtime_automation_foreign", {"child_status": str(child.status), "wait_kind": wait["kind"],
                                                      "smtp_connections": len(opened)})
    assert child.status == RunStatus.WAITING and wait["kind"] == "tool_approval"
    assert wait["details"][0]["arguments"]["to"] == FOREIGN
    assert opened == [] and smtp_self_only.count == 0
