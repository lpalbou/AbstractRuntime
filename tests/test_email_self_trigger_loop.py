"""An email automation never triggers itself (AbstractFramework 0.7.1, RFC 3834 loop guard).

The 0.7.0 end-to-end proof: an automation whose filter matched its own title emailed the user
the result, from the user's account to the user's inbox; the watcher admitted that email and
the automation ran again (5 occurrences in 4 minutes, only the send limits slowed it).

Now, against AbstractCore's hermetic IMAP/SMTP servers:
- mail an automation sends (here the no-model send-email action forwarding to self; agent sends
  go through the same `EmailContext`) carries `Auto-Submitted` + the framework marker;
- the feeder never appends the account's own marked mail (nor a Message-ID the host recorded
  as sent), so exactly one occurrence runs;
- `email.received@1` skips any `Auto-Submitted` mail by default (other auto-responders), a
  typed option admits it; chats (a person is there) send unmarked.
"""

from __future__ import annotations

import email
import email.policy

import pytest

from automation_harness import TARGETS, Clock, children, drive, make_stores, request
from email_wp2_support import (  # noqa: F401 - fixtures
    ALICE,
    _reset_core_resolver,
    add_mail,
    ca,
    imap,
    make_context,
    smtp,
)
from abstractruntime import EffectType, Runtime
from abstractruntime.automations import create_automation, register_controller_bundle
from abstractruntime.core.models import RunStatus
from abstractruntime.email import (
    EmailBinding,
    EmailInboxFeeder,
    JsonFileEventInbox,
    email_action_target,
    register_email_action_workflow,
    wake_email_automations,
)
from abstractruntime.email.binding import automation_marker_for, email_run_scope, resolve_scoped_context
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.triggers import TriggerConfigError
from abstractruntime.triggers.email_received import EmailReceivedTriggerAdapter, message_matches

pytestmark = pytest.mark.basic

REF = "t:alice:1"
T0 = "2026-01-01T00:00:00+00:00"


def _plane(tmp_path, ca, *, imap, smtp, is_own_sent=None):
    from abstractcore.tools.comms_tools import send_email

    run_store, ledger_store = make_stores("json", tmp_path / "plane")
    registry = WorkflowRegistry()
    for spec in TARGETS.values():
        registry.register(spec)
    register_controller_bundle(registry)
    register_email_action_workflow(registry)
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"send_email": send_email}),
                                 policy=ToolApprovalPolicy(auto_approve_tools=set(), require_approval_tools=set()))
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry,
                 artifact_store=FileArtifactStore(tmp_path / "plane" / "artifacts"),
                 effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    rt.set_tool_executor_for_resume(tools)
    ctx = make_context(tmp_path, ca, imap=imap, smtp=smtp, policy_entries=[ALICE])
    rt.set_email_context_resolver(lambda binding: ctx)
    rt.set_email_binding(EmailBinding(REF, ALICE))
    inbox = JsonFileEventInbox(tmp_path / "plane" / "inbox")
    rt.set_event_inbox(inbox)
    return rt, EmailInboxFeeder(inbox, account_ref=REF, is_own_sent=is_own_sent), ctx, inbox


def _self_forwarding_automation(rt, clock, *, config):
    # Forward each matching message to self; the forward's subject still matches the filter
    # ("[<title>] Invoice 7" contains "Invoice"): the exact shape of the 0.7.0 loop.
    target = email_action_target({"to": ["self"], "subject": "[{automation_title}] {subject}", "body": "From {from}\n\n{text}"})
    target["input_data"]["_runtime"] = {"operator_email": ALICE}
    req = request(workflow_id=target["workflow_id"], trigger={"source_id": "email.received", "source_version": 1,
                                                              "config": config})
    req["target"] = target
    return create_automation(rt, req, now=clock.now)[0]


def _deliver(imap, smtp, *, strip=()):
    """What the mail server does: the sent message lands in the account's own INBOX."""
    raw = smtp.messages[-1]["data"]
    if strip:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        for name in strip:
            del msg[name]
        raw = msg.as_bytes()
    return imap.add_message("INBOX", raw)


def _feed(rt, feeder, ctx, now):
    report = feeder.poll(ctx, now=now, force=True)
    wake_email_automations(rt)
    return report


def test_the_result_email_never_triggers_the_automation_again(tmp_path, ca, imap, smtp, monkeypatch):
    rt, feeder, ctx, inbox = _plane(tmp_path, ca, imap=imap, smtp=smtp)
    clock = Clock(monkeypatch)
    feeder.poll(ctx, now=T0, force=True)  # baseline
    aid = _self_forwarding_automation(rt, clock, config={"uses_model": False, "filter": {"subject_contains": "Invoice"}})
    drive(rt, aid)

    add_mail(imap, from_="Carol <carol@example.test>", subject="Invoice 7", text="Please pay.")
    clock.set("2026-01-01T00:00:30+00:00")
    _feed(rt, feeder, ctx, "2026-01-01T00:00:30+00:00")
    drive(rt, aid)
    [occ] = children(rt, aid)
    assert occ.status == RunStatus.COMPLETED and occ.output["success"] is True, occ.output
    [sent] = smtp.messages
    got = email.message_from_bytes(sent["data"], policy=email.policy.default)
    assert got["Subject"] == "[Memory watch] Invoice 7"
    assert got["Auto-Submitted"] == "auto-generated"
    assert got["X-AbstractFramework-Automation"].startswith(f"automation:{aid}/run:")

    # The forward lands in the same inbox and matches the filter: it is never admitted.
    _deliver(imap, smtp)
    for minute in (2, 4, 6):
        now = f"2026-01-01T00:0{minute}:00+00:00"
        clock.set(now)
        report = _feed(rt, feeder, ctx, now)
        drive(rt, aid)
        if minute == 2:
            assert report.own_automatic == 1 and report.appended == []
    assert len(children(rt, aid)) == 1
    assert len(smtp.messages) == 1
    assert inbox.head_seq() == 1  # only Carol's message ever became an event

    # A new message from a person still runs it.
    add_mail(imap, from_="Carol <carol@example.test>", subject="Invoice 8")
    clock.set("2026-01-01T00:08:00+00:00")
    _feed(rt, feeder, ctx, "2026-01-01T00:08:00+00:00")
    drive(rt, aid)
    assert len(children(rt, aid)) == 2


def test_a_marker_a_server_dropped_is_caught_by_auto_submitted(tmp_path, ca, imap, smtp, monkeypatch):
    rt, feeder, ctx, inbox = _plane(tmp_path, ca, imap=imap, smtp=smtp)
    clock = Clock(monkeypatch)
    feeder.poll(ctx, now=T0, force=True)
    aid = _self_forwarding_automation(rt, clock, config={"uses_model": False, "filter": {"subject_contains": "Invoice"}})
    drive(rt, aid)
    add_mail(imap, from_="carol@example.test", subject="Invoice 1")
    clock.set("2026-01-01T00:00:30+00:00")
    _feed(rt, feeder, ctx, "2026-01-01T00:00:30+00:00")
    drive(rt, aid)
    _deliver(imap, smtp, strip=("X-AbstractFramework-Automation",))
    for minute in (2, 4):
        clock.set(f"2026-01-01T00:0{minute}:00+00:00")
        _feed(rt, feeder, ctx, f"2026-01-01T00:0{minute}:00+00:00")
        drive(rt, aid)
    assert inbox.head_seq() == 2  # appended (no marker) but skipped by the trigger (Auto-Submitted)
    assert len(children(rt, aid)) == 1


def test_a_recorded_message_id_is_skipped_even_without_any_header(tmp_path, ca, imap, smtp, monkeypatch):
    sent_ids = set()
    rt, feeder, ctx, inbox = _plane(tmp_path, ca, imap=imap, smtp=smtp, is_own_sent=sent_ids.__contains__)
    clock = Clock(monkeypatch)
    feeder.poll(ctx, now=T0, force=True)
    aid = _self_forwarding_automation(rt, clock, config={"uses_model": False, "filter": {"subject_contains": "Invoice"},
                                                         "auto_submitted": "admit"})
    drive(rt, aid)
    add_mail(imap, from_="carol@example.test", subject="Invoice 1")
    clock.set("2026-01-01T00:00:30+00:00")
    _feed(rt, feeder, ctx, "2026-01-01T00:00:30+00:00")
    drive(rt, aid)
    sent = email.message_from_bytes(smtp.messages[-1]["data"], policy=email.policy.default)
    sent_ids.add(str(sent["Message-ID"]))  # the gateway's outbox records it
    _deliver(imap, smtp, strip=("X-AbstractFramework-Automation", "Auto-Submitted"))
    clock.set("2026-01-01T00:02:00+00:00")
    report = _feed(rt, feeder, ctx, "2026-01-01T00:02:00+00:00")
    drive(rt, aid)
    assert report.own_automatic == 1 and inbox.head_seq() == 1
    assert len(children(rt, aid)) == 1


def _payload(**kw):
    base = {"kind": "email", "folder": "INBOX", "from_address": "bot@example.test", "subject": "Out of office"}
    base.update(kw)
    return base


def test_the_trigger_skips_auto_submitted_mail_by_default_and_the_option_is_typed():
    adapter = EmailReceivedTriggerAdapter()
    cfg = adapter.validate({}, now=T0)
    assert cfg["auto_submitted"] == "skip"
    assert not message_matches(cfg, _payload(auto_submitted="auto-replied"))
    assert not message_matches(cfg, _payload(auto_submitted="auto-generated"))
    assert not message_matches(cfg, _payload(auto_submitted="unknown"))
    assert not message_matches(cfg, _payload(framework_marker="automation:x/run:y"))
    assert message_matches(cfg, _payload(auto_submitted="no"))
    assert message_matches(cfg, _payload(auto_submitted=None))
    admit = adapter.validate({"auto_submitted": "admit"}, now=T0)
    assert message_matches(admit, _payload(auto_submitted="auto-replied"))
    # A stored 0.8.0 config (no key) behaves like the default.
    legacy = {k: v for k, v in cfg.items() if k != "auto_submitted"}
    assert not message_matches(legacy, _payload(auto_submitted="auto-replied"))
    with pytest.raises(TriggerConfigError) as exc:
        adapter.validate({"auto_submitted": "sometimes"}, now=T0)
    assert exc.value.field == "config.auto_submitted"
    schema = adapter.descriptor["config_schema"]["properties"]["auto_submitted"]
    assert schema["enum"] == ["skip", "admit"] and schema["default"] == "skip"


def test_only_automation_runs_mark_their_mail(tmp_path, ca, smtp):
    ctx = make_context(tmp_path, ca, smtp=smtp, policy_entries=[ALICE])
    chat = type("R", (), {"run_id": "chat-1", "vars": {"_runtime": {"email_account": {"account_ref": REF}}}})()
    occurrence = type("R", (), {"run_id": "occ-1", "vars": {
        "_runtime": {"email_account": {"account_ref": REF}},
        "_meta": {"occurrence": {"automation_id": "auto-1", "occurrence_index": 1, "role": "occurrence"}}}})()
    descendant = type("R", (), {"run_id": "sub-1", "vars": {
        "_meta": {"occurrence": {"automation_id": "auto-1", "role": "descendant"}}}})()
    discussion = type("R", (), {"run_id": "d-1", "vars": {"_meta": {"discussion": {"automation_id": "auto-1"}}}})()
    assert automation_marker_for(chat.vars, chat.run_id) == ""
    assert automation_marker_for(discussion.vars, discussion.run_id) == ""  # a person is talking
    assert automation_marker_for(occurrence.vars, occurrence.run_id) == "automation:auto-1/run:occ-1"
    assert automation_marker_for(descendant.vars, descendant.run_id) == "automation:auto-1/run:sub-1"
    with email_run_scope(chat, resolver=lambda b: ctx):
        assert resolve_scoped_context().automation_marker == ""
    with email_run_scope(occurrence, resolver=lambda b: ctx):
        marked = resolve_scoped_context()
    assert marked.automation_marker == "automation:auto-1/run:occ-1"
    assert ctx.automation_marker == ""  # the host's context object is never changed
