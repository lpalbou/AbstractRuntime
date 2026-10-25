"""`email.received@1` on a further mailbox (framework backlog 1044): `config.account: <mailbox id>`.

Runs without AbstractCore's mail servers: the inbox records are appended as the gateway's watcher
(`EmailInboxFeeder`) would, on the streams `email:<tenant>:<user>:<account_id>:<folder>`."""

from __future__ import annotations

import pytest

from automation_harness import TARGETS, Clock, children, drive, request, wake
from abstractruntime import Runtime
from abstractruntime.automations import create_automation, register_controller_bundle
from abstractruntime.email.binding import EMAIL_ACCOUNT_KEY
from abstractruntime.email.inbox import InMemoryEventInbox
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.triggers import EmailReceivedTriggerAdapter, TriggerConfigError
from abstractruntime.triggers.email_received import message_matches

T0 = "2026-01-01T00:00:00+00:00"
PRIMARY = "t:alice:default"
A = EmailReceivedTriggerAdapter()


def test_account_accepts_a_mailbox_id_and_self_stays_the_primary():
    assert A.validate({}, now=T0)["account"] == "self"
    assert A.validate({"account": "second"}, now=T0)["account"] == "second"
    assert A.validate({"account": " work.box-2 "}, now=T0)["account"] == "work.box-2"
    for bad in ("", "  ", "t:alice:second", "two words", 7, None):
        with pytest.raises(TriggerConfigError) as exc:
            A.validate({"account": bad}, now=T0)
        assert exc.value.field == "config.account"
    schema = A.descriptor["config_schema"]["properties"]["account"]
    assert "enum" not in schema and schema["default"] == "self"
    assert A.descriptor["connection_provider"] == "email_accounts"


def _mail(account_id: str, uid: int, subject: str) -> dict:
    return {"kind": "email", "account_ref": f"t:alice:{account_id}", "folder": "INBOX", "uidvalidity": 1, "uid": uid,
            "message_id": f"<{account_id}-{uid}@example.test>", "from": f"x{uid}@example.test",
            "from_address": f"x{uid}@example.test", "to": "alice@example.test", "subject": subject,
            "body_text": f"{subject} body", "date": T0}


def test_message_matches_only_the_named_mailbox():
    second = A.validate({"account": "second"}, now=T0)
    primary = A.validate({}, now=T0)
    assert message_matches(second, _mail("second", 1, "s")) and not message_matches(second, _mail("default", 1, "p"))
    assert message_matches(primary, _mail("default", 1, "p"))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    registry = WorkflowRegistry()
    for spec in TARGETS.values():
        registry.register(spec)
    register_controller_bundle(registry)
    rt = Runtime(run_store=JsonFileRunStore(tmp_path / "runs"), ledger_store=JsonlLedgerStore(tmp_path / "ledger"),
                 workflow_registry=registry, artifact_store=FileArtifactStore(tmp_path / "artifacts"))
    rt.set_event_inbox(InMemoryEventInbox())
    rt.set_email_binding({"account_ref": PRIMARY, "address": "alice@example.test"})
    return rt


def test_an_automation_on_the_second_mailbox_reads_it_and_acts_for_it(runtime, monkeypatch):
    clock = Clock(monkeypatch)
    req = request(trigger={"source_id": "email.received", "source_version": 1,
                           "config": {"account": "second", "start_at": T0, "uses_model": False}},
                  input_data={"prompt": "Tell me what matters"})
    automation_id, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, automation_id)
    later = "2026-01-01T00:05:00+00:00"
    inbox = runtime.event_inbox
    inbox.append(stream="email:t:alice:default:INBOX", event_id="p1", payload=_mail("default", 1, "primary mail"), appended_at=later)
    inbox.append(stream="email:t:alice:second:INBOX", event_id="s1", payload=_mail("second", 1, "second mail"), appended_at=later)
    clock.set("2026-01-01T00:06:00+00:00")
    wake(runtime, automation_id)
    drive(runtime, automation_id)
    (kid,) = children(runtime, automation_id)
    envelope = kid.vars["_meta"]["occurrence"]["trigger_envelope"]
    assert envelope["payload"]["event_ids"] == ["s1"]  # the second mailbox's mail only
    assert envelope["source"] == "abstract://email/accounts/second/INBOX"
    assert kid.vars["_runtime"][EMAIL_ACCOUNT_KEY]["account_ref"] == "t:alice:second"  # acts for that mailbox
