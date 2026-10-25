"""The Flow contract of an automation occurrence: the trigger's data as flat, named inputs beside
`input_data.event`, and the output names the controller reads (`response`, `notify`, `success`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from automation_harness import TARGETS, Clock, children, drive, request, wake
from abstractruntime import Runtime
from abstractruntime.automations import apply_automation_command, create_automation, register_controller_bundle
from abstractruntime.automations.attention import normalize_occurrence_output
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.email.inbox import InMemoryEventInbox
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore, is_artifact_ref
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.triggers import TriggerConnectionStore, connection_principal, wake_connection_automations
from abstractruntime.triggers.feed import FetchResult, poll_feed_connections
from abstractruntime.triggers.webhook import append_webhook_event, sign_standard_webhook

T0 = "2026-01-01T00:00:00+00:00"
ALICE = connection_principal("t", "alice")
RSS = (Path(__file__).parent / "fixtures" / "feeds" / "rss2.xml").read_bytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    registry = WorkflowRegistry()
    for spec in TARGETS.values():
        registry.register(spec)
    register_controller_bundle(registry)
    runtime = Runtime(run_store=JsonFileRunStore(tmp_path / "runs"), ledger_store=JsonlLedgerStore(tmp_path / "ledger"),
                      workflow_registry=registry, artifact_store=FileArtifactStore(tmp_path / "artifacts"))
    runtime.set_event_inbox(InMemoryEventInbox())
    store = TriggerConnectionStore(tmp_path / "plane", sealing_key_file=tmp_path / "data" / "secrets" / "sealing.key")
    runtime.set_trigger_connection_store(store)
    return runtime, store, Clock(monkeypatch)


def _create(runtime, clock, trigger):
    req = request(trigger=trigger)
    req.update({"tenant": "t", "user": "alice"})
    automation_id, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, automation_id)
    return automation_id


def test_schedule_gives_scheduled_at_and_occurrence_manual_gives_nothing(env):
    runtime, _, clock = env
    aid = _create(runtime, clock, {"source_id": "schedule", "source_version": 2, "config": {"kind": "every", "every": "5m", "start_at": T0}})
    (kid,) = children(runtime, aid)
    assert (kid.vars["scheduled_at"], kid.vars["occurrence"]) == (T0, 1)
    manual = _create(runtime, clock, {"source_id": "manual", "source_version": 1, "config": {}})
    apply_automation_command(runtime, automation_id=manual, command_id="go", type="automation.run_now")
    drive(runtime, manual)
    (mk,) = children(runtime, manual)
    assert not {"scheduled_at", "occurrence", "messages", "items", "calls"} & set(mk.vars)


def test_feed_gives_items_and_feed(env):
    runtime, store, clock = env
    conn = store.create(principal=ALICE, source_id="feed.item", source_version=1, config={"url": "https://news.example.test/f", "every": "60s"})
    aid = _create(runtime, clock, {"source_id": "feed.item", "source_version": 1, "config": {"connection_id": conn.connection_id}})
    body = {"b": RSS}
    fetch = lambda url, **kw: FetchResult(status=200, body=body["b"])  # noqa: E731
    poll_feed_connections(runtime, store, fetch=fetch, now=T0)  # baseline
    body["b"] = RSS.replace(b"<item>", b"<item><title>New</title><link>https://n.test/3</link><guid>g3</guid>"
                                       b"<pubDate>Tue, 06 Jan 2026 12:00:00 GMT</pubDate><description>S</description></item><item>", 1)
    clock.set("2026-01-01T00:05:00+00:00")
    poll_feed_connections(runtime, store, fetch=fetch, now="2026-01-01T00:05:00+00:00")
    drive(runtime, aid)
    (kid,) = children(runtime, aid)
    assert kid.vars["items"] == [{"id": "g3", "title": "New", "link": "https://n.test/3", "published": "2026-01-06T12:00:00+00:00",
                                  "summary": "S", "author": ""}]
    assert kid.vars["feed"] == {"title": "Example News", "url": "https://news.example.test/f"}
    assert kid.vars["event"]["data"]["events"][0]["data"]["title"] == "New"  # the envelope stays


def test_webhook_gives_calls(env):
    runtime, store, clock = env
    conn = store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={})
    secret = store.secrets(conn.connection_id, principal=ALICE)["secret"]
    aid = _create(runtime, clock, {"source_id": "webhook.received", "source_version": 1, "config": {"connection_id": conn.connection_id}})
    clock.set("2026-01-01T00:00:10+00:00")
    ts, raw = 1_767_225_605, b'{"order": 7}'
    headers = {"Webhook-Id": "evt_7", "Webhook-Timestamp": str(ts), "Content-Type": "application/json", "X-Source": "shop",
               "Authorization": "Bearer nope", "Webhook-Signature": sign_standard_webhook(secret, webhook_id="evt_7", timestamp=ts, body=raw)}
    assert append_webhook_event(runtime.event_inbox, conn, raw, headers, store=store, now=ts)["status"] == "appended"
    wake_connection_automations(runtime, conn.connection_id)
    drive(runtime, aid)
    (kid,) = children(runtime, aid)
    (call,) = kid.vars["calls"]
    assert call == {"id": "evt_7", "received_at": "2026-01-01T00:00:05+00:00", "body": {"order": 7},
                    "headers": {"webhook-id": "evt_7", "webhook-timestamp": str(ts), "content-type": "application/json", "x-source": "shop"}}


def _mail(account_id, uid, subject):
    return {"kind": "email", "account_ref": f"t:alice:{account_id}", "folder": "INBOX", "uidvalidity": 1, "uid": uid,
            "message_id": f"<{uid}@x.test>", "from": "Bob <bob@x.test>", "from_address": "bob@x.test", "to": "alice@example.test",
            "subject": subject, "date": "Thu, 01 Jan 2026 00:04:00 +0000", "body_text": f"{subject} body",
            "attachments": [{"filename": "a.pdf", "content_type": "application/pdf", "size": 12, "index": 0}]}


@pytest.mark.parametrize("account, mailbox", [("self", "alice@example.test"), ("second", "second")])
def test_email_gives_messages_and_mailbox(env, account, mailbox):
    runtime, _, clock = env
    runtime.set_email_binding({"account_ref": "t:alice:default", "address": "alice@example.test"})
    aid = _create(runtime, clock, {"source_id": "email.received", "source_version": 1,
                                   "config": {"account": account, "start_at": T0, "uses_model": False}})
    acct = "default" if account == "self" else account
    runtime.event_inbox.append(stream=f"email:t:alice:{acct}:INBOX", event_id="m1", payload=_mail(acct, 1, "Hello"),
                               appended_at="2026-01-01T00:05:00+00:00")
    clock.set("2026-01-01T00:06:00+00:00")
    wake(runtime, aid)
    drive(runtime, aid)
    (kid,) = children(runtime, aid)
    assert kid.vars["mailbox"] == mailbox
    assert kid.vars["messages"] == [{"id": "<1@x.test>", "from": "Bob <bob@x.test>", "to": "alice@example.test", "subject": "Hello",
                                     "date": "Thu, 01 Jan 2026 00:04:00 +0000", "body_text": "Hello body",
                                     "attachments": [{"name": "a.pdf", "size": 12, "artifact": None, "index": 0}]}]
    # The controller's own record keeps a ref, never the bodies; the old keys stay for one release.
    assert kid.vars["trigger"]["emails"][0]["subject"] == "Hello" and kid.vars["event"]["data"]["messages"][0]["subject"] == "Hello"
    prepared = [r for r in runtime.ledger_store.list(aid) if "automation.admitted" in str(r.get("effect"))]
    assert prepared and "Hello body" not in str(prepared)


def test_the_output_names_the_controller_reads():
    def done(output, status=RunStatus.COMPLETED):
        run = RunState.new(workflow_id="w", entry_node="n", vars={})
        run.status, run.output = status, output
        return normalize_occurrence_output(run)

    out = done({"response": "All good", "success": True, "notify": {"title": "T", "body": "B"}})
    assert out == {"answer": "All good", "success": True, "notify": {"title": "T", "body": "B"}, "error": None}
    assert done({"response": "x", "success": False})["success"] is False
    assert done({"response": "quiet"})["notify"] is None
