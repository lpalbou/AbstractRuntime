"""`sample_occurrence_inputs`: Run on a triggered flow behaves as if the trigger fired, with the
trigger's own configuration, from real data, read only; the event says `abstracttest: true`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from automation_harness import TARGETS, Clock, children, drive, request, wake
from abstractruntime import Runtime
from abstractruntime.automations import create_automation, register_controller_bundle
from abstractruntime.automations.samples import SAMPLE_WEBHOOK_CALL, SampleError, resolved_sample_inputs, sample_occurrence_inputs
from abstractruntime.email.inbox import InMemoryEventInbox
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore, is_artifact_ref
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.triggers import TriggerConnectionStore, connection_principal
from abstractruntime.triggers.feed import FetchResult
from abstractruntime.triggers.webhook import append_webhook_event, sign_standard_webhook

ALICE = connection_principal("t", "alice")
RSS = (Path(__file__).parent / "fixtures" / "feeds" / "rss2.xml").read_bytes()
NOW = "2026-01-01T00:10:00+00:00"


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


# --- a fake read-only mailbox (AbstractCore's client shape: search newest first, get) -------------

class Detail:
    def __init__(self, uid, subject, frm="bob@x.test"):
        self.data = {"uid": uid, "uidvalidity": 1, "folder": "INBOX", "message_id": f"<{uid}@x.test>", "from": frm,
                     "from_address": frm, "to": "alice@example.test", "subject": subject, "date": "d", "body_text": f"{subject} body",
                     "attachments": [], "flags": [], "seen": False}

    def to_dict(self):
        return dict(self.data)


class Mailbox:
    def __init__(self, n):
        self.messages = {uid: Detail(uid, f"mail {uid}", frm="boss@x.test" if uid % 2 else "bob@x.test") for uid in range(1, n + 1)}
        self.gets, self.writes = 0, 0

    def client(self):
        return self

    def search(self, *, folder=None, limit=None, before_uid=None, criteria=None):
        uids = sorted((u for u in self.messages if before_uid is None or u < before_uid), reverse=True)
        page = uids[: limit or len(uids)]
        more = len(uids) > len(page)
        return {"messages": [SimpleNamespace(uid=u) for u in page], "has_more": more, "next_before_uid": page[-1] if more else None}

    def get(self, uid, *, folder=None):
        self.gets += 1
        return self.messages[uid]


def test_email_sample_takes_the_newest_matching_messages_read_only(env):
    runtime, _, _ = env
    runtime.set_email_binding({"account_ref": "t:alice:default", "address": "alice@example.test"})
    box = Mailbox(500)
    head_before = runtime.event_inbox.head_seq()
    sample = sample_occurrence_inputs(runtime, "email.received", 1, {"filter": {"from_in": ["boss@x.test"]}},
                                      count=3, email_context=box, now=NOW)
    assert runtime.event_inbox.head_seq() == head_before  # nothing appended, no cursor anywhere
    inputs = sample["inputs"]
    assert sample["resolve_vars"] == ["messages"] and is_artifact_ref(inputs["messages"])  # bodies as artifacts
    resolved = resolved_sample_inputs(sample, artifact_store=runtime.artifact_store)
    assert [m["subject"] for m in resolved["messages"]] == ["mail 499", "mail 497", "mail 495"]  # newest first, filtered
    assert resolved["mailbox"] == "alice@example.test"
    event = inputs["event"]
    assert event["abstracttest"] is True and event["id"].startswith("test:") and event["type"] == "abstract.email.received.v1"
    assert sample["truncated"] is False and sample["count"] == 3


def test_email_sample_stops_at_its_budget(env):
    runtime, _, _ = env
    runtime.set_email_binding({"account_ref": "t:alice:default", "address": "alice@example.test"})
    sample = sample_occurrence_inputs(runtime, "email.received", 1, {}, count=100, email_context=Mailbox(100_000),
                                      now=NOW, email_budget_s=0.0)
    assert sample["truncated"] is True


def test_email_sample_without_a_mailbox_is_refused(env):
    runtime, _, _ = env
    with pytest.raises(SampleError) as exc:
        sample_occurrence_inputs(runtime, "email.received", 1, {}, now=NOW)
    assert exc.value.reason_code == "email_not_configured"


def test_feed_sample_is_the_latest_items_without_touching_the_cursor(env):
    runtime, store, _ = env
    conn = store.create(principal=ALICE, source_id="feed.item", source_version=1, config={"url": "https://news.example.test/f", "every": "60s"})
    calls = []
    fetch = lambda url, **kw: calls.append(kw) or FetchResult(status=200, body=RSS)  # noqa: E731
    sample = sample_occurrence_inputs(runtime, "feed.item", 1, {"connection_id": conn.connection_id}, count=1,
                                      principal=ALICE, now=NOW, fetch=fetch)
    assert calls == [{"etag": None, "last_modified": None, "timeout": 20.0}]
    assert [i["title"] for i in sample["inputs"]["items"]] == ["Second story"]  # the newest
    assert sample["inputs"]["feed"] == {"title": "Example News", "url": "https://news.example.test/f"}
    assert runtime.event_inbox.read(stream=conn.stream) == [] and runtime.event_inbox.stream_state(conn.stream) == {}
    with pytest.raises(SampleError) as exc:
        sample_occurrence_inputs(runtime, "feed.item", 1, {"connection_id": conn.connection_id}, principal="t:bob", fetch=fetch)
    assert exc.value.reason_code == "connection_not_found"


def test_webhook_sample_is_the_last_calls_else_one_documented_call(env):
    runtime, store, _ = env
    conn = store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={})
    cfg = {"connection_id": conn.connection_id}
    empty = sample_occurrence_inputs(runtime, "webhook.received", 1, cfg, principal=ALICE, now=NOW)
    (call,) = empty["inputs"]["calls"]
    assert call["id"] == "sample" and call["body"] == SAMPLE_WEBHOOK_CALL["body"]
    secret = store.secrets(conn.connection_id, principal=ALICE)["secret"]
    for i in range(3):
        ts, raw = 1_767_225_600 + i, f'{{"n": {i}}}'.encode()
        headers = {"webhook-id": f"w{i}", "webhook-timestamp": str(ts), "content-type": "application/json",
                   "webhook-signature": sign_standard_webhook(secret, webhook_id=f"w{i}", timestamp=ts, body=raw)}
        assert append_webhook_event(runtime.event_inbox, conn, raw, headers, store=store, now=ts)["status"] == "appended"
    head = runtime.event_inbox.head_seq()
    sample = sample_occurrence_inputs(runtime, "webhook.received", 1, cfg, count=2, principal=ALICE, now=NOW)
    assert [c["id"] for c in sample["inputs"]["calls"]] == ["w2", "w1"] and sample["inputs"]["calls"][0]["body"] == {"n": 2}
    assert runtime.event_inbox.head_seq() == head


def test_schedule_and_manual(env):
    runtime, _, _ = env
    sched = sample_occurrence_inputs(runtime, "schedule", 2, {"kind": "every", "every": "5m"}, now=NOW)["inputs"]
    assert (sched["scheduled_at"], sched["occurrence"], sched["event"]["abstracttest"]) == (NOW, 0, True)
    manual = sample_occurrence_inputs(runtime, "manual", 1, {}, now=NOW)["inputs"]
    assert set(manual) == {"event"}


def test_a_sample_has_the_same_named_inputs_as_a_real_occurrence(env):
    """Same keys, same shapes: the sample is what build_prepared produces (one source checked end to end)."""
    runtime, store, clock = env
    conn = store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={})
    secret = store.secrets(conn.connection_id, principal=ALICE)["secret"]
    req = request(trigger={"source_id": "webhook.received", "source_version": 1, "config": {"connection_id": conn.connection_id}})
    req.update({"tenant": "t", "user": "alice"})
    aid, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, aid)
    ts, raw = 1_767_225_600, b'{"a": 1}'
    headers = {"webhook-id": "w9", "webhook-timestamp": str(ts), "content-type": "application/json",
               "webhook-signature": sign_standard_webhook(secret, webhook_id="w9", timestamp=ts, body=raw)}
    append_webhook_event(runtime.event_inbox, conn, raw, headers, store=store, now=ts)
    from abstractruntime.triggers import wake_connection_automations
    wake_connection_automations(runtime, conn.connection_id)
    drive(runtime, aid)
    (kid,) = children(runtime, aid)
    sample = sample_occurrence_inputs(runtime, "webhook.received", 1, {"connection_id": conn.connection_id}, principal=ALICE, now=NOW)
    assert sample["inputs"]["calls"] == kid.vars["calls"]
    real_event, test_event = kid.vars["event"], sample["inputs"]["event"]
    assert set(real_event) | {"abstracttest"} == set(test_event) and "abstracttest" not in real_event
