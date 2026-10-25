"""A feed connection is polled only while something listens (framework backlog 1041/1043 follow-up):
`paused`, `always_listen`, the derived `listening` / `used_by`, and the feed poller and webhook
ingress honouring them."""

from __future__ import annotations

from pathlib import Path

import pytest

from automation_harness import TARGETS, Clock, children, drive, request
from abstractruntime import Runtime
from abstractruntime.automations import apply_automation_command, create_automation, register_controller_bundle
from abstractruntime.email.inbox import InMemoryEventInbox
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.triggers import TriggerConnectionError, TriggerConnectionStore, connection_principal
from abstractruntime.triggers.connections import connection_usage
from abstractruntime.triggers.feed import FetchResult, poll_feed_connections
from abstractruntime.triggers.webhook import append_webhook_event, sign_standard_webhook

ALICE = connection_principal("t", "alice")
RSS = (Path(__file__).parent / "fixtures" / "feeds" / "rss2.xml").read_bytes()
URL = "https://news.example.test/feed.xml"
T0 = "2026-01-01T00:00:00+00:00"
NOW_S = 1_767_225_600


class Feed:
    def __init__(self, body: bytes) -> None:
        self.body, self.calls = body, 0

    def __call__(self, url, *, etag=None, last_modified=None, timeout=None):
        self.calls += 1
        return FetchResult(status=200, body=self.body, etag=None)


def _with_item(guid: str) -> bytes:
    extra = f"<item><title>{guid}</title><link>https://news.example.test/{guid}</link><guid>{guid}</guid></item>"
    return RSS.replace(b"<item>", extra.encode() + b"<item>", 1)


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


def _bind(runtime, clock, conn, source="feed.item"):
    req = request(trigger={"source_id": source, "source_version": 1, "config": {"connection_id": conn.connection_id}})
    req.update({"tenant": "t", "user": "alice"})
    automation_id, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, automation_id)
    return automation_id


def _feed(store, **kw):
    return store.create(principal=ALICE, source_id="feed.item", source_version=1, config={"url": URL, "every": "60s"}, **kw)


def test_flags_persist_and_the_listening_state_is_derived(env):
    runtime, store, clock = env
    conn = _feed(store)
    assert (conn.paused, conn.always_listen) == (False, False)
    assert connection_usage(runtime, conn) == {"listening": "idle", "used_by": 0}
    automation_id = _bind(runtime, clock, conn)
    assert connection_usage(runtime, conn) == {"listening": "listening", "used_by": 1}
    row = conn.to_dict(usage=connection_usage(runtime, conn))
    assert (row["listening"], row["used_by"], row["status"], row["paused"], row["always_listen"]) == ("listening", 1, "active", False, False)
    # A paused automation does not count.
    apply_automation_command(runtime, automation_id=automation_id, command_id="p1", type="automation.pause")
    assert connection_usage(runtime, store.get(conn.connection_id)) == {"listening": "idle", "used_by": 0}
    lone = store.update(conn.connection_id, principal=ALICE, always_listen=True)
    assert connection_usage(runtime, lone) == {"listening": "listening", "used_by": 0}
    paused = store.update(conn.connection_id, principal=ALICE, paused=True)
    assert store.get(conn.connection_id).paused is True and store.get(conn.connection_id).status == "active"
    assert connection_usage(runtime, paused)["listening"] == "paused"
    disabled = store.update(conn.connection_id, principal=ALICE, paused=False, status="disabled")
    assert connection_usage(runtime, disabled)["listening"] == "paused"
    with pytest.raises(TriggerConnectionError):
        store.update(conn.connection_id, principal=ALICE, paused="yes")
    assert _feed(store, always_listen=True).always_listen is True


def test_an_idle_feed_is_never_fetched_and_stores_nothing(env):
    runtime, store, clock = env
    conn = _feed(store, )
    feed = Feed(RSS)
    rows = poll_feed_connections(runtime, store, fetch=feed, now=T0)
    assert feed.calls == 0 and runtime.event_inbox.read(stream=conn.stream) == []
    assert (rows[0]["listening"], rows[0]["used_by"], rows[0]["skipped"]) == ("idle", 0, True)
    # always_listen: fetched with nothing bound.
    store.update(conn.connection_id, principal=ALICE, always_listen=True)
    poll_feed_connections(runtime, store, fetch=feed, now="2026-01-01T00:01:00+00:00")
    assert feed.calls == 1
    # paused wins over always_listen and over a bound automation.
    store.update(conn.connection_id, principal=ALICE, paused=True)
    _bind(runtime, clock, store.get(conn.connection_id))
    rows = poll_feed_connections(runtime, store, fetch=feed, now="2026-01-01T00:05:00+00:00")
    assert feed.calls == 1 and rows == []  # a paused connection is not even a candidate


def test_binding_after_an_idle_spell_starts_from_a_fresh_baseline(env):
    runtime, store, clock = env
    conn = _feed(store)
    feed = Feed(RSS)
    poll_feed_connections(runtime, store, fetch=feed, now=T0)  # idle: marked, nothing fetched
    feed.body = _with_item("published-while-idle")
    _bind(runtime, clock, conn)
    rows = poll_feed_connections(runtime, store, fetch=feed, now="2026-01-01T00:00:30+00:00")
    assert feed.calls == 1 and rows[0]["baseline"] and rows[0]["appended"] == []  # history, not news
    feed.body = _with_item("fresh")
    rows = poll_feed_connections(runtime, store, fetch=feed, now="2026-01-01T00:02:00+00:00")
    assert len(rows[0]["appended"]) == 1 and rows[0]["listening"] == "listening"


def _signed(secret, wid, body=b'{"n": 1}'):
    return body, {"webhook-id": wid, "webhook-timestamp": str(NOW_S), "content-type": "application/json",
                  "webhook-signature": sign_standard_webhook(secret, webhook_id=wid, timestamp=NOW_S, body=body)}


def test_webhook_idle_accepts_and_stores_paused_refuses(env):
    """UX ruling: a call to an idle webhook connection is accepted and stored (a first test right after
    creating the connection works); only a paused or disabled connection refuses."""
    runtime, store, clock = env
    conn = store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={})
    secret = store.secrets(conn.connection_id, principal=ALICE)["secret"]
    inbox = runtime.event_inbox
    assert connection_usage(runtime, conn)["listening"] == "idle"
    idle = append_webhook_event(inbox, conn, *_signed(secret, "w1"), store=store, now=NOW_S, runtime=runtime)
    assert (idle["status"], idle["http_status"]) == ("appended", 202)
    _bind(runtime, clock, conn, source="webhook.received")
    ok = append_webhook_event(inbox, conn, *_signed(secret, "w2"), store=store, now=NOW_S, runtime=runtime)
    assert ok["status"] == "appended"
    paused = store.update(conn.connection_id, principal=ALICE, paused=True)
    out = append_webhook_event(inbox, paused, *_signed(secret, "w3"), store=store, now=NOW_S, runtime=runtime)
    assert (out["http_status"], out["reason"]) == (409, "connection_disabled")
    out = append_webhook_event(inbox, paused, *_signed(secret, "w4"), secret=secret, now=NOW_S)  # without runtime too
    assert out["reason"] == "connection_disabled"
    assert [e["payload"]["subject"] for e in inbox.read(stream=conn.stream)] == ["w1", "w2"]
    disabled = store.update(conn.connection_id, principal=ALICE, paused=False, status="disabled")
    assert append_webhook_event(inbox, disabled, *_signed(secret, "w5"), secret=secret, now=NOW_S)["reason"] == "connection_disabled"


def test_feed_connection_schema_reads_as_plain_language():
    from abstractruntime.triggers.feed import FeedItemTriggerAdapter

    props = FeedItemTriggerAdapter.descriptor["connection_schema"]["properties"]
    assert set(props) == {"url", "every", "max_items", "initial_items"}  # field names unchanged
    assert props["every"]["title"] == "Check every" and "number and a unit" in props["every"]["description"]
    assert props["initial_items"]["title"] == "When added"
    assert dict(zip(props["initial_items"]["enum"], props["initial_items"]["enumNames"])) == {
        "skip": "Ignore what is already there", "deliver": "Also deliver the latest items"}
    assert all(p.get("title") and p.get("description") for p in props.values())
