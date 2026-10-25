"""`webhook.received@1` (framework backlog 1042, runtime half): Standard Webhooks verification,
the HTTP-body mapping, `append_webhook_event`, and a bound automation."""

from __future__ import annotations

import json

import pytest

from automation_harness import TARGETS, Clock, children, drive, request
from abstractruntime import Runtime
from abstractruntime.automations import create_automation, register_controller_bundle
from abstractruntime.email.inbox import InMemoryEventInbox
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.triggers import TriggerConnectionError, TriggerConnectionStore, connection_principal, wake_connection_automations
from abstractruntime.triggers.webhook import (
    append_webhook_event,
    new_webhook_secret,
    sign_standard_webhook,
    verify_standard_webhook,
    webhook_event_data,
)

ALICE = connection_principal("t", "alice")
NOW = 1_767_225_600  # 2026-01-01T00:00:00Z
# The Standard Webhooks reference vector (standardwebhooks.com libraries).
REF_SECRET = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
REF_ID, REF_TS, REF_BODY = "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330, b'{"test": 2432232314}'
REF_SIG = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _headers(secret, body, *, wid="msg_1", ts=NOW, content_type="application/json"):
    return {"Webhook-Id": wid, "Webhook-Timestamp": str(ts), "Webhook-Signature": sign_standard_webhook(secret, webhook_id=wid, timestamp=ts, body=body),
            "Content-Type": content_type}


# --- verification ----------------------------------------------------------------------------------


def test_reference_vector():
    assert sign_standard_webhook(REF_SECRET, webhook_id=REF_ID, timestamp=REF_TS, body=REF_BODY) == REF_SIG
    headers = {"webhook-id": REF_ID, "webhook-timestamp": str(REF_TS), "webhook-signature": REF_SIG}
    assert verify_standard_webhook(REF_SECRET, headers, REF_BODY, REF_TS + 10).ok


def test_valid_tampered_stale_and_malformed():
    secret = new_webhook_secret()
    body = b'{"event": "push", "n": 1}'
    headers = _headers(secret, body)
    ok = verify_standard_webhook(secret, headers, body, NOW + 5)
    assert ok.ok and ok.webhook_id == "msg_1" and ok.timestamp == NOW
    assert not verify_standard_webhook(secret, headers, body + b" ", NOW).ok  # body tampered
    assert verify_standard_webhook(secret, headers, body.replace(b"1", b"2"), NOW).reason == "bad_signature"
    assert verify_standard_webhook(new_webhook_secret(), headers, body, NOW).reason == "bad_signature"  # wrong secret
    assert verify_standard_webhook(secret, {**headers, "Webhook-Id": "msg_2"}, body, NOW).reason == "bad_signature"  # id is signed
    assert verify_standard_webhook(secret, headers, body, NOW + 301).reason == "stale"
    assert verify_standard_webhook(secret, headers, body, NOW - 301).reason == "stale"  # from the future
    assert verify_standard_webhook(secret, headers, body, NOW + 301, tolerance_s=600).ok
    assert verify_standard_webhook(secret, {k: v for k, v in headers.items() if k != "Webhook-Signature"}, body, NOW).reason == "missing_headers"
    assert verify_standard_webhook(secret, {**headers, "Webhook-Timestamp": "soon"}, body, NOW).reason == "bad_timestamp"
    # Several signatures (key rotation): one valid entry suffices; other versions are ignored.
    multi = {**headers, "Webhook-Signature": "v1a,abc v1,AAAA " + headers["Webhook-Signature"]}
    assert verify_standard_webhook(secret, multi, body, NOW).ok


def test_body_mapping():
    assert webhook_event_data("m", b'{"a": [1]}', "application/json; charset=utf-8") == {"webhook_id": "m", "content_type": "application/json", "body": {"a": [1]}}
    assert webhook_event_data("m", b"[1, 2]", "application/cloudevents+json")["body"] == [1, 2]
    assert webhook_event_data("m", b"hello", "text/plain") == {"webhook_id": "m", "content_type": "text/plain", "body_text": "hello"}
    with pytest.raises(ValueError):
        webhook_event_data("m", b"{nope", "application/json")
    with pytest.raises(ValueError):
        webhook_event_data("m", b"\xff\xfe", "application/octet-stream")


# --- connections and append -------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return TriggerConnectionStore(tmp_path / "plane", sealing_key_file=tmp_path / "data" / "secrets" / "sealing.key")


def test_connection_generates_a_sealed_secret(store):
    conn = store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={"subject_path": "repository.name"})
    assert conn.config == {"subject_path": "repository.name", "tolerance_s": 300}
    secret = store.secrets(conn.connection_id, principal=ALICE)["secret"]
    assert secret.startswith("whsec_") and secret not in (store.base_dir / f"{conn.connection_id}.json").read_text()
    store.update(conn.connection_id, principal=ALICE, config={"tolerance_s": 600})
    assert store.secrets(conn.connection_id, principal=ALICE)["secret"] == secret  # kept across updates
    own = store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={"secret": REF_SECRET})
    assert store.secrets(own.connection_id, principal=ALICE) == {"secret": REF_SECRET}
    with pytest.raises(TriggerConnectionError) as exc:
        store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={"secret": "whsec_short"})
    assert exc.value.field == "config.secret"


def test_append_once_and_refuse_replays_stale_and_tampered(store):
    inbox = InMemoryEventInbox()
    conn = store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={"subject_path": "repository.name"})
    secret = store.secrets(conn.connection_id, principal=ALICE)["secret"]
    body = json.dumps({"repository": {"name": "abstractruntime"}, "ref": "main"}).encode()
    out = append_webhook_event(inbox, conn, body, _headers(secret, body), store=store, now=NOW + 1)
    assert out["status"] == "appended" and out["http_status"] == 202
    (event,) = inbox.read(stream=conn.stream)
    payload = event["payload"]
    assert payload["data"] == {"webhook_id": "msg_1", "content_type": "application/json",
                               "body": {"repository": {"name": "abstractruntime"}, "ref": "main"}}
    assert (payload["subject"], payload["dedupekey"], payload["abstracttrust"], payload["time"]) == (
        "abstractruntime", "msg_1", "untrusted", "2026-01-01T00:00:00+00:00")
    replay = append_webhook_event(inbox, conn, body, _headers(secret, body), store=store, now=NOW + 2)
    assert (replay["status"], replay["http_status"], replay["reason"]) == ("duplicate", 409, "replayed_id")
    stale = append_webhook_event(inbox, conn, body, _headers(secret, body, wid="msg_2"), store=store, now=NOW + 3600)
    assert (stale["status"], stale["http_status"], stale["reason"]) == ("refused", 401, "stale")
    tampered = append_webhook_event(inbox, conn, body + b"x", _headers(secret, body, wid="msg_3"), store=store, now=NOW)
    assert tampered["reason"] == "bad_signature"
    assert store.get(conn.connection_id).last_error["code"] == "webhook_bad_signature"  # recorded
    big = b"{" + b" " * 2048 + b"}"
    assert append_webhook_event(inbox, conn, big, _headers(secret, big, wid="m4"), secret=secret, now=NOW, max_body_bytes=1024)["http_status"] == 413
    binary = b"\xff\x00"
    assert append_webhook_event(inbox, conn, binary, _headers(secret, binary, wid="m5", content_type="application/octet-stream"),
                                secret=secret, now=NOW)["http_status"] == 415
    assert len(inbox.read(stream=conn.stream)) == 1  # only the one valid call
    store.update(conn.connection_id, principal=ALICE, status="disabled")
    disabled = store.get(conn.connection_id)
    assert append_webhook_event(inbox, disabled, body, _headers(secret, body, wid="m6"), secret=secret, now=NOW)["reason"] == "connection_disabled"


def test_a_signed_call_starts_one_occurrence_with_the_body(store, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    registry = WorkflowRegistry()
    for spec in TARGETS.values():
        registry.register(spec)
    register_controller_bundle(registry)
    runtime = Runtime(run_store=JsonFileRunStore(tmp_path / "runs"), ledger_store=JsonlLedgerStore(tmp_path / "ledger"),
                      workflow_registry=registry, artifact_store=FileArtifactStore(tmp_path / "artifacts"))
    runtime.set_event_inbox(InMemoryEventInbox())
    runtime.set_trigger_connection_store(store)
    conn = store.create(principal=ALICE, source_id="webhook.received", source_version=1, config={})
    secret = store.secrets(conn.connection_id, principal=ALICE)["secret"]
    req = request(trigger={"source_id": "webhook.received", "source_version": 1, "config": {"connection_id": conn.connection_id}},
                  input_data={"prompt": "Handle the call"})
    req.update({"tenant": "t", "user": "alice"})
    automation_id, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, automation_id)
    clock.set("2026-01-01T00:00:10+00:00")
    body = b'{"order": 42}'
    out = append_webhook_event(runtime.event_inbox, conn, body, _headers(secret, body, wid="evt_42"), store=store, now=NOW + 10)
    assert out["status"] == "appended"
    assert wake_connection_automations(runtime, conn.connection_id) == [automation_id]
    drive(runtime, automation_id)
    (kid,) = children(runtime, automation_id)
    event = kid.vars["event"]
    assert event["type"] == "abstract.webhook.received.v1" and event["subject"] == "evt_42"
    assert event["connectionid"] == conn.connection_id and event["abstracttrust"] == "untrusted"
    assert event["data"]["events"][0]["data"]["body"] == {"order": 42}
    assert '"order": 42' in kid.vars["prompt"]
