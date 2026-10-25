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


# --- AbstractCore's hermetic IMAP server (TLS, real protocol), read through the real client ------

ALICE_ADDR, ALICE_PW = "alice@example.test", "alice-pw"


@pytest.fixture(scope="module")
def ca(tmp_path_factory):
    from abstractcore.testing.mailserver import TestCA

    return TestCA.create(tmp_path_factory.mktemp("ca"))


@pytest.fixture
def imap(ca):
    from abstractcore.testing.mailserver import FakeImapServer

    server = FakeImapServer(ca, users={ALICE_ADDR: ALICE_PW}, security="ssl")
    yield server
    server.close()


def _context(tmp_path, ca, imap):
    from abstractcore.comms.email import EmailAccount, EmailContext, EmailSecret, ImapSettings, RecipientPolicy
    from abstractcore.comms.email.limits import SendLimits, SendRateLimiter

    account = EmailAccount.build(address=ALICE_ADDR, imap=ImapSettings.build("localhost", port=imap.port, security="ssl",
                                                                              ca_file=str(ca.ca_pem)))
    limits = SendLimits()
    return EmailContext(account=account, secret=EmailSecret(ALICE_PW), policy=RecipientPolicy.build("allowlist", [ALICE_ADDR]),
                        limits=limits, limiter=SendRateLimiter(tmp_path / "sends.json", limits), registered_address=ALICE_ADDR)


def _fill(imap, n, *, big_uid=None):
    from abstractcore.testing.mailserver import build_message

    for i in range(1, n + 1):
        text = ("long line " * 30_000) if i == big_uid else f"body of mail {i}"
        attachments = [("report.pdf", "application/pdf", b"%PDF-1.4 " + b"x" * 5000)] if i % 10 == 0 else []
        frm = "boss@x.test" if i % 2 else "bob@x.test"
        imap.add_message("INBOX", build_message(from_=frm, to=ALICE_ADDR, subject=f"mail {i}", text=text,
                                                html=f"<p>mail {i}</p>" if i % 3 == 0 else "", attachments=attachments))


def test_email_sample_reads_100_messages_on_one_connection_in_a_few_batched_fetches(env, tmp_path, ca, imap):
    import time as _time

    runtime, _, _ = env
    runtime.set_email_binding({"account_ref": "t:alice:default", "address": ALICE_ADDR})
    _fill(imap, 130, big_uid=125)
    head_before = runtime.event_inbox.head_seq()
    imap.fetches.clear()
    imap.logins.clear()
    started = _time.monotonic()
    sample = sample_occurrence_inputs(runtime, "email.received", 1, {}, count=100, email_context=_context(tmp_path, ca, imap), now=NOW)
    elapsed = _time.monotonic() - started
    assert elapsed < 5.0, elapsed
    assert len(imap.logins) == 1  # ONE connection
    # listing + headers + one per distinct text section (4 kinds of structure here): constant, not per message
    assert len(imap.fetches) <= 8, imap.fetches
    assert all("BODY[" not in f.upper().replace("BODY.PEEK[", "") for f in imap.fetches)  # PEEK only
    assert not any("\\Seen" in m.flags for m in imap.folders["INBOX"].messages)  # nothing marked read
    assert runtime.event_inbox.head_seq() == head_before
    msgs = resolved_sample_inputs(sample, artifact_store=runtime.artifact_store)["messages"]
    assert [m["subject"] for m in msgs[:3]] == ["mail 130", "mail 129", "mail 128"] and len(msgs) == 100
    assert msgs[0]["seen"] is False and msgs[0]["labels"] == [] and msgs[0]["id"].endswith(":130:INBOX")  # read live: flags known
    by = {m["subject"]: m for m in msgs}
    assert by["mail 129"]["body_text"] == "body of mail 129"
    assert by["mail 120"]["attachments"] and by["mail 120"]["attachments"][0]["name"] == "report.pdf"
    assert by["mail 125"]["body_text"].startswith("long line") and len(by["mail 125"]["body_text"]) >= 299_000  # whole, never cut
    assert sample["truncated"] is False and sample["inputs"]["event"]["abstracttest"] is True
    assert resolved_sample_inputs(sample, artifact_store=runtime.artifact_store)["mailbox"] == ALICE_ADDR


def test_a_body_over_the_cap_is_skipped_never_cut(tmp_path, ca, imap):
    from abstractruntime.automations.samples import batched_newest_messages

    _fill(imap, 3, big_uid=2)
    details, more, _next, hit = batched_newest_messages(_context(tmp_path, ca, imap).client(), folder="INBOX", count=3,
                                                        body_cap=10_000)
    by = {d.summary.subject: d for d in details}
    assert [d.summary.subject for d in details] == ["mail 3", "mail 2", "mail 1"] and not more and not hit
    assert by["mail 2"].skipped and by["mail 2"].text == ""  # over the cap: a typed record, no cut text
    assert by["mail 3"].text == "body of mail 3"


def test_email_sample_applies_the_trigger_filters(env, tmp_path, ca, imap):
    runtime, _, _ = env
    runtime.set_email_binding({"account_ref": "t:alice:default", "address": ALICE_ADDR})
    _fill(imap, 20)
    sample = sample_occurrence_inputs(runtime, "email.received", 1, {"filter": {"from_in": ["boss@x.test"]}},
                                      count=3, email_context=_context(tmp_path, ca, imap), now=NOW)
    msgs = resolved_sample_inputs(sample, artifact_store=runtime.artifact_store)["messages"]
    assert [m["subject"] for m in msgs] == ["mail 19", "mail 17", "mail 15"]


def test_email_sample_stops_at_its_budget(env, tmp_path, ca, imap):
    runtime, _, _ = env
    runtime.set_email_binding({"account_ref": "t:alice:default", "address": ALICE_ADDR})
    _fill(imap, 5)
    sample = sample_occurrence_inputs(runtime, "email.received", 1, {}, count=100, email_context=_context(tmp_path, ca, imap),
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
