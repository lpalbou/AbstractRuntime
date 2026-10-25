"""Wake the consumers of ONE inbox stream (framework backlog 1042/1044).

`stream_consumers(runtime, stream)` / `wake_stream_consumers(runtime, stream)` generalise
`wake_email_automations`: a feeder that appended to `trigger:<connection_id>` (a webhook or a
feed) or to one mailbox's `email:<tenant>:<user>:<account_id>:<folder>` wakes exactly the
automations whose binding reads that stream, and nobody else."""

from __future__ import annotations

import pytest

from automation_harness import Clock, drive
from email_wp2_support import _reset_core_resolver, ca  # noqa: F401 - fixtures
from test_email_inbox_and_trigger import REF, _email_automation, _plane

from abstractruntime.core.models import RunStatus
from abstractruntime.email import automation_reads_stream, stream_consumers, wake_stream_consumers
from abstractruntime.email.automations import binding_account_id, email_stream_account_id

EMAIL = {"source_id": "email.received", "source_version": 1}


@pytest.mark.parametrize(
    ("trigger", "stream", "self_ref", "reads"),
    [
        ({"source_id": "webhook.received", "source_version": 1, "config": {"connection_id": "tc_a"}}, "trigger:tc_a", None, True),
        ({"source_id": "webhook.received", "source_version": 1, "config": {"connection_id": "tc_a"}}, "trigger:tc_b", None, False),
        ({"source_id": "feed.item", "source_version": 1, "config": {"connection_id": "tc_a"}}, "email:t:u:default:INBOX", None, False),
        ({**EMAIL, "config": {}}, "email:t:u:default:INBOX", None, True),
        ({**EMAIL, "config": {"account": "self"}}, "email:t:u:work:INBOX", None, False),
        ({**EMAIL, "config": {"account": "self"}}, "email:t:u:work:INBOX", "t:u:work", True),
        ({**EMAIL, "config": {"account": "work"}}, "email:t:u:work:Archive/2026", "t:u:default", True),
        ({**EMAIL, "config": {"account": "work"}}, "email:t:u:default:INBOX", "t:u:default", False),
        ({**EMAIL, "config": {"account": "work"}}, "trigger:work", None, False),
        ({"source_id": "schedule", "source_version": 2, "config": {}}, "email:t:u:default:INBOX", None, False),
    ],
)
def test_which_binding_reads_which_stream(trigger, stream, self_ref, reads):
    assert automation_reads_stream(trigger, stream, self_account_ref=self_ref) is reads


def test_stream_account_id_parsing():
    assert email_stream_account_id("email:t:u:work:INBOX") == "work"
    assert email_stream_account_id("email:t:u:work:a:b") == "work"
    assert email_stream_account_id("trigger:tc_x") is None and email_stream_account_id("email:t:u") is None
    assert binding_account_id({"config": {"account": "self"}}) == "default" == binding_account_id({"config": {}})


def test_wake_stream_consumers_wakes_only_the_readers_of_that_stream(tmp_path, monkeypatch):
    rt, _feeder, _ctx = _plane(tmp_path, None)
    clock = Clock(monkeypatch)
    aid = _email_automation(rt, clock, config={"uses_model": True})
    state = drive(rt, aid)
    assert state.status == RunStatus.WAITING and state.waiting.until is None  # idle until an event

    own = f"email:{REF}:INBOX"
    other = "email:t:alice:other:INBOX"
    assert stream_consumers(rt, own) == [aid]
    assert stream_consumers(rt, other) == [] and stream_consumers(rt, "trigger:tc_nobody") == []
    assert wake_stream_consumers(rt, other) == []
    assert rt.run_store.load(aid).status == RunStatus.WAITING
    assert wake_stream_consumers(rt, own) == [aid]


def test_an_automation_on_a_further_mailbox_is_bound_to_that_mailbox():
    from abstractruntime.email import EmailBinding, account_id_of_ref, account_ref_with_id, email_binding_for_trigger

    primary = EmailBinding("t:alice:default", "alice@example.test")
    assert account_id_of_ref("t:alice:work") == "work" and account_id_of_ref("nope") is None
    assert account_ref_with_id("t:alice:default", "work") == "t:alice:work"
    for config in ({}, {"account": "self"}, {"account": "default"}):
        assert email_binding_for_trigger(primary, {**EMAIL, "config": config}) is primary
    work = email_binding_for_trigger(primary, {**EMAIL, "config": {"account": "work"}})
    assert work.account_ref == "t:alice:work"
    assert email_binding_for_trigger(primary, {"source_id": "schedule", "source_version": 2, "config": {"account": "work"}}) is primary
    assert email_binding_for_trigger(None, {**EMAIL, "config": {"account": "work"}}) is None
