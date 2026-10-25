"""Trigger envelopes in CloudEvents 1.0 names + `input_data.event` (framework backlog 1039)."""

from __future__ import annotations

import pytest

from automation_harness import Clock, children, drive, make_runtime, make_stores, request
from abstractruntime.automations import create_automation
from abstractruntime.triggers import EmailReceivedTriggerAdapter, ManualTriggerAdapter, ScheduleV2TriggerAdapter
from abstractruntime.triggers.cloudevents import CLOUDEVENT_JSON_SCHEMA, from_cloudevent, to_cloudevent
from test_trigger_connections import ProbeAdapter

T0 = "2026-01-01T00:00:00+00:00"
ORIGINAL_KEYS = ("event_id", "source_id", "source_version", "fired_at", "payload", "binding_id")


def _validate(event):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(event, CLOUDEVENT_JSON_SCHEMA)


def _binding(source_id, version, config):
    return {"binding_id": "b-1", "source_id": source_id, "source_version": version, "config": config}


def test_every_builtin_source_populates_subject_dedupekey_and_trust():
    sched = ScheduleV2TriggerAdapter()
    env = sched.normalize(_binding("schedule", 2, {}), event_id="schedule@2:b-1:4", fired_at=T0, payload={"tick": 4, "scheduled_at": T0})
    assert (env["subject"], env["dedupekey"], env["abstracttrust"], env["type"]) == (
        "tick:4", "schedule@2:b-1:4", "trusted", "abstract.schedule.v2")
    man = ManualTriggerAdapter().normalize(_binding("schedule", 2, {}), event_id="manual:c1", fired_at=T0, payload={"command_id": "c1"})
    assert (man["subject"], man["dedupekey"], man["abstracttrust"]) == ("command:c1", "manual:c1", "trusted")
    email = EmailReceivedTriggerAdapter()
    one = email.normalize(_binding("email.received", 1, {"account": "self", "folder": "INBOX"}), event_id="email.received@1:b-1:d",
                          fired_at=T0, payload={"count": 1, "first_seq": 7, "last_seq": 7, "messages": [{"message_id": "<m1@x>"}]})
    assert (one["subject"], one["dedupekey"], one["abstracttrust"], one["sequence"]) == ("<m1@x>", "email.received@1:b-1:d", "untrusted", 7)
    assert one["source"] == "abstract://email/accounts/self/INBOX"
    batch = email.normalize(_binding("email.received", 1, {"account": "self"}), event_id="e", fired_at=T0,
                            payload={"count": 2, "first_seq": 3, "last_seq": 4, "messages": [{}, {}]})
    assert batch["subject"] == "batch:3-4"
    conn = ProbeAdapter().normalize(_binding("probe.event", 1, {"connection_id": "tc_a"}), event_id="p", fired_at=T0,
                                    payload={"count": 1, "first_seq": 9, "last_seq": 9, "connection_id": "tc_a",
                                             "content_trust": "untrusted", "events": [{"subject": "item-42"}]})
    assert (conn["subject"], conn["connectionid"], conn["source"], conn["sequence"], conn["abstracttrust"]) == (
        "item-42", "tc_a", "abstract://triggers/connections/tc_a", 9, "untrusted")
    for envelope in (env, man, one, batch, conn):
        assert all(k in envelope for k in ORIGINAL_KEYS)  # the old form stays for one release
        _validate(to_cloudevent(envelope))


def test_round_trip_and_pre_1039_envelopes():
    old = {"event_id": "schedule@1:b:0", "source_id": "schedule", "source_version": 1, "fired_at": T0,
           "payload": {"tick": 0, "scheduled_at": T0}, "binding_id": "b"}
    event = to_cloudevent(old)
    assert event == {
        "specversion": "1.0", "id": "schedule@1:b:0", "type": "abstract.schedule.v1",
        "source": "abstract://triggers/schedule/bindings/b", "time": T0, "datacontenttype": "application/json",
        "bindingid": "b", "data": {"tick": 0, "scheduled_at": T0},
    }
    _validate(event)
    back = from_cloudevent(event)
    assert {k: back[k] for k in ORIGINAL_KEYS} == old
    probe = ProbeAdapter().normalize(_binding("probe.event", 1, {"connection_id": "tc_a"}), event_id="p", fired_at=T0,
                                     payload={"count": 1, "connection_id": "tc_a", "events": [{}]})
    assert from_cloudevent(to_cloudevent(probe)) == probe
    with pytest.raises(ValueError):
        from_cloudevent({"specversion": "1.0", "id": "x", "source": "s", "type": "com.example.other"})


@pytest.fixture
def env(tmp_path, monkeypatch):
    run_store, ledger_store = make_stores("json", tmp_path)
    return make_runtime(run_store, ledger_store), Clock(monkeypatch)


def test_occurrence_receives_the_envelope_as_structured_input(env):
    runtime, clock = env
    req = request(trigger={"source_id": "schedule", "source_version": 2,
                           "config": {"kind": "every", "every": "5m", "start_at": T0}})
    automation_id, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, automation_id)
    (kid,) = children(runtime, automation_id)
    event = kid.vars["event"]
    _validate(event)
    envelope = kid.vars["_meta"]["occurrence"]["trigger_envelope"]
    assert all(k in envelope for k in ORIGINAL_KEYS)
    assert event["id"] == envelope["event_id"] and event["type"] == "abstract.schedule.v2"
    assert event["bindingid"] == envelope["binding_id"] and event["dedupekey"] == envelope["event_id"]
    assert event["abstracttrust"] == "trusted" and event["subject"] == "tick:0"
    assert event["data"]["series"] == {
        "rule": {"kind": "every", "every": "5m"}, "occurrence": 1, "scheduled_at": T0,
        "next_at": "2026-01-01T00:05:00+00:00", "coalesced": None,
    }
    assert kid.vars["prompt"].startswith("[Trigger schedule@2 · occurrence 1")  # the header is unchanged


def test_schedule_v1_stays_frozen_but_its_occurrences_get_the_event(env):
    """schedule@1's adapter answers byte for byte as released (R16.1 corpus); the controller adds the
    CloudEvents attributes to its envelope."""
    from abstractruntime.triggers import ScheduleTriggerAdapter

    v1 = ScheduleTriggerAdapter()
    env1 = v1.normalize(_binding("schedule", 1, {}), event_id="e", fired_at=T0, payload={"tick": 0, "scheduled_at": T0})
    assert set(env1) == set(ORIGINAL_KEYS)
    runtime, clock = env
    automation_id, _ = create_automation(runtime, request(), now=clock.now)  # the harness default: schedule@1
    drive(runtime, automation_id)
    (kid,) = children(runtime, automation_id)
    event = kid.vars["event"]
    _validate(event)
    assert (event["type"], event["subject"], event["abstracttrust"]) == ("abstract.schedule.v1", "tick:0", "trusted")
    assert event["dedupekey"] == kid.vars["_meta"]["occurrence"]["event_id"]
    assert "series" not in event["data"]
