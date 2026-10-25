"""Run origin (framework backlog 1038): `_meta.origin` recorded once at start."""

from __future__ import annotations

import pytest

from automation_harness import Clock, at, children, drive, make_runtime, make_stores, request
from abstractruntime.automations import apply_automation_command, create_automation
from abstractruntime.core.run_origin import origin_for, origin_of, stamp_origin

T0 = "2026-01-01T00:00:00+00:00"


def test_origin_for_builds_and_validates_the_shape():
    assert origin_for("person", actor="t:alice", client="console") == {
        "kind": "person", "actor": "t:alice", "client": "console", "ref": {},
    }
    trig = origin_for("trigger", actor="t:alice", client="automation", automation_id="a1", occurrence_index=3,
                      event_id="schedule@2:b:3", source="schedule@2")
    assert trig["ref"] == {"automation_id": "a1", "occurrence_index": 3, "event_id": "schedule@2:b:3", "source": "schedule@2"}
    assert origin_for("agent", origin_run_id="r1", session_id="s1")["ref"] == {"origin_run_id": "r1", "session_id": "s1"}
    assert origin_for("system", client="bridge:agora", bridge="agora")["kind"] == "system"
    for bad in (
        lambda: origin_for("robot"),
        lambda: origin_for("trigger"),  # names no automation
        lambda: origin_for("agent"),  # names no scheduling run
        lambda: origin_for("person", actor=""),
        lambda: origin_for("trigger", automation_id="a", occurrence_index=0),
    ):
        with pytest.raises(ValueError):
            bad()


def test_origin_of_and_stamp_once():
    assert origin_of({}) is None and origin_of({"_meta": {"origin": {"kind": "nope"}}}) is None
    vars_obj = {"_meta": {"x": 1}}
    stamp_origin(vars_obj, origin_for("person", actor="t:a"))
    stamp_origin(vars_obj, origin_for("system", bridge="b"))  # written once: the first stays
    assert origin_of(vars_obj)["kind"] == "person" and vars_obj["_meta"]["x"] == 1


@pytest.fixture
def env(tmp_path, monkeypatch):
    run_store, ledger_store = make_stores("json", tmp_path)
    return make_runtime(run_store, ledger_store), Clock(monkeypatch)


def test_scheduled_occurrence_records_a_trigger_origin(env):
    runtime, clock = env
    req = request(trigger={"source_id": "schedule", "source_version": 1, "config": {"start_at": T0, "every": "2m"}})
    req.update({"tenant": "t", "user": "alice"})
    automation_id, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, automation_id)
    kids = children(runtime, automation_id)
    assert len(kids) == 1
    origin = origin_of(kids[0].vars)
    assert origin == {
        "kind": "trigger",
        "actor": "t:alice",
        "client": "automation",
        "ref": {
            "automation_id": automation_id,
            "occurrence_index": 1,
            "event_id": kids[0].vars["_meta"]["occurrence"]["event_id"],
            "source": "schedule@1",
        },
    }
    assert "trigger_envelope" not in origin["ref"]  # a pointer, never a copy


def test_run_now_occurrence_records_a_person_origin(env):
    runtime, clock = env
    req = request(trigger={"source_id": "manual", "source_version": 1, "config": {}})
    req.update({"tenant": "t", "user": "bob"})
    automation_id, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, automation_id)
    out = apply_automation_command(runtime, automation_id=automation_id, command_id="go-1", type="automation.run_now")
    assert out["status"] == "applied"
    drive(runtime, automation_id)
    (kid,) = children(runtime, automation_id)
    origin = origin_of(kid.vars)
    assert origin["kind"] == "person" and origin["actor"] == "t:bob" and origin["client"] == "automation.run_now"
    assert origin["ref"]["source"] == "manual@1" and origin["ref"]["event_id"] == "manual:go-1"
