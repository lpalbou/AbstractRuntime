"""Definition validation, identifiers, creation identity and the decision primitive (contract A/B)."""

from __future__ import annotations

import uuid

import pytest

from automation_harness import Clock, make_runtime, make_stores, request
from abstractruntime.automations import (
    AUTOMATION_NAMESPACE,
    AutomationError,
    create_automation,
    find_by_idempotency_key,
    get_automation,
    occurrence_run_id,
)
from abstractruntime.automations.ledger import automation_records, commit_decision, record_key
from abstractruntime.automations.models import build_definition, request_digest

NOW = "2026-01-01T00:00:00+00:00"
AID = str(uuid.uuid4())


def _build(**changes):
    req = request(request_id="r")
    for key, value in changes.items():
        if value is None:
            req.pop(key, None)
        else:
            req[key] = value
    return build_definition(req, automation_id=AID, now=NOW)


def test_definition_defaults():
    d = _build()
    assert d["schema_version"] == 2 and d["revision"] == 1
    assert d["controller"] == {"bundle_ref": "abstractframework.automation-controller@1.0.0", "flow_id": "controller"}
    assert d["context"] == {"mode": "independent", "growing": {}}
    assert d["policy"] == {"serial": True, "misfire": "coalesce", "failure": "continue",
                           "retry": {"max_attempts": 3, "backoff": {"initial": "30s", "factor": 2, "max": "10m"}},
                           "tool_approval": "auto", "email_allowed_recipients": ["self"]}
    assert d["notify"] == {"channels": ["console"]}
    assert d["trigger"]["binding_id"] == str(uuid.uuid5(uuid.UUID(AID), "binding:1"))
    assert d["trigger"]["config"] == {"start_at": NOW, "anchor": NOW, "every": "2m"}
    assert d["session_id"] == f"automation:{AID}" and d["archived_at"] is None and d["created_at"] == NOW


@pytest.mark.parametrize(
    "changes, reason, field",
    [
        ({"surprise": 1}, "invalid_definition", "surprise"),
        ({"title": ""}, "invalid_definition", "title"),
        ({"title": "x" * 121}, "invalid_definition", "title"),
        ({"target": {"workflow_id": "w", "bundle_ref": "b", "flow_id": "@default"}}, "invalid_definition", "target.flow_id"),
        ({"target": {"workflow_id": "w", "bundle_ref": "b", "flow_id": "f", "extra": 1}}, "invalid_definition", "target.extra"),
        ({"trigger": {"source_id": "cron", "source_version": 1, "config": {}}}, "unknown_trigger_source", "trigger.source_id"),
        ({"trigger": {"source_id": "schedule", "source_version": 1, "config": {"every": "1w"}}}, "invalid_definition", "trigger.config.every"),
        ({"trigger": {"source_id": "schedule", "source_version": 1, "config": {"every": "9999999d"}}}, "invalid_definition", "trigger.config.every"),
        ({"trigger": {"source_id": "schedule", "source_version": 1, "config": {"every": "99999999999d"}}}, "invalid_definition", "trigger.config.every"),
        ({"trigger": {"source_id": "schedule", "source_version": 1, "config": {"every": "367d"}}}, "invalid_definition", "trigger.config.every"),
        ({"trigger": {"source_id": "schedule", "source_version": 1, "config": {"every": "1m", "count": 1_000_001}}}, "invalid_definition", "trigger.config.count"),
        ({"policy": {"retry": {"backoff": {"max": "400d"}}}}, "invalid_definition", "policy.retry.backoff.max"),
        ({"context": {"mode": "growing", "growing": {"summary": {"enabled": True, "every_n": 2, "max_tokens": 9}}}},
         "unsupported_feature", "context.growing.summary"),
        ({"context": {"mode": "forking"}}, "invalid_definition", "context.mode"),
        ({"policy": {"retry": {"max_attempts": 11}}}, "invalid_definition", "policy.retry.max_attempts"),
        ({"policy": {"serial": False}}, "unsupported_feature", "policy.serial"),
        ({"policy": {"tool_approval": "never"}}, "invalid_definition", "policy.tool_approval"),
        ({"policy": {"tool_approval": True}}, "invalid_definition", "policy.tool_approval"),
        ({"policy": {"retry": {"backoff": {"initial": "30"}}}}, "invalid_definition", "policy.retry.backoff.initial"),
        ({"workspace_root": "relative/dir"}, "invalid_definition", "workspace_root"),
        ({"workspace_root": None}, "invalid_definition", "workspace_root"),
    ],
)
def test_definition_rejects(changes, reason, field):
    with pytest.raises(AutomationError) as exc:
        _build(**changes)
    assert exc.value.reason_code == reason
    assert exc.value.field == field


def test_deterministic_ids():
    aid = str(uuid.uuid5(AUTOMATION_NAMESPACE, "t:u:req"))
    assert occurrence_run_id(aid, revision=2, index=7) == str(uuid.uuid5(uuid.UUID(aid), "2:7"))
    assert occurrence_run_id(aid, revision=2, index=7, attempt=3) == str(uuid.uuid5(uuid.UUID(aid), "2:7:a3"))
    assert occurrence_run_id(aid, revision=2, index=7, command_id="c9") == str(uuid.uuid5(uuid.UUID(aid), "manual:c9"))
    assert occurrence_run_id(aid, revision=2, index=7, attempt=2, command_id="c9") == str(uuid.uuid5(uuid.UUID(aid), "manual:c9:a2"))


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_create_replays_and_refuses_a_different_request(tmp_path, monkeypatch, kind):
    runtime = make_runtime(*make_stores(kind, tmp_path))
    req = request(request_id="same")
    req.update(tenant="acme", user="ana")
    aid, rev = create_automation(runtime, req, now=NOW)
    assert aid == str(uuid.uuid5(AUTOMATION_NAMESPACE, "acme:ana:same")) and rev == 1
    assert runtime.get_state(aid).vars["_meta"]["creation_digest"] == request_digest(req)
    # Same request later (created_at would differ): the same automation, untouched.
    assert runtime.get_state(aid).actor_id is None
    assert create_automation(runtime, req, now="2026-01-02T00:00:00+00:00") == (aid, 1)
    assert get_automation(runtime.run_store, aid)["definition"]["created_at"] == NOW
    assert [r["name"] for r in automation_records(runtime.ledger_store, aid)] == ["automation.created"]
    other = dict(req, title="Something else")
    with pytest.raises(AutomationError) as exc:
        create_automation(runtime, other, now=NOW)
    assert exc.value.reason_code == "identity_conflict"
    # Another user with the same request id is another automation.
    assert create_automation(runtime, dict(req, user="bob"), now=NOW)[0] != aid


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_commit_decision_checks_the_key_instead_of_assuming_uniqueness(tmp_path, monkeypatch, kind):
    clock = Clock(monkeypatch)
    runtime = make_runtime(*make_stores(kind, tmp_path))
    aid, _ = create_automation(runtime, request(), now=clock.now)
    run = runtime.get_state(aid)
    key = record_key("automation.command_result", aid, "k1")
    kwargs = dict(run_store=runtime.run_store, ledger_store=runtime.ledger_store, name="automation.command_result",
                  key=key, at=NOW, command_id="k1")
    first = commit_decision(run, fields={"status": "applied"}, delta={"state": {"paused": True}}, **kwargs)
    second = commit_decision(run, fields={"status": "applied"}, delta={"state": {"paused": False}}, **kwargs)
    assert second == first
    assert run.vars["_runtime"]["automation"]["paused"] is True
    assert run.vars["_runtime"]["automation"]["state_version"] == 1
    assert find_by_idempotency_key(runtime.ledger_store, aid, key)["idempotency_key"] == key
    assert find_by_idempotency_key(runtime.ledger_store, aid, key + "x") is None
    assert len(automation_records(runtime.ledger_store, aid, "automation.command_result")) == 1


def test_legacy_schedule_roots_are_projected_read_only():
    from abstractruntime import RunState, RunStatus
    from abstractruntime.automations import adopt_legacy_schedule_projection

    run = RunState(run_id="legacy-1", workflow_id="scheduled:abc", status=RunStatus.WAITING, current_node="wait",
                   vars={"_meta": {"schedule": {"kind": "scheduled_run", "target_workflow_id": "b@1:main",
                                                "target_bundle_ref": "b@1", "target_flow_id": "main",
                                                "start_at": NOW, "interval": "15m", "repeat_count": None,
                                                "repeat_until": None, "share_context": True}},
                         "_runtime": {"control": {"paused": True}}})
    before = repr(run)
    summary = adopt_legacy_schedule_projection(run)
    assert repr(run) == before  # nothing written
    assert summary["legacy"] is True and summary["revision"] is None and summary["automation_id"] == "legacy-1"
    assert summary["status"] == "paused" and summary["context_mode"] == "growing"
    assert summary["trigger"]["config"] == {"start_at": NOW, "every": "15m"}
    assert summary["target"] == {"workflow_id": "b@1:main", "bundle_ref": "b@1", "flow_id": "main"}
    with pytest.raises(AutomationError):
        adopt_legacy_schedule_projection(RunState(run_id="x", workflow_id="w", status=RunStatus.COMPLETED,
                                                  current_node="n", vars={}))


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_create_stamps_the_owner_in_the_same_step(tmp_path, kind):
    runtime = make_runtime(*make_stores(kind, tmp_path))
    aid, _ = create_automation(runtime, request(request_id="owned"), now=NOW, actor_id="tenant:ana")
    assert runtime.get_state(aid).actor_id == "tenant:ana"


def test_the_longest_accepted_interval_runs():
    from abstractruntime.triggers import ScheduleTriggerAdapter

    a = ScheduleTriggerAdapter()
    cfg = a.validate({"every": "366d", "count": 1_000_000}, now=NOW)
    binding = {"binding_id": "b", "source_id": "schedule", "source_version": 1, "config": cfg}
    state = a.admit(binding, state=a.initial_state(cfg), now=NOW)["state"]
    assert a.prepare(binding, state=state, now=NOW)["until"] == "2027-01-02T00:00:00+00:00"
