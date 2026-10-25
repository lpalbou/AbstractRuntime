"""Trigger connections (framework backlog 1040): per-user instances of a source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automation_harness import TARGETS, Clock, children, drive, request
from abstractruntime import Runtime
from abstractruntime.automations import apply_automation_command, create_automation, register_controller_bundle
from abstractruntime.automations.models import AutomationError
from abstractruntime.email.inbox import InMemoryEventInbox, JsonFileEventInbox
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.triggers import registry as registry_mod
from abstractruntime.triggers import (
    ConnectionEventAdapter,
    TriggerConfigError,
    TriggerConnectionError,
    TriggerConnectionStore,
    connection_principal,
    connection_stream,
    test_fire,
    trigger_sources,
    validate_binding_connection,
    wake_connection_automations,
)
from abstractruntime.triggers.connections import append_connection_event

ALICE = connection_principal("t", "alice")
BOB = connection_principal("t", "bob")
T0 = "2026-01-01T00:00:00+00:00"


class ProbeAdapter(ConnectionEventAdapter):
    """A connection-bound source for the tests (one plain field, one secret field)."""

    descriptor = {
        "id": "probe.event",
        "version": 1,
        "label": "Probe",
        "config_schema": ConnectionEventAdapter.binding_config_schema(),
        "event_schema": ConnectionEventAdapter.batch_event_schema(),
        "capabilities": {"kind": "event", "inbox": "connection", "content_trust": "untrusted"},
        "needs_connection": True,
        "connection_provider": "trigger_connections",
        "connection_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "token": {"type": "string", "x-secret": True}},
        },
    }

    def validate_connection(self, config, *, now):
        unknown = sorted(k for k in config if k not in ("name", "token"))
        if unknown:
            raise TriggerConfigError(f"unknown field(s) {unknown}", field=f"config.{unknown[0]}")
        if not isinstance(config.get("name"), str) or not config["name"].strip():
            raise TriggerConfigError("name is required", field="config.name")
        return {"name": config["name"].strip(), **({"token": config["token"]} if "token" in config else {})}

    def connection_secrets(self, config):
        return {"generated": "g-secret"}


class OtherProbe(ProbeAdapter):
    descriptor = {**ProbeAdapter.descriptor, "id": "probe.other"}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
def probe(monkeypatch):
    reg = registry_mod._REGISTRY
    reg.rows()  # discover the built-ins first
    monkeypatch.setitem(reg._adapters, ("probe.event", 1), ProbeAdapter())
    monkeypatch.setitem(reg._adapters, ("probe.other", 1), OtherProbe())
    return reg


@pytest.fixture
def store(tmp_path, probe):
    return TriggerConnectionStore(tmp_path / "plane", sealing_key_file=tmp_path / "data" / "secrets" / "sealing.key")


def _create(store, principal=ALICE, source="probe.event", **cfg):
    return store.create(principal=principal, source_id=source, source_version=1, config={"name": "n1", **cfg})


# --- the store ----------------------------------------------------------------------------------


def test_crud_is_scoped_to_the_principal(store):
    a = _create(store)
    b = _create(store, principal=BOB)
    assert [c.connection_id for c in store.list(principal=ALICE)] == [a.connection_id]
    assert [c.connection_id for c in store.list(principal=BOB)] == [b.connection_id]
    assert {c.connection_id for c in store.list()} == {a.connection_id, b.connection_id}  # admin view
    assert store.get(a.connection_id, principal=BOB) is None  # foreign reads like unknown
    with pytest.raises(TriggerConnectionError) as exc:
        store.update(a.connection_id, principal=BOB, label="mine now")
    assert exc.value.reason_code == "connection_not_found"
    assert store.delete(a.connection_id, principal=BOB) is False
    updated = store.update(a.connection_id, principal=ALICE, config={"name": "renamed"}, label="Feed A", status="disabled")
    assert (updated.config, updated.label, updated.status) == ({"name": "renamed"}, "Feed A", "disabled")
    assert store.get(a.connection_id, principal=ALICE).config == {"name": "renamed"}
    assert store.delete(a.connection_id, principal=ALICE) is True
    assert store.get(a.connection_id) is None
    assert store.get(b.connection_id, principal=BOB) is not None


def test_secrets_are_sealed_never_in_the_json(store, tmp_path):
    conn = _create(store, token="tok-123")
    assert conn.config == {"name": "n1"}
    assert conn.secret_fields == ["generated", "token"]
    raw = (store.base_dir / f"{conn.connection_id}.json").read_text()
    assert "tok-123" not in raw and "g-secret" not in raw
    sealed = store.secrets_dir / conn.connection_id / "secret.enc"
    assert sealed.is_file() and "tok-123" not in sealed.read_text()
    assert store.secrets(conn.connection_id, principal=ALICE) == {"token": "tok-123", "generated": "g-secret"}
    with pytest.raises(TriggerConnectionError):
        store.secrets(conn.connection_id, principal=BOB)
    # An update that leaves the secret out keeps it; one that sends it replaces it.
    store.update(conn.connection_id, principal=ALICE, config={"name": "n2"})
    assert store.secrets(conn.connection_id, principal=ALICE)["token"] == "tok-123"
    store.update(conn.connection_id, principal=ALICE, config={"name": "n2", "token": "tok-456"})
    assert store.secrets(conn.connection_id, principal=ALICE) == {"token": "tok-456", "generated": "g-secret"}
    assert store.delete(conn.connection_id, principal=ALICE)
    assert not sealed.exists()


def test_create_validates_config_and_source(store):
    with pytest.raises(TriggerConnectionError) as exc:
        store.create(principal=ALICE, source_id="probe.event", source_version=1, config={"bogus": 1})
    assert exc.value.reason_code == "invalid_connection" and exc.value.field == "config.bogus"
    with pytest.raises(TriggerConnectionError) as exc:
        store.create(principal=ALICE, source_id="schedule", source_version=1, config={})
    assert exc.value.reason_code == "invalid_connection"
    with pytest.raises(TriggerConnectionError) as exc:
        store.create(principal=ALICE, source_id="email.received", source_version=1, config={})
    assert "mail accounts" in str(exc.value)
    with pytest.raises(TriggerConnectionError) as exc:
        store.create(principal=ALICE, source_id="nope", source_version=1, config={})
    assert exc.value.reason_code == "unknown_trigger_source"


# --- binding validation ---------------------------------------------------------------------------


def _binding(cid, source="probe.event"):
    return {"source_id": source, "source_version": 1, "config": {"connection_id": cid}}


def test_binding_validation(store):
    mine = _create(store)
    other_source = _create(store, source="probe.other")
    bobs = _create(store, principal=BOB)
    assert validate_binding_connection(ALICE, _binding(mine.connection_id), store).connection_id == mine.connection_id
    cases = [
        (_binding("tc_unknown"), "connection_not_found"),
        (_binding(bobs.connection_id), "connection_not_found"),
        (_binding(other_source.connection_id), "connection_source_mismatch"),
        ({"source_id": "probe.event", "source_version": 1, "config": {}}, "connection_required"),
    ]
    for binding, code in cases:
        with pytest.raises(TriggerConnectionError) as exc:
            validate_binding_connection(ALICE, binding, store)
        assert exc.value.reason_code == code, binding
        assert exc.value.field == "trigger.config.connection_id"
    with pytest.raises(TriggerConnectionError) as exc:
        validate_binding_connection(ALICE, _binding(mine.connection_id), None)
    assert exc.value.reason_code == "connection_required"
    store.update(mine.connection_id, principal=ALICE, status="disabled")
    with pytest.raises(TriggerConnectionError) as exc:
        validate_binding_connection(ALICE, _binding(mine.connection_id), store)
    assert exc.value.reason_code == "connection_disabled"
    # A source without connections passes untouched.
    assert validate_binding_connection(ALICE, {"source_id": "schedule", "source_version": 1, "config": {}}, store) is None


def test_builtin_descriptors_declare_connections():
    by_ref = {(r["descriptor"]["id"], r["descriptor"]["version"]): r["descriptor"] for r in trigger_sources() if r["available"]}
    for ref in (("schedule", 1), ("schedule", 2), ("manual", 1)):
        assert by_ref[ref]["needs_connection"] is False
    email = by_ref[("email.received", 1)]
    assert email["needs_connection"] is True and email["connection_provider"] == "email_accounts"


def test_registry_refuses_a_connection_source_without_validate_connection():
    class Broken(ProbeAdapter):
        validate_connection = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="validate_connection"):
        registry_mod._check_adapter("probe.event", Broken())
    registry_mod._check_adapter("probe.event", ProbeAdapter())  # the complete one passes


# --- streams and test-fire ------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["memory", "json"])
def test_test_fire_appends_to_the_connection_stream(store, tmp_path, kind):
    inbox = InMemoryEventInbox() if kind == "memory" else JsonFileEventInbox(tmp_path / "inbox")
    conn = _create(store)
    other = _create(store)
    result = test_fire(store, inbox, conn.connection_id, principal=ALICE, now=T0)
    assert not result.duplicate
    events = inbox.read(stream=connection_stream(conn.connection_id), after_seq=0)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["data"] == {"test": True}
    assert payload["connectionid"] == conn.connection_id and payload["type"] == "abstract.probe.event.v1"
    assert inbox.read(stream=connection_stream(other.connection_id)) == []
    assert store.get(conn.connection_id).last_event_at == T0
    with pytest.raises(TriggerConnectionError):
        test_fire(store, inbox, conn.connection_id, principal=BOB)


def test_append_dedupes_on_dedupekey(store):
    inbox = InMemoryEventInbox()
    conn = _create(store)
    first = append_connection_event(inbox, conn, event_id="a", data={"n": 1}, dedupekey="item-1")
    assert first is not None and not first.duplicate
    assert append_connection_event(inbox, conn, event_id="b", data={"n": 1}, dedupekey="item-1") is None
    assert len(inbox.read(stream=conn.stream)) == 1


# --- automations bound to a connection ------------------------------------------------------------


def _runtime(tmp_path, store):
    registry = WorkflowRegistry()
    for spec in TARGETS.values():
        registry.register(spec)
    register_controller_bundle(registry)
    rt = Runtime(
        run_store=JsonFileRunStore(tmp_path / "runs"),
        ledger_store=JsonlLedgerStore(tmp_path / "ledger"),
        workflow_registry=registry,
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
    )
    rt.set_event_inbox(InMemoryEventInbox())
    rt.set_trigger_connection_store(store)
    return rt


def _probe_request(cid, **kw):
    req = request(trigger={"source_id": "probe.event", "source_version": 1, "config": {"connection_id": cid}}, **kw)
    req.update({"tenant": "t", "user": "alice"})
    return req


def test_create_automation_refuses_a_foreign_or_unknown_connection(store, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rt = _runtime(tmp_path, store)
    bobs = _create(store, principal=BOB)
    with pytest.raises(AutomationError) as exc:
        create_automation(rt, _probe_request(bobs.connection_id), now=clock.now)
    assert exc.value.reason_code == "connection_not_found"
    with pytest.raises(AutomationError) as exc:
        create_automation(rt, _probe_request("tc_nope"), now=clock.now)
    assert exc.value.reason_code == "connection_not_found"
    rt.set_trigger_connection_store(None)
    mine = _create(store)
    with pytest.raises(AutomationError) as exc:
        create_automation(rt, _probe_request(mine.connection_id), now=clock.now)
    assert exc.value.reason_code == "connection_required"


def test_revise_onto_a_foreign_connection_is_refused(store, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rt = _runtime(tmp_path, store)
    mine, mine2, bobs = _create(store), _create(store), _create(store, principal=BOB)
    automation_id, _ = create_automation(rt, _probe_request(mine.connection_id), now=clock.now)
    drive(rt, automation_id)
    trig = lambda cid: {"source_id": "probe.event", "source_version": 1, "config": {"connection_id": cid}}  # noqa: E731
    out = apply_automation_command(rt, automation_id=automation_id, command_id="c1", type="automation.revise",
                                   payload={"changes": {"trigger": trig(bobs.connection_id)}})
    assert out["status"] == "rejected" and out["error"]["reason_code"] == "connection_not_found"
    out = apply_automation_command(rt, automation_id=automation_id, command_id="c2", type="automation.revise",
                                   payload={"changes": {"trigger": trig(mine2.connection_id)}})
    assert out["status"] == "applied", out


def test_a_test_fire_starts_an_occurrence_of_the_bound_automation_only(store, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rt = _runtime(tmp_path, store)
    conn, other = _create(store), _create(store)
    automation_id, _ = create_automation(rt, _probe_request(conn.connection_id), now=clock.now)
    drive(rt, automation_id)
    assert children(rt, automation_id) == []
    clock.set("2026-01-01T00:05:00+00:00")
    # An event on another connection never reaches this automation.
    test_fire(store, rt.event_inbox, other.connection_id, now=clock.now)
    wake_connection_automations(rt, other.connection_id)
    drive(rt, automation_id)
    assert children(rt, automation_id) == []
    test_fire(store, rt.event_inbox, conn.connection_id, principal=ALICE, now=clock.now)
    assert wake_connection_automations(rt, conn.connection_id) == [automation_id]
    drive(rt, automation_id)
    kids = children(rt, automation_id)
    assert len(kids) == 1
    envelope = kids[0].vars["_meta"]["occurrence"]["trigger_envelope"]
    assert envelope["source_id"] == "probe.event"
    assert envelope["payload"]["count"] == 1
    assert envelope["payload"]["events"][0]["data"] == {"test": True}
    # A second wake with nothing new admits nothing.
    wake_connection_automations(rt, conn.connection_id)
    drive(rt, automation_id)
    assert len(children(rt, automation_id)) == 1


def test_adapter_admits_only_its_own_stream_in_one_coalesced_batch():
    adapter = ProbeAdapter()
    config = adapter.validate({"connection_id": "tc_a", "max_batch": 2}, now=T0)
    binding = {"binding_id": "b1", "source_id": "probe.event", "source_version": 1, "config": config}
    state = adapter.initial_state(config)
    later = "2026-01-01T00:01:00+00:00"
    events = [
        {"seq": 1, "event_id": "e1", "stream": "trigger:tc_a", "appended_at": later, "payload": {"data": {"n": 1}}},
        {"seq": 2, "event_id": "e2", "stream": "trigger:tc_b", "appended_at": later, "payload": {"data": {"n": 2}}},
        {"seq": 3, "event_id": "e3", "stream": "trigger:tc_a", "appended_at": later, "payload": {"data": {"n": 3}}},
        {"seq": 4, "event_id": "e4", "stream": "trigger:tc_a", "appended_at": later, "payload": {"data": {"n": 4}}},
    ]
    first = adapter.admit(binding, state=state, now=later, events=events)
    assert first["payload"]["event_ids"] == ["e1", "e3"]  # tc_b's event never; max_batch 2
    second = adapter.admit(binding, state=first["state"], now=later, events=events)
    assert second["payload"]["event_ids"] == ["e4"]
    assert adapter.admit(binding, state=second["state"], now=later, events=events) is None
