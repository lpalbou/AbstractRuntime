"""Trigger-source registry (automations contract C): discovery, built-ins, third parties."""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractruntime.triggers import registry as reg
from abstractruntime.triggers import (
    ManualTriggerAdapter,
    ScheduleTriggerAdapter,
    ScheduleV2TriggerAdapter,
    TriggerRegistryError,
    UnknownTriggerSource,
    get_trigger_adapter,
    reset_trigger_registry,
    trigger_sources,
)


class WebhookAdapter(ManualTriggerAdapter):
    """A well-formed third-party source (reuses manual's behaviour)."""

    descriptor = {**ManualTriggerAdapter.descriptor, "id": "webhook", "label": "Webhook",
                  "capabilities": {"kind": "event"}}


class MisnamedAdapter(ManualTriggerAdapter):
    descriptor = {**ManualTriggerAdapter.descriptor, "id": "not-the-entry-name"}


class HalfAdapter:
    descriptor = {**ManualTriggerAdapter.descriptor, "id": "half"}

    def validate(self, config, *, now):  # the other five methods are missing
        return {}


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_trigger_registry()
    yield
    reset_trigger_registry()


def _fake_entry_points(monkeypatch, pairs):
    monkeypatch.setattr(reg, "_entry_points", lambda: sorted(pairs))


def test_builtins_are_available_and_typed():
    listed = [(r["descriptor"]["id"], r["descriptor"]["version"]) for r in trigger_sources() if r["available"]]
    assert listed == [("schedule", 1), ("schedule", 2), ("manual", 1), ("email.received", 1)]
    rows = {(r["descriptor"]["id"], r["descriptor"]["version"]): r for r in trigger_sources() if r["available"]}
    assert rows[("schedule", 1)]["descriptor"]["capabilities"] == {"kind": "time"}
    assert rows[("schedule", 2)]["descriptor"]["capabilities"] == {"kind": "time"}
    assert isinstance(get_trigger_adapter("schedule", 2), ScheduleV2TriggerAdapter)
    assert rows[("manual", 1)]["descriptor"]["capabilities"] == {"kind": "manual"}
    assert isinstance(get_trigger_adapter("schedule", 1), ScheduleTriggerAdapter)
    assert isinstance(get_trigger_adapter("manual", 1), ManualTriggerAdapter)


def test_unknown_source_and_version_are_refused():
    with pytest.raises(UnknownTriggerSource) as exc:
        get_trigger_adapter("schedule", 3)
    assert exc.value.reason_code == "unknown_trigger_source"
    with pytest.raises(UnknownTriggerSource):
        get_trigger_adapter("cron", 1)


def test_third_party_entry_point_is_discovered_without_code_change(monkeypatch):
    mod = __name__
    _fake_entry_points(monkeypatch, [
        ("schedule", reg.BUILTIN_TRIGGER_SOURCES["schedule"]),
        ("manual", reg.BUILTIN_TRIGGER_SOURCES["manual"]),
        ("webhook", f"{mod}:WebhookAdapter"),
    ])
    ids = [r["descriptor"]["id"] for r in trigger_sources() if r["available"]]
    assert ids == ["schedule", "schedule", "manual", "email.received", "webhook"]
    assert isinstance(get_trigger_adapter("webhook", 1), WebhookAdapter)


def test_broken_third_party_sources_are_listed_unavailable(monkeypatch):
    mod = __name__
    _fake_entry_points(monkeypatch, [
        ("ghost", "abstractruntime_no_such_module:Adapter"),
        ("not-matching", f"{mod}:MisnamedAdapter"),
        ("half", f"{mod}:HalfAdapter"),
        ("schedule", f"{mod}:WebhookAdapter"),  # hijacking a built-in name
    ])
    rows = trigger_sources()
    unavailable = {r["name"]: r["unavailable_reason"] for r in rows if not r["available"]}
    assert set(unavailable) == {"ghost", "not-matching", "half", "schedule"}
    assert "ModuleNotFoundError" in unavailable["ghost"]
    assert "differs from descriptor.id" in unavailable["not-matching"]
    assert "lacks method" in unavailable["half"]
    assert "conflicts with the built-in" in unavailable["schedule"]
    # The built-ins still serve; the broken ones are never selectable.
    assert isinstance(get_trigger_adapter("schedule", 1), ScheduleTriggerAdapter)
    with pytest.raises(UnknownTriggerSource):
        get_trigger_adapter("half", 1)


def test_missing_builtin_fails_loud(monkeypatch):
    monkeypatch.setitem(reg.BUILTIN_TRIGGER_SOURCES, "schedule", "abstractruntime.triggers.gone:ScheduleTriggerAdapter")
    with pytest.raises(TriggerRegistryError, match="built-in trigger source 'schedule'"):
        trigger_sources()
    with pytest.raises(TriggerRegistryError):
        get_trigger_adapter("manual", 1)


def test_broken_builtin_descriptor_fails_loud(monkeypatch):
    monkeypatch.setitem(reg.BUILTIN_TRIGGER_SOURCES, "manual", f"{__name__}:HalfAdapter")
    with pytest.raises(TriggerRegistryError, match="'manual'"):
        trigger_sources()


def test_pyproject_declares_the_builtin_entry_points():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split(f'[project.entry-points."{reg.ENTRY_POINT_GROUP}"]', 1)[1].split("\n[", 1)[0]
    declared = {}
    for line in section.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, target = line.split("=", 1)
            declared[name.strip().strip('"')] = target.strip().strip('"')
    assert declared == reg.BUILTIN_TRIGGER_SOURCES
