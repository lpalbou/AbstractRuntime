"""schedule@1 anchor math (automations contract C / C2).

Every expected value is a hand-computed instant on a fixed UTC grid; nothing
reads the wall clock.
"""

from __future__ import annotations

import pytest

from abstractruntime.triggers import ScheduleTriggerAdapter, TriggerConfigError

A = ScheduleTriggerAdapter()
T0 = "2026-01-01T00:00:00+00:00"


def _binding(config, binding_id="b-1"):
    cfg = A.validate(config, now=T0)
    return {"binding_id": binding_id, "source_id": "schedule", "source_version": 1, "config": cfg}


def _state(binding):
    return A.initial_state(binding["config"])


def test_validate_fills_defaults_and_normalizes_utc():
    cfg = A.validate({"every": "5m"}, now="2026-01-01T02:00:00+02:00")
    assert cfg == {"start_at": T0, "anchor": T0, "every": "5m"}
    cfg = A.validate({"start_at": "2026-01-01T10:00:00Z", "every": "1h", "count": 3}, now=T0)
    assert cfg["start_at"] == "2026-01-01T10:00:00+00:00" and cfg["count"] == 3


@pytest.mark.parametrize(
    "config, field, reason",
    [
        ({"every": "5"}, "config.every", "invalid_definition"),
        ({"every": "0m"}, "config.every", "invalid_definition"),
        ({"every": "1.5h"}, "config.every", "invalid_definition"),
        ({"every": "2w"}, "config.every", "invalid_definition"),
        ({"every": "5m", "count": 0}, "config.count", "invalid_definition"),
        ({"every": "5m", "count": True}, "config.count", "invalid_definition"),
        ({"count": 2}, "config.count", "invalid_definition"),
        ({"every": "5m", "until": T0}, "config.until", "invalid_definition"),
        ({"every": "5m", "anchor": "2026-01-01T00:01:00Z"}, "config.anchor", "unsupported_feature"),
        ({"start_at": "2026-01-01T00:00:00"}, "config.start_at", "invalid_definition"),
        ({"every": "5m", "tz": "Europe/Paris"}, "config.tz", "invalid_definition"),
    ],
)
def test_validate_rejects(config, field, reason):
    with pytest.raises(TriggerConfigError) as exc:
        A.validate(config, now=T0)
    assert exc.value.field == field
    assert exc.value.reason_code == reason


def test_grid_is_anchored_not_drifting():
    b = _binding({"every": "5m"})
    st = _state(b)
    assert A.prepare(b, state=st, now=T0) == {"kind": "until", "until": T0}
    # Admission 7s late: the next tick stays on the grid (00:05), not now+5m.
    adm = A.admit(b, state=st, now="2026-01-01T00:00:07+00:00")
    assert adm["event_id"] == "schedule@1:b-1:0"
    assert adm["fired_at"] == T0
    assert "coalesced" not in adm
    st = adm["state"]
    assert st == {"anchor": T0, "tick": 1, "scheduled_count": 1, "exhausted": False}
    assert A.prepare(b, state=st, now="2026-01-01T00:00:07+00:00")["until"] == "2026-01-01T00:05:00+00:00"


def test_not_due_is_none_stale_wake():
    b = _binding({"every": "5m", "start_at": "2026-01-01T01:00:00Z"})
    st = _state(b)
    assert A.admit(b, state=st, now="2026-01-01T00:59:59+00:00") is None
    adm = A.admit(b, state=st, now="2026-01-01T01:00:00+00:00")
    assert adm is not None and adm["payload"] == {"tick": 0, "scheduled_at": "2026-01-01T01:00:00+00:00"}
    # A second wake right after admission finds nothing due.
    assert A.admit(b, state=adm["state"], now="2026-01-01T01:00:01+00:00") is None


def test_missed_ticks_coalesce_into_one_admission():
    b = _binding({"every": "5m"})
    st = _state(b)
    # Down from 00:00 to 00:17: ticks 0,1,2,3 are due; ONE admission for tick 3.
    adm = A.admit(b, state=st, now="2026-01-01T00:17:00+00:00")
    assert adm["event_id"] == "schedule@1:b-1:3"
    assert adm["fired_at"] == "2026-01-01T00:15:00+00:00"
    assert adm["coalesced"] == {"first_tick": 0, "last_tick": 3, "missed_count": 3}
    assert adm["payload"] == {"tick": 3, "scheduled_at": "2026-01-01T00:15:00+00:00",
                              "coalesced": {"first_tick": 0, "last_tick": 3, "missed_count": 3}}
    assert adm["state"]["tick"] == 4
    assert adm["state"]["scheduled_count"] == 1  # one coalesced firing counts once
    assert A.prepare(b, state=adm["state"], now="2026-01-01T00:17:00+00:00")["until"] == "2026-01-01T00:20:00+00:00"


def test_count_counts_scheduled_admissions_and_exhausts():
    b = _binding({"every": "1h", "count": 2})
    st = _state(b)
    a1 = A.admit(b, state=st, now=T0)
    assert a1["state"]["exhausted"] is False
    a2 = A.admit(b, state=a1["state"], now="2026-01-01T05:00:00+00:00")  # coalesced, counts once
    assert a2["state"]["scheduled_count"] == 2
    assert a2["state"]["exhausted"] is True
    assert A.prepare(b, state=a2["state"], now="2026-01-01T05:00:00+00:00") == {"kind": "exhausted"}
    assert A.admit(b, state=a2["state"], now="2026-01-02T00:00:00+00:00") is None


def test_until_is_exclusive():
    b = _binding({"every": "1h", "until": "2026-01-01T03:00:00Z"})
    st = _state(b)
    # Ticks 00,01,02 are eligible; 03:00 == until is not.
    adm = A.admit(b, state=st, now="2026-01-01T09:00:00+00:00")
    assert adm["event_id"] == "schedule@1:b-1:2"
    assert adm["state"]["exhausted"] is True
    assert adm["coalesced"]["last_tick"] == 2
    # Exactly at a boundary one step before until.
    st2 = A.admit(b, state=_state(b), now="2026-01-01T02:00:00+00:00")["state"]
    assert st2["exhausted"] is True


def test_until_not_multiple_of_every():
    b = _binding({"every": "1h", "until": "2026-01-01T02:30:00Z"})
    adm = A.admit(b, state=_state(b), now="2026-01-01T09:00:00+00:00")
    assert adm["event_id"] == "schedule@1:b-1:2"  # 02:00 < 02:30


def test_one_shot_single_tick_then_exhausted():
    b = _binding({"start_at": "2026-01-01T06:00:00Z"})
    st = _state(b)
    assert A.prepare(b, state=st, now=T0) == {"kind": "until", "until": "2026-01-01T06:00:00+00:00"}
    assert A.admit(b, state=st, now=T0) is None
    adm = A.admit(b, state=st, now="2026-01-03T00:00:00+00:00")  # late: still exactly T0
    assert adm["event_id"] == "schedule@1:b-1:0"
    assert adm["fired_at"] == "2026-01-01T06:00:00+00:00"
    assert "coalesced" not in adm
    assert adm["state"]["exhausted"] is True
    assert A.prepare(b, state=adm["state"], now="2026-01-03T00:00:00+00:00") == {"kind": "exhausted"}


def test_rearm_skips_past_ticks_and_never_admits():
    b = _binding({"every": "10m"})
    st = A.admit(b, state=_state(b), now=T0)["state"]  # tick -> 1
    # Paused from 00:01 to 00:45; resume re-arms to the first tick AFTER now.
    re = A.rearm(b, state=st, now="2026-01-01T00:45:00+00:00")
    assert re["tick"] == 5 and re["scheduled_count"] == 1
    assert A.admit(b, state=re, now="2026-01-01T00:45:00+00:00") is None
    assert A.prepare(b, state=re, now="2026-01-01T00:45:00+00:00")["until"] == "2026-01-01T00:50:00+00:00"
    # Exactly on a tick: that tick is "at or before now" and is skipped too.
    assert A.rearm(b, state=st, now="2026-01-01T00:40:00+00:00")["tick"] == 5


def test_rearm_one_shot_that_passed_while_paused_is_exhausted():
    b = _binding({"start_at": "2026-01-01T06:00:00Z"})
    re = A.rearm(b, state=_state(b), now="2026-01-01T07:00:00+00:00")
    assert re["exhausted"] is True
    assert A.admit(b, state=re, now="2026-01-01T07:00:00+00:00") is None
    # Before its time the one-shot stays armed.
    assert A.rearm(b, state=_state(b), now="2026-01-01T05:00:00+00:00")["exhausted"] is False


def test_utc_grid_has_no_dst():
    # 2026-03-29 is the EU spring-forward day: an every-24h grid keeps exact 24h steps.
    b = _binding({"every": "24h", "start_at": "2026-03-28T08:00:00Z"})
    st = A.admit(b, state=_state(b), now="2026-03-28T08:00:00+00:00")["state"]
    assert A.prepare(b, state=st, now="2026-03-28T09:00:00+00:00")["until"] == "2026-03-29T08:00:00+00:00"
    assert A.admit(b, state=st, now="2026-03-30T08:00:00+00:00")["fired_at"] == "2026-03-30T08:00:00+00:00"


def test_new_binding_id_gives_new_event_ids():
    cfg = {"every": "5m"}
    a1 = A.admit(_binding(cfg, "b-1"), state=_state(_binding(cfg)), now=T0)
    a2 = A.admit(_binding(cfg, "b-2"), state=_state(_binding(cfg)), now=T0)
    assert a1["event_id"] != a2["event_id"]


def test_normalize_envelope():
    b = _binding({"every": "5m"})
    env = A.normalize(b, event_id="e", fired_at=T0, payload={"tick": 0, "scheduled_at": T0})
    assert env == {"event_id": "e", "source_id": "schedule", "source_version": 1,
                   "fired_at": T0, "payload": {"tick": 0, "scheduled_at": T0}, "binding_id": "b-1"}


EDITOR_SCHEMA_KEYWORDS = {"type", "enum", "pattern", "format", "minimum", "maximum", "required",
                          "additionalProperties", "properties"}


def _schema_keywords(schema, out):
    for key, value in schema.items():
        out.add(key)
        if key == "properties":
            for sub in value.values():
                _schema_keywords(sub, out)
    return out


@pytest.mark.parametrize("adapter_cls", ["schedule", "manual"])
def test_descriptor_schemas_stay_in_the_editor_subset(adapter_cls):
    from abstractruntime.triggers import ManualTriggerAdapter

    adapter = A if adapter_cls == "schedule" else ManualTriggerAdapter()
    for name in ("config_schema", "event_schema"):
        schema = adapter.descriptor[name]
        assert _schema_keywords(schema, set()) <= EDITOR_SCHEMA_KEYWORDS
        assert schema["additionalProperties"] is False
    if adapter_cls == "schedule":
        every = A.descriptor["config_schema"]["properties"]["every"]
        assert every["format"] == "duration" and every["pattern"] == "^[1-9][0-9]*[smhd]$"
        assert set(A.descriptor["event_schema"]["properties"]) == {"tick", "scheduled_at", "coalesced"}
