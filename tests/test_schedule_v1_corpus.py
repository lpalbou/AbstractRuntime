"""schedule@1 never changes meaning (R16.1 A1).

The fixture holds the trigger configs of real automations (an operator gateway's store, read-only,
anonymized) plus the dialog's synthetic v1 shapes, with every adapter answer the RELEASED 0.9.1
schedule@1 adapter gave for them (validate, re-validate, initial state, prepare/admit/rearm over
several cursors and clocks, normalize). The current adapter must answer byte-identically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abstractruntime.triggers.registry import get_trigger_adapter

CORPUS = json.loads((Path(__file__).parent / "fixtures" / "schedule_v1_corpus.json").read_text(encoding="utf-8"))


def _dump(value):
    return json.dumps(value, sort_keys=True)


def test_corpus_is_not_empty_and_holds_real_automations():
    names = [c["name"] for c in CORPUS["cases"]]
    assert sum(1 for n in names if n.startswith("automation-")) >= 4
    assert all("probes" in c["expected"] for c in CORPUS["cases"])


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda c: c["name"])
def test_schedule_v1_answers_byte_identically(case):
    adapter = get_trigger_adapter("schedule", 1)
    exp = case["expected"]
    cfg = adapter.validate(case["config"], now=CORPUS["now"])
    assert _dump(cfg) == _dump(exp["validated"])
    assert _dump(adapter.validate(cfg, now="2030-01-01T00:00:00+00:00")) == _dump(exp["revalidated"])
    assert _dump(adapter.initial_state(cfg)) == _dump(exp["initial_state"])
    binding = {"binding_id": case["binding_id"], "source_id": "schedule", "source_version": 1, "config": cfg}
    for probe in exp["probes"]:
        st, now = probe["state"], probe["now"]
        assert _dump(adapter.prepare(binding, state=st, now=now)) == _dump(probe["prepare"]), (st, now)
        assert _dump(adapter.admit(binding, state=st, now=now)) == _dump(probe["admit"]), (st, now)
        assert _dump(adapter.rearm(binding, state=st, now=now)) == _dump(probe["rearm"]), (st, now)
    got = adapter.normalize(binding, event_id="e1", fired_at=cfg["start_at"], payload={"tick": 0, "scheduled_at": cfg["start_at"]})
    assert _dump(got) == _dump(exp["normalize"])
