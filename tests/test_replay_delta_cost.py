"""A delta costs what changed, not the size of the view (gateway benchmark, 2026-10-27).

With 50 subscribers at 20 appends/s, every byte of a delta and every copy `apply_delta` makes is
paid 1,000 times a second: one record's delta carries the one run row it touched, and applying it
shares every unchanged item with the previous view instead of copying the view.
"""
from __future__ import annotations

import json

from abstractruntime.replay import apply_delta, replay
from abstractruntime.replay.conformance import load_case


def _tree():
    case = load_case("a2_delegated_approval_completed")
    root = case["manifest"]["root_run_id"]
    assert len(case["records"]) >= 2
    return case, root


def test_one_record_sends_the_one_run_row_it_touched() -> None:
    case, root = _tree()
    child = next(rid for rid in case["records"] if rid != root and case["records"][rid])
    records = {rid: list(recs) for rid, recs in case["records"].items()}
    last = records[child].pop()
    view = replay(records, case["runs"], root_run_id=root)
    delta = view.apply([last])
    assert [r["run_id"] for r in delta["runs"]] == [child]
    assert len(json.dumps(delta)) < len(json.dumps(view.to_json())) / 10


def test_apply_delta_shares_unchanged_items_and_leaves_its_inputs_alone() -> None:
    case, root = _tree()
    child = next(rid for rid in case["records"] if rid != root and case["records"][rid])
    records = {rid: list(recs) for rid, recs in case["records"].items()}
    last = records[child].pop()
    view = replay(records, case["runs"], root_run_id=root)
    before = view.to_json()
    snapshot = json.dumps(before, sort_keys=True)
    delta = view.apply([last])
    delta_copy = json.dumps(delta, sort_keys=True)
    after = apply_delta(before, delta)
    assert after == view.to_json()
    assert json.dumps(before, sort_keys=True) == snapshot and json.dumps(delta, sort_keys=True) == delta_copy
    changed = {a["id"] for a in delta["activities"]}
    kept = [a for a in after["activities"] if a["id"] not in changed]
    assert kept and all(any(a is b for b in before["activities"]) for a in kept)
