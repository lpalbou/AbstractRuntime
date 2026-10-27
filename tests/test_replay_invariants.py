"""Replay reducer invariants over every conformance fixture (framework backlog 0993, ADR-0009).

- replay(all) == fold(apply) under many chunkings: seeded random interleavings of the runs
  (each run's records in its ledger order), random chunk sizes, snapshots given at the start,
  at the end, or both;
- folding is idempotent: re-applying records already folded changes nothing (empty delta lists,
  no typed changes);
- the client side agrees: folding every returned delta with `apply_delta` from {} gives the final
  `to_json()`, and `apply_delta(old, diff_views(old, new)) == new` between any two views;
- cursor tokens parse with `parse_cursor_token`, name every run's last seq and are header-safe.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

import abstractruntime.replay as replay_pkg
from abstractruntime.replay import ReplayView, apply_delta, diff_views, parse_cursor_token, replay

from abstractruntime.replay.conformance import HERE as FIXTURES  # noqa: E402  (shipped with the package)
CASES = sorted(p.name for p in FIXTURES.iterdir() if (p / "manifest.json").exists())
SEEDS = range(12)


def test_suite_runs_against_this_checkout_and_has_cases():
    # The invariants are only meaningful against THIS checkout's reducer, over a real corpus.
    assert Path(replay_pkg.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")
    assert len(CASES) >= 10, CASES


def _load(case):
    d = FIXTURES / case
    records = json.loads((d / "records.json").read_text(encoding="utf-8"))
    runs = json.loads((d / "runs.json").read_text(encoding="utf-8"))
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return records, runs, manifest["root_run_id"]


def _stream(records, rng):
    """All records: each run's in order, the runs interleaved at random."""
    queues = {rid: list(recs) for rid, recs in records.items() if recs}
    out = []
    while queues:
        rid = rng.choice(sorted(queues))
        take = rng.randint(1, 3)
        out.extend(queues[rid][:take])
        del queues[rid][:take]
        if not queues[rid]:
            del queues[rid]
    return out


def _chunks(stream, rng):
    out, i = [], 0
    while i < len(stream):
        n = rng.randint(1, 6)
        out.append(stream[i:i + n])
        i += n
    return out or [[]]


def _fold(root, chunks, runs, mode):
    """Fold chunk by chunk; return (view, deltas)."""
    view = ReplayView(root)
    deltas = []
    for i, chunk in enumerate(chunks):
        give = (mode in ("start", "both") and i == 0) or (mode in ("end", "both") and i == len(chunks) - 1)
        deltas.append(view.apply(copy.deepcopy(chunk), runs_by_id=copy.deepcopy(runs) if give else None))
    return view, deltas


@pytest.mark.parametrize("case", CASES)
def test_replay_all_equals_fold_of_apply(case):
    records, runs, root = _load(case)
    expected = replay(copy.deepcopy(records), copy.deepcopy(runs), root_run_id=root).to_json()
    for seed in SEEDS:
        rng = random.Random(f"{case}:{seed}")
        chunks = _chunks(_stream(records, rng), rng)
        for mode in ("start", "end", "both"):
            view, _ = _fold(root, chunks, runs, mode)
            got = view.to_json()
            assert got == expected, f"{case} seed={seed} snapshots={mode}: fold(apply) != replay(all)"


@pytest.mark.parametrize("case", CASES)
def test_apply_is_idempotent(case):
    records, runs, root = _load(case)
    view = replay(copy.deepcopy(records), copy.deepcopy(runs), root_run_id=root)
    before = view.to_json()
    all_records = [r for rid in sorted(records) for r in records[rid]]
    for again in (all_records, list(reversed(all_records)), all_records[: len(all_records) // 2]):
        delta = view.apply(copy.deepcopy(again), runs_by_id=copy.deepcopy(runs))
        assert view.to_json() == before
        for key in ("runs", "messages", "activities", "artifacts", "changes"):
            assert delta[key] == [], f"{case}: re-applying folded records changed {key}: {delta[key][:2]}"


@pytest.mark.parametrize("case", CASES)
def test_apply_delta_folds_to_the_view(case):
    records, runs, root = _load(case)
    for seed in SEEDS:
        rng = random.Random(f"delta:{case}:{seed}")
        chunks = _chunks(_stream(records, rng), rng)
        mode = ("start", "end", "both")[seed % 3]
        view, deltas = _fold(root, chunks, runs, mode)
        client = {}
        for d in deltas:
            client = apply_delta(client, json.loads(json.dumps(d)))  # what crosses the wire
        assert client == view.to_json(), f"{case} seed={seed} snapshots={mode}"
        # The first full view plus the later deltas gives the same answer (a client that joined late).
        def give(i):
            return (mode in ("start", "both") and i == 0) or (mode in ("end", "both") and i == len(chunks) - 1)

        first = ReplayView(root)
        first.apply(copy.deepcopy(chunks[0]), runs_by_id=copy.deepcopy(runs) if give(0) else None)
        joined = first.to_json()
        for chunk_i, chunk in enumerate(chunks[1:], start=1):
            joined = apply_delta(joined, first.apply(copy.deepcopy(chunk), runs_by_id=copy.deepcopy(runs) if give(chunk_i) else None))
        assert joined == first.to_json() == view.to_json()


@pytest.mark.parametrize("case", CASES)
def test_diff_views_round_trips(case):
    records, runs, root = _load(case)
    rng = random.Random(f"diff:{case}")
    chunks = _chunks(_stream(records, rng), rng)
    view = ReplayView(root)
    snapshots = [{}]
    for i, chunk in enumerate(chunks):
        view.apply(copy.deepcopy(chunk), runs_by_id=copy.deepcopy(runs) if i == len(chunks) - 1 else None)
        snapshots.append(view.to_json())
    final = snapshots[-1]
    for old in snapshots:
        assert apply_delta(old, diff_views(old, final)) == final
    for a, b in zip(snapshots, snapshots[1:]):
        assert apply_delta(a, diff_views(a, b)) == b
    assert diff_views(final, final)["runs"] == [] and diff_views(final, final)["activities"] == []


@pytest.mark.parametrize("case", CASES)
def test_cursor_token_parses_names_every_run_and_is_header_safe(case):
    records, runs, root = _load(case)
    view = replay(copy.deepcopy(records), copy.deepcopy(runs), root_run_id=root)
    cursor = view.to_json()["cursor"]
    token = cursor["token"]
    assert token and not any(ch in token for ch in ',"; \t\r\n'), token
    assert token.isascii()
    parsed = parse_cursor_token(token)
    assert parsed is not None
    order = view.run_ids
    # Every run of the tree appears with its last seq (prefixes are unique in every fixture).
    assert len({rid[:8] for rid in order}) == len(order)
    assert parsed == {rid[:8]: view.last_seq(rid) for rid in order}
    assert parse_cursor_token(f'W/"{token}"') == parsed  # as an ETag
    assert cursor["records"] == sum(len(v) for v in records.values())
    for rid, recs in records.items():
        if recs:
            assert cursor["runs"][rid] == recs[-1]["seq"]


def test_parse_cursor_token_rejects_what_is_not_a_cursor():
    for bad in (None, "", "   ", "abc", "abc:x", ":3", "abc:-1", 42):
        assert parse_cursor_token(bad) is None
    assert parse_cursor_token("abcd1234:7.ef567890:0") == {"abcd1234": 7, "ef567890": 0}


def test_delta_changes_are_typed_ordered_and_ignored_by_apply_delta():
    records, runs, root = _load("a_delegated_child_approval")
    view = ReplayView(root)
    child = next(rid for rid, s in runs.items() if s.get("parent_run_id") and (s.get("waiting") or {}).get("wait_key", "").startswith("tool_approval:"))
    last = records[child][-1]
    head = {rid: [r for r in recs if r is not last] for rid, recs in records.items()}
    d1 = view.apply([r for rid in sorted(head) for r in head[rid]], runs_by_id=None)
    assert d1["changes"], "the first delta reports what appeared"
    kinds = [c["kind"] for c in d1["changes"]]
    rank = {"status": 0, "interaction": 1, "run": 2, "message": 3, "activity": 4, "artifact": 5}
    assert kinds == sorted(kinds, key=rank.__getitem__)
    for c in d1["changes"]:
        assert {"kind", "id", "run_id", "seq"} <= set(c) and isinstance(c["seq"], int)
    status_root = [c for c in d1["changes"] if c["kind"] == "status" and c["run_id"] == root]
    assert status_root and status_root[0]["previous"] is None
    # The approval wait arrives: one interaction `opened` change at the wait's seq.
    d2 = view.apply([last])
    opened = [c for c in d2["changes"] if c["kind"] == "interaction"]
    assert opened == [{"kind": "interaction", "id": last["result"]["wait"]["wait_key"], "run_id": child,
                       "seq": last["seq"], "change": "opened", "interaction_kind": "tool_approval",
                       "wait_key": last["result"]["wait"]["wait_key"]}]
    assert [c for c in d2["changes"] if c["kind"] == "status" and c["run_id"] == child] == [
        {"kind": "status", "id": child, "run_id": child, "seq": last["seq"], "status": "waiting", "previous": "running"}]
    activity = [c for c in d2["changes"] if c["kind"] == "activity"]
    assert [c["seq"] for c in activity] == [last["seq"]]
    # The same content twice yields the same changes (deterministic).
    again = ReplayView(root)
    again.apply([r for rid in sorted(head) for r in head[rid]])
    assert again.apply([last])["changes"] == d2["changes"]
    # apply_delta ignores `changes`.
    base = view.to_json()
    assert apply_delta(base, {**diff_views(base, base), "changes": d2["changes"]}) == base
    # A resume closes it: a `closed` change.
    resume = {"run_id": child, "step_id": "resume-x", "node_id": "act", "status": "completed", "seq": last["seq"] + 1,
              "effect": {"type": "resume", "payload": {"wait_key": last["result"]["wait"]["wait_key"], "payload": {"approved": True}}},
              "result": {"resumed": True}, "started_at": "2099-01-01T00:00:00+00:00", "ended_at": "2099-01-01T00:00:00+00:00"}
    d3 = view.apply([resume])
    closed = [c for c in d3["changes"] if c["kind"] == "interaction"]
    assert [(c["change"], c["seq"], c["interaction_kind"]) for c in closed] == [("closed", last["seq"] + 1, "tool_approval")]
