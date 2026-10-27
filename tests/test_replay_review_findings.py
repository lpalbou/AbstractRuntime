"""Regressions for the adversarial review of the replay view (2026-10-27).

Each test reproduces a confirmed finding: an artifact's provenance depending on fold order, an
approval card listing calls nobody is asked about, an ETag blind to a pause, a sticky snapshot
wait, a paused event wait shown as a question, and an unbounded cursor token.
"""
from __future__ import annotations

import random

from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState
from abstractruntime.automations import attention
from abstractruntime.replay import ReplayView, parse_cursor_token, replay


def _rec(run, seq, step, status="completed", **kw):
    return {"run_id": run, "step_id": step, "node_id": "n", "status": status, "seq": seq,
            "started_at": f"2026-10-27T10:00:{seq:02d}+00:00", "ended_at": f"2026-10-27T10:00:{seq:02d}+00:00", **kw}


def test_an_artifact_names_its_earliest_reference_whatever_the_fold_order() -> None:
    art = {"$artifact": "art-1"}
    sub = {"type": "start_subworkflow", "payload": {"workflow_id": "wf:child"}}
    root = [_rec("R", 1, "s1", "waiting", effect=sub, result={"wait": {"reason": "subworkflow", "wait_key": "subworkflow:C"}}),
            _rec("R", 9, "s2", effect={"type": "llm_call", "payload": {}}, result={"echo": art})]
    child = [_rec("C", 1, "c1", effect={"type": "tool_calls", "payload": {"tool_calls": []}}, result={"results": [{"output": art}]})]
    views = []
    for order in (["R", "C"], ["C", "R"]):
        v = ReplayView("R")
        for rid in order:
            v.apply(root if rid == "R" else child)
        views.append(v.to_json())
    fresh = replay({"R": root, "C": child}, root_run_id="R").to_json()
    assert views[0] == views[1] == fresh
    assert fresh["artifacts"][0]["run_id"] == "C"  # the child produced it first (10:00:01)


def test_an_approval_asks_only_about_the_calls_still_awaiting_it() -> None:
    calls = [{"name": "read_file", "call_id": "0"}, {"name": "execute_command", "call_id": "1"}]
    eff = {"type": "tool_calls", "payload": {"tool_calls": calls}}
    wait = {"reason": "user", "wait_key": "tool_approval:R:act:k", "details": {
        "mode": "approval_required", "tool_calls": [calls[1]],
        "blocked_by_index": {"0": {"call_id": "0", "name": "read_file", "success": False, "output": None, "error": "outside workspace"}},
    }}
    v = replay({"R": [_rec("R", 1, "t", "started", effect=eff), _rec("R", 2, "t", "waiting", effect=eff, result={"wait": wait})]},
               root_run_id="R").to_json()
    assert [c["name"] for c in v["interaction"]["tool_calls"]] == ["execute_command"]
    rows = {r["name"]: r for r in next(a for a in v["activities"] if a["kind"] == "tool")["tool_calls"]}
    assert rows["execute_command"]["status"] == "waiting"
    assert rows["read_file"]["status"] == "failed" and rows["read_file"]["resolved_before_approval"] is True


RUNNING = {"run_id": "R", "workflow_id": "wf", "status": "running"}
PAUSED = {**RUNNING, "status": "waiting", "waiting": {"reason": "user", "wait_key": "pause:R", "details": {"kind": "pause"}}}


def test_a_pause_without_a_record_changes_the_cursor_token() -> None:
    recs = [_rec("R", 1, "a", effect={"type": "llm_call", "payload": {}})]
    running = replay({"R": recs}, {"R": RUNNING}, root_run_id="R")
    paused = replay({"R": recs}, {"R": PAUSED}, root_run_id="R")
    assert running.status() == "running" and paused.status() == "waiting"
    assert running.cursor["token"] != paused.cursor["token"]
    assert parse_cursor_token(paused.cursor["token"]) == {"R": 1}


def test_a_snapshot_wait_lives_as_long_as_the_snapshot_says_and_comes_back() -> None:
    recs = [_rec("R", 1, "a", effect={"type": "llm_call", "payload": {}})]
    v = ReplayView("R")
    v.apply(recs, runs_by_id={"R": RUNNING})
    for snap, records in ((PAUSED, recs), (RUNNING, recs), (PAUSED, recs)):
        v.apply([], runs_by_id={"R": snap})
        fresh = replay({"R": records}, {"R": snap}, root_run_id="R")
        assert v.status() == fresh.status() and v.to_json() == fresh.to_json()


def test_a_paused_event_wait_asks_nobody_like_attention_says() -> None:
    wait = WaitState(reason=WaitReason.EVENT, wait_key="evt:x", prompt="Approve?", choices=["yes", "no"])
    run = RunState(run_id="R", workflow_id="wf", status=RunStatus.WAITING, current_node="n", waiting=wait,
                   vars={"_runtime": {"control": {"paused": True}}})
    eff = {"type": "wait_event", "payload": {"wait_key": "evt:x"}}
    recs = [_rec("R", 1, "s1", "started", effect=eff),
            _rec("R", 2, "s1", "waiting", effect=eff, result={"wait": {"reason": "event", "wait_key": "evt:x", "prompt": "Approve?", "choices": ["yes", "no"]}})]
    assert attention.wait_kind(run) is None
    assert replay({"R": recs}, {"R": run}, root_run_id="R").to_json()["interaction"] is None
    run.vars = {}
    assert replay({"R": recs}, {"R": run}, root_run_id="R").to_json()["interaction"]["kind"] == "event"


def test_a_big_tree_gets_a_bounded_token() -> None:
    kids = [f"child-{i:03d}-{random.Random(i).randrange(10**6):06d}" for i in range(60)]
    root = [_rec("R", i + 1, f"s{i}", "waiting", effect={"type": "start_subworkflow", "payload": {}},
                 result={"wait": {"reason": "subworkflow", "wait_key": f"subworkflow:{k}"}}) for i, k in enumerate(kids)]
    v = replay({"R": root, **{k: [_rec(k, 1, "x", effect={"type": "llm_call", "payload": {}})] for k in kids}}, root_run_id="R")
    token = v.cursor["token"]
    assert len(token) < 64 and parse_cursor_token(token) is None
    assert v.cursor["runs"][kids[0]] == 1
