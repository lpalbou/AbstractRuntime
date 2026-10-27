"""Replay conformance: each fixture's view equals its golden, AND means what the case is about.

Fixtures live in `src/abstractruntime/replay/conformance/<case>/` (shipped with the package) (records.json, runs.json, manifest.json,
expected_view.json; `regen.py` rewrites the goldens after a reviewed change). Most are redacted
from the operator's gateway data; the manifest says which. The golden comparison pins the whole
view; the per-case assertions below are written from the case's meaning and computed from the
raw records where possible, so a golden regenerated over a bug still fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import abstractruntime.replay as replay_pkg
from abstractruntime.replay import PERSON_FACING_KINDS, REPLAY_VIEW_VERSION, replay, wait_kind, workflow_ref

from abstractruntime.replay.conformance import HERE as FIXTURES  # noqa: E402  (shipped with the package)
CASES = sorted(p.name for p in FIXTURES.iterdir() if (p / "manifest.json").exists())


def _load(case):
    d = FIXTURES / case
    records = json.loads((d / "records.json").read_text(encoding="utf-8"))
    runs = json.loads((d / "runs.json").read_text(encoding="utf-8"))
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return records, runs, manifest


def _view(case):
    records, runs, manifest = _load(case)
    return replay(records, runs, root_run_id=manifest["root_run_id"]).to_json(), records, runs, manifest


def _etype(rec):
    return (rec.get("effect") or {}).get("type")


def _wait(rec):
    return ((rec.get("result") or {}).get("wait")) or {}


def _acts(view, **match):
    return [a for a in view["activities"] if all(a.get(k) == v for k, v in match.items())]


def _row(view, run_id):
    return next(r for r in view["runs"] if r["run_id"] == run_id)


# --------------------------------------------------------------------------- every case


def test_corpus_is_present_and_tested_against_this_checkout():
    assert Path(replay_pkg.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")
    expected = {"a_delegated_child_approval", "a2_delegated_approval_completed", "b_timer_without_resume",
                "c_parallel_tool_batch", "d_cancellation", "e_structured_output_agents", "e2_structured_repeated_node",
                "f_windowed_slim", "g_ask_user", "h_denied_approval", "i_wait_kinds", "j_event_answered_then_pause",
                "k_nested_delegation", "l_answer_then_completion"}
    assert expected <= set(CASES), expected - set(CASES)
    for case in CASES:
        d = FIXTURES / case
        size = sum(f.stat().st_size for f in d.iterdir())
        assert size < 200_000, (case, size)
        manifest = json.loads((d / "manifest.json").read_text())
        assert manifest["case"] == case and manifest["pins"] and manifest["provenance"]


@pytest.mark.parametrize("case", CASES)
def test_view_equals_golden(case):
    view, *_ = _view(case)
    golden = json.loads((FIXTURES / case / "expected_view.json").read_text(encoding="utf-8"))
    assert view == golden, f"{case}: the view changed; review, then `python scripts/regen_replay_conformance.py --write {case}`"


@pytest.mark.parametrize("case", CASES)
def test_view_is_plain_json_without_storage_encodings(case):
    view, records, runs, manifest = _view(case)
    text = json.dumps(view, sort_keys=True)
    assert "$slim" not in text
    assert json.loads(text) == view
    assert view["version"] == REPLAY_VIEW_VERSION
    assert view["root_run_id"] == manifest["root_run_id"]
    assert view["runs"][0]["run_id"] == manifest["root_run_id"] and view["runs"][0]["depth"] == 0
    # Every run the records or snapshots name is in the tree once.
    ids = [r["run_id"] for r in view["runs"]]
    assert len(ids) == len(set(ids)) and set(records) | set(runs) <= set(ids)
    # The interaction, when there is one, is a person-facing open wait of a run in the tree.
    it = view["interaction"]
    if it is not None:
        assert it["kind"] in PERSON_FACING_KINDS
        assert any(w["run_id"] == it["run_id"] and w["wait_key"] == it["wait_key"] and w["person_facing"] for w in view["open_waits"])
        assert it["command"] == {"type": "resume", "run_id": it["run_id"], "payload": {"wait_key": it["wait_key"]}}
        assert [r["run_id"] for r in view["runs"] if r["holds_interaction"]] == [it["run_id"]]


@pytest.mark.parametrize("case", CASES)
def test_workflow_fields(case):
    view, records, runs, manifest = _view(case)
    root = manifest["root_run_id"]
    root_wf = runs[root]["workflow_id"]
    expect = workflow_ref(root_wf)
    expect["entrypoint"] = expect["flow_id"]
    assert view["workflow"] == expect
    for row in view["runs"]:
        assert row["workflow"] == workflow_ref(row["workflow_id"])
        if row["run_id"] in runs:
            assert row["workflow_id"] == runs[row["run_id"]]["workflow_id"]


def test_workflow_ref_grammar():
    assert workflow_ref("basic-agent@0.0.5:81795ea9") == {
        "id": "basic-agent@0.0.5:81795ea9", "bundle_id": "basic-agent", "bundle_version": "0.0.5", "flow_id": "81795ea9"}
    assert workflow_ref("basic-agent:81795ea9")["bundle_version"] is None
    assert workflow_ref("visual_react_agent_basic-agent_0_0_5_81795ea9_node-2") == {
        "id": "visual_react_agent_basic-agent_0_0_5_81795ea9_node-2", "bundle_id": None, "bundle_version": None,
        "flow_id": "visual_react_agent_basic-agent_0_0_5_81795ea9_node-2"}
    assert workflow_ref("") is None and workflow_ref(None) is None


# --------------------------------------------------------------------------- per case


def test_a_delegated_child_approval_is_the_childs_question_with_resolved_tool_calls():
    view, records, runs, manifest = _view("a_delegated_child_approval")
    root = manifest["root_run_id"]
    assert view["workflow"] == {"id": "basic-agent@0.0.5:81795ea9", "bundle_id": "basic-agent", "bundle_version": "0.0.5",
                                "flow_id": "81795ea9", "entrypoint": "81795ea9"}
    child = next(rid for rid, s in runs.items() if str((s.get("waiting") or {}).get("wait_key")).startswith("tool_approval:"))
    last = records[child][-1]
    assert last["status"] == "waiting" and "$slim" in last["effect"]["payload"]["tool_calls"]  # the incident's shape
    started = next(r for r in records[child] if r["step_id"] == last["step_id"] and r["status"] == "started")
    it = view["interaction"]
    assert it["kind"] == "tool_approval" and it["run_id"] == child
    assert it["wait_key"] == _wait(last)["wait_key"] and it["seq"] == last["seq"]
    assert it["tool_calls"] == started["effect"]["payload"]["tool_calls"]  # the pointer, resolved
    assert [c["name"] for c in it["tool_calls"]] == ["write_file"]
    assert it["answer_payload"] == "{approved: true|false}"
    assert "tool_calls" not in (it["details"] or {})
    # The root waits on its delegation: an open wait, never the interaction.
    assert view["status"] == "waiting"
    root_wait = next(w for w in view["open_waits"] if w["run_id"] == root)
    assert root_wait["kind"] == "delegation" and root_wait["child_run_id"] == child and not root_wait["person_facing"]
    assert _row(view, child)["parent_run_id"] == root and _row(view, child)["status"] == "waiting"
    act = next(a for a in view["activities"] if a["id"] == it["activity_id"])
    assert act["status"] == "waiting" and [t["status"] for t in act["tool_calls"]] == ["waiting"]


def test_a2_every_approved_call_is_one_activity_and_the_turn_completes():
    view, records, runs, manifest = _view("a2_delegated_approval_completed")
    assert view["status"] == "completed" and view["interaction"] is None and view["open_waits"] == []
    child = next(rid for rid, recs in records.items() if any(_etype(r) == "tool_calls" for r in recs))
    tool_idems = {r["idempotency_key"] for r in records[child] if _etype(r) == "tool_calls"}
    tools = _acts(view, run_id=child, kind="tool")
    assert len(tools) == len(tool_idems) >= 3  # an approved call (new step id, same key) is not a second row
    for a in tools:
        assert a["status"] == "completed" and a["wait"]["kind"] == "tool_approval" and a["wait"]["ended_by"]
        assert a["tool_calls"] and all(t["status"] == "completed" for t in a["tool_calls"])
        assert len(a["step_ids"]) >= 2  # the waiting step and the approved execution's step
    done = next(r for r in records[manifest["root_run_id"]] if str(r.get("idempotency_key")).startswith("system:completion:"))
    final = next(m for m in view["messages"] if m["kind"] == "final")
    assert final["text"] == done["result"]["output"]["response"] and final["seqs"] == [done["seq"]]
    assert view["warnings"] == []


def test_b_timer_without_resume_is_over_once_a_later_step_exists():
    view, records, runs, manifest = _view("b_timer_without_resume")
    root = manifest["root_run_id"]
    recs = records[root]
    assert not any(_etype(r) == "resume" for r in recs)
    timer = next(r for r in recs if _etype(r) == "wait_until" and r["status"] == "waiting")
    assert any(r["seq"] > timer["seq"] and _etype(r) not in (None, "emit_event") for r in recs)
    assert runs[root]["status"] == "waiting"  # the stale snapshot still says waiting
    assert view["status"] == "running" != "waiting"
    assert view["open_waits"] == [] and _row(view, root)["waiting"] is None
    (act,) = _acts(view, kind="timer")
    assert act["wait"]["kind"] == "timer" and act["wait"]["ended_by"] == "later_step" and act["status"] == "completed"


def test_c_parallel_tool_batch_matches_each_call_to_its_result():
    view, records, *_ = _view("c_parallel_tool_batch")
    batches = [r for recs in records.values() for r in recs
               if _etype(r) == "tool_calls" and r["status"] == "completed" and len(r["effect"]["payload"]["tool_calls"]) > 1]
    assert batches
    for done in batches:
        calls = done["effect"]["payload"]["tool_calls"]
        results = {x["runtime_call_id"]: x for x in done["result"]["results"]}
        (act,) = [a for a in view["activities"] if done["step_id"] in a["step_ids"]]
        assert [t["id"] for t in act["tool_calls"]] == [c["runtime_call_id"] for c in calls]
        for row, call in zip(act["tool_calls"], calls):
            assert row["name"] == call["name"] and row["arguments"] == call["arguments"]
            assert row["status"] == "completed" and row["output"] == results[call["runtime_call_id"]]["output"]
        assert len({json.dumps(t["output"]) for t in act["tool_calls"]}) == len(calls)


def test_d_cancellation_leaves_nothing_open():
    view, records, runs, manifest = _view("d_cancellation")
    root = manifest["root_run_id"]
    assert view["status"] == "cancelled" and view["interaction"] is None and view["open_waits"] == []
    agent = next(rid for rid, s in runs.items() if s["status"] == "cancelled" and rid != root)
    assert _row(view, agent)["status"] == "cancelled"
    delegation = next(a for a in _acts(view, run_id=root, kind="subworkflow") if a.get("child_run_id") == agent)
    assert delegation["wait"]["ended_by"] == "terminal"
    cancelled_steps = {r["step_id"] for r in records[agent] if r["status"] == "cancelled"}
    assert cancelled_steps and all(a["status"] == "cancelled" for a in view["activities"] if set(a["step_ids"]) & cancelled_steps)


def _agent_executions(recs):
    """step ids of each Agent node execution's start_subworkflow, from the raw records."""
    return [r for r in recs if _etype(r) == "start_subworkflow" and r["status"] == "started"
            and str(r["effect"].get("result_key") or "").startswith("_temp.agent.")]


def test_e_one_activity_per_agent_execution():
    view, records, runs, manifest = _view("e_structured_output_agents")
    root = manifest["root_run_id"]
    starts = _agent_executions(records[root])
    assert [s["node_id"] for s in starts] == ["generate", "reflect", "rank"]
    for s in starts:
        node_recs = [r for r in records[root] if r["node_id"] == s["node_id"] and _etype(r) != "emit_event"]
        (act,) = _acts(view, run_id=root, node_id=s["node_id"])
        assert act["execution_id"] == s["step_id"] and act["kind"] == "subworkflow"
        assert act["seqs"] == [r["seq"] for r in node_recs]  # started, waiting, resume, structured llm_call
        assert act["name"] == s["effect"]["payload"]["workflow_id"]
    structured = [r for r in records[root] if str(r["effect"] and r["effect"].get("result_key")).endswith(".structured")]
    assert structured and all(any(r["seq"] in a["seqs"] for a in _acts(view, run_id=root, kind="subworkflow")) for r in structured)
    assert not _acts(view, run_id=root, kind="llm")
    assert view["status"] == "cancelled"


def test_e2_the_same_agent_node_twice_is_two_activities():
    view, records, runs, manifest = _view("e2_structured_repeated_node")
    root = manifest["root_run_id"]
    starts = [s for s in _agent_executions(records[root]) if s["node_id"] == "node-8"]
    assert len(starts) == 2
    acts = _acts(view, run_id=root, node_id="node-8")
    assert [a["execution_id"] for a in acts] == [s["step_id"] for s in starts]
    for a, s in zip(acts, starts):
        assert len(a["seqs"]) == 5 and a["seqs"][0] == s["seq"]
        assert set(a["seqs"]).isdisjoint(set().union(*(set(b["seqs"]) for b in acts if b is not a)))
    assert view["warnings"] == []  # the approved call's `$slim` resolves against the waiting step's STARTED


def test_f_windowed_slim_warns_and_still_names_the_tool_calls():
    view, records, runs, manifest = _view("f_windowed_slim")
    child = next(rid for rid, s in runs.items() if str((s.get("waiting") or {}).get("wait_key")).startswith("tool_approval:"))
    (only,) = records[child]
    assert only["seq"] > 1 and "$slim" in only["effect"]["payload"]["tool_calls"]
    for rid, recs in records.items():
        assert _row(view, rid)["first_seq"] == recs[0]["seq"] > 1
        assert any(w.startswith(f"run {rid}: records before seq {recs[0]['seq']}") for w in view["warnings"])
    assert any("`tool_calls` points at a STARTED record" in w and only["step_id"] in w for w in view["warnings"])
    it = view["interaction"]
    assert it["kind"] == "tool_approval" and it["run_id"] == child
    assert it["tool_calls"] == _wait(only)["details"]["tool_calls"]  # from the wait's own details
    assert [c["name"] for c in it["tool_calls"]] == ["write_file"]


def test_g_ask_user_question_and_answer():
    view, records, runs, manifest = _view("g_ask_user")
    root = manifest["root_run_id"]
    waiting = next(r for r in records[root] if _etype(r) == "ask_user" and r["status"] == "waiting")
    resume = next(r for r in records[root] if _etype(r) == "resume" and r["effect"]["payload"]["wait_key"] == _wait(waiting)["wait_key"])
    questions = [m for m in view["messages"] if m["kind"] == "question"]
    answers = [m for m in view["messages"] if m["kind"] == "answer" and m["role"] == "user"]
    assert [(q["text"], q["seqs"]) for q in questions] == [(_wait(waiting)["prompt"], [waiting["seq"]])]
    answer = resume["effect"]["payload"]["payload"]
    answer_text = answer if isinstance(answer, str) else (answer.get("response") or answer.get("text") or answer.get("value"))
    assert [(a["text"], a["seqs"]) for a in answers] == [(answer_text, [resume["seq"]])]
    (act,) = _acts(view, kind="ask_user")
    assert act["wait"] == {"kind": "ask_user", "wait_key": _wait(waiting)["wait_key"], "until": None, "ended_by": "resume"}
    assert view["interaction"] is None and view["status"] == "completed"
    final = next(m for m in view["messages"] if m["kind"] == "final")
    assert final["source"] == "run_output" and final["text"] == runs[root]["output"]["result"]["content"]


def test_h_denied_approval_reads_denied():
    view, records, runs, manifest = _view("h_denied_approval")
    denied = [r for recs in records.values() for r in recs if _etype(r) == "tool_calls" and r["status"] == "completed"
              and any(x.get("error") == "Denied by user" for x in r["result"]["results"])]
    assert denied and not any("approved" in json.dumps(r["effect"]) for recs in records.values() for r in recs)
    for rec in denied:
        (act,) = [a for a in view["activities"] if rec["step_id"] in a["step_ids"]]
        assert act["status"] == "denied" and act["denied_reason"] == "Denied by user"
        assert [t["status"] for t in act["tool_calls"]] == ["denied"] * len(rec["result"]["results"])
        assert act["wait"]["kind"] == "tool_approval"
    assert view["status"] == "completed" and view["interaction"] is None


def test_i_wait_kinds_agree_with_the_attention_classifier():
    view, records, runs, manifest = _view("i_wait_kinds")
    kinds = {w["wait_key"]: (w["kind"], w["person_facing"]) for w in view["open_waits"]}
    by_kind = {k: pf for k, pf in kinds.values()}
    assert by_kind == {"pause": False, "automation_wake": False, "signal": False, "event": True}
    it = view["interaction"]
    event_run = next(rid for rid, s in runs.items() if (s.get("waiting") or {}).get("prompt"))
    assert it["kind"] == "event" and it["run_id"] == event_run
    assert it["prompt"] == runs[event_run]["waiting"]["prompt"] and it["choices"] == ["red", "blue"]
    assert it["answer_payload"] == "{payload}"
    assert view["status"] == "running"


def test_j_answered_event_then_pause_is_waiting_without_interaction():
    view, *_ = _view("j_event_answered_then_pause")
    assert view["status"] == "waiting" and view["interaction"] is None
    assert [(w["kind"], w["person_facing"]) for w in view["open_waits"]] == [("pause", False)]
    (event_act,) = [a for a in view["activities"] if (a.get("wait") or {}).get("kind") == "event"]
    assert event_act["wait"]["ended_by"] == "resume" and event_act["status"] == "completed"


def test_k_nested_runs_are_in_tree_order():
    view, records, runs, manifest = _view("k_nested_delegation")
    root = manifest["root_run_id"]
    by_wf = {s["workflow_id"].rsplit(":", 1)[1]: rid for rid, s in runs.items()}
    assert [r["run_id"] for r in view["runs"]] == [root, by_wf["a"], by_wf["g"], by_wf["b"]]
    assert [r["depth"] for r in view["runs"]] == [0, 1, 2, 1]
    assert _row(view, by_wf["g"])["parent_run_id"] == by_wf["a"]
    # The rows' `order` sorts the same way (what apply_delta uses).
    from abstractruntime.replay.delta import _order_key
    assert sorted(view["runs"], key=lambda r: _order_key("runs", r)) == view["runs"]


def test_l_final_repeating_an_answer_names_it():
    view, records, runs, manifest = _view("l_answer_then_completion")
    (answer,) = [m for m in view["messages"] if m["kind"] == "answer"]
    (final,) = [m for m in view["messages"] if m["kind"] == "final"]
    assert final["text"] == answer["text"] and final["repeats"] == answer["id"] and final["source"] == "ledger"


# --------------------------------------------------------------------------- wait kinds (unit)


@pytest.mark.parametrize("wait, effect_type, run_id, kind", [
    ({"reason": "user", "wait_key": "tool_approval:r1:act:k", "details": {"mode": "approval_required"}}, "tool_calls", "r1", "tool_approval"),
    ({"reason": "user", "wait_key": "k", "details": {"mode": "approval_required"}}, "", "", "tool_approval"),
    ({"reason": "subworkflow", "wait_key": "subworkflow:c1"}, "start_subworkflow", "r1", "delegation"),
    ({"reason": "until", "until": "2026-01-01T00:00:00Z"}, "wait_until", "r1", "timer"),
    ({"reason": "job", "wait_key": "job:1"}, "", "r1", "job"),
    ({"reason": "user", "wait_key": "pause:r1"}, "", "r1", "pause"),
    ({"reason": "user", "wait_key": "pause:other"}, "", "r1", "ask_user"),  # attention: only this run's pause key
    ({"reason": "user", "wait_key": "x", "details": {"kind": "pause"}}, "", "r1", "pause"),
    ({"reason": "user", "wait_key": "user:r1:n", "prompt": "Name?"}, "ask_user", "r1", "ask_user"),
    ({"reason": "event", "wait_key": "automation:a1:wake"}, "wait_event", "r1", "automation_wake"),
    ({"reason": "event", "wait_key": "evt:run:r1:go", "prompt": "Go?"}, "wait_event", "r1", "event"),
    ({"reason": "event", "wait_key": "evt:run:r1:go", "choices": ["a"]}, "wait_event", "r1", "event"),
    ({"reason": "event", "wait_key": "evt:run:r1:go"}, "wait_event", "r1", "signal"),
])
def test_wait_kind(wait, effect_type, run_id, kind):
    assert wait_kind(wait, effect_type, run_id) == kind
    assert (kind in PERSON_FACING_KINDS) == (kind in ("tool_approval", "ask_user", "event"))


def test_wait_kind_matches_the_runtime_attention_classifier():
    from abstractruntime.automations.attention import wait_kind as attention_kind
    from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState

    cases = [
        WaitState(reason=WaitReason.USER, wait_key="tool_approval:r:act:k", details={"mode": "approval_required", "tool_calls": []}),
        WaitState(reason=WaitReason.USER, wait_key="user:r:n", prompt="Name?"),
        WaitState(reason=WaitReason.USER, wait_key="pause:r"),
        WaitState(reason=WaitReason.USER, wait_key="x", details={"kind": "pause"}),
        WaitState(reason=WaitReason.EVENT, wait_key="automation:a:wake"),
        WaitState(reason=WaitReason.EVENT, wait_key="evt:run:r:go", prompt="Go?"),
        WaitState(reason=WaitReason.EVENT, wait_key="evt:run:r:go"),
    ]
    for w in cases:
        run = RunState(run_id="r", workflow_id="wf", status=RunStatus.WAITING, current_node="n", waiting=w)
        theirs = attention_kind(run)
        ours = wait_kind({"reason": w.reason.value, "wait_key": w.wait_key, "prompt": w.prompt, "choices": w.choices,
                          "details": w.details}, "", "r")
        assert (ours if ours in PERSON_FACING_KINDS else None) == theirs, (w, ours, theirs)
