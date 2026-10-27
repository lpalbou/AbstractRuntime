"""A session tool policy's decisions are activities, never interactions (contract v1, section 5).

When a conversation's policy approves or refuses a tool call, the runtime writes one
`abstract.tool_policy_decision` record and NO wait: every client must show "Allowed (or denied)
by this conversation's policy", never an approval gate that flashes and disappears.
"""
from __future__ import annotations

from abstractruntime.replay import replay
from abstractruntime.replay.conformance import load_case

CASE = "m_session_tool_policy"


def _case():
    case = load_case(CASE)
    root = case["manifest"]["root_run_id"]
    child = next(r for r in case["records"] if r != root)
    return case, root, child


def _is_policy(rec) -> bool:
    eff = rec.get("effect") or {}
    return eff.get("type") == "emit_event" and (eff.get("payload") or {}).get("name") == "abstract.tool_policy_decision"


def test_decisions_are_activities_and_never_the_interaction_at_any_point() -> None:
    case, root, child = _case()
    child_recs = case["records"][child]
    # Every prefix of the child's ledger (the run as a client sees it live): never an interaction,
    # never a person-facing wait.
    for cut in range(1, len(child_recs) + 1):
        records = {root: case["records"][root][:2], child: child_recs[:cut]}
        snaps = {rid: {**s, "status": "running"} for rid, s in case["runs"].items()}
        v = replay(records, snaps, root_run_id=root).to_json()
        assert v["interaction"] is None, cut
        assert not [w for w in v["open_waits"] if w["person_facing"]], cut
    v = replay(case["records"], case["runs"], root_run_id=root).to_json()
    policy = [a for a in v["activities"] if a["kind"] == "tool_policy"]
    expected = [r for r in child_recs if _is_policy(r)]
    assert len(policy) == len(expected) == 2
    by_decision = {a["decision"]: a for a in policy}
    assert by_decision["auto_approved"]["name"] == "Allowed by this conversation's policy"
    assert by_decision["denied"]["name"] == "Denied by this conversation's policy"
    assert by_decision["auto_approved"]["policy"]["revision"] == 5
    assert by_decision["auto_approved"]["policy"]["set_by"]["user_id"] == "person-a"


def test_the_tool_rows_say_who_decided_and_a_policy_denial_is_denied_not_failed() -> None:
    case, root, child = _case()
    v = replay(case["records"], case["runs"], root_run_id=root).to_json()
    tool = next(a for a in v["activities"] if a["kind"] == "tool")
    rows = {r["name"]: r for r in tool["tool_calls"]}
    assert rows["fetch_url"]["status"] == "completed" and rows["fetch_url"]["decided_by"] == "session_policy"
    assert rows["execute_command"]["status"] == "denied" and rows["execute_command"]["decided_by"] == "session_policy"


def test_a_decision_folded_after_its_tool_row_still_marks_it() -> None:
    case, root, child = _case()
    recs = list(case["records"][child])
    decisions = [r for r in recs if _is_policy(r)]
    rest = [r for r in recs if not _is_policy(r)]
    # The decision records written after the tool's completion (renumbered in that order).
    moved = [dict(r, seq=i + 1) for i, r in enumerate(rest[:-1] + decisions + rest[-1:])]
    v = replay({root: case["records"][root], child: moved}, case["runs"], root_run_id=root).to_json()
    tool = next(a for a in v["activities"] if a["kind"] == "tool")
    assert {r["name"]: r.get("decided_by") for r in tool["tool_calls"]} == {
        "fetch_url": "session_policy", "execute_command": "session_policy"}
