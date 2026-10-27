"""The conformance cases ship with the package (hosts and clients run them without this repo)."""
from __future__ import annotations

from abstractruntime.replay import replay
from abstractruntime.replay.conformance import case_names, load_case

REQUIRED = {
    "a_delegated_child_approval",       # the 2026-09-29 incident
    "b_timer_without_resume",           # the 2026-10-04 timer wait
    "c_parallel_tool_batch",
    "d_cancellation",
    "e_structured_output_agents",       # the 2026-10-11 executions
    "f_windowed_slim",
    "g_ask_user",
    "h_denied_approval",
}


def test_every_required_case_is_shipped_and_reproduces_its_view() -> None:
    names = set(case_names())
    assert REQUIRED <= names, sorted(REQUIRED - names)
    for name in sorted(names):
        case = load_case(name)
        view = replay(case["records"], case["runs"], root_run_id=case["manifest"]["root_run_id"]).to_json()
        assert view == case["expected_view"], name
