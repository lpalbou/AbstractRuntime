"""An occurrence whose agent completed with `success: false` names the agent's reason, never the
run's status word (2026-11-02: the attention item and the `automation_failed` email said "completed").

The Agent node's output carries its reason in `meta.error` (visualflow compiler); the occurrence
root's status is COMPLETED, so `child.status.value` is "completed" and is never a reason.
"""

from __future__ import annotations

import pytest

from abstractruntime.automations.attention import normalize_occurrence_output
from abstractruntime.core.models import RunState, RunStatus

AT = "2026-11-02T23:00:00+00:00"


def _root(output, status=RunStatus.COMPLETED, error=None) -> RunState:
    return RunState(run_id="occ", workflow_id="wf", status=status, current_node="end", vars={},
                    output=output, error=error, created_at=AT, updated_at=AT)


@pytest.mark.parametrize("lazy", [False, True])
def test_the_agents_meta_error_is_the_occurrence_error(lazy) -> None:
    out = {"response": "The agent failed before producing a final answer.", "success": False,
           "meta": {"schema": "abstractcode.agent.v1.meta", "error": "Effect failed: the provider refused the key"}}
    got = normalize_occurrence_output(_root(out), lazy=lazy)
    assert got["success"] is False
    assert got["error"] == "Effect failed: the provider refused the key"


@pytest.mark.parametrize("lazy", [False, True])
def test_no_reason_anywhere_is_said_plainly_not_as_completed(lazy) -> None:
    got = normalize_occurrence_output(_root({"response": "", "success": False}), lazy=lazy)
    assert got["success"] is False and got["error"] != "completed"
    assert got["error"] == "The run finished without success and gave no reason."


def test_a_failed_run_keeps_its_own_error() -> None:
    got = normalize_occurrence_output(_root(None, status=RunStatus.FAILED, error="boom"))
    assert got == {"answer": "", "success": False, "notify": None, "error": "boom"}
