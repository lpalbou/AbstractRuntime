"""The operator iterations ceiling (laurent c786: hard ceiling, default
100 gateway-side, operator-customizable; seam (b) ruled c792/c805).

Contract pinned here:
- The HOST serves `_limits.max_iterations_ceiling` into run vars; the
  runtime's `start()` is the ONE enforcement site for every lane.
- Workflow-declared max_iterations is authoritative UP TO the ceiling;
  declaring above it refuses LOUD at start (naming both values + the
  override surface) — the run never exists; never mid-run truncation.
- Absent ceiling = no enforcement (server-declared; the runtime never
  invents 100).
- A SILENT workflow under a ceiling below the default gets the default
  clamped down (a default is nobody's word).
- An explicit narrow value (incl. 0) at or under the ceiling is untouched
  (agent's explicit-narrow rule).
"""

from __future__ import annotations

import pytest

from abstractruntime.core.models import StepPlan
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _wf() -> WorkflowSpec:
    def done(run, ctx):
        return StepPlan(node_id="done", complete_output={"ok": True})

    return WorkflowSpec(workflow_id="wf-ceiling", entry_node="done", nodes={"done": done})


def _runtime() -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())


def test_declared_above_ceiling_refuses_loud_at_start() -> None:
    rt = _runtime()
    with pytest.raises(ValueError) as e:
        rt.start(
            workflow=_wf(),
            vars={"_limits": {"max_iterations": 200, "max_iterations_ceiling": 100}},
        )
    msg = str(e.value)
    assert "200" in msg and "100" in msg  # names both values
    assert "ceiling" in msg and "refusing at start" in msg


def test_declared_at_or_under_ceiling_is_untouched() -> None:
    rt = _runtime()
    run_id = rt.start(
        workflow=_wf(),
        vars={"_limits": {"max_iterations": 100, "max_iterations_ceiling": 100}},
    )
    limits = rt._run_store.load(run_id).vars["_limits"]
    assert limits["max_iterations"] == 100

    # Explicit narrow — including 0 — stays the workflow's word.
    run_id2 = rt.start(
        workflow=_wf(),
        vars={"_limits": {"max_iterations": 0, "max_iterations_ceiling": 100}},
    )
    assert rt._run_store.load(run_id2).vars["_limits"]["max_iterations"] == 0


def test_absent_ceiling_means_no_enforcement() -> None:
    rt = _runtime()
    run_id = rt.start(workflow=_wf(), vars={"_limits": {"max_iterations": 10_000}})
    assert rt._run_store.load(run_id).vars["_limits"]["max_iterations"] == 10_000


def test_silent_workflow_under_low_ceiling_clamps_the_default_down() -> None:
    """Ceiling 10 with no declared value: the effective default (20) would
    BREACH the operator's ceiling — the default clamps down (a default is
    nobody's word; clamping it is not an override)."""
    rt = _runtime()
    run_id = rt.start(
        workflow=_wf(), vars={"_limits": {"max_iterations_ceiling": 10}}
    )
    assert rt._run_store.load(run_id).vars["_limits"]["max_iterations"] == 10

    # And a HIGH ceiling leaves the default (20) alone.
    run_id2 = rt.start(
        workflow=_wf(), vars={"_limits": {"max_iterations_ceiling": 100}}
    )
    assert rt._run_store.load(run_id2).vars["_limits"]["max_iterations"] == 20
