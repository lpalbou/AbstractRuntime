"""`Runtime.tick(step_gate=...)` — the host pause seam (2026-09-05).

A host (the gateway's pause switch) needs to stop a run MID-TICK: a tick may
run up to `max_steps` steps and each may be a long LLM/tool call, so "stop
scheduling ticks" alone leaves a paused gateway executing for minutes. The
gate is consulted at every step boundary; when it answers False the tick
returns the persisted state, the run stays RUNNING, and a later tick (gate
open) picks up exactly where it stopped.
"""

from __future__ import annotations

from abstractruntime import Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _workflow(trace: list[str]) -> WorkflowSpec:
    def a(run, ctx):
        trace.append("A")
        return StepPlan(node_id="A", effect=None, next_node="B")

    def b(run, ctx):
        trace.append("B")
        return StepPlan(node_id="B", effect=None, next_node="C")

    def c(run, ctx):
        trace.append("C")
        return StepPlan(node_id="C", complete_output={"ok": True})

    return WorkflowSpec(workflow_id="wf_gate", entry_node="A", nodes={"A": a, "B": b, "C": c})


def _runtime() -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())


def test_tick_without_gate_is_unchanged() -> None:
    trace: list[str] = []
    rt = _runtime()
    wf = _workflow(trace)
    run_id = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status.value == "completed"
    assert trace == ["A", "B", "C"]


def test_closed_gate_executes_nothing_and_keeps_run_running() -> None:
    trace: list[str] = []
    rt = _runtime()
    wf = _workflow(trace)
    run_id = rt.start(workflow=wf, vars={})

    state = rt.tick(workflow=wf, run_id=run_id, step_gate=lambda: False)

    assert trace == []
    assert state.status.value == "running"
    assert state.current_node == "A"
    # Nothing was persisted behind the caller's back either.
    assert rt.get_state(run_id).current_node == "A"


def test_gate_closing_mid_tick_stops_at_a_step_boundary_and_resumes_later() -> None:
    trace: list[str] = []
    rt = _runtime()
    wf = _workflow(trace)
    run_id = rt.start(workflow=wf, vars={})

    calls = {"n": 0}

    def gate() -> bool:
        # Open for the first step only: A runs, then the tick must stop
        # BEFORE B — the transition A→B is already persisted.
        calls["n"] += 1
        return calls["n"] <= 1

    state = rt.tick(workflow=wf, run_id=run_id, step_gate=gate)
    assert trace == ["A"]
    assert state.status.value == "running"
    assert state.current_node == "B"
    assert rt.get_state(run_id).current_node == "B"

    # A later tick (gate open again) continues from B and completes.
    state = rt.tick(workflow=wf, run_id=run_id, step_gate=lambda: True)
    assert trace == ["A", "B", "C"]
    assert state.status.value == "completed"


def test_raising_gate_is_treated_as_open() -> None:
    trace: list[str] = []
    rt = _runtime()
    wf = _workflow(trace)
    run_id = rt.start(workflow=wf, vars={})

    def broken_gate() -> bool:
        raise RuntimeError("gate exploded")

    state = rt.tick(workflow=wf, run_id=run_id, step_gate=broken_gate)
    assert state.status.value == "completed"
    assert trace == ["A", "B", "C"]


def test_closed_gate_leaves_a_due_wait_until_run_untouched() -> None:
    """A due WAIT_UNTIL run must not be flipped to RUNNING (even in memory —
    file stores hand out aliased RunState objects) while the host gate is
    closed; an open gate resumes it exactly once."""
    from abstractruntime import Effect, EffectType

    trace: list[str] = []
    rt = _runtime()

    import datetime as _dt
    import time as _time

    until = (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=1.2)).isoformat()

    def wait_node(run, ctx):
        trace.append("WAIT")
        return StepPlan(
            node_id="WAIT",
            effect=Effect(type=EffectType.WAIT_UNTIL, payload={"until": until, "resume_to_node": "DONE"}),
            next_node="DONE",
        )

    def done_node(run, ctx):
        trace.append("DONE")
        return StepPlan(node_id="DONE", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="wf_gate_wait", entry_node="WAIT", nodes={"WAIT": wait_node, "DONE": done_node})
    run_id = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status.value == "waiting" and state.waiting is not None

    _time.sleep(1.5)  # the deadline is now DUE — an ungated tick would resume
    gated = rt.tick(workflow=wf, run_id=run_id, step_gate=lambda: False)
    assert gated.status.value == "waiting"
    assert gated.waiting is not None and gated.waiting.until == until
    assert rt.get_state(run_id).status.value == "waiting"

    done = rt.tick(workflow=wf, run_id=run_id, step_gate=lambda: True)
    assert done.status.value == "completed"
    assert trace == ["WAIT", "DONE"]
