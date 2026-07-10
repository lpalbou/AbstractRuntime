"""Fair scheduling across N run stores (plan item 11, R3 — 0018 spec pins).

Pins: union due-scan across stores with store+channel stamping (channel
read from the door's stamp, never invented); default due-order; a custom
admission hook ordering by channel; the MECHANICAL starvation floor
(promotes over policy order AND restores policy-dropped starved work);
per-run failure isolation (one failing home never stalls the sweep); and
the eager deadline sweep (a parked EVENT wait with a deadline fires
without a client touch — the D3 gap the request-driven door has).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.scheduler.multi_store import (
    DueOrderAdmission,
    MultiStoreScheduler,
    TickCandidate,
    TickSource,
    apply_starvation_floor,
    stamped_channel,
)
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


# Parks use a SHORT FUTURE deadline: the first tick must genuinely park
# (an already-past deadline auto-unblocks within the same tick), and the
# test sleeps past it so the SWEEP is what wakes the run.
_PARK_S = 0.05


def _sleep_past_deadline() -> None:
    time.sleep(_PARK_S + 0.1)


def _wait_until_workflow(wf_id: str = "wf-until") -> WorkflowSpec:
    """Park on WAIT_UNTIL (due shortly), then complete."""

    def park(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="park",
            effect=Effect(type=EffectType.WAIT_UNTIL, payload={
                "until": _iso(datetime.now(timezone.utc) + timedelta(seconds=_PARK_S)),
            }, result_key="_temp.wake"),
            next_node="done",
        )

    def done(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"ok": True})

    return WorkflowSpec(workflow_id=wf_id, entry_node="park", nodes={"park": park, "done": done})


def _event_deadline_workflow(wf_id: str = "wf-event") -> WorkflowSpec:
    """Park on WAIT_EVENT with a shortly-due deadline (the D3 shape)."""

    def park(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="park",
            effect=Effect(type=EffectType.WAIT_EVENT, payload={
                "wait_key": "visitor_input",
                "until": _iso(datetime.now(timezone.utc) + timedelta(seconds=_PARK_S)),
            }, result_key="_temp.event"),
            next_node="done",
        )

    def done(run: Any, ctx: Any) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"woke": run.vars.get("_temp", {}).get("event")})

    return WorkflowSpec(workflow_id=wf_id, entry_node="park", nodes={"park": park, "done": done})


def _runtime() -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())


def _park_run(rt: Runtime, wf: WorkflowSpec, *, channel: str = "") -> str:
    vars_: Dict[str, Any] = {}
    if channel:
        vars_ = {"_runtime": {"entity": {"channel": channel}}}
    run_id = rt.start(workflow=wf, vars=vars_)
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.WAITING
    return run_id


def _source(name: str, rt: Runtime, wf: WorkflowSpec) -> TickSource:
    return TickSource(name=name, runtime=rt, resolve_workflow=lambda run: wf)


# ------------------------------------------------------------- candidates


def test_candidates_are_stamped_with_store_and_channel() -> None:
    rt = _runtime()
    wf = _wait_until_workflow()
    stamped = _park_run(rt, wf, channel="operator-visit")
    unstamped = _park_run(rt, wf)

    src = _source("castor", rt, wf)
    _sleep_past_deadline()
    cands = src.due_candidates(now_iso=_iso(datetime.now(timezone.utc)))
    by_id = {c.run_id: c for c in cands}
    assert by_id[stamped].store == "castor"
    assert by_id[stamped].channel == "operator-visit"  # read from the stamp
    assert by_id[unstamped].channel == "unstamped"  # labeled, never guessed
    assert all(c.wait_kind == "until" for c in cands)


def test_event_deadline_waits_are_schedulable_work() -> None:
    """The eager D3 sweep: a parked visit's idle deadline fires from the
    sweep alone — no client touch."""
    rt = _runtime()
    wf = _event_deadline_workflow()
    run_id = _park_run(rt, wf)

    sched = MultiStoreScheduler()
    sched.register(_source("home", rt, wf))
    _sleep_past_deadline()
    ticked = sched.sweep_once()
    assert ticked == 1
    state = rt.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    assert (state.output or {}).get("woke", {}).get("timed_out") is True


# ---------------------------------------------------------------- ordering


def test_sweep_unions_stores_and_ticks_each_through_its_own_runtime() -> None:
    rt_a, rt_b = _runtime(), _runtime()
    wf = _wait_until_workflow()
    run_a = _park_run(rt_a, wf)
    run_b = _park_run(rt_b, wf)

    sched = MultiStoreScheduler()
    sched.register(_source("a", rt_a, wf))
    sched.register(_source("b", rt_b, wf))
    _sleep_past_deadline()
    assert sched.sweep_once() == 2
    assert rt_a.get_state(run_a).status == RunStatus.COMPLETED
    assert rt_b.get_state(run_b).status == RunStatus.COMPLETED
    assert sched.stats.per_store["a"].ticked == 1
    assert sched.stats.per_store["b"].ticked == 1


def test_admission_hook_orders_and_floor_restores_dropped_starved() -> None:
    now = datetime.now(timezone.utc)

    def cand(run_id: str, channel: str, waited_s: float) -> TickCandidate:
        return TickCandidate(
            store="s", run_id=run_id, workflow_id="wf", wait_kind="until",
            due_at=_iso(now), channel=channel,
            waiting_since=_iso(now - timedelta(seconds=waited_s)),
        )

    visit = cand("r-visit", "operator-visit", waited_s=1)
    fresh_loop = cand("r-loop", "unstamped", waited_s=1)
    starved_dream = cand("r-dream", "unstamped", waited_s=9999)

    class VisitsFirstDropRest:
        def admit(self, candidates: List[TickCandidate]) -> List[TickCandidate]:
            return [c for c in candidates if c.channel == "operator-visit"]

    ordered = VisitsFirstDropRest().admit([visit, fresh_loop, starved_dream])
    floored = apply_starvation_floor(
        ordered, [visit, fresh_loop, starved_dream], max_starvation_s=300.0, now=now
    )
    # The starved dream is RESTORED and PROMOTED ahead of policy order;
    # the fresh loop stays deferred (policy may defer, never bury).
    assert [c.run_id for c in floored] == ["r-dream", "r-visit"]


def test_starvation_floor_promotes_over_policy_order() -> None:
    now = datetime.now(timezone.utc)

    def cand(run_id: str, waited_s: float) -> TickCandidate:
        return TickCandidate(
            store="s", run_id=run_id, workflow_id="wf", wait_kind="until",
            due_at=_iso(now), channel="unstamped",
            waiting_since=_iso(now - timedelta(seconds=waited_s)),
        )

    fresh = cand("r-fresh", 1)
    old = cand("r-old", 400)
    older = cand("r-older", 800)
    # Policy puts the fresh one first; the floor promotes the starved,
    # oldest first, and keeps the policy's order for the rest.
    floored = apply_starvation_floor(
        [fresh, old, older], [fresh, old, older], max_starvation_s=300.0, now=now
    )
    assert [c.run_id for c in floored] == ["r-older", "r-old", "r-fresh"]


def test_broken_admission_hook_degrades_to_due_order_loudly() -> None:
    rt = _runtime()
    wf = _wait_until_workflow()
    run_id = _park_run(rt, wf)

    class Broken:
        def admit(self, candidates: List[TickCandidate]) -> List[TickCandidate]:
            raise RuntimeError("policy bug")

    sched = MultiStoreScheduler(admission=Broken())
    sched.register(_source("s", rt, wf))
    _sleep_past_deadline()
    assert sched.sweep_once() == 1
    assert rt.get_state(run_id).status == RunStatus.COMPLETED
    assert any("#FALLBACK" in e for e in sched.stats.errors)


# ------------------------------------------------------------- resilience


def test_one_failing_run_never_stalls_the_sweep() -> None:
    rt_ok, rt_bad = _runtime(), _runtime()
    wf = _wait_until_workflow()
    ok_run = _park_run(rt_ok, wf)
    _park_run(rt_bad, wf)

    def broken_resolver(run: Any) -> WorkflowSpec:
        raise RuntimeError("spec rebuild failed for this home")

    sched = MultiStoreScheduler()
    sched.register(TickSource(name="bad", runtime=rt_bad, resolve_workflow=broken_resolver))
    sched.register(_source("ok", rt_ok, wf))
    _sleep_past_deadline()
    ticked = sched.sweep_once()
    assert ticked == 1  # the healthy store's run completed
    assert rt_ok.get_state(ok_run).status == RunStatus.COMPLETED
    assert sched.stats.per_store["bad"].failures == 1
    assert sched.stats.per_store["ok"].ticked == 1


def test_duplicate_source_names_refuse_loudly() -> None:
    rt = _runtime()
    wf = _wait_until_workflow()
    sched = MultiStoreScheduler()
    sched.register(_source("castor", rt, wf))
    try:
        sched.register(_source("castor", rt, wf))
        raise AssertionError("duplicate registration must refuse")
    except ValueError as e:
        assert "already registered" in str(e)


def test_custom_ticker_runs_after_admission() -> None:
    """The admission-ticket-BEFORE-lease pin: per-store tickers (the
    door's lease-acquiring drive) execute strictly after ordering."""
    rt = _runtime()
    wf = _wait_until_workflow()
    run_id = _park_run(rt, wf)
    order: List[str] = []

    class RecordingAdmission(DueOrderAdmission):
        def admit(self, candidates: List[TickCandidate]) -> List[TickCandidate]:
            order.append("admit")
            return super().admit(candidates)

    def ticker(run: Any, workflow: WorkflowSpec) -> Any:
        order.append(f"tick:{run.run_id}")
        return rt.tick(workflow=workflow, run_id=run.run_id)

    sched = MultiStoreScheduler(admission=RecordingAdmission())
    sched.register(TickSource(name="s", runtime=rt, resolve_workflow=lambda r: wf, ticker=ticker))
    _sleep_past_deadline()
    assert sched.sweep_once() == 1
    assert order == ["admit", f"tick:{run_id}"]


def test_background_loop_sweeps_and_stops() -> None:
    rt = _runtime()
    wf = _wait_until_workflow()
    run_id = _park_run(rt, wf)
    sched = MultiStoreScheduler(poll_interval_s=0.02)
    sched.register(_source("s", rt, wf))
    sched.start()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if rt.get_state(run_id).status == RunStatus.COMPLETED:
                break
            time.sleep(0.02)
        assert rt.get_state(run_id).status == RunStatus.COMPLETED
    finally:
        sched.stop()
    assert sched.stats.sweeps >= 1
