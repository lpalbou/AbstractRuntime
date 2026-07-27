"""`_runtime.wait_until_streak` — the O(1) spin discriminator (gateway c4768).

Born from the stale-poller incident (c4757): 25 leaked status-poller runs
re-armed wait_until every ~4.5s for days; the worst single ledger held
86,924 records. The gateway's idle-poller reaper needs a POSITIVE spin
proof that is O(1) to read (deny-safe: absent key or low streak = never
reap) instead of scanning ledger tails.

Semantics pinned here, exactly as adopted on the thread:
- each wait_until PARK increments the streak by 1;
- ANY other effect dispatch resets it to 0 (a scheduler that wakes to
  emit_event / start_subworkflow / llm_call is effect-sparse, not idle);
- a wait_until RESUME (auto-unblock) leaves the streak untouched — the
  spinner's cycle is park -> due -> pure nodes -> park, and a reset on
  resume would oscillate the count 1 -> 0 and never accumulate;
- runs that never spin never grow the key (vars stay lean).
"""

import time
from datetime import datetime, timedelta, timezone

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _runtime() -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())


def _near_future_iso(seconds: float = 0.25) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _spinner_workflow(parks: int) -> WorkflowSpec:
    """A poller-shaped loop: WAIT parks on wait_until, resumes into a PURE
    node (no effect — the class the incident named), which loops back to
    WAIT until `parks` cycles have run, then completes."""

    def wait_node(run, ctx):
        return StepPlan(
            node_id="WAIT",
            effect=Effect(
                type=EffectType.WAIT_UNTIL,
                payload={"until": _near_future_iso(), "resume_to_node": "CHECK"},
            ),
            next_node="CHECK",
        )

    def check_node(run, ctx):
        done = int(run.vars.get("cycles", 0)) + 1
        run.vars["cycles"] = done
        if done >= parks:
            return StepPlan(node_id="CHECK", complete_output={"cycles": done})
        return StepPlan(node_id="CHECK", next_node="WAIT")

    return WorkflowSpec(
        workflow_id="wf_spinner",
        entry_node="WAIT",
        nodes={"WAIT": wait_node, "CHECK": check_node},
    )


def _streak(run) -> object:
    rt = run.vars.get("_runtime") or {}
    return rt.get("wait_until_streak") if isinstance(rt, dict) else None


class TestWaitUntilStreak:
    def test_parks_accumulate_and_pure_node_cycles_do_not_reset(self):
        rt = _runtime()
        wf = _spinner_workflow(parks=3)
        run_id = rt.start(workflow=wf, vars={})

        run = rt.tick(workflow=wf, run_id=run_id)
        assert run.status.value == "waiting"
        assert _streak(run) == 1

        # Second cycle: auto-unblock resumes into the PURE node, which loops
        # back to a fresh park. The resume must not reset; the park must
        # increment — this is the exact spin shape from the incident.
        time.sleep(0.35)
        run = rt.tick(workflow=wf, run_id=run_id)
        assert run.status.value == "waiting"
        assert _streak(run) == 2

        time.sleep(0.35)
        run = rt.tick(workflow=wf, run_id=run_id)
        assert run.status.value == "waiting"
        assert _streak(run) == 3

    def test_any_other_effect_resets_the_streak(self):
        """park -> resume -> emit_event (real work) -> park: the non-wait
        dispatch resets, so the later park reads 1, never 2."""

        def wait_node(run, ctx):
            return StepPlan(
                node_id="WAIT",
                effect=Effect(
                    type=EffectType.WAIT_UNTIL,
                    payload={"until": _near_future_iso(), "resume_to_node": "WORK"},
                ),
                next_node="WORK",
            )

        def work_node(run, ctx):
            return StepPlan(
                node_id="WORK",
                effect=Effect(type=EffectType.EMIT_EVENT, payload={"name": "beat", "payload": {}}),
                next_node="WAIT2",
            )

        def wait2_node(run, ctx):
            return StepPlan(
                node_id="WAIT2",
                effect=Effect(
                    type=EffectType.WAIT_UNTIL,
                    payload={"until": _near_future_iso(seconds=30.0), "resume_to_node": "DONE"},
                ),
                next_node="DONE",
            )

        def done_node(run, ctx):
            return StepPlan(node_id="DONE", complete_output={"ok": True})

        wf = WorkflowSpec(
            workflow_id="wf_worker",
            entry_node="WAIT",
            nodes={"WAIT": wait_node, "WORK": work_node, "WAIT2": wait2_node, "DONE": done_node},
        )
        rt = _runtime()
        run_id = rt.start(workflow=wf, vars={})

        run = rt.tick(workflow=wf, run_id=run_id)
        assert _streak(run) == 1

        time.sleep(0.35)
        run = rt.tick(workflow=wf, run_id=run_id)
        # emit_event dispatched (reset to 0), then WAIT2 parked (0 -> 1):
        # a worker doing real effects between parks never accumulates spin.
        assert run.status.value == "waiting"
        assert _streak(run) == 1

    def test_runs_that_never_park_never_grow_the_key(self):
        def work_node(run, ctx):
            return StepPlan(
                node_id="WORK",
                effect=Effect(type=EffectType.EMIT_EVENT, payload={"name": "beat", "payload": {}}),
                next_node="DONE",
            )

        def done_node(run, ctx):
            return StepPlan(node_id="DONE", complete_output={"ok": True})

        wf = WorkflowSpec(
            workflow_id="wf_no_spin",
            entry_node="WORK",
            nodes={"WORK": work_node, "DONE": done_node},
        )
        rt = _runtime()
        run_id = rt.start(workflow=wf, vars={})
        run = rt.tick(workflow=wf, run_id=run_id)
        assert run.status.value == "completed"
        rt_ns = run.vars.get("_runtime") or {}
        assert "wait_until_streak" not in rt_ns
