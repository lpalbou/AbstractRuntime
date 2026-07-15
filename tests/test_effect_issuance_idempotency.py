"""Effect ISSUANCE idempotency (agent seat P0, commons c1568, 2026-07-13).

The old key was (run_id, node_id, effect_type, normalized_payload) and the
lookup scanned the whole run ledger — so a byte-identical TOOL_CALLS batch
re-issued at the same node (re-read a file after editing it, re-run the
test suite after a fix, any loop with repeated payloads) silently REPLAYED
the first stale result instead of executing. The fix hashes the run's
`_runtime.effect_seq` issuance counter into the key; the tick loop advances
it in the same save that lands each step.

Pins here:
- identical payload at the same node across two loop iterations EXECUTES
  TWICE (distinct results in the durable record);
- crash-replay (vars rolled back to the pre-step save, completed record in
  the ledger) still REUSES the completed result — exact at-most-once;
- retries within one issuance share one key (policy unchanged there).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    Runtime,
    RunState,
    StepPlan,
    WorkflowSpec,
)
from abstractruntime.core.runtime import EffectOutcome


def _two_identical_batches_workflow() -> WorkflowSpec:
    """LOOP node issues the SAME tool batch twice, then completes."""

    def loop_node(run: RunState, ctx: Any) -> StepPlan:
        n = int(run.vars.get("iterations", 0))
        if n >= 2:
            return StepPlan(node_id="LOOP", complete_output={"runs": n})
        run.vars["iterations"] = n + 1
        return StepPlan(
            node_id="LOOP",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={
                    "tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]
                },
                result_key=f"observed.{n}",
            ),
            next_node="LOOP",
        )

    return WorkflowSpec(workflow_id="issuance-test", entry_node="LOOP", nodes={"LOOP": loop_node})


def _counting_tool_handler():
    calls = {"n": 0}

    def handler(run: RunState, effect: Effect, ctx: Any) -> EffectOutcome:
        calls["n"] += 1
        return EffectOutcome.completed({"content": f"result-{calls['n']}"})

    return handler, calls


def test_identical_batch_at_same_node_executes_twice() -> None:
    handler, calls = _counting_tool_handler()
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: handler},
    )
    wf = _two_identical_batches_workflow()
    run_id = rt.start(workflow=wf, vars={})
    run = rt.tick(workflow=wf, run_id=run_id, max_steps=20)
    assert run.status.value == "completed"
    # Both issuances EXECUTED (no stale replay of the first result).
    assert calls["n"] == 2
    assert run.vars["observed"]["0"]["content"] == "result-1"
    assert run.vars["observed"]["1"]["content"] == "result-2"


def test_crash_replay_still_reuses_the_completed_result() -> None:
    """Simulate the crash window: the step's COMPLETED record is in the
    ledger but run vars rolled back to the pre-step save (counter not yet
    advanced). The replay must recompute the SAME key and reuse — the
    handler must NOT run again."""
    handler, calls = _counting_tool_handler()
    run_store = InMemoryRunStore()
    ledger = InMemoryLedgerStore()
    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger,
        effect_handlers={EffectType.TOOL_CALLS: handler},
    )

    def one_shot_node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("done"):
            return StepPlan(node_id="ONE", complete_output={"ok": True})
        run.vars["done"] = True
        return StepPlan(
            node_id="ONE",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "t", "arguments": {}}]},
                result_key="out",
            ),
            next_node="ONE",
        )

    wf = WorkflowSpec(workflow_id="crash-test", entry_node="ONE", nodes={"ONE": one_shot_node})
    run_id = rt.start(workflow=wf, vars={})
    # Snapshot the PRE-STEP state (what a crash-before-save rolls back to).
    import copy

    pre = copy.deepcopy(run_store.load(run_id))
    run = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert run.status.value == "completed"
    assert calls["n"] == 1

    # Crash simulation: restore pre-step vars/state; the ledger keeps the
    # completed record. Replay the tick — the recomputed key must MATCH and
    # the handler must not run a second time.
    from abstractruntime.core.models import RunStatus

    pre.status = RunStatus.RUNNING
    run_store.save(pre)
    replayed = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert replayed.status.value == "completed"
    assert calls["n"] == 1, "crash-replay must reuse, never re-execute"


def test_pause_mid_effect_then_restart_still_reuses_the_completed_result(tmp_path) -> None:
    """Replay adversary P1 (2026-07-14): file-backed run stores ALIAS loaded
    RunStates, so a control-plane save (pause) landing between the probe
    and the step's own save used to serialize an ALREADY-advanced issuance
    counter WITHOUT the step's result. After a restart, the resumed replay
    recomputed a DIFFERENT key, missed the completed record, and the
    effect executed a SECOND time. The advance now binds to the step's own
    saves: an out-of-band save carries the un-advanced counter and the
    replay reuses the completed result."""
    from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

    calls = {"n": 0}
    pause_during_effect: Dict[str, Any] = {"rt": None, "run_id": None}

    def handler(run: RunState, effect: Effect, ctx: Any) -> EffectOutcome:
        calls["n"] += 1
        # Simulate the operator pausing MID-EFFECT: a host-thread
        # control-plane save of the aliased RunState.
        rt = pause_during_effect["rt"]
        if rt is not None and calls["n"] == 1:
            rt.pause_run(run_id=pause_during_effect["run_id"], reason="operator")
        return EffectOutcome.completed({"content": f"result-{calls['n']}"})

    def one_shot_node(run: RunState, ctx: Any) -> StepPlan:
        # Plans from RESULT PRESENCE (no side flag): after the torn pause
        # save (no result on disk), the replay RE-ISSUES the byte-identical
        # effect — the shape where the old probe-time counter advance
        # produced a fresh key and double-executed.
        if run.vars.get("out"):
            return StepPlan(node_id="ONE", complete_output={"ok": True})
        return StepPlan(
            node_id="ONE",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "t", "arguments": {}}]},
                result_key="out",
            ),
            next_node="ONE",
        )

    wf = WorkflowSpec(workflow_id="pause-tear", entry_node="ONE", nodes={"ONE": one_shot_node})

    run_store = JsonFileRunStore(tmp_path)
    ledger = JsonlLedgerStore(tmp_path)
    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger,
        effect_handlers={EffectType.TOOL_CALLS: handler},
    )
    run_id = rt.start(workflow=wf, vars={})
    pause_during_effect["rt"] = rt
    pause_during_effect["run_id"] = run_id

    paused = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert paused.status.value != "completed"  # tick aborted on the pause
    assert calls["n"] == 1

    # RESTART: fresh stores over the same directory (cold caches — what a
    # process restart actually is), resume, tick again.
    run_store2 = JsonFileRunStore(tmp_path)
    ledger2 = JsonlLedgerStore(tmp_path)
    rt2 = Runtime(
        run_store=run_store2,
        ledger_store=ledger2,
        effect_handlers={EffectType.TOOL_CALLS: handler},
    )
    rt2.resume_run(run_id)
    replayed = rt2.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert replayed.status.value == "completed"
    assert calls["n"] == 1, "pause-mid-effect + restart must reuse, never re-execute"
    assert replayed.vars["out"]["content"] == "result-1"


def test_retries_within_one_issuance_share_one_key() -> None:
    """The retry loop precomputes the key once per issuance — unchanged by
    the counter (which advances per ISSUANCE, not per attempt)."""
    attempts = {"n": 0}

    def flaky(run: RunState, effect: Effect, ctx: Any) -> EffectOutcome:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return EffectOutcome.failed("transient")
        return EffectOutcome.completed({"content": "ok"})

    from abstractruntime.core.policy import RetryPolicy

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: flaky},
        effect_policy=RetryPolicy(tool_max_attempts=2, backoff_base=0.0, backoff_max=0.0),
    )

    def node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("done"):
            return StepPlan(node_id="N", complete_output={"ok": True})
        run.vars["done"] = True
        return StepPlan(
            node_id="N",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "t", "arguments": {}}]},
                result_key="out",
            ),
            next_node="N",
        )

    wf = WorkflowSpec(workflow_id="retry-test", entry_node="N", nodes={"N": node})
    run_id = rt.start(workflow=wf, vars={})
    run = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert run.status.value == "completed"
    assert attempts["n"] == 2  # one issuance, two attempts, one key
    assert run.vars["out"]["content"] == "ok"
