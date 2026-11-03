"""A host that stops ON PURPOSE interrupts an effect; it never fails it.

Incident 2026-11-02 (gateway backlog 1103): a polite gateway restart stopped the model host while an
automation's model call was in flight; the call failed (424), the run was marked failed, the
occurrence used a retry and an `automation_failed` email went out. The operator's contract: a
gateway stop or restart suspends the run at its last durable step and the run continues when the
gateway is back; the interrupted model call is sent again, uses no retry and notifies nobody.

The runtime half (`core.effect_cancellation.EffectInterrupted`): no terminal record, the run stays
RUNNING, and the next tick (on ANY Runtime over the same stores, as after a restart) sends the effect
again under the same idempotency key as attempt 1.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from abstractruntime import (
    Effect,
    EffectInterrupted,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    Runtime,
    RunStatus,
    StepPlan,
    WorkflowSpec,
)
from abstractruntime.core.effect_cancellation import inflight_effects
from abstractruntime.core.progress_channel import current_effect_delta_callback
from abstractruntime.core.runtime import EffectOutcome


def _workflow() -> WorkflowSpec:
    def reason(run, ctx):
        return StepPlan(
            node_id="reason",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "digest"}, result_key="llm"),
            next_node="act",
        )

    def act(run, ctx):
        return StepPlan(
            node_id="act",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [{"name": "t", "arguments": {}}]},
                          result_key="tools"),
            next_node="done",
        )

    def done(run, ctx):
        return StepPlan(node_id="done", complete_output={"llm": run.vars.get("llm")})

    return WorkflowSpec(workflow_id="agent", entry_node="reason", nodes={"reason": reason, "act": act, "done": done})


class _Model:
    """An LLM_CALL handler whose host stops during the first call. The handler wraps the call in
    an `except Exception` (as providers and effect handlers do): the interruption must pass it."""

    def __init__(self) -> None:
        self.calls = 0
        self.host_stopping = True

    def __call__(self, run, effect, default_next_node):
        self.calls += 1
        try:
            if self.host_stopping:
                cb = current_effect_delta_callback()
                if cb is not None:
                    cb("partial dig", "content")
                raise EffectInterrupted()
            return EffectOutcome.completed({"content": "The digest."})
        except Exception as e:  # a swallower on the way: must not catch it
            return EffectOutcome.failed(f"swallowed: {e}")


def _runtime(model, tools, run_store, ledger) -> Runtime:
    def tool_handler(run, effect, default_next_node):
        tools.append(effect.payload)
        return EffectOutcome.completed({"results": []})

    return Runtime(run_store=run_store, ledger_store=ledger,
                   effect_handlers={EffectType.LLM_CALL: model, EffectType.TOOL_CALLS: tool_handler})


def _llm(ledger, run_id) -> List[Dict[str, Any]]:
    return [r for r in ledger.list(run_id) if (r.get("effect") or {}).get("type") == "llm_call"]


def test_an_interrupted_effect_leaves_no_terminal_record_and_the_run_stays_running() -> None:
    model, tools = _Model(), []
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = _runtime(model, tools, run_store, ledger)
    wf = _workflow()
    run_id = runtime.start(workflow=wf, vars={})

    with pytest.raises(EffectInterrupted):
        runtime.tick(workflow=wf, run_id=run_id)

    state = runtime.get_state(run_id)
    assert state.status == RunStatus.RUNNING and state.current_node == "reason" and not state.error
    assert [r["status"] for r in _llm(ledger, run_id)] == ["started"], "an interruption wrote a terminal record"
    assert tools == [], "a step after the interrupted one ran"
    assert inflight_effects([run_id]) == []


def test_the_next_start_sends_it_again_as_attempt_1_and_the_run_completes() -> None:
    model, tools = _Model(), []
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    wf = _workflow()
    first = _runtime(model, tools, run_store, ledger)
    run_id = first.start(workflow=wf, vars={})
    with pytest.raises(EffectInterrupted):
        first.tick(workflow=wf, run_id=run_id)

    # The host is back: a NEW Runtime over the same stores, as after a gateway restart.
    model.host_stopping = False
    second = _runtime(model, tools, run_store, ledger)
    state = second.tick(workflow=wf, run_id=run_id)

    assert state.status == RunStatus.COMPLETED, state.error
    assert state.output and state.output.get("llm", {}).get("content") == "The digest."
    records = _llm(ledger, run_id)
    assert [r["status"] for r in records] == ["started", "started", "completed"]
    assert [r["attempt"] for r in records] == [1, 1, 1], "the interruption consumed a retry attempt"
    assert len({r["idempotency_key"] for r in records}) == 1, "the re-sent call is not the same call"
    assert not any(r["status"] == "failed" for r in ledger.list(run_id))
    assert model.calls == 2 and len(tools) == 1


def test_the_live_text_of_the_interrupted_call_ends_cancelled_interrupted() -> None:
    events: List[Dict[str, Any]] = []
    model = _Model()
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = _runtime(model, [], run_store, ledger)
    runtime.set_live_delta_sink(events.append)
    wf = _workflow()
    run_id = runtime.start(workflow=wf, vars={"_runtime": {"stream": True}})
    with pytest.raises(EffectInterrupted):
        runtime.tick(workflow=wf, run_id=run_id)
    assert [e["kind"] for e in events] == ["llm.delta", "llm.delta_end"]
    assert events[-1]["reason"] == "cancelled" and events[-1]["detail"] == "interrupted"
