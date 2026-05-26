from __future__ import annotations

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.runtime import EffectOutcome
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_llm_effect_trace_metadata_includes_runtime_step_identity() -> None:
    seen_payloads = []

    def call_node(_run, _ctx):
        return StepPlan(
            node_id="call",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": "hello", "params": {}},
                result_key="_temp.effects.call",
            ),
        )

    def llm_handler(_run, effect, _default_next_node):
        seen_payloads.append(effect.payload)
        return EffectOutcome.completed({"content": "ok"})

    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )
    workflow = WorkflowSpec(workflow_id="wf-step-trace", entry_node="call", nodes={"call": call_node})

    run_id = runtime.start(workflow=workflow)
    runtime.tick(workflow=workflow, run_id=run_id)

    started = [r for r in ledger_store.list(run_id) if r.get("status") == "started"]
    assert len(started) == 1
    step_id = started[0]["step_id"]
    idempotency_key = started[0]["idempotency_key"]
    trace = started[0]["effect"]["payload"]["params"]["trace_metadata"]
    assert trace["step_id"] == step_id
    assert trace["effect_idempotency_key"] == idempotency_key
    assert trace["attempt"] == 1
    assert seen_payloads[0]["params"]["trace_metadata"]["step_id"] == step_id
