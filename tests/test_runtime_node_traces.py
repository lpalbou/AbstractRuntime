from __future__ import annotations

from typing import Optional

from abstractruntime import InMemoryLedgerStore, InMemoryRunStore, RunState, RunStatus, Runtime, WorkflowSpec
from abstractruntime.core.models import Effect, EffectType, StepPlan
from abstractruntime.core.runtime import EffectOutcome


def test_runtime_records_per_node_traces_for_effect_steps() -> None:
    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, effect, default_next_node
        return EffectOutcome.completed({"content": "ok"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )

    def start(run: RunState, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="start",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi"}, result_key="_temp.llm"),
            next_node="done",
        )

    def done(run: RunState, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="done", complete_output={"success": True, "result": run.vars.get("_temp", {}).get("llm")})

    workflow = WorkflowSpec(workflow_id="trace_test", entry_node="start", nodes={"start": start, "done": done})

    run_id = runtime.start(workflow=workflow, vars={})
    state = runtime.tick(workflow=workflow, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    runtime_ns = state.vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    traces = runtime_ns.get("node_traces")
    assert isinstance(traces, dict)
    node_trace = traces.get("start")
    assert isinstance(node_trace, dict)
    assert node_trace.get("node_id") == "start"
    steps = node_trace.get("steps")
    assert isinstance(steps, list)
    assert len(steps) == 1
    entry = steps[0]
    assert entry.get("node_id") == "start"
    assert entry.get("status") == "completed"
    effect_dict = entry.get("effect")
    assert isinstance(effect_dict, dict)
    assert effect_dict.get("type") == "llm_call"
    result = entry.get("result")
    assert isinstance(result, dict)
    assert result.get("content") == "ok"

    # Public Runtime APIs should expose the same trace information.
    traces_via_api = runtime.get_node_traces(run_id)
    assert isinstance(traces_via_api, dict)
    assert "start" in traces_via_api
    trace_via_api = runtime.get_node_trace(run_id, "start")
    assert isinstance(trace_via_api, dict)
    assert trace_via_api.get("node_id") == "start"
