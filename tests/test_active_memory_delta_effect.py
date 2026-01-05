from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_active_memory_delta_effect_updates_runtime_active_memory() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

    vars: dict[str, Any] = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {},
        "_runtime": {"memory_spans": []},
        "_temp": {},
        "_limits": {},
    }

    delta = {
        "current_tasks": {"upsert": [{"task_id": "t_1", "title": "Do the thing", "status": "doing"}]},
        "key_history": {"add": [{"kind": "event", "summary": "Started task t_1"}]},
    }

    def apply_delta_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="delta",
            effect=Effect(
                type=EffectType.ACTIVE_MEMORY_DELTA,
                payload={"tool_name": "active_memory_delta", "call_id": "mem_1", "delta": delta},
                result_key="_temp.delta_result",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"result": run.vars.get("_temp", {}).get("delta_result")})

    wf = WorkflowSpec(
        workflow_id="wf_active_memory_delta",
        entry_node="delta",
        nodes={"delta": apply_delta_node, "done": done_node},
    )

    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.COMPLETED

    runtime_ns = state.vars.get("_runtime") if isinstance(state.vars.get("_runtime"), dict) else {}
    mem = runtime_ns.get("active_memory") if isinstance(runtime_ns.get("active_memory"), dict) else {}

    tasks = mem.get("tasks")
    assert isinstance(tasks, list)
    assert any(isinstance(t, dict) and t.get("task_id") == "t_1" for t in tasks)

    history = mem.get("key_history")
    assert isinstance(history, list)
    assert any(isinstance(h, dict) and "Started task t_1" in str(h.get("summary") or "") for h in history)



