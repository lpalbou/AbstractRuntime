from __future__ import annotations

from typing import Any

from abstractruntime import Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_tick_completion_appends_completion_record_with_output() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

    vars: dict[str, Any] = {"_temp": {}, "_limits": {}, "_runtime": {}}

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"success": True, "result": {"x": 1}})

    wf = WorkflowSpec(workflow_id="wf_completion_output_ledger", entry_node="done", nodes={"done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)

    st = runtime.tick(workflow=wf, run_id=run_id, max_steps=10)
    assert st.status == RunStatus.COMPLETED

    ledger = runtime.get_ledger(run_id)
    assert isinstance(ledger, list)
    assert ledger, "expected at least one ledger record"
    last = ledger[-1]
    assert isinstance(last, dict)
    assert last.get("status") == "completed"
    result = last.get("result")
    assert isinstance(result, dict)
    assert result.get("completed") is True
    assert result.get("output") == {"success": True, "result": {"x": 1}}


