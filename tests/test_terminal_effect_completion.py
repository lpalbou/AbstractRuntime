from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_terminal_effect_node_completes_run_with_output() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    runtime.set_artifact_store(InMemoryArtifactStore())

    vars: dict[str, Any] = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {},
        "_runtime": {"memory_spans": []},
        "_temp": {},
        "_limits": {},
    }

    def note_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="note",
            effect=Effect(
                type=EffectType.MEMORY_NOTE,
                payload={"note": "Remember: terminal effects should be allowed."},
                result_key="_temp.note",
            ),
            next_node=None,  # terminal on purpose
        )

    wf = WorkflowSpec(workflow_id="wf_terminal_effect", entry_node="note", nodes={"note": note_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output.get("success") is True
    assert isinstance(state.output.get("result"), dict)

    temp = state.vars.get("_temp")
    assert isinstance(temp, dict)
    assert temp.get("note") == state.output.get("result")

