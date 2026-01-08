from __future__ import annotations

from abstractruntime import InMemoryLedgerStore, InMemoryRunStore, Runtime
from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan
from abstractruntime.core.spec import WorkflowSpec


def test_emit_event_succeeds_without_workflow_registry_when_no_listeners() -> None:
    def n1(run: RunState, ctx: object) -> StepPlan:
        del run, ctx
        return StepPlan(
            node_id="n1",
            effect=Effect(
                type=EffectType.EMIT_EVENT,
                payload={"name": "demo", "scope": "session", "payload": {"x": 1}},
                result_key="_temp.effects.n1",
            ),
            next_node="n2",
        )

    def n2(run: RunState, ctx: object) -> StepPlan:
        del run, ctx
        return StepPlan(node_id="n2", complete_output={"success": True, "result": {"ok": True}})

    spec = WorkflowSpec(workflow_id="wf_emit_event_no_registry", entry_node="n1", nodes={"n1": n1, "n2": n2})
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    run_id = rt.start(workflow=spec, vars={})
    run = rt.tick(workflow=spec, run_id=run_id, max_steps=50)

    assert run.status == "completed"
    assert isinstance(run.output, dict)
    assert run.output.get("result", {}).get("ok") is True


