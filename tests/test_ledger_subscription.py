from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime import InMemoryLedgerStore, InMemoryRunStore, RunState, RunStatus, Runtime, WorkflowSpec
from abstractruntime.core.models import Effect, EffectType, StepPlan
from abstractruntime.storage.observable import ObservableLedgerStore


def test_runtime_can_subscribe_to_ledger_appends() -> None:
    ledger = ObservableLedgerStore(InMemoryLedgerStore())
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=ledger)

    def start(run: RunState, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="start",
            effect=Effect(type=EffectType.ANSWER_USER, payload={"message": "hi"}),
            next_node="done",
        )

    def done(run: RunState, ctx) -> StepPlan:
        del run, ctx
        return StepPlan(node_id="done", complete_output={"ok": True})

    workflow = WorkflowSpec(workflow_id="ledger_sub", entry_node="start", nodes={"start": start, "done": done})
    run_id = runtime.start(workflow=workflow, vars={})

    records: List[Dict[str, Any]] = []
    unsubscribe = runtime.subscribe_ledger(lambda r: records.append(r), run_id=run_id)

    state = runtime.tick(workflow=workflow, run_id=run_id)
    unsubscribe()

    assert state.status == RunStatus.COMPLETED
    assert any(r.get("node_id") == "start" and r.get("status") == "started" for r in records)
    assert any(r.get("node_id") == "start" and r.get("status") == "completed" for r in records)


def test_abstractcore_event_bus_bridge_emits_workflow_step_events() -> None:
    from abstractcore.events import EventType, GlobalEventBus

    from abstractruntime.integrations.abstractcore.observability import attach_global_event_bus_bridge

    GlobalEventBus.clear()

    ledger = ObservableLedgerStore(InMemoryLedgerStore())
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=ledger)

    def start(run: RunState, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="start",
            effect=Effect(type=EffectType.ANSWER_USER, payload={"message": "hi"}),
            next_node="done",
        )

    def done(run: RunState, ctx) -> StepPlan:
        del run, ctx
        return StepPlan(node_id="done", complete_output={"ok": True})

    workflow = WorkflowSpec(workflow_id="ledger_bridge", entry_node="start", nodes={"start": start, "done": done})
    run_id = runtime.start(workflow=workflow, vars={})

    events = []

    def _capture(event) -> None:
        events.append(event)

    GlobalEventBus.on(EventType.WORKFLOW_STEP_STARTED, _capture)
    GlobalEventBus.on(EventType.WORKFLOW_STEP_COMPLETED, _capture)
    GlobalEventBus.on(EventType.WORKFLOW_STEP_WAITING, _capture)
    GlobalEventBus.on(EventType.WORKFLOW_STEP_FAILED, _capture)

    unsubscribe = attach_global_event_bus_bridge(runtime=runtime, run_id=run_id)
    state = runtime.tick(workflow=workflow, run_id=run_id)
    unsubscribe()

    assert state.status == RunStatus.COMPLETED

    started = [e for e in events if e.type == EventType.WORKFLOW_STEP_STARTED and e.data.get("node_id") == "start"]
    completed = [e for e in events if e.type == EventType.WORKFLOW_STEP_COMPLETED and e.data.get("node_id") == "start"]
    assert started
    assert completed

    GlobalEventBus.clear()

