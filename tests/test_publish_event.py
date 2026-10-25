"""Publish to channel (`EffectType.PUBLISH_EVENT`): the runtime validates and hands the payload to the
host's handler (`Runtime.set_publish_event_handler`, `handler(run, payload, step_id)`), which appends the
CloudEvents event (and the attention item when `alert`); the effect result is the handler's
`{event_id, sequence, alerted}`."""

from __future__ import annotations

import pytest

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.core.publish import PublishError, validate_publish
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _node(run, ctx):
    return StepPlan(node_id="pub", effect=Effect(type=EffectType.PUBLISH_EVENT, payload=dict(run.vars["publish"]), result_key="out"),
                    next_node="done")


def _done(run, ctx):
    return StepPlan(node_id="done", complete_output={"out": run.vars.get("out")})


FLOW = WorkflowSpec(workflow_id="pubflow", entry_node="pub", nodes={"pub": _node, "done": _done})


class Host:
    """The gateway's role (the contract of abstractgateway publish_host): idempotent on step_id."""

    def __init__(self):
        self.events, self.attention, self.calls = {}, [], []

    def __call__(self, run, payload, step_id):
        self.calls.append((run.run_id, dict(payload), step_id))
        if payload["channel"] == "attention":
            raise ValueError("The channel 'attention' is reserved.")
        event_id = f"publish:{run.run_id}:{step_id}"
        if event_id not in self.events:
            self.events[event_id] = {"type": payload.get("type") or f"{payload['channel']}.message", "channel": payload["channel"],
                                     "data": {**(payload.get("data") or {}), "title": payload["title"]}}
            if payload.get("alert"):
                self.attention.append({"kind": "notify", "record_key": event_id, "title": payload["title"]})
        return {"event_id": event_id, "sequence": len(self.events), "alerted": bool(payload.get("alert"))}


def _runtime(handler=None):
    registry = WorkflowRegistry()
    registry.register(FLOW)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=registry)
    if handler is not None:
        rt.set_publish_event_handler(handler)
    return rt


def _publish(rt, payload):
    run_id = rt.start(workflow=FLOW, vars={"publish": payload})
    return run_id, rt.tick(workflow=FLOW, run_id=run_id)


def test_publish_hands_the_payload_to_the_host_and_returns_its_receipt():
    host = Host()
    rt = _runtime(host)
    run_id, state = _publish(rt, {"channel": "kitchen", "type": "kitchen.order.ready", "title": "Order 12 ready",
                                  "body": "Table 4", "data": {"order": 12}, "subject": "order-12"})
    assert state.status == RunStatus.COMPLETED
    ((rid, payload, step_id),) = host.calls
    assert rid == run_id and payload == {"channel": "kitchen", "type": "kitchen.order.ready", "title": "Order 12 ready",
                                         "body": "Table 4", "subject": "order-12", "data": {"order": 12}}
    assert state.output["out"] == {"event_id": f"publish:{run_id}:{step_id}", "sequence": 1, "alerted": False}
    assert host.attention == []
    _, alerted = _publish(rt, {"channel": "ops", "title": "Backup failed", "alert": True})
    assert alerted.output["out"]["alerted"] is True and host.attention[0]["title"] == "Backup failed"
    assert host.calls[-1][1] == {"channel": "ops", "title": "Backup failed", "alert": True}  # type left to the host


def test_a_replayed_step_passes_the_same_step_id():
    host = Host()
    rt = _runtime(host)
    run_id = rt.start(workflow=FLOW, vars={"publish": {"channel": "kitchen", "title": "t"}})
    run = rt.run_store.load(run_id)
    effect = Effect(type=EffectType.PUBLISH_EVENT, payload={"channel": "kitchen", "title": "t"}, result_key="out")
    first = rt._handle_publish_event(run, effect, "done")
    second = rt._handle_publish_event(run, effect, "done")
    assert host.calls[0][2] == host.calls[1][2] and first.result["event_id"] == second.result["event_id"]
    assert len(host.events) == 1


def test_refusals():
    _, state = _publish(_runtime(), {"channel": "kitchen", "title": "t"})
    assert state.status == RunStatus.FAILED and "no publish_event handler" in str(state.error)
    for bad in ({"channel": "two words", "title": "t"}, {"channel": "k", "type": "has space", "title": "t"}, {"channel": "k"},
                {"channel": "k", "title": "t", "data": [1]}, {"channel": "k", "title": "t", "extra": 1}, {"title": "t"},
                {"channel": "k", "title": "t", "alert": "yes"}):
        with pytest.raises(PublishError):
            validate_publish(bad)
    _, state = _publish(_runtime(Host()), {"channel": "attention", "title": "t"})
    assert state.status == RunStatus.FAILED and "reserved" in str(state.error)


def test_publish_node_bakes_the_effect():
    from abstractruntime.visualflow_compiler.compiler import _create_entity_memory_base_handler

    handler = _create_entity_memory_base_handler(node_id="n", node_type="publish_event", next_node=None, input_key="in", output_key="out")

    class _Run:
        vars = {"in": {"channel": "kitchen", "type": "kitchen.order.ready", "alert": True, "title": "Ready", "effect": "schedule_run"}}

    plan = handler(_Run(), None)
    assert plan.effect.type == EffectType.PUBLISH_EVENT
    assert plan.effect.payload == {"channel": "kitchen", "type": "kitchen.order.ready", "alert": True, "title": "Ready"}
