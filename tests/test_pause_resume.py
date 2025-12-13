import tempfile
from pathlib import Path

from abstractruntime import (
    ActorFingerprint,
    Effect,
    EffectType,
    Runtime,
    StepPlan,
    WorkflowSpec,
)
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore


def test_pause_and_resume_wait_event_in_memory():
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    rt = Runtime(run_store=run_store, ledger_store=ledger_store)

    actor = ActorFingerprint.from_metadata(kind="agent", display_name="test-agent")

    def start_node(run, ctx):
        return StepPlan(node_id="START", effect=None, next_node="WAIT")

    def wait_node(run, ctx):
        return StepPlan(
            node_id="WAIT",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={"wait_key": "evt:hello", "resume_to_node": "DONE"},
                result_key="event_payload",
            ),
            next_node="DONE",
        )

    def done_node(run, ctx):
        return StepPlan(node_id="DONE", complete_output={"ok": True, "payload": run.vars.get("event_payload")})

    wf = WorkflowSpec(
        workflow_id="wf_test",
        entry_node="START",
        nodes={"START": start_node, "WAIT": wait_node, "DONE": done_node},
    )

    run_id = rt.start(workflow=wf, vars={}, actor_id=actor.actor_id)

    run = rt.tick(workflow=wf, run_id=run_id)
    assert run.status.value == "waiting"
    assert run.waiting is not None
    assert run.waiting.wait_key == "evt:hello"

    run = rt.resume(workflow=wf, run_id=run_id, wait_key="evt:hello", payload={"answer": "world"})
    assert run.status.value == "completed"
    assert run.output is not None
    assert run.output["payload"] == {"answer": "world"}

    ledger = rt.get_ledger(run_id)
    assert len(ledger) >= 2  # started + waiting/completed entries


def test_file_persistence_pause_resume():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        run_store = JsonFileRunStore(base)
        ledger_store = JsonlLedgerStore(base)
        rt = Runtime(run_store=run_store, ledger_store=ledger_store)

        def wait_node(run, ctx):
            return StepPlan(
                node_id="WAIT",
                effect=Effect(
                    type=EffectType.ASK_USER,
                    payload={"prompt": "Choose", "choices": ["a", "b"], "resume_to_node": "DONE"},
                    result_key="user_input",
                ),
                next_node="DONE",
            )

        def done_node(run, ctx):
            return StepPlan(node_id="DONE", complete_output={"user": run.vars.get("user_input")})

        wf = WorkflowSpec(workflow_id="wf_file", entry_node="WAIT", nodes={"WAIT": wait_node, "DONE": done_node})

        run_id = rt.start(workflow=wf)
        run = rt.tick(workflow=wf, run_id=run_id)
        assert run.status.value == "waiting"

        # New runtime instance simulates restart
        rt2 = Runtime(run_store=JsonFileRunStore(base), ledger_store=JsonlLedgerStore(base))
        run2 = rt2.get_state(run_id)
        assert run2.status.value == "waiting"

        run3 = rt2.resume(workflow=wf, run_id=run_id, wait_key=run2.waiting.wait_key, payload={"choice": "a"})
        assert run3.status.value == "completed"
        assert run3.output == {"user": {"choice": "a"}}


