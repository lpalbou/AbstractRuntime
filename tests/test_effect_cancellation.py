"""A cancel must reach the effect that is ALREADY RUNNING (Stop must stop the model).

Incident 2026-09-22: the gateway applied a cancel command within seconds
(`cancel_run` → CANCELLED) and the model kept decoding for an hour, because
`cancel_run` only wrote run state and the tick thread blocked inside the
LLM_CALL handler was never told. These pin the runtime half of the fix
(`core/effect_cancellation.py`):

* the cancel event travels OUT OF BAND (never in `effect.payload`, which stays
  JSON) and is set by `cancel_run` on ANY Runtime object in the process — the
  gateway builds a fresh one per command;
* the LLM_CALL handler hands it to the provider as `cancel_event=`;
* the stopped attempt is recorded as `cancelled` (never `failed`, never
  retried) with `cancelled_by` and the stop latency, and nothing after it runs
  (no tool batch, no second model call);
* tree semantics: cancelling the root reaches a synchronous child's model
  call; a child cancelled on its own never strands its parent;
* a cancel that lands before the effect registers is not lost.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    Runtime,
    RunStatus,
    StepPlan,
    WorkflowRegistry,
    WorkflowSpec,
)
from abstractruntime.core.effect_cancellation import (
    current_effect_cancel_event,
    inflight_effects,
    request_effect_cancel,
)
from abstractruntime.core.runtime import EffectOutcome


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class _Decoder:
    """An LLM_CALL handler that 'decodes' until its cancel event is set."""

    def __init__(self, *, honour: bool = True, hard_limit_s: float = 5.0) -> None:
        self.honour = honour
        self.hard_limit_s = hard_limit_s
        self.started = threading.Event()
        self.saw_event: List[Any] = []
        self.stopped_at: List[float] = []
        self.calls = 0

    def __call__(self, run, effect, default_next_node):
        self.calls += 1
        event = current_effect_cancel_event()
        self.saw_event.append(event)
        self.started.set()
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.hard_limit_s:
            if self.honour and event is not None and event.is_set():
                self.stopped_at.append(time.monotonic())
                raise RuntimeError("generation cancelled by the host (test decoder)")
            time.sleep(0.002)  # one "token"
        return EffectOutcome.completed({"content": "finished anyway"})


def _agent_workflow(workflow_id: str = "agent") -> WorkflowSpec:
    def reason(run, ctx):
        return StepPlan(
            node_id="reason",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "write 3000 words"}, result_key="llm"),
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

    return WorkflowSpec(workflow_id=workflow_id, entry_node="reason",
                        nodes={"reason": reason, "act": act, "done": done})


def _runtime(decoder, *, tools=None, run_store=None, ledger=None, registry=None) -> Runtime:
    tool_calls: List[Any] = [] if tools is None else tools

    def tool_handler(run, effect, default_next_node):
        tool_calls.append(effect.payload)
        return EffectOutcome.completed({"results": []})

    return Runtime(
        run_store=run_store or InMemoryRunStore(),
        ledger_store=ledger or InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: decoder, EffectType.TOOL_CALLS: tool_handler},
        workflow_registry=registry,
    )


def _tick_in_thread(runtime, workflow, run_id):
    box: Dict[str, Any] = {}

    def body():
        box["state"] = runtime.tick(workflow=workflow, run_id=run_id)

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    return thread, box


def _llm_records(ledger, run_id):
    return [r for r in ledger.list(run_id) if (r.get("effect") or {}).get("type") == "llm_call"]


def test_cancel_reaches_the_in_flight_handler_out_of_band_and_nothing_runs_after_it():
    decoder = _Decoder()
    tools: List[Any] = []
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = _runtime(decoder, tools=tools, run_store=run_store, ledger=ledger)
    workflow = _agent_workflow()
    run_id = runtime.start(workflow=workflow, vars={})
    thread, box = _tick_in_thread(runtime, workflow, run_id)
    assert decoder.started.wait(5)

    # The gateway applies commands through a FRESH Runtime sharing the stores.
    control_plane = Runtime(run_store=run_store, ledger_store=ledger)
    t_cancel = time.monotonic()
    control_plane.cancel_run(run_id, reason="Stop pressed", cancelled_by="command")
    thread.join(5)
    assert not thread.is_alive()

    assert isinstance(decoder.saw_event[0], threading.Event)
    assert decoder.stopped_at and decoder.stopped_at[0] - t_cancel < 1.0, "the model did not stop within 1 s"
    assert box["state"].status == RunStatus.CANCELLED
    assert tools == [], "a tool batch started after the cancel"

    records = _llm_records(ledger, run_id)
    assert [r["status"] for r in records] == ["started", "cancelled"]
    terminal = records[-1]
    assert terminal["result"]["cancelled"] is True
    assert terminal["result"]["cancelled_by"] == "command"
    assert terminal["result"]["reason"] == "Stop pressed"
    assert terminal["result"]["stopped_after_cancel_s"] < 1.0
    assert "generation cancelled" in terminal["result"]["stop_error"]
    # The payload that reached the ledger is still plain JSON.
    json.dumps(records[0]["effect"])
    assert inflight_effects([run_id]) == []


def test_a_cancelled_attempt_is_never_retried():
    from abstractruntime.core.policy import RetryPolicy

    decoder = _Decoder()
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = Runtime(
        run_store=run_store, ledger_store=ledger,
        effect_handlers={EffectType.LLM_CALL: decoder},
        effect_policy=RetryPolicy(llm_max_attempts=3, tool_max_attempts=1, backoff_base=0.0),
    )
    workflow = _agent_workflow()
    run_id = runtime.start(workflow=workflow, vars={})
    thread, _ = _tick_in_thread(runtime, workflow, run_id)
    assert decoder.started.wait(5)
    Runtime(run_store=run_store, ledger_store=ledger).cancel_run(run_id, cancelled_by="command")
    thread.join(5)
    assert decoder.calls == 1
    assert [r["status"] for r in _llm_records(ledger, run_id)] == ["started", "cancelled"]


def test_a_cancel_requested_before_the_effect_registers_is_not_lost():
    decoder = _Decoder()
    runtime = _runtime(decoder)
    workflow = _agent_workflow()
    run_id = runtime.start(workflow=workflow, vars={})
    # The command landed between the tick's control probe and registration:
    # only the in-memory registry knows (the store still says RUNNING).
    request_effect_cancel(run_id, cancelled_by="command", reason="raced")
    state = runtime.tick(workflow=workflow, run_id=run_id)
    assert decoder.calls == 0, "the handler started for a run whose cancel was already requested"
    assert state.status == RunStatus.CANCELLED
    terminal = _llm_records(runtime._ledger_store, run_id)[-1]
    assert terminal["status"] == "cancelled" and terminal["result"]["cancelled_by"] == "command"


def test_a_handler_that_ignores_the_cancel_still_feeds_nothing_downstream():
    decoder = _Decoder(honour=False, hard_limit_s=0.3)
    tools: List[Any] = []
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = _runtime(decoder, tools=tools, run_store=run_store, ledger=ledger)
    workflow = _agent_workflow()
    run_id = runtime.start(workflow=workflow, vars={})
    thread, box = _tick_in_thread(runtime, workflow, run_id)
    assert decoder.started.wait(5)
    Runtime(run_store=run_store, ledger_store=ledger).cancel_run(run_id, cancelled_by="command")
    thread.join(5)
    assert box["state"].status == RunStatus.CANCELLED
    assert tools == [], "the late answer drove a tool batch"
    assert decoder.calls == 1, "the late answer was fed to another model call"
    # The late completion is recorded truthfully (it DID complete), not relabelled.
    assert [r["status"] for r in _llm_records(ledger, run_id)] == ["started", "completed"]


def test_the_llm_call_handler_hands_the_event_to_the_provider_and_never_persists_it():
    from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient

    seen: Dict[str, Any] = {}
    started = threading.Event()

    class _Provider:
        def generate(self, **kwargs):
            seen.update(kwargs)
            started.set()
            event = kwargs.get("cancel_event")
            if isinstance(event, threading.Event):
                event.wait(5)
            else:
                time.sleep(0.5)  # no event: nothing can stop this "decode"
            from abstractcore.exceptions import GenerationCancelledError

            raise GenerationCancelledError("generation cancelled by the host while decoding (mlx/m)")

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider, client._model = "mlx", "mlx-community/Qwen3.5-4B-4bit"
    client._artifact_store, client._generate_lock, client._capability_defaults = None, None, {}
    client._llm = _Provider()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger,
                      effect_handlers=build_effect_handlers(llm=client))
    workflow = _agent_workflow()
    run_id = runtime.start(workflow=workflow, vars={})
    thread, box = _tick_in_thread(runtime, workflow, run_id)
    assert started.wait(5)
    # Kill-switch attribution: the handler named what it is running.
    live = inflight_effects([run_id])
    assert live and live[0]["provider"] == "mlx" and live[0]["model"] == "mlx-community/Qwen3.5-4B-4bit"
    t_cancel = time.monotonic()
    Runtime(run_store=run_store, ledger_store=ledger).cancel_run(run_id, cancelled_by="command")
    thread.join(5)
    assert isinstance(seen.get("cancel_event"), threading.Event), "the provider never received cancel_event"
    assert time.monotonic() - t_cancel < 0.4, "the provider call did not end on the cancel"
    assert box["state"].status == RunStatus.CANCELLED
    records = _llm_records(ledger, run_id)
    assert records[-1]["status"] == "cancelled"
    assert records[-1]["result"]["provider"] == "mlx"
    for record in ledger.list(run_id):
        text = json.dumps(record, default=lambda o: pytest.fail(f"non-JSON object persisted: {o!r}"))
        assert "Event object" not in text


# --------------------------------------------------------------------------
# Tree semantics
# --------------------------------------------------------------------------


def _parent_workflow(child_id: str) -> WorkflowSpec:
    def spawn(run, ctx):
        return StepPlan(
            node_id="spawn",
            effect=Effect(type=EffectType.START_SUBWORKFLOW, payload={"workflow_id": child_id, "vars": {}},
                          result_key="sub"),
            next_node="after",
        )

    def after(run, ctx):
        return StepPlan(node_id="after", complete_output={"sub": run.vars.get("sub")})

    return WorkflowSpec(workflow_id="parent", entry_node="spawn", nodes={"spawn": spawn, "after": after})


def _tree(decoder):
    child = _agent_workflow("child_agent")
    parent = _parent_workflow("child_agent")
    registry = WorkflowRegistry()
    registry.register(child)
    registry.register(parent)
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = _runtime(decoder, run_store=run_store, ledger=ledger, registry=registry)
    return runtime, parent, run_store, ledger


def _child_id(run_store, parent_id):
    runs = [r for r in run_store.list_runs(limit=100) if getattr(r, "parent_run_id", None) == parent_id]
    return runs[0].run_id if runs else None


def test_cancelling_the_root_reaches_a_synchronous_childs_model_call():
    decoder = _Decoder()
    runtime, parent, run_store, ledger = _tree(decoder)
    root = runtime.start(workflow=parent, vars={})
    thread, box = _tick_in_thread(runtime, parent, root)
    assert decoder.started.wait(5)
    child = _child_id(run_store, root)
    assert child and inflight_effects([child])

    # ONLY the root, as a bare runtime API call (no host walks the tree).
    Runtime(run_store=run_store, ledger_store=ledger).cancel_run(root, cancelled_by="command")
    thread.join(5)
    assert not thread.is_alive()
    assert decoder.stopped_at, "the child's model call never saw the root's cancel"
    assert box["state"].status == RunStatus.CANCELLED
    assert run_store.load(child).status == RunStatus.CANCELLED
    child_llm = _llm_records(ledger, child)
    assert child_llm[-1]["status"] == "cancelled"
    assert child_llm[-1]["result"]["cancelled_by"] == "command"
    spawn = [r for r in ledger.list(root) if (r.get("effect") or {}).get("type") == "start_subworkflow"]
    assert spawn[-1]["status"] == "cancelled"


def test_a_child_cancelled_on_its_own_never_strands_its_parent():
    decoder = _Decoder()
    runtime, parent, run_store, ledger = _tree(decoder)
    root = runtime.start(workflow=parent, vars={})
    thread, box = _tick_in_thread(runtime, parent, root)
    assert decoder.started.wait(5)
    child = _child_id(run_store, root)

    Runtime(run_store=run_store, ledger_store=ledger).cancel_run(child, cancelled_by="command")
    thread.join(5)
    assert not thread.is_alive()
    state = box["state"]
    assert state.status == RunStatus.COMPLETED, "the parent was left waiting/failed on a cancelled child"
    sub = state.output["sub"]
    assert sub["output"]["cancelled"] is True and sub["output"]["success"] is False


def test_a_cancel_during_a_tool_batch_never_starts_the_remaining_calls():
    from abstractruntime.core.effect_cancellation import InflightEffect, effect_inflight_scope
    from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor

    ran: List[str] = []
    entry = InflightEffect(run_id="r-tools", node_id="act", step_id="s", effect_type="tool_calls")

    def first(**kwargs):
        ran.append("first")
        entry.request_cancel(cancelled_by="command", reason="Stop pressed")  # Stop lands mid-batch
        return "one"

    def later(**kwargs):
        ran.append("later")
        return "never"

    executor = MappingToolExecutor({"write_a": first, "write_b": later, "write_c": later})
    calls = [{"name": n, "arguments": {}, "call_id": f"c{i}"} for i, n in enumerate(["write_a", "write_b", "write_c"])]
    with effect_inflight_scope(entry):
        out = executor.execute(tool_calls=calls)
    assert ran == ["first"], "a tool call started after the cancel"
    results = out["results"]
    assert results[0]["success"] is True
    assert [r["cancelled"] for r in results[1:]] == [True, True]
    assert all("not started" in r["error"] for r in results[1:]), "skipped calls must be named, not dropped"


# --------------------------------------------------------------------------
# Hard stop: kill_inflight_effect (the gateway kill switch's in-process lever)
# --------------------------------------------------------------------------


class _DeafLoop:
    """Ignores its cancel event and swallows Exception, like a provider loop."""

    def __init__(self, seconds: float = 20.0) -> None:
        self.seconds = seconds
        self.started = threading.Event()
        self.calls = 0

    def __call__(self, run, effect, default_next_node):
        self.calls += 1
        self.started.set()
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.seconds:
            try:
                sum(range(100))
            except Exception:  # noqa: BLE001
                pass
        return EffectOutcome.completed({"content": "ran to the end"})


def test_kill_inflight_effect_unwinds_a_loop_that_ignores_the_event_and_swallows_exception():
    from abstractruntime.core.effect_cancellation import kill_inflight_effect

    decoder = _DeafLoop()
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = _runtime(decoder, run_store=run_store, ledger=ledger)
    workflow = _agent_workflow()
    run_id = runtime.start(workflow=workflow, vars={})
    thread, box = _tick_in_thread(runtime, workflow, run_id)
    assert decoder.started.wait(5)
    step_id = inflight_effects([run_id])[0]["step_id"]
    t0 = time.monotonic()
    result = kill_inflight_effect(step_id, killed_by="kill_switch", reason="deadline")
    assert result["injected"] is True
    thread.join(5)
    assert not thread.is_alive() and time.monotonic() - t0 < 1.0
    # No cancel_run happened: the runtime itself ends the run CANCELLED, attributed.
    assert box["state"].status == RunStatus.CANCELLED
    terminal = _llm_records(ledger, run_id)[-1]
    assert terminal["status"] == "cancelled" and terminal["result"]["killed_by"] == "kill_switch"
    assert kill_inflight_effect(step_id, killed_by="kill_switch")["injected"] is False  # already gone


def test_a_kill_that_lands_on_the_wrong_effect_re_invokes_it_instead_of_failing_it():
    import ctypes

    from abstractruntime.core.effect_cancellation import EffectKilled

    decoder = _DeafLoop(seconds=0.4)
    run_store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = _runtime(decoder, run_store=run_store, ledger=ledger)
    workflow = _agent_workflow()
    run_id = runtime.start(workflow=workflow, vars={})
    thread, box = _tick_in_thread(runtime, workflow, run_id)
    assert decoder.started.wait(5)
    # A kill meant for ANOTHER effect lands on this thread (no killed_by here).
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(EffectKilled))
    thread.join(5)
    assert decoder.calls == 2, "the interrupted attempt was not re-invoked"
    assert box["state"].status == RunStatus.COMPLETED
