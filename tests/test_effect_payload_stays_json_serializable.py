"""The runtime's progress callback rides BESIDE the effect, never inside it.

`Effect.payload` is JSON to everything downstream — the ledger persists it, the
gateway streams it over SSE, and a long tail of host handlers and tests do a
plain ``json.dumps(effect.payload)`` the moment they are handed one. When the
durable progress channel was first offered to EVERY `LLM_CALL` it was delivered
by deep-copying the payload and injecting a Python callable at
``payload["params"]["on_progress"]``; that put a function inside the JSON and
turned 77 abstractagent tests into
``Object of type function is not JSON serializable``.

These pin the repair (`core/progress_channel.py`) at both ends, so neither half
can be quietly undone:

* a handler executing an LLM_CALL can `json.dumps` the payload it was handed AND
  still obtain the progress callback, whose calls still land as durable
  `abstract.progress` records;
* an effect with no progress channel (anything that is not an LLM_CALL) is
  offered None rather than inheriting the previous effect's callback;
* the callback is a DEFAULT: a caller who put its own callable in
  `params["on_progress"]` keeps it, and the runtime's never reaches the provider.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from abstractruntime import Effect, EffectType, RunState, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.progress_channel import (
    current_effect_progress_callback,
    effect_progress_callback,
)
from abstractruntime.core.runtime import EffectOutcome
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


PHASES = [
    {"kind": "llm", "phase": "prefill", "prompt_tokens": 808, "cached_tokens": 690},
    {"kind": "llm", "phase": "generate", "generated_tokens": 1, "first_token": True},
    {"kind": "llm", "phase": "complete", "generated_tokens": 551, "final": True},
]


def _progress_payloads(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for record in records:
        payload = (record.get("effect") or {}).get("payload") or {}
        if payload.get("name") == "abstract.progress":
            out.append(payload.get("payload") or {})
    return out


def _drive(handlers: Dict[Any, Any], *, with_tool_step: bool = False):
    ledger = InMemoryLedgerStore()
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=ledger, effect_handlers=handlers)

    def reason(run, ctx):
        del run, ctx
        return StepPlan(
            node_id="reason",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": "What is a ledger?", "params": {"max_output_tokens": 600}},
                result_key="answer",
            ),
            next_node="act" if with_tool_step else "done",
        )

    def act(run, ctx):
        del run, ctx
        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"call_id": "c1", "name": "read_file", "arguments": {}}]},
                result_key="tools",
            ),
            next_node="done",
        )

    def done(run, ctx):
        del ctx
        return StepPlan(node_id="done", complete_output={"answer": run.vars.get("answer")})

    nodes = {"reason": reason, "done": done}
    if with_tool_step:
        nodes["act"] = act
    workflow = WorkflowSpec(workflow_id="payload_json", entry_node="reason", nodes=nodes)
    run_id = runtime.start(workflow=workflow, vars={})
    state = runtime.tick(workflow=workflow, run_id=run_id)
    assert state.status.value == "completed", state.error
    return ledger.list(run_id)


def test_the_handler_can_json_dumps_the_payload_and_still_gets_the_callback() -> None:
    seen: Dict[str, Any] = {}

    def llm_handler(run, effect, default_next_node):
        del run, default_next_node
        # THE REGRESSION: a callable anywhere in the payload raises here.
        seen["payload_json"] = json.dumps(effect.payload)
        callback = current_effect_progress_callback()
        seen["callable"] = callable(callback)
        assert callable(callback), "the runtime offered no progress channel to an LLM_CALL"
        for phase in PHASES:
            callback(dict(phase))
        return EffectOutcome.completed({"content": "A ledger is an append-only record of steps."})

    records = _drive({EffectType.LLM_CALL: llm_handler})

    assert seen["callable"] is True
    round_tripped = json.loads(seen["payload_json"])
    # The payload reached the handler intact (plus the invocation trace the
    # runtime does stamp INTO it — that one is JSON by construction).
    assert round_tripped["params"]["max_output_tokens"] == 600
    assert "on_progress" not in round_tripped["params"]

    payloads = _progress_payloads(records)
    assert [p["phase"] for p in payloads] == ["prefill", "generate", "complete"]
    assert all(p["kind"] == "llm" for p in payloads)
    assert payloads[0]["node_id"] == "reason"
    assert payloads[0]["prompt_tokens"] == 808

    # Nothing callable survived into the durable record either.
    json.dumps(records)


def test_an_effect_without_a_progress_channel_inherits_nobodys_callback() -> None:
    offered: Dict[str, Any] = {}

    def llm_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        offered["llm"] = current_effect_progress_callback()
        return EffectOutcome.completed({"content": "ok"})

    def tool_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        # Executed AFTER the llm_call in the same tick, on the same thread: a
        # leaked ContextVar would show up right here and write another step's
        # progress records.
        offered["tools"] = current_effect_progress_callback()
        return EffectOutcome.completed({"mode": "executed", "results": []})

    _drive(
        {EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
        with_tool_step=True,
    )

    assert callable(offered["llm"])
    assert offered["tools"] is None
    # ...and the channel is closed again once the tick is over.
    assert current_effect_progress_callback() is None


class _RecordingLLM:
    """A provider that records the kwargs it was handed, as MLX's would be."""

    def __init__(self) -> None:
        self.seen_kwargs: Dict[str, Any] = {}

    def generate(self, **kwargs):
        self.seen_kwargs = dict(kwargs)
        callback = kwargs.get("on_progress")
        if callable(callback):
            callback({"kind": "llm", "phase": "complete", "final": True})

        class _Response:
            content = "ok"
            model = "mlx-community/Qwen3.5-4B-4bit"
            finish_reason = "stop"
            usage: Dict[str, Any] = {}
            metadata: Dict[str, Any] = {}
            tool_calls: List[Any] = []

        return _Response()


def _client(llm) -> LocalAbstractCoreLLMClient:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "mlx-community/Qwen3.5-4B-4bit"
    client._artifact_store = None
    client._generate_lock = None
    client._capability_defaults = {}
    client._llm = llm
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None
    return client


def test_an_explicit_caller_callback_outranks_the_runtime_offer() -> None:
    llm = _RecordingLLM()
    handler = build_effect_handlers(llm=_client(llm))[EffectType.LLM_CALL]

    caller_events: List[Any] = []
    runtime_events: List[Any] = []

    def caller_callback(event: Any = None, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        caller_events.append(event)

    def runtime_callback(event: Any = None, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        runtime_events.append(event)

    run = RunState.new(workflow_id="precedence", entry_node="reason")
    effect = Effect(
        type=EffectType.LLM_CALL,
        # An in-process caller may hand its own callable through params; that is
        # its payload to keep JSON or not, and its callback to keep.
        payload={"prompt": "hi", "params": {"on_progress": caller_callback}},
        result_key="answer",
    )

    with effect_progress_callback(runtime_callback):
        outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    assert llm.seen_kwargs.get("on_progress") is caller_callback
    assert len(caller_events) == 1
    assert runtime_events == [], "the runtime's offer must not displace the caller's callback"


def test_the_runtime_offer_reaches_the_provider_when_the_caller_names_none() -> None:
    llm = _RecordingLLM()
    handler = build_effect_handlers(llm=_client(llm))[EffectType.LLM_CALL]

    runtime_events: List[Any] = []

    def runtime_callback(event: Any = None, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        runtime_events.append(event)

    run = RunState.new(workflow_id="offer", entry_node="reason")
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={"prompt": "hi", "params": {"max_output_tokens": 32}},
        result_key="answer",
    )
    payload_before = json.dumps(effect.payload)

    with effect_progress_callback(runtime_callback):
        outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    assert llm.seen_kwargs.get("on_progress") is runtime_callback
    assert runtime_events == [{"kind": "llm", "phase": "complete", "final": True}]
    # The handler took its own copy: the caller's payload is untouched, still JSON.
    assert json.dumps(effect.payload) == payload_before


def test_without_the_channel_the_provider_is_offered_nothing() -> None:
    """Absent-input case: no ContextVar installed => no callback, not a stale one."""

    llm = _RecordingLLM()
    handler = build_effect_handlers(llm=_client(llm))[EffectType.LLM_CALL]
    run = RunState.new(workflow_id="bare", entry_node="reason")
    effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi"}, result_key="answer")

    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    assert llm.seen_kwargs.get("on_progress") is None


def test_optional_progress_helpers_are_none_safe() -> None:
    with effect_progress_callback(None):
        assert current_effect_progress_callback() is None
    assert current_effect_progress_callback() is None
