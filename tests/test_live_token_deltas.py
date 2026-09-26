"""Live token deltas: the runtime half of token streaming (S-DESIGN 2026-09-26).

Pins, end to end:

* the delta callback travels beside the effect (core.progress_channel), is
  offered ONLY to an LLM_CALL of a run whose `_runtime.stream is True` while a
  host sink is registered, and never leaks into the next effect;
* the sink receives `llm.delta` dicts (call_id = the LLM_CALL step id, seq
  monotonic, channel content|reasoning) and exactly one `llm.delta_end`
  (completed | failed | cancelled), emitted AFTER the durable record;
* the AbstractCore handler turns streaming on and hands the per-call
  `_on_delta` to the client, which never forwards it to the provider;
* NOTHING reaches the ledger: the streamed run's ledger equals the
  non-streamed run's apart from the `stream` flags and measured timings;
* children inherit `stream` through the `_runtime` rider;
* the remote client (AbstractCore server) stays non-streaming.
"""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    Runtime,
    StepPlan,
    WorkflowRegistry,
    WorkflowSpec,
)
from abstractruntime.core.live_deltas import LiveDeltaEmitter
from abstractruntime.core.progress_channel import (
    current_effect_delta_callback,
    effect_delta_callback,
)
from abstractruntime.core.runtime import EffectOutcome
from abstractruntime.integrations.abstractcore.effect_handlers import (
    _observability_params,
    build_effect_handlers,
)
from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
    _LiveThinkSplitter,
    _normalize_local_streaming_response,
)


# ---------------------------------------------------------------------------
# progress_channel: the second ContextVar pair
# ---------------------------------------------------------------------------


def test_delta_channel_installs_and_resets() -> None:
    assert current_effect_delta_callback() is None
    cb = lambda text, channel="content": None  # noqa: E731
    with effect_delta_callback(cb):
        assert current_effect_delta_callback() is cb
        with effect_delta_callback(None):
            assert current_effect_delta_callback() is None
        assert current_effect_delta_callback() is cb
    assert current_effect_delta_callback() is None


# ---------------------------------------------------------------------------
# LiveDeltaEmitter: batching, ordering, seq, end, failing sink
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _emitter(events: List[Dict[str, Any]], clock: _Clock, **kw: Any) -> LiveDeltaEmitter:
    return LiveDeltaEmitter(events.append, run_id="r1", node_id="n1", call_id="s1", clock=clock, **kw)


def test_first_fragment_is_immediate_then_fragments_coalesce() -> None:
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, flush_interval_s=10.0)  # timer never fires in this test
    em("Hel")
    assert [e["text"] for e in events] == ["Hel"]
    clock.now += 0.001
    em("lo ")
    clock.now += 0.001
    em("world")
    assert len(events) == 1  # coalescing, not per-token
    em.end("completed")
    assert [e["kind"] for e in events] == ["llm.delta", "llm.delta", "llm.delta_end"]
    assert events[1] == {
        "kind": "llm.delta",
        "run_id": "r1",
        "node_id": "n1",
        "call_id": "s1",
        "seq": 1,
        "text": "lo world",
        "channel": "content",
    }
    assert events[2] == {
        "kind": "llm.delta_end",
        "run_id": "r1",
        "node_id": "n1",
        "call_id": "s1",
        "seq": 2,
        "reason": "completed",
    }


def test_timer_flushes_a_pause_so_text_is_never_stranded() -> None:
    events: List[Dict[str, Any]] = []
    em = LiveDeltaEmitter(events.append, run_id="r", node_id="n", call_id="c", flush_interval_s=0.02)
    em("a")
    em("b")  # buffered inside the window
    deadline = time.monotonic() + 2.0
    while len(events) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert [e["text"] for e in events] == ["a", "b"]
    em.end("completed")


def test_channel_switch_flushes_in_order_and_seq_is_monotonic() -> None:
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, flush_interval_s=10.0)
    em("think", "reasoning")
    em("ing", "reasoning")
    em("Answer", "content")
    em(" here", "content")
    em.end("completed")
    deltas = [(e["channel"], e["text"]) for e in events if e["kind"] == "llm.delta"]
    assert deltas == [("reasoning", "think"), ("reasoning", "ing"), ("content", "Answer here")]
    assert [e["seq"] for e in events] == list(range(len(events)))


def test_end_is_emitted_once_and_later_fragments_are_ignored() -> None:
    events: List[Dict[str, Any]] = []
    em = _emitter(events, _Clock())
    em.end("cancelled")
    em.end("completed")
    em("late")
    assert [e["kind"] for e in events] == ["llm.delta_end"]
    assert events[0]["reason"] == "cancelled"


def test_invalid_channel_and_reason_fail_loudly() -> None:
    em = _emitter([], _Clock())
    with pytest.raises(ValueError):
        em("x", "tool")
    with pytest.raises(ValueError):
        em.end("done")


def test_a_raising_sink_is_disabled_without_failing_the_producer() -> None:
    calls: List[Dict[str, Any]] = []

    def bad(event: Dict[str, Any]) -> None:
        calls.append(event)
        raise RuntimeError("hostile sink")

    em = LiveDeltaEmitter(bad, run_id="r", node_id="n", call_id="c", clock=_Clock())
    em("a")  # raises inside, contained
    em.flush()
    em.end("completed")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Runtime: when the callback is offered, call_id, delta_end reasons + timing
# ---------------------------------------------------------------------------


def _llm_workflow(handler_next: str = "done") -> WorkflowSpec:
    def reason(run, ctx):
        del run, ctx
        return StepPlan(
            node_id="reason",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi"}, result_key="answer"),
            next_node=handler_next,
        )

    def act(run, ctx):
        del run, ctx
        return StepPlan(
            node_id="act",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": []}, result_key="tools"),
            next_node="done",
        )

    def done(run, ctx):
        del ctx
        return StepPlan(node_id="done", complete_output={"answer": run.vars.get("answer")})

    return WorkflowSpec(workflow_id="deltas", entry_node="reason", nodes={"reason": reason, "act": act, "done": done})


def _drive(handlers, *, runtime_ns: Optional[Dict[str, Any]], sink=None, next_node: str = "done"):
    ledger = InMemoryLedgerStore()
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=ledger, effect_handlers=handlers)
    if sink is not None:
        rt.set_live_delta_sink(sink)
    wf = _llm_workflow(next_node)
    vars_: Dict[str, Any] = {}
    if runtime_ns is not None:
        vars_["_runtime"] = deepcopy(runtime_ns)
    run_id = rt.start(workflow=wf, vars=vars_)
    state = rt.tick(workflow=wf, run_id=run_id)
    return state, ledger, run_id


def _llm_call_step_ids(ledger, run_id) -> List[str]:
    return [
        r["step_id"]
        for r in ledger.list(run_id)
        if (r.get("effect") or {}).get("type") == "llm_call" and r.get("status") == "completed"
    ]


def test_runtime_streams_deltas_with_the_llm_call_step_id_and_ends_after_the_record() -> None:
    events: List[Dict[str, Any]] = []
    ledger_holder: Dict[str, Any] = {}
    offered: Dict[str, Any] = {}

    def sink(event: Dict[str, Any]) -> None:
        if event["kind"] == "llm.delta_end":
            # The durable completed record is already in the ledger.
            ledger, run_id = ledger_holder["ledger"], ledger_holder["run_id"]
            event = dict(event, _completed_in_ledger=bool(_llm_call_step_ids(ledger, run_id)))
        events.append(event)

    def llm_handler(run, effect, default_next_node):
        del effect, default_next_node
        cb = current_effect_delta_callback()
        offered["llm"] = cb
        cb("Hello", "content")
        cb(" there", "content")
        ledger_holder["run_id"] = run.run_id
        return EffectOutcome.completed({"content": "Hello there"})

    def tool_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        offered["tools"] = current_effect_delta_callback()
        return EffectOutcome.completed({"mode": "executed", "results": []})

    ledger = InMemoryLedgerStore()
    ledger_holder["ledger"] = ledger
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    rt.set_live_delta_sink(sink)
    wf = _llm_workflow("act")
    run_id = rt.start(workflow=wf, vars={"_runtime": {"stream": True}})
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status.value == "completed", state.error

    assert callable(offered["llm"])
    assert offered["tools"] is None  # never leaks into the next effect
    [step_id] = _llm_call_step_ids(ledger, run_id)
    assert {e["call_id"] for e in events} == {step_id}
    assert {e["run_id"] for e in events} == {run_id}
    assert {e["node_id"] for e in events} == {"reason"}
    assert "".join(e["text"] for e in events if e["kind"] == "llm.delta") == "Hello there"
    assert events[-1]["kind"] == "llm.delta_end"
    assert events[-1]["reason"] == "completed"
    assert events[-1]["_completed_in_ledger"] is True
    assert [e["seq"] for e in events] == list(range(len(events)))
    # Nothing about deltas in the ledger.
    assert "llm.delta" not in json.dumps(ledger.list(run_id))


@pytest.mark.parametrize(
    "runtime_ns,with_sink",
    [
        ({"stream": True}, False),  # no host listening
        ({"stream": False}, True),  # explicit off
        ({"stream": "true"}, True),  # strict: only the boolean True enables
        ({"stream": 1}, True),
        ({}, True),
        (None, True),
    ],
)
def test_no_delta_callback_unless_stream_is_true_and_a_sink_is_registered(runtime_ns, with_sink) -> None:
    offered: Dict[str, Any] = {}
    events: List[Any] = []

    def llm_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        offered["cb"] = current_effect_delta_callback()
        return EffectOutcome.completed({"content": "ok"})

    state, _, _ = _drive(
        {EffectType.LLM_CALL: llm_handler},
        runtime_ns=runtime_ns,
        sink=events.append if with_sink else None,
    )
    assert state.status.value == "completed"
    assert offered["cb"] is None
    assert events == []


def test_set_live_delta_sink_rejects_non_callables_and_accepts_none() -> None:
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    with pytest.raises(TypeError):
        rt.set_live_delta_sink("not a sink")  # type: ignore[arg-type]
    rt.set_live_delta_sink(lambda e: None)
    rt.set_live_delta_sink(None)


def test_delta_end_reason_failed_when_the_handler_raises() -> None:
    events: List[Dict[str, Any]] = []

    def llm_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        current_effect_delta_callback()("partial", "content")
        raise RuntimeError("provider exploded")

    from abstractruntime.core.policy import NoRetryPolicy

    ledger = InMemoryLedgerStore()
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        effect_policy=NoRetryPolicy(),
    )
    rt.set_live_delta_sink(events.append)
    wf = _llm_workflow()
    run_id = rt.start(workflow=wf, vars={"_runtime": {"stream": True}})
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status.value == "failed"
    assert [e["kind"] for e in events] == ["llm.delta", "llm.delta_end"]
    assert events[-1]["reason"] == "failed"


def test_delta_end_reason_cancelled_for_a_cancelled_outcome() -> None:
    events: List[Dict[str, Any]] = []

    def llm_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        current_effect_delta_callback()("half an ans", "content")
        return EffectOutcome.cancelled({"cancelled_by": "test"})

    _drive({EffectType.LLM_CALL: llm_handler}, runtime_ns={"stream": True}, sink=events.append)
    assert events[-1]["kind"] == "llm.delta_end"
    assert events[-1]["reason"] == "cancelled"


def test_each_retry_attempt_is_its_own_call_with_its_own_end() -> None:
    events: List[Dict[str, Any]] = []
    attempts = {"n": 0}

    def llm_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        attempts["n"] += 1
        current_effect_delta_callback()(f"try{attempts['n']}", "content")
        if attempts["n"] == 1:
            return EffectOutcome.failed("transient")
        return EffectOutcome.completed({"content": "ok"})

    from abstractruntime.core.policy import RetryPolicy

    ledger = InMemoryLedgerStore()
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        effect_policy=RetryPolicy(llm_max_attempts=2, backoff_base=0.0),
    )
    rt.set_live_delta_sink(events.append)
    wf = _llm_workflow()
    run_id = rt.start(workflow=wf, vars={"_runtime": {"stream": True}})
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status.value == "completed", state.error
    ends = [e for e in events if e["kind"] == "llm.delta_end"]
    assert [e["reason"] for e in ends] == ["failed", "completed"]
    assert ends[0]["call_id"] != ends[1]["call_id"]


# ---------------------------------------------------------------------------
# Child runs inherit `stream` (the `_runtime` rider)
# ---------------------------------------------------------------------------


def _child_runtime_ns(parent_ns: Dict[str, Any], child_ns: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    child = WorkflowSpec(
        workflow_id="child",
        entry_node="end",
        nodes={"end": lambda r, c: StepPlan(node_id="end", complete_output={})},
    )
    parent = WorkflowSpec(
        workflow_id="parent",
        entry_node="spawn",
        nodes={
            "spawn": lambda r, c: StepPlan(
                node_id="spawn",
                effect=Effect(
                    type=EffectType.START_SUBWORKFLOW,
                    payload={
                        "workflow_id": "child",
                        "vars": {"_runtime": deepcopy(child_ns or {})},
                        "async": True,
                        "wait": True,
                    },
                    result_key="child",
                ),
            )
        },
    )
    registry = WorkflowRegistry()
    registry.register(child)
    registry.register(parent)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=registry)
    run_id = rt.start(workflow=parent, vars={"_runtime": deepcopy(parent_ns)})
    state = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    child_id = state.waiting.wait_key.split(":", 1)[1]
    return rt.get_state(child_id).vars.get("_runtime") or {}


@pytest.mark.parametrize(
    "parent,child,expected",
    [
        (True, None, True),
        (False, None, False),
        (True, False, False),  # explicit child value wins
        (False, True, True),
        ("true", None, None),  # not a boolean: not inherited
        (None, None, None),
    ],
)
def test_children_inherit_stream(parent, child, expected) -> None:
    parent_ns = {} if parent is None else {"stream": parent}
    child_ns = None if child is None else {"stream": child}
    ns = _child_runtime_ns(parent_ns, child_ns)
    assert ns.get("stream") == expected
    if expected is None:
        assert "stream" not in ns


# ---------------------------------------------------------------------------
# Full stack: handler + LocalAbstractCoreLLMClient + fake streaming provider
# ---------------------------------------------------------------------------

_RAW = {"id": "resp-1", "object": "chat.completion"}
_USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


class _FakeStreamingProvider:
    """Streams (reasoning deltas, then content) or answers at once, same answer."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate(self, **kwargs: Any):
        self.calls.append(kwargs)
        if kwargs.get("stream"):

            def gen():
                base = dict(tool_calls=None, usage=None, model="fake-model", finish_reason=None)
                yield NS(content=None, metadata={"reasoning_delta": "Let me "}, raw_response={"i": 1}, **base)
                yield NS(content=None, metadata={"reasoning_delta": "think."}, raw_response={"i": 2}, **base)
                yield NS(content="Hel", metadata=None, raw_response={"i": 3}, **base)
                yield NS(
                    content="lo world",
                    tool_calls=None,
                    usage=_USAGE,
                    model="fake-model",
                    finish_reason="stop",
                    metadata={"reasoning": "Let me think."},
                    raw_response=_RAW,
                )

            return gen()
        return NS(
            content="Hello world",
            tool_calls=None,
            usage=_USAGE,
            model="fake-model",
            finish_reason="stop",
            metadata={"reasoning": "Let me think."},
            raw_response=_RAW,
            gen_time=None,
        )


def _local_client(provider: Any) -> LocalAbstractCoreLLMClient:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "fake"
    client._model = "fake-model"
    client._artifact_store = None
    client._generate_lock = None
    client._capability_defaults = {}
    client._llm = provider
    client._on_token = None
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None
    client._prompt_cache_state_lock = threading.Lock()
    client._prompt_cache_state = {}
    return client


def _run_full_stack(*, stream: bool, sink=None, payload_params: Optional[Dict[str, Any]] = None):
    provider = _FakeStreamingProvider()
    ledger = InMemoryLedgerStore()
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers=build_effect_handlers(llm=_local_client(provider)),
    )
    if sink is not None:
        rt.set_live_delta_sink(sink)
    payload: Dict[str, Any] = {"prompt": "hi"}
    if payload_params is not None:
        payload["params"] = payload_params
    wf = WorkflowSpec(
        workflow_id="full",
        entry_node="a",
        nodes={
            "a": lambda r, c: StepPlan(
                node_id="a",
                effect=Effect(type=EffectType.LLM_CALL, payload=deepcopy(payload), result_key="ans"),
                next_node="b",
            ),
            "b": lambda r, c: StepPlan(node_id="b", complete_output={"a": r.vars.get("ans")}),
        },
    )
    run_id = rt.start(workflow=wf, vars={"_runtime": {"stream": True}} if stream else {})
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status.value == "completed", state.error
    return ledger.list(run_id), provider, run_id


def test_full_stack_streams_reasoning_and_content_channels() -> None:
    events: List[Dict[str, Any]] = []
    records, provider, run_id = _run_full_stack(stream=True, sink=events.append)

    [call] = provider.calls
    assert call["stream"] is True
    assert "_on_delta" not in call  # never a provider kwarg
    reasoning = "".join(e["text"] for e in events if e.get("channel") == "reasoning")
    content = "".join(e["text"] for e in events if e.get("channel") == "content")
    assert reasoning == "Let me think."
    assert content == "Hello world"
    assert events[-1]["kind"] == "llm.delta_end" and events[-1]["reason"] == "completed"
    completed = [r for r in records if (r.get("effect") or {}).get("type") == "llm_call" and r["status"] == "completed"]
    assert completed[0]["step_id"] == events[0]["call_id"]
    assert completed[0]["result"]["content"] == "Hello world"
    assert completed[0]["result"]["reasoning"] == "Let me think."


_TIMING_KEYS = {"gen_time", "ttft_ms"}
_ID_KEYS = {"step_id", "idempotency_key", "effect_idempotency_key", "started_at", "ended_at"}


def _diff_paths(a: Any, b: Any, path: str = "") -> List[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            out += _diff_paths(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff_paths(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [path]


def test_ledger_is_identical_streamed_or_not_apart_from_the_stream_flags_and_timings() -> None:
    """Streamed-vs-non-streamed parity: same prompt, same fake provider answer ->
    the same persisted records. The only differences allowed are the honest
    `stream` flags and the measured timings (streaming measures gen_time and
    ttft_ms itself). No delta ever reaches the ledger."""
    events: List[Dict[str, Any]] = []
    streamed, _, run_a = _run_full_stack(stream=True, sink=events.append)
    plain, _, run_b = _run_full_stack(stream=False)
    assert events, "the streamed arm produced no deltas: the switch is not wired"

    a = json.loads(json.dumps(streamed).replace(run_a, "RUN"))
    b = json.loads(json.dumps(plain).replace(run_b, "RUN"))
    assert len(a) == len(b)
    diffs = _diff_paths(a, b)
    unexpected = [
        p
        for p in diffs
        if p.rsplit(".", 1)[-1] not in _ID_KEYS | _TIMING_KEYS
        and not p.endswith("._provider_request.payload.stream")
        and not p.endswith("._runtime_observability.llm_generate_kwargs.params.stream")
    ]
    assert unexpected == []
    # The stream flag difference is real (the switch did something).
    assert any(p.endswith("._provider_request.payload.stream") for p in diffs)
    # Parity fields present in the streamed record.
    [done] = [r for r in a if (r.get("effect") or {}).get("type") == "llm_call" and r["status"] == "completed"]
    result = done["result"]
    assert result["raw_response"] == _RAW
    assert result["usage"] == _USAGE
    assert isinstance(result["ttft_ms"], float)
    assert "reasoning_delta" not in (result.get("metadata") or {})


def test_explicit_stream_false_in_the_payload_wins() -> None:
    events: List[Dict[str, Any]] = []
    _, provider, _ = _run_full_stack(stream=True, sink=events.append, payload_params={"stream": False})
    assert provider.calls[0]["stream"] is False
    assert [e["kind"] for e in events] == ["llm.delta_end"]


def test_on_delta_never_reaches_persisted_params() -> None:
    cleaned = _observability_params({"_on_delta": "not-callable-on-purpose", "temperature": 0})
    assert "_on_delta" not in cleaned
    assert cleaned == {"temperature": 0}


# ---------------------------------------------------------------------------
# llm_client: per-call on_delta, think splitting, per-client on_token untouched
# ---------------------------------------------------------------------------


def test_inline_think_markup_is_split_live_even_across_fragments() -> None:
    got: List[tuple] = []
    splitter = _LiveThinkSplitter(lambda text, channel: got.append((channel, text)))
    for piece in ["Sure. <th", "ink>let me ", "reason</thi", "nk>The answer", " is 4.<"]:
        splitter.feed(piece)
    splitter.finish()
    joined: Dict[str, str] = {"content": "", "reasoning": ""}
    for channel, text in got:
        joined[channel] += text
    assert joined == {"content": "Sure. The answer is 4.<", "reasoning": "let me reason"}
    assert all("<think" not in t and "</think" not in t for _, t in got)


def test_per_call_on_delta_is_separate_from_the_per_client_on_token() -> None:
    chunks = [{"content": "<think>hm</think>"}, {"content": "Hi", "metadata": {"reasoning_delta": "x"}}]
    tokens: List[str] = []
    deltas: List[tuple] = []
    result = _normalize_local_streaming_response(
        iter(chunks), on_token=lambda d, m: tokens.append(d), on_delta=lambda t, c: deltas.append((c, t))
    )
    assert tokens == ["<think>hm</think>", "Hi"]  # unchanged legacy surface
    assert ("reasoning", "hm") in deltas and ("content", "Hi") in deltas and ("reasoning", "x") in deltas
    assert result["content"] == "Hi"


def test_a_raising_on_delta_is_contained() -> None:
    def bad(text: str, channel: str) -> None:
        raise RuntimeError("boom")

    result = _normalize_local_streaming_response(iter([{"content": "a"}, {"content": "b"}]), on_delta=bad)
    assert result["content"] == "ab"


class _StubSender:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"url": url, "json": json})
        return {
            "model": json["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def test_remote_client_stays_non_streaming_and_drops_on_delta() -> None:
    sender = _StubSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://localhost:1",
        model="openai-compatible/default",
        request_sender=sender,
    )
    seen: List[Any] = []
    result = client.generate(prompt="hello", params={"stream": True, "_on_delta": lambda t, c: seen.append(t)})
    assert result["content"] == "ok"
    body = sender.calls[-1]["json"]
    assert body["stream"] is False
    assert "_on_delta" not in body
    assert seen == []  # remote mode: no live deltas (documented)
