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
        "parent_run_id": None,
        "node_id": "n1",
        "call_id": "s1",
        "seq": 1,
        "text": "lo world",
        "channel": "content",
    }
    assert events[2] == {
        "kind": "llm.delta_end",
        "run_id": "r1",
        "parent_run_id": None,
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
    em("b")
    em.flush()  # disabled: not delivered
    em.end("completed")  # one more try, for the end event
    assert [c["kind"] for c in calls] == ["llm.delta", "llm.delta_end"]
    assert calls[-1]["reason"] == "unavailable"
    assert calls[-1]["detail"] == "sink_error"
    assert em.unavailable_detail == "sink_error"


def test_unavailable_detail_turns_completed_into_unavailable_but_not_failed() -> None:
    events: List[Dict[str, Any]] = []
    em = _emitter(events, _Clock())
    em.mark_unavailable("structured_output")
    em.mark_unavailable("usage_unavailable")  # first reason wins
    em.end("completed")
    assert events[-1]["reason"] == "unavailable" and events[-1]["detail"] == "structured_output"

    failed: List[Dict[str, Any]] = []
    em2 = _emitter(failed, _Clock())
    em2.mark_unavailable("provider_cannot_stream")
    em2.end("failed")
    assert failed[-1]["reason"] == "failed" and "detail" not in failed[-1]

    with pytest.raises(ValueError):
        _emitter([], _Clock()).mark_unavailable("because")
    with pytest.raises(ValueError):
        _emitter([], _Clock()).end("unavailable")  # needs a detail


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
        ({"stream": None}, True),  # unset
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


@pytest.mark.parametrize("bad", ["true", 1, 0, "yes", {"on": True}])
def test_a_non_boolean_stream_switch_is_refused_at_start(bad) -> None:
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    with pytest.raises(ValueError, match="_runtime.stream must be a boolean"):
        rt.start(workflow=_llm_workflow(), vars={"_runtime": {"stream": bad}})


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


# ---------------------------------------------------------------------------
# S-2 amendment: parent_run_id, the parity gate and every "no stream" detail
# ---------------------------------------------------------------------------


def test_events_carry_parent_run_id_none_for_roots_and_the_parent_for_children() -> None:
    events: List[Dict[str, Any]] = []

    def llm_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        current_effect_delta_callback()("hi", "content")
        return EffectOutcome.completed({"content": "hi"})

    child = WorkflowSpec(
        workflow_id="child",
        entry_node="reason",
        nodes={
            "reason": lambda r, c: StepPlan(
                node_id="reason",
                effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi"}, result_key="a"),
                next_node="done",
            ),
            "done": lambda r, c: StepPlan(node_id="done", complete_output={}),
        },
    )
    parent = WorkflowSpec(
        workflow_id="parent",
        entry_node="reason",
        nodes={
            "reason": lambda r, c: StepPlan(
                node_id="reason",
                effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "hi"}, result_key="a"),
                next_node="spawn",
            ),
            "spawn": lambda r, c: StepPlan(
                node_id="spawn",
                effect=Effect(
                    type=EffectType.START_SUBWORKFLOW,
                    payload={"workflow_id": "child", "vars": {}},
                    result_key="child",
                ),
                next_node="done",
            ),
            "done": lambda r, c: StepPlan(node_id="done", complete_output={}),
        },
    )
    registry = WorkflowRegistry()
    registry.register(child)
    registry.register(parent)
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        workflow_registry=registry,
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )
    rt.set_live_delta_sink(events.append)
    root_id = rt.start(workflow=parent, vars={"_runtime": {"stream": True}})
    state = rt.tick(workflow=parent, run_id=root_id)
    assert state.status.value == "completed", state.error

    by_run: Dict[str, set] = {}
    for e in events:
        by_run.setdefault(e["run_id"], set()).add(e["parent_run_id"])
    assert by_run[root_id] == {None}
    child_ids = [rid for rid in by_run if rid != root_id]
    assert len(child_ids) == 1, "the child run did not stream: stream was not inherited"
    assert by_run[child_ids[0]] == {root_id}


class _ConfigurableProvider:
    """Fake AbstractCore provider for the parity gate.

    mode:
      "ok"             streams with usage on the terminal chunk
      "no_usage"       streams, never reports usage
      "reject_options" the first streamed request discovers the server
                       rejects `stream_options` (sets the provider latch the
                       way openai_compatible_provider does, before any token)
      "one_piece"      answers in one piece even when asked to stream
    """

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls: List[Dict[str, Any]] = []
        self._stream_options_unsupported = False

    def _final(self) -> NS:
        return NS(
            content="Hello world",
            tool_calls=None,
            usage=_USAGE,
            model="fake-model",
            finish_reason="stop",
            metadata={"prompt_cache": {"mode": "key", "outcome": "hit_extend", "cached_tokens": 5}},
            raw_response=_RAW,
            gen_time=None,
        )

    def generate(self, **kwargs: Any):
        self.calls.append(kwargs)
        if not kwargs.get("stream") or self.mode == "one_piece":
            return self._final()
        provider = self

        def gen():
            if provider.mode in ("reject_options", "reject_options_usage_anyway"):
                provider._stream_options_unsupported = True
            usage = None if provider.mode in ("no_usage", "reject_options") else _USAGE
            yield NS(content="Hel", tool_calls=None, usage=None, model="fake-model", finish_reason=None,
                     metadata=None, raw_response={"i": 1})
            yield NS(
                content="lo world",
                tool_calls=None,
                usage=usage,
                model="fake-model",
                finish_reason="stop",
                metadata={"prompt_cache": {"mode": "key", "outcome": "hit_extend", "cached_tokens": 5}},
                raw_response=_RAW,
            )

        return gen()


def _run_with(provider: Any, *, client: Any = None, payload: Optional[Dict[str, Any]] = None, sink=None):
    events: List[Dict[str, Any]] = []
    ledger = InMemoryLedgerStore()
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers=build_effect_handlers(llm=client if client is not None else _local_client(provider)),
    )
    rt.set_live_delta_sink(sink if sink is not None else events.append)
    body = payload if payload is not None else {"prompt": "hi"}
    wf = WorkflowSpec(
        workflow_id="gate",
        entry_node="a",
        nodes={
            "a": lambda r, c: StepPlan(
                node_id="a",
                effect=Effect(type=EffectType.LLM_CALL, payload=deepcopy(body), result_key="ans"),
                next_node="b",
            ),
            "b": lambda r, c: StepPlan(node_id="b", complete_output={}),
        },
    )
    run_id = rt.start(workflow=wf, vars={"_runtime": {"stream": True}})
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status.value == "completed", state.error
    [record] = [
        r for r in ledger.list(run_id) if (r.get("effect") or {}).get("type") == "llm_call" and r["status"] == "completed"
    ]
    return events, record


def _stream_unavailable(record: Dict[str, Any]) -> Optional[str]:
    return ((record["result"].get("metadata") or {}).get("_runtime_observability") or {}).get("stream_unavailable")


def test_streamed_record_keeps_prompt_cache_raw_response_and_usage() -> None:
    events, record = _run_with(_ConfigurableProvider("ok"))
    assert any(e["kind"] == "llm.delta" for e in events)
    assert events[-1]["reason"] == "completed"
    result = record["result"]
    assert result["metadata"]["prompt_cache"]["outcome"] == "hit_extend"
    assert result["raw_response"] == _RAW
    assert result["usage"] == _USAGE
    assert _stream_unavailable(record) is None


def test_the_provider_flag_alone_does_not_refuse_streaming() -> None:
    """A server that rejected `stream_options` may still send usage (LM Studio
    puts it on the last content chunk): the flag alone never refuses."""
    provider = _ConfigurableProvider("ok")
    provider._stream_options_unsupported = True
    events, record = _run_with(provider)
    assert provider.calls[-1]["stream"] is True
    assert events[-1]["reason"] == "completed"
    assert record["result"]["usage"] == _USAGE


def test_a_server_that_rejects_options_but_sends_usage_keeps_streaming() -> None:
    provider = _ConfigurableProvider("reject_options_usage_anyway")
    client = _local_client(provider)
    events, record = _run_with(provider, client=client)
    assert provider._stream_options_unsupported is True
    assert events[-1]["reason"] == "completed" and record["result"]["usage"] == _USAGE
    events2, _ = _run_with(provider, client=client)
    assert provider.calls[-1]["stream"] is True
    assert events2[-1]["reason"] == "completed"


def test_usage_missing_at_the_end_is_reported_and_the_next_call_streams_again() -> None:
    """An unexplained missing usage is reported for THAT call only; it must not
    switch streaming off for the model until restart (REVIEW/17 S2)."""
    provider = _ConfigurableProvider("no_usage")
    client = _local_client(provider)
    events, record = _run_with(provider, client=client)
    # The reply DID stream: the live view ends `completed`; only the record
    # says usage was missing (REVIEW/21).
    assert any(e["kind"] == "llm.delta" for e in events)
    assert events[-1]["reason"] == "completed" and "detail" not in events[-1]
    assert _stream_unavailable(record) == "usage_unavailable"
    assert record["result"]["metadata"]["usage_estimated"] is True

    provider.mode = "ok"  # the hiccup is over
    events2, record2 = _run_with(provider, client=client)
    assert provider.calls[-1]["stream"] is True
    assert any(e["kind"] == "llm.delta" for e in events2)
    assert events2[-1]["reason"] == "completed"
    assert _stream_unavailable(record2) is None


def test_flag_plus_a_missing_usage_refuses_until_a_reprobe_gets_usage(monkeypatch) -> None:
    """Into the refusal: the provider flagged the rejection AND the streamed
    answer had no usage. Out of it: a periodic re-probe streams again, and one
    streamed answer with usage lifts the refusal."""
    from abstractruntime.integrations.abstractcore import llm_client as lc

    monkeypatch.setattr(lc, "_STREAM_USAGE_REPROBE_EVERY", 3)
    provider = _ConfigurableProvider("reject_options")
    client = _local_client(provider)
    events, record = _run_with(provider, client=client)
    assert provider.calls[-1]["stream"] is True  # discovery call streamed
    assert events[-1]["reason"] == "completed" and _stream_unavailable(record) == "usage_unavailable"

    provider.mode = "reject_options_usage_anyway"  # the server now sends usage
    for _ in range(2):  # refused calls 1 and 2
        ev, rec = _run_with(provider, client=client)
        assert provider.calls[-1]["stream"] is False
        assert [e["kind"] for e in ev] == ["llm.delta_end"] and ev[-1]["detail"] == "usage_unavailable"
        assert rec["result"]["usage"] == _USAGE
    ev, _ = _run_with(provider, client=client)  # 3rd: re-probe streams, gets usage
    assert provider.calls[-1]["stream"] is True and ev[-1]["reason"] == "completed"
    ev, _ = _run_with(provider, client=client)  # released
    assert provider.calls[-1]["stream"] is True and ev[-1]["reason"] == "completed"


def test_a_provider_listed_without_streamed_prompt_cache_does_not_stream(monkeypatch) -> None:
    """The gate mechanism, independent of which providers are listed today."""
    from abstractruntime.integrations.abstractcore import llm_client as lc

    monkeypatch.setattr(lc, "_STREAM_LANES_WITHOUT_PROMPT_CACHE_TELEMETRY", frozenset({"fakelane"}))
    provider = _ConfigurableProvider("ok")
    client = _local_client(provider)
    client._provider = "fakelane"
    events, record = _run_with(
        provider, client=client, payload={"prompt": "hi", "params": {"prompt_cache_key": "sess:1"}}
    )
    assert provider.calls[-1]["stream"] is False
    assert events[-1]["detail"] == "prompt_cache_unavailable"
    assert _stream_unavailable(record) == "prompt_cache_unavailable"


class _RealMLXShapes:
    """The REAL MLXProvider's streamed and sync lanes (abstractcore 8d59974),
    over a fake mlx-lm generator: no model load, real chunk/metadata shapes."""

    TELEMETRY = {"mode": "key", "outcome": "hit_extend", "cached_tokens": 4, "fed_tokens": 7}

    def __init__(self) -> None:
        from unittest.mock import Mock

        from abstractcore.providers.mlx_provider import MLXProvider

        words = ["Hello", " there", " friend"]
        p = MLXProvider.__new__(MLXProvider)
        p.model = "fake/mlx-lm"
        p.logger = Mock()
        p.llm = object()
        p.tokenizer = NS(encode=lambda text: list(range(len(str(text).split()))))
        p._mtp_processor = None
        p._native_runtime = None
        p._build_mlx_sampler = lambda *a, **k: None

        def stream_generate_fn(model, tokenizer, prompt, **kwargs):
            for i, w in enumerate(words):
                yield NS(text=w, generation_tokens=i + 1, prompt_tokens=7,
                         finish_reason="stop" if i == len(words) - 1 else None)

        p.stream_generate_fn = stream_generate_fn
        p.generate_fn = lambda model, tokenizer, prompt=None, **kwargs: "".join(words)
        self.p = p
        self.calls: List[Dict[str, Any]] = []

    def generate(self, **kwargs: Any):
        self.calls.append(kwargs)
        telemetry = dict(self.TELEMETRY, key=kwargs.get("prompt_cache_key"))
        if kwargs.get("stream"):
            return self.p._stream_generate(
                "the prompt", 16, 0.0, 1.0, usage_prompt="the full prompt", prompt_cache_telemetry=telemetry
            )
        resp = self.p._single_generate("the prompt", 16, 0.0, 1.0, usage_prompt="the full prompt")
        resp.metadata = dict(resp.metadata or {}, prompt_cache=self.p._final_prompt_cache_telemetry(telemetry))
        return resp


def test_mlx_streams_and_records_the_same_prompt_cache_and_usage_as_non_streamed() -> None:
    pytest.importorskip("abstractcore.providers.mlx_provider")
    payload = {"prompt": "hi", "params": {"prompt_cache_key": "sess:1"}}

    streamed_provider = _RealMLXShapes()
    client = _local_client(streamed_provider)
    client._provider = "mlx"
    events, streamed = _run_with(streamed_provider, client=client, payload=payload)
    assert streamed_provider.calls[-1]["stream"] is True
    assert "".join(e["text"] for e in events if e["kind"] == "llm.delta") == "Hello there friend"
    assert events[-1]["reason"] == "completed"
    assert _stream_unavailable(streamed) is None

    sync_provider = _RealMLXShapes()
    sync_client = _local_client(sync_provider)
    sync_client._provider = "mlx"
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_effect_handlers(llm=sync_client),
    )
    wf = WorkflowSpec(
        workflow_id="sync",
        entry_node="a",
        nodes={
            "a": lambda r, c: StepPlan(
                node_id="a",
                effect=Effect(type=EffectType.LLM_CALL, payload=deepcopy(payload), result_key="ans"),
                next_node="b",
            ),
            "b": lambda r, c: StepPlan(node_id="b", complete_output={}),
        },
    )
    run_id = rt.start(workflow=wf, vars={})
    rt.tick(workflow=wf, run_id=run_id)
    [plain] = [
        r for r in rt.ledger_store.list(run_id)
        if (r.get("effect") or {}).get("type") == "llm_call" and r["status"] == "completed"
    ]
    assert sync_provider.calls[-1]["stream"] is False

    s_res, p_res = streamed["result"], plain["result"]
    assert s_res["metadata"]["prompt_cache"] == p_res["metadata"]["prompt_cache"]
    assert s_res["metadata"]["prompt_cache"]["key"] == "sess:1"
    assert s_res["usage"] == p_res["usage"] and s_res["usage"]
    assert s_res["finish_reason"] == p_res["finish_reason"] == "stop"
    assert s_res["content"] == p_res["content"] == "Hello there friend"


def test_structured_output_does_not_stream_and_says_so() -> None:
    class _JSONProvider(_ConfigurableProvider):
        def _final(self) -> NS:
            out = super()._final()
            out.content = '{"x": 1}'
            return out

    provider = _JSONProvider("ok")
    events, record = _run_with(
        provider,
        payload={
            "prompt": "hi",
            "response_schema": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
        },
    )
    assert all(c["stream"] is False for c in provider.calls)
    assert events[-1]["detail"] == "structured_output"
    assert _stream_unavailable(record) == "structured_output"


def test_provider_answering_in_one_piece_is_reported() -> None:
    events, record = _run_with(_ConfigurableProvider("one_piece"))
    assert events[-1]["detail"] == "provider_cannot_stream"
    assert _stream_unavailable(record) == "provider_cannot_stream"


def test_node_stream_off_is_reported() -> None:
    events, record = _run_with(_ConfigurableProvider("ok"), payload={"prompt": "hi", "params": {"stream": False}})
    assert events[-1]["detail"] == "node_stream_off"
    assert _stream_unavailable(record) == "node_stream_off"


def test_a_failing_sink_is_recorded_as_sink_error() -> None:
    delivered: List[Dict[str, Any]] = []

    def flaky(event: Dict[str, Any]) -> None:
        delivered.append(event)
        if event["kind"] == "llm.delta":
            raise RuntimeError("client went away")

    _, record = _run_with(_ConfigurableProvider("ok"), sink=flaky)
    assert delivered[-1]["kind"] == "llm.delta_end"
    assert delivered[-1]["detail"] == "sink_error"
    assert _stream_unavailable(record) == "sink_error"
    assert record["result"]["content"] == "Hello world"


def test_remote_mode_reports_remote_core() -> None:
    emitter_events: List[Dict[str, Any]] = []
    em = LiveDeltaEmitter(emitter_events.append, run_id="r", node_id="n", call_id="c")
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://localhost:1", model="openai-compatible/default", request_sender=_StubSender()
    )
    client.generate(prompt="hello", params={"stream": True, "_on_delta": em})
    em.end("completed")
    assert emitter_events[-1]["reason"] == "unavailable" and emitter_events[-1]["detail"] == "remote_core"


def test_think_tag_split_across_provider_chunks_is_split_server_side() -> None:
    """S-2 #10: clients never parse <think>; the runtime splits it, even when
    the tags themselves are cut across provider chunks."""

    class _ThinkProvider(_ConfigurableProvider):
        def generate(self, **kwargs: Any):
            self.calls.append(kwargs)
            pieces = ["<th", "ink>weigh ", "options</th", "ink>", "Answer", ": 4"]

            def gen():
                for i, piece in enumerate(pieces):
                    last = i == len(pieces) - 1
                    yield NS(content=piece, tool_calls=None, usage=_USAGE if last else None, model="m",
                             finish_reason="stop" if last else None, metadata=None, raw_response=None)

            return gen()

    events, record = _run_with(_ThinkProvider("ok"))
    content = "".join(e["text"] for e in events if e.get("channel") == "content")
    reasoning = "".join(e["text"] for e in events if e.get("channel") == "reasoning")
    assert content == "Answer: 4"
    assert reasoning == "weigh options"
    assert not any("<" in e.get("text", "") for e in events)
    assert record["result"]["content"] == "Answer: 4"
    assert record["result"]["reasoning"] == "weigh options"


@pytest.mark.parametrize(
    "marker",
    ["<tool_call>", "<|tool_call|>", "<|tool_call>", "<|tool_call_start|>", "<function_call>", "```tool_code"],
)
def test_tool_call_envelopes_never_reach_the_live_content(marker) -> None:
    got: List[tuple] = []
    splitter = _LiveThinkSplitter(lambda text, channel: got.append((channel, text)))
    cut = len(marker) // 2
    for piece in ["Let me check. ", marker[:cut], marker[cut:], '{"name": "read_file", "arguments": {}}', "more"]:
        splitter.feed(piece)
    splitter.finish()
    content = "".join(t for c, t in got if c == "content")
    assert content == "Let me check. "
    assert "{" not in content


def test_a_lone_angle_bracket_is_not_swallowed() -> None:
    got: List[tuple] = []
    splitter = _LiveThinkSplitter(lambda text, channel: got.append((channel, text)))
    for piece in ["a <", "b", " and <tool", "s> c"]:
        splitter.feed(piece)
    splitter.finish()
    assert "".join(t for _, t in got) == "a <b and <tools> c"


# ---------------------------------------------------------------------------
# Harmony (gpt-oss) transcripts (REVIEW/17 S1)
# ---------------------------------------------------------------------------


def _split_live(pieces: List[str]):
    got: List[tuple] = []
    splitter = _LiveThinkSplitter(lambda text, channel: got.append((channel, text)))
    for piece in pieces:
        splitter.feed(piece)
    splitter.finish()
    joined: Dict[str, str] = {"content": "", "reasoning": ""}
    for channel, text in got:
        joined[channel] += text
    return joined, got, splitter


def _chop(text: str, size: int = 3) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


HARMONY_ANSWER = (
    "<|channel|>analysis<|message|>User asks 2+2. Easy.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>"
)
HARMONY_TOOL = (
    "<|channel|>analysis<|message|>Need to read the file.<|end|>"
    "<|start|>assistant<|channel|>commentary to=functions.read_file <|constrain|>json"
    '<|message|>{"file_path": "a.txt"}<|call|>'
)


@pytest.mark.parametrize("size", [1, 3, 7, 1000])
def test_harmony_final_streams_and_analysis_is_reasoning(size) -> None:
    joined, got, splitter = _split_live(_chop(HARMONY_ANSWER, size))
    assert joined == {"content": "2 + 2 = 4.", "reasoning": "User asks 2+2. Easy."}
    assert all("<|" not in t for _, t in got)  # framing tokens never emitted
    assert splitter.emitted_content and not splitter.held_back


@pytest.mark.parametrize("size", [1, 4, 1000])
def test_harmony_commentary_tool_call_is_held_back(size) -> None:
    joined, got, splitter = _split_live(_chop(HARMONY_TOOL, size))
    assert joined == {"content": "", "reasoning": "Need to read the file."}
    assert "file_path" not in "".join(t for _, t in got)
    assert splitter.held_back and not splitter.emitted_content


def test_harmony_commentary_preamble_without_recipient_is_content() -> None:
    joined, _, _ = _split_live(["<|channel|>commentary<|message|>Checking the file now.<|end|>"])
    assert joined["content"] == "Checking the file now."


def test_plain_text_with_angle_pipes_is_untouched() -> None:
    joined, _, splitter = _split_live(["a <| b |> c", " <|not a token|>"])
    assert joined["content"] == "a <| b |> c <|not a token|>"
    assert not splitter.held_back


class _HarmonyProvider(_ConfigurableProvider):
    def __init__(self, text: str) -> None:
        super().__init__("ok")
        self.text = text

    def generate(self, **kwargs: Any):
        self.calls.append(kwargs)
        pieces = _chop(self.text, 5)

        def gen():
            for i, piece in enumerate(pieces):
                last = i == len(pieces) - 1
                yield NS(content=piece, tool_calls=None, usage=_USAGE if last else None, model="gpt-oss",
                         finish_reason="stop" if last else None, metadata=None, raw_response=None)

        return gen()


def test_harmony_answer_streams_end_to_end() -> None:
    events, _ = _run_with(_HarmonyProvider(HARMONY_ANSWER))
    content = "".join(e["text"] for e in events if e.get("channel") == "content")
    reasoning = "".join(e["text"] for e in events if e.get("channel") == "reasoning")
    assert content == "2 + 2 = 4."
    assert reasoning == "User asks 2+2. Easy."
    assert events[-1]["reason"] == "completed"


def test_a_call_whose_whole_answer_is_held_back_says_so() -> None:
    events, record = _run_with(_HarmonyProvider(HARMONY_TOOL))
    assert not any(e.get("channel") == "content" for e in events)
    assert events[-1]["reason"] == "unavailable"
    assert events[-1]["detail"] == "tool_envelope_holdback"
    assert _stream_unavailable(record) == "tool_envelope_holdback"


def test_text_then_a_tool_call_is_not_reported_as_held_back() -> None:
    events, record = _run_with(_HarmonyProvider('Let me look. <tool_call>{"name": "x", "arguments": {}}</tool_call>'))
    assert "".join(e["text"] for e in events if e.get("channel") == "content") == "Let me look. "
    assert events[-1]["reason"] == "completed"
    assert _stream_unavailable(record) is None


# ---------------------------------------------------------------------------
# Core's UnifiedStreamProcessor output through the runtime path: the core
# splits harmony (7dddf90) and gates ```json tool blocks (4d9260b); the runtime
# splitter is defence in depth and must never process the same text twice.
# ---------------------------------------------------------------------------


def _through_core_and_runtime(transcript: str, *, size: int, tools):
    usp = pytest.importorskip("abstractcore.providers.streaming")
    from abstractcore.core.types import GenerateResponse

    chunks = [GenerateResponse(content=transcript[i : i + size], model="m") for i in range(0, len(transcript), size)]
    core_out = list(usp.UnifiedStreamProcessor(model_name="openai/gpt-oss-20b").process_stream(iter(chunks), tools))
    got: List[tuple] = []
    result = _normalize_local_streaming_response(iter(core_out), on_delta=lambda t, c: got.append((c, t)))
    live = {"content": "", "reasoning": ""}
    for channel, text in got:
        live[channel] += text
    core_content = "".join(c.content or "" for c in core_out)
    core_reasoning = "".join((c.metadata or {}).get("reasoning_delta", "") for c in core_out)
    return live, result, core_content, core_reasoning


@pytest.mark.parametrize("size", [1, 5, 1000])
def test_core_split_harmony_is_streamed_once(size) -> None:
    transcript = (
        "<|channel|>analysis<|message|>User asks 2+2. Simple.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>The answer is 4."
    )
    live, result, core_content, core_reasoning = _through_core_and_runtime(transcript, size=size, tools=None)
    assert live == {"content": "The answer is 4.", "reasoning": "User asks 2+2. Simple."}
    assert live["content"] == core_content and live["reasoning"] == core_reasoning  # not doubled, not dropped
    assert result["content"] == "The answer is 4."


@pytest.mark.parametrize("size", [1, 5, 1000])
def test_core_split_harmony_tool_call_streams_preamble_once_and_keeps_the_call(size) -> None:
    transcript = (
        "<|channel|>analysis<|message|>Need the weather.<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>Checking now.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"city": "Paris"}<|call|>'
    )
    live, result, core_content, _ = _through_core_and_runtime(transcript, size=size, tools=[{"name": "get_weather"}])
    assert live["content"] == core_content == "Checking now."
    assert live["reasoning"] == "Need the weather."
    assert "Paris" not in live["content"]
    assert [c.get("name") for c in (result["tool_calls"] or [])] == ["get_weather"]


def test_a_json_block_that_is_not_a_tool_call_reaches_the_live_text() -> None:
    """Core only treats ```json as a tool call when it names an offered tool;
    the runtime must not hold back what core decided is content."""
    usp = pytest.importorskip("abstractcore.providers.streaming")
    from abstractcore.core.types import GenerateResponse

    text = 'Here is the config:\n```json\n{"theme": "dark"}\n```\nDone.'
    chunks = [GenerateResponse(content=text[i : i + 4], model="m") for i in range(0, len(text), 4)]
    core_out = list(usp.UnifiedStreamProcessor(model_name="qwen3-4b").process_stream(iter(chunks), [{"name": "read_file"}]))
    got: List[str] = []
    _normalize_local_streaming_response(iter(core_out), on_delta=lambda t, c: got.append(t) if c == "content" else None)
    assert "".join(got) == "".join(c.content or "" for c in core_out)
    assert '"theme": "dark"' in "".join(got)



# ---------------------------------------------------------------------------
# REVIEW/21: no "not streamed" under text that streamed; the aborted-
# generation detector is not blind on streams without usage.
# ---------------------------------------------------------------------------


def test_usage_missing_before_any_text_still_ends_unavailable() -> None:
    from abstractruntime.integrations.abstractcore.llm_client import _report_stream_usage_missing

    events: List[Dict[str, Any]] = []
    em = LiveDeltaEmitter(events.append, run_id="r", node_id="n", call_id="c")
    _report_stream_usage_missing(em)
    em.end("completed")
    assert events[-1]["reason"] == "unavailable" and events[-1]["detail"] == "usage_unavailable"

    streamed: List[Dict[str, Any]] = []
    em2 = LiveDeltaEmitter(streamed.append, run_id="r", node_id="n", call_id="c", flush_interval_s=10.0)
    em2("hello")
    em2(" again")  # still buffered: counts as streamed text
    _report_stream_usage_missing(em2)
    em2.end("completed")
    assert streamed[-1]["reason"] == "completed"
    assert em2.record_only_detail == "usage_unavailable"


class _AbortedStreamProvider(_ConfigurableProvider):
    """Streams text with no usage; `finish` is the terminal chunk's reason."""

    def __init__(self, text: str, finish: Optional[str]) -> None:
        super().__init__("ok")
        self.text, self.finish = text, finish

    def generate(self, **kwargs: Any):
        self.calls.append(kwargs)
        pieces = _chop(self.text, 6)

        def gen():
            for i, piece in enumerate(pieces):
                last = i == len(pieces) - 1
                yield NS(content=piece, tool_calls=None, usage=None, model="m",
                         finish_reason=self.finish if last else None, metadata=None, raw_response=None)

        return gen()


@pytest.mark.parametrize(
    "text,finish,aborted",
    [
        ("Let me read the configuration file first:", "stop", True),  # tool-call preface, call dropped
        ("Here is the whole answer, cut", None, True),  # no terminal chunk at all
        ("The answer is 4.", "stop", False),
        ("x" * 500 + ":", "stop", False),  # long text ending with ':' is a real answer
    ],
)
def test_streamed_answers_without_usage_are_judged_by_the_heuristic(text, finish, aborted) -> None:
    _, record = _run_with(_AbortedStreamProvider(text, finish))
    meta = record["result"]["metadata"]
    assert meta["usage_estimated"] is True
    assert bool(meta.get("generation_aborted")) is aborted
    observed = (meta.get("_runtime_observability") or {}).get("aborted_generation")
    assert (observed is not None) is aborted


def test_non_streamed_answers_without_usage_stay_unknown() -> None:
    from abstractruntime.integrations.abstractcore.effect_handlers import _looks_like_aborted_generation

    result = {"content": "Let me check:", "finish_reason": None, "usage": None, "tool_calls": None,
              "metadata": {"_provider_request": {"payload": {"stream": False}}}}
    assert _looks_like_aborted_generation(result) is False
    assert "usage_estimated" not in result["metadata"]


def test_reprobe_counter_is_thread_safe(monkeypatch) -> None:
    import threading as _t

    from abstractruntime.integrations.abstractcore import llm_client as lc

    monkeypatch.setattr(lc, "_STREAM_USAGE_REPROBE_EVERY", 10)
    provider = _ConfigurableProvider("ok")
    provider._stream_options_unsupported = True
    client = _local_client(provider)
    client._stream_usage_refused = True
    client._stream_usage_refused_calls = 0
    results: List[Optional[str]] = []
    lock = _t.Lock()

    def worker() -> None:
        for _ in range(250):
            r = client._stream_parity_refusal({})
            with lock:
                results.append(r)

    threads = [_t.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert client._stream_usage_refused_calls == 2000
    assert results.count(None) == 200  # exactly one re-probe per ten calls


# ---------------------------------------------------------------------------
# Ordering: every live fragment precedes the call's durable record
# ---------------------------------------------------------------------------


def test_the_last_batched_fragment_is_sent_before_the_record_and_nothing_after() -> None:
    timeline: List[tuple] = []
    held: Dict[str, Any] = {}

    class _RecordingLedger(InMemoryLedgerStore):
        def append(self, record):  # type: ignore[override]
            out = super().append(record)
            rec = record if isinstance(record, dict) else getattr(record, "to_dict", lambda: {})()
            status = getattr(record, "status", None)
            status = getattr(status, "value", status) or rec.get("status")
            effect = getattr(record, "effect", None) or rec.get("effect") or {}
            etype = effect.get("type") if isinstance(effect, dict) else getattr(getattr(effect, "type", None), "value", None)
            timeline.append(("record", etype, status))
            return out

    def llm_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        cb = current_effect_delta_callback()
        held["cb"] = cb
        cb("Hello", "content")  # first fragment: sent at once
        cb(" world", "content")  # inside the 40 ms window: still batched on return
        return EffectOutcome.completed({"content": "Hello world"})

    ledger = _RecordingLedger()
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=ledger, effect_handlers={EffectType.LLM_CALL: llm_handler})
    rt.set_live_delta_sink(lambda e: timeline.append((e["kind"], e.get("text"), e["seq"])))
    wf = _llm_workflow()
    run_id = rt.start(workflow=wf, vars={"_runtime": {"stream": True}})
    assert rt.tick(workflow=wf, run_id=run_id).status.value == "completed"

    # A fragment arriving after the call returned (a late provider thread) is refused.
    held["cb"]("late", "content")
    time.sleep(0.08)  # longer than the batch window: a stray timer would have fired

    record_at = next(i for i, t in enumerate(timeline) if t[0] == "record" and t[1] == "llm_call" and t[2] == "completed")
    delta_positions = [i for i, t in enumerate(timeline) if t[0] == "llm.delta"]
    end_at = next(i for i, t in enumerate(timeline) if t[0] == "llm.delta_end")
    assert delta_positions and max(delta_positions) < record_at < end_at
    assert "".join(t[1] for t in timeline if t[0] == "llm.delta") == "Hello world"
    assert [t[2] for t in timeline if t[0].startswith("llm.")] == [0, 1, 2]  # seq: deltas, then end


def test_seal_flushes_and_refuses_more_text_but_allows_the_end() -> None:
    events: List[Dict[str, Any]] = []
    em = _emitter(events, _Clock(), flush_interval_s=10.0)
    em("a")
    em("b")  # batched
    em.seal()
    assert [e["text"] for e in events] == ["a", "b"]
    em("c")
    em.end("completed")
    assert [e["kind"] for e in events] == ["llm.delta", "llm.delta", "llm.delta_end"]


def test_a_kill_reinvoke_gets_its_own_call_and_the_first_ends_cancelled() -> None:
    """REVIEW/23: a kill aimed at an effect that had already finished lands on
    this attempt and the runtime re-invokes it. The interrupted invocation's
    partial text must be closed as cancelled, and the re-invoke streams under
    its own call id."""
    from abstractruntime.core.effect_cancellation import EffectKilled

    events: List[Dict[str, Any]] = []
    invocations = {"n": 0}

    def llm_handler(run, effect, default_next_node):
        del run, effect, default_next_node
        invocations["n"] += 1
        cb = current_effect_delta_callback()
        if invocations["n"] == 1:
            cb("partial ans", "content")
            raise EffectKilled()  # unattributed: meant for an effect that already finished
        cb("The full answer.", "content")
        return EffectOutcome.completed({"content": "The full answer."})

    state, ledger, run_id = _drive({EffectType.LLM_CALL: llm_handler}, runtime_ns={"stream": True}, sink=events.append)
    assert state.status.value == "completed", state.error
    assert invocations["n"] == 2
    calls: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        calls.setdefault(e["call_id"], []).append(e)
    assert len(calls) == 2
    first, second = list(calls.values())
    assert [e["kind"] for e in first] == ["llm.delta", "llm.delta_end"]
    assert first[0]["text"] == "partial ans" and first[-1]["reason"] == "cancelled"
    assert "".join(e["text"] for e in second if e["kind"] == "llm.delta") == "The full answer."
    assert second[-1]["kind"] == "llm.delta_end" and second[-1]["reason"] == "completed"
    # The first call is closed before the second one starts.
    assert events.index(first[-1]) < events.index(second[0])
    [step_id] = _llm_call_step_ids(ledger, run_id)
    assert first[0]["call_id"] == step_id and second[0]["call_id"] == f"{step_id}:reinvoke"
