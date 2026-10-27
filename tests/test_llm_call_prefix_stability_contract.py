"""Prefix-stability contract of consecutive LLM calls (local KV / prefix caches).

Operator report 2026-10-27: an agent run on a local MLX hybrid model
(`Jundot/Qwen3.8-27B-oQ4e-mtp`) re-prefilled the WHOLE prompt at every step
("prefill 16,384 / 24,552 tokens" at step 10; ledger: 10 of 10 calls `cold`).
A hybrid model restores only an EXACT stored prefix, so a single byte that moves
between two calls of a run costs the entire conversation.

What the runtime hands the LLM client for call N+1 of a tool loop must be call N
plus appended turns. The ONLY permitted per-call difference is a block of
TRAILING `system` riders (the session attachment index), which providers
exclude from their reusable snapshot (AbstractCore MLX native lane:
`MLXProvider._native_rider_tail_tokens`). Concretely, across consecutive calls:

- `system_prompt` and `tools` are byte-identical;
- the messages of call N minus its trailing system riders are a prefix of call
  N+1's messages;
- no `system` message sits between non-system messages (a rider that moved into
  the middle of the history would destabilize every later byte);
- a per-call timestamp (the grounding envelope moves every second) reaches the
  model only through the durable, stamped user turn, never through the prefix.

Driven through the real Runtime (ledger grounding pass) and the real LLM_CALL
handler (attachment injection), with a stub client capturing what it receives.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Dict, List, Optional

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_client
from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.integrations.abstractcore.session_attachments import session_memory_owner_run_id
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.turn_grounding import SYNTHETIC_MESSAGE_KEY, stamp_user_turn_grounding

pytestmark = pytest.mark.basic

SYSTEM = "## MY PERSONA\nYou are an autonomous ReAct agent (Reason -> Act -> Observe)."
TOOLS = [
    {"name": "web_search", "description": "Search the web.", "parameters": {"query": {"type": "string"}}},
    {"name": "open_attachment", "description": "Open a stored attachment.", "parameters": {"artifact_id": {"type": "string"}}},
]
SESSION = "s-prefix-stability"


def _moving_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every grounding read returns a NEW second, like a real loop does."""
    tick = itertools.count()

    def grounding(trace_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        s = next(tick)
        return {
            "local_datetime": f"2026-10-26T21:11:{s:02d}+01:00",
            "display": f"[2026-10-26 21:11:{s:02d}]",
            "source": "abstractruntime",
            "prompt_injected": False,
        }

    monkeypatch.setattr(llm_client, "_runtime_grounding_metadata", grounding)


def _session_with_attachment() -> InMemoryArtifactStore:
    store = InMemoryArtifactStore()
    content = b"RIFF....WAVEfmt "
    store.store(
        content,
        content_type="audio/wav",
        run_id=session_memory_owner_run_id(SESSION),
        tags={"kind": "attachment", "path": "ask.wav", "filename": "ask.wav", "session_id": SESSION,
              "sha256": hashlib.sha256(content).hexdigest()},
    )
    return store


def _transcript(iterations: int) -> List[Dict[str, Any]]:
    """The durable transcript of a tool loop after `iterations` reasoning steps."""
    task = {"role": "user", "content": "Create a report on neural fields I can listen to."}
    msgs: List[Dict[str, Any]] = [task]
    stamp_user_turn_grounding(msgs, grounding={"local_datetime": "2026-10-26T21:11:47+01:00",
                                               "display": "[2026-10-26 21:11:47]"})
    for i in range(1, iterations):
        msgs.append({"role": "assistant", "content": f"Step {i}: searching.",
                     "tool_calls": [{"id": f"call_{i}", "name": "web_search", "arguments": {"query": f"q{i}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"[web_search]: result {i}"})
        msgs.append({"role": "user", "content": f"[loop] iteration {i + 1} of 20.", SYNTHETIC_MESSAGE_KEY: "loop_tail"})
    return msgs


def _call(store: InMemoryArtifactStore, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    class _StubLLM:
        def generate(self, **kwargs: Any) -> Dict[str, Any]:
            captured.update(json.loads(json.dumps(kwargs, default=str)))
            return {"content": "ok", "tool_calls": [], "metadata": {}}

    def node(run: RunState, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="reason",
            effect=Effect(type=EffectType.LLM_CALL, payload={
                "prompt": "", "messages": json.loads(json.dumps(messages)), "system_prompt": SYSTEM,
                "tools": TOOLS, "params": {"temperature": 0.2},
            }),
            next_node=None,
        )

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: make_llm_call_handler(llm=_StubLLM(), artifact_store=store)},
    )
    wf = WorkflowSpec(workflow_id="wf-prefix", entry_node="reason", nodes={"reason": node})
    run_id = runtime.start(workflow=wf, session_id=SESSION)
    runtime.tick(workflow=wf, run_id=run_id)
    assert captured, "non-vacuity: the LLM client was never called"
    return captured


def _without_trailing_system(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = len(msgs)
    while keep and msgs[keep - 1].get("role") == "system":
        keep -= 1
    return msgs[:keep]


def _stable_params(req: Dict[str, Any]) -> Dict[str, Any]:
    """Generation params minus per-call plumbing (trace ids, progress/cancel callbacks)."""
    params = dict(req.get("params") or {})
    for key in ("trace_metadata", "on_progress", "cancel_event"):
        params.pop(key, None)
    return params


def test_consecutive_tool_loop_calls_only_append_and_riders_only_trail(monkeypatch: pytest.MonkeyPatch) -> None:
    _moving_clock(monkeypatch)
    store = _session_with_attachment()
    calls = [_call(store, _transcript(k)) for k in (1, 2, 3, 4)]

    # Non-vacuity: the rider this contract is about is really there, at the tail.
    for req in calls:
        msgs = req["messages"]
        assert msgs[-1]["role"] == "system" and "ask.wav" in str(msgs[-1]["content"])

    for prev, cur in zip(calls, calls[1:]):
        assert cur.get("system_prompt") == prev.get("system_prompt"), "system prompt moved between calls"
        assert cur.get("tools") == prev.get("tools"), "tool list moved between calls"
        assert _stable_params(cur) == _stable_params(prev)
        head = _without_trailing_system(prev["messages"])
        assert len(head) < len(prev["messages"])
        assert cur["messages"][: len(head)] == head, "call N (minus trailing riders) is not a prefix of call N+1"
        assert len(cur["messages"]) > len(head)

    for req in calls:
        msgs = req["messages"]
        body = _without_trailing_system(msgs)
        lead = 0
        while lead < len(body) and body[lead].get("role") == "system":
            lead += 1
        assert all(m.get("role") != "system" for m in body[lead:]), "a system message sits mid-history"


def test_a_moving_timestamp_never_reaches_the_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clock moves on every read; nothing outside the riders may change with it."""
    _moving_clock(monkeypatch)
    store = _session_with_attachment()
    first = _call(store, _transcript(3))
    again = _call(store, _transcript(3))  # same transcript, later second
    assert again["system_prompt"] == first["system_prompt"]
    assert _without_trailing_system(again["messages"]) == _without_trailing_system(first["messages"])
