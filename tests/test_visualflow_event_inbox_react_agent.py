"""Scripted test of the generic event-inbox resident agent flow.

Loads the real flow document (abstractflow/examples/flows/event-inbox-react-agent.json)
and drives it step-by-step with a scripted LLM + a fake tool, emulating the gateway
runner's durable mailbox delivery (append envelope to `events_inbox` between ticks).

Proves the four properties the design exists for:
1. Idle = durable park on the open channel key (`evt:global:global:<mailbox>`).
2. A wake event folds into the first burst cycle's prompt.
3. INTERLEAVE: an event appended while the burst is mid-flight (between ticks,
   exactly like the runner thread does) appears in the NEXT cycle's prompt.
4. A `{kind: "stop"}` payload ends the resident; a plain answer re-parks it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from abstractruntime.core.models import EffectType, RunStatus, WaitReason
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow

FLOW_PATH = Path(__file__).resolve().parents[2] / "abstractflow" / "examples" / "flows" / "event-inbox-react-agent.json"

MAILBOX = "test-box"
WAIT_KEY = f"evt:global:global:{MAILBOX}"


def _push_mail(run_store: InMemoryRunStore, run_id: str, *, body: str, kind: str = "message", sender: str = "laurent") -> int:
    """Emulate GatewayRunner._deliver_durable_event: append envelope with per-run seq."""
    run = run_store.load(run_id)
    assert run is not None
    vars_obj = run.vars
    assert vars_obj.get("events_mailbox") == MAILBOX  # receiver declared the channel
    inbox = vars_obj.get("events_inbox")
    if not isinstance(inbox, list):
        inbox = []
        vars_obj["events_inbox"] = inbox
    seq = int(vars_obj.get("events_inbox_seq") or 0) + 1
    vars_obj["events_inbox_seq"] = seq
    inbox.append(
        {
            "event_id": f"e-{seq}",
            "name": MAILBOX,
            "scope": "global",
            "payload": {"kind": kind, "from": sender, "body": body},
            "emitted_at": "2026-07-08T00:00:00Z",
            "emitter": {"source": "external", "client_id": sender},
            "seq": seq,
        }
    )
    run_store.save(run)
    return seq


def _drive(runtime: Runtime, spec, run_id: str, *, max_ticks: int = 200):
    """Tick one step at a time until the run parks, completes, or fails."""
    state = runtime.tick(workflow=spec, run_id=run_id, max_steps=1)
    ticks = 0
    while state.status == RunStatus.RUNNING and ticks < max_ticks:
        state = runtime.tick(workflow=spec, run_id=run_id, max_steps=1)
        ticks += 1
    return state


def test_event_inbox_resident_full_lifecycle_with_interleave() -> None:
    raw = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
    spec = compile_visualflow(raw)

    run_store = InMemoryRunStore()
    journal: List[str] = []
    llm_prompts: List[str] = []

    def note(*, text: str) -> Dict[str, Any]:
        """Record a note (test tool)."""
        journal.append(str(text))
        return {"noted": text}

    interleave_armed = {"done": False}

    def llm_stub(run, effect, default_next_node):
        del default_next_node
        payload = dict(effect.payload or {})
        prompt = str(payload.get("prompt") or "")
        llm_prompts.append(prompt)
        cycle = len(llm_prompts)
        if cycle == 1:
            return EffectOutcome.completed(
                {
                    "content": None,
                    "tool_calls": [{"name": "note", "arguments": {"text": "hello"}, "call_id": "c-1"}],
                }
            )
        if cycle == 2:
            return EffectOutcome.completed(
                {
                    "content": None,
                    "tool_calls": [{"name": "note", "arguments": {"text": "weather"}, "call_id": "c-2"}],
                }
            )
        return EffectOutcome.completed({"content": "Burst report: greeted the room and covered the weather.", "tool_calls": None})

    runtime = Runtime(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm_stub,
            EffectType.TOOL_CALLS: make_tool_calls_handler(tools=MappingToolExecutor.from_tools([note])),
        },
    )

    run_id = runtime.start(
        workflow=spec,
        vars={"mailbox": MAILBOX, "task": "Serve the room.", "tools": ["note"], "max_burst_cycles": 6},
    )

    # --- 1. Empty channel: the resident parks durably on the open channel key.
    state = _drive(runtime, spec, run_id)
    assert state.status == RunStatus.WAITING
    assert state.waiting is not None and state.waiting.reason == WaitReason.EVENT
    assert state.waiting.wait_key == WAIT_KEY
    assert state.vars.get("events_mailbox") == MAILBOX  # channel declared for durable delivery

    # --- 2. Wake: durable delivery (append) + resume, as the gateway runner does.
    _push_mail(run_store, run_id, body="Say hello to the room")
    runtime.resume(workflow=spec, run_id=run_id, wait_key=WAIT_KEY, payload={"name": MAILBOX}, max_steps=0)

    # Drive the burst one step at a time; interleave a second event after the
    # first tool executed (i.e. mid-burst, between ticks - the runner thread's slot).
    state = run_store.load(run_id)
    ticks = 0
    while state.status == RunStatus.RUNNING and ticks < 300:
        state = runtime.tick(workflow=spec, run_id=run_id, max_steps=1)
        ticks += 1
        if not interleave_armed["done"] and journal == ["hello"]:
            _push_mail(run_store, run_id, body="Also mention the weather", sender="castor")
            interleave_armed["done"] = True

    # --- 3. Burst finished with a report; resident re-parked.
    assert interleave_armed["done"], "interleave never armed - first tool call did not happen"
    assert state.status == RunStatus.WAITING
    assert state.waiting is not None and state.waiting.wait_key == WAIT_KEY
    assert journal == ["hello", "weather"]

    # Cycle 1 saw the wake event; cycle 2 saw the interleaved event; cycle 3 saw none.
    assert "Say hello to the room" in llm_prompts[0]
    assert "from laurent" in llm_prompts[0]
    assert "Also mention the weather" in llm_prompts[1]
    assert "from castor" in llm_prompts[1]
    assert "(no new events)" in llm_prompts[2]
    # The interleaved event must NOT have been visible before it was sent.
    assert "weather" not in llm_prompts[0]

    # Cursor advanced past both events; burst state reset after the report.
    latest = run_store.load(run_id)
    assert int(latest.vars.get("cursor") or 0) == 2
    assert latest.vars.get("in_burst") is False
    assert latest.vars.get("trace") == ""

    # --- 4. Stop control ends the resident.
    _push_mail(run_store, run_id, body="", kind="stop", sender="operator")
    runtime.resume(workflow=spec, run_id=run_id, wait_key=WAIT_KEY, payload={"name": MAILBOX}, max_steps=0)
    state = _drive(runtime, spec, run_id)

    assert state.status == RunStatus.COMPLETED
    assert isinstance(state.output, dict)
    assert state.output.get("answer") == "Resident event agent stopped by stop event."
    # The stop burst never called the LLM again.
    assert len(llm_prompts) == 3


def test_event_inbox_resident_flush_on_budget_exhaustion() -> None:
    """A model that never concludes is flushed with a labeled #FALLBACK report and re-parks."""
    raw = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
    spec = compile_visualflow(raw)

    run_store = InMemoryRunStore()
    calls = {"n": 0}

    def busy_tool() -> str:
        """No-op tool used to keep the burst spinning."""
        return "ok"

    def llm_stub(run, effect, default_next_node):
        del run, default_next_node
        calls["n"] += 1
        return EffectOutcome.completed(
            {"content": None, "tool_calls": [{"name": "busy_tool", "arguments": {}, "call_id": f"c-{calls['n']}"}]}
        )

    runtime = Runtime(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm_stub,
            EffectType.TOOL_CALLS: make_tool_calls_handler(tools=MappingToolExecutor.from_tools([busy_tool])),
        },
    )

    run_id = runtime.start(
        workflow=spec,
        vars={"mailbox": MAILBOX, "task": "Spin.", "tools": ["busy_tool"], "max_burst_cycles": 3},
    )
    state = _drive(runtime, spec, run_id)
    assert state.status == RunStatus.WAITING  # parked, empty channel

    _push_mail(run_store, run_id, body="go")
    runtime.resume(workflow=spec, run_id=run_id, wait_key=WAIT_KEY, payload={"name": MAILBOX}, max_steps=0)
    state = _drive(runtime, spec, run_id, max_ticks=400)

    # Budget hit: burst flushed, resident re-parked (not failed, not spinning).
    assert state.status == RunStatus.WAITING
    assert calls["n"] == 3  # exactly max_burst_cycles LLM calls
    latest = run_store.load(run_id)
    assert latest.vars.get("in_burst") is False
    assert latest.vars.get("trace") == ""
