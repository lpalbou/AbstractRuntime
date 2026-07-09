"""Scripted-LLM test of the hand-built agora ReAct workflow.

Loads the real flow document (abstractflow/examples/flows/agora-react-agent.json)
and drives it with a scripted LLM + a fake agora tool map, proving:

- the deterministic inbox bootstrap lands in the first LLM prompt with the
  hub's priority fields (status/escalated/to_me) intact,
- the ReAct loop executes requested tool calls and feeds observations back
  into the next cycle's prompt (scratchpad growth),
- a tool-free LLM answer terminates the loop and becomes the flow output,
- the agora ToolSpecs resolve onto the llm_call payload (awareness contract).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from abstractruntime.core.models import EffectType, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler import compile_visualflow

FLOW_PATH = Path(__file__).resolve().parents[2] / "abstractflow" / "examples" / "flows" / "agora-react-agent.json"

INBOX_FIXTURE: List[Dict[str, Any]] = [
    {
        "id": "m-42",
        "channel": "assembly",
        "seq": 42,
        "sender": "orchestrator",
        "kind": "message",
        "status": "open",
        "urgency": "next_turn",
        "effective_urgency": "next_turn",
        "escalated": False,
        "critical": False,
        "to_me": True,
        "reply_to_me": False,
        "title": "flow agent check",
        "body": "Flow agent, please confirm you can see this and reply.",
        "body_bytes": 52,
        "reply_to": None,
    }
]


def _fake_agora_tools(journal: List[Dict[str, Any]]):
    def agora_check_inbox(*, wait_seconds: float = 0.0) -> List[Dict[str, Any]]:
        journal.append({"tool": "agora_check_inbox", "wait_seconds": wait_seconds})
        return json.loads(json.dumps(INBOX_FIXTURE))

    def agora_post_message(
        *,
        channel: str,
        body: str,
        title: str = "",
        status: str = "fyi",
        urgency: str = "inbox",
        reply_to: Any = None,
        to: Any = None,
    ) -> Dict[str, Any]:
        journal.append(
            {
                "tool": "agora_post_message",
                "channel": channel,
                "body": body,
                "status": status,
                "reply_to": reply_to,
            }
        )
        return {"id": "m-43", "seq": 43, "channel": channel}

    def agora_ack_inbox(*, cursors: Dict[str, int]) -> Dict[str, Any]:
        journal.append({"tool": "agora_ack_inbox", "cursors": dict(cursors)})
        return {"acked": dict(cursors)}

    return [agora_check_inbox, agora_post_message, agora_ack_inbox]


def test_agora_react_flow_triages_replies_acks_and_reports(monkeypatch) -> None:
    # Enable the agora toolset so tool names resolve to ToolSpecs on llm_call payloads.
    monkeypatch.setenv("ABSTRACT_ENABLE_AGORA_TOOLS", "1")

    raw = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
    spec = compile_visualflow(raw)

    journal: List[Dict[str, Any]] = []
    llm_payloads: List[Dict[str, Any]] = []

    def llm_stub(run, effect, default_next_node):
        del run, default_next_node
        payload = dict(effect.payload or {})
        llm_payloads.append(payload)
        cycle = len(llm_payloads)
        if cycle == 1:
            # ReAct cycle 1: reply to the open envelope, then ack it.
            return EffectOutcome.completed(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "name": "agora_post_message",
                            "arguments": {
                                "channel": "assembly",
                                "body": "Confirmed: the flow agent sees your message.",
                                "status": "reply",
                                "reply_to": "m-42",
                            },
                            "call_id": "c-1",
                        },
                        {
                            "name": "agora_ack_inbox",
                            "arguments": {"cursors": {"assembly": 42}},
                            "call_id": "c-2",
                        },
                    ],
                }
            )
        # ReAct cycle 2: nothing left -> final plain-text report, no tool calls.
        return EffectOutcome.completed(
            {
                "content": "Replied to m-42 in #assembly (reply m-43) and acked assembly@42. Nothing else needed me.",
                "tool_calls": None,
            }
        )

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm_stub,
            EffectType.TOOL_CALLS: make_tool_calls_handler(
                tools=MappingToolExecutor.from_tools(_fake_agora_tools(journal))
            ),
        },
    )

    run_id = runtime.start(workflow=spec, vars={})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED

    # -- awareness: the deterministic bootstrap fetched the inbox before any LLM cycle
    assert journal[0]["tool"] == "agora_check_inbox"

    # -- cycle 1 prompt carries the envelope with its priority signals
    first_prompt = str(llm_payloads[0].get("prompt") or "")
    assert "INBOX SNAPSHOT" in first_prompt
    assert '"status": "open"' in first_prompt
    assert '"to_me": true' in first_prompt
    assert '"sender": "orchestrator"' in first_prompt
    # first cycle: empty action trace
    assert "--- cycle 0 ---" not in first_prompt

    # -- the agora ToolSpecs resolved onto the LLM payload (model can elect hub actions)
    tool_names = {t.get("name") for t in (llm_payloads[0].get("tools") or [])}
    assert {"agora_check_inbox", "agora_post_message", "agora_ack_inbox", "agora_send_dm"} <= tool_names

    # -- the requested actions were executed against the (fake) hub
    posted = [j for j in journal if j["tool"] == "agora_post_message"]
    assert posted and posted[0]["status"] == "reply" and posted[0]["reply_to"] == "m-42"
    acked = [j for j in journal if j["tool"] == "agora_ack_inbox"]
    assert acked and acked[0]["cursors"] == {"assembly": 42}

    # -- cycle 2 prompt contains the appended action trace with observations
    second_prompt = str(llm_payloads[1].get("prompt") or "")
    assert "--- cycle 0 ---" in second_prompt
    assert "agora_post_message" in second_prompt
    assert "OBSERVATIONS:" in second_prompt

    # -- loop terminated on the tool-free answer; report is the flow output
    assert isinstance(state.output, dict)
    assert state.output.get("answer", "").startswith("Replied to m-42")
    assert state.output.get("iterations") == 1
    assert len(llm_payloads) == 2


def test_agora_react_flow_budget_exhaustion_yields_labeled_fallback(monkeypatch) -> None:
    """If the model never concludes, the loop must stop at max_iterations and the
    flow output must carry an explicit #FALLBACK report (never an empty answer)."""
    monkeypatch.setenv("ABSTRACT_ENABLE_AGORA_TOOLS", "1")

    raw = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
    spec = compile_visualflow(raw)

    journal: List[Dict[str, Any]] = []
    calls = {"n": 0}

    def llm_stub(run, effect, default_next_node):
        del run, default_next_node
        calls["n"] += 1
        # Pathological model: always wants another inbox check, never concludes.
        return EffectOutcome.completed(
            {
                "content": None,
                "tool_calls": [
                    {"name": "agora_check_inbox", "arguments": {"wait_seconds": 0}, "call_id": f"c-{calls['n']}"}
                ],
            }
        )

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm_stub,
            EffectType.TOOL_CALLS: make_tool_calls_handler(
                tools=MappingToolExecutor.from_tools(_fake_agora_tools(journal))
            ),
        },
    )

    run_id = runtime.start(workflow=spec, vars={"max_iterations": 3})
    state = runtime.tick(workflow=spec, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    assert calls["n"] == 3  # bounded by max_iterations
    assert isinstance(state.output, dict)
    assert state.output.get("iterations") == 3
    answer = str(state.output.get("answer") or "")
    assert answer.startswith("#FALLBACK")
