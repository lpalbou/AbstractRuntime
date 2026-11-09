"""A visit never fails (operator 2026-11-09): a turn whose model call fails answers the error for THAT turn and
parks - the visit continues and the next turn is answered (gateway slice finding: a failed mirrored turn ended the
visited entity's whole visit run). Both arms: the v0 single call and the ReAct middle the gateway runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")
sys.path.insert(0, str(Path(__file__).parent))

from abstractruntime.core.models import Effect, EffectType, RunStatus  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    HARVEST_NODE,
    VISITOR_WAIT_KEY,
    ReactMiddle,
    build_visit_workflow,
)
from test_visit_midturn_elections import _make_home  # noqa: E402


class _FlakyLLM:
    """Answers in order; a reply that is an Exception instance FAILS that call."""

    def __init__(self, replies: List[Any]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        self.calls += 1
        nxt = self.replies.pop(0) if self.replies else "..."
        if isinstance(nxt, Exception):
            return EffectOutcome.failed(str(nxt))
        return EffectOutcome.completed(dict(nxt) if isinstance(nxt, dict) else {"content": str(nxt)})


def _answers(ert: Any, run_id: str) -> List[Dict[str, Any]]:
    out = []
    for rec in ert.runtime.get_ledger(run_id):
        eff = rec.get("effect") or {}
        if eff.get("type") == "answer_user" and rec.get("status") == "completed":
            out.append(dict(eff.get("payload") or {}))
    return out


def _say(ert: Any, wf: Any, run_id: str, text: str):
    return ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                              payload={"text": text, "speaker": "person:admin"}, max_steps=300)


def _middle():
    pytest.importorskip("abstractagent")
    from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn
    from abstractagent.logic.react import ReActLogic

    react = create_react_workflow(
        logic=ReActLogic(tools=[]), workflow_id="entity-visit-react", provider="stub", model="stub",
        allowed_tools=[], final_next_node=HARVEST_NODE,
    )
    return ReactMiddle(nodes=react.nodes, entry="reason", reset_turn=reset_react_turn)


@pytest.mark.parametrize("arm", ["v0", "react"])
def test_a_failed_model_call_answers_that_turn_and_the_visit_continues(tmp_path: Path, arm: str) -> None:
    home_dir = _make_home(tmp_path)
    llm = _FlakyLLM(["First answer.", RuntimeError("provider timed out after 1800 s"), "Third answer.",
                     "Goodbye reflection."])
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    try:
        wf = build_visit_workflow(
            ert.home, participants=["person:admin"], idle_seconds=3600,
            model_info={"provider": "test", "model": "scripted"}, visit_id=f"visit-flaky-{arm}",
            react_middle=_middle() if arm == "react" else None,
        )
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id=f"visit-flaky-{arm}")
        assert ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50).status == RunStatus.WAITING
        assert _say(ert, wf, run_id, "hello").status == RunStatus.WAITING
        state = _say(ert, wf, run_id, "are you there?")
        assert state.status == RunStatus.WAITING, f"the visit failed with the turn: {state.error}"
        answers = _answers(ert, run_id)
        assert answers[-1]["level"] == "error" and "provider timed out" in answers[-1]["turn_error"]
        assert answers[-1]["message"].startswith("This turn could not be answered")
        state = _say(ert, wf, run_id, "and now?")
        assert state.status == RunStatus.WAITING
        assert _answers(ert, run_id)[-1]["message"].startswith("Third answer")
        # The failed turn is folded with an explicit outcome (Codex #5): the history alternates, the next turn
        # never stacks two user messages.
        history = list((state.vars.get("_visit") or {}).get("history") or [])
        assert [m["role"] for m in history] == ["user", "assistant"] * 3, history
        assert history[3]["content"].startswith("[this turn could not be answered: ")
        assert "provider timed out" in history[3]["content"]
        if arm == "react":
            roles = [m.get("role") for m in (state.vars.get("context") or {}).get("messages") or []]
            assert "user" in roles and not any(a == b == "user" for a, b in zip(roles, roles[1:])), roles
        state = ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                                   payload={"kind": "close"}, max_steps=400)
        assert state.status == RunStatus.COMPLETED
    finally:
        ert.close()


def test_a_failed_react_turn_keeps_what_it_already_did(tmp_path: Path) -> None:
    """Codex #5's probe: the turn ran write_file, then its next model call failed. The completed tool result stays in
    the transcript, the failed-turn outcome closes the turn, and the error answer reports the tool that really ran."""
    home_dir = _make_home(tmp_path)
    ran: List[str] = []

    def tools(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        results = []
        for tc in ((effect.payload or {}).get("tool_calls") or []):
            ran.append(str(tc.get("name")))
            results.append({"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True,
                            "output": "wrote note.md (12 bytes)", "error": None})
        return EffectOutcome.completed({"mode": "executed", "results": results})

    llm = _FlakyLLM([
        {"content": "Hello.", "tool_calls": []},
        {"content": "", "tool_calls": [{"name": "write_file", "arguments": {"path": "note.md", "content": "x"},
                                        "call_id": "w1"}]},
        RuntimeError("provider timed out after 1800 s"),
        {"content": "Back now.", "tool_calls": []},
    ])
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm, EffectType.TOOL_CALLS: tools})
    try:
        pytest.importorskip("abstractagent")
        from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn
        from abstractagent.logic.react import ReActLogic
        from abstractcore.tools import ToolDefinition

        react = create_react_workflow(
            logic=ReActLogic(tools=[ToolDefinition(name="write_file", description="Write", parameters={})]),
            workflow_id="entity-visit-react", provider="stub", model="stub",
            allowed_tools=["write_file"], final_next_node=HARVEST_NODE,
        )
        wf = build_visit_workflow(
            ert.home, participants=["person:admin"], idle_seconds=3600,
            model_info={"provider": "test", "model": "scripted"}, visit_id="visit-keeps",
            react_middle=ReactMiddle(nodes=react.nodes, entry="reason", reset_turn=reset_react_turn),
        )
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-keeps")
        assert ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50).status == RunStatus.WAITING
        assert _say(ert, wf, run_id, "hello").status == RunStatus.WAITING
        state = _say(ert, wf, run_id, "please write a note")
        assert state.status == RunStatus.WAITING, state.error
        assert ran == ["write_file"]
        answer = _answers(ert, run_id)[-1]
        assert answer["level"] == "error" and answer["tools_ran"] == ["write_file"], answer
        msgs = state.vars["context"]["messages"]
        assert any("wrote note.md" in str(m.get("content")) for m in msgs), "the completed tool result stays"
        assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"].startswith("[this turn could not be answered")
        state = _say(ert, wf, run_id, "and now?")
        assert state.status == RunStatus.WAITING and _answers(ert, run_id)[-1]["message"].startswith("Back now")
    finally:
        ert.close()
