"""The react visit arm — the gateway's ONLY entity arm
(`build_visit_workflow(react_middle=...)`) — sends THE history window
(runtime re-gate blocker, 2026-09-28; the reviewer's attack
gate-runtime2/attack/test_react_visit_window.py, adopted).

The arm's transcript is the durable `context.messages`, which BRIDGE appends
to and the adapter's react loop sends on every call. BRIDGE sets
`_runtime.history_window_tokens`; the adapter sends
`window_transcript(context.messages)` — the newest whole turns up to 50k
tokens, a tool result kept with its turn — and records the receipt at
`_runtime.session_history`. The stored transcript stays whole. (A middle that
records no window is recorded as `window_applied: false`: test_entity_history_window.py.)

Needs abstractagent (it depends on the runtime, not the other way round):
present in the framework workspace, absent from a standalone checkout.
"""
from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")
pytest.importorskip("abstractagent")

from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn  # noqa: E402
from abstractagent.logic.react import ReActLogic  # noqa: E402
from abstractcore.tools import ToolDefinition  # noqa: E402

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    HARVEST_NODE,
    VISITOR_WAIT_KEY,
    ReactMiddle,
    build_visit_workflow,
)
from abstractruntime.memory.token_budget import estimate_message_tokens  # noqa: E402
from abstractruntime.session_history import HISTORY_REPLAY_MAX_TOKENS  # noqa: E402

BIG = "tide " * 8000  # ~10k tokens

# The 2026-08-01 incident's tool result: 494,932 chars of PNG bytes as text.
_MONSTER_HEADER = (
    "[read_file]: --- shared/Screenshot_2026-08-01_at_5.25.26_PM.png "
    "(5104148 bytes; truncated view) ---\n\x89PNG\r\n\x1a\n"
)
_MONSTER = (_MONSTER_HEADER + ("�PNG\x89garbage" * (494_932 // 12 + 1)))[:494_932]


def _make_home(tmp_path: Path, slug: str) -> Path:
    from abstractmemory import DEFAULT_SPARK_TEMPLATE, MemorySystem, SQLiteJournal, SQLiteTripleStore, engram, lint_spark

    home_dir = tmp_path / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}@home-test"
    spark = _copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = slug.capitalize()
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store, journal = SQLiteTripleStore(db), SQLiteJournal(db)
    assert engram(MemorySystem(store=store, journal=journal), spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


class _ScriptedLLM:
    def __init__(self, replies: List[Dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        self.calls.append(json.loads(json.dumps(effect.payload or {}, default=str)))
        return EffectOutcome.completed(dict(self.replies.pop(0) if self.replies else {"content": "..."}))


def _poison_tools(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
    results = [
        {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True,
         "output": _MONSTER[len("[read_file]: "):], "error": None}
        for tc in ((effect.payload or {}).get("tool_calls") or [])
    ]
    return EffectOutcome.completed({"mode": "executed", "results": results})


def _lane(tmp_path: Path, replies: List[Dict[str, Any]], *, middle: Any = None):
    ert = open_entity_runtime(
        _make_home(tmp_path, "windowling"),
        extra_handlers={EffectType.LLM_CALL: (llm := _ScriptedLLM(replies)), EffectType.TOOL_CALLS: _poison_tools},
    )
    if middle is None:
        react = create_react_workflow(
            logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={})]),
            workflow_id="entity-visit-react", provider="stub", model="stub",
            allowed_tools=["read_file"], final_next_node=HARVEST_NODE,
        )
        middle = ReactMiddle(nodes=react.nodes, entry="reason", reset_turn=reset_react_turn)
    wf = build_visit_workflow(
        ert.home, participants=["person:albou"], idle_seconds=3600,
        model_info={"provider": "test", "model": "scripted"}, visit_id="visit-w1", react_middle=middle,
    )
    rid = ert.runtime.start(workflow=wf, vars={}, session_id="visit-w")
    assert ert.runtime.tick(workflow=wf, run_id=rid, max_steps=50).status == RunStatus.WAITING
    return ert, wf, rid, llm


def _say(ert, wf, rid, text):
    return ert.runtime.resume(workflow=wf, run_id=rid, wait_key=VISITOR_WAIT_KEY,
                              payload={"text": text, "speaker": "person:albou"}, max_steps=200)


def _tok(msgs):
    return sum(estimate_message_tokens(m) for m in msgs)


def test_react_visit_history_is_windowed_at_50k(tmp_path: Path) -> None:
    ert, wf, rid, llm = _lane(tmp_path, [{"content": f"ok {i}", "tool_calls": []} for i in range(20)])
    try:
        for i in range(12):
            assert _say(ert, wf, rid, f"turn {i} " + BIG).status == RunStatus.WAITING
        wire = llm.calls[-1]["messages"]
        state = ert.runtime.get_state(rid)
        report = state.vars["_runtime"]["session_history"]
        # The request is the window: at most 50k tokens of history plus the
        # labeled notice, the oldest turns dropped whole.
        assert _tok(wire) <= HISTORY_REPLAY_MAX_TOKENS + 200
        assert report["max_tokens"] == HISTORY_REPLAY_MAX_TOKENS and report["dropped_messages"] > 0
        assert report["window_applied"] is True
        assert report["replayed_messages"] == len(wire)
        assert wire[0]["role"] == "user" and "#TRUNCATION" in wire[0]["content"]
        assert wire[-1]["role"] == "user" and "turn 11 " in wire[-1]["content"]
        # The stored transcript stays whole: every turn, every reply.
        stored = state.vars["context"]["messages"]
        assert len(stored) == 24 and "turn 0 " in stored[0]["content"]
        assert all("#TRUNCATION" not in str(m.get("content")) for m in stored)
    finally:
        ert.close()


def test_react_visit_poisoned_turn_drops_out_once_a_newer_turn_exists(tmp_path: Path) -> None:
    replies = [
        {"content": "", "tool_calls": [{"name": "read_file", "arguments": {"path": "shared/x.png"}, "call_id": "c1"}]},
        {"content": "I looked at the file.", "tool_calls": []},
    ] + [{"content": f"fine {i}", "tool_calls": []} for i in range(10)]
    ert, wf, rid, llm = _lane(tmp_path, replies)
    try:
        assert _say(ert, wf, rid, "read the screenshot").status == RunStatus.WAITING
        for i in range(5):
            assert _say(ert, wf, rid, f"small talk {i}").status == RunStatus.WAITING
            wire = llm.calls[-1]["messages"]
            # The poisoned turn (~124k tokens even clamped at 200k chars) no
            # longer fits beside a newer turn: dropped whole, with its call.
            assert [m for m in wire if m.get("role") == "tool"] == []
            assert not any(m.get("tool_calls") for m in wire)
            assert _tok(wire) < 5_000
        state = ert.runtime.get_state(rid)
        assert state.vars["_runtime"]["session_history"]["dropped_messages"] == 4
        stored_tool = [m for m in state.vars["context"]["messages"] if m.get("role") == "tool"]
        assert len(stored_tool) == 1 and len(stored_tool[0]["content"]) == len(_MONSTER)  # durable truth kept
    finally:
        ert.close()



# ------------------------------------------------ user-role messages mid-turn
# The loop adds user-role messages INSIDE a turn: the `[User response]` after
# an ask_user wait, and drained operator guidance. The window keeps the turn in
# progress whole from `_runtime.history_window_turn_start` (the visitor's
# message), so neither splits off the turn's own question and tool results
# (runtime re-gate 3; the reviewer's gate-runtime3/attack/test_midturn.py).

BIGTOOL = " ".join(f"line{i}" for i in range(30000))  # ~290k chars: the two results alone pass the window
QUESTION = "QUESTION-SENTINEL please compare files a and b"
_TWO_READS = {"content": "", "tool_calls": [
    {"name": "read_file", "arguments": {"path": "a"}, "call_id": "r1"},
    {"name": "read_file", "arguments": {"path": "b"}, "call_id": "r2"},
]}


def _midturn_lane(tmp_path: Path, replies: List[Dict[str, Any]], tools: Any):
    from abstractagent.logic.builtins import ASK_USER_TOOL

    ert = open_entity_runtime(
        _make_home(tmp_path, "midturn"),
        extra_handlers={EffectType.LLM_CALL: (llm := _ScriptedLLM(replies)), EffectType.TOOL_CALLS: tools},
    )
    react = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={}), ASK_USER_TOOL]),
        workflow_id="entity-visit-react", provider="stub", model="stub",
        allowed_tools=["read_file", "ask_user"], final_next_node=HARVEST_NODE,
    )
    wf = build_visit_workflow(
        ert.home, participants=["person:albou"], idle_seconds=3600, model_info={"provider": "test", "model": "scripted"},
        visit_id="visit-m1", react_middle=ReactMiddle(nodes=react.nodes, entry="reason", reset_turn=reset_react_turn),
    )
    rid = ert.runtime.start(workflow=wf, vars={}, session_id="visit-m")
    assert ert.runtime.tick(workflow=wf, run_id=rid, max_steps=50).status == RunStatus.WAITING
    return ert, wf, rid, llm


def _big_reads(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
    return EffectOutcome.completed({"mode": "executed", "results": [
        {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": BIGTOOL, "error": None}
        for tc in ((effect.payload or {}).get("tool_calls") or [])
    ]})


def _assert_whole_turn(wire: List[Dict[str, Any]], report: Dict[str, Any], marker: str) -> None:
    assert report["window_applied"] is True
    assert any(QUESTION in str(m.get("content")) for m in wire), "the visitor's question of THIS turn was dropped"
    # Both tool results of the turn ride (each clamped by the agent's 200k-char guard).
    assert sum(1 for m in wire if m.get("role") == "tool" and str(m.get("content")).startswith("[read_file]: line0 ")) == 2
    assert any(marker in str(m.get("content")) for m in wire)
    # The turn alone is past the window: kept whole, and said so; the older
    # small turn is what was dropped.
    assert report["oversize_turn_kept"] is True and report["dropped_messages"] == 2


def test_midturn_ask_after_big_tool_reads_keeps_the_visitors_question(tmp_path: Path) -> None:
    replies = [{"content": "hi", "tool_calls": []}, _TWO_READS,
               {"content": "", "tool_calls": [{"name": "ask_user", "arguments": {"question": "Which section?"}, "call_id": "q1"}]},
               {"content": "Section 2 says X.", "tool_calls": []}]
    ert, wf, rid, llm = _midturn_lane(tmp_path, replies, _big_reads)
    try:
        assert _say(ert, wf, rid, "hello").status == RunStatus.WAITING
        st = _say(ert, wf, rid, QUESTION)
        assert st.status == RunStatus.WAITING and st.waiting.wait_key != VISITOR_WAIT_KEY  # parked on the ask
        stored_before = json.dumps(ert.runtime.get_state(rid).vars["context"]["messages"])
        st = ert.runtime.resume(workflow=wf, run_id=rid, wait_key=st.waiting.wait_key,
                                payload={"response": "section 2"}, max_steps=200)
        assert st.status == RunStatus.WAITING
        state = ert.runtime.get_state(rid)
        stored = state.vars["context"]["messages"]
        assert stored_before[:-1] in json.dumps(stored), "the stored transcript was rewritten"
        _assert_whole_turn(llm.calls[-1]["messages"], state.vars["_runtime"]["session_history"], "section 2")
    finally:
        ert.close()


def test_midturn_operator_guidance_keeps_the_visitors_question(tmp_path: Path) -> None:
    def reads_then_guidance(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        # Guidance delivered while the tools ran; the next reason drains it
        # into the transcript as a user-role interjection.
        run.vars.setdefault("_runtime", {}).setdefault("inbox", []).append(
            {"role": "system", "content": "GUIDANCE-SENTINEL focus on section 2"})
        return _big_reads(run, effect, dnn)

    replies = [{"content": "hi", "tool_calls": []}, _TWO_READS, {"content": "Section 2 says X.", "tool_calls": []}]
    ert, wf, rid, llm = _midturn_lane(tmp_path, replies, reads_then_guidance)
    try:
        assert _say(ert, wf, rid, "hello").status == RunStatus.WAITING
        assert _say(ert, wf, rid, QUESTION).status == RunStatus.WAITING
        state = ert.runtime.get_state(rid)
        stored = state.vars["context"]["messages"]
        assert any(m.get("role") == "user" and "GUIDANCE-SENTINEL" in str(m.get("content")) for m in stored)
        _assert_whole_turn(llm.calls[-1]["messages"], state.vars["_runtime"]["session_history"], "GUIDANCE-SENTINEL")
    finally:
        ert.close()
