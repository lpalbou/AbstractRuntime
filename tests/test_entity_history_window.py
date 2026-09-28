"""The entity lanes select history through THE runtime history window
(operator ruling 2026-09-28, ADR-0026; tag gate B2).

The entity chat driver (`ChatSession`) and the entity visit workflow kept only
the last 10 turns (`history[-2 * history_turns:]`) with no notice, and the
visit seeded per-message re-send caps (32,000 chars per tool result, 80,000
per message) into `_limits`. Both are gone: every raw turn is kept, and each
prompt carries `session_history.window_transcript` — the most recent 50,000
tokens of whole turns, a labeled `#TRUNCATION` notice when older turns were
dropped — with the window's receipt recorded (visit: `_runtime.session_history`
in the run; chat: `TurnReport.history_window`).
"""
from __future__ import annotations

import copy as _copy
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    HARVEST_NODE,
    VISITOR_WAIT_KEY,
    ReactMiddle,
    build_visit_workflow,
)
from abstractruntime.session_history import HISTORY_REPLAY_MAX_TOKENS, window_transcript  # noqa: E402

# Past the retired 10-turn slice.
TURNS = 14
# ~10k tokens per visitor turn: six of them overflow the 50k window.
BIG = "tide " * 8000


def _make_home(tmp_path: Path, slug: str) -> Path:
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

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
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


def test_history_turns_is_gone_from_both_entity_lanes() -> None:
    assert "history_turns" not in inspect.signature(build_visit_workflow).parameters
    assert "history_turns" not in inspect.signature(ChatSession.__init__).parameters


def test_window_transcript_keeps_whole_turns_and_announces_the_drop() -> None:
    msgs: List[Dict[str, Any]] = []
    for i in range(TURNS):
        msgs += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    whole = window_transcript(msgs)
    assert list(whole) == msgs and whole.report["dropped_messages"] == 0
    assert whole.report["max_tokens"] == HISTORY_REPLAY_MAX_TOKENS

    big = [{"role": "user", "content": BIG}, {"role": "assistant", "content": "ok"}] * 8
    window = window_transcript(big)
    assert 0 < window.report["dropped_messages"] < len(big) and window.report["dropped_messages"] % 2 == 0
    assert window[0]["role"] == "user" and window[0]["content"].startswith("[#TRUNCATION: ")
    assert window[0]["content"].endswith(BIG)  # the kept message itself is never cut
    assert set(window[0]) == {"role", "content"}  # plain messages stay plain for the model client
    assert big[0]["content"] == BIG  # the input is not mutated


# ------------------------------------------------------------ visit workflow


def _drive_visit(tmp_path: Path, slug: str, texts: List[str], *, react_middle=None):
    home_dir = _make_home(tmp_path, slug)
    calls: List[Dict[str, Any]] = []

    def llm(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        calls.append(dict(effect.payload))
        return EffectOutcome.completed({"content": f"reply {len(calls)}"})

    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    try:
        wf = build_visit_workflow(
            ert.home,
            participants=["person:albou"],
            idle_seconds=3600,
            model_info={"provider": "test", "model": "scripted"},
            react_middle=react_middle,
        )
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id=f"visit-{slug}")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        assert state.status == RunStatus.WAITING
        for text in texts:
            state = ert.runtime.resume(
                workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                payload={"text": text, "speaker": "person:albou"}, max_steps=200,
            )
            assert state.status == RunStatus.WAITING
        return calls, ert.runtime.get_state(run_id).vars
    finally:
        ert.close()


def test_visit_prompt_carries_every_turn_inside_the_window_and_records_it(tmp_path: Path) -> None:
    calls, run_vars = _drive_visit(tmp_path, "keeper", [f"turn {i}" for i in range(TURNS)])
    assert len(calls) == TURNS
    last = calls[-1]["messages"]
    assert len(last) == 2 * (TURNS - 1) + 1  # every prior turn, not the last 10
    assert last[0]["content"].endswith("turn 0") and last[1]["content"] == "reply 1"
    assert len(run_vars["_visit"]["history"]) == 2 * TURNS  # the whole visit is kept
    report = run_vars["_runtime"]["session_history"]
    assert report["replayed_messages"] == 2 * (TURNS - 1) and report["dropped_messages"] == 0
    assert report["max_tokens"] == HISTORY_REPLAY_MAX_TOKENS


def test_visit_window_drops_oldest_whole_turns_with_a_notice(tmp_path: Path) -> None:
    calls, run_vars = _drive_visit(tmp_path, "tidal", [BIG] * 8)
    last = calls[-1]["messages"]
    report = run_vars["_runtime"]["session_history"]
    assert report["dropped_messages"] > 0 and report["replayed_tokens"] <= HISTORY_REPLAY_MAX_TOKENS
    assert len(last) == report["replayed_messages"] + 1
    assert last[0]["role"] == "user" and last[0]["content"].startswith("[#TRUNCATION: ")
    assert len(run_vars["_visit"]["history"]) == 16


def test_bridge_seeds_no_per_message_char_caps(tmp_path: Path) -> None:
    seen: List[Dict[str, Any]] = []

    def reason(run: Any, ctx: Any) -> StepPlan:
        seen.append(dict(run.vars.get("_limits") or {}))
        # A contract-faithful middle sends the window and records it.
        runtime_ns = run.vars["_runtime"]
        assert runtime_ns["history_window_tokens"] == HISTORY_REPLAY_MAX_TOKENS
        runtime_ns["session_history"] = dict(window_transcript(run.vars["context"]["messages"]).report)
        temp = run.vars.setdefault("_temp", {})
        temp["final_answer"] = "seen"
        temp["turn_captures"] = {"diary_entries": [], "act_only_warnings": []}
        return StepPlan(node_id="reason", next_node=HARVEST_NODE)

    _drive_visit(tmp_path, "uncapped", ["hello"], react_middle=ReactMiddle(nodes={"reason": reason}, entry="reason"))
    assert seen and "max_tool_message_chars" not in seen[0] and "max_message_chars" not in seen[0]



def test_a_react_middle_that_records_no_window_is_recorded_not_silent(tmp_path: Path, caplog) -> None:
    """The react arm's transcript is sent by the adapter; one too old to honor
    `_runtime.history_window_tokens` (abstractagent < 0.3.17) sends the whole
    visit. The runtime cannot require its own dependent, so the turn completes
    — and the gap is stated: one warning, `window_applied: false` in the run."""
    def reason(run: Any, ctx: Any) -> StepPlan:
        assert run.vars["_runtime"]["history_window_tokens"] == HISTORY_REPLAY_MAX_TOKENS
        temp = run.vars.setdefault("_temp", {})
        temp["final_answer"] = "no window"
        temp["turn_captures"] = {"diary_entries": [], "act_only_warnings": []}
        # The loop stores its reply after the request: not part of what was sent.
        run.vars["context"]["messages"].append({"role": "assistant", "content": "no window"})
        return StepPlan(node_id="reason", next_node=HARVEST_NODE)

    with caplog.at_level("WARNING", logger="abstractruntime.identity.visit_workflow"):
        _calls, run_vars = _drive_visit(
            tmp_path, "oldadapter", ["hello"], react_middle=ReactMiddle(nodes={"reason": reason}, entry="reason")
        )
    report = run_vars["_runtime"]["session_history"]
    assert report["window_applied"] is False and report["reason"] == "agent_too_old"
    assert report["max_tokens"] == HISTORY_REPLAY_MAX_TOKENS and report["replayed_messages"] == 1
    warned = [r for r in caplog.records if "does not apply the history window" in r.getMessage()]
    assert len(warned) == 1 and "upgrade abstractagent" in warned[0].getMessage()
    assert len(run_vars["_visit"]["history"]) == 2  # the turn completed


# --------------------------------------------------------------- chat driver


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _LLM:
    def __init__(self) -> None:
        self.calls: List[List[Dict[str, Any]]] = []

    def generate(self, *, messages, system_prompt):
        self.calls.append(list(messages))
        return _Resp(f"reply {len(self.calls)}")


def _chat(tmp_path: Path, slug: str) -> tuple:
    llm = _LLM()
    session = ChatSession(
        open_home(_make_home(tmp_path, slug)), llm,
        participants=["person:albou"], session_id="s1", context_window=262144,
        enable_tools=False, out=lambda s: None,
    )
    return session, llm


def test_chat_prompt_carries_every_turn_inside_the_window(tmp_path: Path) -> None:
    session, llm = _chat(tmp_path, "chatter")
    for i in range(TURNS):
        _, report = session.turn(f"turn {i}")
    last = llm.calls[-1]
    assert len(last) == 2 * (TURNS - 1) + 1
    assert last[0]["content"] == "turn 0"
    assert len(session.history) == 2 * TURNS
    assert report.history_window["replayed_messages"] == 2 * (TURNS - 1)
    assert report.history_window["dropped_messages"] == 0


def test_chat_window_drops_oldest_whole_turns_with_a_notice(tmp_path: Path) -> None:
    session, llm = _chat(tmp_path, "tidechat")
    for _ in range(8):
        _, report = session.turn(BIG)
    last = llm.calls[-1]
    assert report.history_window["dropped_messages"] > 0
    assert len(last) == report.history_window["replayed_messages"] + 1
    assert last[0]["content"].startswith("[#TRUNCATION: ") and last[0]["content"].endswith(BIG)
    assert len(session.history) == 16


def test_window_transcript_keeps_the_turn_in_progress_whole_from_its_start() -> None:
    older = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "a"}]
    # ~30k + ~30k tokens: the turn alone is past the 50k window.
    current = [{"role": "user", "content": "QUESTION " + BIG * 3}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]},
               {"role": "tool", "content": BIG * 3}, {"role": "user", "content": "[User response]: section 2"}]
    msgs = older + current
    # Grouped at every user message, the ask answer alone is the newest turn.
    split = window_transcript(msgs)
    assert split[-1]["content"].endswith("section 2") and not any("QUESTION" in m["content"] for m in split)
    # From the turn's start it is one turn: kept whole, reported oversize.
    whole = window_transcript(msgs, current_turn_start=len(older))
    assert [m["content"] for m in whole][-4:][1:] == [m["content"] for m in current][1:]
    assert whole[-4]["content"].endswith("QUESTION " + BIG * 3)
    assert whole.report["oversize_turn_kept"] is True and whole.report["dropped_messages"] == 2
    for bad in (-1, len(msgs) + 1, True, "2"):
        with pytest.raises(ValueError):
            window_transcript(msgs, current_turn_start=bad)
