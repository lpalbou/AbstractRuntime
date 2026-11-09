"""A visit never fails and the entity is reachable within a minute or two
(operator 2026-11-09, aletheia incident): the visit door writes the visiting
posture and waits for the own-time loop with no limit; the loop's tick stops
starting new work the moment it sees that posture (ChatSession.yield_check),
skips the tick's look-back, and yields at the boundary right after.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.life import (  # noqa: E402
    LifeLoop,
    await_loop_quiescent,
    read_loop_status,
    write_entity_state,
)
from test_entity_life_loop import _make_home  # noqa: E402

_ELECTS_A_WRITE = "Let me keep this.\n\n```tool name=write_file path=note.md\nhello\n```"


class _LLM:
    """Scripted replies; `on_call(n)` runs after the n-th call (1-based)."""

    def __init__(self, replies: List[str], on_call=None) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.on_call = on_call

    def generate(self, *, messages, system_prompt):
        self.calls += 1

        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "SCRIPT EXHAUSTED"
        if self.on_call is not None:
            self.on_call(self.calls)
        return r


def _session(home_dir: Path, llm: _LLM) -> ChatSession:
    return ChatSession(
        open_home(home_dir), llm, participants=["entity:liveling@home-test"],
        context_window=20000, enable_tools=True, enable_workspace=True,
        phase="personal", out=lambda s: None,
    )


def test_a_waiting_visitor_stops_the_tools_the_reply_elected(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _LLM([_ELECTS_A_WRITE, "SURPLUS - the continuation must never be asked"])
    s = _session(home_dir, llm)
    s.yield_check = lambda: True
    try:
        reply, report = s.turn("your own time")
    finally:
        s.home.close()
    assert not (home_dir / "workspace" / "note.md").exists(), "the elected tool must not run"
    assert report.tools == []
    assert "[tool call not run: write_file - a visitor arrived]" in reply
    assert "[used tool: write_file]" not in reply
    assert any("a visitor is at the door" in n for n in report.notices)
    assert llm.calls == 1 and llm.replies == ["SURPLUS - the continuation must never be asked"]


def test_without_a_visitor_the_tools_run_as_before(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _LLM([_ELECTS_A_WRITE, "Kept."])
    s = _session(home_dir, llm)
    s.yield_check = lambda: False
    try:
        _reply, report = s.turn("your own time")
    finally:
        s.home.close()
    assert (home_dir / "workspace" / "note.md").exists()
    assert report.tools == ["write_file"]
    assert llm.calls == 2


def test_the_loop_yields_after_the_in_flight_call_when_a_visitor_knocks(tmp_path: Path) -> None:
    """The visitor knocks WHILE the tick's first model call runs: the elected
    tool never runs, no continuation and no tick look-back are asked for, and
    the day closes for the visit at the boundary right after."""
    home_dir = _make_home(tmp_path)
    stop = home_dir / "STOP"

    def knock(n: int) -> None:
        if n == 1:
            write_entity_state(
                home_dir, "asleep",
                reason="in conversation with person:test [visit v-1] (auto-yield)",
                mode="visiting", written_by="visit-door",
            )

    llm = _LLM([_ELECTS_A_WRITE, "SURPLUS - must stay unconsumed"], on_call=knock)
    lines: List[str] = []

    def out(line: str) -> None:
        lines.append(line)
        if "a visitor is at the door" in line:
            stop.touch()  # end the yielded-gate idle deterministically

    loop = LifeLoop(
        lambda: _session(home_dir, llm), tick_seconds=0, ticks_per_day=8, max_ticks=8,
        stop_file=stop, state_home=home_dir, sleep_fn=lambda s: None, out=out,
    )
    report = loop.run()
    assert report.ticks == 1
    assert not (home_dir / "workspace" / "note.md").exists()
    assert any("a visitor is at the door" in ln for ln in lines)
    assert llm.calls == 1 and llm.replies == ["SURPLUS - must stay unconsumed"]
    assert read_loop_status(home_dir).get("phase") != "day"


def test_await_loop_quiescent_without_a_limit_waits_for_the_yield(tmp_path: Path) -> None:
    (tmp_path / "loop_status").write_text(json.dumps({"phase": "day", "pid": os.getpid()}), encoding="utf-8")
    polls: Dict[str, Any] = {"n": 0}

    def sleeper(_s: float) -> None:
        polls["n"] += 1
        if polls["n"] == 50:  # far beyond any bound a caller could have meant
            (tmp_path / "loop_status").write_text(json.dumps({"phase": "between", "pid": os.getpid()}), encoding="utf-8")

    assert await_loop_quiescent(tmp_path, timeout_seconds=None, poll_seconds=60.0, sleep_fn=sleeper)
    assert polls["n"] == 50


def _knocking_loop(home_dir: Path, replies: List[str], *, lines: List[str], **kw):
    """A loop whose visitor knocks during the tick's first model call."""

    def knock(n: int) -> None:
        if n == 1:
            write_entity_state(
                home_dir, "asleep",
                reason="in conversation with person:test [visit v-1] (auto-yield)",
                mode="visiting", written_by="visit-door",
            )

    llm = _LLM(replies, on_call=knock)
    stop = home_dir / "STOP"
    kw.setdefault("sleep_fn", lambda s: stop.touch())  # the yielded-gate idle ends the run
    loop = LifeLoop(
        lambda: _session(home_dir, llm), stop_file=stop, state_home=home_dir, out=lines.append, **kw,
    )
    return loop, llm


def test_a_knock_on_the_days_last_tick_skips_the_day_close_look_back(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    lines: List[str] = []
    loop, llm = _knocking_loop(
        home_dir, ["A quiet thought.\nnext: more", "SURPLUS - no reflection while he waits"],
        lines=lines, tick_seconds=0, ticks_per_day=1, max_ticks=8, rest_minutes=30,
    )
    loop.run()
    assert llm.calls == 1 and llm.replies == ["SURPLUS - no reflection while he waits"]
    assert any("yielding to the visit" in ln for ln in lines)


def test_rest_elected_as_a_visitor_knocks_never_sleeps_over_the_visit(tmp_path: Path) -> None:
    from abstractruntime.identity.life import read_entity_state

    home_dir = _make_home(tmp_path)
    lines: List[str] = []
    loop, llm = _knocking_loop(
        home_dir, ["Enough for today.\n\n```rest\ntired\n```", "SURPLUS"],
        lines=lines, tick_seconds=0, ticks_per_day=8, max_ticks=8, rest_minutes=30,
    )
    loop.run()
    state = read_entity_state(home_dir)
    assert state["state"] == "asleep" and state.get("mode") == "visiting"
    assert "[visit v-1]" in str(state.get("reason") or "")
    assert any("the visit comes first" in ln for ln in lines)


def test_no_pause_between_ticks_while_a_visitor_waits(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    stop = home_dir / "STOP"
    lines: List[str] = []
    slept_before_yield: List[float] = []

    def sleeper(s: float) -> None:
        if any("a visitor is at the door" in ln for ln in lines):
            stop.touch()
        else:
            slept_before_yield.append(s)

    loop, _llm = _knocking_loop(
        home_dir, ["A quiet thought.\nnext: more"], lines=lines,
        tick_seconds=20, ticks_per_day=8, max_ticks=8, sleep_fn=sleeper,
    )
    loop.run()
    assert any("a visitor is at the door" in ln for ln in lines)
    assert slept_before_yield == []


def test_a_waiting_visitor_gets_no_correction_call(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _LLM(["[used tool: read_file] I read it.", "SURPLUS - no correction call while he waits"])
    s = _session(home_dir, llm)
    s.yield_check = lambda: True
    try:
        _reply, report = s.turn("your own time")
    finally:
        s.home.close()
    assert llm.calls == 1
    assert not any("marker imitation caught in-turn" in n for n in report.notices)


def _waiting_session(home_dir: Path, replies: List[str], *, phase: str = "personal"):
    llm = _LLM(replies)
    s = ChatSession(
        open_home(home_dir), llm, participants=["entity:liveling@home-test"],
        context_window=20000, enable_tools=True, enable_workspace=True, phase=phase, out=lambda s: None,
    )
    s.yield_check = lambda: True
    return s, llm


def test_a_waiting_visitor_gets_no_format_nudge_and_it_is_noted(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s, llm = _waiting_session(home_dir, ["Saved:\n\n```python title=coherence.py\nprint(1)\n```", "SURPLUS"])
    try:
        _reply, report = s.turn("your own time")
    finally:
        s.home.close()
    assert llm.calls == 1
    assert any("format-repair nudge skipped: a visitor is at the door" in n for n in report.notices)


def test_a_waiting_visitor_gets_no_liveness_correction_and_it_is_noted(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s, llm = _waiting_session(home_dir, ["I just fetched the latest news: all calm.", "SURPLUS"])
    try:
        _reply, report = s.turn("your own time")
    finally:
        s.home.close()
    assert llm.calls == 1
    assert any("liveness claim without a lookup left uncorrected" in n for n in report.notices)


def test_a_cut_work_turn_gets_no_speak_now_call_and_no_false_marker(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    s, llm = _waiting_session(
        home_dir,
        ["[used tool: write_file]\n```tool name=write_file path=a.md\nx\n```", "SURPLUS - no speak-now call"],
        phase="work",
    )
    try:
        reply, report = s.turn("the task")
    finally:
        s.home.close()
    assert llm.calls == 1
    assert "[used tool: write_file]" not in reply
    assert not (home_dir / "workspace" / "a.md").exists()
