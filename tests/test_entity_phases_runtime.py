"""The runtime half of the entity phases (contract-phases v2 §10, restart #3; TARGET 2026-11-08 19:02/19:11; spec
v23 OWN-TIME-IS-A-PRIVILEGE). Each section pins one acceptance item; every test was red on the code before it."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home  # noqa: E402

from abstractruntime.identity.life import (  # noqa: E402
    LifeLoop,
    read_entity_state,
    read_loop_status,
    write_entity_state,
    write_personal_grant,
)


def _never():  # noqa: ANN202
    raise AssertionError("no session may open here")


def _history(home_dir: Path) -> List[Dict[str, Any]]:
    p = home_dir / "state_history.jsonl"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


# ---------------------------------------------------------------- §10.1 no grant: own time ends AWAKE (root 1178)


@pytest.mark.parametrize("rest_minutes", [0.0, 30.0])
def test_personal_off_ends_own_time_awake_and_visitable(tmp_path: Path, rest_minutes: float) -> None:
    """Turning personal off used to land him ASLEEP for 6 h (24/7 mode: written_by grant-gate, wake_at +6h) and the
    visit door refused him. Now own time ends awake, in both modes; nothing writes a sleep."""
    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "awake", reason="living")
    write_personal_grant(home_dir, mode="disabled")
    lines: List[str] = []
    report = LifeLoop(_never, tick_seconds=0, ticks_per_day=2, max_ticks=4, rest_minutes=rest_minutes,
                      state_home=home_dir, sleep_fn=lambda s: None, out=lines.append).run()
    assert report.stopped_by == "personal_off"
    assert report.days == 0
    state = read_entity_state(home_dir)
    assert state["state"] == "awake" and not state.get("wake_at"), state
    assert read_loop_status(home_dir).get("stopped_by") == "personal_off"
    assert not [h for h in _history(home_dir) if h.get("state") == "asleep"], _history(home_dir)


def test_mid_day_revocation_closes_the_day_and_he_stays_awake(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["One.\nnext: two", "Reflection.", "SURPLUS - never a tick 2"])
    steps: Dict[str, int] = {"n": 0}

    def sleeper(_s: float) -> None:
        steps["n"] += 1
        if steps["n"] == 1:
            write_personal_grant(home_dir, mode="disabled")

    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=5, ticks_per_day=4, max_ticks=4, rest_minutes=30,
                      state_home=home_dir, sleep_fn=sleeper, out=lambda s: None).run()
    assert report.ticks == 1 and report.stopped_by == "personal_off"
    assert llm.replies == ["SURPLUS - never a tick 2"]  # the day closed with its look-back
    assert read_entity_state(home_dir)["state"] == "awake"


@pytest.mark.parametrize("writer", ["grant-gate", "day-gate", "loop-exit", "self", "personal-cycle"])
def test_a_sleep_left_by_an_earlier_loop_is_woken_by_the_next(tmp_path: Path, writer: str) -> None:
    """A sleep the loop wrote for itself never outlives it: a new loop over the home wakes it (contract §1.2 rule 6,
    the old gate/exit landings and a self sleep left by a dead loop). An operator sleep is never touched."""
    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "asleep", reason="left behind", written_by=writer,
                       wake_at="2999-01-01T00:00:00+00:00")
    write_personal_grant(home_dir, mode="disabled")
    report = LifeLoop(_never, tick_seconds=0, ticks_per_day=1, max_ticks=2, rest_minutes=30,
                      state_home=home_dir, sleep_fn=lambda s: None, out=lambda s: None).run()
    assert report.stopped_by == "personal_off"
    assert read_entity_state(home_dir)["state"] == "awake"


def test_an_operator_sleep_is_never_woken_by_the_loop_exit(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "asleep", reason="sleep now", written_by="operator")
    (home_dir / "STOP").write_text("halt", encoding="utf-8")
    LifeLoop(_never, tick_seconds=0, ticks_per_day=1, max_ticks=1, rest_minutes=30, stop_file=home_dir / "STOP",
             state_home=home_dir, sleep_fn=lambda s: None, out=lambda s: None).run()
    state = read_entity_state(home_dir)
    assert state["state"] == "asleep" and state.get("written_by") == "operator"


def test_an_ended_loop_leaves_him_awake_never_asleep(tmp_path: Path) -> None:
    """v22 retires IDLE-IS-SLEEP: an exit over an awake state leaves him awake (idle), never a loop-exit sleep."""
    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "awake", reason="living")
    llm = _ScriptedLLM(["One.", "Reflection."])
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=1,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None).run()
    assert report.stopped_by == "max_ticks"
    state = read_entity_state(home_dir)
    assert state["state"] == "awake" and state.get("written_by") != "loop-exit", state
