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


# ---------------------------------------------------------------- §10.3-§10.5 transitions: the gate, the self-loops,
# sleep as his decision, no sleep after a sleep, own time is the operator's privilege (spec v23)


class _HookLLM(_ScriptedLLM):
    """Scripted replies; `hooks[i]` runs just before reply i is returned (a visit knocking, a toggle, an order)."""

    def __init__(self, replies: List[str], hooks: Dict[int, Any] | None = None) -> None:
        super().__init__(replies)
        self.hooks = dict(hooks or {})
        self.n = 0

    def generate(self, *, messages, system_prompt):
        hook = self.hooks.get(self.n)
        self.n += 1
        if hook is not None:
            hook()
        return super().generate(messages=messages, system_prompt=system_prompt)


def _markers(home_dir: Path) -> List[tuple]:
    return [(h["from"], h["to"], h["cause"]) for h in _history(home_dir) if h.get("marker") == "phase_changed"]


def _phase_factory(home_dir: Path, llm: Any):
    """Like test_entity_life_loop._factory_for, but opens the day in the phase the gate decided (the loop passes it,
    as it does to the production factory)."""
    from abstractruntime.identity.chat import ChatSession, open_home
    from abstractruntime.identity.life import OWN_TIME_CONTRACT

    def _open(phase: str = "personal") -> ChatSession:
        home = open_home(home_dir)
        session = ChatSession(home, llm, participants=[home.entity_id], context_window=20000,
                              enable_workspace=True, out=lambda s: None, phase=phase)
        session.system_base += "\n\n" + OWN_TIME_CONTRACT
        return session

    return _open


LOOP_EDGES = [
    # (from, to, the cause the loop writes)
    ("personal", "personal", "personal_cycle"),
    ("work", "work", "task_complete"),
    ("work", "personal", "no_task"),
    ("work", "personal", "operator"),
    ("personal", "work", "operator"),
    ("sleep", "personal", "maintenance_done"),
    ("sleep", "work", "maintenance_done"),
    ("personal", "sleep", "self_elected"),
    ("work", "sleep", "self_elected"),
]


def test_transition_table_every_loop_edge_is_the_graphs_and_sleep_after_sleep_is_refused(tmp_path: Path) -> None:
    from abstractruntime.identity.life import _boundary_cause, read_day_gate, sleep_refusal
    from abstractruntime.identity.phase_graph import load_effective_graph

    graph, _w = load_effective_graph()
    for frm, to, cause in LOOP_EDGES:
        assert graph.legal_to(frm, to, cause) is not None, f"{frm}->{to}#{cause} missing from the vendored graph"
    home_dir = _make_home(tmp_path)
    # The causes the loop computes at a boundary are those edges.
    assert _boundary_cause("personal", "personal", home_dir) == "personal_cycle"
    assert _boundary_cause("work", "work", home_dir) == "task_complete"
    assert _boundary_cause("work", "personal", home_dir) == "no_task"
    (home_dir / "work_order.md").write_text("still standing", encoding="utf-8")
    assert _boundary_cause("work", "personal", home_dir) == "operator"
    assert _boundary_cause("personal", "work", home_dir) == "operator"
    assert _boundary_cause("sleep", "personal", home_dir) == "maintenance_done"
    assert _boundary_cause(None, "personal", home_dir) is None  # a start is the operator's act, not a loop edge
    assert _boundary_cause("visit", "personal", home_dir) is None  # a visit's close is the door's edge
    # sleep -> sleep: no edge in the graph under any cause, and the gate refuses his decision after a sleep.
    for cause in ("self_elected", "operator", "maintenance_done", "personal_cycle"):
        assert graph.legal_to("sleep", "sleep", cause) is None
    assert sleep_refusal(home_dir, last_phase="sleep") is not None
    assert sleep_refusal(home_dir, last_phase="personal") is None
    (home_dir / "work_order.md").unlink()
    after_sleep = read_day_gate(home_dir, sleep_elected="tired again", last_phase="sleep")
    assert after_sleep["phase"] == "personal" and "a sleep cannot follow a sleep" in after_sleep["note"]
    assert read_day_gate(home_dir, sleep_elected="tired", last_phase="personal")["phase"] == "sleep"


def test_personal_self_loop_chains_days_with_no_sleep_between(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["One.", "R1.", "Two.", "R2.", "Three.", "R3."])
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=3, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None).run()
    assert report.days == 3 and report.sleeps == 0
    assert _markers(home_dir) == []  # a self-loop is no phase moment
    assert not [h for h in _history(home_dir) if h.get("state") == "asleep"]


def test_work_self_loop_then_work_to_personal_directly(tmp_path: Path) -> None:
    """work -> work (another order stands when the first is done) -> personal (no task left, grant standing), with
    no sleep forced between; then personal -> work when an order arrives during own time."""
    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")

    def second_order() -> None:  # during day 1's close (its order is archived): another order stands
        (home_dir / "work_order.md").write_text("Label the boxes.", encoding="utf-8")

    def third_order() -> None:
        (home_dir / "work_order.md").write_text("One more.", encoding="utf-8")

    llm = _HookLLM(
        ["Done.\n```work\ndone: shelf sorted\n```", "R1.",
         "Done.\n```work\ndone: boxes labelled\n```", "R2.",
         "A free thought.", "The tick's look-back.", "R3.",
         "Done.\n```work\ndone: one more done\n```", "R4."],
        hooks={1: second_order, 5: third_order},
    )
    report = LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=4, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None).run()
    assert report.days == 4 and report.sleeps == 0
    assert _markers(home_dir) == [("work", "personal", "no_task"), ("personal", "work", "operator")]
    assert "one more done" in (home_dir / "work_order.done.md").read_text(encoding="utf-8")


def test_sleep_is_his_decision_and_he_returns_to_personal(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    passes: List[Any] = []

    def on_sleep(**kw):
        passes.append(read_entity_state(home_dir).get("written_by"))
        return {"formed": False}

    llm = _ScriptedLLM(["Enough for now.\n```rest\nthe day is full\n```", "R1.", "Back.", "R2."])
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=2, rest_minutes=0.001,
                      stop_file=home_dir / "STOP", state_home=home_dir, on_sleep=on_sleep,
                      sleep_fn=lambda s: None, out=lambda s: None).run()
    assert passes == ["self"] and report.sleeps == 1
    assert report.days == 2, "personal resumed after his sleep (grant standing)"
    assert _markers(home_dir) == [("personal", "sleep", "self_elected"), ("sleep", "personal", "maintenance_done")]
    sleep_rows = [h for h in _history(home_dir) if h.get("state") == "asleep"]
    assert {h.get("written_by") for h in sleep_rows} == {"self"}, "nothing but his decision slept him"


def test_personal_revoked_during_his_sleep_wakes_him_idle_never_personal(tmp_path: Path) -> None:
    """Spec v23 OWN-TIME-IS-A-PRIVILEGE: the interruption (his sleep) returns to personal only while the grant
    stands; revoked meanwhile, he ends idle - awake, visitable - and the loop ends."""
    home_dir = _make_home(tmp_path)

    def on_sleep(**kw):
        write_personal_grant(home_dir, mode="disabled")
        return {"formed": False}

    llm = _ScriptedLLM(["Enough.\n```rest\nfull\n```", "R1.", "NEVER - no personal after the revocation"])
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=3, rest_minutes=0.001,
                      stop_file=home_dir / "STOP", state_home=home_dir, on_sleep=on_sleep,
                      sleep_fn=lambda s: None, out=lambda s: None).run()
    assert report.days == 1 and report.stopped_by == "personal_off"
    assert read_entity_state(home_dir)["state"] == "awake"
    assert llm.replies == ["NEVER - no personal after the revocation"]


def test_a_rest_elected_as_personal_is_revoked_never_sleeps_him(tmp_path: Path) -> None:
    """PERSONAL-OFF-NEVER-SLEEPS: his decision came from own time the operator revoked in the same tick - the gate
    drops it; own time ends awake."""
    home_dir = _make_home(tmp_path)
    slept: List[int] = []
    llm = _HookLLM(["Enough.\n```rest\nfull\n```", "R1."],
                   hooks={0: lambda: write_personal_grant(home_dir, mode="disabled")})
    lines: List[str] = []
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=3, rest_minutes=0.001,
                      stop_file=home_dir / "STOP", state_home=home_dir,
                      on_sleep=lambda **kw: slept.append(1) or {"formed": False},
                      sleep_fn=lambda s: None, out=lines.append).run()
    assert slept == [] and report.sleeps == 0
    assert report.stopped_by == "personal_off" and read_entity_state(home_dir)["state"] == "awake"
    assert any("his own time was turned off" in ln for ln in lines), lines


def _visit_during_tick(home_dir: Path, *, revoke: bool = False, close_after_polls: int = 3):
    """A visitor knocks during the tick (the door writes the visiting posture); after a few polls the door closes
    and restores awake (contract §4.4). `revoke`: the operator turns personal off during the visit."""
    polls = {"n": 0, "armed": False}

    def knock() -> None:
        write_entity_state(home_dir, "asleep", mode="visiting", written_by="visit-door",
                           reason="in conversation with person:laurent (auto-yield)")
        polls["armed"] = True

    def sleeper(_s: float) -> None:
        if not polls["armed"]:
            return
        polls["n"] += 1
        if polls["n"] == 1 and revoke:
            write_personal_grant(home_dir, mode="disabled")
        if polls["n"] == close_after_polls:
            write_entity_state(home_dir, "awake", written_by="visit-door",
                               reason="visitor session ended (person:laurent; 1 turn)")
            polls["armed"] = False

    return knock, sleeper


@pytest.mark.parametrize("revoke", [False, True])
def test_personal_visit_close_returns_to_personal_only_while_granted(tmp_path: Path, revoke: bool) -> None:
    home_dir = _make_home(tmp_path)
    knock, sleeper = _visit_during_tick(home_dir, revoke=revoke)
    llm = _HookLLM(["One - a knock arrives.", "The salvaged look-back of day 1.", "Back to my own thread.", "R2."],
                   hooks={0: knock})
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=2, max_ticks=2, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=sleeper,
                      out=lambda s: None).run()
    if revoke:
        assert report.days == 1 and report.stopped_by == "personal_off", "revoked meanwhile: idle, never personal"
        assert read_entity_state(home_dir)["state"] == "awake"
    else:
        assert report.days == 2 and report.ticks == 2, "the close returned him to personal"
    assert _markers(home_dir) == [], "the visit's edges are the door's; no loop marker"


def test_work_visit_close_resumes_the_work(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    knock, sleeper = _visit_during_tick(home_dir)
    llm = _HookLLM(["Working - a knock.", "The salvaged look-back of day 1.", "Done.\n```work\ndone: sorted\n```",
                    "R2."], hooks={0: knock})
    report = LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=2, max_ticks=2, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=sleeper,
                      out=lambda s: None).run()
    assert report.days == 2 and report.ticks == 2
    assert "sorted" in (home_dir / "work_order.done.md").read_text(encoding="utf-8"), "the work resumed and finished"


def test_work_sleep_wake_returns_to_the_standing_work(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    llm = _ScriptedLLM(["Tired.\n```rest\nlong task\n```", "R1.", "Done.\n```work\ndone: sorted\n```", "R2."])
    report = LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=2, rest_minutes=0.001,
                      stop_file=home_dir / "STOP", state_home=home_dir, on_sleep=lambda **kw: {"formed": False},
                      sleep_fn=lambda s: None, out=lambda s: None).run()
    assert report.sleeps == 1 and report.days == 2
    assert _markers(home_dir) == [("work", "sleep", "self_elected"), ("sleep", "work", "maintenance_done")]
    assert (home_dir / "work_order.done.md").exists(), "the order survived his sleep and finished"


def test_nothing_pulls_he_rests_awake_and_a_work_order_ends_the_rest(tmp_path: Path) -> None:
    from test_entity_life_loop import _make_home_no_drives

    home_dir = _make_home_no_drives(tmp_path)
    naps: List[float] = []

    def sleeper(s: float) -> None:
        naps.append(s)
        if len(naps) == 4:
            (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")

    llm = _ScriptedLLM(["Done.\n```work\ndone: sorted\n```", "R1."])
    report = LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=1, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=sleeper,
                      out=lambda s: None).run()
    rest_naps = [n for n in naps if n == 5.0]  # the rest polls (the work day's tick pause is 1 s)
    assert len(rest_naps) == 4 and sum(naps) < 30 * 60, "the order ended the rest early"
    assert report.ticks == 1 and report.sleeps == 0
    assert not [h for h in _history(home_dir) if h.get("state") == "asleep"], "resting is never sleep"


def test_the_operators_start_opens_the_first_day_unconditionally(tmp_path: Path) -> None:
    from test_entity_life_loop import _make_home_no_drives

    home_dir = _make_home_no_drives(tmp_path)
    llm = _ScriptedLLM(["A first look around.", "R1."])
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=5, rest_minutes=0,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None, first_day_unconditional=True).run()
    assert report.days == 1 and report.ticks == 1
    assert report.stopped_by == "settled_desk", "after the start's day nothing pulls: supervised runs end"


def test_due_maintenance_reaches_his_cue_as_facts(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory

    home_dir = _make_home(tmp_path)
    monkeypatch.setattr(abstractmemory, "maintenance_due", lambda *a, **k: {
        "due": True, "reasons": ["x"], "new_records": 14, "duplicate_groups": 2, "isolated": 0})
    llm = _ScriptedLLM(["One.", "R1."])
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=1,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None).run()
    cue = report.records[0].cue
    assert "14 new record(s) since your last sleep, 2 group(s) of records sharing a title" in cue, cue
    assert "rest" in cue and "owed" not in cue

