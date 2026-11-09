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

# ---------------------------------------------------------------- §10.6 he wakes when the maintenance is done;
# §10.8 the pass yields to a visitor and the visit's close hands the same sleep back (contract §4.4)


def test_he_wakes_when_the_maintenance_is_done_with_no_nap(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    naps: List[float] = []
    llm = _ScriptedLLM(["Enough.\n```rest\nfull\n```", "R1.", "Back.", "R2."])
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=0, ticks_per_day=1, max_ticks=2, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir,
                      on_sleep=lambda **kw: {"formed": False}, sleep_fn=naps.append, out=lambda s: None).run()
    assert report.sleeps == 1 and report.days == 2
    assert sum(naps) == 0, "no fixed nap after the pass (was rest_minutes x 60 = 1800 s)"
    wake = [h for h in _history(home_dir) if h.get("state") == "awake"][-1]
    assert wake.get("written_by") == "self"
    for h in _history(home_dir):
        if h.get("state") == "asleep":
            assert not h.get("wake_at"), f"no clock stamped on his sleep: {h}"


def test_a_visit_during_his_sleep_hands_back_the_same_sleep_whose_maintenance_resumes(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    passes: List[str] = []
    saved: Dict[str, Any] = {}

    def on_sleep(**kw):
        passes.append(str(read_entity_state(home_dir).get("written_by")))
        if len(passes) == 1:
            # The door opens mid-pass: it records the pre-visit state and writes the visiting posture; the engine
            # yields at its next stage.
            saved.update(read_entity_state(home_dir))
            write_entity_state(home_dir, "asleep", mode="visiting", written_by="visit-door",
                               reason="in conversation with person:laurent (auto-yield)")
        return {"formed": False}

    polls = {"n": 0}

    def sleeper(_s: float) -> None:
        if str(read_entity_state(home_dir).get("mode") or "") != "visiting":
            return
        polls["n"] += 1
        if polls["n"] == 3:  # the door closes: the interrupted sleep is restored as it was
            write_entity_state(home_dir, "asleep", reason=str(saved.get("reason")), mode=str(saved.get("mode")),
                               written_by=str(saved.get("written_by")))

    llm = _ScriptedLLM(["Enough.\n```rest\nfull\n```", "R1.", "Back after the sleep.", "R2."])
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=0, ticks_per_day=1, max_ticks=2, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, on_sleep=on_sleep,
                      sleep_fn=sleeper, out=lambda s: None).run()
    assert passes == ["self", "self"], "the pass yielded to the visit, then resumed in the same sleep"
    assert report.sleeps == 1, "one sleep, not two (visit->sleep is the same sleep)"
    assert report.days == 2 and read_entity_state(home_dir)["state"] == "awake"
    assert _markers(home_dir) == [("personal", "sleep", "self_elected"), ("sleep", "personal", "maintenance_done")]


def test_the_night_has_no_clock_and_its_voice_yields_to_a_visitor(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory

    from abstractruntime.identity.life import build_consolidator

    home_dir = _make_home(tmp_path)
    seen: Dict[str, Any] = {}

    def fake_sleep_pass(ms, *, scopes, owner_id, should_continue=None, **kw):
        seen["continue"] = should_continue()
        return {"maintenance": {}, "dream": {"created": True, "signals": ["tension"], "dream_record_id": "d1"}}

    monkeypatch.setattr(abstractmemory, "sleep_pass", fake_sleep_pass)
    narrated: List[int] = []
    import abstractruntime.identity.night_voice as nv

    monkeypatch.setattr(nv, "run_night_voice", lambda *a, **k: narrated.append(1) or {"narrated": True})
    # A self sleep whose (legacy) stamped wake_at is long past: the night still runs - no clock ends it.
    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self",
                       wake_at="2000-01-01T00:00:00+00:00")
    build_consolidator(home_dir, narrator_llm_factory=lambda: object(), out=lambda s: None)()
    assert seen["continue"] is True
    assert narrated == [1]
    # A visitor knocked during the pass: the voice (a model call) yields, the dream stands.
    def knocking_pass(ms, *, scopes, owner_id, should_continue=None, **kw):
        write_entity_state(home_dir, "asleep", mode="visiting", written_by="visit-door", reason="auto-yield")
        return {"maintenance": {}, "dream": {"created": True, "signals": ["tension"], "dream_record_id": "d2"}}

    monkeypatch.setattr(abstractmemory, "sleep_pass", knocking_pass)
    result = build_consolidator(home_dir, narrator_llm_factory=lambda: object(), out=lambda s: None)()
    assert narrated == [1] and result["formed"] is True


# ---------------------------------------------------------------- §10.2 / §6.4 the Work toggle: one runtime-owned
# bucket, the sidecar migration, the gate, the durable pause at the next effect boundary, preserved across restarts


def _sidecar(home_dir: Path, rec: Dict[str, Any]) -> Path:
    p = home_dir.parent / ".life_phase" / f"{home_dir.name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def test_work_toggle_bucket_is_field_merged_with_one_writer(tmp_path: Path) -> None:
    import yaml

    from abstractruntime.identity.life import read_personal_grant, read_work_toggle, write_work_toggle

    home_dir = _make_home(tmp_path)
    assert read_work_toggle(home_dir)["enabled"] is True and read_work_toggle(home_dir)["source"] == "default"
    doc = yaml.safe_load((home_dir / "phases.yaml").read_text(encoding="utf-8"))
    doc["work"] = None
    doc["visit"] = {"tools": ["read_file"]}
    (home_dir / "phases.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    out = write_work_toggle(home_dir, enabled=False, written_by="person:laurent")
    assert out == {"enabled": False, "written_by": "person:laurent", "written_at": out["written_at"],
                   "source": "phases.yaml"}
    doc = yaml.safe_load((home_dir / "phases.yaml").read_text(encoding="utf-8"))
    assert doc["visit"] == {"tools": ["read_file"]}, "other sections ride through"
    assert read_personal_grant(home_dir)["mode"] == "until_revoked", "the personal grant is untouched"
    with pytest.raises(ValueError):
        write_work_toggle(home_dir, enabled="no", written_by="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        write_work_toggle(home_dir, enabled=True, written_by="")
    (home_dir / "phases.yaml").write_text("{corrupt: [", encoding="utf-8")
    with pytest.raises(ValueError):
        write_work_toggle(home_dir, enabled=True, written_by="x")
    assert read_work_toggle(home_dir)["enabled"] is False, "a corrupt file fails closed"


def test_sidecar_migration_is_idempotent_then_the_sidecar_is_projection_only(tmp_path: Path) -> None:
    from abstractruntime.identity.life import (
        migrate_phase_sidecar,
        read_resume_to,
        read_work_toggle,
        work_enabled,
    )

    home_dir = _make_home(tmp_path)
    side = _sidecar(home_dir, {"rev": 4, "work": False, "resume_to": "personal"})
    # Before the migration the runtime reads the gateway's sidecar.
    assert read_work_toggle(home_dir)["source"] == "gateway-sidecar" and work_enabled(home_dir) is False
    assert read_resume_to(home_dir) == {"phase": "personal", "written_by": None, "written_at": None,
                                        "source": "gateway-sidecar"}
    first = migrate_phase_sidecar(home_dir, migrated_by="gateway:test")
    assert first["work"] == "migrated" and first["resume_to"] == "migrated"
    toggle = read_work_toggle(home_dir)
    assert toggle["source"] == "phases.yaml" and toggle["enabled"] is False
    assert toggle["written_by"].startswith("gateway:test")
    assert read_resume_to(home_dir)["phase"] == "personal"
    before = (home_dir / "phases.yaml").read_text(encoding="utf-8")
    assert migrate_phase_sidecar(home_dir) == {"sidecar": str(side), "work": "present", "resume_to": "present"}
    assert (home_dir / "phases.yaml").read_text(encoding="utf-8") == before, "the second run writes nothing"
    # From now on the sidecar is a projection: a stale value there changes nothing.
    _sidecar(home_dir, {"rev": 5, "work": True, "resume_to": None})
    assert work_enabled(home_dir) is False and read_resume_to(home_dir)["phase"] == "personal"
    assert migrate_phase_sidecar(home_dir)["work"] == "present"


def test_the_gate_skips_work_while_work_is_off(tmp_path: Path) -> None:
    from abstractruntime.identity.life import read_day_gate, write_work_toggle

    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    write_work_toggle(home_dir, enabled=False, written_by="person:laurent")
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "personal" and "Work is off" in decision["note"]
    write_personal_grant(home_dir, mode="disabled")
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "idle" and decision["cause"] == "work_off"
    write_work_toggle(home_dir, enabled=True, written_by="person:laurent")
    assert read_day_gate(home_dir)["phase"] == "work"


def test_work_off_pauses_the_work_day_at_its_next_effect_boundary_then_resumes(tmp_path: Path) -> None:
    """Work off during a work day: the in-flight call finishes, the tool its reply elected does not run, the day
    closes at the tick boundary (no tick 2), the order stands and the pause is recorded; own time follows while the
    grant stands (work->personal#operator). Work on again: the next boundary resumes the work, and he is told."""
    from abstractruntime.identity.life import read_work_pause, write_work_toggle

    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    turned_off = lambda: write_work_toggle(home_dir, enabled=False, written_by="person:laurent")  # noqa: E731
    turned_on = lambda: write_work_toggle(home_dir, enabled=True, written_by="person:laurent")  # noqa: E731
    llm = _HookLLM(
        ["Let me note it.\n\n```tool name=write_file path=note.md\nshelf plan\n```",  # Work goes off during it
         "R1.",
         "A free thought.", "The tick's look-back.", "R2.",  # own time; Work comes back on during it
         "Done.\n```work\ndone: sorted\n```", "R3."],
        hooks={0: turned_off, 3: turned_on},
    )
    lines: List[str] = []
    report = LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=3, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lines.append).run()
    assert not (home_dir / "workspace" / "note.md").exists(), "the elected tool did not run"
    assert report.records[0].tools == []
    assert any("Work was turned off - the work pauses here" in ln for ln in lines), lines
    assert report.days == 3 and report.ticks == 3
    assert _markers(home_dir) == [("work", "personal", "operator"), ("personal", "work", "operator")]
    assert "Your work resumes" in report.records[2].cue
    assert read_work_pause(home_dir) is None, "the pause is marked resumed"
    pause = json.loads((home_dir / "work_pause.json").read_text(encoding="utf-8"))
    assert pause["reason"] == "work_off" and pause["resumed_at"]
    assert "sorted" in (home_dir / "work_order.done.md").read_text(encoding="utf-8")


def test_toggles_and_work_orders_survive_a_restart_and_his_sleep(tmp_path: Path) -> None:
    from abstractruntime.identity.life import read_personal_grant, read_work_toggle, write_work_toggle

    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    write_work_toggle(home_dir, enabled=False, written_by="person:laurent")
    llm = _ScriptedLLM(["Enough.\n```rest\nfull\n```", "R1.", "Back.", "Look-back.", "R2."])
    LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=2, rest_minutes=0.001,
             stop_file=home_dir / "STOP", state_home=home_dir, on_sleep=lambda **kw: {"formed": False},
             sleep_fn=lambda s: None, out=lambda s: None).run()
    # His sleep and the loop's end changed neither toggle nor the order.
    assert read_work_toggle(home_dir)["enabled"] is False
    assert read_personal_grant(home_dir)["mode"] == "until_revoked"
    assert (home_dir / "work_order.md").read_text(encoding="utf-8") == "Sort the shelf."
    # A new loop (a restart) reads the same toggles: Work still off -> personal, the order still waits.
    llm2 = _ScriptedLLM(["Thinking.", "Look-back.", "R."])
    r2 = LifeLoop(_phase_factory(home_dir, llm2), tick_seconds=1, ticks_per_day=1, max_ticks=1, rest_minutes=30,
                  stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                  out=lambda s: None).run()
    assert r2.ticks == 1 and (home_dir / "work_order.md").exists()
    write_work_toggle(home_dir, enabled=True, written_by="person:laurent")
    llm3 = _ScriptedLLM(["Done.\n```work\ndone: sorted\n```", "R."])
    LifeLoop(_phase_factory(home_dir, llm3), tick_seconds=1, ticks_per_day=1, max_ticks=1, rest_minutes=30,
             stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None, out=lambda s: None).run()
    assert not (home_dir / "work_order.md").exists() and (home_dir / "work_order.done.md").exists()


def test_a_work_turn_cut_by_work_off_names_its_cause(tmp_path: Path) -> None:
    from test_visitor_at_the_door import _ELECTS_A_WRITE, _LLM, _session

    home_dir = _make_home(tmp_path)
    llm = _LLM([_ELECTS_A_WRITE, "SURPLUS - never asked"])
    s = _session(home_dir, llm)
    s.yield_check = lambda: "Work was turned off"
    try:
        reply, report = s.turn("the task")
    finally:
        s.home.close()
    assert not (home_dir / "workspace" / "note.md").exists() and report.tools == []
    assert "[tool call not run: write_file - Work was turned off]" in reply
    assert any(n.startswith("#NOTE Work was turned off: 1 tool call(s) not run") for n in report.notices)


# ---------------------------------------------------------------- the kill switch's runtime half (gateway slice
# findings 2026-11-09): hard_stop_loop signals only a live own-time loop incarnation and says what it did


def _stand_in_loop(argv_mark: str = "-m abstractruntime.identity.life"):
    import subprocess

    # A process whose command line carries the loop module's argv, like spawn_loop_process's child.
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)", argv_mark])


def _status_for(home_dir: Path, pid: int, token: Any = None, phase: str = "day") -> None:
    from datetime import datetime, timezone

    rec = {"phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(), "pid": pid}
    if token:
        rec["pid_started_at"] = token
    (home_dir / "loop_status").write_text(json.dumps(rec) + "\n", encoding="utf-8")


def test_hard_stop_kills_the_loop_and_never_records_the_caller(tmp_path: Path) -> None:
    import os

    from abstractruntime.identity.life import _pid_start_time, hard_stop_loop

    home_dir = _make_home(tmp_path)
    proc = _stand_in_loop()
    try:
        _status_for(home_dir, proc.pid, _pid_start_time(proc.pid))
        first = hard_stop_loop(home_dir, reason="drill", requested_by="admin")
        assert first["killed"] is True and first["pid"] == proc.pid and first["signal"] == "SIGTERM"
        assert first["reason"] == "killed" and first["requested_reason"] == "drill"
        assert proc.poll() is not None
        raw = json.loads((home_dir / "loop_status").read_text(encoding="utf-8"))
        assert raw.get("pid") != os.getpid(), "the status never names the calling process"
        assert raw["phase"] == "stopped" and raw["stopped_by"] == "kill_switch"
        # A second stop finds nothing to signal - and never signals the caller.
        second = hard_stop_loop(home_dir, reason="again")
        assert second["killed"] is False and second["signal"] is None
        assert second["reason"] in ("no live process at the recorded pid", "no loop pid recorded")
    finally:
        if proc.poll() is None:
            proc.kill()


def test_hard_stop_refuses_the_callers_own_pid_a_recycled_pid_and_a_foreign_process(tmp_path: Path) -> None:
    import os

    from abstractruntime.identity.life import _pid_start_time, hard_stop_loop, read_loop_status

    home_dir = _make_home(tmp_path)
    # The 2026-11-09 shape: a status that names the caller (written by an earlier hard stop from this process).
    _status_for(home_dir, os.getpid(), _pid_start_time(os.getpid()), phase="between")
    own = hard_stop_loop(home_dir)
    assert own["killed"] is False and own["signal"] is None and "calling process" in own["reason"]
    assert read_loop_status(home_dir).get("pid") is None, "the bogus pid is cleared, never kept"
    foreign = _stand_in_loop(argv_mark="--not-the-loop")
    try:
        _status_for(home_dir, foreign.pid, _pid_start_time(foreign.pid))
        out = hard_stop_loop(home_dir)
        assert out["killed"] is False and "not an own-time loop" in out["reason"]
        assert foreign.poll() is None, "a process that is not the loop is never signalled"
        _status_for(home_dir, foreign.pid, "Mon Jan  1 00:00:00 2001")
        out = hard_stop_loop(home_dir)
        assert out["killed"] is False and "recycled" in out["reason"] and foreign.poll() is None
    finally:
        foreign.kill()


def test_the_operators_start_reaches_the_spawned_loop(tmp_path: Path, monkeypatch) -> None:
    """spawn_loop_process(first_day_unconditional=True) - the operator's own act - passes the flag the CLI turns
    into LifeLoop(first_day_unconditional=True); a restore (the default) does not."""
    import subprocess

    from abstractruntime.identity.life import spawn_loop_process

    home_dir = _make_home(tmp_path)
    seen: List[List[str]] = []

    class _Proc:
        pid = 999999

        def poll(self):
            return None

    def fake_popen(argv, *a, **k):
        seen.append(list(argv))
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    for flag in (True, False):
        try:
            spawn_loop_process(home_dir, provider="lmstudio", model="m", first_day_unconditional=flag)
        except Exception:  # noqa: BLE001 - only the argv handed to the child matters here
            pass
    assert len(seen) == 2, seen
    assert "--first-day-unconditional" in seen[0] and "--first-day-unconditional" not in seen[1]


# ---------------------------------------------------------------- Codex review 2026-11-09


def test_a_sleep_handed_back_while_he_rests_resumes_its_maintenance_never_spins(tmp_path: Path) -> None:
    """Codex #1: with nothing pulling, the gate chose the personal rest; the rest returned at once because he was
    asleep (a visit's close had handed his sleep back) and the loop repeated forever without resuming the pass.
    A held state (sleep, visit, pause) now goes to the state gate before any leg is chosen."""
    from test_entity_life_loop import _make_home_no_drives

    home_dir = _make_home_no_drives(tmp_path)
    passes: List[str] = []
    polls = {"n": 0}

    def sleeper(_s: float) -> None:
        polls["n"] += 1
        if polls["n"] == 2:  # the door's close restores the sleep it interrupted
            write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming",
                               written_by="self")
        if polls["n"] > 200:
            raise AssertionError("the loop spun without resuming his sleep")

    def on_sleep(**kw):
        passes.append(str(read_entity_state(home_dir).get("written_by")))
        (home_dir / "STOP").write_text("halt", encoding="utf-8")
        return {"formed": False}

    report = LifeLoop(_factory_for(home_dir, _ScriptedLLM([])), tick_seconds=1, ticks_per_day=1, max_ticks=1,
                      rest_minutes=30, stop_file=home_dir / "STOP", state_home=home_dir, on_sleep=on_sleep,
                      sleep_fn=sleeper, out=lambda s: None).run()
    assert passes == ["self"], "the handed-back sleep's maintenance resumed"
    assert report.stopped_by == "stop_file" and read_entity_state(home_dir)["state"] == "awake"
