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
    read_personal_grant,
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
    # Spec v25 removed work->personal#operator with the work toggle: no loop path names it.
    assert graph.legal_to("work", "personal", "operator") is None
    home_dir = _make_home(tmp_path)
    # The causes the loop computes at a boundary are those edges.
    assert _boundary_cause("personal", "personal", home_dir) == "personal_cycle"
    assert _boundary_cause("work", "work", home_dir) == "task_complete"
    assert _boundary_cause("work", "personal", home_dir) == "no_task"
    (home_dir / "work_order.md").write_text("still standing", encoding="utf-8")
    assert _boundary_cause("work", "personal", home_dir) == "no_task", "never the removed operator edge"
    # A job still queued and the work landing removed by the blueprint: no landing out of work is legal -> idle,
    # never personal (even with his own time granted and remembered).
    skipped = read_day_gate(home_dir, last_phase="work", return_to="personal", skip_phases=frozenset({"work"}))
    assert skipped["phase"] == "idle" and skipped["cause"] == "edge_removed", skipped
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
    after_sleep = read_day_gate(home_dir, sleep_elected="tired again", last_phase="sleep", return_to="personal")
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


# ---------------------------------------------------------------- CONTRACT v3 D1: there is no work toggle. Work starts
# when a job is queued; the toggle bucket, its sidecar key and the work-off pause (a24350a) are retired and never read


def _sidecar(home_dir: Path, rec: Dict[str, Any]) -> Path:
    p = home_dir.parent / ".life_phase" / f"{home_dir.name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def _old_work_off(home_dir: Path) -> None:
    """A home a24350a left with Work OFF: the phases.yaml `work` bucket and the gateway sidecar's `work` key."""
    import yaml

    doc = yaml.safe_load((home_dir / "phases.yaml").read_text(encoding="utf-8"))
    doc["work"] = {"enabled": False, "written_by": "person:laurent", "written_at": "2026-11-09T15:00:00+00:00"}
    (home_dir / "phases.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    _sidecar(home_dir, {"rev": 4, "work": False})


def test_an_old_work_off_is_never_read_a_standing_order_runs_its_work_day(tmp_path: Path) -> None:
    """D1: the work toggle is gone. A home that still carries Work off (the old bucket and the sidecar key) runs the
    standing order like any other: the gate chooses work, the day works it, the verdict archives it."""
    from abstractruntime.identity.life import read_day_gate

    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    _old_work_off(home_dir)
    assert read_day_gate(home_dir, last_phase="personal")["phase"] == "work"
    write_personal_grant(home_dir, mode="disabled")
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "work" and decision["cause"] == "work_order", decision
    llm = _ScriptedLLM(["Done.\n```work\ndone: sorted\n```", "R1."])
    report = LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=3, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None).run()
    assert report.days == 1 and "sorted" in (home_dir / "work_order.done.md").read_text(encoding="utf-8")


def test_the_sidecar_migration_carries_resume_to_only_and_leaves_an_old_work_bucket_untouched(tmp_path: Path) -> None:
    import yaml

    from abstractruntime.identity.life import migrate_phase_sidecar, read_resume_to, write_resume_to

    home_dir = _make_home(tmp_path)
    _old_work_off(home_dir)
    side = _sidecar(home_dir, {"rev": 4, "work": False, "resume_to": "personal"})
    # Before the migration the runtime reads the gateway's sidecar.
    assert read_resume_to(home_dir) == {"phase": "personal", "written_by": None, "written_at": None,
                                        "source": "gateway-sidecar"}
    first = migrate_phase_sidecar(home_dir, migrated_by="gateway:test")
    assert first == {"sidecar": str(side), "resume_to": "migrated"}, first
    assert read_resume_to(home_dir)["phase"] == "personal"
    before = (home_dir / "phases.yaml").read_text(encoding="utf-8")
    assert migrate_phase_sidecar(home_dir) == {"sidecar": str(side), "resume_to": "present"}
    assert (home_dir / "phases.yaml").read_text(encoding="utf-8") == before, "the second run writes nothing"
    # From now on the sidecar is a projection: a stale value there changes nothing.
    _sidecar(home_dir, {"rev": 5, "work": True, "resume_to": None})
    assert read_resume_to(home_dir)["phase"] == "personal"
    # The old work bucket rides through every write untouched and unread (it is not this runtime's any more).
    write_resume_to(home_dir, None, written_by="gateway:test")
    doc = yaml.safe_load((home_dir / "phases.yaml").read_text(encoding="utf-8"))
    assert doc["work"]["enabled"] is False and doc["resume_to"]["phase"] is None
    assert read_personal_grant(home_dir)["mode"] == "until_revoked", "the personal grant is untouched"
    with pytest.raises(ValueError):
        write_resume_to(home_dir, "sleep", written_by="x")
    with pytest.raises(ValueError):
        write_resume_to(home_dir, None, written_by="")
    (home_dir / "phases.yaml").write_text("{corrupt: [", encoding="utf-8")
    with pytest.raises(ValueError):
        write_resume_to(home_dir, "personal", written_by="x")


def test_no_work_toggle_is_read_anywhere_in_the_runtime() -> None:
    """D1 grep pin: no reader or writer of a work toggle, its sidecar key or the work-off pause survives in the
    runtime's code (the vendored phase spec is the Entity owner's document and is checked byte for byte elsewhere)."""
    import re

    import abstractruntime

    root = Path(abstractruntime.__file__).parent
    pattern = re.compile(
        r"work_enabled|work_toggle|WORK_BUCKET|work_pause|WORK_PAUSE|Work (is|was turned) off|work_paused"
        r"|[\[(]\s*[\"']work[\"']\s*[\])]"
    )
    hits = [
        f"{path.relative_to(root)}:{n}: {line.strip()}"
        for path in sorted(root.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == [], "\n".join(hits)


def test_the_order_and_the_grant_survive_his_sleep_and_a_restart(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    llm = _ScriptedLLM(["Tired.\n```rest\nlong task\n```", "R1.", "Working.", "Look-back.", "R2."])
    LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=2, rest_minutes=0.001,
             stop_file=home_dir / "STOP", state_home=home_dir, on_sleep=lambda **kw: {"formed": False},
             sleep_fn=lambda s: None, out=lambda s: None).run()
    # His sleep and the loop's end changed neither the grant nor the order.
    assert read_personal_grant(home_dir)["mode"] == "until_revoked"
    assert (home_dir / "work_order.md").read_text(encoding="utf-8") == "Sort the shelf."
    # A new loop (a restart) finds the same order and works it.
    llm2 = _ScriptedLLM(["Done.\n```work\ndone: sorted\n```", "R."])
    LifeLoop(_phase_factory(home_dir, llm2), tick_seconds=1, ticks_per_day=1, max_ticks=1, rest_minutes=30,
             stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None, out=lambda s: None).run()
    assert not (home_dir / "work_order.md").exists() and (home_dir / "work_order.done.md").exists()


# ---------------------------------------------------------------- the kill switch's runtime half (gateway slice
# findings 2026-11-09; Codex review 2026-11-09 #4/#7): hard_stop_loop signals only a verified own-time loop of THIS
# home, fails closed when anything is unknown, and says killed only on a confirmed exit


def _fake_loop(tmp_path: Path, home_dir: Path, *, home_arg: Any = None):
    """A process launched EXACTLY like spawn_loop_process's child (`<python> -m abstractruntime.identity.life --home
    <home> ...`), whose module is a stand-in that only sleeps (found first on its PYTHONPATH)."""
    import os
    import subprocess

    mods = tmp_path / "fakemods" / "abstractruntime" / "identity"
    mods.mkdir(parents=True, exist_ok=True)
    (mods.parent / "__init__.py").write_text("", encoding="utf-8")
    (mods / "__init__.py").write_text("", encoding="utf-8")
    (mods / "life.py").write_text("import time\ntime.sleep(300)\n", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "abstractruntime.identity.life", "--home", str(home_arg or home_dir),
         "--provider", "stub"],
        env={**os.environ, "PYTHONPATH": str(tmp_path / "fakemods")}, cwd=str(tmp_path),
    )


def _status_for(home_dir: Path, pid: int, token: Any = None, phase: str = "day") -> None:
    from datetime import datetime, timezone

    rec = {"phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(), "pid": pid}
    if token:
        rec["pid_started_at"] = token
    (home_dir / "loop_status").write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _settle(proc) -> None:
    import time as _t

    deadline = _t.monotonic() + 5
    while _t.monotonic() < deadline:
        from abstractruntime.identity.life import _pid_command

        if "--home" in (_pid_command(proc.pid) or ""):
            return
        _t.sleep(0.05)


def test_hard_stop_kills_this_homes_loop_and_never_records_the_caller(tmp_path: Path) -> None:
    import os

    from abstractruntime.identity.life import _pid_start_time, hard_stop_loop

    home_dir = _make_home(tmp_path)
    proc = _fake_loop(tmp_path, home_dir)
    try:
        _settle(proc)
        _status_for(home_dir, proc.pid, _pid_start_time(proc.pid))
        first = hard_stop_loop(home_dir, reason="drill", requested_by="admin")
        assert first["killed"] is True and first["pid"] == proc.pid and first["signal"] == "SIGTERM", first
        assert first["reason"] == "killed" and first["requested_reason"] == "drill"
        assert proc.poll() is not None
        raw = json.loads((home_dir / "loop_status").read_text(encoding="utf-8"))
        assert raw.get("pid") != os.getpid(), "the status never names the calling process"
        assert raw["phase"] == "stopped" and raw["stopped_by"] == "kill_switch"
        second = hard_stop_loop(home_dir, reason="again")
        assert second["killed"] is False and second["signal"] is None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_hard_stop_fails_closed_on_anything_but_this_homes_verified_loop(tmp_path: Path) -> None:
    import os
    import subprocess

    from abstractruntime.identity.life import _pid_start_time, hard_stop_loop, read_loop_status

    home_dir = _make_home(tmp_path)
    # The 2026-11-09 shape: a status naming the caller.
    _status_for(home_dir, os.getpid(), _pid_start_time(os.getpid()), phase="between")
    own = hard_stop_loop(home_dir)
    assert own["killed"] is False and own["signal"] is None and "calling process" in own["reason"]
    assert read_loop_status(home_dir).get("pid") is None, "the bogus pid is cleared, never kept"
    # Codex #4's probe: the module name as a LATER argument is not a loop invocation.
    probe = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)", "-m abstractruntime.identity.life"])
    other_home = _fake_loop(tmp_path, home_dir, home_arg=tmp_path / "another-home")
    (tmp_path / "x").mkdir()
    mine = _fake_loop(tmp_path / "x", home_dir)
    try:
        import time as _t

        _settle(other_home)
        _settle(mine)
        _t.sleep(0.3)  # the probe's interpreter is up
        for p, why in ((probe, "not this home's own-time loop"), (other_home, "not this home's own-time loop")):
            _status_for(home_dir, p.pid, _pid_start_time(p.pid))
            out = hard_stop_loop(home_dir)
            assert out["killed"] is False and out["signal"] is None and why in out["reason"], out
            assert p.poll() is None, "never signalled"
        # This home's loop with NO start-time token: the incarnation is unknown - fail closed.
        _status_for(home_dir, mine.pid, None)
        out = hard_stop_loop(home_dir)
        assert out["killed"] is False and "no start-time token" in out["reason"] and mine.poll() is None
        # A recycled pid (the token does not match).
        _status_for(home_dir, mine.pid, "Mon Jan  1 00:00:00 2001")
        out = hard_stop_loop(home_dir)
        assert out["killed"] is False and "recycled" in out["reason"] and mine.poll() is None
    finally:
        for p in (probe, other_home, mine):
            if p is not None and p.poll() is None:
                p.kill()


def test_killed_only_on_a_confirmed_exit_never_on_a_denied_permission(tmp_path: Path, monkeypatch) -> None:
    """Codex #7: after SIGTERM the process stops answering signal 0 with EPERM (a permission change); that is not a
    death - killed stays False and no SIGKILL goes to an incarnation that cannot be re-verified."""
    import os
    import signal as _signal

    from abstractruntime.identity import life as life_mod
    from abstractruntime.identity.life import _pid_start_time, hard_stop_loop

    home_dir = _make_home(tmp_path)
    proc = _fake_loop(tmp_path, home_dir)
    real_kill = os.kill
    sent: List[int] = []
    state = {"term": False}

    def fake_kill(pid, sig):
        if pid != proc.pid:
            return real_kill(pid, sig)
        if sig == _signal.SIGTERM:
            sent.append(sig)
            state["term"] = True
            return None  # delivered, but the process carries on
        if sig == 0 and state["term"]:
            raise PermissionError(1, "Operation not permitted")
        if sig == _signal.SIGKILL:
            sent.append(sig)
            return None
        return real_kill(pid, sig)

    try:
        _settle(proc)
        _status_for(home_dir, proc.pid, _pid_start_time(proc.pid))
        monkeypatch.setattr(os, "kill", fake_kill)
        out = hard_stop_loop(home_dir)
        assert out["killed"] is False, out
        assert sent == [_signal.SIGTERM], "no SIGKILL to an incarnation that cannot be re-verified"
        assert "exit not confirmed" in out["reason"]
    finally:
        monkeypatch.undo()
        proc.kill()


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
            spawn_loop_process(home_dir, provider="lmstudio", model="m", first_day_unconditional=flag,
                               return_to="personal" if flag else None)
        except Exception:  # noqa: BLE001 - only the argv handed to the child matters here
            pass
    assert len(seen) == 2, seen
    assert "--first-day-unconditional" in seen[0] and "--first-day-unconditional" not in seen[1]
    # Codex #2: the run's remembered return reaches the child; a host that predates it passes none (the home's
    # resume_to decides).
    assert seen[0][seen[0].index("--return-to") + 1] == "personal"
    assert seen[1][seen[1].index("--return-to") + 1] == "none"


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


@pytest.mark.parametrize("last_phase", [None, "work", "sleep", "visit"])
def test_drives_alone_never_admit_personal(tmp_path: Path, last_phase: Any) -> None:
    """Codex #2 (spec v23 OWN-TIME-IS-A-PRIVILEGE): the grant armed and drives pulling are not enough - personal
    needs the operator's start, a continuation of personal, or a remembered return to it."""
    from abstractruntime.identity.life import read_day_gate, write_resume_to

    home_dir = _make_home(tmp_path)  # grant armed, one standing drive
    # Right after a work day with nothing queued the same refusal is stated as jobs_done (CONTRACT v3 D1).
    no_return = "jobs_done" if last_phase == "work" else "no_return"
    refused = read_day_gate(home_dir, last_phase=last_phase)
    assert refused["phase"] == "idle" and refused["cause"] == no_return, refused
    assert read_day_gate(home_dir, last_phase=last_phase, first_day=True)["phase"] == "personal"
    assert read_day_gate(home_dir, last_phase="personal")["phase"] == "personal"  # continuation
    assert read_day_gate(home_dir, last_phase=last_phase, return_to="personal")["phase"] == "personal"
    assert read_day_gate(home_dir, last_phase=last_phase, return_to="work")["cause"] == no_return
    # The remembered return persisted by the gateway (sidecar today, the runtime bucket after migration).
    _sidecar(home_dir, {"resume_to": "personal"})
    assert read_day_gate(home_dir, last_phase=last_phase)["phase"] == "personal"
    write_resume_to(home_dir, None, written_by="gateway:test")
    assert read_day_gate(home_dir, last_phase=last_phase)["cause"] == no_return


def test_a_run_started_for_work_alone_ends_idle_after_it(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    llm = _ScriptedLLM(["Done.\n```work\ndone: sorted\n```", "R1.", "NEVER - no own time was granted"])
    report = LifeLoop(_phase_factory(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=3, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None, return_to="work").run()
    assert report.days == 1 and report.stopped_by == "jobs_done"
    assert llm.replies == ["NEVER - no own time was granted"]
    assert read_entity_state(home_dir)["state"] == "awake"
    # The operator's grant during the work names the return (resume_to personal): own time follows.
    from abstractruntime.identity.life import write_resume_to

    (home_dir / "work_order.md").write_text("Label the boxes.", encoding="utf-8")
    llm2 = _HookLLM(["Done.\n```work\ndone: labelled\n```", "R1.", "My own thread.", "Look-back.", "R2."],
                    hooks={0: lambda: write_resume_to(home_dir, "personal", written_by="person:laurent")})
    r2 = LifeLoop(_phase_factory(home_dir, llm2), tick_seconds=1, ticks_per_day=1, max_ticks=2, rest_minutes=30,
                  stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                  out=lambda s: None, return_to="work").run()
    assert r2.days == 2 and r2.ticks == 2


# ---------------------------------------------------------------- CONTRACT v3 D1/D4: work is driven by the job queue
# (the standing work order and the inbox's pending/taken tasks); all jobs done -> his own time if still granted and
# remembered, else idle with the stated reason jobs_done


def _add_task(home_dir: Path, task_id: str, title: str, brief: str = "", by: str = "person:laurent") -> None:
    """A task left with him, as AbstractGateway's entity_tasks.append_task writes it."""
    with (home_dir / "task_inbox.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "added", "task_id": task_id, "title": title, "brief": brief,
                             "origin": "operator", "at": "2026-11-09T18:00:00+00:00", "by": by}) + "\n")


def _task_status(home_dir: Path, task_id: str) -> List[tuple]:
    rows = [json.loads(ln) for ln in (home_dir / "task_inbox.jsonl").read_text(encoding="utf-8").splitlines()]
    return [(r["status"], r.get("note")) for r in rows if r.get("event") == "status" and r["task_id"] == task_id]


def _job_factory(home_dir: Path, llm: Any, seen: List[tuple]):
    """The production factory's shape (phase + job): a work day carries its job's contract in the system prompt."""
    from abstractruntime.identity.chat import ChatSession, open_home
    from abstractruntime.identity.life import OWN_TIME_CONTRACT, work_day_text

    def _open(phase: str = "personal", job: Any = None) -> ChatSession:
        seen.append((phase, (job or {}).get("id")))
        home = open_home(home_dir)
        session = ChatSession(home, llm, participants=[home.entity_id], context_window=20000,
                              enable_workspace=True, out=lambda s: None, phase=phase)
        session.system_base += "\n\n" + (work_day_text(job) if phase == "work" and job else OWN_TIME_CONTRACT)
        return session

    return _open


def test_the_job_queue_is_the_order_then_the_taken_then_the_pending_tasks(tmp_path: Path) -> None:
    from abstractruntime.identity.life import append_task_status, jobs_queued, read_day_gate, read_job_queue

    home_dir = _make_home(tmp_path)
    write_personal_grant(home_dir, mode="disabled")
    for tid in ("t1", "t2", "t3", "t4"):
        _add_task(home_dir, tid, f"Task {tid}")
    append_task_status(home_dir, "t3", "taken", by="entity:liveling")
    append_task_status(home_dir, "t1", "done", by="person:laurent")
    append_task_status(home_dir, "t4", "parked", by="entity:liveling")
    with (home_dir / "task_inbox.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{torn\n")
    (home_dir / "work_order.md").write_text("Sort the shelf.\nCarefully.", encoding="utf-8")
    queue = read_job_queue(home_dir)
    assert [(j["kind"], j["id"]) for j in queue["jobs"]] == [("work_order", "work_order.md"), ("task", "t3"),
                                                              ("task", "t2")]
    assert queue["jobs_queued"] == jobs_queued(home_dir) == 3 and "1 unparseable" in queue["note"]
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "work" and decision["cause"] == "work_order" and decision["jobs_queued"] == 3
    (home_dir / "work_order.md").unlink()
    decision = read_day_gate(home_dir)
    assert decision["cause"] == "task" and decision["job"]["id"] == "t3", decision
    with pytest.raises(KeyError):
        append_task_status(home_dir, "nope", "done", by="x")
    with pytest.raises(ValueError):
        append_task_status(home_dir, "t2", "finished", by="x")


def test_no_work_leg_without_a_job(tmp_path: Path) -> None:
    """Done and parked tasks, an empty order file: nothing is queued, so no work day opens - the gate never runs a
    work leg on its own (personal off: idle)."""
    from abstractruntime.identity.life import append_task_status, read_day_gate

    home_dir = _make_home(tmp_path)
    write_personal_grant(home_dir, mode="disabled")
    _add_task(home_dir, "t1", "Done already")
    _add_task(home_dir, "t2", "Parked")
    append_task_status(home_dir, "t1", "done", by="person:laurent")
    append_task_status(home_dir, "t2", "parked", by="person:laurent")
    (home_dir / "work_order.md").write_text("   \n", encoding="utf-8")
    for last in (None, "work", "personal", "sleep", "visit"):
        assert read_day_gate(home_dir, last_phase=last)["phase"] == "idle", last
    report = LifeLoop(_never, tick_seconds=0, ticks_per_day=1, max_ticks=2, rest_minutes=30, state_home=home_dir,
                      sleep_fn=lambda s: None, out=lambda s: None).run()
    assert report.days == 0 and report.stopped_by == "personal_off"


def test_a_queued_job_with_personal_off_runs_a_work_day_and_ends_idle(tmp_path: Path) -> None:
    from abstractruntime.identity.life import read_resume_to

    home_dir = _make_home(tmp_path)
    write_personal_grant(home_dir, mode="disabled")
    _add_task(home_dir, "t1", "Label the boxes", "Every box on the shelf gets a label.", by="entity:sol@home-x")
    llm = _ScriptedLLM(["Labelled.\n```work\ndone: every box labelled\n```", "R1.", "NEVER - no own time"])
    seen: List[tuple] = []
    lines: List[str] = []
    report = LifeLoop(_job_factory(home_dir, llm, seen), tick_seconds=1, ticks_per_day=1, max_ticks=3,
                      rest_minutes=30, stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lines.append, return_to=None).run()
    assert seen == [("work", "t1")] and report.days == 1
    system = llm.calls[0]["system_prompt"]
    assert "THE TASK, left by entity:sol@home-x:\nLabel the boxes\n\nEvery box on the shelf gets a label." in system
    assert "The task below was left for you" in system and "Your operator left you" not in system
    assert _task_status(home_dir, "t1") == [("taken", None), ("done", "done: every box labelled")]
    assert report.stopped_by == "jobs_done" and read_loop_status(home_dir)["stopped_by"] == "jobs_done"
    assert llm.replies == ["NEVER - no own time"]
    assert read_entity_state(home_dir)["state"] == "awake"
    assert read_resume_to(home_dir)["source"] == "default", "no own time was interrupted: nothing remembered"
    assert any("all jobs are done" in ln for ln in lines), lines


def test_a_blocked_verdict_parks_the_task_and_the_next_job_follows(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    write_personal_grant(home_dir, mode="disabled")
    _add_task(home_dir, "t1", "Fetch the key")
    _add_task(home_dir, "t2", "Open the door")
    llm = _ScriptedLLM(["Stuck.\n```work\nblocked: the key is not in the drawer\n```", "R1.",
                        "Opened.\n```work\ndone: door open\n```", "R2."])
    seen: List[tuple] = []
    report = LifeLoop(_job_factory(home_dir, llm, seen), tick_seconds=1, ticks_per_day=1, max_ticks=5,
                      rest_minutes=30, stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None, return_to=None).run()
    assert seen == [("work", "t1"), ("work", "t2")] and report.stopped_by == "jobs_done"
    assert _task_status(home_dir, "t1")[-1] == ("parked", "blocked: the key is not in the drawer")
    assert _task_status(home_dir, "t2")[-1] == ("done", "done: door open")


def test_a_job_during_his_own_time_interrupts_it_and_he_returns_to_it(tmp_path: Path) -> None:
    from abstractruntime.identity.life import read_resume_to

    home_dir = _make_home(tmp_path)  # grant armed; the run is his own time (return_to personal)
    llm = _HookLLM(["A free thought.", "The tick's look-back.", "R1.",
                    "Labelled.\n```work\ndone: labelled\n```", "R2.",
                    "Back to my thread.", "Look-back.", "R3."],
                   hooks={0: lambda: _add_task(home_dir, "t1", "Label the boxes")})
    seen: List[tuple] = []
    report = LifeLoop(_job_factory(home_dir, llm, seen), tick_seconds=1, ticks_per_day=1, max_ticks=3,
                      rest_minutes=30, stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None).run()
    assert seen == [("personal", None), ("work", "t1"), ("personal", None)] and report.days == 3
    assert _markers(home_dir) == [("personal", "work", "operator"), ("work", "personal", "no_task")]
    remembered = read_resume_to(home_dir)
    assert remembered["phase"] == "personal" and remembered["written_by"] == "loop:job-interrupts-own-time"


def test_own_time_revoked_during_the_work_ends_idle_when_the_jobs_are_done(tmp_path: Path) -> None:
    """D4: the job interrupted his own time (resume_to personal); the operator revokes it during the work - the work
    finishes (work never yields to the grant) and he ends idle (jobs_done); the remembered return is cancelled."""
    from abstractruntime.identity.life import read_resume_to

    home_dir = _make_home(tmp_path)
    _add_task(home_dir, "t1", "Label the boxes")
    llm = _HookLLM(["Half done.", "Labelled.\n```work\ndone: labelled\n```", "R1.", "NEVER - own time was revoked"],
                   hooks={0: lambda: write_personal_grant(home_dir, mode="disabled")})
    seen: List[tuple] = []
    report = LifeLoop(_job_factory(home_dir, llm, seen), tick_seconds=1, ticks_per_day=2, max_ticks=4,
                      rest_minutes=30, stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None, first_day_unconditional=True).run()
    assert seen == [("work", "t1")] and report.ticks == 2, "the revocation did not cut the work"
    assert report.stopped_by == "jobs_done" and llm.replies == ["NEVER - own time was revoked"]
    assert read_entity_state(home_dir)["state"] == "awake"
    assert read_resume_to(home_dir)["phase"] is None, "own time is cancelled, never given back"
    assert _task_status(home_dir, "t1")[-1][0] == "done"


# ---------------------------------------------------------------- CONTRACT v3 D3: Stop cancels the remembered return;
# nothing is restored after a hard stop


def test_the_kill_switch_clears_resume_to_even_with_no_loop_running(tmp_path: Path) -> None:
    from abstractruntime.identity.life import hard_stop_loop, read_resume_to, write_resume_to

    home_dir = _make_home(tmp_path)
    write_resume_to(home_dir, "personal", written_by="loop:job-interrupts-own-time")
    out = hard_stop_loop(home_dir, reason="test")
    remembered = read_resume_to(home_dir)
    assert remembered["phase"] is None and remembered["written_by"] == "kill_switch"
    assert out["killed"] is False and out["resume_to_cleared"] is True
    # A home still on the gateway sidecar: the stop's null shadows its stale "personal".
    other = tmp_path / "other"
    other_home = _make_home(other)
    _sidecar(other_home, {"resume_to": "personal"})
    hard_stop_loop(other_home)
    assert read_resume_to(other_home)["phase"] is None
    # A corrupt phases.yaml: the stop stands and says what it could not clear.
    (home_dir / "phases.yaml").write_text("{corrupt: [", encoding="utf-8")
    bad = hard_stop_loop(home_dir)
    assert bad["resume_to_cleared"] is False and "resume_to not cleared" in bad["resume_to_note"]
    assert read_loop_status(home_dir)["stopped_by"] == "kill_switch"


@pytest.mark.parametrize("operator_grants_again", [False, True])
def test_after_a_hard_stop_the_next_loop_works_its_job_and_restores_nothing(tmp_path: Path,
                                                                            operator_grants_again: bool) -> None:
    """Stop, then a job queued: the next loop works the job and ends idle (jobs_done) even when its host passes the
    old return (return_to personal) and the grant still reads armed. The operator's own act (a grant/start:
    first_day_unconditional) is what opens own time again."""
    from abstractruntime.identity.life import hard_stop_loop

    home_dir = _make_home(tmp_path)  # grant armed
    hard_stop_loop(home_dir, reason="operator stop")
    # Stop revoked own time (Codex-2 #7). The grant is armed again here, so the restore guard is what is tested:
    # without the operator's start act (first_day_unconditional) nothing of the stopped loop comes back.
    write_personal_grant(home_dir, mode="until_revoked", granted_by="person:laurent")
    _add_task(home_dir, "t1", "Label the boxes")
    llm = _ScriptedLLM(["Labelled.\n```work\ndone: labelled\n```", "R1.", "My own thread.", "Look-back.", "R2."])
    seen: List[tuple] = []
    report = LifeLoop(_job_factory(home_dir, llm, seen), tick_seconds=1, ticks_per_day=1, max_ticks=2,
                      rest_minutes=30, stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None, return_to="personal",
                      first_day_unconditional=operator_grants_again).run()
    if operator_grants_again:
        assert seen == [("work", "t1"), ("personal", None)], "the operator's grant opens own time after the job"
    else:
        assert seen == [("work", "t1")] and report.stopped_by == "jobs_done", seen
        assert llm.replies == ["My own thread.", "Look-back.", "R2."]


def test_a_task_status_appends_without_flock_where_there_is_none(tmp_path: Path, monkeypatch) -> None:
    """Cross-OS: where fcntl does not exist the status is still appended (one line), as the gateway's writer does."""
    from abstractruntime.identity.life import append_task_status

    home_dir = _make_home(tmp_path)
    _add_task(home_dir, "t1", "Label the boxes")
    monkeypatch.setitem(sys.modules, "fcntl", None)  # `import fcntl` raises ImportError
    append_task_status(home_dir, "t1", "done", by="entity:liveling", note="done: labelled")
    assert _task_status(home_dir, "t1") == [("done", "done: labelled")]


# ---------------------------------------------------------------- Codex review 2 (2026-11-09)


def test_queued_work_is_admitted_at_the_spawn_door_with_own_time_off(tmp_path: Path, monkeypatch) -> None:
    """Codex-2 #5: the spawn door enforced the personal grant unconditionally, so a job queued with own time off
    never started. A queued job is admitted whatever the grant says; nothing queued and no grant still refuses, and
    the ownership checks (one start at a time, a running loop) stand."""
    import subprocess

    from abstractruntime.identity.life import spawn_loop_process

    home_dir = _make_home(tmp_path)
    write_personal_grant(home_dir, mode="disabled")
    spawned: List[List[str]] = []

    class _Proc:
        pid = 999999

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda argv, *a, **k: spawned.append(list(argv)) or _Proc())
    with pytest.raises(RuntimeError, match="no personal time"):
        spawn_loop_process(home_dir, provider="lmstudio", model="m")
    assert spawned == []
    _add_task(home_dir, "t1", "Label the boxes")
    try:
        spawn_loop_process(home_dir, provider="lmstudio", model="m")
    except Exception:  # noqa: BLE001 - only whether the child was launched matters here
        pass
    assert len(spawned) == 1 and "--home" in spawned[0], spawned
    from abstractruntime.identity.life import loop_start_refusal

    assert loop_start_refusal(home_dir) is None


def test_the_kill_switch_revokes_own_time_once(tmp_path: Path) -> None:
    """Codex-2 #7 (runtime half): Stop cleared the return but left the grant armed, so a later restore could open
    own time again. The runtime Stop revokes personal too (the gateway's own write: mode disabled, by kill_switch),
    and a second Stop writes nothing new."""
    from abstractruntime.identity.life import hard_stop_loop, loop_start_refusal

    home_dir = _make_home(tmp_path)  # grant until_revoked
    out = hard_stop_loop(home_dir, reason="operator stop")
    assert read_personal_grant(home_dir)["mode"] == "disabled"
    assert out["personal_revoked"] is True and "personal_note" not in out
    assert loop_start_refusal(home_dir) is not None, "nothing queued and own time revoked: no start"
    before = (home_dir / "phases.yaml").read_text(encoding="utf-8")
    again = hard_stop_loop(home_dir, reason="second stop")
    assert again["personal_revoked"] is False and read_personal_grant(home_dir)["mode"] == "disabled"
    import yaml

    assert yaml.safe_load((home_dir / "phases.yaml").read_text(encoding="utf-8"))["personal"] == \
        yaml.safe_load(before)["personal"], "idempotent"


# ---------------------------------------------------------------- Codex-2 #1: one home state lock + lifecycle generation


def test_a_bucket_write_racing_stop_never_restores_what_stop_took(tmp_path: Path, monkeypatch) -> None:
    """Codex-2 #1: a resume_to write read phases.yaml, Stop revoked personal and cleared the return, then the write
    replaced the whole file with its stale copy - personal until_revoked and resume_to personal were back after Stop.
    Every phases.yaml write now reads and merges inside the home state lock, so Stop waits for the write and lands
    after it."""
    import threading

    import abstractruntime.utils.atomic_files as atomic_files
    from abstractruntime.identity.life import hard_stop_loop, read_resume_to, write_resume_to

    home_dir = _make_home(tmp_path)  # personal until_revoked
    real_write = atomic_files.atomic_write_text
    writer = {"thread": None, "fired": False}
    stopper = threading.Thread(target=lambda: hard_stop_loop(home_dir, reason="operator stop"))

    def racing_write(path, text, *a, **k):
        if threading.current_thread() is writer["thread"] and not writer["fired"]:
            writer["fired"] = True  # the writer has read the file and is about to replace it: Stop arrives now
            stopper.start()
            stopper.join(timeout=1.0)
        return real_write(path, text, *a, **k)

    monkeypatch.setattr(atomic_files, "atomic_write_text", racing_write)
    t = threading.Thread(target=lambda: write_resume_to(home_dir, "personal", written_by="loop:test"))
    writer["thread"] = t
    t.start()
    t.join(timeout=10)
    stopper.join(timeout=10)
    assert writer["fired"] and not stopper.is_alive()
    assert read_personal_grant(home_dir)["mode"] == "disabled", "Stop's revocation survived the racing write"
    assert read_resume_to(home_dir)["phase"] is None, "Stop's cleared return survived the racing write"


def test_a_write_decided_before_stop_is_refused_after_it(tmp_path: Path) -> None:
    from abstractruntime.identity.life import (
        StaleLifecycleGeneration,
        bump_lifecycle_generation,
        hard_stop_loop,
        read_lifecycle_generation,
        read_resume_to,
        write_resume_to,
    )

    home_dir = _make_home(tmp_path)
    seen = read_lifecycle_generation(home_dir)
    assert seen == 0
    out = hard_stop_loop(home_dir, reason="operator stop")
    assert out["lifecycle_generation"] == 1 == read_lifecycle_generation(home_dir)
    before = (home_dir / "phases.yaml").read_text(encoding="utf-8")
    for late in (
        lambda: write_personal_grant(home_dir, mode="until_revoked", granted_by="person:x", expected_generation=seen),
        lambda: write_resume_to(home_dir, "personal", written_by="loop:late", expected_generation=seen),
        lambda: write_entity_state(home_dir, "awake", reason="late wake", expected_generation=seen),
    ):
        with pytest.raises(StaleLifecycleGeneration):
            late()
    assert (home_dir / "phases.yaml").read_text(encoding="utf-8") == before, "nothing written"
    assert not [h for h in _history(home_dir) if h.get("reason") == "late wake"], "no late state written"
    # Current-generation writers pass; the bump returns the new value.
    write_resume_to(home_dir, None, written_by="gateway:test", expected_generation=1)
    assert bump_lifecycle_generation(home_dir, written_by="gateway:test") == 2 and read_resume_to(home_dir)["phase"] is None


def test_the_home_state_lock_is_reentrant_in_a_thread_and_times_out_across_threads(tmp_path: Path) -> None:
    import threading

    from abstractruntime.identity.life import HomeStateLockTimeout, home_state_lock, write_resume_to

    home_dir = _make_home(tmp_path)
    with home_state_lock(home_dir):
        with home_state_lock(home_dir):  # a host holds it across "check, then call the writers"
            write_resume_to(home_dir, "personal", written_by="gateway:test")
        failures: List[BaseException] = []

        def other() -> None:
            try:
                with home_state_lock(home_dir, timeout_s=0.2):
                    pass
            except BaseException as e:  # noqa: BLE001
                failures.append(e)

        t = threading.Thread(target=other)
        t.start()
        t.join(timeout=5)
        assert len(failures) == 1 and isinstance(failures[0], HomeStateLockTimeout)


def test_a_loop_ends_when_stop_moves_the_generation_and_its_own_late_writes_are_refused(tmp_path: Path) -> None:
    from abstractruntime.identity.life import bump_lifecycle_generation

    home_dir = _make_home(tmp_path)
    llm = _HookLLM(["One - the operator stops him now.", "The look-back.", "R1.", "NEVER - a tick after Stop"],
                   hooks={0: lambda: bump_lifecycle_generation(home_dir, written_by="kill_switch")})
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=3, max_ticks=5, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lambda s: None).run()
    # The in-flight turn stops at its next effect boundary and forms nothing (Codex-2 #2): no tick is recorded.
    assert report.stopped_by == "kill_switch" and report.ticks == 0 and report.failures == 0, report
    assert "NEVER - a tick after Stop" in llm.replies


# ---------------------------------------------------------------- Codex-2 #2: a cancelled turn stops at its next effect


def _two_writes() -> str:
    return ("Two notes.\n\n```tool name=write_file path=a.md\nfirst\n```\n\n"
            "```tool name=write_file path=b.md\nsecond\n```")


def test_a_cancelled_turn_runs_no_further_tool_no_model_call_and_forms_nothing(tmp_path: Path, monkeypatch) -> None:
    """Codex-2 #2: chat.py ran the whole tool batch after cancellation, then another model call, and the turn formed.
    Now the check runs before EACH tool and before every model call: the first write ran, the second did not, no
    continuation was asked, nothing formed, and TurnCancelled carries the honest report."""
    from test_visitor_at_the_door import _LLM, _session

    import abstractruntime.identity.tools as tools_mod
    from abstractruntime.identity.chat import TurnCancelled

    home_dir = _make_home(tmp_path)
    llm = _LLM([_two_writes(), "SURPLUS - no model call after Stop"])
    s = _session(home_dir, llm)
    stop = {"now": False}
    s.cancel_check = lambda: stop["now"]
    real = tools_mod.TOOL_DESCRIPTORS["write_file"].executor

    def write_then_stop(e, ctx):  # the operator's Stop lands while the first write runs
        out = real(e, ctx)
        stop["now"] = True
        return out

    import dataclasses

    monkeypatch.setitem(tools_mod.TOOL_DESCRIPTORS, "write_file",
                        dataclasses.replace(tools_mod.TOOL_DESCRIPTORS["write_file"], executor=write_then_stop))
    before = len(s.home.ms.store.query(__import__("abstractmemory").TripleQuery(limit=0)))
    try:
        with pytest.raises(TurnCancelled) as cancelled:
            s.turn("your own time")
        after = len(s.home.ms.store.query(__import__("abstractmemory").TripleQuery(limit=0)))
    finally:
        s.home.close()
    assert (home_dir / "workspace" / "a.md").exists() and not (home_dir / "workspace" / "b.md").exists()
    assert llm.replies == ["SURPLUS - no model call after Stop"], "no continuation after Stop"
    report = cancelled.value.report
    assert report.tools == ["write_file"] and [f["path"] for f in report.files] == ["a.md"]
    assert any("1 tool call(s) not run" in n for n in report.notices)
    assert after == before, "nothing formed after Stop"


def test_the_authoring_drainer_writes_nothing_after_stop(tmp_path: Path) -> None:
    """Codex-2 #2: the drainer wrote after its model returned without rechecking Stop. Now a Stop seen after the
    capability answered writes nothing and hands the tasks back unspent."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_memory_authoring import TITLE, _Capability, _rows  # noqa: F401
    from test_title_failures import _session as _tf_session, _workspace_home

    from abstractruntime.identity.authoring import drain_authoring

    home_dir = _workspace_home(tmp_path)
    home, s = _tf_session(home_dir, ["Back.\n```interest\nwhy rivers bend\n```\n", "A quiet close."])
    s.llm_for_capability = None  # forms pending
    s.turn("hi")
    stop = {"now": False}

    class _StopsWhileAnswering(_Capability):
        def generate(self, **kw):
            stop["now"] = True  # the operator's Stop lands while the capability answers
            return super().generate(**kw)

    receipt = drain_authoring(home.ms, llm_for_capability=lambda name: _StopsWhileAnswering(), artifacts=home.artifacts,
                              home_dir=home_dir, entity=home.entity_id, holder="test", cap=3, window="turn",
                              should_stop=lambda: stop["now"])
    tasks = [t for t in home.ms.tasks.tasks(kind=("title.author",))]
    home.close()
    assert receipt["authored"] == 0 and receipt.get("cancelled", 0) >= 1, receipt
    interest = _rows(home_dir, "interest")[-1]
    assert interest.attributes.get("title") in (None, "") and isinstance(interest.attributes.get("pending"), dict)
    assert all(int(t.get("attempts") or 0) == 0 and t["status"] in ("deferred", "pending") for t in tasks), tasks
