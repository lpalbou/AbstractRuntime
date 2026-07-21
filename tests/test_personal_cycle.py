"""v11 personal<->sleep maintenance cycle (laurent dm#104) + the blueprint
tunables reader (the operator's dials consumed FROM the artifact)."""
from __future__ import annotations

import json
from pathlib import Path

from abstractruntime.identity.phase_spec import (
    PHASE_SPEC_ENV,
    load_phase_tunables,
    vendored_spec_path,
)


def test_tunables_read_the_vendored_v11_seeds() -> None:
    t, warns = load_phase_tunables()
    assert t["personal_cycle"]["personal_window_h"] == 2.0
    assert t["personal_cycle"]["sleep_window_h"] == 1.0
    assert t["sleep_bound_h"] == 1.0
    assert t["unattended_wake_cadence_h"] == 6.0
    assert t["grant_unused_floor_h"] == 2.0
    assert warns == [], warns


def test_operator_override_wins_and_partial_edits_fill(tmp_path: Path, monkeypatch) -> None:
    """The modulation path: an edited blueprint changes the numbers with
    zero code change; a PARTIAL edit fills missing keys from the ruled
    defaults (never KeyErrors a consumer)."""
    edited = {"tunables": {"personal_cycle": {"personal_window_h": 0.5}}}
    p = tmp_path / "edited_spec.json"
    p.write_text(json.dumps(edited), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    t, _w = load_phase_tunables()
    assert t["personal_cycle"]["personal_window_h"] == 0.5, "the operator's dial"
    assert t["personal_cycle"]["sleep_window_h"] == 1.0, "partial edit fills"
    assert t["sleep_bound_h"] == 1.0


def test_unreadable_override_falls_to_vendored_loudly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(PHASE_SPEC_ENV, str(tmp_path / "missing.json"))
    t, warns = load_phase_tunables()
    assert t["personal_cycle"]["personal_window_h"] == 2.0, "vendored stands"
    assert any("#FALLBACK" in w for w in warns), "degradation is loud"


def test_bad_values_keep_ruled_defaults_loudly(tmp_path: Path, monkeypatch) -> None:
    edited = {"tunables": {"personal_cycle": {"personal_window_h": -3},
                           "sleep_bound_h": "soon"}}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(edited), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    t, warns = load_phase_tunables()
    assert t["personal_cycle"]["personal_window_h"] == 2.0
    assert t["sleep_bound_h"] == 1.0
    assert sum("#FALLBACK" in w for w in warns) >= 2


def test_vendored_artifact_carries_the_cycle_tunables() -> None:
    data = json.loads(vendored_spec_path().read_text(encoding="utf-8"))
    assert data["version"] >= 12
    assert "personal_cycle" in (data.get("tunables") or {})
    assert "sleep_bound_h" in (data.get("tunables") or {}), "the v12 rename"


def test_cycle_sleep_fires_after_the_window_and_wakes_into_personal(tmp_path: Path, monkeypatch) -> None:
    """The cycle end-to-end in a 24/7 loop: after >= personal_window_h of
    lived personal time, a bounded maintenance sleep runs (consolidator
    invoked, state honest) and the loop wakes BACK INTO PERSONAL with the
    grant standing."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home

    from abstractruntime.identity.life import LifeLoop, read_entity_state

    home_dir = _make_home(tmp_path)
    # Tiny windows via the operator-override lane (the modulation path IS
    # the test seam - no monkeypatching internals).
    edited = {"tunables": {"personal_cycle": {"personal_window_h": 0.000001,
                                              "sleep_window_h": 0.0001}}}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(edited), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))

    slept = {"n": 0}

    def on_sleep():
        slept["n"] += 1
        return {"formed": False, "signals": []}

    llm = _ScriptedLLM(["One.", "R1.", "Two.", "R2.", "Three.", "R3."])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=3,
        stop_file=home_dir / "STOP", state_home=home_dir,
        rest_minutes=0.001,  # 24/7 mode: the cycle lane is armed
        on_sleep=on_sleep,
        sleep_fn=lambda s: None, out=lambda s: None,
    )
    report = loop.run()
    assert slept["n"] >= 1, "the maintenance window ran the consolidator"
    assert report.sleeps >= 1
    assert report.days >= 2, "personal time RESUMED after the cycle sleep"
    # The wake left an honest awake state (not stranded asleep).
    state = read_entity_state(home_dir)
    assert state.get("state") in ("awake", "asleep")  # asleep only via exit landing


def test_gateway_operator_copy_wins_over_vendored(tmp_path: Path) -> None:
    """The PUT lane's persistence (gateway c-t-i 350): laurent's edited
    blueprint at <data_dir>/config/entity_phases.json reaches a detached
    loop as a plain file read - located from the home dir, no env, no
    HTTP. Absent = vendored stands silently (the normal un-edited state)."""
    data_dir = tmp_path / "runtime"
    home = data_dir / "entities" / "tester"
    home.mkdir(parents=True)
    t, _ = load_phase_tunables(home_dir=home)
    assert t["personal_cycle"]["personal_window_h"] == 2.0, "vendored when absent"
    cfg = data_dir / "config"
    cfg.mkdir()
    (cfg / "entity_phases.json").write_text(json.dumps(
        {"tunables": {"personal_cycle": {"personal_window_h": 3.5}}}), encoding="utf-8")
    t2, w2 = load_phase_tunables(home_dir=home)
    assert t2["personal_cycle"]["personal_window_h"] == 3.5, "laurent's dial reached the loop"
    assert t2["personal_cycle"]["sleep_window_h"] == 1.0, "partial edit fills"


def test_cycle_preconditions_hold_the_clock(tmp_path: Path, monkeypatch) -> None:
    """v12 P0-1: a standing work order or a non-plain-awake state HOLDS the
    cycle (the clock keeps, the sleep never opens over the desk or the
    door's posture)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home

    from abstractruntime.identity.life import LifeLoop, write_cycle_clock

    home_dir = _make_home(tmp_path)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"tunables": {"personal_cycle": {
        "personal_window_h": 0.000001, "sleep_window_h": 0.0001}}}), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    # The clock is ALREADY over the window (persisted from prior personal
    # days) and a work order stands: the cycle must hold, never sleep over
    # the desk. (The order routes the day to work, so the clock cannot
    # accumulate in-test - the pre-seeded file is the honest fixture.)
    write_cycle_clock(home_dir, 99999.0)
    (home_dir / "work_order.md").write_text("stand by", encoding="utf-8")
    outs: list = []
    slept = {"n": 0}
    llm = _ScriptedLLM(["One.", "R1.", "Two.", "R2."])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=2,
        stop_file=home_dir / "STOP", state_home=home_dir,
        rest_minutes=0.001,
        on_sleep=lambda: slept.__setitem__("n", slept["n"] + 1) or {"formed": False},
        sleep_fn=lambda s: None, out=outs.append,
    )
    loop.run()
    assert slept["n"] == 0, "the cycle never fired over a standing order"
    assert any("cycle held" in o for o in outs), "the hold is loud"


def test_cycle_clock_persists_across_respawns(tmp_path: Path) -> None:
    """v12 P1-4: the clock survives a respawn (read/write helpers)."""
    from abstractruntime.identity.life import read_cycle_clock, write_cycle_clock

    assert read_cycle_clock(tmp_path) == 0.0
    write_cycle_clock(tmp_path, 5400.5)
    assert read_cycle_clock(tmp_path) == 5400.5
    (tmp_path / "personal_cycle_clock.json").write_text("{corrupt", encoding="utf-8")
    assert read_cycle_clock(tmp_path) == 0.0, "corrupt reads fresh, never raises"


def test_unwired_dial_edit_warns(tmp_path: Path, monkeypatch) -> None:
    """v12 P0-4 machinery pin: ALL dials are wired as of v14 (the ledger is
    all-True - zero dead dials), so the unwired-edit warning is exercised
    through a patched ledger; the machinery must hold for any FUTURE dial
    born unwired. Missing known keys stay noted."""
    from abstractruntime.identity import phase_spec as ps

    assert all(ps.TUNABLE_WIRED.values()), "v14: zero dead dials - all wired"
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"tunables": {
        "personal_cycle": {"personal_window_h": 2.0, "sleep_window_h": 1.0},
        "sleep_bound_h": 2.5,
    }}), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    monkeypatch.setitem(ps.TUNABLE_WIRED, "sleep_bound_h", False)
    t, warns = load_phase_tunables()
    assert t["sleep_bound_h"] == 2.5, "the edit is READ (recorded)"
    assert any("NOT YET WIRED" in w for w in warns), "and loudly not applied"
    assert any("unattended_wake_cadence_h absent" in w for w in warns), "missing known key noted"


def test_gate_dials_are_wired(tmp_path, monkeypatch) -> None:
    """v13: unattended_wake_cadence_h + grant_unused_floor_h edits GOVERN
    the day gate (no longer dead dials); personal_cycle.enabled=false
    disables the cycle."""
    from abstractruntime.identity.life import read_day_gate, write_personal_grant

    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"tunables": {
        "personal_cycle": {"personal_window_h": 2.0, "sleep_window_h": 1.0, "enabled": False},
        "sleep_bound_h": 1.0,
        "unattended_wake_cadence_h": 3.0,
        "grant_unused_floor_h": 0.5,
    }}), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    t, _ = load_phase_tunables()
    assert t["personal_cycle"]["enabled"] is False

    home = tmp_path / "home"
    home.mkdir()
    # No grant -> sleep leg carries the EDITED cadence (3h, not the 6h seed).
    d = read_day_gate(home)
    assert d["phase"] == "sleep" and d["need_check_s"] == int(3.0 * 3600)
    # Grant armed, zero drives, zero use -> the EDITED 0.5h floor governs.
    write_personal_grant(home, mode="until_revoked", granted_by="operator:test")
    d = read_day_gate(home)
    if d["cause"] == "granted_unused":
        assert "0.5h floor" in d["detail"]


def test_need_check_is_no_churn(tmp_path, monkeypatch) -> None:
    """v13 cadence_need_check: a quiet check re-sleeps WITHOUT a marker
    pair (no awake/asleep churn - the same sleep continues); a sanctioned
    day lands ONE wake with written_by=need-check."""
    import sys
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home

    from abstractruntime.identity.life import (
        LifeLoop, read_entity_state, write_entity_state,
    )

    home_dir = _make_home(tmp_path)
    # The quiet premise: NO grant (the fixture arms one - drop it), no
    # order; the seeded drive alone cannot sanction a day without a grant.
    from abstractruntime.identity.life import PHASES_FILENAME

    (home_dir / PHASES_FILENAME).unlink()
    llm = _ScriptedLLM(["One."] * 4)
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=1,
        stop_file=home_dir / "STOP", state_home=home_dir,
        rest_minutes=0.001, sleep_fn=lambda s: None, out=lambda s: None,
    )
    # A gate-written unarmed sleep whose deadline is already past.
    write_entity_state(
        home_dir, "asleep", reason="day gate: no_grant (x) - need-check in 6h",
        written_by="grant-gate",
        wake_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
    )
    before = json.loads((home_dir / "state_history.jsonl").read_text().strip().splitlines()[-1])
    # Quiet check: no grant, no order, no drives -> the same sleep continues.
    still = loop._bounded_asleep_or_paused(read_entity_state(home_dir))
    assert still is True, "nothing sanctioned - keeps idling"
    after_lines = (home_dir / "state_history.jsonl").read_text().strip().splitlines()
    assert json.loads(after_lines[-1]) == before, "NO marker churn on a quiet check"
    assert loop._need_check_at is not None, "next deadline held in memory"
    # Sanction a day (work order) and force the in-memory deadline due.
    (home_dir / "work_order.md").write_text("desk", encoding="utf-8")
    loop._need_check_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    still = loop._bounded_asleep_or_paused(read_entity_state(home_dir))
    assert still is False, "a sanctioned day wakes"
    state = read_entity_state(home_dir)
    assert state["state"] == "awake" and state["written_by"] == "need-check"


def test_sleep_bound_reads_the_blueprint_dial(tmp_path, monkeypatch) -> None:
    """v14: sleep_bound_h threads through sleep_bound_deadline(home_dir=);
    no home_dir = the ruled seed (sweeper-compatible until it adopts)."""
    from datetime import datetime, timedelta, timezone

    from abstractruntime.identity.life import SLEEP_BOUND_SECONDS, sleep_bound_deadline

    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"tunables": {
        "personal_cycle": {"personal_window_h": 2.0, "sleep_window_h": 1.0},
        "sleep_bound_h": 2.5,
    }}), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    changed = datetime(2026, 7, 21, 10, 0, 0, tzinfo=timezone.utc)
    state = {"state": "asleep", "changed_at": changed.isoformat(), "reason": "nap"}
    # Dial governs with a home; the ruled seed governs without one.
    assert sleep_bound_deadline(state, home_dir=tmp_path) == changed + timedelta(hours=2.5)
    assert sleep_bound_deadline(state) == changed + timedelta(seconds=SLEEP_BOUND_SECONDS)
    # An explicit wake_at stamp always wins over any bound.
    stamped = dict(state, wake_at=(changed + timedelta(minutes=5)).isoformat())
    assert sleep_bound_deadline(stamped, home_dir=tmp_path) == changed + timedelta(minutes=5)


def test_night_predicate_honors_stamped_window_deadline(tmp_path, monkeypatch) -> None:
    """memory c370 wiring: the night's should_continue reads the stamped
    wake_at - past (stamp - grace) it says stop (engine sheds at the next
    phase boundary); a future stamp keeps the night."""
    import sys
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _make_home

    import abstractmemory
    from abstractruntime.identity.life import build_consolidator, write_entity_state

    home_dir = _make_home(tmp_path)
    captured = {}

    def _fake_sleep_pass(ms, *, scopes, owner_id, should_continue=None, **kw):
        captured["pred"] = should_continue
        return {"maintenance": {}, "dream": {}}

    monkeypatch.setattr(abstractmemory, "sleep_pass", _fake_sleep_pass)
    consolidate = build_consolidator(home_dir, out=lambda s: None)

    # Future stamp: the night keeps its window.
    write_entity_state(
        home_dir, "asleep", reason="personal_cycle maintenance (1h)",
        written_by="personal-cycle",
        wake_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    )
    consolidate()
    assert captured["pred"] is not None, "the engine receives the predicate"
    assert captured["pred"]() is True, "future stamp - night continues"

    # Past-the-grace stamp: the night sheds.
    write_entity_state(
        home_dir, "asleep", reason="personal_cycle maintenance (1h)",
        written_by="personal-cycle",
        wake_at=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
    )
    assert captured["pred"]() is False, "inside the 60s grace - night sheds"


def test_loop_status_carries_the_tunables_receipt(tmp_path) -> None:
    """dm#112 R3: the loop stamps WHAT dials it read (+when) in loop_status;
    the receipt survives later phase heartbeats (substrate-preservation
    pattern)."""
    from abstractruntime.identity.life import read_loop_status, write_loop_status

    write_loop_status(tmp_path, "between", tunables={
        "personal_cycle": {"personal_window_h": 2.0, "sleep_window_h": 1.0, "enabled": True},
        "sleep_bound_h": 1.0, "$comment": "stripped",
    })
    st = read_loop_status(tmp_path)
    assert st["tunables"]["sleep_bound_h"] == 1.0
    assert "$comment" not in st["tunables"], "doc keys never ride the receipt"
    assert st.get("tunables_at"), "the read moment is stamped"
    first_at = st["tunables_at"]
    # A later heartbeat WITHOUT tunables preserves the receipt.
    write_loop_status(tmp_path, "day")
    st2 = read_loop_status(tmp_path)
    assert st2["tunables"] == st["tunables"] and st2["tunables_at"] == first_at


def test_cue_attention_kwargs_exist_on_the_engine(tmp_path) -> None:
    """dm#112 R1: the resident-window fix must construct on the REAL engine
    signature (attention_config=/window_limit=) - the wrong-kwarg version
    silently fell to window 512 via a swallowed TypeError."""
    from abstractmemory import AttentionConfig, MemorySystem, SQLiteJournal, SQLiteTripleStore

    db = tmp_path / "m.sqlite3"
    system = MemorySystem(
        store=SQLiteTripleStore(db), journal=SQLiteJournal(db),
        attention_config=AttentionConfig(window_limit=8192),
    )
    assert system is not None
    # And the life.py site uses exactly these names (source pin).
    import inspect

    from abstractruntime.identity import life

    src = inspect.getsource(life)
    assert "attention=AttentionConfig(window=" not in src, "the dead-code shape is gone"
    # v16: no hardcoded horizon anywhere - both sites read the dial.
    assert "window_limit=8192" not in src, "the constant died with the v16 rows"
    assert 'attention_window=int(_tun["window_limit"])' in src
    assert 'drive_window_limit=int(_tun["drive_window_limit"])' in src


def test_cycle_window_is_maintenance_only(tmp_path, monkeypatch) -> None:
    """memory c379 wiring: the cycle window calls the consolidator with
    include_dream=False (quality passes only); the engine receives the flag
    through build_consolidator; hooks without the kwarg still work."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home

    import abstractmemory
    from abstractruntime.identity.life import LifeLoop, build_consolidator

    home_dir = _make_home(tmp_path)
    captured = {}

    def _fake_sleep_pass(ms, *, scopes, owner_id, should_continue=None, include_dream=True, **kw):
        captured["include_dream"] = include_dream
        return {"maintenance": {}, "dream": {}}

    monkeypatch.setattr(abstractmemory, "sleep_pass", _fake_sleep_pass)
    consolidate = build_consolidator(home_dir, out=lambda s: None)
    consolidate(include_dream=False)
    assert captured["include_dream"] is False, "the flag reaches the engine"
    consolidate()
    assert captured["include_dream"] is True, "nightly default unchanged"

    # The cycle branch passes include_dream=False through _sleep_window;
    # a zero-arg on_sleep hook (older consolidators, test doubles) still
    # works - the flag is composition, never a requirement.
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"tunables": {"personal_cycle": {
        "personal_window_h": 0.000001, "sleep_window_h": 0.0001}}}), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    seen = {"flags": []}

    def _hook(include_dream=True):
        seen["flags"].append(include_dream)
        return {"formed": False}

    llm = _ScriptedLLM(["One.", "R1.", "Two.", "R2.", "Three.", "R3."])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=2,
        stop_file=home_dir / "STOP", state_home=home_dir,
        rest_minutes=0.001, on_sleep=_hook,
        sleep_fn=lambda s: None, out=lambda s: None,
    )
    loop.run()
    assert False in seen["flags"], "the cycle window ran maintenance-only"


def test_window_dials_resolve_from_the_blueprint(tmp_path, monkeypatch) -> None:
    """v16 M2: window_limit/drive_window_limit are dials - an operator edit
    governs both home-open sites; seeds 8192/256 stand un-edited."""
    t, _ = load_phase_tunables()
    assert t["window_limit"] == 8192 and t["drive_window_limit"] == 256
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"tunables": {
        "personal_cycle": {"personal_window_h": 2.0, "sleep_window_h": 1.0},
        "sleep_bound_h": 1.0, "unattended_wake_cadence_h": 6.0,
        "grant_unused_floor_h": 2.0,
        "window_limit": 16384, "drive_window_limit": 512,
    }}), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    t, warns = load_phase_tunables()
    assert t["window_limit"] == 16384 and t["drive_window_limit"] == 512
    assert not any("NOT YET WIRED" in w for w in warns), "runtime half is wired"
