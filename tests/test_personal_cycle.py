"""The blueprint tunables reader (the operator's dials consumed FROM the artifact), and the v22 retirement of the
v11 personal<->sleep maintenance cycle (laurent dm#104), the v13 need-check and the sleep bound: TARGET 2026-11-08
'a fixed clock never forces sleep' (contract-phases v2 §10.4-§10.6). Those dials stay readable and govern nothing."""
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


def test_no_clock_opens_a_sleep_window(tmp_path: Path, monkeypatch) -> None:
    """Re-based (v22): even with the cycle dials edited to tiny windows and the switch on, no timed maintenance sleep
    opens - personal days chain, and sleep is only his own decision (a ```rest block) or the operator's."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home

    from abstractruntime.identity.life import LifeLoop, write_cycle_clock

    home_dir = _make_home(tmp_path)
    edited = {"tunables": {"personal_cycle": {"personal_window_h": 0.000001,
                                              "sleep_window_h": 0.0001, "enabled": True}}}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(edited), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    write_cycle_clock(home_dir, 99999.0)  # a clock far past any window: still nothing opens
    slept = {"n": 0}

    def on_sleep(**_kw):
        slept["n"] += 1
        return {"formed": False, "signals": []}

    llm = _ScriptedLLM(["One.", "R1.", "Two.", "R2.", "Three.", "R3."])
    report = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=3,
        stop_file=home_dir / "STOP", state_home=home_dir,
        rest_minutes=0.001, on_sleep=on_sleep,
        sleep_fn=lambda s: None, out=lambda s: None,
    ).run()
    assert slept["n"] == 0 and report.sleeps == 0, "no clock opens a sleep window"
    assert report.days == 3, "personal days chain (personal->personal)"
    assert not any("maintenance sleep is over" in r.cue for r in report.records)


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


def test_cycle_clock_persists_across_respawns(tmp_path: Path) -> None:
    """v12 P1-4: the clock survives a respawn (read/write helpers)."""
    from abstractruntime.identity.life import read_cycle_clock, write_cycle_clock

    assert read_cycle_clock(tmp_path) == 0.0
    write_cycle_clock(tmp_path, 5400.5)
    assert read_cycle_clock(tmp_path) == 5400.5
    (tmp_path / "personal_cycle_clock.json").write_text("{corrupt", encoding="utf-8")
    assert read_cycle_clock(tmp_path) == 0.0, "corrupt reads fresh, never raises"


def test_unwired_dial_edit_warns(tmp_path: Path, monkeypatch) -> None:
    """v12 P0-4 machinery pin, re-based for v22: the timed dials are RETIRED (wired:false) - an edit is read, and the
    warning says it governs nothing; a dial born unwired still says NOT YET WIRED. Missing known keys stay noted."""
    from abstractruntime.identity import phase_spec as ps

    assert ps.TUNABLE_WIRED["sleep_bound_h"] is False and "sleep_bound_h" in ps.TUNABLE_RETIRED
    assert ps.TUNABLE_WIRED["window_limit"] is True
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"tunables": {
        "personal_cycle": {"personal_window_h": 2.0, "sleep_window_h": 1.0},
        "sleep_bound_h": 2.5, "window_limit": 9000,
    }}), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    monkeypatch.setitem(ps.TUNABLE_WIRED, "window_limit", False)
    t, warns = load_phase_tunables()
    assert t["sleep_bound_h"] == 2.5, "the edit is READ (recorded)"
    assert any("sleep_bound_h=2.5 is edited but RETIRED" in w for w in warns), warns
    assert any("window_limit=9000.0 is edited but NOT YET WIRED" in w for w in warns), warns
    assert any("unattended_wake_cadence_h absent" in w for w in warns), "missing known key noted"


def test_retired_gate_dials_govern_nothing(tmp_path, monkeypatch) -> None:
    """Re-based (v22): the 6h cadence and the 2h use floor drove the gate's sleep leg, which is gone. Edited, they
    change nothing: no grant -> own time ends (idle, no need-check deadline); armed + zero drives -> he rests."""
    from abstractruntime.identity.life import read_day_gate, write_personal_grant

    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"tunables": {
        "personal_cycle": {"personal_window_h": 2.0, "sleep_window_h": 1.0, "enabled": False},
        "sleep_bound_h": 1.0,
        "unattended_wake_cadence_h": 3.0,
        "grant_unused_floor_h": 0.5,
    }}), encoding="utf-8")
    monkeypatch.setenv(PHASE_SPEC_ENV, str(p))
    home = tmp_path / "home"
    home.mkdir()
    d = read_day_gate(home)
    assert d["phase"] == "idle" and d["cause"] == "no_grant" and "need_check_s" not in d
    write_personal_grant(home, mode="until_revoked", granted_by="operator:test")
    d = read_day_gate(home)
    assert d["phase"] == "personal" and d["cause"] in ("resting", "grant_degraded"), d


def test_no_need_check_and_no_bound_wake_an_operator_sleep(tmp_path, monkeypatch) -> None:
    """Re-based (v22 wake_conditions: no clock wakes him; the 6h need-check and the 1h sleep bound are retired): an
    operator sleep past its stamped wake_at, over a standing work order, keeps idling - no need-check wake, no
    sleep-bound wake - until the operator's wake lands; then the work day opens."""
    import sys
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home

    from abstractruntime.identity.life import LifeLoop, read_entity_state, write_entity_state

    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("desk", encoding="utf-8")
    write_entity_state(
        home_dir, "asleep", reason="sleep now", written_by="operator",
        wake_at=(datetime.now(timezone.utc) - timedelta(hours=7)).isoformat(),
    )
    polls = {"n": 0}
    seen = []

    def sleeper(_s: float) -> None:
        polls["n"] += 1
        st = read_entity_state(home_dir)
        seen.append((st["state"], st.get("written_by")))
        if polls["n"] == 20:
            write_entity_state(home_dir, "awake", reason="operator wake")

    llm = _ScriptedLLM(["Working.\n```work\ndone: sorted\n```", "R."])
    report = LifeLoop(
        _factory_for(home_dir, llm), tick_seconds=1, ticks_per_day=1, max_ticks=1,
        stop_file=home_dir / "STOP", state_home=home_dir, rest_minutes=0.001,
        sleep_fn=sleeper, out=lambda s: None,
    ).run()
    assert seen[:19] == [("asleep", "operator")] * 19, "no clock woke him"
    assert report.ticks == 1, "the operator's wake opened the work day"
    writers = [json.loads(ln).get("written_by") for ln in
               (home_dir / "state_history.jsonl").read_text().splitlines() if ln.strip()]
    assert "need-check" not in writers and "sleep-bound" not in writers


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


def test_night_predicate_has_no_clock(tmp_path, monkeypatch) -> None:
    """Re-based (v22; memory c370's stamped-window deadline retired with the cycle): the night's should_continue
    ignores any wake_at - a pass runs to its end unless a visitor, a wake, a pause or the STOP brake ends it."""
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
    write_entity_state(
        home_dir, "asleep", reason="self-elected sleep: full", written_by="self",
        wake_at=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
    )
    consolidate()
    assert captured["pred"] is not None, "the engine receives the predicate"
    assert captured["pred"]() is True, "a stamp inside the old 60 s grace no longer sheds the night"
    write_entity_state(home_dir, "asleep", mode="visiting", written_by="visit-door", reason="auto-yield")
    assert captured["pred"]() is False, "a visitor ends it at the next stage"
    write_entity_state(home_dir, "awake", reason="operator wake")
    assert captured["pred"]() is False, "a wake ends it"


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


def test_consolidator_flags_reach_the_engine_and_his_sleep_runs_the_whole_night(tmp_path, monkeypatch) -> None:
    """memory c379 wiring: build_consolidator passes include_dream to the engine. Re-based (v22): the maintenance-
    only cycle window is gone - his own sleep (a ```rest block) runs the nightly composition (dream on); a hook
    without the kwarg still works."""
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

    seen = {"flags": []}

    def _hook(include_dream=True):
        seen["flags"].append(include_dream)
        return {"formed": False}

    llm = _ScriptedLLM(["Enough.\n```rest\nthe day is full\n```", "R1.", "Two.", "R2."])
    LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=2,
        stop_file=home_dir / "STOP", state_home=home_dir,
        rest_minutes=0.001, on_sleep=_hook,
        sleep_fn=lambda s: None, out=lambda s: None,
    ).run()
    assert seen["flags"] == [True], "his own sleep ran the nightly composition"


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


def test_explored_ratio_derives_from_the_shared_fold(tmp_path) -> None:
    """skill c413 P0: the day-open ratio's explored count derives from
    unexplored_interests (the explores= stamp lane), never from phantom
    attributes keys nothing writes. An interest DISCHARGED by a believed
    explores= stamp counts as explored."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _make_home

    from abstractmemory import (
        MemoryRecordInput,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
    )

    from abstractruntime.identity.life import _standing_interest_note

    home_dir = _make_home(tmp_path)
    entity_id = "entity:liveling@home-test"
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)

    def _interest(title: str, digest: str) -> MemoryRecordInput:
        return MemoryRecordInput(
            kind="interest", title=title, digest=digest, keywords=(),
            payload_ref=None, attributes={}, provenance={"source": "entity-reflection-v1"},
            edges=(),
        )

    out1 = ms.remember_many(
        [_interest("interest: tidal pools", "the taste of tidal pools")],
        scope="self", owner_id=entity_id, turn_id="t-i1", idempotency_key="t-i1-k",
    )
    ms.remember_many(
        [_interest("interest: bridges", "how bridges remember load")],
        scope="self", owner_id=entity_id, turn_id="t-i2", idempotency_key="t-i2-k",
    )
    explored_gid = str((out1.record_ids if hasattr(out1, "record_ids") else out1)[0])
    # A believed diary-plane record stamping explores= on interest 1 (his act).
    ms.remember_many(
        [MemoryRecordInput(
            kind="diary", title="walked the pools",
            digest="today I walked the pools and wrote what held",
            keywords=(), payload_ref=None,
            attributes={"explores": explored_gid, "entry_id": "diary_pools01"},
            provenance={"source": "diary-projection"}, edges=(),
        )],
        scope="diary", owner_id=entity_id, turn_id="t-exp", idempotency_key="t-exp-k",
    )
    store.close()
    journal.close()
    note, _ids = _standing_interest_note(home_dir, entity_id, rotation_key=0)
    assert "1 ever explored" in note, f"the stamp discharges into the ratio: {note!r}"


def test_an_operator_sleep_on_a_bare_desk_never_wakes_into_a_day(tmp_path) -> None:
    """Re-based (v22): an operator sleep on a bare desk (no grant, no order) past its stamped wake_at stays a sleep -
    no need-check, no bound, no marker churn - until a STOP ends the loop; the sleep is still the operator's."""
    import sys
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _ScriptedLLM, _factory_for, _make_home

    from abstractruntime.identity.life import (
        PHASES_FILENAME, LifeLoop, read_entity_state, write_entity_state,
    )

    home_dir = _make_home(tmp_path)
    (home_dir / PHASES_FILENAME).unlink()
    write_entity_state(
        home_dir, "asleep", reason="sleep now", written_by="operator",
        wake_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
    )
    before = (home_dir / "state_history.jsonl").read_text().strip().splitlines()
    polls = {"n": 0}

    def sleeper(_s: float) -> None:
        polls["n"] += 1
        if polls["n"] == 10:
            (home_dir / "STOP").write_text("halt", encoding="utf-8")

    LifeLoop(
        _factory_for(home_dir, _ScriptedLLM([])), tick_seconds=1, ticks_per_day=1, max_ticks=1,
        stop_file=home_dir / "STOP", state_home=home_dir, rest_minutes=0.001,
        sleep_fn=sleeper, out=lambda s: None,
    ).run()
    after = (home_dir / "state_history.jsonl").read_text().strip().splitlines()
    assert after == before, "no marker churn - his one act stays one moment"
    st = read_entity_state(home_dir)
    assert st["state"] == "asleep" and st["written_by"] == "operator"
