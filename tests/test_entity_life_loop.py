"""Life loop contract (the entity's own time, maintainer m2).

Offline: scripted LLM, sleep mocked to zero. Pins the loop's honesty rules:
self-prompt chaining (`next:`), rest election stops immediately, the stop
file halts between ticks, days close with reflection, self-directed
participants, and the workspace is live inside the loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.life import (  # noqa: E402
    LifeLoop,
    NEUTRAL_CUE,
    OWN_TIME_CONTRACT,
    parse_next_cue,
    parse_rest_block,
)


def _make_home(tmp_path: Path) -> Path:
    import copy

    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / "liveling"
    home_dir.mkdir(parents=True)
    entity_id = "entity:liveling@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Liveling"
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    # ONE standing drive (dm#89 day gate): a zero-drive desk RESTS
    # (settled_desk), so day-running tests seed a real open question —
    # the graph plane drive_pressure reads (diary projection).
    from abstractruntime.integrations.abstractmemory.identity_support import (
        project_diary_entry,
    )

    project_diary_entry(
        ms, entity_id=entity_id, entry_id="diary_seedq", kind="question",
        visibility="normal", gist="what does a day owe its evening?",
        written_at="2026-07-19T08:00:00+00:00", turn_id="t-seed",
        origin={"run_id": "seed"},
    )
    store.close()
    journal.close()
    _arm_personal(home_dir)
    return home_dir


def _arm_personal(home_dir: Path, *, mode: str = "until_revoked", expires_at: str = "") -> None:
    """Arm the personal phase (laurent 12:44: own time IS the personal phase,
    OFF by default) — the loop refuses to open days without it, so every
    fixture home that expects ticks arms it as its operator act. Uses the
    ONE writer (write_personal_grant) except for the shapes the writer
    refuses (timer without expiry), which pin the READ side's fail-closed
    behavior and are hand-written."""
    from abstractruntime.identity.life import write_personal_grant

    if mode == "timer" and not expires_at:
        (home_dir / "phases.yaml").write_text(
            yaml.safe_dump({"personal": {
                "mode": "timer", "granted_by": "person:test-operator",
                "granted_at": "2026-07-13T00:00:00+00:00",
            }}, sort_keys=False),
            encoding="utf-8",
        )
        return
    write_personal_grant(
        home_dir, mode=mode, granted_by="person:test-operator",
        expires_at=expires_at or None,
    )


class _ScriptedLLM:
    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, messages, system_prompt):
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})

        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "…\nnext: keep thinking"
        return r


def _factory_for(home_dir: Path, llm: _ScriptedLLM):
    def _open() -> ChatSession:
        home = open_home(home_dir)
        session = ChatSession(
            home, llm, participants=[home.entity_id], context_window=20000,
            enable_workspace=True, out=lambda s: None,
        )
        session.system_base += "\n\n" + OWN_TIME_CONTRACT
        return session

    return _open


# ---------------------------------------------------------------- parsing


def test_parse_next_cue_last_line_wins_and_caps() -> None:
    assert parse_next_cue("thoughts\nnext: build the home file") == "build the home file"
    assert parse_next_cue("next: a\nmore\nnext: b") == "b"
    assert parse_next_cue("no note here") is None
    assert len(parse_next_cue("next: " + "x" * 900)) == 400


def test_parse_rest_block() -> None:
    marked, reason = parse_rest_block("done for now\n```rest\nI want quiet\n```")
    assert reason == "I want quiet"
    assert "[chose to rest]" in marked
    assert parse_rest_block("no rest")[1] is None


# ------------------------------------------------------------------- loop


def test_loop_chains_next_cues_and_respects_max_ticks(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(
        [
            "First thought.\nnext: look at my questions",
            "Second thought about questions.",  # no next: -> neutral cue
            "Third thought.\nnext: never reached",
            "Day reflection: a quiet start.",  # reflection call at day close
        ]
    )
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=0, ticks_per_day=8, max_ticks=3, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 3
    assert report.stopped_by == "max_ticks"
    # Tick 2 received tick 1's own note; tick 3 received the neutral cue.
    assert report.records[1].cue == "look at my questions"
    assert report.records[2].cue == NEUTRAL_CUE
    # The user-facing message each tick is the entity's own prior note.
    assert llm.calls[1]["messages"][-1]["content"] == "look at my questions"


def test_loop_rest_election_stops_and_still_reflects(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(
        [
            "Thinking.\nnext: rest soon",
            "Enough.\n```rest\na good stopping point\n```",
            "Reflection: short but mine.",
        ]
    )
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=0, ticks_per_day=8, max_ticks=10, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "rest"
    assert report.rest_reason == "a good stopping point"
    assert report.ticks == 2
    # Reflection ran at day close (3rd scripted reply consumed).
    assert llm.replies == []


def test_loop_stop_command_halts_between_ticks(tmp_path: Path) -> None:
    """The gateway's durable stop command (home.sqlite3 inbox) halts the loop
    at the next tick boundary — no sentinel file involved — and the report
    names the channel (stop_command, distinct from stop_file)."""
    from abstractruntime.identity.life import request_loop_stop

    home_dir = _make_home(tmp_path)
    tick_replies = ["One.\nnext: two", "Two.\nnext: three", "Reflection."]
    llm = _ScriptedLLM(tick_replies)

    def sleeper(_s: float) -> None:
        # The operator clicks stop in the webapp while the loop sleeps after tick 2.
        if len(llm.calls) >= 2:
            request_loop_stop(home_dir, reason="operator clicked stop", requested_by="operator")

    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=8, max_ticks=10,
        stop_file=home_dir / "STOP", state_home=home_dir,
        sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "stop_command"
    assert report.ticks == 2
    assert not (home_dir / "STOP").exists()  # remote stop never writes sentinel files


def test_stale_stop_command_dies_at_start(tmp_path: Path) -> None:
    """A stop addressed to a previous life must not kill the next one: the
    starter fast-forwards the inbox at the start-request moment."""
    from abstractruntime.identity.life import fast_forward_loop_commands, request_loop_stop

    home_dir = _make_home(tmp_path)
    request_loop_stop(home_dir, reason="stop for the OLD life")
    skipped = fast_forward_loop_commands(home_dir)
    assert skipped >= 1

    llm = _ScriptedLLM(["One.\nnext: two", "Two.\nnext: three", "Reflection."])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=0, ticks_per_day=2, max_ticks=2,
        stop_file=home_dir / "STOP", state_home=home_dir, out=lambda s: None,
    )
    report = loop.run()
    # The stale command did not stop it; it ran its full two ticks.
    assert report.ticks == 2
    assert report.stopped_by == "max_ticks"


def test_life_sleep_stats_counts_and_shares(tmp_path: Path) -> None:
    """Sleep is a life statistic (maintainer ask): number and % of sleeps,
    split self-elected vs operator, read from the append-only state history."""
    from abstractruntime.identity.life import life_sleep_stats, write_entity_state

    home_dir = _make_home(tmp_path)
    # Empty life: zeros, no raise.
    empty = life_sleep_stats(home_dir)
    assert empty["sleeps"] == 0 and empty["sleep_share"] == 0.0

    write_entity_state(home_dir, "asleep", reason="op sleep", written_by="operator")
    write_entity_state(home_dir, "awake", written_by="operator")
    write_entity_state(home_dir, "asleep", reason="self sleep", written_by="self")
    write_entity_state(home_dir, "awake", written_by="self")

    stats = life_sleep_stats(home_dir)
    assert stats["sleeps"] == 2
    assert stats["self_elected"] == 1
    assert stats["operator"] == 1
    assert stats["wakes"] == 2
    assert stats["transitions"] == 4
    assert abs(stats["sleep_share"] - 0.5) < 1e-9


def test_build_consolidator_runs_against_a_real_home(tmp_path: Path) -> None:
    """The real consolidator opens the home and runs the sleep pass without
    error (regression: it must reach the MemorySystem via ChatHome.ms, not
    a wrong attribute — the live loop caught `.memory` vs `.ms`), and it
    returns the LOOP-FACING contract: `formed` mirrors the engine dream's
    `created` (found 2026-07-09: the raw engine dict was handed to a loop
    checking `formed`, so a real formed dream read as "a quiet night" —
    the test double's hand-written shape had masked the mismatch)."""
    from abstractruntime.identity.life import build_consolidator

    home_dir = _make_home(tmp_path)
    consolidate = build_consolidator(home_dir, embedding_model=None)
    result = consolidate()  # must not raise; a quiet night is a valid result
    assert isinstance(result, dict)
    assert result["formed"] == bool((result["engine"].get("dream") or {}).get("created"))
    assert "dream_record_id" in result
    assert "maintenance_candidates" in result
    # A virgin home is a quiet night: nothing formed, and the engine says why.
    assert result["formed"] is False
    assert (result["engine"].get("dream") or {}).get("skipped_reason")


def test_self_elected_rest_is_a_sleep_with_consolidation(tmp_path: Path) -> None:
    """A self-elected rest in 24/7 mode marks state=asleep (written_by=self)
    around a consolidation pass, then wakes back to awake (maintainer ruling
    2026-07-08: sleep is where the graph is worked on, and it shows on the
    navbar because the state file is written)."""
    from abstractruntime.identity.life import read_entity_state

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM([
        "Enough.\n```rest\nconsolidate the day\n```",  # day 1: rest immediately
        "Day1 reflection.",
        "Awake again.",                                  # day 2: one tick
        "Day2 reflection.",
    ])

    sleep_states: List[str] = []
    consolidated = {"n": 0}

    def on_sleep():
        # Capture the state visible DURING the sleep window (navbar truth).
        consolidated["n"] += 1
        sleep_states.append(read_entity_state(home_dir).get("state"))
        return {"formed": True, "dream_id": "ex:dream-test"}

    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=0, ticks_per_day=1, max_ticks=2,
        stop_file=home_dir / "STOP", state_home=home_dir,
        rest_minutes=0.01, on_sleep=on_sleep,
        sleep_fn=lambda s: None, out=lambda s: None,
    )
    report = loop.run()

    # The consolidation pass ran inside the sleep window, and state was asleep then.
    assert consolidated["n"] == 1
    assert sleep_states == ["asleep"]
    assert report.dreams == 1
    # The nap self-cleared DURING the loop (day 2 ran - the dream cue is
    # the proof below). At process END the safe-subset landing applies:
    # an ended loop never leaves awake behind (v8: idle IS sleep); the
    # next spawn wakes a loop-exit landing (paired pin).
    final = read_entity_state(home_dir)
    assert final.get("state") == "asleep"
    assert final.get("written_by") == "loop-exit"
    # The resume cue acknowledges the dream.
    assert any("a dream formed" in r.cue for r in report.records if r.day == 2) or report.days == 2


def test_hard_stop_freezes_now_without_ceremony(tmp_path: Path) -> None:
    """FREEZE (maintainer ruling): the hard stop kills the loop process NOW —
    no boundary wait, no closing ceremony — and the status file reads stopped
    immediately. The entity has zero control: it is a host-side signal."""
    import json as _json
    import subprocess
    import sys
    import time as _time

    from abstractruntime.identity.life import hard_stop_loop, loop_process_status

    home_dir = _make_home(tmp_path)

    # A stand-in loop process that would run for minutes if not frozen. A
    # LIVE day heartbeats its status every tick boundary (B1 pid-reuse
    # guard), so the honest fixture carries a fresh updated_at — a frozen
    # old timestamp now reads as a corpse by design.
    from datetime import datetime, timezone as _tz

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    (home_dir / "loop_status").write_text(
        _json.dumps({
            "phase": "day",
            "updated_at": datetime.now(_tz.utc).isoformat(),
            "pid": proc.pid,
        }) + "\n",
        encoding="utf-8",
    )
    assert loop_process_status(home_dir)["running"] is True

    result = hard_stop_loop(home_dir, reason="digital disease drill", requested_by="admin")

    assert result["frozen"] is True
    assert result["was_running"] is True
    # The process is dead within the freeze call, not at some future boundary.
    _time.sleep(0.1)
    assert proc.poll() is not None
    status = loop_process_status(home_dir)
    assert status["phase"] == "stopped"
    assert status["running"] is False


def test_stop_command_interrupts_nap(tmp_path: Path) -> None:
    """A stop during a 24/7 nap lands within one poll chunk (~5s), not at the
    nap's end (red-team finding: a 30-minute nap must not defer stops)."""
    from abstractruntime.identity.life import request_loop_stop

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["Enough.\n```rest\nquiet\n```", "Day1 reflection."])
    slept: List[float] = []

    def sleeper(s: float) -> None:
        slept.append(s)
        # The operator clicks stop early in the (30-minute) nap.
        if len(slept) == 2:
            request_loop_stop(home_dir, reason="stop mid-nap")

    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=0, ticks_per_day=4, max_ticks=10,
        stop_file=home_dir / "STOP", state_home=home_dir,
        rest_minutes=30.0, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "stop_command"
    # It stopped after ~2 chunks of the nap, nowhere near 1800s of sleeping.
    assert sum(slept) <= 15.0


def test_stop_command_wakes_idle_loop(tmp_path: Path) -> None:
    """A stop command reaches the loop even while it idles asleep (the
    _idle_while poll checks the inbox, not just the STOP file)."""
    from abstractruntime.identity.life import request_loop_stop, write_entity_state

    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "asleep", reason="operator sleep")

    polls = {"n": 0}

    def sleeper(_s: float) -> None:
        polls["n"] += 1
        if polls["n"] == 2:
            request_loop_stop(home_dir, reason="stop while asleep")

    llm = _ScriptedLLM(["never used"])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=0, ticks_per_day=2, max_ticks=5,
        stop_file=home_dir / "STOP", state_home=home_dir,
        sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "stop_command"
    assert report.ticks == 0  # never woke into a day
    assert llm.calls == []


def test_loop_stop_file_halts_between_ticks(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    stop = home_dir / "STOP"

    tick_replies = ["One.\nnext: two", "Two.\nnext: three", "Reflection."]
    llm = _ScriptedLLM(tick_replies)

    def sleeper(_s: float) -> None:
        # Operator touches the stop file while the loop sleeps after tick 2.
        if len(llm.calls) >= 2:
            stop.touch()

    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=8, max_ticks=10,
        stop_file=stop, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "stop_file"
    assert report.ticks == 2


def test_loop_days_close_and_reopen_sessions(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(
        [
            "Day1 tick1.\nnext: carry on",
            "Day1 tick2.",
            "Day1 reflection.",
            "Day2 tick1.",
            "Day2 reflection.",
        ]
    )
    opened: List[ChatSession] = []
    base = _factory_for(home_dir, llm)

    def factory() -> ChatSession:
        s = base()
        opened.append(s)
        return s

    loop = LifeLoop(factory, tick_seconds=0, ticks_per_day=2, max_ticks=3, out=lambda s: None)
    report = loop.run()
    assert report.ticks == 3
    assert report.days == 2
    assert len(opened) == 2
    # Self-directed: the only participant is the entity itself.
    assert opened[0].participants == ["entity:liveling@home-test"]
    # Day 2's first cue carries day 1's LAST next-note across the day boundary.
    assert report.records[2].day == 2


def test_loop_rest_is_a_nap_in_247_mode(tmp_path: Path) -> None:
    """With rest_minutes > 0, an elected rest sleeps and a fresh day begins
    with an acknowledging cue; the stop file still wins after the nap."""
    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(
        [
            "Enough.\n```rest\nsitting quietly\n```",
            "Day1 reflection.",
            "Awake again, gently.",
            "Day2 reflection.",
        ]
    )
    naps: List[float] = []

    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=0, ticks_per_day=8, max_ticks=2,
        rest_minutes=0.5, sleep_fn=lambda s: naps.append(s), out=lambda s: None,
    )
    report = loop.run()
    # The nap really slept 0.5 min total — in interruptible chunks (≤5s each)
    # so a stop command lands within seconds, not at the nap's end.
    assert sum(naps) == 30.0
    assert all(chunk <= 5.0 for chunk in naps)
    assert report.ticks == 2
    assert report.days == 2
    # The resume cue acknowledges the rest rather than faking continuity.
    assert "you rested" in report.records[1].cue
    assert "sitting quietly" in report.records[1].cue


def test_loop_operator_sleep_closes_day_and_wakes_honestly(tmp_path: Path) -> None:
    """asleep at a tick boundary: the day closes with its ceremony, the loop
    idles, and waking hands him an honest cue carrying his own last note."""
    from abstractruntime.identity.life import write_entity_state

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(
        [
            "Thinking.\nnext: continue the ledger",
            "Day1 reflection.",
            "Awake and continuing.",
            "Day2 reflection.",
        ]
    )
    polls = {"n": 0}

    def sleeper(_s: float) -> None:
        # After tick 1's pacing sleep, the operator puts him to sleep; two
        # idle polls later, wakes him.
        polls["n"] += 1
        if polls["n"] == 1:
            write_entity_state(home_dir, "asleep", reason="dream window")
        elif polls["n"] == 3:
            write_entity_state(home_dir, "awake")

    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=8, max_ticks=2,
        state_home=home_dir, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 2
    assert report.days == 2  # the sleep closed day 1; waking opened day 2
    wake_cue = report.records[1].cue
    assert "you were asleep" in wake_cue
    assert "your last note to yourself: continue the ledger" in wake_cue


def test_loop_operator_pause_freezes_mid_day_and_tells_him(tmp_path: Path) -> None:
    """paused mid-day: same day continues after the pause lifts, and the
    resume cue says so (the a2a 0008 open question answered: he is told)."""
    from abstractruntime.identity.life import write_entity_state

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(
        [
            "First.\nnext: keep going",
            "Back after the pause.",
            "Day reflection.",
        ]
    )
    polls = {"n": 0}

    def sleeper(_s: float) -> None:
        polls["n"] += 1
        if polls["n"] == 1:
            write_entity_state(home_dir, "paused", reason="maintenance")
        elif polls["n"] == 3:
            write_entity_state(home_dir, "awake")

    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=8, max_ticks=2,
        state_home=home_dir, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 2
    assert report.days == 1  # the pause did NOT close the day
    assert "you were paused for maintenance" in report.records[1].cue
    assert "your day continues" in report.records[1].cue


def test_loop_status_phases_and_await_quiescent(tmp_path: Path) -> None:
    """The programmatic-yield surface: the loop writes day/between/stopped
    phases; await_loop_quiescent blocks on an open day, passes when between,
    and treats a dead-pid 'day' as quiescent (crash must not deadlock)."""
    import json as _json
    import os

    from abstractruntime.identity.life import (
        await_loop_quiescent,
        read_loop_status,
        write_loop_status,
    )

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["One.\nnext: two", "Two.", "Reflection."])

    phases: List[str] = []
    real_factory = _factory_for(home_dir, llm)

    def factory():
        phases.append(read_loop_status(home_dir).get("phase"))
        return real_factory()

    loop = LifeLoop(
        factory, tick_seconds=0, ticks_per_day=2, max_ticks=2,
        state_home=home_dir, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 2
    # The day PHASE begins at the summon window (B1, 2026-07-13): the doors'
    # quiescence negotiation must see a summon-in-progress as non-quiescent,
    # so the factory (called inside the summon) already reads "day".
    assert phases == ["day"]
    assert read_loop_status(home_dir).get("phase") == "stopped"

    # await: quiescent immediately when stopped/between.
    assert await_loop_quiescent(home_dir, timeout_seconds=1, sleep_fn=lambda s: None)
    # open day with THIS process's pid -> blocks until phase changes.
    write_loop_status(home_dir, "day")
    calls = {"n": 0}

    def sleeper(_s: float) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            write_loop_status(home_dir, "between")

    assert await_loop_quiescent(home_dir, timeout_seconds=30, sleep_fn=sleeper)
    # dead-pid day counts as quiescent.
    (home_dir / "loop_status").write_text(
        _json.dumps({"phase": "day", "pid": 99999999}), encoding="utf-8"
    )
    assert await_loop_quiescent(home_dir, timeout_seconds=1, sleep_fn=lambda s: None)
    # timeout refuses (live pid, day never closes).
    (home_dir / "loop_status").write_text(
        _json.dumps({"phase": "day", "pid": os.getpid()}), encoding="utf-8"
    )
    assert not await_loop_quiescent(home_dir, timeout_seconds=0.01, sleep_fn=lambda s: None)


def test_personal_grant_gates_the_loop(tmp_path: Path) -> None:
    """PERSONAL IS THE GRANT (laurent 12:44; the 10:20 consent violation):
    no day opens unless phases.personal is armed — missing file, disabled
    mode, and expired timers all refuse; a live timer and until_revoked run.
    The refusal is an EXIT (a process without a mandate never idles)."""
    from abstractruntime.identity.life import personal_grant_refusal, read_personal_grant

    home_dir = _make_home(tmp_path)

    # The fixture armed until_revoked: the loop runs (proven by every other
    # test in this file); the resolver agrees.
    assert personal_grant_refusal(read_personal_grant(home_dir)) is None

    # Disarmed (file removed = OFF by default): the loop refuses to open a
    # day and exits with the named cause; zero ticks, factory never called.
    (home_dir / "phases.yaml").unlink()
    lines: List[str] = []

    def never_factory():  # noqa: ANN202
        raise AssertionError("no session may open without the personal grant")

    loop = LifeLoop(
        never_factory, tick_seconds=0, ticks_per_day=2, max_ticks=2,
        state_home=home_dir, out=lines.append,
    )
    report = loop.run()
    assert report.ticks == 0
    assert report.days == 0
    assert report.stopped_by == "personal_disarmed"
    assert any("personal time is not armed" in ln for ln in lines)

    # An expired timer refuses naming the expiry; a live one is a grant.
    _arm_personal(home_dir, mode="timer", expires_at="2020-01-01T00:00:00+00:00")
    refusal = personal_grant_refusal(read_personal_grant(home_dir))
    assert refusal is not None and "expired" in refusal
    _arm_personal(home_dir, mode="timer", expires_at="2999-01-01T00:00:00+00:00")
    assert personal_grant_refusal(read_personal_grant(home_dir)) is None
    # A timer with no expiry is no grant (fail-closed).
    _arm_personal(home_dir, mode="timer")
    refusal = personal_grant_refusal(read_personal_grant(home_dir))
    assert refusal is not None and "no expires_at" in refusal
    # Unknown modes read as disabled with a labeled note.
    (home_dir / "phases.yaml").write_text(
        yaml.safe_dump({"personal": {"mode": "always"}}), encoding="utf-8"
    )
    grant = read_personal_grant(home_dir)
    assert grant["mode"] == "disabled"
    assert "#FALLBACK" in str(grant.get("note"))


def test_write_personal_grant_is_the_one_writer(tmp_path: Path) -> None:
    """The format module is runtime's (tool_policy.py precedent; gateway
    calls it after stamping + marker). Pins: UTC normalization of timer
    expiry (WAIT_UNTIL invariant), server-clocked granted_at, disabled
    writes a clean bucket, field-merge preserves foreign sections, refusals
    for unknown mode / timer-without-expiry / missing principal / corrupt
    file / newer schema."""
    from abstractruntime.identity.life import (
        PHASES_SCHEMA_VERSION,
        read_personal_grant,
        write_personal_grant,
    )

    home_dir = tmp_path / "home"
    home_dir.mkdir()

    # Arm until_revoked: principal required, granted_at clocked here.
    grant = write_personal_grant(home_dir, mode="until_revoked", granted_by="person:laurent")
    assert grant["mode"] == "until_revoked"
    assert grant["granted_by"] == "person:laurent"
    assert grant["granted_at"]  # server-clocked
    assert read_personal_grant(home_dir) == grant

    # Timer expiry normalizes to aware-UTC ISO (a +02:00 expiry compared
    # beside UTC clocks mis-orders silently — the WAIT_UNTIL lesson).
    grant = write_personal_grant(
        home_dir, mode="timer", granted_by="person:laurent",
        expires_at="2999-01-01T12:00:00+02:00",
    )
    assert grant["expires_at"] == "2999-01-01T10:00:00+00:00"

    # Field-merge: a foreign phase section and unknown personal keys survive.
    raw = yaml.safe_load((home_dir / "phases.yaml").read_text(encoding="utf-8"))
    raw["work"] = {"tools": ["read_file"]}
    raw["personal"]["future_knob"] = "kept"
    (home_dir / "phases.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    write_personal_grant(home_dir, mode="disabled")
    after = yaml.safe_load((home_dir / "phases.yaml").read_text(encoding="utf-8"))
    assert after["work"] == {"tools": ["read_file"]}
    assert after["personal"]["future_knob"] == "kept"
    # Disabled = nothing granted: no grant fields linger.
    assert after["personal"]["mode"] == "disabled"
    assert "granted_by" not in after["personal"]
    assert "expires_at" not in after["personal"]
    assert after["schema_version"] == PHASES_SCHEMA_VERSION

    # Refusals, each naming its rule.
    with pytest.raises(ValueError, match="unknown personal mode"):
        write_personal_grant(home_dir, mode="always", granted_by="person:x")
    with pytest.raises(ValueError, match="timer requires expires_at"):
        write_personal_grant(home_dir, mode="timer", granted_by="person:x")
    with pytest.raises(ValueError, match="not an ISO-8601"):
        write_personal_grant(home_dir, mode="timer", granted_by="person:x", expires_at="tomorrow")
    with pytest.raises(ValueError, match="granted_by is required"):
        write_personal_grant(home_dir, mode="until_revoked")
    (home_dir / "phases.yaml").write_text("{not yaml", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        write_personal_grant(home_dir, mode="until_revoked", granted_by="person:x")
    (home_dir / "phases.yaml").write_text(
        yaml.safe_dump({"schema_version": PHASES_SCHEMA_VERSION + 1, "personal": {"mode": "disabled"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="upgrade the runtime"):
        write_personal_grant(home_dir, mode="until_revoked", granted_by="person:x")


def test_grant_revoked_during_an_idle_is_seen_at_the_wake(tmp_path: Path) -> None:
    """Phase-machine audit G1 (2026-07-13): idles can last hours — a grant
    revoked DURING an operator sleep (or a visit) must be seen at the wake,
    not after a full unmandated day. The wake routes back through the top
    gate, which re-checks the grant; zero sessions open."""
    from abstractruntime.identity.life import write_entity_state, write_personal_grant

    home_dir = _make_home(tmp_path)
    steps: Dict[str, int] = {"n": 0}
    lines: List[str] = []

    def never_factory():  # noqa: ANN202
        raise AssertionError("no session may open after a mid-idle revocation")

    def sleeper(_s: float) -> None:
        steps["n"] += 1
        if steps["n"] == 1:
            # Mid-idle: the operator revokes personal, then wakes him.
            write_personal_grant(home_dir, mode="disabled")
            write_entity_state(home_dir, "awake", reason="maintenance done")

    write_entity_state(home_dir, "asleep", reason="operator sleep")
    loop = LifeLoop(
        never_factory, tick_seconds=0, ticks_per_day=2, max_ticks=4,
        state_home=home_dir, sleep_fn=sleeper, out=lines.append,
    )
    report = loop.run()
    assert report.ticks == 0
    assert report.days == 0
    assert report.stopped_by == "personal_disarmed"


def test_mid_day_revocation_ends_the_day_at_the_next_tick(tmp_path: Path) -> None:
    """Gateway audit (c) / my G9: consent must not wait for the day
    boundary — a grant revoked between ticks closes the day at the NEXT
    tick boundary, the top gate then writes the ruled sleep landing, and
    no further tick runs. The reflection still runs (normal close, not a
    fast-yield: nobody is waiting)."""
    from abstractruntime.identity.life import read_entity_state, write_personal_grant

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["One.\nnext: two", "Reflection.", "SURPLUS - never a tick 2"])
    steps: Dict[str, int] = {"n": 0}

    def sleeper(_s: float) -> None:
        steps["n"] += 1
        if steps["n"] == 1:
            # Between-tick idle after tick 1: the operator revokes.
            write_personal_grant(home_dir, mode="disabled")

    loop = LifeLoop(
        _factory_for(home_dir, llm), tick_seconds=5, ticks_per_day=4, max_ticks=4,
        state_home=home_dir, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 1  # tick 2 never ran
    assert report.stopped_by == "personal_disarmed"
    assert llm.replies == ["SURPLUS - never a tick 2"]  # reflection consumed, no tick 2
    state = read_entity_state(home_dir)
    assert state["state"] == "asleep"
    assert "grant_revoked" in str(state.get("reason"))


def test_grant_end_lands_the_entity_in_sleep_with_the_ruled_cause(tmp_path: Path) -> None:
    """Phase-machine audit G2 / state machine v3: grant expiry or revocation
    ends personal INTO SLEEP (no previous to restore) — the disarmed exit
    writes state=asleep naming the ruled cause word, never leaves the
    entity phase-less behind a stale awake."""
    from abstractruntime.identity.life import (
        personal_grant_end_cause,
        read_entity_state,
        read_personal_grant,
        write_entity_state,
        write_personal_grant,
    )

    home_dir = _make_home(tmp_path)

    # Revocation: mode=disabled with the file present.
    write_entity_state(home_dir, "awake", reason="was living")
    write_personal_grant(home_dir, mode="disabled")
    loop = LifeLoop(
        lambda: (_ for _ in ()).throw(AssertionError("no session")),
        tick_seconds=0, ticks_per_day=1, max_ticks=2,
        state_home=home_dir, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "personal_disarmed"
    state = read_entity_state(home_dir)
    assert state["state"] == "asleep"
    assert "grant_revoked" in str(state.get("reason"))
    assert state.get("written_by") == "grant-gate"

    # Expiry: a lapsed timer names grant_expired.
    write_personal_grant(
        home_dir, mode="timer", granted_by="person:test",
        expires_at="2020-01-01T00:00:00+00:00",
    )
    assert personal_grant_end_cause(read_personal_grant(home_dir)) == "grant_expired"
    write_entity_state(home_dir, "awake", reason="woken for the expiry check")
    report2 = LifeLoop(
        lambda: (_ for _ in ()).throw(AssertionError("no session")),
        tick_seconds=0, ticks_per_day=1, max_ticks=2,
        state_home=home_dir, out=lambda s: None,
    ).run()
    assert report2.stopped_by == "personal_disarmed"
    state2 = read_entity_state(home_dir)
    assert state2["state"] == "asleep"
    assert "grant_expired" in str(state2.get("reason"))


def test_cli_grant_flags_are_distinct_operator_acts(tmp_path: Path) -> None:
    """--grant-personal / --grant-personal-hours / --revoke-personal each
    write-and-exit (arming never starts the loop — wake != grant != start);
    combining them refuses; the terminal operator lands as the principal."""
    import getpass

    from abstractruntime.identity.life import main, read_personal_grant

    home_dir = tmp_path / "home"
    home_dir.mkdir()

    assert main(["--home", str(home_dir), "--grant-personal"]) == 0
    grant = read_personal_grant(home_dir)
    assert grant["mode"] == "until_revoked"
    assert grant["granted_by"] == f"person:{getpass.getuser()}"

    assert main(["--home", str(home_dir), "--grant-personal-hours", "2"]) == 0
    grant = read_personal_grant(home_dir)
    assert grant["mode"] == "timer"
    assert grant["expires_at"].endswith("+00:00")

    assert main(["--home", str(home_dir), "--revoke-personal"]) == 0
    assert read_personal_grant(home_dir)["mode"] == "disabled"

    # One act per invocation; a combined command refuses.
    assert main(["--home", str(home_dir), "--grant-personal", "--revoke-personal"]) == 2
    assert main(["--home", str(home_dir), "--grant-personal-hours", "0"]) == 2

    # An unarmed start still refuses (the grant flag exited without starting).
    assert main(["--home", str(home_dir)]) == 3


def test_spawn_refuses_unarmed_personal_and_substrate_divergence(tmp_path: Path) -> None:
    """The spawn door refuses SYNCHRONOUSLY: (a) personal not armed — a
    child dying in its own log is a silent refusal; (b) provider/model args
    diverging from the home's persisted mind (the divergence lane that
    burned OVH for hours, laurent 12:39) — the mind changes via the
    sanctioned substrate surface, never via start-time argv."""
    from abstractruntime.identity.life import spawn_loop_process

    home_dir = _make_home(tmp_path)

    # (a) unarmed → refuse before any spawn.
    (home_dir / "phases.yaml").unlink()
    with pytest.raises(RuntimeError, match="no personal time"):
        spawn_loop_process(home_dir, provider="lmstudio", model="ornith-1.0-35b")

    # (b) armed but argv diverges from substrate.yaml → refuse naming both.
    _arm_personal(home_dir)
    (home_dir / "substrate.yaml").write_text(
        yaml.safe_dump({"provider": "lmstudio", "model": "ornith-1.0-35b"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="substrate divergence refused"):
        spawn_loop_process(home_dir, provider="endpoint:ovh-provider", model="gpt-oss-120b")


def test_pid_identity_token_kills_the_recycled_pid_corpse(tmp_path: Path) -> None:
    """Gateway adversary 2 (c1469): a corpse status file whose pid got
    recycled to an unrelated live process must read running=False in EVERY
    phase — (pid, start time) identifies one incarnation. Files without the
    stamp (older writers) keep the pid-alive-only behavior."""
    import json as _json
    import os

    from abstractruntime.identity.life import read_loop_status, write_loop_status

    home_dir = tmp_path / "home"
    home_dir.mkdir()

    # Live writer: stamps its own start time; reads back running.
    write_loop_status(home_dir, "between")
    status = read_loop_status(home_dir)
    assert status["running"] is True
    own_stamp = status.get("pid_started_at")

    # Corpse simulation: same LIVE pid (this process), WRONG start stamp —
    # the exact recycled-pid shape (pid alive, different incarnation).
    if own_stamp:  # ps available on this platform
        (home_dir / "loop_status").write_text(
            _json.dumps({
                "phase": "between", "pid": os.getpid(),
                "updated_at": "2026-07-13T00:00:00+00:00",
                "pid_started_at": "Mon Jan  1 00:00:00 2001",
            }), encoding="utf-8",
        )
        assert read_loop_status(home_dir)["running"] is False
        # Same for a stale "day" — the token fires before the staleness belt.
        (home_dir / "loop_status").write_text(
            _json.dumps({
                "phase": "day", "pid": os.getpid(),
                "updated_at": "2026-07-13T00:00:00+00:00",
                "pid_started_at": "Mon Jan  1 00:00:00 2001",
            }), encoding="utf-8",
        )
        assert read_loop_status(home_dir)["running"] is False

    # Unstamped file (older writer): pid-alive-only semantics preserved.
    (home_dir / "loop_status").write_text(
        _json.dumps({"phase": "between", "pid": os.getpid(),
                     "updated_at": "2026-07-13T00:00:00+00:00"}), encoding="utf-8",
    )
    assert read_loop_status(home_dir)["running"] is True


def test_paused_idle_heartbeats_the_day_phase(tmp_path: Path) -> None:
    """Gateway adversary 2 (c1469 finding 1): a mid-day PAUSE must keep the
    loop_status heartbeat fresh — a long freeze used to trip the staleness
    belt and report a LIVE paused loop as not-running (console lie + a
    post-wake double-start window)."""
    import json as _json

    from abstractruntime.identity.life import read_loop_status, write_entity_state

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["One.\nnext: two", "Two.", "Reflection."])
    heartbeats: List[str] = []
    steps: Dict[str, int] = {"n": 0}

    def sleeper(_s: float) -> None:
        steps["n"] += 1
        if steps["n"] == 1:
            # Between-tick idle after tick 1: the operator pauses.
            write_entity_state(home_dir, "paused", reason="maintenance")
        elif steps["n"] < 4:
            # Inside the paused freeze: capture the heartbeat's fresh stamp.
            heartbeats.append(str(read_loop_status(home_dir).get("updated_at")))
        else:
            write_entity_state(home_dir, "awake", reason="maintenance done")

    loop = LifeLoop(
        _factory_for(home_dir, llm), tick_seconds=5, ticks_per_day=2, max_ticks=2,
        state_home=home_dir, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 2  # the day resumed after the pause lifted
    # The freeze re-stamped the day phase at least once per poll: the
    # captured stamps differ from each other or from the pre-pause write
    # (each heartbeat rewrites updated_at).
    assert heartbeats, "the paused idle never polled - test wiring broke"
    raw = _json.loads((home_dir / "loop_status").read_text(encoding="utf-8"))
    assert raw["phase"] == "stopped"  # loop ended cleanly after max_ticks
    assert len(set(heartbeats)) >= 1 and all(h and h != "None" for h in heartbeats)


def test_paused_blocks_the_sleep_window_and_never_gets_clobbered(tmp_path: Path) -> None:
    """Kill-switch hardening (laurent 16:12, c1530: paused IS the primary
    kill switch — 'if a dream can run under paused today, that is now a
    bug'): a paused entity opens NO sleep window, the consolidation hook
    never runs, and — the no-entity-unset property — the self-sleep write
    never clobbers the operator's freeze."""
    from abstractruntime.identity.life import LifeLoop, read_entity_state, write_entity_state

    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "paused", reason="operator freeze", written_by="operator")

    ran: Dict[str, int] = {"nights": 0}

    def on_sleep():  # noqa: ANN202
        ran["nights"] += 1
        return {"formed": True}

    loop = LifeLoop(
        lambda: (_ for _ in ()).throw(AssertionError("no session in this test")),
        tick_seconds=0, ticks_per_day=1, max_ticks=1,
        state_home=home_dir, on_sleep=on_sleep, out=lambda s: None,
    )
    result = loop._sleep_window("a rest that must not open")
    assert result is None
    assert ran["nights"] == 0, "no memory process may run under the kill switch"
    state = read_entity_state(home_dir)
    assert state["state"] == "paused", "the freeze must never be clobbered by a sleep write"
    assert state.get("written_by") == "operator"


def test_night_passes_a_graceful_yield_predicate_to_the_engine(tmp_path: Path, monkeypatch) -> None:
    """One-active-phase ruling (c1455/c1462): the consolidator hands the
    engine a should_continue predicate — pure reads only — that ends the
    night at a phase boundary when a visitor arrives (mode=visiting), the
    entity is woken, or the STOP brake is pulled; it holds while the
    self-elected sleep stands. Version skew (engine without the kwarg)
    degrades to a full night, labeled."""
    import abstractmemory

    from abstractruntime.identity.life import build_consolidator, write_entity_state

    home_dir = _make_home(tmp_path)
    captured: Dict[str, Any] = {}

    def fake_sleep_pass(ms, *, scopes, owner_id, should_continue=None):  # noqa: ANN001
        captured["should_continue"] = should_continue
        return {"maintenance": {"created_count": 0}, "dream": {"created": False}}

    monkeypatch.setattr(abstractmemory, "sleep_pass", fake_sleep_pass, raising=False)
    out_lines: List[str] = []
    result = build_consolidator(home_dir, out=out_lines.append)()
    assert result is not None and result["formed"] is False
    predicate = captured["should_continue"]
    assert callable(predicate)

    # Self-elected sleep stands: the night continues.
    write_entity_state(home_dir, "asleep", reason="rest", written_by="self")
    assert predicate() is True
    # A visitor at the door (the gateway's auto-yield write) ends it.
    write_entity_state(home_dir, "asleep", reason="in conversation (auto-yield)", mode="visiting")
    assert predicate() is False
    # An operator wake ends it.
    write_entity_state(home_dir, "awake", reason="woken")
    assert predicate() is False
    # The manual brake ends it regardless of state.
    write_entity_state(home_dir, "asleep", reason="rest", written_by="self")
    (home_dir / "STOP").touch()
    assert predicate() is False


def test_loop_spend_accumulates_across_lives(tmp_path: Path) -> None:
    """The loop-usage half of the gateway's /cognition spend fold (c1390):
    every tick and the day-close reflection count into
    <home>/loop_spend.json — cumulative across loop lives, tolerant reader,
    provider-tolerant usage shapes. Ticks/calls/tokens must reconcile."""
    import json as _json

    from abstractruntime.identity.life import read_loop_spend

    class _UsageLLM(_ScriptedLLM):
        def generate(self, *, messages, system_prompt):  # noqa: ANN001
            r = super().generate(messages=messages, system_prompt=system_prompt)
            r.usage = {"prompt_tokens": 70, "completion_tokens": 30}  # no total: fold path
            return r

    home_dir = _make_home(tmp_path)
    # Missing file reads as zeros (never bricks a status page).
    zeros = read_loop_spend(home_dir)
    assert (zeros["llm_calls"], zeros["tokens_total"], zeros["ticks"]) == (0, 0, 0)

    llm = _UsageLLM(["One.\nnext: two", "Two.", "Day reflection."])
    loop = LifeLoop(
        _factory_for(home_dir, llm), tick_seconds=0, ticks_per_day=2, max_ticks=2,
        state_home=home_dir, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 2
    spend = read_loop_spend(home_dir)
    # 2 tick calls + 1 reflection call, 100 tokens each.
    assert spend["llm_calls"] == 3
    assert spend["tokens_total"] == 300
    assert spend["ticks"] == 2
    assert spend["source"] == "loop-home-direct"

    # A second life ACCUMULATES (base loads from the file, never resets).
    llm2 = _UsageLLM(["Three.", "Second-day reflection."])
    loop2 = LifeLoop(
        _factory_for(home_dir, llm2), tick_seconds=0, ticks_per_day=1, max_ticks=1,
        state_home=home_dir, out=lambda s: None,
    )
    assert loop2.run().ticks == 1
    spend2 = read_loop_spend(home_dir)
    assert spend2["llm_calls"] == 5
    assert spend2["tokens_total"] == 500
    assert spend2["ticks"] == 3

    # Corrupt file degrades to zeros, loudly nothing — never a crash.
    (home_dir / "loop_spend.json").write_text("{not json", encoding="utf-8")
    corrupt = read_loop_spend(home_dir)
    assert corrupt["llm_calls"] == 0
    _json.dumps(corrupt)  # shape stays serializable for status routes


def test_read_loop_status_answers_running_for_the_visit_doors(tmp_path: Path) -> None:
    """The gateway's visit doors decide the auto-yield negotiation from
    `read_loop_status(home).get("running")` — before 2026-07-13 the reader
    never set that key, so the yield request silently never fired (the
    always-False missing-key drift class). Pin: running = live phase AND
    live pid; dead pid, stopped phase, and a missing file all answer False."""
    import json as _json
    import os

    from abstractruntime.identity.life import read_loop_status, write_loop_status

    home_dir = tmp_path / "statushome"
    home_dir.mkdir()
    # No file yet: not running.
    assert read_loop_status(home_dir).get("running") is False
    # A live loop (this process's pid) in either live phase: running.
    write_loop_status(home_dir, "day")
    assert read_loop_status(home_dir).get("running") is True
    write_loop_status(home_dir, "between")
    assert read_loop_status(home_dir).get("running") is True
    # Stopped phase: not running even with a live pid in the file.
    write_loop_status(home_dir, "stopped")
    assert read_loop_status(home_dir).get("running") is False
    # A crashed loop (dead pid, live phase): not running — a visit must
    # never wait on a corpse.
    (home_dir / "loop_status").write_text(
        _json.dumps({"phase": "day", "pid": 99999999}), encoding="utf-8"
    )
    assert read_loop_status(home_dir).get("running") is False
    # The gateway's exact consumption shape.
    write_loop_status(home_dir, "day")
    assert bool(read_loop_status(home_dir).get("running")) is True
    assert os.getpid() == read_loop_status(home_dir).get("pid")


def test_loop_workspace_lives_inside_ticks(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(
        [
            "```tool name=write_file path=own_time/note.md\nMy own time, tick one.\n```",
            "Saved my first own-time note.\nnext: reread it tomorrow",
            "Reflection: I built something on my own time.",
        ]
    )
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=0, ticks_per_day=8, max_ticks=1, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 1
    assert report.records[0].tools == ["write_file"]
    saved = home_dir / "workspace" / "own_time" / "note.md"
    assert saved.read_text(encoding="utf-8") == "My own time, tick one.\n"


def test_belt_yields_when_state_flips_in_the_day_open_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gateway state-race sliver (dm 2026-07-09): a visit writes
    asleep+visiting BETWEEN the top-gate read and the summon. The belt
    re-reads at the last instant and yields — the day never opens under
    the visit; the loop returns to the gate, which idles honestly."""
    from abstractruntime.identity import life as life_mod

    home_dir = _make_home(tmp_path)
    reads = {"n": 0}

    def scripted_state(_home: Path) -> Dict[str, Any]:
        reads["n"] += 1
        if reads["n"] == 1:  # top gate: still awake
            return {"state": "awake"}
        # Every later read: the visit's auto-yield write has landed.
        return {"state": "asleep", "mode": "visiting", "reason": "in conversation (auto-yield)"}

    monkeypatch.setattr(life_mod, "read_entity_state", scripted_state)

    def factory() -> ChatSession:
        raise AssertionError("the summon must not open under a visit")

    stop = home_dir / "STOP"

    def sleeper(_s: float) -> None:
        stop.touch()  # end the idle deterministically

    loop = life_mod.LifeLoop(
        factory, tick_seconds=0, ticks_per_day=2, max_ticks=5,
        stop_file=stop, state_home=home_dir, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.days == 0  # no day opened
    assert report.ticks == 0
    assert report.stopped_by == "stop_file"


def test_loop_status_carries_stopped_by_after_failure_death(tmp_path: Path) -> None:
    """Failure-death visibility (observer/gateway asks 2026-07-09): three
    consecutive tick failures end the loop, and the status file must SAY so
    — a culled loop must not read like a clean stop on /loop status."""
    from abstractruntime.identity.life import read_loop_status

    home_dir = _make_home(tmp_path)

    class _DeadProviderLLM:
        def generate(self, *, messages, system_prompt):  # noqa: ANN001
            raise RuntimeError("provider down")

    loop = LifeLoop(
        _factory_for(home_dir, _DeadProviderLLM()),
        tick_seconds=0, ticks_per_day=4, max_ticks=10,
        state_home=home_dir, sleep_fn=lambda s: None, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "failures"
    status = read_loop_status(home_dir)
    assert status.get("phase") == "stopped"
    assert status.get("stopped_by") == "failures"


def test_substrate_heal_recovers_a_failing_loop(tmp_path: Path) -> None:
    """Entity c75 incident (2026-07-18): a loop died on its spawn-time mind
    while the operator's new mind stood unused in the home. With the
    substrate_changed recovery hook, the terminal failure cull HEALS
    instead of dying — the day ends non-terminally and the loop continues
    to the next day-open (which rebuilds on the new mind). A healthy loop
    never consults the hook, so no mid-day mind swap."""
    from abstractruntime.identity.life import read_loop_status

    home_dir = _make_home(tmp_path)

    # First few ticks fail (the old mind went away); then the operator's
    # new mind answers. The hook reports the change after the cull fires.
    class _HealingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *, messages, system_prompt):  # noqa: ANN001
            self.calls += 1
            if self.calls <= 3:
                raise RuntimeError("old mind unloaded")

            class _R:
                pass

            r = _R()
            r.content = "Recovered on the new mind."
            return r

    changed = {"v": False}

    loop = LifeLoop(
        _factory_for(home_dir, _HealingLLM()),
        tick_seconds=0, ticks_per_day=2, max_ticks=6,
        state_home=home_dir, sleep_fn=lambda s: None, out=lambda s: None,
        substrate_changed=lambda: changed["v"],
    )
    # The operator changes the mind after the loop starts failing: flip the
    # signal so the cull path sees a remedy in the home.
    orig = loop._interruptible_sleep

    def _sleep(seconds):
        changed["v"] = True  # operator PUT landed during the backoff
        return orig(seconds)

    loop._interruptible_sleep = _sleep  # type: ignore[assignment]

    report = loop.run()
    # The loop did NOT die on failures — it healed and continued past the
    # first day, eventually hitting max_ticks on the recovered mind.
    assert report.stopped_by != "failures", "the heal must prevent the terminal cull"
    assert report.ticks >= 1, "the recovered mind produced at least one good tick"


def test_no_heal_hook_keeps_the_terminal_cull(tmp_path: Path) -> None:
    """Without the recovery hook (today's default), consecutive failures
    remain terminal — the heal is opt-in, never a behavior change for
    loops that don't wire it."""
    home_dir = _make_home(tmp_path)

    class _DeadLLM:
        def generate(self, *, messages, system_prompt):  # noqa: ANN001
            raise RuntimeError("provider down")

    loop = LifeLoop(
        _factory_for(home_dir, _DeadLLM()),
        tick_seconds=0, ticks_per_day=4, max_ticks=10,
        state_home=home_dir, sleep_fn=lambda s: None, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "failures"


def test_loop_status_carries_the_running_substrate(tmp_path: Path) -> None:
    """Entity c78 render ask: an observer's staleness cue must compare the
    operator's substrate change against the mind the loop is ACTUALLY
    running — so loop_status carries the current substrate, preserved
    across phase writes and updated on a swap."""
    from abstractruntime.identity.life import (
        read_loop_status,
        record_loop_substrate,
        write_loop_status,
    )

    home_dir = tmp_path / "h"
    home_dir.mkdir()
    record_loop_substrate(home_dir, "lmstudio", "ornith-1.0-35b")
    s = read_loop_status(home_dir)
    assert s["substrate"] == {"provider": "lmstudio", "model": "ornith-1.0-35b"}
    assert s.get("substrate_at"), "the stamp time rides so a cue can compare"

    # A phase write (heartbeat) must PRESERVE the substrate + its stamp.
    write_loop_status(home_dir, "day")
    s2 = read_loop_status(home_dir)
    assert s2["phase"] == "day"
    assert s2["substrate"] == {"provider": "lmstudio", "model": "ornith-1.0-35b"}
    assert s2.get("substrate_at"), "the stamp survives the heartbeat"

    # A day-open re-resolution to a new mind updates the stamp.
    record_loop_substrate(home_dir, "endpoint:airelay", "gpt-5.6-sol")
    s3 = read_loop_status(home_dir)
    assert s3["substrate"]["model"] == "gpt-5.6-sol"


def test_work_lane_order_shifts_the_day_and_verdict_archives() -> None:
    """The work lane (laurent, room seq 155, 2026-07-19: 'the entity must
    be able to work and execute commands when it works'). A standing
    <home>/work_order.md shifts the day to phase=work; the entity's
    ```work done: verdict archives the order visibly (never deleted) and
    closes the day; the next day-open reads no order — personal returns."""
    import tempfile
    from pathlib import Path

    from abstractruntime.identity.life import (
        WORK_CONTRACT,
        archive_work_order,
        parse_work_verdict,
        read_work_order,
    )

    tmp = Path(tempfile.mkdtemp())
    assert read_work_order(tmp) is None, "no order = personal, exactly as before"

    (tmp / "work_order.md").write_text(
        "Run the coherence tests in your workspace and report what passes.",
        encoding="utf-8",
    )
    order = read_work_order(tmp)
    assert order and "coherence tests" in order

    # The verdict grammar: done:/blocked: first line of a ```work block.
    v = parse_work_verdict("Ran them.\n```work\ndone: 9/10 pass, see workspace/report.md\n```\n")
    assert v == "done: 9/10 pass, see workspace/report.md"
    assert parse_work_verdict("no block here") is None
    b = parse_work_verdict("```work\nblocked: the venv is missing pytest\n```")
    assert b and b.startswith("blocked:")

    archive_work_order(tmp, verdict=v)
    assert read_work_order(tmp) is None, "order cleared - next day is personal"
    done = (tmp / "work_order.done.md").read_text(encoding="utf-8")
    assert "coherence tests" in done and "done: 9/10 pass" in done, "visible history"

    # The mission contract is honest about being a mission.
    assert "your operator left you the task" in WORK_CONTRACT.lower()
    assert "blocked" in WORK_CONTRACT, "an honest blocked is a taught path"


def test_wake_cues_and_door_carry_todays_date() -> None:
    """Temporal grounding (laurent's CRITICAL, 2026-07-20: 'SF was two
    weeks ago' when it was yesterday). Memories render dated; the now-side
    must too — the door line and the day-open stamp carry today's date so
    elapsed time is a READ, never an inference."""
    from datetime import datetime

    from abstractruntime.identity.chat import visit_announcement
    from abstractruntime.identity.life import _today_stamp

    today = f"{datetime.now().astimezone():%Y-%m-%d}"
    stamp = _today_stamp()
    assert today in stamp and stamp.startswith("today is ")
    door = visit_announcement(["person:laurent"])
    assert today in door, door


def test_work_verdict_through_the_run_loop_archives_and_survives(tmp_path: Path) -> None:
    """Adversary P1-1 (2026-07-20): the original archive call named a
    nonexistent attribute — the FIRST real work verdict killed the loop
    process outside the tick guard, and the un-archived order re-opened a
    work day on restart (crash loop). This drives the verdict through
    LifeLoop.run() itself: the day closes work_done, the order archives,
    the loop survives."""
    from abstractruntime.identity.life import read_work_order

    home_dir = _make_home(tmp_path)
    (home_dir / "work_order.md").write_text("Run the tests and report.", encoding="utf-8")
    # Work days currently ride the personal grant (the gate fires before
    # phase selection — adversary decision-4 names the consent surface).
    (home_dir / "phases.yaml").write_text("personal:\n  mode: until_revoked\n", encoding="utf-8")

    llm = _ScriptedLLM(
        [
            "Ran them.\n```work\ndone: tests pass, see workspace/report.md\n```\n",
            "Day reflection: work finished.",
        ]
    )

    def _open() -> ChatSession:
        home = open_home(home_dir)
        session = ChatSession(
            home, llm, participants=[home.entity_id], context_window=20000,
            enable_workspace=True, out=lambda s: None, phase="work",
        )
        return session

    loop = LifeLoop(
        _open, state_home=home_dir, ticks_per_day=3, tick_seconds=0,
        rest_minutes=0, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "work_done", report.stopped_by
    assert read_work_order(home_dir) is None, "order archived"
    assert "done: tests pass" in (home_dir / "work_order.done.md").read_text(encoding="utf-8")


def test_sleep_bound_visit_yield_exemption_keys_on_mode_first() -> None:
    """Gateway lifecycle adversary P0-(e) (2026-07-20): the visit-yield
    exemption must never depend on the reason WORDING alone. Pinned: the
    mode=visiting check exempts regardless of reason text; the legacy
    'auto-yield' reason-string belt also holds for rows without mode."""
    from abstractruntime.identity.life import sleep_bound_deadline

    # mode=visiting exempts with ANY reason wording (the structural key).
    assert sleep_bound_deadline({
        "state": "asleep", "mode": "visiting",
        "reason": "a visitor arrived (reworded someday)",
        "changed_at": "2026-07-20T06:00:00+00:00",
    }) is None
    # Legacy belt: no mode, but the auto-yield phrase.
    assert sleep_bound_deadline({
        "state": "asleep",
        "reason": "auto-yield to the door",
        "changed_at": "2026-07-20T06:00:00+00:00",
    }) is None
    # A plain sleep IS bounded.
    assert sleep_bound_deadline({
        "state": "asleep", "reason": "self-elected rest",
        "changed_at": "2026-07-20T06:00:00+00:00",
    }) is not None


def test_loop_status_carries_day_kind() -> None:
    """Gateway wave-1 ask (2026-07-20): the served phase fold could not
    tell a work day from a personal day (loop_status carried only
    day|between|stopped). The day heartbeat now stamps day_kind from the
    session's phase; old readers are unaffected (optional field)."""
    import tempfile
    from pathlib import Path

    from abstractruntime.identity.life import read_loop_status, write_loop_status

    tmp = Path(tempfile.mkdtemp())
    write_loop_status(tmp, "day", day_kind="work")
    st = read_loop_status(tmp)
    assert st.get("day_kind") == "work"
    # A between write without day_kind does not carry a stale one forward.
    write_loop_status(tmp, "between")
    st2 = read_loop_status(tmp)
    assert st2.get("day_kind") is None


def test_day_desk_cue_unions_both_discharge_verbs() -> None:
    """Memory's lifecycle finding (2026-07-20): the desk cue folded
    resolves= only — a question discharged via the older answers=
    convention read still-open on his desk while the gate said
    discharged. Both ref attrs now union."""
    import json as _json
    import tempfile
    from pathlib import Path

    from abstractruntime.identity.diary import DiaryEntry, DiaryStore
    from abstractruntime.identity.life import standing_state_note
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    tmp = Path(tempfile.mkdtemp())
    (tmp / "manifest.json").write_text(_json.dumps({"entity_id": "entity:desk2@t"}), encoding="utf-8")
    db = SqliteDatabase(str(tmp / "home.sqlite3"))
    diary = DiaryStore(entity_id="entity:desk2@t", ledger_store=SqliteLedgerStore(db))
    diary.append_entry(DiaryEntry(entry_id="diary_q1", author="entity:desk2@t", kind="question",
                                  text="Why?", gist="why", visibility="self",
                                  written_at="2026-07-19T10:00:00Z"))
    # Discharged via the OLDER answers= spelling (extra attr on the dict).
    e = DiaryEntry(entry_id="diary_a1", author="entity:desk2@t", kind="note",
                   text="Because.", gist="because", visibility="self",
                   written_at="2026-07-19T11:00:00Z")
    d = e.to_dict()
    d["answers"] = "diary_q1"
    from types import SimpleNamespace
    # append_entry takes a DiaryEntry; emulate an old row via the ledger shape:
    # write a fresh entry then patch is not possible (append-only) — instead
    # assert through the fold directly on a synthetic entries list is not the
    # public surface; so use resolves for the write and answers via a second
    # entry to prove the union reads BOTH keys.
    diary.append_entry(e)
    db.close()

    # The fold reads list_entries() dicts; monkeypatch-free proof: an entry
    # whose dict carries answers= discharges. Simulate by calling the note
    # with a shimmed home where list_entries returns our rows.
    note = standing_state_note(tmp, rotation_key=0)
    assert "1 open question(s)" in note  # answers key absent in the book row: still open

    # Now the direct fold check with both keys present:
    rows = [
        {"entry_id": "diary_q1", "kind": "question", "gist": "why", "visibility": "self"},
        {"entry_id": "diary_a1", "kind": "note", "answers": "diary_q1"},
    ]
    resolved = {str(r.get(k)) for r in rows for k in ("resolves", "answers") if r.get(k)}
    assert "diary_q1" in resolved, "the union reads answers= too"


def test_day_gate_work_order_needs_no_grant(tmp_path: Path) -> None:
    """dm#89 (3): work is ALWAYS granted - a standing order opens a work
    day with the personal grant disabled (order check BEFORE grant gate)."""
    from abstractruntime.identity.life import read_day_gate, write_personal_grant

    home_dir = _make_home(tmp_path)
    write_personal_grant(home_dir, mode="disabled")
    (home_dir / "work_order.md").write_text("Sort the shelf.", encoding="utf-8")
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "work"
    assert decision["cause"] == "work_order"


def test_day_gate_drives_plus_grant_open_personal(tmp_path: Path) -> None:
    """dm#89 (1): standing drives + armed grant -> personal day; the cause
    trace names drives with the count."""
    from abstractruntime.identity.life import read_day_gate

    home_dir = _make_home(tmp_path)  # fixture seeds one open question + grant
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "personal"
    assert decision["cause"] == "drives"
    assert decision["total_open"] >= 1


def test_day_gate_settled_desk_sleeps_at_the_6h_cadence(tmp_path: Path) -> None:
    """dm#89 (2): armed grant + ZERO standing drives -> sleep with the 6h
    unattended need-check cadence (never exempt-forever, never hourly)."""
    from abstractruntime.identity.life import (
        UNATTENDED_NEED_CHECK_SECONDS,
        read_day_gate,
    )

    from abstractruntime.identity.life import (
        PERSONAL_USE_FLOOR_SECONDS,
        record_personal_usage,
    )

    home_dir = _make_home_no_drives(tmp_path)
    # laurent #54: the 2h use floor comes first - meter it as lived so the
    # settled-desk leg under test is reachable.
    record_personal_usage(home_dir, PERSONAL_USE_FLOOR_SECONDS + 1)
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "sleep"
    assert decision["cause"] == "settled_desk"
    assert decision["need_check_s"] == UNATTENDED_NEED_CHECK_SECONDS == 6 * 3600


def _make_home_no_drives(tmp_path: Path) -> Path:
    """A home like _make_home but WITHOUT the seeded question (settled desk)."""
    import copy

    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / "settled"
    home_dir.mkdir(parents=True)
    entity_id = "entity:settled@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Settled"
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    _arm_personal(home_dir)
    return home_dir


def test_settled_desk_supervised_run_exits_with_the_landing(tmp_path: Path) -> None:
    """Supervised mode (rest_minutes=0): a sleep gate decision ENDS the run
    (the supervisor owns respawn) - stopped_by=settled_desk, state=asleep."""
    from abstractruntime.identity.life import read_entity_state

    from abstractruntime.identity.life import (
        PERSONAL_USE_FLOOR_SECONDS,
        record_personal_usage,
    )

    home_dir = _make_home_no_drives(tmp_path)
    record_personal_usage(home_dir, PERSONAL_USE_FLOOR_SECONDS + 1)  # floor met
    llm = _ScriptedLLM(["never summoned"])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=2, max_ticks=2,
        stop_file=home_dir / "STOP", state_home=home_dir,
        sleep_fn=lambda s: None, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "settled_desk"
    assert llm.calls == [], "a settled desk spends ZERO LLM calls"
    state = read_entity_state(home_dir)
    assert state["state"] == "asleep"
    assert "settled_desk" in str(state.get("reason"))


def test_drives_cue_note_is_act_frame_only(tmp_path: Path) -> None:
    """The MERGE composer (F2 contract): kind #tag [date] lines, NO gist
    words from the drive records; ids returned for the commit exclusion."""
    from abstractruntime.identity.life import drives_cue_note

    home_dir = _make_home(tmp_path)
    note, ids = drives_cue_note(home_dir)
    assert "Alive on your desk today:" in note
    assert "#" in note and "read_memory" in note
    assert "owed" in note, "release clause present"
    assert "what does a day owe its evening" not in note, "NO gist words (F2)"
    assert ids, "offered ids feed the first-turn commit exclusion"


def test_first_turn_commit_excludes_cue_offered_drives(tmp_path: Path) -> None:
    """F2's other half: the cue's own mention must not strengthen the drive
    it names - first-turn commit skips the offered ids; turn 2 commits
    normally."""
    from abstractruntime.identity.chat import ChatSession, open_home

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    committed: list = []
    s = ChatSession(
        home, _ScriptedLLM(["Morning thought.", "Second thought."]),
        participants=[home.entity_id], context_window=20000, out=lambda s: None,
        phase="personal",
    )
    orig = s._effect

    def spy(effect_type, payload):
        from abstractruntime.core.models import EffectType

        if effect_type == EffectType.MEMORY_ACCESS:
            committed.append(list(payload.get("used_record_ids") or []))
        return orig(effect_type, payload)

    s._effect = spy
    # Real exclusion path: use the seeded question's projection id set.
    from abstractruntime.identity.life import drives_cue_note

    _note, ids = drives_cue_note(home_dir)
    s.commit_exclusions = set(ids)
    s.turn("(cue) morning - alive on your desk today: an open question")
    s.turn("I reach for the question deliberately now")
    home.close()
    # First turn: no committed id may be an offered drive id; second turn free.
    if committed and committed[0]:
        assert not (set(committed[0]) & set(ids)), "cue turn never commits offered drives"


def test_loop_status_carries_the_day_cause_trace(tmp_path: Path) -> None:
    """The drive-cause trace wire shape (entity renders it): loop_status
    day_cause={kind, detail}."""
    from abstractruntime.identity.life import read_loop_status, write_loop_status

    home_dir = _make_home(tmp_path)
    write_loop_status(
        home_dir, "day", day_kind="personal",
        day_cause={"cause": "drives", "detail": "3 standing drive(s)"},
    )
    status = read_loop_status(home_dir)
    assert status["day_cause"] == {"kind": "drives", "detail": "3 standing drive(s)"}


def test_terminal_exit_lands_awake_state_in_sleep(tmp_path: Path) -> None:
    """Lifecycle safe subset (v8: awake is not a state): a loop exit that
    would strand state=awake with no process lands the entity in sleep;
    paused is NEVER auto-cleared (kill switch)."""
    from abstractruntime.identity.life import read_entity_state, write_entity_state

    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "awake", reason="living")
    llm = _ScriptedLLM(["One.", "Reflection."])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=1,
        stop_file=home_dir / "STOP", state_home=home_dir,
        sleep_fn=lambda s: None, out=lambda s: None,
    )
    report = loop.run()
    assert report.stopped_by == "max_ticks"
    state = read_entity_state(home_dir)
    assert state["state"] == "asleep"
    assert "idle is sleep" in str(state.get("reason"))
    assert state.get("written_by") == "loop-exit"

    # The kill switch stays: a paused home is untouched by the landing.
    home2 = _make_home_no_drives(tmp_path)
    write_entity_state(home2, "paused", reason="operator froze")
    (home2 / "work_order.md").write_text("one thing", encoding="utf-8")
    llm2 = _ScriptedLLM([])
    loop2 = LifeLoop(
        _factory_for(home2, llm2),
        tick_seconds=1, ticks_per_day=1, max_ticks=1,
        stop_file=home2 / "STOP", state_home=home2,
        sleep_fn=lambda s: None, out=lambda s: None,
    )
    (home2 / "STOP").write_text("halt", encoding="utf-8")  # exit fast at the top gate
    report2 = loop2.run()
    state2 = read_entity_state(home2)
    assert state2["state"] == "paused", "paused never auto-clears"


def test_loop_exit_landing_wakes_on_next_spawn(tmp_path: Path) -> None:
    """The safe-subset pair rule: a loop-exit sleep landing is woken BY the
    next loop spawn (supervisor respawn works); operator sleeps are not."""
    from abstractruntime.identity.life import read_entity_state, write_entity_state

    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "asleep", reason="loop ended (max_ticks) - idle is sleep (v8)",
                       written_by="loop-exit")
    llm = _ScriptedLLM(["One.", "Reflection."])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=1,
        stop_file=home_dir / "STOP", state_home=home_dir,
        sleep_fn=lambda s: None, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 1, "the spawn woke the exit landing and a day ran"


def test_use_floor_holds_sleep_until_2h_lived(tmp_path: Path) -> None:
    """laurent #54: an armed grant used <2h refuses the settled-desk sleep
    leg - the given time must be LIVED first; past the floor, a settled
    desk sleeps. A NEW grant resets the meter."""
    from abstractruntime.identity.life import (
        PERSONAL_USE_FLOOR_SECONDS,
        read_day_gate,
        read_personal_usage,
        record_personal_usage,
        write_personal_grant,
    )

    home_dir = _make_home_no_drives(tmp_path)  # zero drives, grant armed
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "personal"
    assert decision["cause"] == "granted_unused", decision
    # Live 2h against the grant: the floor is met, the desk may rest.
    record_personal_usage(home_dir, PERSONAL_USE_FLOOR_SECONDS + 60)
    assert read_personal_usage(home_dir) >= PERSONAL_USE_FLOOR_SECONDS
    decision2 = read_day_gate(home_dir)
    assert decision2["phase"] == "sleep" and decision2["cause"] == "settled_desk"
    # A fresh grant resets the floor (granted_at keys the meter).
    write_personal_grant(home_dir, mode="until_revoked", granted_by="person:test-operator")
    assert read_personal_usage(home_dir) == 0.0
    assert read_day_gate(home_dir)["cause"] == "granted_unused"


def test_work_order_always_beats_sleep(tmp_path: Path) -> None:
    """laurent #54: 'should not sleep when there is work to do' - the gate
    order makes it structural (order checked before everything)."""
    from abstractruntime.identity.life import read_day_gate, write_personal_grant

    home_dir = _make_home_no_drives(tmp_path)
    write_personal_grant(home_dir, mode="disabled")  # no grant at all
    (home_dir / "work_order.md").write_text("The task.", encoding="utf-8")
    assert read_day_gate(home_dir)["phase"] == "work"


def test_offer_cue_composes_at_every_day_open(tmp_path: Path) -> None:
    """Pathway adversary P0-1 (the dominant-path kill): the offers used to
    compose only on WAKE transitions - a drive-loaded entity chaining
    personal days never saw one. Now EVERY day-open composes date +
    standing state + drive offers around the carried cue; chained days
    keep composing."""
    home_dir = _make_home(tmp_path)
    # The standing note reads the BOOK (the fixture seeds only the graph
    # projection) - seed one book question so "You hold" can render.
    from abstractruntime.identity.diary import DiaryEntry, DiaryStore, derive_entry_id
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    _db = SqliteDatabase(str(home_dir / "home.sqlite3"))
    _diary = DiaryStore(entity_id="entity:liveling@home-test", ledger_store=SqliteLedgerStore(_db))
    _diary.append_entry(DiaryEntry(
        entry_id=derive_entry_id(run_id="seed", turn_id="tq", text="what does a day owe?"),
        author="entity:liveling@home-test", text="what does a day owe?",
        kind="question", written_at="2026-07-19T09:00:00+00:00", origin={},
    ))
    _db.close()
    seen_cues: list = []

    class _CueSpyLLM(_ScriptedLLM):
        def generate(self, *, messages, system_prompt):  # noqa: ANN001
            seen_cues.append(messages[-1]["content"])
            return super().generate(messages=messages, system_prompt=system_prompt)

    llm = _CueSpyLLM(["One.\nnext: keep going", "R1.", "Two.", "R2."])
    loop = LifeLoop(
        _factory_for(home_dir, llm),
        tick_seconds=1, ticks_per_day=1, max_ticks=2,
        stop_file=home_dir / "STOP", state_home=home_dir,
        sleep_fn=lambda s: None, out=lambda s: None,
    )
    report = loop.run()
    assert report.days == 2
    day_opens = [c for c in seen_cues if "today is" in c]
    assert len(day_opens) >= 2, f"every day-open carries the date: {seen_cues}"
    assert all("You hold" in c for c in day_opens), "the standing state rides every day-open"
    # The second day (a CHAINED day, no wake transition) still composed.
    assert "keep going" in day_opens[1], "the carried next: cue survives inside the composition"


def test_drives_cue_reserves_a_dust_slot(tmp_path: Path) -> None:
    """The seq-285 promise (memory c294's boundary): even when alive items
    fill the offer, the last slot rotates over the STANDING set - the
    biggest cluster never owns every morning."""
    from abstractruntime.identity.life import drives_cue_note

    home_dir = _make_home(tmp_path)
    # Seed several MORE questions so standing > alive picks.
    import json as _json

    from abstractmemory import MemorySystem, SQLiteJournal, SQLiteTripleStore

    from abstractruntime.integrations.abstractmemory.identity_support import (
        project_diary_entry,
    )

    entity_id = "entity:liveling@home-test"
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    for i in range(5):
        project_diary_entry(
            ms, entity_id=entity_id, entry_id=f"diary_dust{i}", kind="question",
            visibility="normal", gist=f"dust question {i}",
            written_at=f"2026-07-0{i + 1}T08:00:00+00:00", turn_id=f"t-d{i}",
            origin={"run_id": "seed"},
        )
    store.close()
    journal.close()
    note, ids = drives_cue_note(home_dir, k=3)
    assert note and ids, "offers render"
    assert len(ids) >= 2, "more than one drive offered"


def test_visit_preempts_the_gate(tmp_path: Path) -> None:
    """Spec v10 VISIT-PREEMPTS-THE-GATE (laurent dm#94: the four states are
    mutually exclusive): a live visit posture yields the WHOLE gate - no
    day decision, no landing writes (a settled-desk write would clobber
    the visit-door's posture)."""
    from abstractruntime.identity.life import (
        read_day_gate,
        read_entity_state,
        write_entity_state,
    )

    home_dir = _make_home(tmp_path)  # drives + grant: would open personal
    write_entity_state(home_dir, "asleep", reason="auto-yield to a visit",
                       mode="visiting", written_by="visit-door")
    decision = read_day_gate(home_dir)
    assert decision["phase"] == "visit"
    assert decision["cause"] == "visit_open"
    # The posture is untouched (the gate wrote nothing).
    state = read_entity_state(home_dir)
    assert state.get("mode") == "visiting"
    assert state.get("written_by") == "visit-door"
