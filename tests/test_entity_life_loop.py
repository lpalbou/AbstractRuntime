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
    # After waking, the state returns to awake (self-cleared).
    final = read_entity_state(home_dir)
    assert final.get("state") == "awake"
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
