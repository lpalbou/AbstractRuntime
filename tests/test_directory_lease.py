"""Directory writer lease (plan item 1, GW-A): ONE writer per directory.

Pins the primitive (`storage/lease.py` — re-homed from `identity/lease.py`
under the 2026-07-10 vocabulary sign-off) and the loop's writer windows:
mutual exclusion in-process and cross-process, loud refusal naming the
holder, crash-releases (kernel drops flock with the fd), stale-copy
inertness (a copied directory acquires freely), and the loop yielding at
the gate when the home is held.

B1 (laurent 04:58 "i should always be able to visit"; memory c1322 ruled
time-sliced alternation M2-clean): the loop holds the lease PER WINDOW
(summon, one tick's turn, the close look-back) — never per day. The pins
here: the between-tick idle is lease-free (the visit's slot), a held lease
at a tick boundary is a wait (never a failure, never a consumed slot), a
visit yield defers the look-back to the write-ahead marker, and the next
open salvages it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.storage.lease import (
    LEASE_FILENAME,
    DirectoryLease,
    DirectoryLeaseHeld,
    acquire_directory_lease,
    read_directory_lease,
)


def test_second_acquire_refuses_naming_the_holder(tmp_path: Path) -> None:
    with acquire_directory_lease(tmp_path, holder="visit-host", session_id="s-1"):
        with pytest.raises(DirectoryLeaseHeld) as exc:
            acquire_directory_lease(tmp_path, holder="loop")
        # The refusal names the incumbent (diagnostics from the file).
        assert exc.value.holder is not None
        assert exc.value.holder["holder"] == "visit-host"
        assert "one writer per directory" in str(exc.value)
    # Released -> the home is free again.
    lease = acquire_directory_lease(tmp_path, holder="loop")
    assert lease.metadata["holder"] == "loop"
    lease.release()


def test_release_is_idempotent_and_reacquirable(tmp_path: Path) -> None:
    lease = acquire_directory_lease(tmp_path, holder="dream")
    lease.release()
    lease.release()  # second release is a no-op, never an error
    with acquire_directory_lease(tmp_path, holder="maintenance"):
        pass


def test_read_directory_lease_reports_held_via_probe_not_metadata(tmp_path: Path) -> None:
    assert read_directory_lease(tmp_path) is None  # no file yet
    with acquire_directory_lease(tmp_path, holder="visit-host", run_id="r-9"):
        state = read_directory_lease(tmp_path)
        assert state["held"] is True
        assert state["holder"] == "visit-host"
        assert state["run_id"] == "r-9"
    # After release the metadata says released AND the probe says free —
    # the probe is the truth (stale metadata alone must never read as held).
    state = read_directory_lease(tmp_path)
    assert state["held"] is False
    assert state.get("released") is True


def test_stale_copy_is_inert(tmp_path: Path) -> None:
    """A copied home carries the origin's lease FILE but no kernel lock:
    the copy must acquire freely (flock state does not travel with bytes)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    lease = acquire_directory_lease(origin, holder="loop", session_id="day-3")
    # Copy the home while the origin is held (worst case).
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(origin, copy)
    stale = read_directory_lease(copy)
    assert stale["holder"] == "loop"  # stale bytes travelled...
    assert stale["held"] is False  # ...but the lock did not
    with acquire_directory_lease(copy, holder="visit-host"):
        pass  # acquires freely; the origin stays held
    with pytest.raises(DirectoryLeaseHeld):
        acquire_directory_lease(origin, holder="visit-host")
    lease.release()


def test_crashed_holder_releases_with_the_process(tmp_path: Path) -> None:
    """A holder killed mid-window must not wedge the home: the kernel drops
    flock when the process dies. Cross-process is also the honest test of
    the mutual exclusion itself (two real processes, one home)."""
    child_src = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {json.dumps(str(Path(__file__).resolve().parents[1] / "src"))})
        from abstractruntime.storage.lease import acquire_directory_lease
        lease = acquire_directory_lease({json.dumps(str(tmp_path))}, holder="visit-host")
        print("HELD", flush=True)
        time.sleep(60)
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_src], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "HELD"
        # Held by a LIVE foreign process -> this process is refused.
        with pytest.raises(DirectoryLeaseHeld):
            acquire_directory_lease(tmp_path, holder="loop")
        # Kill the holder (crash, no release path runs)...
        child.kill()
        child.wait(timeout=10)
        # ...and the home is free (kernel released the flock with the fd).
        with acquire_directory_lease(tmp_path, holder="loop"):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_identity_lease_shim_is_the_same_primitive(tmp_path: Path) -> None:
    """Migration-window shim: `identity.lease` old spellings alias the ONE
    storage implementation — same class objects (an `except HomeLeaseHeld`
    catches what storage raises) and the SAME lock file, so mixed old/new
    callers keep excluding each other. The shim dies before release."""
    from abstractruntime.identity import lease as shim

    assert shim.HomeLease is DirectoryLease
    assert shim.HomeLeaseHeld is DirectoryLeaseHeld
    assert shim.acquire_home_lease is acquire_directory_lease
    assert shim.read_home_lease is read_directory_lease
    assert shim.LEASE_FILENAME == LEASE_FILENAME
    # One lock file: an old-spelling holder refuses a new-spelling acquire.
    with shim.acquire_home_lease(tmp_path, holder="visit-host"):
        with pytest.raises(DirectoryLeaseHeld):
            acquire_directory_lease(tmp_path, holder="loop")


# ------------------------------------------------------------- loop wiring


def _make_home(tmp_path: Path) -> Path:
    import copy

    yaml = pytest.importorskip("yaml")
    pytest.importorskip("abstractmemory")
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / "leaseling"
    home_dir.mkdir(parents=True)
    entity_id = "entity:leaseling@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Leaseling"
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
    # Personal phase armed (laurent 12:44: own time is OFF by default) —
    # these fixtures run loops, so the operator act is part of the fixture.
    (home_dir / "phases.yaml").write_text(
        yaml.safe_dump({"personal": {
            "mode": "until_revoked",
            "granted_by": "person:test-operator",
            "granted_at": "2026-07-13T00:00:00+00:00",
        }}, sort_keys=False),
        encoding="utf-8",
    )
    return home_dir


class _ScriptedLLM:
    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)

    def generate(self, *, messages, system_prompt):  # noqa: ANN001
        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "…\nnext: keep thinking"
        return r


def test_loop_day_holds_the_lease_and_releases_between_days(tmp_path: Path) -> None:
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.chat import ChatSession, open_home
    from abstractruntime.identity.life import OWN_TIME_CONTRACT, LifeLoop

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["Tick one.\nnext: go on", "Reflection."])
    seen_during_day: Dict[str, Any] = {}

    def factory() -> ChatSession:
        # The summon happens INSIDE the day window: the lease must be held.
        seen_during_day.update(read_directory_lease(home_dir) or {})
        home = open_home(home_dir)
        session = ChatSession(
            home, llm, participants=[home.entity_id], context_window=20000,
            enable_workspace=True, out=lambda s: None,
        )
        session.system_base += "\n\n" + OWN_TIME_CONTRACT
        return session

    loop = LifeLoop(
        factory, tick_seconds=0, ticks_per_day=1, max_ticks=1,
        state_home=home_dir, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 1
    assert seen_during_day.get("held") is True
    assert seen_during_day.get("holder") == "loop"
    # Between days / after the loop: the home is free.
    final = read_directory_lease(home_dir)
    assert final["held"] is False


def test_loop_yields_at_the_gate_while_a_visit_holds_the_home(tmp_path: Path) -> None:
    """A held home never crashes the loop: the day-open acquire refuses,
    the loop idles one poll and returns to the gate (where the stop file
    ends the test deterministically). The summon never opens."""
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.life import LifeLoop

    home_dir = _make_home(tmp_path)
    visit = acquire_directory_lease(home_dir, holder="visit-host", session_id="live-visit")

    def factory():  # noqa: ANN202
        raise AssertionError("the summon must not open while the visit holds the home")

    stop = home_dir / "STOP"

    def sleeper(_s: float) -> None:
        stop.touch()  # end the yield-idle deterministically

    loop = LifeLoop(
        factory, tick_seconds=0, ticks_per_day=2, max_ticks=5,
        stop_file=stop, state_home=home_dir, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.days == 0
    assert report.ticks == 0
    assert report.stopped_by == "stop_file"
    visit.release()


# ------------------------------------------------- B1: per-tick lease windows


def _loop_session_factory(home_dir: Path, llm: Any, probes: Dict[str, Any], out: Any = None):
    """A real-ChatSession factory whose turn() and salvage probe the lease
    mid-write — the only honest way to observe that the TICK and SUMMON
    windows hold the lease while they write. `out` (when given) collects the
    session's own announcements (salvage attribution rides them)."""
    from abstractruntime.identity.chat import ChatSession, open_home

    def factory():  # noqa: ANN202
        home = open_home(home_dir)
        session = ChatSession(
            home, llm, participants=[home.entity_id], context_window=20000,
            enable_workspace=True, out=out or (lambda s: None),
        )
        probes.setdefault("session_ids", []).append(session.session_id)
        original_turn = session.turn
        original_lookback = session.run_pending_lookback

        def probed_turn(cue: str):  # noqa: ANN202
            probes.setdefault("during_turn", []).append(read_directory_lease(home_dir) or {})
            return original_turn(cue)

        def probed_lookback():  # noqa: ANN202
            result = original_lookback()
            if result is not None:
                # Only a REAL salvage (marker found and reflected) records a
                # probe — the no-op path proves nothing about the window.
                probes.setdefault("during_salvage", []).append(read_directory_lease(home_dir) or {})
            return result

        session.turn = probed_turn  # type: ignore[method-assign]
        session.run_pending_lookback = probed_lookback  # type: ignore[method-assign]
        return session

    return factory


def test_between_tick_idle_is_lease_free_and_the_turn_is_leased(tmp_path: Path) -> None:
    """B1 core: the tick's turn runs UNDER the lease; the between-tick idle
    runs WITHOUT it — that free window is where a waiting visit slots in
    (~tick_seconds bound instead of a day-long hold)."""
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.life import LifeLoop

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["One.\nnext: go on", "Two.", "Reflection."])
    probes: Dict[str, Any] = {}

    def sleeper(_s: float) -> None:
        probes.setdefault("during_idle", []).append(read_directory_lease(home_dir) or {})

    loop = LifeLoop(
        _loop_session_factory(home_dir, llm, probes),
        tick_seconds=5, ticks_per_day=2, max_ticks=2,
        state_home=home_dir, sleep_fn=sleeper, out=lambda s: None,
    )
    report = loop.run()
    assert report.ticks == 2
    # Both turns saw the loop's own writer window.
    assert [p.get("held") for p in probes["during_turn"]] == [True, True]
    assert all(p.get("holder") == "loop" for p in probes["during_turn"])
    # Every between-tick idle saw a FREE home (the visit's slot).
    assert probes["during_idle"], "the idle probe never ran - test wiring broke"
    assert all(p.get("held") is False for p in probes["during_idle"])
    # After the loop: free.
    assert (read_directory_lease(home_dir) or {}).get("held") is False


def test_held_lease_at_a_tick_boundary_is_a_wait_never_a_failure(tmp_path: Path) -> None:
    """A writer that takes the home between ticks (a visit's exact move)
    makes the loop WAIT at the boundary — no failure counted, no tick slot
    consumed, and the tick runs once the writer leaves."""
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.life import LifeLoop

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM(["One.\nnext: go on", "Two.", "Reflection."])
    probes: Dict[str, Any] = {}
    lines: List[str] = []
    visitor: Dict[str, Any] = {}

    def sleeper(_s: float) -> None:
        n = visitor.get("sleeps", 0) + 1
        visitor["sleeps"] = n
        if n == 1:
            # Between-tick idle after tick 1: the visit takes the home.
            visitor["lease"] = acquire_directory_lease(home_dir, holder="visit-host")
        elif "lease" in visitor and visitor["lease"] is not None:
            # First wait poll: the visit leaves; the loop's next boundary
            # acquire must succeed.
            visitor["lease"].release()
            visitor["lease"] = None

    loop = LifeLoop(
        _loop_session_factory(home_dir, llm, probes),
        tick_seconds=5, ticks_per_day=2, max_ticks=2,
        state_home=home_dir, sleep_fn=sleeper, out=lines.append,
    )
    report = loop.run()
    assert report.ticks == 2
    assert report.failures == 0
    assert any("waiting at the tick boundary" in ln for ln in lines)
    # No tick slot was consumed by the wait: both ticks fit in ONE day
    # (a slot-consuming wait would close the day after tick 1 and land
    # tick 2 in a second day — mutation caught by this pin).
    assert report.days == 1


def test_visit_yield_defers_the_look_back_and_the_next_open_salvages_it(tmp_path: Path) -> None:
    """B1 fast-yield: when the gateway writes the visiting posture, the day
    closes WITHOUT the inline reflection (quiescence lands right after the
    in-flight turn); the write-ahead marker carries the sheet and the NEXT
    open over the home runs the salvage look-back, attributed to the
    yielded session."""
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.life import LifeLoop, read_loop_status, write_entity_state

    home_dir = _make_home(tmp_path)
    marker = home_dir / "pending_reflection.json"
    stop = home_dir / "STOP"
    # ONE surplus reply: if a mutation reflects inline despite the yield, it
    # consumes this reply and the unconsumed-count pin below catches it (the
    # scripted default-on-empty fallback would mask a bare empty-list check).
    llm = _ScriptedLLM(["Mid-day thought.\nnext: keep going", "SURPLUS - must stay unconsumed"])
    lines: List[str] = []
    steps: Dict[str, int] = {"n": 0}
    probes1: Dict[str, Any] = {}

    def sleeper(_s: float) -> None:
        steps["n"] += 1
        if steps["n"] == 1:
            # Between-tick idle: a visitor arrives. mode="visiting" ALONE
            # (no auto-yield phrase in the reason) — each detection channel
            # must work by itself.
            write_entity_state(
                home_dir, "asleep",
                reason="in conversation with person:test", mode="visiting",
            )
        else:
            stop.touch()  # end the yielded-gate idle deterministically

    loop = LifeLoop(
        _loop_session_factory(home_dir, llm, probes1),
        tick_seconds=5, ticks_per_day=4, max_ticks=4,
        stop_file=stop, state_home=home_dir, sleep_fn=sleeper, out=lines.append,
    )
    report = loop.run()
    assert report.ticks == 1
    assert any("a visitor is at the door" in ln for ln in lines)
    assert any("look-back is deferred" in ln for ln in lines)
    # The look-back did NOT run inline: the marker still pends and the
    # scripted LLM was never asked for a reflection (surplus unconsumed).
    assert marker.exists(), "fast-yield must leave the write-ahead marker in place"
    assert llm.replies == ["SURPLUS - must stay unconsumed"]
    # Quiescence: the day is closed from the visit door's point of view.
    assert read_loop_status(home_dir).get("phase") != "day"
    yielded_session = probes1["session_ids"][0]

    # --- the next open salvages the deferred look-back -------------------
    stop.unlink()
    write_entity_state(home_dir, "awake", reason="visitor session ended (auto-yield return)")
    llm2 = _ScriptedLLM(["Salvage reflection: that day mattered.", "New day tick.", "New day reflection."])
    lines2: List[str] = []
    probes2: Dict[str, Any] = {}
    loop2 = LifeLoop(
        _loop_session_factory(home_dir, llm2, probes2, out=lines2.append),
        tick_seconds=0, ticks_per_day=1, max_ticks=1,
        state_home=home_dir, out=lines2.append,
    )
    report2 = loop2.run()
    assert report2.ticks == 1
    assert any("salvaged look-back" in ln for ln in lines2)
    assert not marker.exists(), "the salvage must retire the write-ahead marker"
    # The salvage ran UNDER the summon window's lease (a home writer), and
    # it was attributed to the YIELDED session, not the salvaging one.
    assert [p.get("held") for p in probes2.get("during_salvage", [])] == [True]
    assert probes2["during_salvage"][0].get("holder") == "loop"
    assert any(yielded_session in ln for ln in lines2), (
        "the salvage announcement must name the yielded session (attribution)"
    )


def test_salvage_helper_repays_the_deferred_look_back_for_doors(tmp_path: Path) -> None:
    """salvage_pending_lookback is the DOOR half of fast-yield: a door that
    hosts no ChatSession (the durable visit lane) can still be "the next
    open" and repay the deferred look-back. No marker = cheap no-op."""
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.chat import ChatSession, open_home, salvage_pending_lookback

    home_dir = _make_home(tmp_path)
    marker = home_dir / "pending_reflection.json"

    # No marker: nothing to do, nothing opened.
    assert salvage_pending_lookback(home_dir, _ScriptedLLM([])) is None

    # A session turns and dies unreflected — the write-ahead marker pends.
    home = open_home(home_dir)
    session = ChatSession(
        home, _ScriptedLLM(["A thought worth keeping."]),
        participants=[home.entity_id], context_window=20000, out=lambda s: None,
    )
    session.turn("what stays with you?")
    home.close()
    assert marker.exists()

    # The door salvages it (caller holds the lease, as the contract says).
    with acquire_directory_lease(home_dir, holder="visit-host"):
        result = salvage_pending_lookback(home_dir, _ScriptedLLM(["Salvage look-back."]))
    assert result is not None
    assert not marker.exists(), "the salvage must retire the marker"


def test_visit_yield_detects_the_auto_yield_reason_channel(tmp_path: Path) -> None:
    """The gateway's OTHER spelling: some writers carry only the auto-yield
    phrase in the reason (no mode field). Each channel must fire alone."""
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.life import LifeLoop, write_entity_state

    home_dir = _make_home(tmp_path)
    stop = home_dir / "STOP"
    llm = _ScriptedLLM(["One tick.\nnext: more", "SURPLUS"])
    lines: List[str] = []
    steps: Dict[str, int] = {"n": 0}

    def sleeper(_s: float) -> None:
        steps["n"] += 1
        if steps["n"] == 1:
            write_entity_state(
                home_dir, "asleep",
                reason="in conversation with person:test (auto-yield)",
            )
        else:
            stop.touch()

    loop = LifeLoop(
        _loop_session_factory(home_dir, llm, {}),
        tick_seconds=5, ticks_per_day=4, max_ticks=4,
        stop_file=stop, state_home=home_dir, sleep_fn=sleeper, out=lines.append,
    )
    report = loop.run()
    assert report.ticks == 1
    assert any("a visitor is at the door" in ln for ln in lines)
    assert (home_dir / "pending_reflection.json").exists()
    assert llm.replies == ["SURPLUS"]
