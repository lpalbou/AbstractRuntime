"""Per-home lease (plan item 1, GW-A): ONE writer per home at a time.

Pins the primitive (`identity/lease.py`) and the loop's day/dream windows:
mutual exclusion in-process and cross-process, loud refusal naming the
holder, crash-releases (kernel drops flock with the fd), stale-copy
inertness (a copied home acquires freely), and the loop yielding at the
gate when the home is held.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.identity.lease import (
    LEASE_FILENAME,
    HomeLease,
    HomeLeaseHeld,
    acquire_home_lease,
    read_home_lease,
)


def test_second_acquire_refuses_naming_the_holder(tmp_path: Path) -> None:
    with acquire_home_lease(tmp_path, holder="visit-host", session_id="s-1"):
        with pytest.raises(HomeLeaseHeld) as exc:
            acquire_home_lease(tmp_path, holder="loop")
        # The refusal names the incumbent (diagnostics from the file).
        assert exc.value.holder is not None
        assert exc.value.holder["holder"] == "visit-host"
        assert "one writer per home" in str(exc.value)
    # Released -> the home is free again.
    lease = acquire_home_lease(tmp_path, holder="loop")
    assert lease.metadata["holder"] == "loop"
    lease.release()


def test_release_is_idempotent_and_reacquirable(tmp_path: Path) -> None:
    lease = acquire_home_lease(tmp_path, holder="dream")
    lease.release()
    lease.release()  # second release is a no-op, never an error
    with acquire_home_lease(tmp_path, holder="maintenance"):
        pass


def test_read_home_lease_reports_held_via_probe_not_metadata(tmp_path: Path) -> None:
    assert read_home_lease(tmp_path) is None  # no file yet
    with acquire_home_lease(tmp_path, holder="visit-host", run_id="r-9"):
        state = read_home_lease(tmp_path)
        assert state["held"] is True
        assert state["holder"] == "visit-host"
        assert state["run_id"] == "r-9"
    # After release the metadata says released AND the probe says free —
    # the probe is the truth (stale metadata alone must never read as held).
    state = read_home_lease(tmp_path)
    assert state["held"] is False
    assert state.get("released") is True


def test_stale_copy_is_inert(tmp_path: Path) -> None:
    """A copied home carries the origin's lease FILE but no kernel lock:
    the copy must acquire freely (flock state does not travel with bytes)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    lease = acquire_home_lease(origin, holder="loop", session_id="day-3")
    # Copy the home while the origin is held (worst case).
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(origin, copy)
    stale = read_home_lease(copy)
    assert stale["holder"] == "loop"  # stale bytes travelled...
    assert stale["held"] is False  # ...but the lock did not
    with acquire_home_lease(copy, holder="visit-host"):
        pass  # acquires freely; the origin stays held
    with pytest.raises(HomeLeaseHeld):
        acquire_home_lease(origin, holder="visit-host")
    lease.release()


def test_crashed_holder_releases_with_the_process(tmp_path: Path) -> None:
    """A holder killed mid-window must not wedge the home: the kernel drops
    flock when the process dies. Cross-process is also the honest test of
    the mutual exclusion itself (two real processes, one home)."""
    child_src = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {json.dumps(str(Path(__file__).resolve().parents[1] / "src"))})
        from abstractruntime.identity.lease import acquire_home_lease
        lease = acquire_home_lease({json.dumps(str(tmp_path))}, holder="visit-host")
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
        with pytest.raises(HomeLeaseHeld):
            acquire_home_lease(tmp_path, holder="loop")
        # Kill the holder (crash, no release path runs)...
        child.kill()
        child.wait(timeout=10)
        # ...and the home is free (kernel released the flock with the fd).
        with acquire_home_lease(tmp_path, holder="loop"):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


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
        seen_during_day.update(read_home_lease(home_dir) or {})
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
    final = read_home_lease(home_dir)
    assert final["held"] is False


def test_loop_yields_at_the_gate_while_a_visit_holds_the_home(tmp_path: Path) -> None:
    """A held home never crashes the loop: the day-open acquire refuses,
    the loop idles one poll and returns to the gate (where the stop file
    ends the test deterministically). The summon never opens."""
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.life import LifeLoop

    home_dir = _make_home(tmp_path)
    visit = acquire_home_lease(home_dir, holder="visit-host", session_id="live-visit")

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
