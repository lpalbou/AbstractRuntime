"""THE Stop fence (STOP-LANE-DESIGN 2026-11-09; Codex review 3 #1/#2/#5/#6): one primitive (`LifecycleFence`), a
closed set of commit points that take it as a REQUIRED argument, fail closed, and an int-compare hot path.

Every test here goes red with the fence removed (a writer that takes no fence, a check outside the lock, a fail-open
read, a re-parse per check). Only the model and process creation are stubbed; the admission path is the real one."""

from __future__ import annotations

import ast
import builtins
import collections
import contextlib
import fcntl
import inspect
import io
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from test_entity_life_loop import _factory_for, _make_home  # noqa: E402
from test_entity_phases_runtime import _HookLLM, _history  # noqa: E402

import abstractruntime.identity.life as life  # noqa: E402
import abstractruntime.identity.lifecycle_fence as lf  # noqa: E402
from abstractruntime.identity.chat import TurnCancelled  # noqa: E402
from abstractruntime.identity.life import (  # noqa: E402
    LifeLoop,
    StaleLifecycleGeneration,
    bump_lifecycle_generation,
    hard_stop_loop,
    migrate_phase_sidecar,
    read_entity_state,
    read_lifecycle_generation,
    read_resume_to,
    write_entity_state,
    write_loop_status,
    write_personal_grant,
    write_resume_to,
)
from abstractruntime.identity.lifecycle_fence import (  # noqa: E402
    KILL_SWITCH,
    RESUME,
    LifecycleFence,
    LifecycleUnreadable,
    StoppedByOperator,
)

RUNTIME_SRC = Path(life.__file__).resolve().parents[1]

# THE CLOSED SET of runtime state commit points (design §2.2): each takes `fence` keyword-only, with NO default.
COMMIT_POINTS = (
    "write_entity_state", "_mutate_phases", "write_personal_grant", "write_resume_to", "_write_phases_bucket",
    "migrate_phase_sidecar", "write_loop_status", "touch_loop_heartbeat", "record_loop_substrate",
    "append_phase_changed", "_beat_before_each_call",
)
# Modules whose unfenced calls are still to be fenced in a later step of this lane (chat.py is another worker's file
# until the lead frees it). The scan below fails when such a module is clean, so the exemption cannot outlive its fix.
PENDING_FENCE = {"identity/chat.py"}


def _bump_in_another_process(home_dir: Path) -> int:
    """The gateway's Stop, from ANOTHER process (this one is then no bumper: it sees the Stop by the signal stat)."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from abstractruntime.identity.life import (bump_lifecycle_generation, home_state_lock, write_entity_state,\n"
        "    write_loop_status, write_personal_grant, write_resume_to)\n"
        "from abstractruntime.identity.lifecycle_fence import KILL_SWITCH\n"
        "home = Path(sys.argv[1])\n"
        "with home_state_lock(home, timeout_s=None):\n"
        "    write_personal_grant(home, mode='disabled', granted_by='kill_switch', fence=KILL_SWITCH)\n"
        "    write_resume_to(home, None, written_by='kill_switch', fence=KILL_SWITCH)\n"
        "    g = bump_lifecycle_generation(home, written_by='gateway:kill_switch', timeout_s=None)\n"
        "    write_entity_state(home, 'paused', reason='stopped [by test via another process]', fence=KILL_SWITCH)\n"
        "write_loop_status(home, 'stopped', stopped_by='kill_switch', owner=(None, None), fence=KILL_SWITCH)\n"
        "print(g)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(RUNTIME_SRC.parent)}  # the same runtime tree as this test imports
    out = subprocess.run([sys.executable, "-P", "-c", code, str(home_dir)], capture_output=True, text=True,
                         timeout=120, env=env)
    assert out.returncode == 0, out.stderr
    return int(out.stdout.strip().splitlines()[-1])


def _stop_here(home_dir: Path) -> int:
    """The gateway's Stop, from THIS process (the bumper: its fences see it with zero I/O). The loop under test runs in
    this process too, so hard_stop_loop refuses to signal it (the recorded pid is the caller) - the refused kill."""
    out = hard_stop_loop(home_dir, reason="operator stop")
    assert out["killed"] is False, out
    write_entity_state(home_dir, "paused", reason="stopped [by test]", fence=KILL_SWITCH)
    return int(out["lifecycle_generation"])


def _snapshot(home_dir: Path) -> Dict[str, Any]:
    """Every file under the home (bytes), sqlite files as their table row counts (a WAL checkpoint is no change)."""
    import sqlite3

    snap: Dict[str, Any] = {}
    for p in sorted(home_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(home_dir))
        if rel.endswith(("-wal", "-shm")) or rel in (life.HOME_STATE_LOCK_FILENAME, ".writer_lease"):
            continue  # locks, not records: a stopping loop lets go of the home's writer lease
        if rel.endswith(".sqlite3"):
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                snap[rel] = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
            finally:
                con.close()
            continue
        snap[rel] = p.read_bytes()
    return snap


# ------------------------------------------------------------------ I6 / T6: an unfenced write does not run


def test_every_commit_point_requires_a_fence_and_none_has_a_default() -> None:
    """I6 (design §3): the signature makes the unfenced write impossible - `fence` keyword-only, no default, and the
    old optional `expected_generation` gone. Red when any commit point regains a default or loses the parameter."""
    for name in COMMIT_POINTS:
        params = inspect.signature(getattr(life, name)).parameters
        assert "fence" in params, f"{name} takes no fence"
        assert params["fence"].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params["fence"].default is inspect.Parameter.empty, f"{name}'s fence has a default"
        assert "expected_generation" not in params, f"{name} still takes the optional expected_generation"


def _unfenced_calls(path: Path) -> List[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if name in COMMIT_POINTS and not any(k.arg in ("fence", None) for k in node.keywords):
                lines.append(node.lineno)
    return lines


def test_no_runtime_module_calls_a_commit_point_without_a_fence() -> None:
    """The static half of I6: a call site that never runs in a test still cannot drop its fence."""
    offenders: Dict[str, List[int]] = {}
    pending_clean: List[str] = []
    for path in sorted(RUNTIME_SRC.rglob("*.py")):
        rel = str(path.relative_to(RUNTIME_SRC))
        found = _unfenced_calls(path)
        if rel in PENDING_FENCE:
            if not found:
                pending_clean.append(rel)
            continue
        if found:
            offenders[rel] = found
    assert not offenders, f"unfenced commit-point calls: {offenders}"
    assert not pending_clean, f"{pending_clean} is fenced now - drop it from PENDING_FENCE"


def test_an_unfenced_or_misused_write_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "asleep", reason="operator sleep", fence=LifecycleFence.admit(home_dir))
    phases_before = (home_dir / "phases.yaml").read_text(encoding="utf-8")
    state_before = (home_dir / "state").read_bytes()
    history_before = (home_dir / "state_history.jsonl").read_bytes()
    with pytest.raises(TypeError):
        write_entity_state(home_dir, "awake", reason="no fence")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        write_resume_to(home_dir, "personal", written_by="x")  # type: ignore[call-arg]
    # The two sentinels pass unchecked ONLY for what a Stop / a Resume writes.
    for misuse in (
        lambda: write_entity_state(home_dir, "awake", reason="x", fence=KILL_SWITCH),
        lambda: write_entity_state(home_dir, "paused", reason="x", fence=RESUME),
        lambda: write_entity_state(home_dir, "asleep", reason="x", fence=RESUME),
        lambda: write_personal_grant(home_dir, mode="until_revoked", granted_by="person:x", fence=KILL_SWITCH),
        lambda: write_personal_grant(home_dir, mode="until_revoked", granted_by="person:x", fence=RESUME),
        lambda: write_resume_to(home_dir, "personal", written_by="x", fence=RESUME),
        lambda: write_loop_status(home_dir, "day", fence=KILL_SWITCH),
        lambda: write_loop_status(home_dir, "stopped", fence=RESUME),
        lambda: migrate_phase_sidecar(home_dir, fence=KILL_SWITCH),
        lambda: write_entity_state(home_dir, "awake", reason="x", fence=7),
    ):
        with pytest.raises(TypeError):
            misuse()
    other = tmp_path / "other-home"
    other.mkdir()
    with pytest.raises(ValueError):
        write_entity_state(home_dir, "awake", reason="x", fence=LifecycleFence.admit(other))
    assert (home_dir / "phases.yaml").read_text(encoding="utf-8") == phases_before
    assert (home_dir / "state").read_bytes() == state_before
    assert (home_dir / "state_history.jsonl").read_bytes() == history_before
    assert not (home_dir / "loop_status").exists()


def test_a_migration_decided_before_stop_is_refused_after_it(tmp_path: Path) -> None:
    """T6 / Codex review 3 #5: migrate_phase_sidecar checked the bucket's absence, then wrote unfenced - a migration
    racing a Stop brought back the sidecar's personal return. Now it carries a fence: refused after the Stop, and the
    Stop's cleared return stands."""
    home_dir = _make_home(tmp_path)
    sidecar = life.gateway_phase_sidecar_path(home_dir)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"resume_to": "personal"}), encoding="utf-8")
    decided = LifecycleFence.admit(home_dir)
    _stop_here(home_dir)
    phases_before = (home_dir / "phases.yaml").read_text(encoding="utf-8")
    with pytest.raises(StaleLifecycleGeneration):
        migrate_phase_sidecar(home_dir, fence=decided)
    assert (home_dir / "phases.yaml").read_text(encoding="utf-8") == phases_before
    assert read_resume_to(home_dir)["phase"] is None
    # Admitted after the Stop, the migration finds the bucket the Stop wrote and leaves it.
    assert migrate_phase_sidecar(home_dir, fence=LifecycleFence.admit(home_dir))["resume_to"] == "present"
    assert read_resume_to(home_dir)["phase"] is None


def test_a_migration_racing_stop_never_restores_the_old_return(tmp_path: Path, monkeypatch) -> None:
    """The interleaving itself: the Stop arrives while the migration has read the sidecar. The read and the write
    are one step inside the home state lock now, so the Stop waits for it and clears the return after it."""
    home_dir = _make_home(tmp_path)
    sidecar = life.gateway_phase_sidecar_path(home_dir)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"resume_to": "personal"}), encoding="utf-8")
    real_read = life._read_gateway_phase_sidecar
    stopper = threading.Thread(target=lambda: hard_stop_loop(home_dir, reason="operator stop"))
    fired = {"n": 0}

    def racing_read(h):
        out = real_read(h)
        if threading.current_thread() is not stopper and not fired["n"]:
            fired["n"] += 1
            stopper.start()  # the migration has read the sidecar: the operator's Stop lands now
            stopper.join(timeout=1.0)
        return out

    monkeypatch.setattr(life, "_read_gateway_phase_sidecar", racing_read)
    fence = LifecycleFence.admit(home_dir)
    try:
        migrate_phase_sidecar(home_dir, fence=fence)
    except StaleLifecycleGeneration:
        pass  # refused is as good: nothing restored
    stopper.join(timeout=10)
    assert fired["n"] == 1 and not stopper.is_alive()
    assert read_resume_to(home_dir)["phase"] is None, "the migration brought back the personal return after Stop"


# ------------------------------------------------------------------ the primitive


def test_the_fence_moves_with_the_stop_and_its_refusal_is_the_operators_stop(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    fence = LifecycleFence.admit(home_dir)
    assert fence.generation == 0 and not fence.moved()
    fence.check("before nothing")  # passes
    assert _stop_here(home_dir) == 1
    assert fence.moved() and fence.truth_moved()
    with pytest.raises(StoppedByOperator) as stopped:
        fence.check("before a model call")
    # One exception: a TurnCancelled (lanes map it to a clean stopped-by-the-operator record), never a failure.
    assert isinstance(stopped.value, TurnCancelled) and stopped.value.where == "before a model call"
    assert issubclass(StaleLifecycleGeneration, StoppedByOperator)
    with pytest.raises(StaleLifecycleGeneration):
        fence.commit("a host's own write", lambda: (_ for _ in ()).throw(AssertionError("ran after Stop")))
    # Every phases.yaml writer decided before the Stop is refused after it, the file byte-identical.
    phases_before = (home_dir / "phases.yaml").read_bytes()
    for late in (
        lambda: write_personal_grant(home_dir, mode="until_revoked", granted_by="person:late", fence=fence),
        lambda: write_resume_to(home_dir, "personal", written_by="loop:late", fence=fence),
        lambda: life._write_phases_bucket(home_dir, "resume_to", {"phase": "personal"}, fence=fence),
    ):
        with pytest.raises(StaleLifecycleGeneration):
            late()
    assert (home_dir / "phases.yaml").read_bytes() == phases_before
    assert life.read_personal_grant(home_dir)["mode"] == "disabled" and read_resume_to(home_dir)["phase"] is None
    fresh = LifecycleFence.admit(home_dir)
    assert fresh.generation == 1 and not fresh.moved()
    assert fresh.commit("a host's own write", lambda: "ran") == "ran"
    assert (home_dir / lf.LIFECYCLE_SIGNAL_FILENAME).read_text(encoding="utf-8").strip() == "1"


def test_a_stop_in_another_process_is_seen_at_the_next_check(tmp_path: Path) -> None:
    """Non-bumper mode: this process did not bump; it sees the Stop at its next check through the signal file's stat
    (one read of the int when it changed), with no re-parse of phases.yaml."""
    home_dir = _make_home(tmp_path)
    fence = LifecycleFence.admit(home_dir)
    assert not fence.moved()
    assert _bump_in_another_process(home_dir) == 1
    with _counting() as counts:
        assert fence.moved()
        assert fence.moved()
    assert counts["yaml.safe_load"] == 0 and counts["os.stat"] == 1, counts  # the second compare needs no I/O
    assert counts["open"] == 1, counts  # the int is read once, when the stat changed
    with pytest.raises(StaleLifecycleGeneration):
        write_entity_state(home_dir, "awake", reason="late wake", fence=fence)
    assert read_entity_state(home_dir)["state"] == "paused"


def test_a_stale_reader_never_undoes_a_published_stop(tmp_path: Path, monkeypatch) -> None:
    """An admission that parsed phases.yaml just BEFORE a Stop and publishes just AFTER it must not move this
    process's table back (the Stop's own fences would read not-moved): its fence is the stale one, refused."""
    home_dir = _make_home(tmp_path)
    LifecycleFence.admit(home_dir)
    real_load = life._load_phases_for_write
    raced = {"done": False}

    def load_then_stop(h, what):
        data = real_load(h, what)  # the reader has the pre-Stop file in hand
        if not raced["done"] and "reading" in what:
            raced["done"] = True
            bump_lifecycle_generation(home_dir, written_by="test:racing-stop")
        return data

    monkeypatch.setattr(life, "_load_phases_for_write", load_then_stop)
    stale = LifecycleFence.admit(home_dir)
    monkeypatch.setattr(life, "_load_phases_for_write", real_load)
    assert raced["done"] and stale.generation == 0
    assert stale.moved(), "the stale admission undid the Stop in this process's table"
    assert not LifecycleFence.admit(home_dir).moved() and read_lifecycle_generation(home_dir) == 1


# ------------------------------------------------------------------ T8 / I7: fail closed


def test_an_unreadable_generation_fails_closed_and_the_loop_refuses(tmp_path: Path) -> None:
    """T8 (design §1 F: the loop's fence was fail-open twice - `_generation=None` "not fenced" and a False on any
    exception). An unreadable phases.yaml now reads as MOVED, admits nothing, and a loop does not start: it writes
    nothing at all."""
    home_dir = _make_home(tmp_path)
    fence = LifecycleFence.admit(home_dir)
    (home_dir / "phases.yaml").write_text("personal: [unclosed\n", encoding="utf-8")
    with pytest.raises(LifecycleUnreadable):
        LifecycleFence.admit(home_dir)
    with pytest.raises(LifecycleUnreadable):
        fence.truth_moved()
    with pytest.raises(ValueError):
        write_entity_state(home_dir, "awake", reason="x", fence=fence)
    before = _snapshot(home_dir)
    lines: List[str] = []

    def _never():  # noqa: ANN202
        raise AssertionError("no session may open over an unreadable generation")

    report = LifeLoop(_never, tick_seconds=0, ticks_per_day=2, max_ticks=3, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lines.append).run()
    assert report.stopped_by == "generation_unreadable" and report.ticks == 0, report
    assert _snapshot(home_dir) == before, "a loop over an unreadable generation wrote to the home"
    assert not any("#FALLBACK" in ln for ln in lines) and any("unreadable" in ln for ln in lines), lines


def test_a_vanished_or_garbled_signal_reads_as_moved(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    _bump_in_another_process(home_dir)  # the signal file exists (this process never bumped: the stat path runs)
    fence = LifecycleFence.admit(home_dir)
    assert fence.generation == 1 and not fence.moved()
    signal = home_dir / lf.LIFECYCLE_SIGNAL_FILENAME
    signal.write_text("not a number\n", encoding="utf-8")
    assert fence.moved(), "a garbled signal file read as not moved"
    signal.unlink()
    assert fence.moved(), "a signal file that vanished read as not moved"


# ------------------------------------------------------------------ the loop: one fence, nothing after the Stop


@pytest.mark.parametrize("where", ["this process", "another process"])
def test_a_stop_mid_day_ends_the_loop_with_nothing_written_after_it(tmp_path: Path, where: str) -> None:
    """A Stop during the second tick (the first already ran, so the day has something to look back on): no model
    call after it (the day's look-back used to run at the close), the loop never writes its status, state, spend or
    a marker over the Stop's records, and nothing in its log reads as a failure."""
    home_dir = _make_home(tmp_path)
    seen: Dict[str, Any] = {}

    def _stop() -> None:
        seen["generation"] = _stop_here(home_dir) if where == "this process" else _bump_in_another_process(home_dir)
        seen["snapshot"] = _snapshot(home_dir)

    llm = _HookLLM(["First thought.\nnext: keep going", "A look-back.", "Second thought - the Stop lands now.",
                    "NEVER - no model call after the Stop", "NEVER 2", "NEVER 3"], hooks={2: _stop})
    lines: List[str] = []
    report = LifeLoop(_factory_for(home_dir, llm), tick_seconds=0, ticks_per_day=4, max_ticks=6, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lines.append).run()
    assert seen["generation"] == 1
    assert report.stopped_by == "kill_switch" and report.failures == 0, report
    assert llm.n == 3 and llm.replies[0] == "NEVER - no model call after the Stop", (llm.n, llm.replies)
    after = _snapshot(home_dir)
    changed = sorted(k for k in set(after) | set(seen["snapshot"]) if after.get(k) != seen["snapshot"].get(k))
    assert not changed, f"the loop wrote to the home after the Stop: {changed}"
    status = json.loads((home_dir / "loop_status").read_text(encoding="utf-8"))
    assert status["phase"] == "stopped" and status["stopped_by"] == "kill_switch", status
    assert read_entity_state(home_dir)["state"] == "paused"
    assert _history(home_dir)[-1]["state"] == "paused", "a row landed after the paused row"
    assert not [ln for ln in lines if "#FALLBACK" in ln or "failed" in ln.lower()], lines
    assert sum("this loop ends here" in ln for ln in lines) == 1, lines


def test_a_loop_given_a_moved_fence_writes_nothing(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    fence = LifecycleFence.admit(home_dir)
    _stop_here(home_dir)
    before = _snapshot(home_dir)
    lines: List[str] = []

    def _never():  # noqa: ANN202
        raise AssertionError("no session may open after the Stop")

    report = LifeLoop(_never, tick_seconds=0, ticks_per_day=2, max_ticks=3, rest_minutes=30,
                      stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                      out=lines.append, fence=fence).run()
    assert report.stopped_by == "kill_switch" and _snapshot(home_dir) == before
    assert any("stopped before it started" in ln for ln in lines) and not any("#FALLBACK" in ln for ln in lines)


# ------------------------------------------------------------------ §7: the hot path budget


_COUNTED = (
    (builtins, "open", "open"), (io, "open", "open"), (os, "open", "os.open"), (os, "stat", "os.stat"),
    (pathlib.Path, "read_text", "Path.read_text"), (pathlib.Path, "read_bytes", "Path.read_bytes"),
    (pathlib.Path, "exists", "Path.exists"), (fcntl, "flock", "flock"), (subprocess, "run", "subprocess.run"),
    (subprocess, "Popen", "subprocess.Popen"), (yaml, "safe_load", "yaml.safe_load"),
)


@contextlib.contextmanager
def _counting() -> Iterator[collections.Counter]:
    counts: collections.Counter = collections.Counter()
    saved = []
    for owner, name, label in _COUNTED:
        real = getattr(owner, name)
        saved.append((owner, name, real))

        def counted(*a, _real=real, _label=label, **k):
            counts[_label] += 1
            return _real(*a, **k)

        setattr(owner, name, counted)
    try:
        yield counts
    finally:
        for owner, name, real in reversed(saved):
            setattr(owner, name, real)


N_EFFECTS = 10_000
BUDGET_CEILING_NS = 500_000  # design §7 (4): budget < 50 us per fenced effect; ceiling 500 us so a loaded box never flakes


def _bench(fence: LifecycleFence) -> tuple:
    with _counting() as counts:
        t0 = time.perf_counter_ns()
        for _ in range(N_EFFECTS):
            fence.check("a fenced effect")
        elapsed = time.perf_counter_ns() - t0
    return dict(counts), elapsed / N_EFFECTS


def test_fence_hot_path_budget(tmp_path: Path, capsys) -> None:
    """Design §7 (4): the check is an int compare in process. In the bumping process (the gateway in every operator
    Stop) it does ZERO I/O; in any other process exactly one stat per check and nothing else - no open, read, flock,
    subprocess or yaml parse (today's check parsed phases.yaml at every call: ~300 us)."""
    bumper_home = _make_home(tmp_path / "bumper")
    bump_lifecycle_generation(bumper_home, written_by="test:bench")  # this process bumped this home
    bumper_counts, bumper_ns = _bench(LifecycleFence.admit(bumper_home))
    assert sum(bumper_counts.values()) == 0, bumper_counts

    other_home = _make_home(tmp_path / "other")
    other_counts, other_ns = _bench(LifecycleFence.admit(other_home))
    assert other_counts.get("os.stat") == N_EFFECTS, other_counts
    assert sum(v for k, v in other_counts.items() if k != "os.stat") == 0, other_counts

    assert bumper_ns < BUDGET_CEILING_NS and other_ns < BUDGET_CEILING_NS, (bumper_ns, other_ns)
    with capsys.disabled():
        print(f"\n[fence hot path] bumper {bumper_ns / 1000:.3f} us/check, non-bumper {other_ns / 1000:.3f} us/check "
              f"(N={N_EFFECTS}, budget 50 us)")
