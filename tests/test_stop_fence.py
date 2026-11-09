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
    # the spawn reservation (Codex review 3 #6) and the fenced builders of the loop process
    "_reserve_loop_start", "_record_spawned_child", "_release_failed_reservation",
    "build_consolidator", "build_session_factory", "build_capability_llm_factory",
)
# The lane openers a fence enters through (open_home's fence is optional for readers; every runtime call passes it).
OPENERS = ("open_home", "open_entity_runtime")
# Modules whose unfenced calls are still to be fenced in a later step of this lane (chat.py is another worker's file
# until the lead frees it). The scan below fails when such a module is clean, so the exemption cannot outlive its fix.
PENDING_FENCE: set = set()  # chat.py was fenced in step 4 (its CLI visit and salvage pass the fence)


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
    from abstractruntime.identity.entity_runtime import open_entity_runtime

    runtime_fence = inspect.signature(open_entity_runtime).parameters["fence"]
    assert runtime_fence.kind is inspect.Parameter.KEYWORD_ONLY and runtime_fence.default is inspect.Parameter.empty
    should_stop = inspect.signature(life._sleep_authoring).parameters["should_stop"]
    assert should_stop.kind is inspect.Parameter.KEYWORD_ONLY and should_stop.default is inspect.Parameter.empty


def _unfenced_calls(path: Path) -> List[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if name in COMMIT_POINTS + OPENERS and not any(k.arg in ("fence", None) for k in node.keywords):
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


# ------------------------------------------------------------------ T7 / I3: one loop per home, the spawn reservation


class _FakeChild:
    """Popen's stand-in for the loop child: `pid` is a real live process (a sleeper) that never registers - the child
    that boots late. Every other Popen (ps, the sleeper itself) is the real one."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self):
        return None


def _stub_loop_popen(monkeypatch, sleeper_pid: int) -> List[List[str]]:
    launched: List[List[str]] = []
    real_popen = subprocess.Popen

    def popen(argv, *a, **k):
        if life.LOOP_MODULE_ARGV.split()[-1] in list(argv):
            launched.append(list(argv))
            return _FakeChild(sleeper_pid)
        return real_popen(argv, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", popen)
    return launched


def test_spawn_reservation_single_loop(tmp_path: Path, monkeypatch) -> None:
    """T7 (Codex review 3 #6): spawn released its lock after a fixed 0.6 s whatever the child had done; a child slow
    to register let a second start spawn a second loop over the home. Now the reservation (phase starting, the
    generation, the spawner and then the child pid) stands until the child replaces it: a second start - concurrent
    or after the first returned - is refused, one Popen; and a Stop between Popen and the child's boot makes the child
    refuse itself, writing nothing."""
    from abstractruntime.identity.life import read_loop_status, spawn_loop_process

    home_dir = _make_home(tmp_path)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        launched = _stub_loop_popen(monkeypatch, sleeper.pid)
        results: List[Any] = []

        def start() -> None:
            try:
                results.append(spawn_loop_process(home_dir, provider="lmstudio", model="m",
                                                  first_day_unconditional=True))
            except Exception as e:  # noqa: BLE001
                results.append(e)

        racers = [threading.Thread(target=start) for _ in range(2)]
        for t in racers:
            t.start()
        for t in racers:
            t.join(timeout=30)
        assert len(launched) == 1, launched
        assert sum(isinstance(r, dict) for r in results) == 1, results
        status = read_loop_status(home_dir)
        assert status["phase"] == "starting" and status["running"] is True, status
        assert status["pid"] == sleeper.pid and status["generation"] == 0, status
        # The first start returned; its child has not registered: a later start is refused too.
        with pytest.raises(RuntimeError, match="already starting"):
            spawn_loop_process(home_dir, provider="lmstudio", model="m")
        assert len(launched) == 1
        argv = launched[0]
        assert argv[argv.index("--generation") + 1] == "0"

        # The operator's Stop between Popen and the boot: the unverified child is not signalled (no token)...
        _stop_here(home_dir)
        assert sleeper.poll() is None
        before = _snapshot(home_dir)
        # ...and its own boot (the exact argv it was spawned with) refuses, writing nothing.
        code = life.main(argv[argv.index(life.LOOP_MODULE_ARGV.split()[-1]) + 1:])
        assert code == 0 and _snapshot(home_dir) == before
    finally:
        sleeper.kill()
        sleeper.wait()


def test_the_childs_boot_replaces_the_reservation_with_its_own_registration(tmp_path: Path, monkeypatch) -> None:
    """The success half of the reservation: a child booting at the reserved generation claims the home (its own
    pid in place of the reservation) and a kill switch that preceded the reservation is still known to it."""
    from abstractruntime.identity.life import read_loop_status, spawn_loop_process

    home_dir = _make_home(tmp_path)
    write_loop_status(home_dir, "stopped", stopped_by="kill_switch", owner=(None, None), fence=KILL_SWITCH)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        launched = _stub_loop_popen(monkeypatch, sleeper.pid)
        spawn_loop_process(home_dir, provider="lmstudio", model="m", return_to="personal")
        assert read_loop_status(home_dir)["after_stopped_by"] == "kill_switch"
        g = int(launched[0][launched[0].index("--generation") + 1])
        lines: List[str] = []

        def _never():  # noqa: ANN202
            raise AssertionError("after a kill switch nothing is restored: no day opens")

        report = LifeLoop(_never, tick_seconds=0, ticks_per_day=1, max_ticks=1, rest_minutes=0,
                          stop_file=home_dir / "STOP", state_home=home_dir, sleep_fn=lambda s: None,
                          out=lines.append, return_to="personal", fence=LifecycleFence(home_dir, g)).run()
    finally:
        sleeper.kill()
        sleeper.wait()
    assert report.stopped_by != "kill_switch", report
    assert any("nothing is restored" in ln for ln in lines), lines  # the kill switch before the reservation
    status = json.loads((home_dir / "loop_status").read_text(encoding="utf-8"))
    # The reservation is gone: the record is the loop's own (its pid, no spawner), never the sleeper's.
    assert status.get("pid") == os.getpid() and "spawner_pid" not in status and status["phase"] != "starting", status


# ------------------------------------------------------------------ T5: the night's completion is fenced


def test_sleep_completion_fenced(tmp_path: Path) -> None:
    """T5 (Codex review 3 #2; design class H): the loop's night checked the state word, never the generation, and
    passed memory no cancellation - a Stop during the final dream stage still recorded a completed night (resetting
    the idle-sleep boundary) and the repair step ran on. Now the night carries the loop's fence: the stage in flight
    ends, memory records no completion (cancelled_after set), the repair step and the night voice never run, and the
    night's only word about it is one clean line."""
    import warnings

    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.chat import open_home
    from abstractruntime.identity.life import build_consolidator

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    eid = home.entity_id
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key, kind, digest, kws in (
                ("s-1", "episode", "The connection pool saturated at noon.", ("pool", "saturation", "noon")),
                ("s-3", "episode", "Planted the garden sensors at noon.", ("garden", "sensors", "noon")),
                ("s-2", "lesson", "Batching the noon cron fixed it.", ("pool", "batching"))):
            home.ms.remember_many([MemoryRecordInput(kind=kind, title=f"Exchange {key}", digest=digest, keywords=kws)],
                                  scope="life", owner_id=eid, idempotency_key=key)
    home.close()
    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self",
                       fence=LifecycleFence.admit(home_dir))
    fence = LifecycleFence.admit(home_dir)
    calls: List[str] = []

    class _StopsWhileAuthoring:
        model_info = {"provider": "fake", "model": "summarizer"}

        def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
            calls.append(prompt)
            if len(calls) == 1:
                _stop_here(home_dir)  # the operator's Stop while this stage is being authored
            return response_model(title="A noon dream", summary="Noon tied the pool and the garden.")

    lines: List[str] = []
    result = build_consolidator(home_dir, out=lines.append, llm_for_capability=lambda name: _StopsWhileAuthoring(),
                                narrator_llm_factory=lambda: (_ for _ in ()).throw(AssertionError("no night voice")),
                                fence=fence)()
    assert len(calls) == 1, f"a capability call ran after the Stop: {len(calls)}"
    assert result["stopped_by"] == "kill_switch" and result["engine"].get("cancelled_after"), result
    home = open_home(home_dir)
    try:
        assert home.ms.tasks.last_maintenance([("self", eid), ("diary", eid), ("life", eid)]) is None, \
            "a stopped night recorded a completed maintenance"
    finally:
        home.close()
    assert result["authoring"] == {"skipped": "the operator's Stop"}, result["authoring"]
    # T12 / I5: clean - one line, no #FALLBACK, nothing called failed.
    assert [ln for ln in lines if "operator's Stop" in ln] == [
        "(the operator's Stop - the night ends here; nothing more is written)"], lines
    assert not [ln for ln in lines if "#FALLBACK" in ln or "fail" in ln.lower()], lines


# ------------------------------------------------------------------ T11 / T12: a real loop process, the kill refused


_FAKE_SITE = '''
import json, os, time
_dir = os.environ.get("FENCE_FAKE_LLM_DIR")
if _dir:
    import abstractcore, abstractmemory

    class _Resp:
        def __init__(self, content):
            self.content = content
            self.usage = {"total_tokens": 7}
            self.tool_calls = None

    class _FakeLLM:
        """THE MODEL STUB of the real child: the first call announces itself and waits for the test's Stop."""
        provider = "lmstudio"
        model = "fake-model"
        model_info = {"provider": "lmstudio", "model": "fake-model"}

        def generate(self, *args, **kwargs):
            calls = os.path.join(_dir, "calls.jsonl")
            n = len(open(calls).readlines()) if os.path.exists(calls) else 0
            with open(calls, "a") as fh:
                fh.write(json.dumps({"n": n, "pid": os.getpid()}) + "\\n")
            if n == 0:
                open(os.path.join(_dir, "called"), "w").close()
                deadline = time.monotonic() + 120
                while not os.path.exists(os.path.join(_dir, "go")) and time.monotonic() < deadline:
                    time.sleep(0.02)
            return _Resp("A thought of my own, finished after the Stop.\\nnext: keep going")

    abstractcore.create_llm = lambda provider, **kwargs: _FakeLLM()

    class _NoEmbedder:
        def __init__(self, *a, **k):
            raise RuntimeError("this test runs vectorless")

    abstractmemory.OpenAICompatTextEmbedder = _NoEmbedder
'''


def _wait_for(predicate, what: str, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def test_loop_process_refused_after_stop(tmp_path: Path, monkeypatch) -> None:
    """T11 (design class F): a REAL loop process (spawned through the real door, a stub model inside it) whose kill
    the Stop cannot deliver (the incarnation reads unknown). The Stop lands while its model call is in flight; the
    child sees the moved generation by itself, forms nothing, writes nothing under the home, and exits saying so
    once - no #FALLBACK, nothing failed (T12)."""
    from abstractruntime.identity.life import spawn_loop_process

    home_dir = _make_home(tmp_path)
    site = tmp_path / "fake-model-site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_FAKE_SITE, encoding="utf-8")
    talk = tmp_path / "model-talk"
    talk.mkdir()
    monkeypatch.setenv("FENCE_FAKE_LLM_DIR", str(talk))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(site), str(RUNTIME_SRC.parent)]))
    monkeypatch.delenv("ABSTRACTGATEWAY_AUTH_TOKEN", raising=False)
    # The Stop's kill is refused: the incarnation cannot be verified (EPERM, no token, a recycled pid...).
    monkeypatch.setattr(life, "_verified_loop_incarnation",
                        lambda pid, token, home: (False, "refused: the incarnation is unknown (test)"))
    started = spawn_loop_process(home_dir, provider="lmstudio", model="fake-model", tick_seconds=0.0,
                                 ticks_per_day=3, rest_minutes=30.0, first_day_unconditional=True)
    pid = int(started["pid"])
    try:
        _wait_for(lambda: (talk / "called").exists(), "the child's first model call")
        stop = hard_stop_loop(home_dir, reason="operator stop")
        assert stop["killed"] is False and stop["lifecycle_generation"] == 1, stop
        write_entity_state(home_dir, "paused", reason="stopped [by test]", fence=KILL_SWITCH)
        before = _snapshot(home_dir)
        before.pop(life.LOOP_LOG_FILENAME, None)
        (talk / "go").write_text("", encoding="utf-8")  # the in-flight call returns now, after the Stop
        _wait_for(lambda: life._pid_state(pid) == "gone", "the child to end by itself")
    finally:
        if life._pid_state(pid) != "gone":
            os.kill(pid, 9)
    after = _snapshot(home_dir)
    log = after.pop(life.LOOP_LOG_FILENAME).decode("utf-8", "replace").splitlines()
    changed = sorted(k for k in set(after) | set(before) if after.get(k) != before.get(k))
    assert not changed, f"the stopped loop wrote under the home: {changed}\n" + "\n".join(log[-20:])
    assert len((talk / "calls.jsonl").read_text().splitlines()) == 1, "a model call ran after the Stop"
    status = json.loads((home_dir / "loop_status").read_text(encoding="utf-8"))
    assert status["phase"] == "stopped" and status["stopped_by"] == "kill_switch", status
    assert _history(home_dir)[-1]["state"] == "paused"
    first = next(i for i, ln in enumerate(log) if "operator's Stop" in ln)
    tail = log[first:]
    assert not [ln for ln in tail if "#FALLBACK" in ln or "fail" in ln.lower() or "Traceback" in ln], "\n".join(log)
    assert sum("this loop ends here" in ln for ln in tail) == 1 and "stopped by kill_switch" in tail[-1], "\n".join(log)


def test_the_sleeps_repair_step_does_nothing_after_stop(tmp_path: Path) -> None:
    """`_sleep_authoring(should_stop=)`: declared, re-armed and authored nothing once the fence moved (the drainer
    takes the same check per task and after each answer, ab060bf)."""
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.chat import open_home

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    try:
        home.ms.remember_many([MemoryRecordInput(
            kind="lesson", title="untitled lesson [3fa2c1]", digest="Waiting lets the other finish.",
            attributes={"title_missing": True, "title_failure_reason": "missing"},
        )], scope="life", owner_id=home.entity_id, idempotency_key="legacy-1", turn_id="legacy-1")
        before = len(list(home.ms.tasks.tasks()))
        receipt = life._sleep_authoring(home, home.entity_id, [("life", home.entity_id)],
                                        lambda name: (_ for _ in ()).throw(AssertionError("no authoring call")),
                                        8, lambda s: None, should_stop=lambda: True)
        after = len(list(home.ms.tasks.tasks()))
    finally:
        home.close()
    assert receipt.get("cancelled") and receipt["declared"] == 0 and receipt["authored"] == 0, receipt
    assert after == before, "the repair step declared work after the Stop"


def test_an_engine_without_cancelled_still_ends_the_night_on_stop(tmp_path: Path, monkeypatch) -> None:
    """The version-skew rung: an engine that predates `cancelled=` gets the fence through `should_continue` - a Stop
    ends its night at the next stage."""
    import abstractmemory

    from abstractruntime.identity.life import build_consolidator

    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "asleep", reason="self-elected sleep: full", mode="dreaming", written_by="self",
                       fence=LifecycleFence.admit(home_dir))
    fence = LifecycleFence.admit(home_dir)
    seen: List[bool] = []

    def old_sleep_pass(system, *, scopes, owner_id, should_continue=None, include_dream=True, author=None):
        seen.append(should_continue())
        hard_stop_loop(home_dir, reason="operator stop")  # the runtime's own Stop: it writes no state word
        assert read_entity_state(home_dir)["state"] == "asleep"  # only the generation says a Stop happened
        seen.append(should_continue())
        return {"maintenance": {}, "dream": {}, "cancelled_after": "consolidation"}

    monkeypatch.setattr(abstractmemory, "sleep_pass", old_sleep_pass)
    result = build_consolidator(home_dir, out=lambda s: None, fence=fence)()
    assert seen == [True, False], seen
    assert result["stopped_by"] == "kill_switch"


def test_the_spawner_never_records_its_child_over_a_registration(tmp_path: Path) -> None:
    """The child may register before Popen returns to the spawner: the spawner's pid write then leaves the child's
    own record alone (and never revives a reservation over it)."""
    home_dir = _make_home(tmp_path)
    fence = LifecycleFence.admit(home_dir)
    life._reserve_loop_start(home_dir, fence=fence)
    write_loop_status(home_dir, "between", fence=fence)  # the child's boot claim, in-process
    claimed = json.loads((home_dir / "loop_status").read_text(encoding="utf-8"))
    assert life._record_spawned_child(home_dir, 4_000_000, fence=fence) is False
    assert json.loads((home_dir / "loop_status").read_text(encoding="utf-8")) == claimed
    life._release_failed_reservation(home_dir, fence=fence)  # nor does a failed-start release touch it
    assert json.loads((home_dir / "loop_status").read_text(encoding="utf-8")) == claimed


# ------------------------------------------------------------------ step 3: the home opened with its fence


def _fenced_home(home_dir: Path):
    from abstractruntime.identity.chat import open_home

    fence = LifecycleFence.admit(home_dir)
    return open_home(home_dir, fence=fence), fence


def _artifact_files(home_dir: Path) -> List[str]:
    return sorted(str(p.relative_to(home_dir)) for p in (home_dir / "artifacts").rglob("*") if p.is_file()
                  and not p.name.startswith("artifact_catalog"))


def test_stop_fence_formation_authoring(tmp_path: Path) -> None:
    """T3 (Codex review 3 #1; design class B): `_effect(MEMORY_FORM)` authorized neither its authoring call nor the
    write - a Stop while the summaries model answered still stored the call as an artifact and formed the record (or
    formed it pending with a failure line). On a fenced home the authoring's artifact is refused, the write never
    runs, no failure line, and the turn sees the operator's Stop."""
    from abstractruntime.core.models import EffectType
    from abstractruntime.identity.chat import ChatSession

    home_dir = _make_home(tmp_path)
    home, fence = _fenced_home(home_dir)
    calls: List[str] = []

    class _StopsWhileAnswering:
        model_info = {"provider": "fake", "model": "summarizer"}

        def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
            calls.append(prompt)
            _stop_here(home_dir)  # the operator's Stop lands while the summaries model answers
            return response_model(title="An authored title", summary="An authored summary.")

    s = ChatSession(home, _HookLLM([]), participants=["person:laurent"], context_window=20000, out=lambda _s: None,
                    phase="visit", model_info={"provider": "lmstudio", "model": "qwen-test"})
    s.llm_for_capability = lambda name: _StopsWhileAnswering()
    inner = home.handlers[EffectType.MEMORY_FORM].__lifecycle_inner__  # the table is the fenced one
    ran: List[Any] = []
    home.handlers.update({EffectType.MEMORY_FORM: lf.fence_handlers(
        {EffectType.MEMORY_FORM: lambda run, effect, nxt=None: ran.append(effect) or inner(run, effect, nxt)},
        fence)[EffectType.MEMORY_FORM]})
    before_rows = _snapshot(home_dir).get("memory.sqlite3")
    before_artifacts = _artifact_files(home_dir)
    payload = {"records": [{"kind": "lesson", "title": None, "pending": ["title", "summary"],
                            "_authoring": {"body": "Waiting lets the other finish.", "raw_block": "Waiting.",
                                           "template_id": "test.t3"},
                            "digest": "Waiting lets the other finish.", "provenance": {"source": "test"}}],
               "scope": "life", "owner_id": home.entity_id, "turn_id": "t3"}
    try:
        with pytest.raises(StoppedByOperator):
            s._effect(EffectType.MEMORY_FORM, payload)
    finally:
        home.close()
    assert len(calls) == 1 and ran == [], "the write ran after the Stop"
    assert _artifact_files(home_dir) == before_artifacts, "the authoring call was kept as an artifact after the Stop"
    assert _snapshot(home_dir).get("memory.sqlite3") == before_rows
    assert not (tmp_path / "entities" / "untracked" / "authoring-failures").exists()


def test_a_fenced_home_refuses_every_write_after_stop(tmp_path: Path) -> None:
    """The home's three stores carry the fence: memory's commit guard (installed by open_home, rolled back inside the
    transaction), the diary book and the artifact store - after a Stop each raises the operator's Stop and nothing
    lands."""
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.diary import DiaryEntry

    home_dir = _make_home(tmp_path)
    home, fence = _fenced_home(home_dir)
    try:
        assert home.store._commit_guard == fence.check and home.journal._commit_guard == fence.check
        _stop_here(home_dir)
        before = _snapshot(home_dir)
        with pytest.raises(StoppedByOperator):
            home.ms.remember_many([MemoryRecordInput(kind="lesson", title="Late", digest="A late lesson.")],
                                  scope="life", owner_id=home.entity_id, idempotency_key="late-1")
        with pytest.raises(StoppedByOperator):
            home.diary.append_entry(DiaryEntry(entry_id="diary_late01", author=home.entity_id, kind="note",
                                               gist="late", text="Written after the Stop.", visibility="self",
                                               written_at="2026-11-09T21:00:00+00:00"))
        with pytest.raises(StoppedByOperator):
            home.artifacts.store(b"late bytes", content_type="text/plain")
    finally:
        home.close()
    after = _snapshot(home_dir)
    changed = sorted(k for k in set(after) | set(before) if after.get(k) != before.get(k))
    assert not changed, changed


def test_the_runtime_lane_ends_cancelled_never_failed(tmp_path: Path) -> None:
    """The Runtime lane (a visit workflow, MEMORY_CONSOLIDATE...): the fenced table abandons a model reply that
    returned after the Stop - cancelled (KILL_SWITCH_REASON), never failed, never retried - and refuses every
    effect after it without calling its handler."""
    from abstractruntime.core.models import Effect, EffectType, StepPlan
    from abstractruntime.core.spec import WorkflowSpec
    from abstractruntime.identity.entity_runtime import open_entity_runtime

    home_dir = _make_home(tmp_path)
    fence = LifecycleFence.admit(home_dir)
    llm_calls: List[Any] = []

    def llm_handler(run, effect, default_next_node=None):
        from abstractruntime.core.runtime import EffectOutcome

        llm_calls.append(effect)
        _stop_here(home_dir)  # the operator's Stop while the model answers
        return EffectOutcome.completed({"content": "a reply that returned after the Stop"})

    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm_handler}, fence=fence)
    try:
        wf = WorkflowSpec(workflow_id="fenced", entry_node="a", nodes={
            "a": lambda r, c: StepPlan(node_id="a", effect=Effect(type=EffectType.LLM_CALL,
                                                                    payload={"prompt": "hi"}, result_key="ans"),
                                       next_node="b"),
            "b": lambda r, c: StepPlan(node_id="b", complete_output={"a": r.vars.get("ans")}),
        })
        run_id = ert.runtime.start(workflow=wf, vars={})
        state = ert.runtime.tick(workflow=wf, run_id=run_id)
        records = list(ert.ledger_store.list(run_id))
        inner_form = ert.runtime._handlers[EffectType.MEMORY_FORM]
        late = inner_form(None, Effect(type=EffectType.MEMORY_FORM, payload={"records": []}), None)
    finally:
        ert.home.close()
    assert len(llm_calls) == 1, "the model call was retried after the Stop"
    assert state.status.value == "cancelled" and lf.KILL_SWITCH_REASON in str(state.error), (state.status, state.error)
    assert not [r for r in records if str(r.get("status") or "").lower() == "failed"], records
    assert lf.is_stopped_outcome(late)


def test_fence_hot_path_budget_through_the_handler_table_and_the_commit_guard(tmp_path: Path, capsys) -> None:
    """Design §7 (4), the effect half: N no-op effects through the wrapped table and N commit-guard calls (the memory
    guard is `fence.check`). Bumper: zero I/O. Non-bumper: one stat per fenced call, nothing else."""
    from abstractruntime.core.models import Effect, EffectType
    from abstractruntime.core.runtime import EffectOutcome

    done = EffectOutcome.completed({})

    def bench(home_dir: Path) -> tuple:
        fence = LifecycleFence.admit(home_dir)
        table = lf.fence_handlers({EffectType.MEMORY_QUERY: lambda run, effect, nxt=None: done}, fence)
        handler, guard = table[EffectType.MEMORY_QUERY], fence.check
        effect = Effect(type=EffectType.MEMORY_QUERY, payload={})
        with _counting() as counts:
            t0 = time.perf_counter_ns()
            for _ in range(N_EFFECTS):
                handler(None, effect, None)
                guard("SQLiteJournal._txn")
            elapsed = time.perf_counter_ns() - t0
        return dict(counts), elapsed / N_EFFECTS

    bumper = _make_home(tmp_path / "bumper")
    bump_lifecycle_generation(bumper, written_by="test:bench")
    bumper_counts, bumper_ns = bench(bumper)
    other_counts, other_ns = bench(_make_home(tmp_path / "other"))
    assert sum(bumper_counts.values()) == 0, bumper_counts
    assert other_counts.get("os.stat") == 2 * N_EFFECTS, other_counts
    assert sum(v for k, v in other_counts.items() if k != "os.stat") == 0, other_counts
    assert bumper_ns < BUDGET_CEILING_NS and other_ns < BUDGET_CEILING_NS, (bumper_ns, other_ns)
    with capsys.disabled():
        print(f"\n[fence effect + commit guard] bumper {bumper_ns / 1000:.3f} us, non-bumper {other_ns / 1000:.3f} us "
              f"per (effect + guard) pair (N={N_EFFECTS}, budget 50 us per fenced effect)")


def test_a_stop_inside_chat_effects_is_the_operators_stop_never_a_failure(tmp_path: Path) -> None:
    """ChatSession's two halves of the effect fence: the fenced table's refusal raises the operator's Stop (never
    "<effect> failed", which ended the loop's tick as a failure), and a Stop raised inside formation authoring (a
    fenced artifact, a fenced heartbeat) propagates instead of forming the record pending with a failure line."""
    from abstractruntime.core.models import EffectType
    from abstractruntime.identity.chat import ChatSession

    home_dir = _make_home(tmp_path)
    home, fence = _fenced_home(home_dir)
    s = ChatSession(home, _HookLLM([]), participants=["person:laurent"], context_window=20000, out=lambda _s: None,
                    phase="visit", model_info={"provider": "lmstudio", "model": "qwen-test"})

    class _RefusedMidCall:
        model_info = {"provider": "fake", "model": "summarizer"}

        def generate(self, **kw: Any) -> Any:
            raise StoppedByOperator("before a model call (a fenced heartbeat)")

    s.llm_for_capability = lambda name: _RefusedMidCall()
    payload = {"records": [{"kind": "lesson", "title": None, "pending": ["title"],
                            "_authoring": {"body": "b", "raw_block": "b", "template_id": "t"}, "digest": "d"}],
               "scope": "life", "owner_id": home.entity_id, "turn_id": "t"}
    try:
        with pytest.raises(StoppedByOperator):
            s._author_records(payload)
        _stop_here(home_dir)
        with pytest.raises(StoppedByOperator):
            s._effect(EffectType.MEMORY_FORM, {"records": [], "scope": "life", "owner_id": home.entity_id,
                                               "turn_id": "t"})
    finally:
        home.close()


# ------------------------------------------------------------------ step 4: ChatSession carries the fence


def _session_over(home_dir: Path, llm: Any):
    from abstractruntime.identity.chat import ChatSession

    home, fence = _fenced_home(home_dir)
    s = ChatSession(home, llm, participants=["person:laurent"], context_window=20000, out=lambda _s: None,
                    phase="visit", model_info={"provider": "lmstudio", "model": "qwen-test"})
    assert s.fence is fence  # the session takes the home's fence
    return home, s, fence


def test_stop_fence_retry_model_call(tmp_path: Path) -> None:
    """T1 (Codex review 3 #1; design class A): `_generate` checked cancellation once, before its harmony retry loop -
    a Stop during attempt 1 still sent attempt 2 to the model. Now every attempt is checked: the first fails with the
    known harmony 400 while the Stop lands, the second never runs."""
    from abstractruntime.identity.chat import ChatSession

    home_dir = _make_home(tmp_path)
    calls: List[int] = []

    class _HarmonyRace:
        def generate(self, *, messages, system_prompt, **kw):
            calls.append(len(calls))
            _stop_here(home_dir)  # the operator's Stop during attempt 1
            raise RuntimeError(f"400: {ChatSession._HARMONY_HEADER_400}")

    home, s, _fence = _session_over(home_dir, _HarmonyRace())
    try:
        with pytest.raises(StoppedByOperator) as stopped:
            s.turn("hello")
    finally:
        home.close()
    assert calls == [0], f"a model attempt ran after the Stop: {calls}"
    assert "attempt 2" in stopped.value.where or "answering" in stopped.value.where, stopped.value.where


def test_stop_fence_abandons_late_reply(tmp_path: Path) -> None:
    """T2 (design classes A/B/C): the model's reply returns AFTER the Stop: it is abandoned - the turn raises the
    operator's Stop, nothing forms (no MEMORY_FORM), the book takes nothing, the spend does not count it, and no
    further model call runs."""
    home_dir = _make_home(tmp_path)
    at_stop: Dict[str, Any] = {}

    def _stop() -> None:
        _stop_here(home_dir)
        at_stop["snapshot"] = _snapshot(home_dir)

    llm = _HookLLM(["A reply finished after the Stop.\n```diary kind=note\ngist: late\nLate words.\n```",
                    "NEVER - no second call"], hooks={0: _stop})
    home, s, _fence = _session_over(home_dir, llm)
    spend_before = dict(s.spend)
    try:
        with pytest.raises(StoppedByOperator):
            s.turn("hello")
        spend_after = dict(s.spend)
    finally:
        home.close()
    assert llm.n == 1 and llm.replies == ["NEVER - no second call"], (llm.n, llm.replies)
    assert spend_after == spend_before, "the abandoned reply was counted as spend"
    after = _snapshot(home_dir)
    changed = sorted(k for k in set(after) | set(at_stop["snapshot"])
                     if after.get(k) != at_stop["snapshot"].get(k))
    assert not changed, f"the late reply landed: {changed}"


def test_the_bump_stops_the_decode_in_flight_through_cancel_event(tmp_path: Path) -> None:
    """§2.3: in the process that bumps (the gateway's Stop), a client that honours `cancel_event=` sees it set while
    it is still decoding - the provider stops within a token; the reply is abandoned anyway."""
    home_dir = _make_home(tmp_path)
    seen: Dict[str, Any] = {}

    class _Cancellable:
        def generate(self, *, messages, system_prompt, cancel_event=None, **kw):
            seen["before"] = cancel_event.is_set()
            _stop_here(home_dir)
            seen["after"] = cancel_event.is_set()
            return type("R", (), {"content": "partial", "usage": None})()

    home, s, _fence = _session_over(home_dir, _Cancellable())
    try:
        with pytest.raises(StoppedByOperator):
            s.turn("hello")
    finally:
        home.close()
    assert seen == {"before": False, "after": True}, seen
    assert not lf._INFLIGHT, "an in-flight cancel event outlived its call"


def test_the_host_set_cancel_check_is_gone_loudly(tmp_path: Path) -> None:
    """The fence replaces `cancel_check`: a host that still sets it fails at once (silently ignoring it would leave
    its turns unfenced)."""
    home_dir = _make_home(tmp_path)
    home, s, _fence = _session_over(home_dir, _HookLLM([]))
    try:
        with pytest.raises(AttributeError, match="LifecycleFence"):
            s.cancel_check = lambda: True
    finally:
        home.close()


def test_the_cli_visit_restores_never_land_after_stop(tmp_path: Path) -> None:
    """T6 / Codex review 3 #5: the CLI visit's timeout restore wrote awake unconditionally and its close restore
    checked then wrote. Both go through one locked, fenced step now: over a Stop's paused nothing is written; over
    its own posture (no Stop) it wakes him."""
    from abstractruntime.identity.chat import _restore_own_yield

    home_dir = _make_home(tmp_path)
    write_entity_state(home_dir, "asleep", reason="in conversation with person:laurent (auto-yield)",
                       mode="visiting", written_by="visit-door", fence=LifecycleFence.admit(home_dir))
    visit = LifecycleFence.admit(home_dir)
    _stop_here(home_dir)
    before = (home_dir / "state").read_bytes(), (home_dir / "state_history.jsonl").read_bytes()
    assert _restore_own_yield(home_dir, visit, "visitor session ended (auto-yield return)") is False
    assert ((home_dir / "state").read_bytes(), (home_dir / "state_history.jsonl").read_bytes()) == before
    assert read_entity_state(home_dir)["state"] == "paused"
    # No Stop: the visit's own posture is restored.
    other = _make_home(tmp_path / "other")
    write_entity_state(other, "asleep", reason="in conversation with person:laurent (auto-yield)",
                       mode="visiting", written_by="visit-door", fence=LifecycleFence.admit(other))
    assert _restore_own_yield(other, LifecycleFence.admit(other), "visitor session ended") is True
    assert read_entity_state(other)["state"] == "awake"


def test_a_refused_book_write_is_the_stop_and_rescues_nothing(tmp_path: Path) -> None:
    """Codex review 3 #1 (durable diary capture): the visit lane's capture rescued the raw reply into
    <home>/rescue/ whenever the book refused - after a Stop that is a write. A fenced refusal is the Stop now:
    nothing written, no rescue."""
    from abstractruntime.core.models import EffectType
    from abstractruntime.identity.act_only import capture_diary_elections

    home_dir = _make_home(tmp_path)
    home, fence = _fenced_home(home_dir)
    try:
        _stop_here(home_dir)
        before = _snapshot(home_dir)
        with pytest.raises(StoppedByOperator):
            capture_diary_elections("Hello.\n```diary kind=note\ngist: a note\nMy words.\n```", run=None,
                                    turn_id="t-1", diary_write_handler=home.handlers[EffectType.DIARY_WRITE],
                                    rescue_dir=home_dir)
    finally:
        home.close()
    assert not (home_dir / "rescue").exists(), "the reply was rescued into the home after the Stop"
    after = _snapshot(home_dir)
    assert sorted(k for k in set(after) | set(before) if after.get(k) != before.get(k)) == []


def test_the_visit_lanes_diary_authoring_call_never_runs_after_stop(tmp_path: Path) -> None:
    """The visit lane authors a public diary entry's metadata with the host's summaries handler before the book
    write: after a Stop that model call does not run, and the run ends CANCELLED."""
    from abstractruntime.core.models import Effect, EffectType, StepPlan
    from abstractruntime.core.runtime import EffectOutcome
    from abstractruntime.core.spec import WorkflowSpec
    from abstractruntime.identity.entity_runtime import open_entity_runtime

    home_dir = _make_home(tmp_path)
    summaries_calls: List[Any] = []

    def mind(run, effect, default_next_node=None):
        _stop_here(home_dir)  # the Stop lands while the mind answers
        return EffectOutcome.completed({"content": "Here.\n```diary kind=note\ngist: a note\nMy words.\n```"})

    def summaries(run, effect, default_next_node=None):
        summaries_calls.append(effect)
        return EffectOutcome.completed({"content": json.dumps({"title": "T", "summary": "S"})})

    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: mind},
                              capability_handlers={"summaries": summaries}, fence=LifecycleFence.admit(home_dir))
    try:
        wf = WorkflowSpec(workflow_id="visit-ish", entry_node="a", nodes={
            "a": lambda r, c: StepPlan(node_id="a", effect=Effect(type=EffectType.LLM_CALL,
                                                                    payload={"prompt": "hi", "turn_id": "t-1"},
                                                                    result_key="ans"), next_node="b"),
            "b": lambda r, c: StepPlan(node_id="b", complete_output={"a": r.vars.get("ans")}),
        })
        state = ert.runtime.tick(workflow=wf, run_id=ert.runtime.start(workflow=wf, vars={}))
    finally:
        ert.home.close()
    assert summaries_calls == [], "the summaries model was called after the Stop"
    assert state.status.value == "cancelled" and lf.KILL_SWITCH_REASON in str(state.error), (state.status, state.error)
    assert not (home_dir / "rescue").exists()


def test_a_stop_between_two_model_attempts_stops_the_retry(tmp_path: Path) -> None:
    """T1's narrow case: attempt 1 fails with the harmony 400 BEFORE the Stop, the Stop lands while the failure is
    being classified - the retry is the next model call, and it is checked."""
    from abstractruntime.identity.chat import ChatSession

    home_dir = _make_home(tmp_path)
    calls: List[int] = []

    class _Harmony400(RuntimeError):
        stopped = False

        def __str__(self) -> str:
            if not _Harmony400.stopped:
                _Harmony400.stopped = True
                _stop_here(home_dir)  # the Stop lands between attempt 1 and attempt 2
            return f"400: {ChatSession._HARMONY_HEADER_400}"

    class _HarmonyRace:
        def generate(self, *, messages, system_prompt, **kw):
            calls.append(len(calls))
            raise _Harmony400()

    home, s, _fence = _session_over(home_dir, _HarmonyRace())
    try:
        with pytest.raises(StoppedByOperator) as stopped:
            s.turn("hello")
    finally:
        home.close()
    assert calls == [0] and stopped.value.where == "before model attempt 2", (calls, stopped.value.where)


def test_an_effect_after_stop_never_calls_its_authoring_model(tmp_path: Path) -> None:
    """`_effect` checks the fence BEFORE its authoring call: no summaries model call after a Stop."""
    from abstractruntime.core.models import EffectType

    home_dir = _make_home(tmp_path)
    home, s, _fence = _session_over(home_dir, _HookLLM([]))
    calls: List[str] = []
    s.llm_for_capability = lambda name: calls.append(name) or (_ for _ in ()).throw(AssertionError("called"))
    payload = {"records": [{"kind": "lesson", "title": None, "pending": ["title"],
                            "_authoring": {"body": "b", "raw_block": "b", "template_id": "t"}, "digest": "d"}],
               "scope": "life", "owner_id": home.entity_id, "turn_id": "t"}
    try:
        _stop_here(home_dir)
        with pytest.raises(StoppedByOperator):
            s._effect(EffectType.MEMORY_FORM, payload)
    finally:
        home.close()
    assert calls == [], "the authoring model was asked for after the Stop"


def test_a_late_authoring_reply_is_abandoned_even_over_an_unguarded_home(tmp_path: Path) -> None:
    """The session's own check after the authoring reply (a harness home opened without a fence, the session given
    one): the late reply is not kept as an artifact and the write never runs."""
    from abstractruntime.core.models import EffectType
    from abstractruntime.identity.chat import ChatSession, open_home

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)  # unguarded stores
    s = ChatSession(home, _HookLLM([]), participants=["person:laurent"], context_window=20000, out=lambda _s: None,
                    phase="visit", model_info={"provider": "lmstudio", "model": "qwen-test"})
    s.fence = LifecycleFence.admit(home_dir)

    class _StopsWhileAnswering:
        model_info = {"provider": "fake", "model": "summarizer"}

        def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
            _stop_here(home_dir)
            return response_model(title="An authored title", summary="An authored summary.")

    s.llm_for_capability = lambda name: _StopsWhileAnswering()
    payload = {"records": [{"kind": "lesson", "title": None, "pending": ["title", "summary"],
                            "_authoring": {"body": "b", "raw_block": "b", "template_id": "t"}, "digest": "d"}],
               "scope": "life", "owner_id": home.entity_id, "turn_id": "t"}
    before = _artifact_files(home_dir)
    try:
        with pytest.raises(StoppedByOperator):
            s._effect(EffectType.MEMORY_FORM, payload)
    finally:
        home.close()
    assert _artifact_files(home_dir) == before, "the late authoring reply was kept"
