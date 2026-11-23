"""`utils.os_process`: one contract over two operating systems (2026-11-22).

The own-time loop keeps a pid on disk and decides about the process behind it. On Windows that
crashed at the first read (`os.WNOHANG` does not exist there), the liveness probe was a console
control event, the incarnation token and the command line came from a `ps` that is not installed,
and the pid recorded for a spawned loop was a launcher's. Every test here runs on every system: real
processes, no mocks of the system calls."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List

import pytest

import abstractruntime.identity.life as life
from abstractruntime.utils import os_process

WINDOWS = sys.platform == "win32"
# POSIX reads the token, the zombie state and the command line from `ps` (a slim container may not have it).
pytestmark = pytest.mark.skipif(not WINDOWS and shutil.which("ps") is None, reason="needs ps")
SLEEPER = "import time; time.sleep(60)"
NOBODY = 2**31 - 4  # above every system's pid range in use: no process has it


def _wait_for(done: Callable[[], bool], what: str, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if done():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def sleeper():
    """Real children that sleep; whatever a test leaves alive is killed, and every child collected."""
    started: List[subprocess.Popen] = []

    def start(*extra: str) -> subprocess.Popen:
        python, env = os_process.direct_python()
        proc = subprocess.Popen([python, "-c", SLEEPER, *extra], env=env, stdin=subprocess.DEVNULL)
        started.append(proc)
        _wait_for(lambda: os_process.start_token(proc.pid) is not None, "the child's start token")
        return proc

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


# ------------------------------------------------------------------ this process, and no process


def test_this_process_is_alive_and_has_a_stable_token() -> None:
    me = os.getpid()
    assert os_process.exists(me) is True
    assert os_process.state(me) == "alive"
    token = os_process.start_token(me)
    assert token and token == os_process.start_token(me)
    assert token.startswith(os_process.WINDOWS_TOKEN_PREFIX) is WINDOWS


@pytest.mark.parametrize("pid", [None, "", "not-a-pid", object(), float("inf"), float("nan")])
def test_a_pid_that_is_not_a_number_is_never_alive_and_never_dead(pid: object) -> None:
    assert os_process.exists(pid) is False
    assert os_process.state(pid) == "unknown"  # only "gone" confirms a death
    assert os_process.start_token(pid) is None
    assert os_process.command(pid) is None
    assert os_process.reap(pid) is False


def test_a_pid_nobody_holds_is_gone() -> None:
    assert os_process.state(NOBODY) == "gone"
    assert os_process.exists(NOBODY) is False
    assert os_process.start_token(NOBODY) is None
    assert os_process.command(NOBODY) is None
    with pytest.raises(ProcessLookupError):
        os_process.terminate(NOBODY)


# ------------------------------------------------------------------ another process


def test_a_live_child_is_alive_with_its_own_token(sleeper) -> None:
    proc = sleeper()
    assert os_process.state(proc.pid) == "alive" and os_process.exists(proc.pid) is True
    token = os_process.start_token(proc.pid)
    assert token and token == os_process.start_token(proc.pid)
    assert os_process.reap(proc.pid) is False  # it has not exited: nothing to collect


def test_two_incarnations_never_share_a_token(sleeper) -> None:
    first, second = sleeper(), sleeper()
    tokens = {os_process.start_token(first.pid), os_process.start_token(second.pid), os_process.start_token(os.getpid())}
    if WINDOWS:
        assert len(tokens) == 3  # 100 ns creation times
    else:
        assert None not in tokens  # ps prints whole seconds: equal stamps are possible, missing ones are not


def test_an_exited_child_still_held_by_its_parent_is_gone(sleeper) -> None:
    """The corpse: POSIX keeps a zombie that answers signal 0 until its parent collects it; Windows
    keeps the process object (and its creation time) while a handle is open. Neither is alive."""
    proc = sleeper()
    name = os_process.terminate(proc.pid, token=os_process.start_token(proc.pid))
    assert name == ("TerminateProcess" if WINDOWS else "SIGTERM")
    _wait_for(lambda: os_process.state(proc.pid) == "gone", "the terminated child to read gone")
    assert proc.returncode is None  # this test never collected it through Popen: the corpse is held
    # The loop's own read: collect it when it is ours (a POSIX zombie still answers signal 0), then probe.
    assert life._pid_alive_reaped(proc.pid) is False
    assert os_process.exists(proc.pid) is False
    with pytest.raises(ProcessLookupError):
        os_process.terminate(proc.pid)


def test_force_ends_a_child(sleeper) -> None:
    proc = sleeper()
    assert os_process.terminate(proc.pid, force=True) == ("TerminateProcess" if WINDOWS else "SIGKILL")
    _wait_for(lambda: os_process.state(proc.pid) == "gone", "the killed child to read gone")


@pytest.mark.skipif(not WINDOWS, reason="POSIX has no handle to check the token on; the caller's verification stands")
def test_windows_terminate_refuses_another_incarnation(sleeper) -> None:
    proc = sleeper()
    with pytest.raises(ProcessLookupError):
        os_process.terminate(proc.pid, token=os_process.WINDOWS_TOKEN_PREFIX + "1")
    assert os_process.state(proc.pid) == "alive"


def test_the_command_of_a_child_carries_its_arguments(sleeper) -> None:
    proc = sleeper("--home", "plain-home", "tail")
    command = os_process.command(proc.pid)
    if WINDOWS:
        assert isinstance(command, list) and command[1:] == ["-c", SLEEPER, "--home", "plain-home", "tail"]
    else:
        assert isinstance(command, str) and command.endswith("--home plain-home tail")


@pytest.mark.skipif(not WINDOWS, reason="ps joins arguments with spaces: only Windows gives the exact list back")
def test_windows_arguments_survive_spaces_quotes_and_backslashes(sleeper) -> None:
    awkward = ["C:\\Users\\Some One\\entities\\gan dalf\\", 'say "hi"', "", "a\\\\b"]
    proc = sleeper(*awkward)
    assert os_process.command(proc.pid)[3:] == awkward


def test_reap_collects_our_exited_child_once(sleeper) -> None:
    proc = sleeper()
    proc.kill()
    if WINDOWS:
        _wait_for(lambda: os_process.state(proc.pid) == "gone", "the child to exit")
        assert os_process.reap(proc.pid) is False  # nothing to collect on Windows
        return
    _wait_for(lambda: os_process.reap(proc.pid), "the zombie to be collected")
    assert os_process.reap(proc.pid) is False
    assert os_process.state(proc.pid) == "gone"


@pytest.mark.parametrize("pid", [0, -1, -4])
def test_terminate_never_takes_a_group_for_a_process(pid: int) -> None:
    """POSIX reads 0 and negative numbers as process groups (0: the caller's own)."""
    with pytest.raises(ProcessLookupError):
        os_process.terminate(pid)
    with pytest.raises(ProcessLookupError):
        os_process.terminate(pid, force=True)


@pytest.mark.skipif(not WINDOWS, reason="a Windows quirk: OpenProcess ignores the two low bits of the id")
def test_windows_a_number_next_to_a_pid_is_not_that_process(sleeper) -> None:
    proc = sleeper()
    token = os_process.start_token(proc.pid)
    for other in (proc.pid + 1, proc.pid + 3, proc.pid + 2**32):
        assert os_process.state(other) == "gone" and os_process.exists(other) is False
        assert os_process.start_token(other) is None and os_process.command(other) is None
        with pytest.raises(ProcessLookupError):
            os_process.terminate(other, token=token)
    assert os_process.state(proc.pid) == "alive"


def test_a_detached_child_starts_and_is_ours_to_end() -> None:
    python, env = os_process.direct_python()
    proc = subprocess.Popen([python, "-c", SLEEPER], env=env, stdin=subprocess.DEVNULL, **os_process.detached())
    try:
        _wait_for(lambda: os_process.state(proc.pid) == "alive", "the detached child")
        os_process.terminate(proc.pid, force=True)
        _wait_for(lambda: os_process.state(proc.pid) == "gone", "the detached child to end")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


# ------------------------------------------------------------------ a child whose pid is the one Popen returns


def test_direct_python_child_has_popens_pid_and_this_environment(tmp_path: Path) -> None:
    """A Windows virtual environment's `sys.executable` is a launcher: its child is the interpreter,
    so the pid `Popen` returned was never the pid the child saw. `direct_python` starts the
    interpreter itself and the child still runs in this environment."""
    out = tmp_path / "child.json"
    code = ("import json, os, sys; "
            "json.dump({'pid': os.getpid(), 'ppid': os.getppid(), 'prefix': sys.prefix, 'executable': sys.executable}, "
            "open(sys.argv[1], 'w'))")
    python, env = os_process.direct_python()
    proc = subprocess.Popen([python, "-c", code, str(out)], env=env)
    assert proc.wait(timeout=30) == 0
    child = json.loads(out.read_text(encoding="utf-8"))
    assert child["pid"] == proc.pid and child["ppid"] == os.getpid()
    assert os.path.normcase(child["prefix"]) == os.path.normcase(sys.prefix)
    assert os.path.normcase(child["executable"]) == os.path.normcase(sys.executable)


def test_direct_python_keeps_the_callers_environment_and_does_not_touch_it() -> None:
    given = {"ONLY": "this"}
    _python, env = os_process.direct_python(given)
    assert env["ONLY"] == "this" and given == {"ONLY": "this"}
    assert set(env) - {"ONLY"} <= {"__PYVENV_LAUNCHER__"}


# ------------------------------------------------------------------ the own-time loop's readers


def _write_status(home: Path, **fields: object) -> None:
    (home / "loop_status").write_text(json.dumps({"updated_at": "2026-11-22T01:29:20+00:00", **fields}),
                                      encoding="utf-8")


def test_a_reservation_whose_child_died_reads_not_running_and_never_raises(tmp_path: Path, sleeper) -> None:
    """The incident (Windows, 2026-11-22): the loop child died at its boot, the status stayed
    `starting` with its pid, and EVERY later read raised, so own time could never be started again."""
    proc = sleeper()
    token = os_process.start_token(proc.pid)
    proc.kill()
    proc.wait(timeout=10)
    _write_status(tmp_path, phase="starting", generation=0, spawner_pid=os.getpid(), pid=proc.pid, pid_started_at=token)
    status = life.read_loop_status(tmp_path)
    assert status["phase"] == "starting" and status["running"] is False


def test_a_reservation_whose_child_lives_reads_running_and_a_reused_pid_does_not(tmp_path: Path, sleeper) -> None:
    proc = sleeper()
    token = os_process.start_token(proc.pid)
    _write_status(tmp_path, phase="starting", generation=0, spawner_pid=os.getpid(), pid=proc.pid, pid_started_at=token)
    assert life.read_loop_status(tmp_path)["running"] is True
    _write_status(tmp_path, phase="starting", generation=0, spawner_pid=os.getpid(), pid=proc.pid,
                  pid_started_at="the start time of an earlier holder of this pid")
    assert life.read_loop_status(tmp_path)["running"] is False


def test_a_process_that_only_mentions_the_loop_module_is_not_a_loop(tmp_path: Path, sleeper) -> None:
    """Stop signals a process only when it is this home's loop BY ITS OWN ARGUMENTS, read from the
    system: the module directly after the interpreter. `python -c <code> -m abstractruntime...` is
    some other program that carries the words."""
    home = tmp_path / "entities with space" / "gandalf"
    home.mkdir(parents=True)
    proc = sleeper("-m", "abstractruntime.identity.life", "--home", str(home), "--provider", "stub")
    command = os_process.command(proc.pid)
    assert command and "abstractruntime.identity.life" in (command if isinstance(command, str) else " ".join(command))
    assert life._is_loop_invocation(command, home) is False


def test_loop_invocation_by_argument_list(tmp_path: Path) -> None:
    home = tmp_path / "entities with space" / "gandalf"
    home.mkdir(parents=True)
    head = ["C:\\Program Files\\Python312\\python.exe", "-m", "abstractruntime.identity.life", "--home"]
    assert life._is_loop_invocation(head + [str(home), "--provider", "stub"], home) is True
    assert life._is_loop_invocation(head + [str(home)], home) is True
    assert life._is_loop_invocation(head + [str(home) + "2"], home) is False
    assert life._is_loop_invocation(head, home) is False
    assert life._is_loop_invocation(head[:1] + ["-c", "x"] + head[1:] + [str(home)], home) is False
    assert life._is_loop_invocation(["python", "-m", "another.module", "--home", str(home)], home) is False
    assert life._is_loop_invocation([], home) is False and life._is_loop_invocation(None, home) is False


def test_loop_invocation_by_ps_line_is_unchanged(tmp_path: Path) -> None:
    home = tmp_path / "gandalf"
    home.mkdir()
    line = f"/usr/bin/python3 -m abstractruntime.identity.life --home {home}"
    assert life._is_loop_invocation(line + " --provider stub", home) is True
    assert life._is_loop_invocation(line, home) is True
    assert life._is_loop_invocation(line + "2", home) is False
    assert life._is_loop_invocation(f"/usr/bin/python3 -c x -m abstractruntime.identity.life --home {home}", home) is False


@pytest.mark.skipif(not WINDOWS, reason="8.3 short names are a Windows spelling")
def test_windows_a_home_named_by_its_short_path_is_the_same_home(tmp_path: Path) -> None:
    """One folder, two spellings (a short ASTROD~1 form and the long name): a loop started with one must be
    recognised by a Stop that names the other, else the Stop refuses a loop that is plainly this home's."""
    import ctypes

    home = (tmp_path / "a rather long folder name" / "gandalf").resolve()
    home.mkdir(parents=True)
    buffer = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetShortPathNameW(str(home), buffer, len(buffer))
    short = buffer.value
    if not short or short.lower() == str(home).lower():
        pytest.skip("this volume keeps no short names")
    head = ["python.exe", "-m", "abstractruntime.identity.life", "--home"]
    assert life._is_loop_invocation(head + [short], home) is True
    assert life._is_loop_invocation(head + [str(home)], Path(short)) is True
    assert life._is_loop_invocation(head + [str(home).upper()], home) is True
    assert life._is_loop_invocation(head + [short + "x"], home) is False
