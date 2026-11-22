"""`utils.file_lock`: one cross-process lock on every system (2026-11-22).

The own-time loop and its host are two processes writing one home. Their exclusion was `fcntl.flock`, which
Windows does not have: there, two simultaneous starts both launched a loop and a Stop's generation bump could land
between a write's fence check and the write. Real files and real processes, no mocks of the system calls."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import abstractruntime.identity.life as life
import abstractruntime.identity.lifecycle_fence as lf
from abstractruntime.utils import file_lock, os_process

SRC = str(Path(life.__file__).resolve().parents[2])

# A child that takes the lock on argv[1] (shared when argv[2] == "shared"), says so, and holds it until killed.
HOLDER = (
    "import sys, time\n"
    f"sys.path.insert(0, {SRC!r})\n"
    "from abstractruntime.utils import file_lock\n"
    "fh = open(sys.argv[1], 'a+', encoding='utf-8')\n"
    "file_lock.lock(fh, shared=sys.argv[2] == 'shared')\n"
    "print('held', flush=True)\n"
    "time.sleep(30)\n"  # never outlives a killed test run by long
)


@pytest.fixture
def holder():
    """Start a process that holds the lock on a path; every holder is killed at the end."""
    procs = []

    def start(path: Path, mode: str = "exclusive") -> subprocess.Popen:
        python, env = os_process.direct_python()
        proc = subprocess.Popen([python, "-c", HOLDER, str(path), mode], env=env, stdout=subprocess.PIPE, text=True)
        procs.append(proc)
        assert proc.stdout is not None and proc.stdout.readline().strip() == "held"
        return proc

    yield start
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()


def _open(path: Path):
    return open(path, "a+", encoding="utf-8")


def test_this_system_has_a_file_lock() -> None:
    assert file_lock.available() is True


def test_two_opens_of_one_file_exclude_each_other_in_one_process(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    with _open(path) as first, _open(path) as second:
        file_lock.lock(first)
        with pytest.raises(BlockingIOError):
            file_lock.lock(second, blocking=False)
        file_lock.unlock(first)
        file_lock.lock(second, blocking=False)
        file_lock.unlock(second)


def test_another_process_holding_it_is_refused_and_its_death_releases_it(tmp_path: Path, holder) -> None:
    path = tmp_path / "x.lock"
    proc = holder(path)
    with _open(path) as fh:
        with pytest.raises(BlockingIOError):
            file_lock.lock(fh, blocking=False)
        proc.kill()
        proc.wait(timeout=10)
        deadline = time.monotonic() + 10
        while True:
            try:
                file_lock.lock(fh, blocking=False)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline, "the dead holder's lock was never released"
                time.sleep(0.02)
        file_lock.unlock(fh)


def test_a_blocking_lock_waits_for_the_holder(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    with _open(path) as first, _open(path) as second:
        file_lock.lock(first)
        threading.Timer(0.3, lambda: file_lock.unlock(first)).start()
        started = time.monotonic()
        file_lock.lock(second)  # no limit: returns when the holder lets go
        assert time.monotonic() - started >= 0.25
        file_lock.unlock(second)


def test_first_use_from_many_threads_at_once(tmp_path: Path) -> None:
    """The first lock of a process builds the Windows call signatures: once, whatever the number of threads."""
    code = (
        "import sys, threading\n"
        f"sys.path.insert(0, {SRC!r})\n"
        "sys.setswitchinterval(1e-6)\n"
        "from abstractruntime.utils import file_lock\n"
        "errors = []\n"
        "go = threading.Event()\n"
        "def work(i):\n"
        "    go.wait()\n"
        "    try:\n"
        "        with open(sys.argv[1] + str(i), 'a+') as fh:\n"
        "            file_lock.lock(fh); file_lock.unlock(fh)\n"
        "    except BaseException as e:\n"
        "        errors.append(repr(e))\n"
        "threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]\n"
        "[t.start() for t in threads]; go.set(); [t.join() for t in threads]\n"
        "print(errors)\n"
    )
    python, env = os_process.direct_python()
    for _ in range(5):
        out = subprocess.run([python, "-c", code, str(tmp_path / "f")], env=env, capture_output=True, text=True,
                             timeout=60)
        assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_shared_holders_coexist_and_exclude_an_exclusive_one(tmp_path: Path, holder) -> None:
    path = tmp_path / "x.lock"
    proc = holder(path, "shared")
    with _open(path) as mine, _open(path) as writer:
        file_lock.lock(mine, shared=True, blocking=False)  # two processes hold it shared
        with pytest.raises(BlockingIOError):
            file_lock.lock(writer, blocking=False)
        file_lock.unlock(mine)
        with pytest.raises(BlockingIOError):
            file_lock.lock(writer, blocking=False)  # the other process still holds it shared
        proc.kill()
        proc.wait(timeout=10)
        deadline = time.monotonic() + 10
        while True:
            try:
                file_lock.lock(writer, blocking=False)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline
                time.sleep(0.02)


def test_the_content_of_a_locked_file_stays_readable_and_writable(tmp_path: Path, holder) -> None:
    """The task inbox is locked on the data file itself. A Windows byte-range lock is mandatory for the bytes it
    covers, so the lock sits far past the end of the file: its content is never refused to anyone."""
    path = tmp_path / "inbox.jsonl"
    path.write_text("one\n", encoding="utf-8")
    holder(path)
    assert path.read_text(encoding="utf-8") == "one\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("two\n")
    assert path.read_text(encoding="utf-8") == "one\ntwo\n"


# ------------------------------------------------------------------ the home's locks, across processes

# A child that holds a home's lock (argv[2]: "state", or the write barrier "shared"/"exclusive") until killed.
HOME_HOLDER = (
    "import sys, time\n"
    f"sys.path.insert(0, {SRC!r})\n"
    "from pathlib import Path\n"
    "import abstractruntime.identity.life as life\n"
    "import abstractruntime.identity.lifecycle_fence as lf\n"
    "home = Path(sys.argv[1])\n"
    "held = life.home_state_lock(home, timeout_s=None) if sys.argv[2] == 'state' else "
    "lf.write_barrier(home, exclusive=sys.argv[2] == 'exclusive')\n"
    "with held:\n"
    "    print('held', flush=True)\n"
    "    time.sleep(30)\n"
)


@pytest.fixture
def home_holder():
    procs = []

    def start(home: Path, what: str) -> subprocess.Popen:
        python, env = os_process.direct_python()
        proc = subprocess.Popen([python, "-c", HOME_HOLDER, str(home), what], env=env, stdout=subprocess.PIPE,
                                text=True)
        procs.append(proc)
        assert proc.stdout is not None and proc.stdout.readline().strip() == "held"
        return proc

    yield start
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()


def test_the_home_state_lock_excludes_another_process(tmp_path: Path, home_holder) -> None:
    proc = home_holder(tmp_path, "state")
    with pytest.raises(life.HomeStateLockTimeout):
        with life.home_state_lock(tmp_path, timeout_s=0.3):
            pass
    proc.kill()
    proc.wait(timeout=10)
    with life.home_state_lock(tmp_path, timeout_s=10):
        pass


def test_the_spawn_lock_admits_one_start_at_a_time_across_processes(tmp_path: Path, holder) -> None:
    proc = holder(tmp_path / ".loop_spawn.lock")
    with pytest.raises(RuntimeError, match="another start is already in progress"):
        with life._spawn_lock(tmp_path):
            pass
    proc.kill()
    proc.wait(timeout=10)
    deadline = time.monotonic() + 10
    while True:
        try:
            with life._spawn_lock(tmp_path):
                break
        except RuntimeError:
            assert time.monotonic() < deadline
            time.sleep(0.02)


def test_the_write_barrier_holds_a_bump_out_while_another_process_writes(tmp_path: Path, home_holder) -> None:
    """A write section (shared) in one process, the Stop's bump (exclusive) in another: the bump waits."""
    proc = home_holder(tmp_path, "shared")
    entered = threading.Event()

    def bump() -> None:
        with lf.write_barrier(tmp_path, exclusive=True):
            entered.set()

    thread = threading.Thread(target=bump, daemon=True)
    thread.start()
    assert not entered.wait(0.5), "the bump entered while another process held a write section"
    with lf.write_barrier(tmp_path, exclusive=False):  # other writers are still admitted meanwhile
        pass
    proc.kill()
    proc.wait(timeout=10)
    assert entered.wait(10), "the bump never entered after the writer was gone"
    thread.join(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows always has a file lock")
def test_without_any_file_lock_the_home_state_lock_still_excludes_threads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "fcntl", None)  # `import fcntl` raises ImportError
    assert file_lock.available() is False
    order = []

    def second() -> None:
        with life.home_state_lock(tmp_path, timeout_s=10):
            order.append("second")

    with life.home_state_lock(tmp_path, timeout_s=10):
        thread = threading.Thread(target=second)
        thread.start()
        time.sleep(0.2)
        order.append("first")
    thread.join(timeout=10)
    assert order == ["first", "second"]
