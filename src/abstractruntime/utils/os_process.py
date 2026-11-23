"""Another process, by pid: is it there, which incarnation is it, what does it run, and ending it.

One implementation per operating system, behind one small contract, for every caller that keeps a
pid on disk and later has to decide about the process behind it (the own-time loop's status file,
its Stop). The POSIX half is `os.kill(pid, 0)`, `os.waitpid` and `ps`, as the own-time loop has
always used them. Windows has none of the three:

- `os.kill(pid, 0)` is not a probe there. Signal 0 is `CTRL_C_EVENT`, so the call either sends a
  console control event or, from a process without a console (a service, `pythonw`), raises for a
  process that is alive.
- `os.WNOHANG` and `signal.SIGKILL` do not exist, and there is no `ps`.

So the Windows half asks the kernel through a process handle (ctypes over kernel32/ntdll, no
dependency): a handle names one process object for as long as it is open, which also closes the
window between "verified" and "terminated" that a pid number leaves open.

The contract:

- `exists(pid)`: the cheap answer for a status read. No subprocess. A process this user may not
  look at reads False (the safe direction for a caller that would otherwise wait on it or signal
  it). On POSIX this process's own exited child still reads True until it is collected: `reap`
  first when the pid may be a child of the caller.
- `reap(pid)`: collect `pid` when it is this process's own exited child (POSIX zombies).
- `state(pid)`: `"alive"`, `"gone"` or `"unknown"`. Only `"gone"` confirms a death; a denied
  permission is `"unknown"`, never a death.
- `start_token(pid)`: the incarnation token. A pid number is reused; (pid, start time) names one
  process. A writer stamps its own, a reader compares the stamp with the current holder of the pid.
  Tokens are compared for equality only and are written at rest, so their spelling is fixed:
  POSIX keeps `ps -o lstart=` under `LC_ALL=C TZ=UTC`; Windows writes `win-filetime:<n>`, the
  process creation time in 100 ns units. A token of the other spelling never equals a live one
  (a home copied between machines reads "not that process", which is the truth).
- `command(pid)`: what the process runs. The exact argument list where the system can give it
  (Windows), else the command line as `ps` prints it (POSIX: arguments joined by spaces).
- `terminate(pid, force=..., token=...)`: end it. POSIX sends SIGTERM, or SIGKILL when forced.
  Windows has one way to end another process and it is immediate; when `token` is given it is
  checked on the very handle the termination uses.
- `direct_python(env)`: the interpreter and environment to start a Python child with so that the
  pid `Popen` returns is the pid the child sees as its own. In a Windows virtual environment
  `sys.executable` is a launcher that starts the real interpreter as its child; a pid recorded
  from `Popen` would name the launcher.
- `detached()`: the `Popen` keyword arguments for a child that must not end with its parent's
  terminal (a new session on POSIX; on Windows its own process group and a console without a
  window, so a Ctrl+C in the parent's console does not reach it).
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

__all__ = [
    "WINDOWS_TOKEN_PREFIX",
    "command",
    "detached",
    "direct_python",
    "exists",
    "reap",
    "start_token",
    "state",
    "terminate",
]

# The spelling of a Windows start token (written at rest in status files: do not change it).
WINDOWS_TOKEN_PREFIX = "win-filetime:"

_WINDOWS = sys.platform == "win32"


def _as_pid(pid: Any) -> Optional[int]:
    try:
        return int(pid)
    except Exception:  # noqa: BLE001 - not a number (None, text, infinity, an object that refuses): no pid
        return None


def exists(pid: Any) -> bool:
    """True when `pid` answers, without starting a subprocess. Anything else is False: no such
    process, a pid that is not a number, or a process this user may not look at. (POSIX: this
    process's own exited child answers until collected, see `reap`.)"""
    number = _as_pid(pid)
    if number is None:
        return False
    if _WINDOWS:
        return _windows_state(number) == "alive"
    try:
        os.kill(number, 0)
        return True
    except OSError:
        return False


def reap(pid: Any) -> bool:
    """Collect `pid` when it is OUR exited child; True when it was (it is gone). A child that
    died stays a zombie in its parent until collected, and a zombie still answers signal 0.
    Windows keeps no zombies: always False there."""
    number = _as_pid(pid)
    if number is None or _WINDOWS:
        return False
    try:
        done, _status = os.waitpid(number, os.WNOHANG)
    except OSError:  # not our child (ChildProcessError), or no such process
        return False
    return done == number


def state(pid: Any) -> str:
    """`"alive"`, `"gone"` (collected here, no such process, or a zombie) or `"unknown"`
    (permission denied, or no answer). Only `"gone"` confirms an exit."""
    number = _as_pid(pid)
    if number is None:
        return "unknown"
    if _WINDOWS:
        return _windows_state(number)
    if reap(number):
        return "gone"
    try:
        os.kill(number, 0)
    except ProcessLookupError:
        return "gone"
    except OSError:  # PermissionError included: it exists, and it is not ours to check
        return "unknown"
    stat = _ps(number, "stat=")
    if stat is None:
        return "alive"  # it answered signal 0: it exists
    return "gone" if stat.upper().startswith("Z") else "alive"


def start_token(pid: Any) -> Optional[str]:
    """The incarnation token of `pid` (see the module text), or None when it cannot be read.
    The token is an upgrade over the pid number, never a gate: callers decide what None means."""
    number = _as_pid(pid)
    if number is None:
        return None
    if _WINDOWS:
        return _windows_start_token(number)
    return _ps(number, "lstart=") or None


def command(pid: Any) -> Union[List[str], str, None]:
    """What `pid` runs: its exact argument list (Windows), or its command line as `ps` prints it
    (POSIX). None when it cannot be read."""
    number = _as_pid(pid)
    if number is None:
        return None
    if _WINDOWS:
        return _windows_argv(number)
    return _ps(number, "command=") or None


def terminate(pid: int, *, force: bool = False, token: Optional[str] = None) -> str:
    """End `pid`; returns the name of what was used (`"SIGTERM"`, `"SIGKILL"`,
    `"TerminateProcess"`). Raises ProcessLookupError when that process is not there (or, on
    Windows with `token`, when the pid now belongs to another incarnation) and OSError when the
    system refuses. The caller verifies WHICH process this is before calling."""
    number = int(pid)
    if number <= 0:  # POSIX reads 0 and negative numbers as process GROUPS: never a process to end
        raise ProcessLookupError(f"no process {number}")
    if _WINDOWS:
        _windows_terminate(number, token)
        return "TerminateProcess"
    import signal

    chosen = signal.SIGKILL if force else signal.SIGTERM
    os.kill(number, chosen)
    return chosen.name


def detached() -> Dict[str, Any]:
    """`Popen` keyword arguments for a child that outlives its parent's terminal (see the module text)."""
    if _WINDOWS:
        import subprocess

        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def direct_python(env: Optional[Mapping[str, str]] = None) -> Tuple[str, Dict[str, str]]:
    """(executable, environment) to start a Python child of this interpreter whose own pid is the
    pid `Popen` returns. Everywhere but a Windows virtual environment that is `sys.executable` and
    the environment unchanged. There, the base interpreter is started directly and told which
    environment it serves (`__PYVENV_LAUNCHER__`), as `multiprocessing` starts its own children:
    the child keeps the environment's `sys.executable`, `sys.prefix` and packages."""
    environment = dict(os.environ if env is None else env)
    executable = sys.executable
    base = str(getattr(sys, "_base_executable", "") or "")
    if _WINDOWS and base and os.path.normcase(base) != os.path.normcase(executable) and os.path.isfile(base):
        environment["__PYVENV_LAUNCHER__"] = executable
        return base, environment
    return executable, environment


# ----------------------------------------------------------------------------------------------
# POSIX
# ----------------------------------------------------------------------------------------------


def _ps(pid: int, column: str) -> Optional[str]:
    """One `ps` column of one process, stripped; None when ps fails or is not there.

    PINNED ENVIRONMENT: a writer under LC_ALL=C and a reader under fr_FR.UTF-8 print DIFFERENT
    `lstart` strings for the same process, and the token then reads a live process as another
    one. Writer and reader must print identically whoever launched them."""
    import subprocess

    try:
        out = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", column],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
    except Exception:  # noqa: BLE001 - no ps, a timeout: the caller degrades
        return None
    return (out.stdout or "").strip()


# ----------------------------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------------------------

_SYNCHRONIZE = 0x00100000
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WAIT_OBJECT_0 = 0x0
_WAIT_TIMEOUT = 0x102
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87  # OpenProcess: no process has that id
_PROCESS_COMMAND_LINE_INFORMATION = 60  # NtQueryInformationProcess (Windows 8.1 and later)

_windows_api: Dict[str, Any] = {}
_windows_api_built = threading.Lock()


def _api() -> Dict[str, Any]:
    """kernel32, ntdll and shell32 with their signatures declared. Handles and pointers are
    pointer-sized: left to ctypes' default `int` they are cut to 32 bits."""
    if _windows_api:
        return _windows_api
    with _windows_api_built:  # built once, whatever the number of threads at first use
        if not _windows_api:
            _build_api()
    return _windows_api


def _build_api() -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    ntdll = ctypes.WinDLL("ntdll")  # type: ignore[attr-defined]
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)  # type: ignore[attr-defined]

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetProcessId.argtypes = [wintypes.HANDLE]
    kernel32.GetProcessId.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    ntdll.NtQueryInformationProcess.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
    ]
    ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)

    class UnicodeString(ctypes.Structure):
        _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT), ("Buffer", ctypes.c_void_p)]

    _windows_api.update(ctypes=ctypes, wintypes=wintypes, kernel32=kernel32, ntdll=ntdll, shell32=shell32,
                        UnicodeString=UnicodeString)


class _Handle:
    """A process handle, closed on exit. `error` is the Windows error when it could not be opened."""

    def __init__(self, pid: int, access: int) -> None:
        api = _api()
        self._kernel32 = api["kernel32"]
        self.value = None
        self.error = 0
        if not 0 < pid <= 0xFFFFFFFF:  # no process has such an id (and ctypes would cut a larger one to 32 bits)
            self.error = _ERROR_INVALID_PARAMETER
            return
        # Always with the right to ask the handle which process it is (GetProcessId answers 0 without it).
        self.value = self._kernel32.OpenProcess(access | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not self.value:
            self.value = None
            self.error = api["ctypes"].get_last_error() or -1
            return
        opened = self._kernel32.GetProcessId(self.value)
        if opened != pid:
            # OpenProcess ignores the two low bits of the id: pid+1 opens pid. That is another number, not this
            # process (a status file naming pid+1 would otherwise pass for it, the caller's own pid included).
            # No answer (0) is unknown, never "no such process".
            self._kernel32.CloseHandle(self.value)
            self.value = None
            self.error = _ERROR_INVALID_PARAMETER if opened else -1

    def __enter__(self) -> "_Handle":
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.value:
            self._kernel32.CloseHandle(self.value)
            self.value = None


def _handle_state(handle: Any) -> str:
    """A process object outlives its process while any handle to it is open (ours, its parent's):
    it has exited when the handle is signalled, whatever its exit code says."""
    waited = _api()["kernel32"].WaitForSingleObject(handle, 0)
    if waited == _WAIT_OBJECT_0:
        return "gone"
    return "alive" if waited == _WAIT_TIMEOUT else "unknown"


def _handle_token(handle: Any) -> Optional[str]:
    api = _api()
    created, exited, kernel, user = (api["wintypes"].FILETIME() for _ in range(4))
    if not api["kernel32"].GetProcessTimes(handle, created, exited, kernel, user):
        return None
    return f"{WINDOWS_TOKEN_PREFIX}{(created.dwHighDateTime << 32) | created.dwLowDateTime}"


def _windows_state(pid: int) -> str:
    try:
        with _Handle(pid, _SYNCHRONIZE) as handle:
            if handle.value:
                return _handle_state(handle.value)
            return "gone" if handle.error == _ERROR_INVALID_PARAMETER else "unknown"
    except Exception:  # noqa: BLE001 - no answer is not a death
        return "unknown"


def _windows_start_token(pid: int) -> Optional[str]:
    try:
        with _Handle(pid, _PROCESS_QUERY_LIMITED_INFORMATION) as handle:
            return _handle_token(handle.value) if handle.value else None
    except Exception:  # noqa: BLE001
        return None


def _windows_argv(pid: int) -> Optional[List[str]]:
    """The process's own command line (read from the process by the kernel), split as the C
    runtime splits it into argv."""
    try:
        api = _api()
        ctypes = api["ctypes"]
        with _Handle(pid, _PROCESS_QUERY_LIMITED_INFORMATION) as handle:
            if not handle.value:
                return None
            needed = api["wintypes"].ULONG(0)
            api["ntdll"].NtQueryInformationProcess(
                handle.value, _PROCESS_COMMAND_LINE_INFORMATION, None, 0, ctypes.byref(needed))
            if not needed.value:
                return None
            buffer = ctypes.create_string_buffer(needed.value)
            status = api["ntdll"].NtQueryInformationProcess(
                handle.value, _PROCESS_COMMAND_LINE_INFORMATION, buffer, needed.value, ctypes.byref(needed))
            if status != 0:
                return None
            text = api["UnicodeString"].from_buffer(buffer)
            line = ctypes.wstring_at(text.Buffer, text.Length // 2) if text.Buffer and text.Length else ""
        if not line.strip():
            return None  # CommandLineToArgvW answers the CALLER's program path for an empty line
        count = ctypes.c_int(0)
        parts = api["shell32"].CommandLineToArgvW(line, ctypes.byref(count))
        if not parts:
            return None
        try:
            return [parts[i] for i in range(count.value)]
        finally:
            api["kernel32"].LocalFree(parts)
    except Exception:  # noqa: BLE001
        return None


def _windows_terminate(pid: int, token: Optional[str]) -> None:
    api = _api()
    access = _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE
    with _Handle(pid, access) as handle:
        if not handle.value:
            if handle.error == _ERROR_INVALID_PARAMETER:
                raise ProcessLookupError(f"no process {pid}")
            if handle.error == _ERROR_ACCESS_DENIED:
                raise PermissionError(f"process {pid} may not be ended by this user")
            raise OSError(f"process {pid} could not be opened (Windows error {handle.error})")
        if _handle_state(handle.value) == "gone":
            raise ProcessLookupError(f"process {pid} has already exited")
        if token and _handle_token(handle.value) != str(token):
            raise ProcessLookupError(f"pid {pid} now belongs to another process")
        if not api["kernel32"].TerminateProcess(handle.value, 1):
            error = api["ctypes"].get_last_error()
            if _handle_state(handle.value) == "gone":
                raise ProcessLookupError(f"process {pid} has already exited")
            raise OSError(f"process {pid} could not be ended (Windows error {error})")
