"""An advisory lock on an open file, across processes: `fcntl.flock` where it exists, `LockFileEx` on Windows.

The own-time loop and its host are two processes writing one home (the status file, the grant, the lifecycle
generation). Their exclusion is a lock on a dedicated file. `fcntl` does not exist on Windows, and every
caller used to fall back to a lock inside one process there, or to none: a Stop in the host and a write in the
loop did not exclude each other.

Same meaning on both systems:

- the lock belongs to the OPEN FILE (`fh`): two opens of one path exclude each other, in one process or two;
- one lock per open file: open the file, lock it once, unlock or close it. Locking an open file that already
  holds the lock, or turning its shared lock into an exclusive one, is NOT portable (`flock` converts in place,
  Windows waits on itself);
- `shared=True` admits other shared holders and excludes an exclusive one;
- `blocking=True` waits without a limit and can be interrupted (Ctrl+C) while it waits; `blocking=False` raises
  `BlockingIOError` when it is held elsewhere;
- closing the file releases it (a process that dies releases it).

`available()` is False only on a system with neither mechanism; callers there keep what they did before this
module existed (an exclusion inside one process, or none), and say so.

On Windows a byte-range lock is mandatory for the bytes it covers, so the lock is taken on ONE byte far past the
end of any real file (`_WINDOWS_LOCK_OFFSET`): readers and writers of the file's content are never refused, also
when the locked file is a data file and not a dedicated lock file.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import IO, Any, Dict

__all__ = ["available", "lock", "unlock"]

_WINDOWS = sys.platform == "win32"
_WINDOWS_LOCK_OFFSET = 1 << 62
_LOCKFILE_FAIL_IMMEDIATELY = 0x1
_LOCKFILE_EXCLUSIVE_LOCK = 0x2
_ERROR_LOCK_VIOLATION = 33
_ERROR_IO_PENDING = 997
# A blocking LockFileEx cannot be interrupted (a Ctrl+C would wait for the holder): the wait is a retry instead.
_WINDOWS_RETRY_S = 0.02

_windows_api: Dict[str, Any] = {}
_windows_api_built = threading.Lock()


def available() -> bool:
    """Whether this system has a cross-process file lock (`fcntl.flock`, or Windows)."""
    if _WINDOWS:
        return True
    try:
        import fcntl  # noqa: F401
    except ImportError:
        return False
    return True


def lock(fh: IO[Any], *, shared: bool = False, blocking: bool = True) -> None:
    """Take the lock on `fh` (see the module text). Raises BlockingIOError when `blocking` is False and another
    open file holds it; OSError for any other refusal."""
    if _WINDOWS:
        _windows_lock(fh, shared=shared, blocking=blocking)
        return
    import fcntl

    fcntl.flock(fh.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB))


def unlock(fh: IO[Any]) -> None:
    """Release the lock on `fh` (closing the file also does)."""
    if _WINDOWS:
        _windows_unlock(fh)
        return
    import fcntl

    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _api() -> Dict[str, Any]:
    if _windows_api:
        return _windows_api
    with _windows_api_built:  # built once: the argument types name ONE Overlapped class
        if not _windows_api:
            _build_api()
    return _windows_api


def _build_api() -> None:
    import ctypes
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                    ctypes.POINTER(Overlapped)]
    kernel32.LockFileEx.restype = wintypes.BOOL
    kernel32.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.POINTER(Overlapped)]
    kernel32.UnlockFileEx.restype = wintypes.BOOL
    _windows_api.update(ctypes=ctypes, kernel32=kernel32, Overlapped=Overlapped)


def _windows_range(fh: IO[Any]) -> Any:
    import msvcrt

    api = _api()
    where = api["Overlapped"]()
    where.Offset = _WINDOWS_LOCK_OFFSET & 0xFFFFFFFF
    where.OffsetHigh = _WINDOWS_LOCK_OFFSET >> 32
    return msvcrt.get_osfhandle(fh.fileno()), where


def _windows_lock(fh: IO[Any], *, shared: bool, blocking: bool) -> None:
    api = _api()
    handle, where = _windows_range(fh)
    flags = (0 if shared else _LOCKFILE_EXCLUSIVE_LOCK) | _LOCKFILE_FAIL_IMMEDIATELY
    while True:
        if api["kernel32"].LockFileEx(handle, flags, 0, 1, 0, api["ctypes"].byref(where)):
            return
        error = api["ctypes"].get_last_error()
        if error not in (_ERROR_LOCK_VIOLATION, _ERROR_IO_PENDING):
            raise OSError(f"the file lock could not be taken (Windows error {error})")
        if not blocking:
            raise BlockingIOError(f"the file lock is held by another open file (Windows error {error})")
        time.sleep(_WINDOWS_RETRY_S)


def _windows_unlock(fh: IO[Any]) -> None:
    api = _api()
    handle, where = _windows_range(fh)
    if not api["kernel32"].UnlockFileEx(handle, 0, 1, 0, api["ctypes"].byref(where)):
        raise OSError(f"the file lock could not be released (Windows error {api['ctypes'].get_last_error()})")
