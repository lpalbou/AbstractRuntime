"""Atomic text-file writes for operator config files.

The home's operator config files (tool_policy.yaml, system_prompt.yaml,
substrate.yaml) are read at every summon/resolve; a crash mid-write must
never leave a TORN file behind (a half-written yaml reads as malformed →
the resolver degrades to defaults with a #FALLBACK — survivable, but the
operator's word was silently lost). The classic fix: write to a unique
temp file in the SAME directory, fsync, then `os.replace` (atomic on
POSIX within one filesystem). Readers see either the old bytes or the new
bytes, never a mixture.

(Adversary P2 from the prompt-overlay review, 2026-07-11 — one shared fix
for every yaml writer, per the one-source rule.)
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import IO, Any, Callable, TypeVar, Union

__all__ = ["atomic_write_text", "open_replaced", "replace_file"]

# How long `replace_file` waits for readers of the target to close it, and `open_replaced`
# for a rename over its file to finish (Windows only).
REPLACE_WAIT_S = 5.0

_T = TypeVar("_T")


def _retry_denied(call: Callable[[], _T]) -> _T:
    """`call()`, tried again on Windows while it answers PermissionError, for `REPLACE_WAIT_S`."""
    if os.name != "nt":
        return call()
    deadline = time.monotonic() + REPLACE_WAIT_S
    pause = 0.001
    while True:
        try:
            return call()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(pause)
            pause = min(pause * 2, 0.05)


def open_replaced(path: Union[str, Path], mode: str = "r", **kwargs: Any) -> IO[Any]:
    """`open()` for a file `replace_file` may be renaming over at this instant.

    The reader's half of `replace_file`: on Windows the name answers PermissionError for the
    instant of the rename, which a reader that maps every OSError to "no such file" (the run
    store's load) would report as a run that does not exist."""
    return _retry_denied(lambda: open(path, mode, **kwargs))


def replace_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """`os.replace(src, dst)` that also works on Windows while `dst` is being read.

    POSIX renames over an open file. Windows refuses (PermissionError, WinError 5 or 32)
    for as long as ANY handle to `dst` is open without delete sharing, and Python's `open()`
    never asks for it: a run file read by one thread while another saves it failed the save,
    and with it the run (2026-11-18, a fresh Windows install: three of the first five runs,
    AbstractNote's processor among them, ended `failed` with "Access is denied"). Readers
    hold the file for well under a millisecond, so the rename is tried again, with a short
    growing pause, until it goes through or `REPLACE_WAIT_S` has passed (then the error is
    raised: something really holds the file). Readers of such a file use `open_replaced`."""
    _retry_denied(lambda: os.replace(src, dst))


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` to `path` atomically (tmp file + fsync + os.replace +
    directory fsync).

    The directory fsync matters (whole-package adversary finding 11,
    2026-07-13): without it the `os.replace` itself can be lost on power
    failure — for phases.yaml that means a REVOKED personal grant reporting
    success and resurrecting ARMED after reboot. The consent record deserves
    the full discipline; failure to fsync the directory degrades silently
    (some filesystems refuse O_DIRECTORY opens — the file fsync still
    happened)."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        replace_file(tmp_name, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # best-effort: the rename is durable on fsync-honoring fs
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
