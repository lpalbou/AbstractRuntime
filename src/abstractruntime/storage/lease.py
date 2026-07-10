"""Directory writer lease: ONE writer per directory at a time.

Generic mutual exclusion for any directory whose STORES must have a single
writer — entity homes are the first consumer (plan item 1, GW-A: visit host,
own-time loop day, dream window, reembed maintenance), project workplaces
(plan item 15) are the designed second. Re-homed from `identity/lease.py`
under the 2026-07-10 vocabulary sign-off (a2a/fs/renaming.md, approved by
the maintainer): the mechanism serializes writers of a directory's stores —
storage's domain — and "home" only named its first consumer.

Design validation (maintainer-driven adversarial passes, 2026-07-10): the
lease arbitrates PROCESSES, never principals (visitors' write-rights are
denied structurally at the deposit gate, lease or no lease). It exists
because legitimate writers do NOT share one process: the own-time loop is a
detached process by ruling, CLI sessions are home-direct, and maintenance
verbs open stores from the invoking process. The damage class under
collision is PERMANENT on append-only stores (hash-chain forks, journal
seq-axis interleave), which is what justifies a kernel lock despite low
collision frequency. flock was chosen over TTL leases / pid files / sockets
/ ports / O_EXCL because it alone combines crash-release, copy-inertness,
works-for-every-entry-point, and zero standing machinery.

Mechanics (the spawn-lock precedent, life.py `.loop_spawn.lock`):

- ONE dotfile inside the directory: `<dir>/.writer_lease`. The file is never
  unlinked (unlink races a holder's fd onto a dead inode and two "holders"
  end up on different files); release truncates to a released record.
- `flock(LOCK_EX | LOCK_NB)` on an open fd is the mutual exclusion. flock
  conflicts are per-open-file-description, so a second acquire in the SAME
  process conflicts too (testable in-process), and the kernel releases the
  lock when the holder dies (fd close) — a crashed writer never wedges the
  directory. Non-POSIX platforms degrade to best-effort with a LABELED
  warning in the acquired metadata, same posture as the spawn lock.
- Holder metadata (kind, pid, acquired_at, session/run id) is written into
  the file for the "who holds this directory?" question and refusal
  messages — DIAGNOSTICS ONLY, never trusted for exclusion (flock is the
  truth).
- Acquisition failure raises `DirectoryLeaseHeld` naming the holder — a
  loud 409-class refusal, never a silent wait. Callers that can yield (the
  loop at its day gate) yield at their own boundary.
- STALE COPIES ARE INERT: a copied directory carries `.writer_lease` bytes,
  but flock state does not travel with file contents — the copy acquires
  freely and the stale metadata is overwritten. This is why the file may
  travel on copy without operator discipline.
- THE ONE-LEASE RELAY RULE (plan invariant): any cross-directory delivery
  acquires ONE lease at a time, never two — deadlock-free by shape.
  Nothing in this module can hold two directories atomically; keep it that
  way.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LEASE_FILENAME = ".writer_lease"

# Declared holder kinds (diagnostics vocabulary; free strings are not
# refused — a future writer kind must not need a lease.py release — but
# the four plan windows should use these exact words).
HOLDER_KINDS = ("visit-host", "loop", "dream", "maintenance")


class DirectoryLeaseHeld(RuntimeError):
    """Acquisition refused: another writer holds the directory.

    `holder` carries the incumbent's metadata when it was readable
    (diagnostics from the lease file — the flock refusal itself is the
    authoritative fact)."""

    def __init__(self, message: str, *, holder: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.holder = dict(holder) if isinstance(holder, dict) else None


def _lease_path(dir_path: Path) -> Path:
    return Path(dir_path) / LEASE_FILENAME


def _read_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort read of the lease file's JSON body (None when absent or
    unreadable — the file is diagnostics; a torn write must not raise)."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001 - torn concurrent write reads as no metadata
        return None
    return data if isinstance(data, dict) else None


class DirectoryLease:
    """One writer window over one directory. Use as a context manager or via
    explicit acquire()/release(); acquire() raises DirectoryLeaseHeld when
    the directory is already held."""

    def __init__(
        self,
        dir_path: Path,
        *,
        holder: str,
        session_id: str = "",
        run_id: str = "",
    ) -> None:
        self.dir_path = Path(dir_path)
        self.holder = str(holder or "").strip() or "unknown"
        self.session_id = str(session_id or "")
        self.run_id = str(run_id or "")
        self._fh: Optional[Any] = None
        self.metadata: Optional[Dict[str, Any]] = None

    @property
    def acquired(self) -> bool:
        return self._fh is not None

    def acquire(self) -> Dict[str, Any]:
        if self._fh is not None:  # idempotent within the window
            return dict(self.metadata or {})
        path = _lease_path(self.dir_path)
        # "a+" never truncates an incumbent's metadata on open; we truncate
        # only AFTER the flock says the window is ours.
        fh = open(path, "a+", encoding="utf-8")
        warning = ""
        try:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:
                # Non-POSIX: best-effort (no kernel exclusion). Labeled, not
                # silent — the caller decides whether to surface it.
                warning = "#FALLBACK flock unavailable on this platform; lease is advisory-only"
            except OSError:
                incumbent = _read_metadata(path)
                fh.close()
                who = ""
                if incumbent:
                    who = (
                        f" (held by {incumbent.get('holder', 'unknown')}"
                        f" pid {incumbent.get('pid', '?')}"
                        f" since {incumbent.get('acquired_at', '?')})"
                    )
                raise DirectoryLeaseHeld(
                    f"the directory at {self.dir_path} already has a writer{who} - "
                    "one writer per directory; retry at the holder's next boundary",
                    holder=incumbent,
                )
        except Exception:
            if not fh.closed:
                fh.close()
            raise
        meta: Dict[str, Any] = {
            "holder": self.holder,
            "pid": os.getpid(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.session_id:
            meta["session_id"] = self.session_id
        if self.run_id:
            meta["run_id"] = self.run_id
        if warning:
            meta["warning"] = warning
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(meta) + "\n")
            fh.flush()
        except OSError:
            # Metadata is diagnostics; the flock (already held) is the truth.
            meta.setdefault("warning", "#FALLBACK lease metadata write failed; lock held regardless")
        self._fh = fh
        self.metadata = meta
        return dict(meta)

    def release(self) -> None:
        if self._fh is None:
            return
        fh, self._fh = self._fh, None
        released = {
            "released": True,
            "holder": self.holder,
            "pid": os.getpid(),
            "released_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(released) + "\n")
            fh.flush()
        except OSError:
            pass  # diagnostics only; closing the fd is what releases the flock
        finally:
            try:
                fh.close()  # closes the description -> kernel drops the lock
            except OSError:
                pass

    def __enter__(self) -> "DirectoryLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def acquire_directory_lease(
    dir_path: Path, *, holder: str, session_id: str = "", run_id: str = ""
) -> DirectoryLease:
    """Acquire the directory's writer lease NOW (raises DirectoryLeaseHeld
    when held).

    Returns the DirectoryLease already acquired — usable as a context
    manager (`with acquire_directory_lease(...):` — enter is idempotent) or
    released explicitly via `.release()`."""
    lease = DirectoryLease(dir_path, holder=holder, session_id=session_id, run_id=run_id)
    lease.acquire()
    return lease


def read_directory_lease(dir_path: Path) -> Optional[Dict[str, Any]]:
    """The directory's lease state for diagnostics: None when no lease file
    exists; otherwise the last-written metadata plus `held` — determined by
    a real non-destructive flock PROBE (metadata alone cannot answer it: a
    crashed holder leaves stale bytes while the kernel already released the
    lock, and a copied directory carries the origin's bytes with no lock at
    all)."""
    path = _lease_path(dir_path)
    if not path.exists():
        return None
    out: Dict[str, Any] = dict(_read_metadata(path) or {})
    held: Optional[bool] = None
    try:
        import fcntl

        probe = open(path, "a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
                held = False
            except OSError:
                held = True
        finally:
            probe.close()
    except ImportError:
        held = None  # advisory-only platform: honestly unknown
    out["held"] = held
    return out
