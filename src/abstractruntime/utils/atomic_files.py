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
from pathlib import Path

__all__ = ["atomic_write_text"]


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` to `path` atomically (tmp file + fsync + os.replace)."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
