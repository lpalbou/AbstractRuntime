"""Title failures are recorded, never repaired (operator ruling 2026-11-09).

"we shouldn't need a fallback. if it fails, it should add information to a temporary error file in untracked/ so we
know what's failing and how to fix it at the source."

Every memory's title and summary are authored with it (identity/authoring.py; operator 2026-11-09). When that
authoring fails, the record forms PENDING through AbstractMemory (its task queued) and, one JSON line is appended to

    <workspace>/untracked/authoring-failures/<YYYY-MM-DD>.jsonl

where <workspace> is the parent of the data dir that holds the home (`<data_dir>/entities/<slug>`), when that parent
holds an `untracked/` folder. No environment variable is read. When the folder does not resolve (a home outside a
workspace, a test home), or the append fails, the same line goes to the home's own-time log after the
`#AUTHORING_FAILURE` marker, and to the caller's `out` when one is given. Never silently.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

__all__ = [
    "TITLE_FAILURE_KINDS",
    "TITLE_FAILURE_MARKER",
    "TITLE_FAILURE_REASONS",
    "TITLE_MAX_CHARS",
    "TITLE_TEMPLATE_VERSION",
    "record_authoring_failure",
    "record_title_failure",
    "title_failures_dir",
    "is_placeholder_title",
]

TITLE_FAILURE_REASONS = ("missing", "empty", "multi_line", "too_long", "malformed")
TITLE_FAILURE_KINDS = ("interest", "lesson", "realization", "diary", "digest", "card")
# Past this a `title:` line is not a title (the prompts ask for "a short, complete name"); its words stay in the
# record's body, whole.
TITLE_MAX_CHARS = 200
# The date the prompts began asking for the `title:` first line (operator 2026-11-08). Bump it with any change to
# the wording that asks for a title, so failures group by the prompt that produced them.
TITLE_TEMPLATE_VERSION = "2026-11-08"
# Renamed 2026-11-09 (the title AND summary are mandatory; the file records either missing): authoring-failures.
TITLE_FAILURE_MARKER = "#AUTHORING_FAILURE"
TITLE_FAILURES_DIRNAME = "authoring-failures"
AUTHORING_MISSING = ("title", "summary", "both")


_PLACEHOLDER_KINDS = ("interest", "lesson", "realization", "diary", "digest", "card")
# Memory's BIRTH LABELS (abstractmemory engram: a spark item with no `name` is titled `<section stem>-<ordinal>`;
# before c5260 the honesty items got `trait-<n>` too - a live home's two identity traits were both "trait-0"). Exact formats
# only: the stem, one hyphen, the ordinal's digits - never words that merely contain them.
_BIRTH_LABEL_STEMS = ("trait", "value", "purpose", "limit")


def _is_birth_label(text: str) -> bool:
    stem, sep, ordinal = text.partition("-")
    return bool(sep) and stem in _BIRTH_LABEL_STEMS and ordinal.isdigit() and ordinal.isascii()


def is_placeholder_title(title: Any) -> bool:
    """True for a title that is a template label, never his words: the retired storage placeholder
    `untitled <kind> [<6 hex>]` (runtime 8e72975, before memory stored a pending title as None), and memory's birth
    labels `trait-N` / `value-N` / `purpose-N` / `limit-N` (an unnamed spark item; operator 2026-11-10). Renderers show
    no title for it, and the sleep's repair declares the record pending so its title is authored."""
    text = str(title or "")
    if _is_birth_label(text):
        return True
    if not text.startswith("untitled ") or not text.endswith("]"):
        return False
    kind, sep, key = text[len("untitled "):-1].partition(" [")
    return bool(sep) and kind in _PLACEHOLDER_KINDS and len(key) == 6 and all(c in "0123456789abcdef" for c in key)


def title_failures_dir(home_dir: Any) -> Optional[Path]:
    """`<workspace>/untracked/authoring-failures` for a home at `<data_dir>/entities/<slug>`, or None when the data dir's
    parent holds no `untracked/` folder."""
    if home_dir is None:
        return None
    try:
        workspace = Path(home_dir).resolve().parent.parent.parent
    except (OSError, RuntimeError, ValueError):
        return None
    untracked = workspace / "untracked"
    return untracked / TITLE_FAILURES_DIRNAME if untracked.is_dir() else None


def record_title_failure(home_dir: Any, **kw: Any) -> Dict[str, Any]:
    """A missing title (record_authoring_failure with missing="title")."""
    kw.setdefault("missing", "title")
    return record_authoring_failure(home_dir, **kw)


def record_authoring_failure(
    home_dir: Any,
    *,
    entity: Optional[str],
    record_id: Optional[str],
    kind: str,
    raw_block: str,
    template_id: str,
    provider: Optional[str],
    model: Optional[str],
    reason: str,
    missing: str = "title",
    template_version: str = TITLE_TEMPLATE_VERSION,
    extra: Optional[Dict[str, Any]] = None,
    out: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Append one failure line (see the module doc). Returns `{"line": <dict>, "path": <file> | None,
    "logged": <the marker line> | None}`; never raises."""
    now = datetime.now(timezone.utc)
    line: Dict[str, Any] = {
        "observed_at": now.isoformat(),
        "entity": entity or None,
        "record_id": record_id or None,
        "kind": str(kind),
        "raw_block": str(raw_block or ""),
        "template_id": str(template_id),
        "template_version": str(template_version),
        "provider": provider or None,
        "model": model or None,
        "reason": str(reason),
        "missing": str(missing) if str(missing) in AUTHORING_MISSING else "title",
    }
    for key, value in (extra or {}).items():
        line.setdefault(str(key), value)
    text = json.dumps(line, ensure_ascii=False, sort_keys=True)
    folder = title_failures_dir(home_dir)
    problem = ""
    if folder is not None:
        path = folder / f"{now:%Y-%m-%d}.jsonl"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
            return {"line": line, "path": str(path), "logged": None}
        except OSError as e:
            problem = f" (the failure file {path} could not be written: {e})"
    marked = f"{TITLE_FAILURE_MARKER} {text}"
    if home_dir is not None:
        try:
            from .life import LOOP_LOG_FILENAME

            with (Path(home_dir) / LOOP_LOG_FILENAME).open("a", encoding="utf-8") as fh:
                fh.write(marked + "\n")
        except OSError as e:
            problem += f" (the own-time log could not be written either: {e})"
    if out is not None:
        try:
            out(marked + problem)
        except Exception:  # noqa: BLE001 - reporting must never break the record it reports on
            pass
    return {"line": line, "path": None, "logged": marked + problem}
