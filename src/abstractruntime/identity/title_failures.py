"""Title failures are recorded, never repaired (operator ruling 2026-11-09).

"we shouldn't need a fallback. if it fails, it should add information to a temporary error file in untracked/ so we
know what's failing and how to fix it at the source."

A record the entity names itself (an elected interest, lesson or realization: the block's `title:` first line; a
public diary entry: its `gist:` line) gets NO title from his words when that line is missing or unusable. No text
heuristic stands in for it (a first line or a first sentence used to: "Dr. Smith taught me patience." became "Dr.").
AbstractMemory refuses a record without a title (`MemoryRecordInput` requires a non-empty one), so until it accepts
a missing title the record carries the visibly-untitled label `untitled_label(kind, block)` ("untitled lesson
[3fa2c1]": the kind and a key of the block, none of his words; the key keeps two untitled records from reading as
duplicates). Instead of a guess, one JSON line is appended to

    <workspace>/untracked/title-failures/<YYYY-MM-DD>.jsonl

where <workspace> is the parent of the data dir that holds the home (`<data_dir>/entities/<slug>`), when that parent
holds an `untracked/` folder. No environment variable is read. When the folder does not resolve (a home outside a
workspace, a test home), or the append fails, the same line goes to the home's own-time log after the
`#TITLE_FAILURE` marker, and to the caller's `out` when one is given. Never silently.
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
    "record_title_failure",
    "title_failures_dir",
    "is_placeholder_title",
    "untitled_attributes",
    "untitled_label",
]

TITLE_FAILURE_REASONS = ("missing", "empty", "multi_line", "too_long", "malformed")
TITLE_FAILURE_KINDS = ("interest", "lesson", "realization", "diary", "digest", "card")
# Past this a `title:` line is not a title (the prompts ask for "a short, complete name"); its words stay in the
# record's body, whole.
TITLE_MAX_CHARS = 200
# The date the prompts began asking for the `title:` first line (operator 2026-11-08). Bump it with any change to
# the wording that asks for a title, so failures group by the prompt that produced them.
TITLE_TEMPLATE_VERSION = "2026-11-08"
TITLE_FAILURE_MARKER = "#TITLE_FAILURE"
TITLE_FAILURES_DIRNAME = "title-failures"


def untitled_label(kind: str, raw_block: str) -> str:
    """The stored label of a record he did not title: `untitled <kind> [<6 hex>]`, the hex a key of the block he
    wrote (sha256), never his words."""
    import hashlib

    key = hashlib.sha256(str(raw_block or "").encode("utf-8")).hexdigest()[:6]
    return f"untitled {kind} [{key}]"


_PLACEHOLDER_KINDS = ("interest", "lesson", "realization", "diary", "digest", "card")


def is_placeholder_title(title: Any) -> bool:
    """True for a label `untitled_label` made (`untitled <kind> [<6 hex>]`): the runtime's own storage placeholder,
    never his words. Renderers show no title for it (Codex review 2026-11-09 #9, interim until memory stores None)."""
    text = str(title or "")
    if not text.startswith("untitled ") or not text.endswith("]"):
        return False
    kind, sep, key = text[len("untitled "):-1].partition(" [")
    return bool(sep) and kind in _PLACEHOLDER_KINDS and len(key) == 6 and all(c in "0123456789abcdef" for c in key)


def untitled_attributes(reason: str) -> Dict[str, Any]:
    """The record attributes that say its title is a storage placeholder (title_missing) and why."""
    return {"title_missing": True, "title_failure_reason": str(reason)}


def title_failures_dir(home_dir: Any) -> Optional[Path]:
    """`<workspace>/untracked/title-failures` for a home at `<data_dir>/entities/<slug>`, or None when the data dir's
    parent holds no `untracked/` folder."""
    if home_dir is None:
        return None
    try:
        workspace = Path(home_dir).resolve().parent.parent.parent
    except (OSError, RuntimeError, ValueError):
        return None
    untracked = workspace / "untracked"
    return untracked / TITLE_FAILURES_DIRNAME if untracked.is_dir() else None


def record_title_failure(
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
