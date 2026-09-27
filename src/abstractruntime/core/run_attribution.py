"""Automation attribution of a run, as stored in the run index (contracts C11/E).

Every run index row carries four columns derived here, at save/index time, from
the run's INLINE `vars._meta` (the offloader never moves these keys):

- `automation_id`    the automation the run belongs to (a controller's own id)
- `role`             controller | occurrence | descendant | discussion | legacy_schedule | None
- `occurrence_index` the occurrence number (occurrences, descendants, discussions)
- `session_kind`     chat | automation | occurrence | discussion

Derivation, first match wins:
- `_meta.automation`                    -> controller, session_kind "automation"
- `_meta.occurrence`                    -> its `role`, its `session_kind`
                                           ("automation" in growing mode: the
                                           occurrence is a turn of the
                                           automation's session; "occurrence"
                                           in independent mode: its own session)
- `_meta.discussion`                    -> discussion
- `_meta.schedule.kind=="scheduled_run"` -> legacy_schedule, session_kind "automation"
                                           (the gateway's pre-automation scheduled
                                           wrapper runs)
- otherwise                              -> chat, role None

"Turn roots" (what `root_only=True` listings and session history return) are the
parent-less runs that are not automation controllers, plus the `occurrence` runs.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

SESSION_KINDS = ("chat", "automation", "occurrence", "discussion")
ROLES = ("controller", "occurrence", "descendant", "discussion", "legacy_schedule")

INDEX_FIELD_NAMES = ("automation_id", "role", "occurrence_index", "session_kind")


def _text(value: Any) -> Optional[str]:
    text = str(value).strip() if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
    return text or None


def _index(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def automation_index_fields(vars_obj: Any, *, run_id: Optional[str] = None) -> Dict[str, Any]:
    """The four attribution index fields for a run's vars (see module docstring)."""
    meta = vars_obj.get("_meta") if isinstance(vars_obj, Mapping) else None
    meta = meta if isinstance(meta, Mapping) else {}

    automation = meta.get("automation")
    if isinstance(automation, Mapping):
        return {
            "automation_id": _text(run_id),
            "role": "controller",
            "occurrence_index": None,
            "session_kind": "automation",
        }

    occurrence = meta.get("occurrence")
    if isinstance(occurrence, Mapping):
        role = occurrence.get("role")
        role = role if role in ("occurrence", "descendant") else "occurrence"
        kind = occurrence.get("session_kind")
        kind = kind if kind in ("automation", "occurrence") else "occurrence"
        return {
            "automation_id": _text(occurrence.get("automation_id")),
            "role": role,
            "occurrence_index": _index(occurrence.get("occurrence_index")),
            "session_kind": kind,
        }

    discussion = meta.get("discussion")
    if isinstance(discussion, Mapping):
        return {
            "automation_id": _text(discussion.get("automation_id")),
            "role": "discussion",
            "occurrence_index": _index(discussion.get("occurrence_index")),
            "session_kind": "discussion",
        }

    schedule = meta.get("schedule")
    if isinstance(schedule, Mapping) and schedule.get("kind") == "scheduled_run":
        return {
            "automation_id": None,
            "role": "legacy_schedule",
            "occurrence_index": None,
            "session_kind": "automation",
        }

    return {"automation_id": None, "role": None, "occurrence_index": None, "session_kind": "chat"}


def filter_values(want: Any) -> Optional[frozenset]:
    """A `list_run_index` filter argument as a set: None (no filter), one
    string, a comma-separated string ("chat,discussion"), or an iterable."""
    if want is None:
        return None
    items = want.split(",") if isinstance(want, str) else list(want)
    values = frozenset(str(v).strip() for v in items if str(v).strip())
    return values


def row_matches(
    row: Mapping[str, Any],
    *,
    automation_id: Any = None,
    role: Any = None,
    session_kind: Any = None,
) -> bool:
    """True when an index row passes the attribution filters."""
    for field, want in (("automation_id", automation_id), ("role", role), ("session_kind", session_kind)):
        values = filter_values(want)
        if values is not None and row.get(field) not in values:
            return False
    return True


def latest_occurrence_of(rows: Any) -> Optional[Dict[str, Any]]:
    """Among occurrence index rows, the highest `occurrence_index`, newest
    attempt (latest `created_at`, then `run_id`), or None."""
    best: Optional[Dict[str, Any]] = None
    best_key: Any = None
    for row in rows or []:
        index = row.get("occurrence_index")
        key = (index if isinstance(index, int) else -1, str(row.get("created_at") or ""), str(row.get("run_id") or ""))
        if best_key is None or key > best_key:
            best, best_key = row, key
    return best


def is_turn_root(*, parent_run_id: Any, role: Any) -> bool:
    """A session turn root: parent-less and not a controller, or an occurrence."""
    if role == "occurrence":
        return True
    if role == "controller":
        return False
    return not str(parent_run_id or "").strip()


__all__ = [
    "INDEX_FIELD_NAMES",
    "ROLES",
    "SESSION_KINDS",
    "automation_index_fields",
    "filter_values",
    "is_turn_root",
    "latest_occurrence_of",
    "row_matches",
]
