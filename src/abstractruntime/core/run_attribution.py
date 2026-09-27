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


class SessionAttributionError(LookupError):
    """A session's attribution could not be resolved (fail closed)."""

    reason_code = "session_attribution_failed"


def resolve_discussion_root(run_store: Any, session_id: str) -> tuple:
    """`(root_run, root_discussion)` of a discussion session, validated.

    Every discussion member of the session (all runs indexed with role
    `discussion`) must name the SAME `discussion_root_run_id`; that root must
    be one of those members (so it lives in this session), be parent-less,
    name itself as the root, and carry `seed_messages`. Anything else raises
    `SessionAttributionError` — a planted or foreign root is never followed.
    """
    sid = str(session_id or "").strip()
    rows = run_store.list_run_index(session_id=sid, role="discussion", limit=1_000_000)
    members: Dict[str, Any] = {}
    root_ids = set()
    for row in rows:
        rid = str(row.get("run_id") or "")
        run = run_store.load(rid)
        meta = ((run.vars or {}).get("_meta") or {}) if run is not None else {}
        discussion = meta.get("discussion") if isinstance(meta, Mapping) else None
        root_id = _text(discussion.get("discussion_root_run_id")) if isinstance(discussion, Mapping) else None
        if run is None or not root_id:
            raise SessionAttributionError(f"discussion session {sid}: run {rid} has no discussion_root_run_id")
        members[rid] = run
        root_ids.add(root_id)
    if len(root_ids) != 1:
        raise SessionAttributionError(
            f"discussion session {sid}: members disagree on the discussion root ({sorted(root_ids)})"
        )
    (root_id,) = root_ids
    root = members.get(root_id)
    if root is None:
        raise SessionAttributionError(f"discussion session {sid}: root run {root_id} is not a run of this session")
    root_discussion = ((root.vars or {}).get("_meta") or {}).get("discussion")
    if (
        str(root.parent_run_id or "").strip()
        or str(root.session_id or "").strip() != sid
        or not isinstance(root_discussion, Mapping)
        or "seed_messages" not in root_discussion
    ):
        raise SessionAttributionError(f"discussion session {sid}: {root_id} is not a valid discussion root")
    return root, root_discussion


def store_session_kinds(run_store: Any, session_id: str) -> frozenset:
    """`session_kind` values of a session's runs: the store's own indexed
    `session_kinds` when it has one (all built-in stores), else derived from
    its `list_run_index(session_id=...)` rows (third-party stores keep working,
    at their index's cost). A store with neither raises
    `SessionAttributionError`."""
    sid = str(session_id or "").strip()
    if not sid:
        return frozenset()
    session_kinds = getattr(run_store, "session_kinds", None)
    if callable(session_kinds):
        return frozenset(session_kinds(sid))
    list_run_index = getattr(run_store, "list_run_index", None)
    if not callable(list_run_index):
        raise SessionAttributionError(
            f"run store {type(run_store).__name__} has no run index; session {sid} cannot be attributed"
        )
    return frozenset(
        k for k in (r.get("session_kind") for r in list_run_index(session_id=sid, limit=1_000_000)) if k
    )


_KIND_PRECEDENCE = ("discussion", "automation", "occurrence", "chat")


def session_attribution(run_store: Any, session_id: str) -> Optional[Dict[str, Any]]:
    """What kind of session `session_id` is.

    Returns None for a session with no runs yet, else `{"kind": ...}` (the
    session's most specific kind: discussion > automation > occurrence >
    chat); a discussion adds `discussion_root_run_id`, `automation_id`,
    `occurrence_index`, `revision`, `workspace_root` and `discussion` (the
    root's `_meta.discussion` without its seed). The persisted, validated
    discussion root is the authority (automations contract B, amendment 4).

    Cost: one `store_session_kinds` lookup (SQLite: an index read; JSON: the
    in-memory session index); only discussion sessions load runs. Raises
    `SessionAttributionError` when the lookup cannot be completed (a store
    without a run index, or an invalid discussion root).
    """
    sid = str(session_id or "").strip()
    if not sid:
        return None
    kinds = store_session_kinds(run_store, sid)
    if not kinds:
        return None
    kind = next((k for k in _KIND_PRECEDENCE if k in kinds), "chat")
    if kind != "discussion":
        return {"kind": kind}
    root, root_discussion = resolve_discussion_root(run_store, sid)
    workspace_root = (root.vars or {}).get("workspace_root")
    return {
        "kind": "discussion",
        "discussion_root_run_id": str(root.run_id),
        "automation_id": _text(root_discussion.get("automation_id")),
        "occurrence_index": _index(root_discussion.get("occurrence_index")),
        "revision": root_discussion.get("revision"),
        "workspace_root": workspace_root if isinstance(workspace_root, str) and workspace_root.strip() else None,
        "discussion": {k: v for k, v in root_discussion.items() if k != "seed_messages"},
    }


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
    "SessionAttributionError",
    "is_turn_root",
    "resolve_discussion_root",
    "session_attribution",
    "store_session_kinds",
    "latest_occurrence_of",
    "row_matches",
]
