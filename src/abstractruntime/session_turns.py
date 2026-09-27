"""One selector for "the turns of a session" (automations contract C3 / E).

Every history path — history bundles (`history_bundle._best_effort_session_turns`),
session replay (`session_history.session_chat_messages`) and the gateway's
growing-mode seeding — takes its turns from `select_session_turns`, so an
automation's occurrences are conversation turns everywhere or nowhere.

A session's turns are its TURN ROOTS (`core.run_attribution.is_turn_root`):

- its parent-less runs, and
- its automation occurrence runs (`vars._meta.occurrence.role == "occurrence"`),

never descendants (children of a turn), never automation controllers (a run
with `vars._meta.automation`), never the runtime's internal runs (dunder
workflow ids such as `__session_memory__`), never the legacy scheduled wrapper
runs, and — unless asked — never draft-test runs.

Legacy scheduled wrappers are the gateway's pre-automation "scheduled run"
roots: `vars._meta.schedule.kind == "scheduled_run"` (indexed as role
`legacy_schedule`) or a workflow id with the gateway's own `scheduled:` prefix.
They are controllers of the old kind, so they are not turns; the prefix is a
structural naming convention of the gateway, not a text heuristic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .core.models import RunState
from .core.run_attribution import automation_index_fields, is_turn_root
from .core.run_lifecycle import is_draft_lifecycle, run_lifecycle_index_fields

LEGACY_SCHEDULED_WORKFLOW_PREFIX = "scheduled:"


def _iso_ms(raw: Any) -> Optional[int]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return int(datetime.fromisoformat(text).timestamp() * 1000)
    except Exception:
        return None


def _order_key(row: Dict[str, Any]) -> tuple:
    # (parsed ms, raw ISO string): the raw string keeps sub-ms precision as the
    # tiebreak so same-millisecond turns keep their true order.
    for key in ("created_at", "updated_at"):
        raw = row.get(key)
        ms = _iso_ms(raw)
        if ms is not None:
            return (float(ms), str(raw or ""), str(row.get("run_id") or ""))
    return (0.0, "", str(row.get("run_id") or ""))


def is_internal_workflow_id(workflow_id: Any) -> bool:
    """The runtime's RESERVED dunder ids (`__session_memory__`): both start
    AND end with `__` (tenant catalog ids such as `__catalog__v2__...@v:flow`
    only start with it and are ordinary turns)."""
    wid = str(workflow_id or "")
    return wid.startswith("__") and wid.endswith("__")


def is_legacy_scheduled(row: Dict[str, Any]) -> bool:
    return row.get("role") == "legacy_schedule" or str(row.get("workflow_id") or "").startswith(
        LEGACY_SCHEDULED_WORKFLOW_PREFIX
    )


def _run_row(run: RunState) -> Dict[str, Any]:
    """An index-shaped row for a RunState (scan fallback for index-less stores)."""
    return {
        "run_id": str(run.run_id),
        "workflow_id": str(run.workflow_id or ""),
        "status": str(getattr(run.status, "value", run.status)),
        "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
        "session_id": str(run.session_id) if run.session_id else None,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        **run_lifecycle_index_fields(run.vars),
        **automation_index_fields(run.vars, run_id=str(run.run_id)),
    }


def _candidate_rows(run_store: Any, session_id: str, fetch: int) -> tuple[List[Dict[str, Any]], Dict[str, RunState]]:
    list_run_index = getattr(run_store, "list_run_index", None)
    if callable(list_run_index):
        return list(list_run_index(session_id=session_id, root_only=True, limit=fetch) or []), {}
    # Stores without a run index (older/duck-typed stores): scan full runs.
    list_runs = getattr(run_store, "list_runs", None)
    if not callable(list_runs):
        raise TypeError(f"run store {type(run_store).__name__} can list neither run index rows nor runs")
    loaded: Dict[str, RunState] = {}
    rows: List[Dict[str, Any]] = []
    for run in list_runs(limit=fetch) or []:
        if str(getattr(run, "session_id", "") or "").strip() != session_id:
            continue
        row = _run_row(run)
        if not is_turn_root(parent_run_id=row["parent_run_id"], role=row["role"]):
            continue
        loaded[row["run_id"]] = run
        rows.append(row)
    return rows, loaded


class OccurrenceNotInSession(LookupError):
    """`through_occurrence=N` names an occurrence the session does not hold."""

    reason_code = "occurrence_not_found"


def _occurrence_cutoff(
    run_store: Any, session_id: str, automation_id: Optional[str], index: int
) -> tuple:
    """The order key of occurrence `index` (its newest attempt) in the
    session, read from the index directly — never from a newest-first window,
    so an old occurrence of a long session is found. Raises
    `OccurrenceNotInSession` when there is none."""
    filters: Dict[str, Any] = {"session_id": session_id, "role": "occurrence", "limit": 1_000_000}
    if automation_id is not None:
        filters["automation_id"] = str(automation_id)
    rows = [r for r in run_store.list_run_index(**filters) if r.get("occurrence_index") == index]
    if not rows:
        raise OccurrenceNotInSession(
            f"session {session_id} has no occurrence {index}"
            + (f" of automation {automation_id}" if automation_id is not None else "")
        )
    return max(_order_key(r) for r in rows)


def _is_chat_like(run: RunState, row: Dict[str, Any]) -> bool:
    if row.get("role") == "occurrence":
        return True  # occurrences count as chat (contract E)
    vars_obj = run.vars if isinstance(run.vars, dict) else {}
    ctx = vars_obj.get("context")
    return isinstance(ctx, dict) and isinstance(ctx.get("messages"), list)


def select_session_turns(
    run_store: Any,
    session_id: str,
    *,
    include_occurrences: bool = True,
    until_ms: Optional[int] = None,
    automation_id: Optional[str] = None,
    through_occurrence: Optional[int] = None,
    include_drafts: bool = False,
    limit: int = 50,
) -> List[RunState]:
    """The session's turns, chronological (oldest first), the newest `limit`.

    - `include_occurrences`: include automation occurrence runs (default).
    - `automation_id`: keep only that automation's occurrences (other turns stay).
    - `through_occurrence=N`: drop occurrences after N and every turn created
      after occurrence N (the history as it stood when N ran). Occurrence N
      is looked up in the index directly, however old; a session without it
      raises `OccurrenceNotInSession`.
    - `until_ms`: drop turns created after this epoch-millisecond instant.
    - A retried occurrence contributes one turn: its newest attempt.
    - When the session holds chat-like turns (a `context.messages` list, or an
      occurrence), other root runs (plain workflow runs) are left out, as the
      history bundle always did.
    """
    sid = str(session_id or "").strip()
    lim = int(limit)
    if not sid or lim <= 0:
        return []

    # A time bound (`through_occurrence`, `until_ms`) must be applied BEFORE
    # any newest-first window, or an old bound selects the wrong turns in a
    # long session (review 44 F1): bounded reads fetch the session's whole
    # (column-only) index, unbounded reads the usual window.
    cutoff: Optional[tuple] = None
    if through_occurrence is not None:
        cutoff = _occurrence_cutoff(run_store, sid, automation_id, int(through_occurrence))
    bounded_read = through_occurrence is not None or until_ms is not None
    rows, preloaded = _candidate_rows(run_store, sid, 1_000_000 if bounded_read else max(1000, lim * 5))

    by_id: Dict[str, Dict[str, Any]] = {}
    newest_attempt: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("run_id") or "").strip()
        if not rid or rid in by_id:
            continue
        if str(row.get("session_id") or "").strip() != sid:
            continue
        role = row.get("role")
        if not is_turn_root(parent_run_id=row.get("parent_run_id"), role=role):
            continue
        if is_internal_workflow_id(row.get("workflow_id")) or is_legacy_scheduled(row):
            continue
        if not include_drafts and is_draft_lifecycle(row.get("run_lifecycle")):
            continue
        if role == "occurrence":
            if not include_occurrences:
                continue
            if automation_id is not None and row.get("automation_id") != str(automation_id):
                continue
            # One turn per logical occurrence: its newest attempt (attempts are
            # created in order, so the newest creation is the highest attempt).
            key = (row.get("automation_id"), row.get("occurrence_index"))
            prior = newest_attempt.get(key)
            if prior is not None:
                if _order_key(prior) >= _order_key(row):
                    continue
                by_id.pop(str(prior["run_id"]), None)
            newest_attempt[key] = row
        by_id[rid] = row

    selected = sorted(by_id.values(), key=_order_key)

    if cutoff is not None:
        target = int(through_occurrence)  # type: ignore[arg-type]
        selected = [
            r for r in selected
            if _order_key(r) <= cutoff
            and not (r.get("role") == "occurrence" and isinstance(r.get("occurrence_index"), int) and r["occurrence_index"] > target)
        ]

    if until_ms is not None:
        bounded = []
        for row in selected:
            ms = _iso_ms(row.get("created_at"))
            if ms is None:
                ms = _iso_ms(row.get("updated_at"))
            if ms is None or ms <= until_ms:
                bounded.append(row)
        selected = bounded

    # Load a bounded newest window of full RunStates (the chat-like preference
    # below needs vars; the result keeps the newest `limit` anyway). The 3x
    # over-fetch absorbs turns the preference drops.
    window = selected[-max(lim * 3, lim + 8):]
    loaded: List[tuple[RunState, Dict[str, Any]]] = []
    for row in window:
        rid = str(row["run_id"])
        run = preloaded.get(rid)
        if run is None:
            run = run_store.load(rid)
        if run is None:
            continue
        loaded.append((run, row))

    if any(_is_chat_like(run, row) for run, row in loaded):
        loaded = [(run, row) for run, row in loaded if _is_chat_like(run, row)]
    return [run for run, _row in loaded][-lim:]


__all__ = [
    "LEGACY_SCHEDULED_WORKFLOW_PREFIX",
    "OccurrenceNotInSession",
    "is_internal_workflow_id",
    "is_legacy_scheduled",
    "select_session_turns",
]
