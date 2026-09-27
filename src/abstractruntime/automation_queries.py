"""Read-side automation queries over the run index (automations contract E).

An automation IS its controller run (`automation_id == run_id`, role
`controller`); its occurrences are child runs with role `occurrence`. These
functions read the attribution columns every run store indexes
(`core.run_attribution`) and load full RunStates only for the rows they return.

v1 clients poll complete pages: there is no aggregate change cursor, so
`changed_since` is refused (`ChangedSinceUnsupported`, reason_code
`unsupported_feature`) instead of being approximated.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core.models import RunState, RunStatus, WaitReason

# Upper bound for one automation's index scan (occurrence rows, controllers).
# Rows are column-only and cheap; a bound keeps a runaway store from turning a
# summary into an unbounded walk.
_SCAN_LIMIT = 1_000_000

AUTOMATION_STATUSES = ("active", "paused", "completed", "failed", "cancelled", "archived")


class ChangedSinceUnsupported(NotImplementedError):
    """`changed_since` needs an aggregate change cursor, which v1 does not keep."""

    reason_code = "unsupported_feature"

    def __init__(self) -> None:
        super().__init__(
            "changed_since is not supported in automations v1: there is no correct aggregate "
            "change cursor across a controller and its occurrences yet; poll complete pages"
        )


class InvalidCursor(ValueError):
    reason_code = "invalid_request"


@dataclass
class Page:
    items: List[Dict[str, Any]] = field(default_factory=list)
    next_cursor: Optional[str] = None


def _list_index(run_store: Any, **filters: Any) -> List[Dict[str, Any]]:
    # Direct call: a store without the attribution-aware index fails loudly.
    return run_store.list_run_index(limit=_SCAN_LIMIT, **filters)


def latest_occurrence(run_store: Any, automation_id: str) -> Optional[Dict[str, Any]]:
    """The index row of the automation's highest-numbered occurrence (its
    newest attempt when retried), or None when it has not fired yet."""
    rows = _list_index(run_store, automation_id=str(automation_id), role="occurrence")
    best: Optional[Dict[str, Any]] = None
    best_key: Any = None
    for row in rows:
        index = row.get("occurrence_index")
        key = (index if isinstance(index, int) else -1, str(row.get("created_at") or ""), str(row.get("run_id") or ""))
        if best_key is None or key > best_key:
            best, best_key = row, key
    return best


def automation_status(controller: RunState) -> str:
    """Contract A status: archived > failed > completed > paused > active."""
    meta = (controller.vars or {}).get("_meta") or {}
    definition = meta.get("automation") if isinstance(meta, dict) else None
    if isinstance(definition, dict) and definition.get("archived_at"):
        return "archived"
    if controller.status == RunStatus.FAILED:
        return "failed"
    if controller.status == RunStatus.COMPLETED:
        return "completed"
    if controller.status == RunStatus.CANCELLED:
        return "cancelled"
    if _automation_state(controller).get("paused") is True:
        return "paused"
    return "active"


def _automation_state(controller: RunState) -> Dict[str, Any]:
    runtime_ns = (controller.vars or {}).get("_runtime")
    state = runtime_ns.get("automation") if isinstance(runtime_ns, dict) else None
    return state if isinstance(state, dict) else {}


def _wake_until(controller: RunState) -> Optional[str]:
    """The deadline of the controller's own wake wait, if it is parked on one."""
    waiting = controller.waiting
    if controller.status != RunStatus.WAITING or waiting is None:
        return None
    if waiting.reason != WaitReason.EVENT or waiting.wait_key != f"automation:{controller.run_id}:wake":
        return None
    return waiting.until or None


def automation_summary(controller: RunState) -> Dict[str, Any]:
    """Summary fields for one controller run (what `list_automations` returns).

    `next_fire_at` is the deadline of the controller's wake wait while no
    occurrence is pending (the next scheduled tick); `retry_at` the same
    deadline while an occurrence is in retry backoff. `occurrence_count` counts
    admitted occurrences (`_runtime.automation.next_index - 1`), manual ones
    included; a controller that has not ticked yet has the contract's initial
    state (no occurrence).
    """
    meta = (controller.vars or {}).get("_meta") or {}
    definition = meta.get("automation") if isinstance(meta, dict) else None
    if not isinstance(definition, dict):
        raise ValueError(f"run {controller.run_id} is not an automation controller (no vars._meta.automation)")
    state = _automation_state(controller)
    pending = state.get("pending_occurrence")
    until = _wake_until(controller)
    in_backoff = isinstance(pending, dict) and pending.get("phase") == "backoff"
    context = definition.get("context") if isinstance(definition.get("context"), dict) else {}
    next_index = state.get("next_index", 1)
    return {
        "automation_id": controller.run_id,
        "title": definition.get("title"),
        "status": automation_status(controller),
        "revision": definition.get("revision"),
        "trigger": definition.get("trigger"),
        "context_mode": context.get("mode", "independent"),
        "target": definition.get("target"),
        "session_id": controller.session_id,
        "workspace_root": definition.get("workspace_root"),
        "next_fire_at": until if (until and pending is None) else None,
        "retry_at": until if (until and in_backoff) else None,
        "occurrence_count": max(0, int(next_index) - 1) if isinstance(next_index, int) else 0,
        "pending_occurrence": pending if isinstance(pending, dict) else None,
        "last_outcome": state.get("last_outcome"),
        "archived_at": definition.get("archived_at"),
        "created_at": controller.created_at,
        "updated_at": controller.updated_at,
    }


def _encode_cursor(row: Dict[str, Any]) -> str:
    raw = json.dumps([str(row.get("created_at") or ""), str(row.get("run_id") or "")], separators=(",", ":"))
    return "auto1:" + base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str) -> tuple:
    try:
        if not cursor.startswith("auto1:"):
            raise ValueError("prefix")
        created_at, run_id = json.loads(base64.urlsafe_b64decode(cursor[len("auto1:"):].encode("ascii")))
        return (str(created_at), str(run_id))
    except Exception as exc:
        raise InvalidCursor(f"invalid automations cursor {cursor!r}") from exc


def list_automations(
    run_store: Any,
    *,
    status: Any = None,
    cursor: Optional[str] = None,
    limit: int = 50,
    changed_since: Optional[str] = None,
) -> Page:
    """Automation summaries, newest first, paged with an opaque restart-stable
    keyset cursor (`(created_at, run_id)` of the last item). Archived
    automations stay listable. `status` filters on the summary status (a value,
    comma-separated values or a list)."""
    if changed_since is not None:
        raise ChangedSinceUnsupported()
    from .core.run_attribution import filter_values

    wanted = filter_values(status)
    lim = max(1, int(limit or 50))
    rows = _list_index(run_store, role="controller")
    rows.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("run_id") or "")), reverse=True)
    if cursor:
        after = _decode_cursor(cursor)
        rows = [r for r in rows if (str(r.get("created_at") or ""), str(r.get("run_id") or "")) < after]

    items: List[Dict[str, Any]] = []
    last_row: Optional[Dict[str, Any]] = None
    more = False
    for row in rows:
        controller = run_store.load(str(row["run_id"]))
        if controller is None:
            continue
        summary = automation_summary(controller)
        if wanted is not None and summary["status"] not in wanted:
            continue
        if len(items) >= lim:
            more = True
            break
        items.append(summary)
        last_row = row
    return Page(items=items, next_cursor=_encode_cursor(last_row) if (more and last_row is not None) else None)


__all__ = [
    "AUTOMATION_STATUSES",
    "ChangedSinceUnsupported",
    "InvalidCursor",
    "Page",
    "automation_status",
    "automation_summary",
    "latest_occurrence",
    "list_automations",
]
