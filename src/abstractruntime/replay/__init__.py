"""One ledger replay, many views (ADR-0010, framework backlog 0993).

`replay(records_by_run, runs_by_id)` folds a run tree's ledger records and durable snapshots
into a `ReplayView` (JSON `replay_view_v1` through `to_json()`); `apply(view, records)` folds
newer records into it and leaves the change in `view.last_delta`; `apply_delta(view_json,
delta)` is the client side of that change and `diff_views(old, new)` the delta between two
views (a reconnect). Hosts serve the view; clients render it and act through the host's
commands, never by re-deriving run state from raw records.
"""

from .delta import apply_delta, diff_views, parse_cursor_token, resolve_cursor_token, token_run_names
from .executions import ExecutionTracker
from .view import PERSON_FACING_KINDS, REPLAY_VIEW_VERSION, TERMINAL_STATUSES, ReplayView, apply, replay, wait_kind, workflow_ref

__all__ = [
    "REPLAY_VIEW_VERSION",
    "TERMINAL_STATUSES",
    "PERSON_FACING_KINDS",
    "ReplayView",
    "ExecutionTracker",
    "replay",
    "apply",
    "apply_delta",
    "diff_views",
    "parse_cursor_token",
    "resolve_cursor_token",
    "token_run_names",
    "wait_kind",
    "workflow_ref",
]
