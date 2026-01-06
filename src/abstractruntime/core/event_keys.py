"""abstractruntime.core.event_keys

Durable event key conventions.

Why this exists:
- `WAIT_EVENT` needs a stable `wait_key` that external hosts can compute.
- Visual editors and other hosts (AbstractCode, servers) must agree on the same
  key format without importing UI-specific code.

We keep this module dependency-light (stdlib only).
"""

from __future__ import annotations

from typing import Optional


def build_event_wait_key(
    *,
    scope: str,
    name: str,
    session_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Build a durable wait_key for event-driven workflows.

    Format:
        evt:{scope}:{scope_id}:{name}

    Scopes:
    - session: `scope_id` is the workflow instance/session identifier (recommended default)
    - workflow: `scope_id` is the workflow_id
    - run: `scope_id` is the run_id
    - global: `scope_id` is the literal string "global"
    """
    scope_norm = str(scope or "session").strip().lower()
    name_norm = str(name or "").strip()
    if not name_norm:
        raise ValueError("event name is required")

    scope_id: Optional[str]
    if scope_norm == "session":
        scope_id = str(session_id or "").strip() if session_id is not None else ""
    elif scope_norm == "workflow":
        scope_id = str(workflow_id or "").strip() if workflow_id is not None else ""
    elif scope_norm == "run":
        scope_id = str(run_id or "").strip() if run_id is not None else ""
    elif scope_norm == "global":
        scope_id = "global"
    else:
        raise ValueError(f"unknown event scope: {scope!r}")

    if not scope_id:
        raise ValueError(f"missing scope id for scope={scope_norm!r}")

    return f"evt:{scope_norm}:{scope_id}:{name_norm}"





