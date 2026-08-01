"""abstractruntime.core.event_keys

Durable event key conventions.

Why this exists:
- `WAIT_EVENT` needs a stable `wait_key` that external hosts can compute.
- Visual editors and other hosts (AbstractCode, servers) must agree on the same
  key format without importing UI-specific code.

We keep this module dependency-light (stdlib only).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional, Sequence


TOOL_APPROVAL_WAIT_KEY_PREFIX = "tool_approval"


def build_tool_approval_wait_key(
    *,
    run_id: str,
    node_id: Optional[str] = None,
    effect_idempotency_key: Optional[str] = None,
    effect_seq: Optional[int] = None,
    tool_calls: Optional[Sequence[Any]] = None,
) -> str:
    """Build the wait_key for ONE tool-approval instance.

    Format:
        tool_approval:{run_id}:{node_id}:{identity}

    Two properties matter, and the old key had neither:

    1. DISTINCT across approvals. The previous fallback was
       `tool_calls:{run_id}:{node_id}` -- constant for every approval round of
       an agent node, since an agent loops on the same node. Any approver that
       deduplicates by key (a sane idempotency measure; the flow editor's
       approval panel and both AbstractFlow drivers do it) answered the first
       approval and then parked the run forever on the second. Reproduced live
       on multiagent/bugfix, 2026-07-31.
    2. STABLE across a crash-replay of the SAME approval. The other branch used
       `tool_approval:{uuid4}`, which is distinct but re-randomises on replay,
       so a resumed host could not recompute the key it had handed out.

    `identity` is the effect's idempotency key when the caller can supply it:
    that is the runtime's OWN at-most-once identity (`EffectPolicy.
    idempotency_key` -- run + node + payload + the run's effect-issuance
    counter), so it is already unique per issuance and already replay-stable.
    A byte-identical tool batch re-issued at the same node gets a different key
    because the issuance counter advanced. Without it, we digest the same
    facts we can see: the issuance counter and the semantic tool calls.
    """

    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    node = str(node_id or "").strip() or "-"

    identity = str(effect_idempotency_key or "").strip()
    if not identity:
        payload = {
            "effect_seq": int(effect_seq) if isinstance(effect_seq, int) and not isinstance(effect_seq, bool) else 0,
            "tool_calls": [_tool_call_identity(tc) for tc in (tool_calls or [])],
        }
        try:
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            raw = repr(payload)
        identity = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{TOOL_APPROVAL_WAIT_KEY_PREFIX}:{rid}:{node}:{identity[:32]}"


def _tool_call_identity(tool_call: Any) -> Any:
    """The semantic half of a tool call: what it does, not who labelled it."""
    if not isinstance(tool_call, dict):
        return str(tool_call)
    arguments = tool_call.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            parsed = arguments
        arguments = parsed
    if not isinstance(arguments, (dict, list, str, int, float, bool)) and arguments is not None:
        arguments = str(arguments)
    return {
        "name": str(tool_call.get("name") or "").strip(),
        "arguments": arguments,
    }


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





