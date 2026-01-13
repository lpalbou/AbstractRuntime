"""Context node adapters (active context helpers).

These nodes mutate `run.vars["context"]` durably so pause/resume works and other
nodes (Agent/Subflow/LLM_CALL with use_context) can read the updated history.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def _persist_node_output(run_vars: Dict[str, Any], node_id: str, value: Dict[str, Any]) -> None:
    temp = run_vars.get("_temp")
    if not isinstance(temp, dict):
        temp = {}
        run_vars["_temp"] = temp
    persisted = temp.get("node_outputs")
    if not isinstance(persisted, dict):
        persisted = {}
        temp["node_outputs"] = persisted
    persisted[node_id] = value


def create_add_message_node_handler(
    *,
    node_id: str,
    next_node: Optional[str],
    data_aware_handler: Optional[Callable[[Any], Any]],
    flow: Any,
) -> Callable:
    """Create a handler for `add_message` visual nodes.

    Contract:
    - Inputs:
      - `role`: message role
      - `content`: message content
    - Behavior:
      - builds a canonical message object (timestamp + metadata.message_id)
      - appends it to `run.vars.context.messages` (durable)
      - does NOT clobber `_last_output` (pass-through)
    """
    from abstractruntime.core.models import StepPlan
    from ..compiler import _sync_effect_results_to_node_outputs

    def handler(run: Any, ctx: Any) -> "StepPlan":
        del ctx
        if flow is not None and hasattr(flow, "_node_outputs") and hasattr(flow, "_data_edge_map"):
            _sync_effect_results_to_node_outputs(run, flow)

        last_output = run.vars.get("_last_output", {})
        resolved = data_aware_handler(last_output) if callable(data_aware_handler) else {}
        payload = resolved if isinstance(resolved, dict) else {}

        message = payload.get("message")
        if not isinstance(message, dict):
            # Fallback: build a minimal message object if the visual handler did not.
            role = payload.get("role")
            role_str = role if isinstance(role, str) else str(role or "user")
            content = payload.get("content")
            content_str = content if isinstance(content, str) else str(content or "")

            try:
                from datetime import datetime, timezone

                timestamp = datetime.now(timezone.utc).isoformat()
            except Exception:
                from datetime import datetime

                timestamp = datetime.utcnow().isoformat() + "Z"

            import uuid

            message = {
                "role": role_str,
                "content": content_str,
                "timestamp": timestamp,
                "metadata": {"message_id": f"msg_{uuid.uuid4().hex}"},
            }

        ctx_ns = run.vars.get("context")
        if not isinstance(ctx_ns, dict):
            ctx_ns = {}
            run.vars["context"] = ctx_ns

        msgs = ctx_ns.get("messages")
        if not isinstance(msgs, list):
            msgs = []
            ctx_ns["messages"] = msgs

        msgs.append(message)

        # Persist node outputs for pause/resume and data-edge reads.
        _persist_node_output(
            run.vars,
            node_id,
            {
                "message": message,
                "context": ctx_ns,
                "task": ctx_ns.get("task") if isinstance(ctx_ns.get("task"), str) else str(ctx_ns.get("task") or ""),
                "messages": list(msgs),
            },
        )

        if next_node:
            return StepPlan(node_id=node_id, next_node=next_node)
        return StepPlan(node_id=node_id, complete_output={"success": True, "message": message})

    return handler

