"""MemAct / Active Memory node adapters.

These nodes mutate `run.vars["_runtime"]["active_memory"]` durably so pause/resume
works and workflow authors can inspect the resulting memory blocks.
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


def create_memact_compose_node_handler(
    *,
    node_id: str,
    next_node: Optional[str],
    data_aware_handler: Optional[Callable[[Any], Any]],
    flow: Any,
) -> Callable:
    """Create a handler for `memact_compose` visual nodes."""
    from abstractruntime.core.models import StepPlan
    from ..compiler import _sync_effect_results_to_node_outputs

    def handler(run: Any, ctx: Any) -> "StepPlan":
        del ctx
        if flow is not None and hasattr(flow, "_node_outputs") and hasattr(flow, "_data_edge_map"):
            _sync_effect_results_to_node_outputs(run, flow)

        last_output = run.vars.get("_last_output", {})
        resolved = data_aware_handler(last_output) if callable(data_aware_handler) else {}
        payload = resolved if isinstance(resolved, dict) else {}

        kg_result = payload.get("kg_result")
        if kg_result is None:
            kg_result = payload.get("raw") if "raw" in payload else None
        if kg_result is None:
            kg_result = payload.get("result") if "result" in payload else None

        stimulus = payload.get("stimulus")
        if stimulus is None:
            stimulus = payload.get("query_text") if "query_text" in payload else ""
        stimulus_text = str(stimulus or "").strip()

        marker = payload.get("marker")
        marker_text = str(marker or "KG:").strip() or "KG:"

        max_items_raw = payload.get("max_items")
        max_items: Optional[int] = None
        if max_items_raw is not None and not isinstance(max_items_raw, bool):
            try:
                mi = int(float(max_items_raw))
            except Exception:
                mi = None
            if isinstance(mi, int) and mi > 0:
                max_items = mi

        out: Dict[str, Any]
        if not isinstance(kg_result, dict):
            out = {
                "ok": False,
                "error": "memact_compose requires a dict kg_result (connect memory_kg_query.raw or output)",
                "delta": {},
                "trace": {},
                "active_memory": {},
                "memact_system_prompt": "",
                "memact_blocks": [],
            }
        else:
            from abstractruntime.memory.active_memory import render_memact_blocks, render_memact_system_prompt
            from abstractruntime.memory.memact_composer import compose_memact_current_context_from_kg_result

            composed = compose_memact_current_context_from_kg_result(
                run.vars,
                kg_result=kg_result,
                stimulus=stimulus_text,
                marker=marker_text,
                max_items=max_items,
            )
            out = {
                "ok": bool(composed.get("ok")),
                "delta": composed.get("delta") if isinstance(composed.get("delta"), dict) else {},
                "trace": composed.get("trace") if isinstance(composed.get("trace"), dict) else {},
                "active_memory": composed.get("active_memory") if isinstance(composed.get("active_memory"), dict) else {},
                "memact_system_prompt": render_memact_system_prompt(run.vars),
                "memact_blocks": render_memact_blocks(run.vars),
            }

        _persist_node_output(run.vars, node_id, out)

        # Standard pipeline semantics: this node becomes the `_last_output`.
        run.vars["_last_output"] = dict(out)
        try:
            if flow is not None and hasattr(flow, "_node_outputs") and isinstance(flow._node_outputs, dict):  # type: ignore[attr-defined]
                flow._node_outputs[node_id] = out  # type: ignore[attr-defined]
        except Exception:
            pass

        if next_node:
            return StepPlan(node_id=node_id, next_node=next_node)
        return StepPlan(node_id=node_id, complete_output=out)

    return handler

