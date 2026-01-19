"""KG/spans → MemAct Active Memory composition helpers (v0).

This module provides deterministic, bounded mapping from durable memory sources
(starting with the temporal KG) into MemAct's structured Active Memory blocks.

Design goals:
- deterministic: stable ordering + stable statements
- bounded: honors packet budgets and optional max_items caps
- auditable: returns a JSON-safe trace describing inputs + selections
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _normalize_marker(marker: Any) -> str:
    s = marker if isinstance(marker, str) else str(marker or "")
    s2 = s.strip()
    if not s2:
        return "KG:"
    return s2


def _has_marker(text: Any, marker: str) -> bool:
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    # Case-insensitive prefix match.
    return s.casefold().startswith(marker.casefold())


def _packet_statement(pkt: Dict[str, Any]) -> str:
    stmt = pkt.get("statement")
    if isinstance(stmt, str) and stmt.strip():
        return stmt.strip()
    subj = pkt.get("subject")
    pred = pkt.get("predicate")
    obj = pkt.get("object")
    parts = [subj, pred, obj]
    if all(isinstance(p, str) and p.strip() for p in parts):
        return f"{subj.strip()} —{pred.strip()}→ {obj.strip()}"
    return ""


def _render_memact_context_line(pkt: Dict[str, Any], *, marker: str) -> str:
    stmt = _packet_statement(pkt)
    if not stmt:
        return ""
    marker2 = _normalize_marker(marker)

    # Keep the injected statement stable for dedupe (avoid including retrieval_score).
    suffix_parts: List[str] = []
    span_id = pkt.get("span_id")
    if isinstance(span_id, str) and span_id.strip():
        suffix_parts.append(f"span:{span_id.strip()}")
    writer_workflow_id = pkt.get("writer_workflow_id")
    if isinstance(writer_workflow_id, str) and writer_workflow_id.strip():
        suffix_parts.append(f"wf:{writer_workflow_id.strip()}")

    suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
    return f"{marker2} {stmt}{suffix}".strip()


def _packets_from_kg_result(kg_result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    packets = _as_dict_list(kg_result.get("packets"))
    if packets:
        return packets, "kg_result.packets"
    # Fallback: derive packets from raw items (no packing / max_input_tokens=0).
    items = _as_dict_list(kg_result.get("items"))
    if not items:
        return [], "none"
    try:
        from .kg_packets import packetize_assertions

        return packetize_assertions(items), "packetize_assertions(items)"
    except Exception:
        return [], "packetize_failed"


def compose_memact_current_context_from_kg_result(
    vars: Dict[str, Any],
    *,
    kg_result: Dict[str, Any],
    stimulus: str,
    marker: str = "KG:",
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply a KG-derived reconstruction of MemAct CURRENT CONTEXT.

    Behavior (v0):
    - Remove any existing CURRENT CONTEXT entries prefixed with `marker`
      (treat them as "previous KG reconstruction").
    - Add new entries derived from the KG packets (or packetized items fallback).
    - Returns `{ok, delta, trace}` where `delta` matches MemAct envelope fields.
    """
    if not isinstance(vars, dict):
        return {"ok": False, "error": "vars must be a dict"}
    if not isinstance(kg_result, dict):
        return {"ok": False, "error": "kg_result must be a dict"}

    try:
        from .active_memory import ensure_memact_memory, apply_memact_envelope
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": f"memact active_memory unavailable: {e}"}

    mem = ensure_memact_memory(vars)
    marker2 = _normalize_marker(marker)

    existing_ctx = mem.get("current_context")
    existing_ctx_list = existing_ctx if isinstance(existing_ctx, list) else []
    removed: List[str] = []
    for rec in existing_ctx_list:
        if not isinstance(rec, dict):
            continue
        text = rec.get("text")
        if _has_marker(text, marker2):
            removed.append(str(text))

    packets, packet_source = _packets_from_kg_result(kg_result)
    if isinstance(max_items, int) and max_items > 0 and len(packets) > max_items:
        packets = packets[: int(max_items)]

    added: List[str] = []
    selected: List[Dict[str, Any]] = []
    for pkt in packets:
        line = _render_memact_context_line(pkt, marker=marker2)
        if not line:
            continue
        added.append(line)
        if len(selected) < 50:
            selected.append(
                {
                    "statement": _packet_statement(pkt),
                    "span_id": pkt.get("span_id"),
                    "observed_at": pkt.get("observed_at"),
                    "retrieval_score": pkt.get("retrieval_score"),
                }
            )

    delta: Dict[str, Any] = {"current_context": {"added": added, "removed": removed}}
    applied = apply_memact_envelope(vars, envelope=delta)

    trace: Dict[str, Any] = {
        "stimulus": str(stimulus or "").strip(),
        "marker": marker2,
        "packet_source": packet_source,
        "kg": {
            "ok": bool(kg_result.get("ok")) if "ok" in kg_result else None,
            "count": kg_result.get("count"),
            "packed_count": kg_result.get("packed_count"),
            "estimated_tokens": kg_result.get("estimated_tokens"),
            "dropped": kg_result.get("dropped"),
            "effort": kg_result.get("effort"),
            "warnings": kg_result.get("warnings"),
        },
        "delta_preview": {"removed": len(removed), "added": len(added)},
        "selected": selected,
        "applied": applied,
    }
    ok = bool(applied.get("ok")) if isinstance(applied, dict) else False
    return {"ok": ok, "delta": delta, "trace": trace, "active_memory": mem}

