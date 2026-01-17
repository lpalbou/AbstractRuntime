"""KG → Active Memory packetization (v0).

This module bridges symbolic KG assertions (subject/predicate/object + metadata)
into a bounded, LLM-friendly Active Memory block.

Goals:
- deterministic ordering + formatting
- token-budgeted packing (uses AbstractCore TokenUtils when available)
- provenance-preserving packets (span_id, writer ids, retrieval scores)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .token_budget import estimate_tokens


KG_MEMORY_PACKETS_VERSION = 0

_WS = re.compile(r"\s+")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    return _WS.sub(" ", s).strip()


def packetize_assertions(items: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert assertion dicts into compact, JSON-safe memory packets (v0)."""
    packets: list[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        subject = _clean_text(item.get("subject"))
        predicate = _clean_text(item.get("predicate"))
        obj = _clean_text(item.get("object"))
        if not (subject and predicate and obj):
            continue

        observed_at = _clean_text(item.get("observed_at"))

        # Retrieval metadata (semantic queries) is stored in attributes._retrieval.
        retrieval_score: Optional[float] = None
        attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        ret = attrs.get("_retrieval") if isinstance(attrs, dict) else None
        if isinstance(ret, dict):
            s = ret.get("score")
            if isinstance(s, (int, float)):
                retrieval_score = float(s)

        prov = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        span_id = _clean_text(prov.get("span_id"))
        writer_run_id = _clean_text(prov.get("writer_run_id"))
        writer_workflow_id = _clean_text(prov.get("writer_workflow_id"))

        scope = _clean_text(item.get("scope"))
        owner_id = _clean_text(item.get("owner_id"))

        # Stable statement surface for LLM consumption.
        statement = f"{subject} —{predicate}→ {obj}"

        key = (subject.casefold(), predicate.casefold(), obj.casefold(), observed_at)
        if key in seen:
            continue
        seen.add(key)

        pkt: Dict[str, Any] = {
            "version": KG_MEMORY_PACKETS_VERSION,
            "statement": statement,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
        }
        if observed_at:
            pkt["observed_at"] = observed_at
        if scope:
            pkt["scope"] = scope
        if owner_id:
            pkt["owner_id"] = owner_id
        if span_id:
            pkt["span_id"] = span_id
        if writer_run_id:
            pkt["writer_run_id"] = writer_run_id
        if writer_workflow_id:
            pkt["writer_workflow_id"] = writer_workflow_id
        if retrieval_score is not None:
            pkt["retrieval_score"] = retrieval_score

        packets.append(pkt)

    return packets


def pack_active_memory_text(
    packets: Iterable[Dict[str, Any]],
    *,
    scope: str,
    max_input_tokens: int,
    model: Optional[str] = None,
    title: str = "KG ACTIVE MEMORY",
    include_scores: bool = True,
) -> Tuple[str, List[Dict[str, Any]], int, int]:
    """Render a token-budgeted Active Memory block.

    Returns:
      (text, kept_packets, estimated_tokens, dropped_packets)
    """
    budget = int(max_input_tokens) if isinstance(max_input_tokens, int) else int(max_input_tokens or 0)
    if budget <= 0:
        return "", [], 0, 0

    scope2 = _clean_text(scope) or "session"

    packets_list = [p for p in packets if isinstance(p, dict)]

    header_lines = [
        f"## {title}",
        f"(scope={scope2}; newest-first; budget={budget} max_input_tokens)",
        "",
    ]
    lines = list(header_lines)
    tokens = estimate_tokens("\n".join(lines), model=model)

    kept: list[Dict[str, Any]] = []

    for pkt in packets_list:
        stmt = _clean_text(pkt.get("statement"))
        if not stmt:
            continue

        ts = _clean_text(pkt.get("observed_at"))
        prefix = f"[{ts}] " if ts else ""

        parts: list[str] = []
        span_id = _clean_text(pkt.get("span_id"))
        if span_id:
            parts.append(f"span:{span_id}")

        if include_scores:
            score = pkt.get("retrieval_score")
            if isinstance(score, (int, float)):
                parts.append(f"score:{float(score):.3f}")

        suffix = f" ({'; '.join(parts)})" if parts else ""
        line = f"- {prefix}{stmt}{suffix}"

        add = estimate_tokens("\n" + line, model=model)
        if tokens + add > budget:
            break

        tokens += add
        lines.append(line)
        kept.append(pkt)

    text = "\n".join(lines).strip() if kept else ""
    dropped_total = max(0, len(packets_list) - len(kept))
    return text, kept, int(tokens if kept else 0), int(dropped_total)
