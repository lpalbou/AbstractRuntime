"""Authoring-time reroute lowering (VisualFlow JSON -> internal VisualFlow).

A reroute ("relay") node is visual sugar from the editor: a small dot with one
input and one output that lets a source feed several inputs along a tidy path.
It has no behaviour. Before anything else reads the flow, every chain
`source -> reroute (-> reroute ...) -> target` is rewritten into
`source -> target` and the reroute nodes are removed, as if they had never been
drawn (Unreal Engine's reroute node semantics).

Rules:
- A reroute's incoming edge (at most one) names its source; a chain resolves to
  the first node upstream that is not a reroute.
- A reroute with no source feeds nothing: its outgoing edges are dropped, so
  the targets keep their own defaults (the editor warns about it).
- A cycle made only of reroutes is refused loudly.
- Per-path values (`inputRouteOverrides`) that name a reroute as their source
  are rewritten to the resolved source, so multi-entry lowering (which runs
  next) sees the graph without reroutes.
- Reroutes carry data wires only for now: an execution wire through a reroute
  is refused loudly (entry routes are keyed by the execution wire's source, so
  it would need its own rewrite rule).

This module must remain stdlib-only.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from .models import VisualEdge, VisualFlow, VisualNode

REROUTE_NODE_TYPE = "reroute"


def _type_of(node: VisualNode) -> str:
    t = node.type
    return str(t.value if hasattr(t, "value") else t)


def lower_reroutes(visual: VisualFlow) -> VisualFlow:
    """Remove reroute nodes, wiring each source straight to the reroutes' targets."""
    reroute_ids = {n.id for n in visual.nodes if _type_of(n) == REROUTE_NODE_TYPE}
    if not reroute_ids:
        return visual

    exec_outputs = set()
    for n in visual.nodes:
        for pin in (n.data or {}).get("outputs") or []:
            if isinstance(pin, dict) and pin.get("type") == "execution":
                exec_outputs.add((n.id, pin.get("id")))

    def is_exec(e: VisualEdge) -> bool:
        return (e.source, e.sourceHandle) in exec_outputs or e.sourceHandle.startswith("exec") or e.targetHandle.startswith("exec")

    incoming: Dict[str, VisualEdge] = {}
    for e in visual.edges:
        touches = e.source in reroute_ids or e.target in reroute_ids
        if touches and (
            is_exec(e)
            or (e.source in reroute_ids and e.sourceHandle != "out")
            or (e.target in reroute_ids and e.targetHandle != "in")
        ):
            raise ValueError(
                f"Reroute wire {e.source}.{e.sourceHandle} -> {e.target}.{e.targetHandle}: a reroute has one data "
                "input 'in' and one data output 'out' (execution wires through a reroute are not supported)."
            )
        if e.target in reroute_ids:
            if e.target in incoming:
                raise ValueError(f"Reroute '{e.target}' has more than one incoming wire; a reroute takes one source.")
            incoming[e.target] = e

    def resolve(node_id: str, handle: str) -> Optional[Tuple[str, str]]:
        seen: List[str] = []
        while node_id in reroute_ids:
            if node_id in seen:
                raise ValueError(f"Reroutes form a cycle: {' -> '.join(seen + [node_id])}.")
            seen.append(node_id)
            edge = incoming.get(node_id)
            if edge is None:
                return None
            node_id, handle = edge.source, edge.sourceHandle
        return node_id, handle

    edges: List[VisualEdge] = []
    for e in visual.edges:
        if e.target in reroute_ids:
            continue
        if e.source in reroute_ids:
            src = resolve(e.source, e.sourceHandle)
            if src is None:
                continue
            e = replace(e, source=src[0], sourceHandle=src[1])
        edges.append(e)

    def rewrite_ref(ref: Any) -> Any:
        if not isinstance(ref, dict):
            return ref
        sid = ref.get("sourceNodeId") or ref.get("source")
        if not isinstance(sid, str) or sid not in reroute_ids:
            return ref
        handle = ref.get("sourceHandle") or ref.get("handle") or "out"
        src = resolve(sid, str(handle))
        if src is None:
            return None
        out = {k: v for k, v in ref.items() if k not in ("source", "handle")}
        out["sourceNodeId"], out["sourceHandle"] = src
        return out

    nodes: List[VisualNode] = []
    for n in visual.nodes:
        if n.id in reroute_ids:
            continue
        data = n.data if isinstance(n.data, dict) else {}
        overrides = data.get("inputRouteOverrides")
        if isinstance(overrides, dict):
            new_overrides: Dict[str, Any] = {}
            for pin, per_route in overrides.items():
                if not isinstance(per_route, dict):
                    continue
                kept = {rk: r for rk, r in ((rk, rewrite_ref(ref)) for rk, ref in per_route.items()) if r is not None}
                if kept:
                    new_overrides[pin] = kept
            n = replace(n, data={**data, "inputRouteOverrides": new_overrides})
        nodes.append(n)

    return replace(visual, nodes=nodes, edges=edges)
