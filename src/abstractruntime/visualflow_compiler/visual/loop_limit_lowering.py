"""Authoring-time loop-limit lowering (VisualFlow JSON -> internal VisualFlow).

An execution wire may carry a limit, `edge.data.loopLimit = {"max": N}` (the editor draws it as
a ↺N badge on a wire that makes execution repeat; operator 2026-10-14: "the loop can only happen
X times.. and then it should get out of the loop and resume where it was"). Before anything else
reads the flow, every such wire S.h -> T is rewritten into

    S.h -> L.exec-in,   L.again -> T,   [L.limit_reached -> X]

where L is a `loop_limit` node named after the wire and X is where execution resumes once the
limit is reached: when S is an If/Else, the target of its OTHER branch (the flow continues forward
from the If/Else as if it had chosen the other way); for any other source, an unwired other
branch, or an other branch that is itself a limited wire (no way forward), there is no X: at the
limit the path ends (inside a loop body the runtime returns to the loop; at the top level the run
ends; the editor's run checks warn about it).

Per-path values (`inputRouteOverrides`) and the entry list (`entryRoutes`) of T keyed on the
rewritten wire (`S::h`) are re-keyed to the new entry (`L::again`), and X's entry list (when it
has one) gains the new way in, so multi-entry lowering (which runs next) sees a consistent graph.

This module must remain stdlib-only.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from .models import VisualEdge, VisualFlow, VisualNode

LOOP_LIMIT_NODE_TYPE = "loop_limit"
OTHER_IF_BRANCH = {"true": "false", "false": "true"}


def _type_of(node: VisualNode) -> str:
    t = node.type
    return str(t.value if hasattr(t, "value") else t)


def wire_loop_limit(edge: VisualEdge) -> Optional[int]:
    """The limit a wire carries, or None."""
    raw = edge.data.get("loopLimit") if isinstance(edge.data, dict) else None
    if not isinstance(raw, dict):
        return None
    mx = raw.get("max")
    if isinstance(mx, bool) or not isinstance(mx, (int, float)):
        raise ValueError(f"Wire {edge.source}.{edge.sourceHandle} -> {edge.target}: loopLimit.max must be a number, got {mx!r}")
    return max(0, int(mx))


def _route_key(source: str, handle: str) -> str:
    return f"{source}::{handle or 'exec-out'}"


def lower_loop_limits(visual: VisualFlow) -> VisualFlow:
    """Turn every wire limit into a loop_limit node on that wire."""
    limited = [(i, e, wire_loop_limit(e)) for i, e in enumerate(visual.edges)]
    limited = [(i, e, mx) for i, e, mx in limited if mx is not None]
    if not limited:
        return visual

    by_id = {n.id: n for n in visual.nodes}

    nodes: List[VisualNode] = list(visual.nodes)
    edges: List[VisualEdge] = list(visual.edges)
    rekey: Dict[str, Dict[str, str]] = {}  # target node id -> {old route key: new route key}
    new_entries: Dict[str, List[Dict[str, str]]] = {}  # exit target id -> entries to add to its entryRoutes
    limit_ids: List[Tuple[VisualEdge, str]] = []  # (original wire, its limit node id)
    for index, e, mx in limited:
        if e.targetHandle != "exec-in":
            raise ValueError(
                f"Wire {e.source}.{e.sourceHandle} -> {e.target}.{e.targetHandle} carries a loop limit but is not an execution wire."
            )
        limit_id = f"loop-limit:{e.source}.{e.sourceHandle}->{e.target}"
        nodes.append(
            VisualNode(
                id=limit_id,
                type=LOOP_LIMIT_NODE_TYPE,
                data={
                    "nodeType": LOOP_LIMIT_NODE_TYPE,
                    "label": f"Loop limit {mx}",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "max", "label": "max", "type": "number"},
                    ],
                    "outputs": [
                        {"id": "again", "label": "again", "type": "execution"},
                        {"id": "limit_reached", "label": "limit reached", "type": "execution"},
                        {"id": "count", "label": "count", "type": "number"},
                    ],
                    "pinDefaults": {"max": mx},
                },
            )
        )
        plain = {k: v for k, v in e.data.items() if k != "loopLimit"}
        edges[index] = VisualEdge(source=e.source, target=limit_id, sourceHandle=e.sourceHandle, targetHandle="exec-in", data=plain)
        edges.append(VisualEdge(source=limit_id, target=e.target, sourceHandle="again", targetHandle=e.targetHandle))
        rekey.setdefault(e.target, {})[_route_key(e.source, e.sourceHandle)] = _route_key(limit_id, "again")
        limit_ids.append((e, limit_id))

    # Where execution resumes at the limit: the If/Else's other branch. When that branch is itself a
    # limited wire there is no way forward (both branches loop back; each limit would hand over to
    # the other for ever), so the path ends at the limit instead; the editor warns about it.
    limited_wires = {(e.source, e.sourceHandle) for e, _ in limit_ids}
    for e, limit_id in limit_ids:
        source = by_id.get(e.source)
        if source is None or _type_of(source) != "if":
            continue
        other = OTHER_IF_BRANCH.get(e.sourceHandle)
        if not other or (e.source, other) in limited_wires:
            continue
        exits = [x for x in visual.edges if x.source == e.source and x.sourceHandle == other and x.targetHandle == "exec-in"]
        if not exits:
            continue
        edges.append(VisualEdge(source=limit_id, target=exits[0].target, sourceHandle="limit_reached", targetHandle="exec-in"))
        new_entries.setdefault(exits[0].target, []).append(
            {"key": _route_key(limit_id, "limit_reached"), "sourceNodeId": limit_id, "sourceHandle": "limit_reached"}
        )

    lowered: List[VisualNode] = []
    for n in nodes:
        keys = rekey.get(n.id)
        data = n.data if isinstance(n.data, dict) else {}
        changed: Dict[str, Any] = {}
        overrides = data.get("inputRouteOverrides")
        if keys and isinstance(overrides, dict):
            new_overrides: Dict[str, Any] = {}
            for pin, per_route in overrides.items():
                if not isinstance(per_route, dict):
                    continue
                new_overrides[pin] = {keys.get(rk, rk): ref for rk, ref in per_route.items()}
            changed["inputRouteOverrides"] = new_overrides
        # The editor's entry list of a node entered by several wires names each way in; it must
        # match the wires after the rewrite (multi-entry lowering refuses a mismatch).
        routes = data.get("entryRoutes")
        if isinstance(routes, list) and (keys or n.id in new_entries):
            new_routes: List[Any] = []
            for r in routes:
                if isinstance(r, dict) and keys:
                    # Matched by the way in (source node and pin), which is what multi-entry
                    # lowering checks; the key is rewritten with it.
                    src = str(r.get("sourceNodeId") or r.get("source") or "")
                    handle = str(r.get("sourceHandle") or "exec-out")
                    new_key = keys.get(_route_key(src, handle))
                    if new_key:
                        limit_id = new_key.split("::", 1)[0]
                        r = {**r, "key": new_key, "sourceNodeId": limit_id, "sourceHandle": "again"}
                new_routes.append(r)
            new_routes.extend(new_entries.get(n.id, []))
            changed["entryRoutes"] = new_routes
        if changed:
            n = replace(n, data={**data, **changed})
        lowered.append(n)
    return replace(visual, nodes=lowered, edges=edges)
