"""Authoring-time multi-entry lowering (VisualFlow JSON -> internal VisualFlow).

AbstractFlow authoring allows nodes to have multiple incoming execution edges into
their `exec-in` pin. Some *data* pins may also need to vary based on which exec
entry triggered the node (e.g. chat loops: first entry uses the initial prompt,
re-entry uses AskUser.response).

We keep the user-facing authoring graph clean and expressive by storing route
metadata on the target node (in `node.data`) and lowering it at compile time
into explicit internal junction nodes:
- `join_exec`: merges exec fan-in and emits `which` / `from`
- `path_mux`: selects a value based on `join_exec.which`

This keeps VisualFlow compilation semantics centralized in AbstractRuntime
(see framework ADR-0012), while enabling simple loop authoring in the editor.

This module must remain stdlib-only.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import VisualEdge, VisualFlow, VisualNode


def _route_key(source_node_id: str, source_handle: str) -> str:
    return f"{source_node_id}::{source_handle}"


def _is_exec_edge(e: VisualEdge) -> bool:
    return getattr(e, "targetHandle", None) == "exec-in"


def _normalize_handle(handle: Any) -> str:
    if isinstance(handle, str) and handle.strip():
        return handle.strip()
    return "exec-out"


def _as_str(v: Any) -> Optional[str]:
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def _parse_entry_routes(
    *,
    node: VisualNode,
    incoming_exec: List[VisualEdge],
) -> List[Dict[str, str]]:
    """Return normalized entry route descriptors in stable order.

    Each route dict contains:
      - key: str             (stable identifier)
      - sourceNodeId: str
      - sourceHandle: str    (exec handle, e.g. exec-out, true, false, then:0)
    """

    # 1) Attempt to use explicit authoring metadata (preferred).
    raw = node.data.get("entryRoutes") if isinstance(node.data, dict) else None
    routes: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for r in raw:
            if not isinstance(r, dict):
                continue
            src = _as_str(r.get("sourceNodeId")) or _as_str(r.get("source"))
            h = _normalize_handle(r.get("sourceHandle"))
            if not src:
                continue
            key = _as_str(r.get("key")) or _route_key(src, h)
            routes.append({"key": key, "sourceNodeId": src, "sourceHandle": h})

    # 2) Fallback: derive order from the incoming exec edges as they appear in the
    # VisualFlow edge list (deterministic for a given JSON document).
    if not routes:
        for e in incoming_exec:
            src = _as_str(getattr(e, "source", None))
            if not src:
                continue
            h = _normalize_handle(getattr(e, "sourceHandle", None))
            routes.append({"key": _route_key(src, h), "sourceNodeId": src, "sourceHandle": h})

    # Validate uniqueness.
    seen: set[str] = set()
    dups: list[str] = []
    for r in routes:
        key = r.get("key")
        if not isinstance(key, str) or not key:
            continue
        if key in seen:
            dups.append(key)
        else:
            seen.add(key)
    if dups:
        raise ValueError(f"Node '{node.id}' has duplicate entryRoutes keys: {dups}")

    # Validate alignment with incoming exec edges.
    # We validate using (sourceNodeId, sourceHandle) pairs (not keys), because keys can be customized.
    edge_pairs: List[Tuple[str, str]] = []
    for e in incoming_exec:
        src = _as_str(getattr(e, "source", None))
        if not src:
            continue
        edge_pairs.append((src, _normalize_handle(getattr(e, "sourceHandle", None))))

    # Detect duplicate exec edges (ambiguous route definition).
    counts: Dict[Tuple[str, str], int] = {}
    for p in edge_pairs:
        counts[p] = counts.get(p, 0) + 1
    dups = [p for p, n in counts.items() if n > 1]
    if dups:
        raise ValueError(
            f"Node '{node.id}' has duplicate incoming exec edges (ambiguous routes): {dups}. "
            "Ensure each exec entry is unique by (source node id, source handle)."
        )

    route_pairs = [(r["sourceNodeId"], _normalize_handle(r["sourceHandle"])) for r in routes]
    missing = [p for p in route_pairs if p not in counts]
    extra = [p for p in edge_pairs if p not in set(route_pairs)]
    if missing or extra:
        raise ValueError(
            f"Node '{node.id}' entryRoutes mismatch with incoming exec edges. "
            f"missing={missing} extra={extra}. Reconnect the exec wires or resave the flow."
        )

    return routes


def _parse_input_route_overrides(node: VisualNode) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Parse `inputRouteOverrides` from node.data.

    Returns:
      pin_id -> { route_key -> (source_node_id, source_handle) }
    """
    raw = node.data.get("inputRouteOverrides") if isinstance(node.data, dict) else None
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for pin_id, per_route in raw.items():
        if not isinstance(pin_id, str) or not pin_id:
            continue
        if not isinstance(per_route, dict):
            continue
        norm: Dict[str, Tuple[str, str]] = {}
        for route_key, ref in per_route.items():
            if not isinstance(route_key, str) or not route_key.strip():
                continue
            if not isinstance(ref, dict):
                continue
            src = _as_str(ref.get("sourceNodeId")) or _as_str(ref.get("source"))
            h = _as_str(ref.get("sourceHandle")) or _as_str(ref.get("handle"))
            if not src or not h:
                continue
            norm[route_key.strip()] = (src, h)
        if norm:
            out[pin_id] = norm
    return out


def _pin_default(node: VisualNode, pin_id: str) -> Tuple[bool, Any]:
    data = node.data if isinstance(node.data, dict) else {}
    pd = data.get("pinDefaults")
    if not isinstance(pd, dict):
        return (False, None)
    if pin_id not in pd:
        return (False, None)
    return (True, pd.get(pin_id))


def lower_authoring_multi_entry(visual: VisualFlow) -> VisualFlow:
    """Lower multi-entry authoring metadata into internal junction nodes.

    Input (authoring) expectations:
    - Multiple incoming execution edges to `exec-in` are allowed.
    - `node.data.entryRoutes` provides stable ordering for multi-entry nodes.
    - `node.data.inputRouteOverrides` provides per-entry data overrides for input pins.

    Output (internal) graph:
    - Inserts `join_exec` before each multi-entry node.
    - Inserts `path_mux` per overridden input pin and routes values by `join_exec.which`.
    """

    # Fast path: no exec fan-in anywhere.
    incoming_exec: Dict[str, List[VisualEdge]] = {}
    for e in visual.edges:
        if _is_exec_edge(e):
            incoming_exec.setdefault(e.target, []).append(e)
    multi_targets = {tid: ins for tid, ins in incoming_exec.items() if len(ins) > 1}
    if not multi_targets:
        return visual

    nodes_by_id: Dict[str, VisualNode] = {n.id: n for n in visual.nodes}

    # Validate targets exist as nodes.
    for tid in list(multi_targets.keys()):
        if tid not in nodes_by_id:
            multi_targets.pop(tid, None)
    if not multi_targets:
        return visual

    # We'll build a new edge list by removing rewritten edges and appending new ones.
    removed_exec_edges: set[int] = set()
    removed_data_edges: set[int] = set()
    added_edges: List[VisualEdge] = []
    added_nodes: List[VisualNode] = []

    used_node_ids: set[str] = set(nodes_by_id.keys())

    # Deterministic processing order.
    for target_id in sorted(multi_targets.keys()):
        target_node = nodes_by_id[target_id]
        ins = list(multi_targets[target_id])

        routes = _parse_entry_routes(node=target_node, incoming_exec=ins)
        route_index_by_key: Dict[str, int] = {r["key"]: i for i, r in enumerate(routes)}

        join_id = f"__internal_join_exec__{target_id}"
        if join_id in used_node_ids:
            raise ValueError(f"Internal join_exec id collision: '{join_id}' already exists")
        used_node_ids.add(join_id)
        added_nodes.append(VisualNode(id=join_id, type="join_exec", data={"_internal": True}))

        # Remove existing incoming exec edges to the target (they'll be rerouted into join_exec).
        for idx, e in enumerate(visual.edges):
            if idx in removed_exec_edges:
                continue
            if not _is_exec_edge(e):
                continue
            if e.target != target_id:
                continue
            removed_exec_edges.add(idx)

        # Add exec edges into join_exec in route order, then join_exec -> target.
        for r in routes:
            added_edges.append(
                VisualEdge(
                    source=r["sourceNodeId"],
                    sourceHandle=_normalize_handle(r["sourceHandle"]),
                    target=join_id,
                    targetHandle="exec-in",
                )
            )
        added_edges.append(
            VisualEdge(
                source=join_id,
                sourceHandle="exec-out",
                target=target_id,
                targetHandle="exec-in",
            )
        )

        overrides_by_pin = _parse_input_route_overrides(target_node)
        if not overrides_by_pin:
            continue

        # Validate override route keys and referenced nodes.
        for pin_id, per_route in overrides_by_pin.items():
            for rk, (src, _h) in per_route.items():
                if rk not in route_index_by_key:
                    raise ValueError(
                        f"Node '{target_id}' override for pin '{pin_id}' references unknown route '{rk}'. "
                        "Ensure entryRoutes is up to date."
                    )
                if src not in nodes_by_id:
                    raise ValueError(
                        f"Node '{target_id}' override for pin '{pin_id}' references missing node '{src}'."
                    )

        # For each overridden pin, insert a path_mux.
        for pin_id in sorted(overrides_by_pin.keys()):
            mux_id = f"__internal_path_mux__{target_id}__{pin_id}"
            if mux_id in used_node_ids:
                raise ValueError(f"Internal path_mux id collision: '{mux_id}' already exists")
            used_node_ids.add(mux_id)

            mux_data: Dict[str, Any] = {"_internal": True}

            # If the target pin has a default and no base edge, preserve it as mux.fallback default.
            has_default, default_val = _pin_default(target_node, pin_id)

            # Find (and remove) the base incoming data edge to this pin, if any.
            base_edge: Optional[VisualEdge] = None
            base_edge_idx: Optional[int] = None
            for idx, e in enumerate(visual.edges):
                if idx in removed_data_edges or idx in removed_exec_edges:
                    continue
                if _is_exec_edge(e):
                    continue
                if e.target == target_id and e.targetHandle == pin_id:
                    if base_edge is not None:
                        raise ValueError(
                            f"Node '{target_id}' has multiple incoming data edges to pin '{pin_id}'. "
                            "Use inputRouteOverrides (per-entry overrides) instead of multi-wiring."
                        )
                    base_edge = e
                    base_edge_idx = idx

            if base_edge_idx is not None:
                removed_data_edges.add(base_edge_idx)
                # base -> mux.fallback
                added_edges.append(
                    VisualEdge(
                        source=base_edge.source,
                        sourceHandle=base_edge.sourceHandle,
                        target=mux_id,
                        targetHandle="fallback",
                    )
                )
            else:
                if has_default:
                    mux_data["pinDefaults"] = {"fallback": default_val}

            added_nodes.append(VisualNode(id=mux_id, type="path_mux", data=mux_data))

            # join.which -> mux.select
            added_edges.append(
                VisualEdge(
                    source=join_id,
                    sourceHandle="which",
                    target=mux_id,
                    targetHandle="select",
                )
            )

            # Per-route overrides: routeKey -> mux.in{idx}
            for rk, (src, h) in overrides_by_pin.get(pin_id, {}).items():
                idx = route_index_by_key[rk]
                added_edges.append(
                    VisualEdge(
                        source=src,
                        sourceHandle=h,
                        target=mux_id,
                        targetHandle=f"in{idx}",
                    )
                )

            # mux.out -> target.pin
            added_edges.append(
                VisualEdge(
                    source=mux_id,
                    sourceHandle="out",
                    target=target_id,
                    targetHandle=pin_id,
                )
            )

    # If we didn't add any nodes (unexpected), keep original.
    if not added_nodes and not added_edges:
        return visual

    # Build final nodes/edges.
    next_nodes = list(visual.nodes) + added_nodes
    next_edges: List[VisualEdge] = []
    for idx, e in enumerate(visual.edges):
        if idx in removed_exec_edges or idx in removed_data_edges:
            continue
        next_edges.append(e)
    next_edges.extend(added_edges)

    return replace(visual, nodes=next_nodes, edges=next_edges)
