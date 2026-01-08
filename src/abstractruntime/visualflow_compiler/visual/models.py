"""Stdlib-only models for the VisualFlow JSON schema.

These are intentionally minimal and permissive:
- They accept unknown/extra fields (ignored).
- `type` is stored as a string for forward compatibility.

The compiler/executor code is responsible for interpreting node types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    # Event/Trigger nodes (entry points)
    ON_FLOW_START = "on_flow_start"
    ON_USER_REQUEST = "on_user_request"
    ON_AGENT_MESSAGE = "on_agent_message"
    ON_SCHEDULE = "on_schedule"
    ON_EVENT = "on_event"
    # Flow IO nodes
    ON_FLOW_END = "on_flow_end"
    # Core execution nodes
    AGENT = "agent"
    FUNCTION = "function"
    CODE = "code"
    SUBFLOW = "subflow"


@dataclass(frozen=True)
class VisualNode:
    id: str
    type: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualEdge:
    source: str
    target: str
    sourceHandle: str
    targetHandle: str


@dataclass(frozen=True)
class VisualFlow:
    id: str
    name: str = ""
    description: str = ""
    interfaces: List[str] = field(default_factory=list)
    nodes: List[VisualNode] = field(default_factory=list)
    edges: List[VisualEdge] = field(default_factory=list)
    entryNode: Optional[str] = None


def _coerce_type(value: Any) -> str:
    if isinstance(value, str):
        s = value.strip()
        # Pydantic (and other serializers) may stringify enums as "NodeType.X".
        # Normalize to the underlying value-like form ("x") for downstream matching.
        if s.startswith("NodeType.") and "." in s:
            member = s.split(".", 1)[1].strip()
            if member:
                return member.lower()
        return s
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, dict):
        # Best-effort support for enum-like objects serialized as {"value": "..."}.
        v = value.get("value")
        if isinstance(v, str):
            return v
    return str(value or "")


def load_visualflow_json(raw: Any) -> VisualFlow:
    """Parse a VisualFlow JSON object (dict) into stdlib dataclasses.

    Also accepts Pydantic-like models by calling `model_dump()` or `dict()`.
    """
    if hasattr(raw, "model_dump"):
        # Prefer JSON mode so enums are dumped as their values (not "NodeType.X").
        try:
            raw = raw.model_dump(mode="json")  # type: ignore[assignment]
        except TypeError:
            raw = raw.model_dump()  # type: ignore[assignment]
    elif hasattr(raw, "dict"):
        raw = raw.dict()  # type: ignore[assignment]

    if not isinstance(raw, dict):
        raise TypeError("VisualFlow must be a JSON object (dict)")

    fid = str(raw.get("id") or raw.get("flow_id") or raw.get("workflow_id") or "").strip()
    if not fid:
        raise ValueError("VisualFlow missing required 'id'")

    name = str(raw.get("name") or "")
    description = str(raw.get("description") or "")

    interfaces_raw = raw.get("interfaces")
    interfaces: list[str] = []
    if isinstance(interfaces_raw, list):
        for it in interfaces_raw:
            if isinstance(it, str) and it.strip():
                interfaces.append(it.strip())

    nodes_raw = raw.get("nodes")
    nodes: list[VisualNode] = []
    if isinstance(nodes_raw, list):
        for n in nodes_raw:
            if not isinstance(n, dict):
                continue
            nid = str(n.get("id") or "").strip()
            if not nid:
                continue
            t = _coerce_type(n.get("type"))
            data = n.get("data")
            data_d = dict(data) if isinstance(data, dict) else {}
            nodes.append(VisualNode(id=nid, type=str(t), data=data_d))

    edges_raw = raw.get("edges")
    edges: list[VisualEdge] = []
    if isinstance(edges_raw, list):
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            src = str(e.get("source") or "").strip()
            tgt = str(e.get("target") or "").strip()
            if not src or not tgt:
                continue
            sh = e.get("sourceHandle")
            th = e.get("targetHandle")
            # VisualFlow edges are defined by their pin handles; skip malformed edges.
            if not isinstance(sh, str) or not sh.strip():
                continue
            if not isinstance(th, str) or not th.strip():
                continue
            edges.append(
                VisualEdge(
                    source=src,
                    target=tgt,
                    sourceHandle=sh.strip(),
                    targetHandle=th.strip(),
                )
            )

    entry = raw.get("entryNode")
    entry_node = str(entry).strip() if isinstance(entry, str) and entry.strip() else None

    return VisualFlow(
        id=fid,
        name=name,
        description=description,
        interfaces=interfaces,
        nodes=nodes,
        edges=edges,
        entryNode=entry_node,
    )
