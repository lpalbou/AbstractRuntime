"""Refuse an execution wire that loops back out of a loop body (AbstractFramework backlog 1018).

A ForEach / For / While runs its body one pass at a time: the body ends, control returns to the
loop node, which starts the next pass or pops its frame through `done`. An execution wire from
inside a body back to a node that is not inside that body (the loop node itself, or anything
before it) re-enters the loop while its frame is still on the control stack: the stack grows by
one frame per pass and the run never ends (a Loop Limit on that wire counts per pass and never
reaches its max). Operator ruling 2026-10-15: such a wire is refused when the flow compiles, with
the words the editor's run checks use.

A loop-back is found the way the editor finds its ↺ badges (AbstractFlow `loopBackEdges.ts`): a
depth-first walk over the execution wires from the entry node, children in the order of the
source's output pins; a wire landing on a node still on the walk's stack closes a loop, and it
leaves a loop body when the stack from that node to the wire's source holds a loop node that was
left through its `loop` pin.

This module must remain stdlib-only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import VisualEdge, VisualFlow, VisualNode

LOOP_NODE_TYPES = {"loop", "for", "while"}
BODY_PIN = "loop"


def _type_of(node: VisualNode) -> str:
    t = node.type
    return str(t.value if hasattr(t, "value") else t)


def _data(node: VisualNode) -> dict:
    return node.data if isinstance(node.data, dict) else {}


def _label(node: Optional[VisualNode], node_id: str) -> str:
    label = _data(node).get("label") if node is not None else None
    return str(label).strip() if isinstance(label, str) and label.strip() else node_id


def loop_back_out_of_body_message(edge: VisualEdge, loop: VisualNode, source: Optional[VisualNode], target: Optional[VisualNode]) -> str:
    return (
        f"Wire {_label(source, edge.source)}.{edge.sourceHandle or 'exec-out'} -> {_label(target, edge.target)} "
        f"goes back out of {_label(loop, loop.id)}'s loop body: the run would never end. "
        "A path inside a loop body can only repeat within the body; to repeat the whole loop, "
        "wire the loop-back after the loop's done."
    )


def refuse_loop_backs_out_of_bodies(visual: VisualFlow) -> VisualFlow:
    """Raise ValueError on the first execution wire that loops back out of a loop body; else return `visual` unchanged."""
    by_id: Dict[str, VisualNode] = {n.id: n for n in visual.nodes}
    out: Dict[str, List[VisualEdge]] = {}
    for e in visual.edges:
        if e.targetHandle != "exec-in" or e.source not in by_id or e.target not in by_id:
            continue
        out.setdefault(e.source, []).append(e)

    def pin_index(e: VisualEdge) -> int:
        outputs = _data(by_id[e.source]).get("outputs") or []
        ids = [str(p.get("id")) for p in outputs if isinstance(p, dict)]
        handle = e.sourceHandle or ""
        return ids.index(handle) if handle in ids else len(ids)

    for wires in out.values():
        wires.sort(key=lambda e: (pin_index(e), e.target))

    done: set = set()
    on_stack: Dict[str, int] = {}
    stack: List[List[str]] = []  # [node id, output pin it is being left through]

    def walk(root: str) -> None:
        on_stack[root] = len(stack)
        stack.append([root, ""])
        pending: List[List[VisualEdge]] = [list(out.get(root, []))]
        while pending:
            if not pending[-1]:
                pending.pop()
                nid, _ = stack.pop()
                on_stack.pop(nid, None)
                done.add(nid)
                continue
            e = pending[-1].pop(0)
            stack[-1][1] = e.sourceHandle or ""
            at = on_stack.get(e.target)
            if at is not None:
                for sid, handle in stack[at:]:
                    node = by_id[sid]
                    if _type_of(node) in LOOP_NODE_TYPES and handle == BODY_PIN:
                        raise ValueError(loop_back_out_of_body_message(e, node, by_id.get(e.source), by_id.get(e.target)))
            elif e.target not in done:
                on_stack[e.target] = len(stack)
                stack.append([e.target, ""])
                pending.append(list(out.get(e.target, [])))

    if visual.entryNode and visual.entryNode in by_id:
        walk(visual.entryNode)
    for n in visual.nodes:
        if n.id not in done:
            walk(n.id)
    return visual
