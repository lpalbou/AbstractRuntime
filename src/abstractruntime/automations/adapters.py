"""VisualFlow adapters for the controller's `automation` nodes (contract D).

A node `{"type": "automation", "id": "<node_id>"}` runs adapter
`automation.<node_id>`. The VisualFlow compiler dispatches `automation` nodes
here (like `set_var` / `add_message`); an unknown adapter id fails the
compile, never the run.

Routing uses the node's execution outputs: a node with several outcomes has
one `case:<outcome>` edge per outcome; a node with one outcome uses
`exec-out`. A missing edge for an outcome is a compile-time error.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..core.models import StepPlan
from . import controller as c

# adapter id -> (outcomes needing a `case:` edge, or () for a single exec-out)
ADAPTER_OUTCOMES: Dict[str, tuple] = {
    "automation.read_definition": ("end", "continue"),
    "automation.wait": ("end", "go"),
    "automation.admit": ("prepare", "dispatch", "rearm"),
    "automation.prepare_context": (),
    "automation.dispatch": (),
    "automation.record_outcome": ("next", "dispatch"),
    "automation.next": (),
}


def _target(outcome: str, *, node_id: str, next_node: Optional[str], branch_map: Optional[Dict[str, str]]) -> str:
    if outcome:
        target = (branch_map or {}).get(f"case:{outcome}")
    else:
        target = next_node
    if not target:
        raise ValueError(
            f"automation node {node_id!r} has no execution edge for outcome "
            f"{('case:' + outcome) if outcome else 'exec-out'!r}"
        )
    return target


def create_automation_node_handler(
    *,
    node_id: str,
    next_node: Optional[str],
    branch_map: Optional[Dict[str, str]],
) -> Callable[[Any, Any], StepPlan]:
    adapter_id = f"automation.{node_id}"
    if adapter_id not in ADAPTER_OUTCOMES:
        raise ValueError(f"unknown automation adapter {adapter_id!r} (known: {sorted(ADAPTER_OUTCOMES)})")
    routes = {
        outcome: _target(outcome, node_id=node_id, next_node=next_node, branch_map=branch_map)
        for outcome in (ADAPTER_OUTCOMES[adapter_id] or ("",))
    }

    def handler(run: Any, ctx: Any) -> StepPlan:
        turn = c.Turn.from_run(run)
        if adapter_id == "automation.read_definition":
            return StepPlan(node_id=node_id, next_node=routes[c.read_definition(turn)])
        if adapter_id == "automation.wait":
            decision = c.wait_decision(turn)
            if decision.get("end"):
                return StepPlan(node_id=node_id, next_node=routes["end"])
            if decision.get("go"):
                return StepPlan(node_id=node_id, next_node=routes["go"])
            return StepPlan(node_id=node_id, effect=c.wake_effect(turn, decision.get("until")), next_node=routes["go"])
        if adapter_id == "automation.admit":
            return StepPlan(node_id=node_id, next_node=routes[c.admit(turn)])
        if adapter_id == "automation.prepare_context":
            c.prepare_context(turn)
            return StepPlan(node_id=node_id, next_node=routes[""])
        if adapter_id == "automation.dispatch":
            return StepPlan(node_id=node_id, effect=c.dispatch(turn), next_node=routes[""])
        if adapter_id == "automation.record_outcome":
            return StepPlan(node_id=node_id, next_node=routes[c.record_outcome(turn)])
        c.next_step(turn)
        return StepPlan(node_id=node_id, next_node=routes[""])

    handler.__name__ = f"automation_{node_id}"
    return handler


__all__ = ["ADAPTER_OUTCOMES", "create_automation_node_handler"]
