"""abstractruntime.core.spec

Workflow specification and node contract.

In v0.1 we keep the spec simple:
- The workflow is a graph: nodes identified by string IDs.
- Nodes are implemented as Python callables (handlers).
- This kernel does not define a UI DSL; AbstractFlow can produce specs.

Durability note:
- We persist *RunState* and a *ledger*.
- We assume the workflow spec + node handlers are available at resume time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

from .models import RunState, StepPlan


class RunContext(Protocol):
    """Dependency injection surface for node handlers.

    This is intentionally small and can be extended later.
    """

    def now_iso(self) -> str: ...


NodeHandler = Callable[[RunState, RunContext], StepPlan]


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    entry_node: str
    nodes: Dict[str, NodeHandler]

    def get_node(self, node_id: str) -> NodeHandler:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node_id '{node_id}' in workflow '{self.workflow_id}'")
        return self.nodes[node_id]

    def is_terminal(self, node_id: str) -> bool:
        """A workflow is terminal when the node returns StepPlan.complete_output.

        The runtime decides termination based on StepPlan, not a dedicated node type.
        """

        return False  # evaluated at runtime

