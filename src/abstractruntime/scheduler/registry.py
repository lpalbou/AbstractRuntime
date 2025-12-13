"""abstractruntime.scheduler.registry

Workflow registry for scheduler lookups.

The scheduler needs to map workflow_id -> WorkflowSpec to call tick().
This registry provides that mapping.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..core.spec import WorkflowSpec


class WorkflowRegistry:
    """Registry mapping workflow_id to WorkflowSpec.

    Used by the Scheduler to look up workflow specs when resuming runs.

    Example:
        registry = WorkflowRegistry()
        registry.register(my_workflow)
        registry.register(another_workflow)

        # Later, scheduler can look up:
        spec = registry.get("my_workflow_id")
    """

    def __init__(self) -> None:
        self._workflows: Dict[str, WorkflowSpec] = {}

    def register(self, workflow: WorkflowSpec) -> None:
        """Register a workflow spec.

        Args:
            workflow: The workflow spec to register.

        Raises:
            ValueError: If a workflow with the same ID is already registered.
        """
        if workflow.workflow_id in self._workflows:
            raise ValueError(
                f"Workflow '{workflow.workflow_id}' is already registered. "
                "Use unregister() first if you want to replace it."
            )
        self._workflows[workflow.workflow_id] = workflow

    def unregister(self, workflow_id: str) -> None:
        """Unregister a workflow spec.

        Args:
            workflow_id: The workflow ID to unregister.

        Raises:
            KeyError: If the workflow is not registered.
        """
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow '{workflow_id}' is not registered.")
        del self._workflows[workflow_id]

    def get(self, workflow_id: str) -> Optional[WorkflowSpec]:
        """Get a workflow spec by ID.

        Args:
            workflow_id: The workflow ID to look up.

        Returns:
            The WorkflowSpec if found, None otherwise.
        """
        return self._workflows.get(workflow_id)

    def get_or_raise(self, workflow_id: str) -> WorkflowSpec:
        """Get a workflow spec by ID, raising if not found.

        Args:
            workflow_id: The workflow ID to look up.

        Returns:
            The WorkflowSpec.

        Raises:
            KeyError: If the workflow is not registered.
        """
        spec = self._workflows.get(workflow_id)
        if spec is None:
            raise KeyError(
                f"Workflow '{workflow_id}' is not registered. "
                "Register it with registry.register(workflow) before starting the scheduler."
            )
        return spec

    def list_ids(self) -> list[str]:
        """List all registered workflow IDs."""
        return list(self._workflows.keys())

    def __len__(self) -> int:
        return len(self._workflows)

    def __contains__(self, workflow_id: str) -> bool:
        return workflow_id in self._workflows
