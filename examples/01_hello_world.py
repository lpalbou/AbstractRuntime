#!/usr/bin/env python3
"""
01_hello_world.py - Minimal AbstractRuntime workflow

Demonstrates:
- Zero-config runtime creation
- Simple two-node workflow
- Running to completion

This is the simplest possible workflow: start → end.
"""

from abstractruntime import (
    create_scheduled_runtime,
    WorkflowSpec,
    StepPlan,
)


def main():
    # Create runtime with zero configuration
    # Uses in-memory storage, auto-starts scheduler
    sr = create_scheduled_runtime()

    # Define a simple workflow with two nodes
    def start_node(run, ctx):
        """First node: set a greeting in vars."""
        run.vars["greeting"] = "Hello from AbstractRuntime!"
        return StepPlan(node_id="start", next_node="end")

    def end_node(run, ctx):
        """Final node: complete with output."""
        return StepPlan(
            node_id="end",
            complete_output={"message": run.vars["greeting"]},
        )

    workflow = WorkflowSpec(
        workflow_id="hello_world",
        entry_node="start",
        nodes={
            "start": start_node,
            "end": end_node,
        },
    )

    # Run the workflow
    # run() combines start() + tick() until completion
    run_id, state = sr.run(workflow)

    print(f"Run ID: {run_id}")
    print(f"Status: {state.status.value}")
    print(f"Output: {state.output}")

    # Clean up
    sr.stop()


if __name__ == "__main__":
    main()
