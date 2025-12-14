#!/usr/bin/env python3
"""
04_multi_step.py - Multi-node workflow with branching

Demonstrates:
- Multiple nodes in a workflow
- Conditional branching based on state
- Accumulating state across nodes
- Loop patterns

This workflow processes a list of items, branching based on conditions.
"""

from abstractruntime import (
    create_scheduled_runtime,
    WorkflowSpec,
    StepPlan,
)


def main():
    sr = create_scheduled_runtime()

    def init_node(run, ctx):
        """Initialize with items to process."""
        run.vars["items"] = ["apple", "banana", "cherry", "date", "elderberry"]
        run.vars["processed"] = []
        run.vars["index"] = 0
        print("Initialized with 5 items to process")
        return StepPlan(node_id="init", next_node="check")

    def check_node(run, ctx):
        """Check if there are more items to process."""
        index = run.vars["index"]
        items = run.vars["items"]
        
        if index >= len(items):
            print("All items processed, finishing...")
            return StepPlan(node_id="check", next_node="finish")
        
        current = items[index]
        print(f"Processing item {index + 1}/{len(items)}: {current}")
        return StepPlan(node_id="check", next_node="process")

    def process_node(run, ctx):
        """Process the current item with conditional logic."""
        index = run.vars["index"]
        items = run.vars["items"]
        current = items[index]
        
        # Conditional processing
        if len(current) > 5:
            result = f"{current.upper()} (long)"
        else:
            result = f"{current.capitalize()} (short)"
        
        run.vars["processed"].append(result)
        run.vars["index"] = index + 1
        
        print(f"  → Result: {result}")
        
        # Loop back to check
        return StepPlan(node_id="process", next_node="check")

    def finish_node(run, ctx):
        """Complete with summary."""
        processed = run.vars["processed"]
        
        return StepPlan(
            node_id="finish",
            complete_output={
                "total_processed": len(processed),
                "results": processed,
                "summary": f"Processed {len(processed)} items",
            },
        )

    workflow = WorkflowSpec(
        workflow_id="multi_step_demo",
        entry_node="init",
        nodes={
            "init": init_node,
            "check": check_node,
            "process": process_node,
            "finish": finish_node,
        },
    )

    # Run to completion
    run_id, state = sr.run(workflow)
    
    print()
    print(f"Status: {state.status.value}")
    print(f"Output: {state.output}")

    sr.stop()


if __name__ == "__main__":
    main()
