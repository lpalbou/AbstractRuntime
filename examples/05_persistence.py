#!/usr/bin/env python3
"""
05_persistence.py - File-based storage, survive restart

Demonstrates:
- File-based run store (JSON)
- File-based ledger store (JSONL)
- Resuming a workflow after "restart"
- Inspecting persisted state

Run this example twice:
1. First run: starts workflow, pauses for input, saves state
2. Second run: loads state, resumes from where it left off
"""

import os
import sys
import json
from pathlib import Path

from abstractruntime import (
    Runtime,
    WorkflowSpec,
    StepPlan,
    Effect,
    EffectType,
    RunStatus,
    JsonFileRunStore,
    JsonlLedgerStore,
)


# Persistence files
DATA_DIR = Path(__file__).parent / "data"
RUN_STORE_FILE = DATA_DIR / "runs.json"
LEDGER_DIR = DATA_DIR / "ledger"
STATE_FILE = DATA_DIR / "current_run.txt"


def create_workflow():
    """Create the demo workflow."""
    
    def step1_node(run, ctx):
        """First step: record start."""
        run.vars["step1_done"] = True
        run.vars["progress"] = ["Step 1 completed"]
        print("Step 1: Initialized")
        return StepPlan(node_id="step1", next_node="step2")

    def step2_node(run, ctx):
        """Second step: pause for user input."""
        run.vars["progress"].append("Step 2: Waiting for input")
        print("Step 2: Pausing for user input...")
        
        return StepPlan(
            node_id="step2",
            effect=Effect(
                type=EffectType.ASK_USER,
                payload={
                    "prompt": "Enter a message to continue:",
                    "allow_free_text": True,
                },
                result_key="user_message",
            ),
            next_node="step3",
        )

    def step3_node(run, ctx):
        """Third step: complete with user's message."""
        message = run.vars.get("user_message", {}).get("response", "no message")
        run.vars["progress"].append(f"Step 3: Received '{message}'")
        print(f"Step 3: Received message: {message}")
        
        return StepPlan(
            node_id="step3",
            complete_output={
                "progress": run.vars["progress"],
                "user_message": message,
            },
        )

    return WorkflowSpec(
        workflow_id="persistence_demo",
        entry_node="step1",
        nodes={
            "step1": step1_node,
            "step2": step2_node,
            "step3": step3_node,
        },
    )


def main():
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    LEDGER_DIR.mkdir(exist_ok=True)
    
    # Create file-based stores
    run_store = JsonFileRunStore(str(RUN_STORE_FILE))
    ledger_store = JsonlLedgerStore(str(LEDGER_DIR))
    
    # Create runtime with file-based persistence
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    
    workflow = create_workflow()
    
    # Check if we have a saved run to resume
    if STATE_FILE.exists():
        run_id = STATE_FILE.read_text().strip()
        print(f"Found saved run: {run_id}")
        
        try:
            state = runtime.get_state(run_id)
            print(f"Current status: {state.status.value}")
            print(f"Current node: {state.current_node}")
            print(f"Progress: {state.vars.get('progress', [])}")
            
            if state.status == RunStatus.WAITING:
                print(f"\nWorkflow is waiting for: {state.waiting.prompt}")
                user_input = input("Your response: ").strip()
                
                state = runtime.resume(
                    workflow=workflow,
                    run_id=run_id,
                    wait_key=state.waiting.wait_key,
                    payload={"response": user_input},
                )
                
                # Continue until done
                while state.status == RunStatus.RUNNING:
                    state = runtime.tick(workflow=workflow, run_id=run_id)
            
            if state.status == RunStatus.COMPLETED:
                print(f"\nWorkflow completed!")
                print(f"Output: {state.output}")
                # Clean up state file
                STATE_FILE.unlink()
                print("\nState file cleaned up. Run again to start fresh.")
            
        except Exception as e:
            print(f"Error resuming: {e}")
            print("Starting fresh...")
            STATE_FILE.unlink(missing_ok=True)
            main()  # Restart
    
    else:
        # Start a new run
        print("Starting new workflow run...")
        run_id = runtime.start(workflow=workflow, vars={})
        
        # Save run_id for later
        STATE_FILE.write_text(run_id)
        print(f"Saved run ID: {run_id}")
        
        # Run until waiting or complete
        state = runtime.tick(workflow=workflow, run_id=run_id)
        while state.status == RunStatus.RUNNING:
            state = runtime.tick(workflow=workflow, run_id=run_id)
        
        if state.status == RunStatus.WAITING:
            print(f"\nWorkflow paused at: {state.current_node}")
            print(f"Waiting for: {state.waiting.prompt}")
            print("\n" + "="*50)
            print("State has been saved to disk.")
            print("Run this script again to resume!")
            print("="*50)
        elif state.status == RunStatus.COMPLETED:
            print(f"\nWorkflow completed: {state.output}")
            STATE_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
