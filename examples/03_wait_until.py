#!/usr/bin/env python3
"""
03_wait_until.py - Schedule a task for later

Demonstrates:
- WAIT_UNTIL effect for time-based pauses
- Scheduler automatically resuming when time is reached
- Background polling

The workflow pauses until a specific time, then continues.
The scheduler polls and resumes automatically.
"""

import time
from datetime import datetime, timedelta, timezone

from abstractruntime import (
    create_scheduled_runtime,
    WorkflowSpec,
    StepPlan,
    Effect,
    EffectType,
    RunStatus,
)


def main():
    # Create runtime with fast polling for demo
    sr = create_scheduled_runtime(poll_interval_s=0.5)

    def start_node(run, ctx):
        """Start and schedule a wake-up in 3 seconds."""
        now = datetime.now(timezone.utc)
        wake_time = now + timedelta(seconds=3)
        run.vars["scheduled_for"] = wake_time.isoformat()
        
        print(f"[{now.strftime('%H:%M:%S')}] Scheduling wake-up for 3 seconds from now...")
        
        return StepPlan(
            node_id="start",
            effect=Effect(
                type=EffectType.WAIT_UNTIL,
                payload={"until": wake_time.isoformat()},
                result_key="wake_result",
            ),
            next_node="woke_up",
        )

    def woke_up_node(run, ctx):
        """Called when the scheduled time is reached."""
        now = datetime.now(timezone.utc)
        scheduled = run.vars.get("scheduled_for", "unknown")
        
        return StepPlan(
            node_id="woke_up",
            complete_output={
                "message": "Woke up on schedule!",
                "scheduled_for": scheduled,
                "actual_time": now.isoformat(),
            },
        )

    workflow = WorkflowSpec(
        workflow_id="wait_until_demo",
        entry_node="start",
        nodes={
            "start": start_node,
            "woke_up": woke_up_node,
        },
    )

    # Start the workflow
    run_id, state = sr.run(workflow)
    
    if state.status == RunStatus.WAITING:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Workflow is waiting...")
        print("Scheduler will automatically resume when time is reached.")
        print()
        
        # Wait for scheduler to resume
        while state.status == RunStatus.WAITING:
            time.sleep(0.5)
            state = sr.get_state(run_id)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {state.status.value}")
    
    print()
    print(f"Final status: {state.status.value}")
    print(f"Output: {state.output}")

    sr.stop()


if __name__ == "__main__":
    main()
