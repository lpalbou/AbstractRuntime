#!/usr/bin/env python3
"""
02_ask_user.py - Pause for user input, resume with response

Demonstrates:
- ASK_USER effect for durable pauses
- Workflow waiting for external input
- Resuming with respond()
- Choices and free text input

The workflow pauses and waits for user input. This pause is durable -
the process could restart and the workflow would still be waiting.
"""

from abstractruntime import (
    create_scheduled_runtime,
    WorkflowSpec,
    StepPlan,
    Effect,
    EffectType,
    RunStatus,
)


def main():
    sr = create_scheduled_runtime()

    # Workflow that asks the user a question
    def start_node(run, ctx):
        """Ask the user for their name."""
        return StepPlan(
            node_id="start",
            effect=Effect(
                type=EffectType.ASK_USER,
                payload={
                    "prompt": "What is your name?",
                    "choices": None,  # Free text only
                    "allow_free_text": True,
                },
                result_key="user_name",
            ),
            next_node="greet",
        )

    def greet_node(run, ctx):
        """Greet the user by name, then ask a follow-up."""
        name = run.vars.get("user_name", {}).get("response", "stranger")
        run.vars["name"] = name
        
        return StepPlan(
            node_id="greet",
            effect=Effect(
                type=EffectType.ASK_USER,
                payload={
                    "prompt": f"Hello {name}! What would you like to do?",
                    "choices": ["Learn about workflows", "See an example", "Exit"],
                    "allow_free_text": True,
                },
                result_key="user_choice",
            ),
            next_node="respond",
        )

    def respond_node(run, ctx):
        """Respond based on user's choice."""
        choice = run.vars.get("user_choice", {}).get("response", "")
        name = run.vars.get("name", "friend")
        
        if "learn" in choice.lower() or choice == "Learn about workflows":
            message = f"Great choice, {name}! Workflows are durable state machines."
        elif "example" in choice.lower() or choice == "See an example":
            message = f"Check out the other examples in this directory, {name}!"
        else:
            message = f"Goodbye, {name}!"
        
        return StepPlan(
            node_id="respond",
            complete_output={"message": message, "name": name, "choice": choice},
        )

    workflow = WorkflowSpec(
        workflow_id="ask_user_demo",
        entry_node="start",
        nodes={
            "start": start_node,
            "greet": greet_node,
            "respond": respond_node,
        },
    )

    # Start the workflow
    run_id, state = sr.run(workflow)
    
    # The workflow is now waiting for user input
    while state.status == RunStatus.WAITING:
        # Display the question
        if state.waiting:
            print(f"\n{'='*50}")
            print(f"Question: {state.waiting.prompt}")
            if state.waiting.choices:
                print("Choices:")
                for i, choice in enumerate(state.waiting.choices, 1):
                    print(f"  [{i}] {choice}")
                print("  [0] Type your own response")
            print(f"{'='*50}")
            
            # Get user input
            user_input = input("\nYour response: ").strip()
            
            # Handle choice selection
            if state.waiting.choices:
                try:
                    choice_num = int(user_input)
                    if choice_num == 0:
                        user_input = input("Type your response: ").strip()
                    elif 1 <= choice_num <= len(state.waiting.choices):
                        user_input = state.waiting.choices[choice_num - 1]
                except ValueError:
                    pass  # Use as-is
            
            # Resume with the response
            state = sr.respond(run_id, {"response": user_input})
    
    # Workflow completed
    print(f"\n{'='*50}")
    print(f"Status: {state.status.value}")
    print(f"Output: {state.output}")
    print(f"{'='*50}")

    sr.stop()


if __name__ == "__main__":
    main()
