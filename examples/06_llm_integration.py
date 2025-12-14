#!/usr/bin/env python3
"""
06_llm_integration.py - LLM call with AbstractCore

Demonstrates:
- LLM_CALL effect for calling language models
- AbstractCore integration via create_local_runtime
- Passing prompts and receiving responses

Requirements:
- abstractcore package
- Ollama running locally with qwen3:4b-instruct-2507-q4_K_M model

To install Ollama and the model:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen3:4b-instruct-2507-q4_K_M
"""

from abstractruntime import (
    WorkflowSpec,
    StepPlan,
    Effect,
    EffectType,
    RunStatus,
)

# This import requires abstractcore
try:
    from abstractruntime.integrations.abstractcore import create_local_runtime
except ImportError:
    print("This example requires abstractcore.")
    print("Install with: pip install abstractcore")
    exit(1)


def main():
    # Create runtime with AbstractCore LLM integration
    # This automatically sets up the LLM_CALL effect handler
    runtime = create_local_runtime(
        provider="ollama",
        model="qwen3:4b-instruct-2507-q4_K_M",
    )

    def ask_llm_node(run, ctx):
        """Call the LLM with a prompt."""
        topic = run.vars.get("topic", "Python programming")
        
        return StepPlan(
            node_id="ask_llm",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": f"Give me one interesting fact about {topic}. Keep it under 50 words.",
                },
                result_key="llm_response",
            ),
            next_node="process_response",
        )

    def process_response_node(run, ctx):
        """Process the LLM response."""
        response = run.vars.get("llm_response", {})
        content = response.get("content", "No response")
        
        return StepPlan(
            node_id="process_response",
            complete_output={
                "topic": run.vars.get("topic"),
                "fact": content,
                "model": response.get("model"),
            },
        )

    workflow = WorkflowSpec(
        workflow_id="llm_demo",
        entry_node="ask_llm",
        nodes={
            "ask_llm": ask_llm_node,
            "process_response": process_response_node,
        },
    )

    # Run with a topic
    print("Asking LLM for an interesting fact...")
    run_id = runtime.start(workflow=workflow, vars={"topic": "durable workflows"})
    
    state = runtime.tick(workflow=workflow, run_id=run_id)
    while state.status == RunStatus.RUNNING:
        state = runtime.tick(workflow=workflow, run_id=run_id)
    
    if state.status == RunStatus.COMPLETED:
        print(f"\nTopic: {state.output.get('topic')}")
        print(f"Model: {state.output.get('model')}")
        print(f"\nFact: {state.output.get('fact')}")
    else:
        print(f"Status: {state.status.value}")
        if state.error:
            print(f"Error: {state.error}")


if __name__ == "__main__":
    main()
