#!/usr/bin/env python3
"""
07_react_agent.py - Full ReAct agent with tools

Demonstrates:
- ReAct (Reason + Act) agent pattern
- Durable tool execution via ToolExecutor
- Multi-turn LLM conversation
- Agent completing a task autonomously

Requirements:
- abstractcore package
- abstractagent package
- Ollama running locally with qwen3:4b-instruct-2507-q4_K_M model

This example shows how AbstractRuntime powers an autonomous agent
that can reason about tasks and use tools to complete them.
"""

import sys

# Check dependencies
try:
    from abstractcore.tools import ToolDefinition
    from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime
except ImportError:
    print("This example requires abstractcore.")
    print("Install with: pip install abstractcore")
    sys.exit(1)

try:
    from abstractagent.agents.react import ReactAgent
    from abstractagent.tools import ALL_TOOLS
except ImportError:
    print("This example requires abstractagent.")
    print("Install from the abstractagent directory.")
    sys.exit(1)

from abstractruntime import RunStatus


def on_step(step: str, data: dict) -> None:
    """Callback to display agent progress."""
    if step == "init":
        print(f"\n{'='*60}")
        print(f"Task: {data.get('task', '')}")
        print(f"{'='*60}\n")
    elif step == "reason":
        print(f"[Step {data.get('iteration', '?')}] Thinking...")
    elif step == "parse":
        if data.get("has_tool_calls"):
            print("  → Decided to use tools")
    elif step == "act":
        tool = data.get("tool", "unknown")
        args = data.get("args", {})
        print(f"  → Tool: {tool}({args})")
    elif step == "observe":
        result = data.get("result", "")[:60]
        print(f"  → Result: {result}...")
    elif step == "done":
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(data.get("answer", "No answer"))
        print(f"{'='*60}")


def main():
    tools = list(ALL_TOOLS)
    tool_executor = MappingToolExecutor.from_tools(tools)

    # Create runtime with AbstractCore LLM integration + durable tool execution
    runtime = create_local_runtime(
        provider="ollama",
        model="qwen3:4b-instruct-2507-q4_K_M",
        tool_executor=tool_executor,
    )
    
    print("Available tools:")
    for t in tools:
        tool_def = getattr(t, "_tool_definition", None) or ToolDefinition.from_function(t)
        print(f"  - {tool_def.name}: {tool_def.description[:50]}...")
    
    # Create the ReAct agent
    agent = ReactAgent(
        runtime=runtime,
        tools=tools,
        on_step=on_step,
    )
    
    # Run the agent with a task
    task = "List the files in the current directory"
    
    agent.start(task)
    
    # Run to completion (or waiting state)
    while True:
        state = agent.step()
        
        if state.status == RunStatus.COMPLETED:
            print(f"\nCompleted in {state.output.get('iterations', '?')} steps")
            break
        elif state.status == RunStatus.WAITING:
            # Handle user questions if agent asks
            question = agent.get_pending_question()
            if question:
                print(f"\nAgent asks: {question.get('prompt')}")
                # Only prompt for input if running interactively
                import sys
                if sys.stdin.isatty():
                    response = input("Your response: ").strip()
                    agent.resume(response)
                else:
                    print("(Non-interactive mode, skipping)")
                    break
            else:
                print("Agent is waiting for unknown reason")
                break
        elif state.status == RunStatus.FAILED:
            print(f"\nFailed: {state.error}")
            break


if __name__ == "__main__":
    main()
