# AbstractRuntime Examples

Runnable examples demonstrating AbstractRuntime capabilities.

## Quick Start

```bash
cd examples
python 01_hello_world.py
```

## Examples

| Example | Description | Dependencies |
|---------|-------------|--------------|
| 01_hello_world.py | Minimal workflow with zero-config | None |
| 02_ask_user.py | Pause for user input, resume with response | None |
| 03_wait_until.py | Schedule a task for later | None |
| 04_multi_step.py | Multi-node workflow with branching | None |
| 05_persistence.py | File-based storage, survive restart | None |
| 06_llm_integration.py | LLM call with AbstractCore | abstractcore, ollama |
| 07_react_agent.py | Full ReAct agent with tools | abstractcore, abstractagent, ollama |

## Requirements

Examples 1-5 only require abstractruntime:
```bash
pip install abstractruntime
```

Examples 6-7 require additional packages:
```bash
pip install abstractcore abstractagent
# Also requires Ollama running locally with qwen3:4b-instruct-2507-q4_K_M
```

## Running Examples

Each example is self-contained. Run directly:

```bash
python 01_hello_world.py
```

For interactive examples (02, 05), follow the prompts.
