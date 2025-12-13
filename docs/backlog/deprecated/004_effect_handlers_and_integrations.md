# 004_effect_handlers_and_integrations

> Legacy note: preserved for history. See `docs/backlog/completed/005_abstractcore_integration.md`.

## Goal
Define effect handler interfaces and reference adapters for:
- tool execution
- LLM calls

## Design
AbstractRuntime defines protocols/interfaces only.
Adapters can live in AbstractFlow or integration packages.

### Tool execution
- `ToolExecutor.execute(tool_calls) -> tool_results`

Adapters:
- in-process AbstractCore tool executor
- pass-through executor (returns tool calls to host)

### LLM calls
- `LLMClient.generate(...) -> response`

Adapters:
- LocalAbstractCoreLLMClient (create_llm + generate)
- RemoteAbstractCoreLLMClient (HTTP to AbstractCore server)

## Acceptance criteria
- A workflow can emit a `tool_calls` effect and be executed via a provided ToolExecutor.
- A workflow can emit an `llm_call` effect and be executed via a provided LLMClient.


