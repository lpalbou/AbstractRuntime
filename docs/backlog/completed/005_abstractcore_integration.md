## 005_abstractcore_integration (completed)

### Goal
Execute `llm_call` and `tool_calls` effects via AbstractCore in multiple topologies:
- local
- remote
- hybrid

### What shipped
- `src/abstractruntime/integrations/abstractcore/`
  - `llm_client.py`: local + remote LLM clients
  - `tool_executor.py`: executed + passthrough
  - `effect_handlers.py`: `LLM_CALL` + `TOOL_CALLS`
  - `factory.py`: convenience runtime factories

### Notes
- Remote mode targets AbstractCore server `/v1/chat/completions`.
- Tool passthrough returns `WAITING` with `WaitState.details` holding tool calls.

### Related ADRs
- [ADR 0001: Layered Coupling](../../adr/0001_layered_coupling_with_abstractcore.md) — Why integration is a separate module
- [ADR 0002: Execution Modes](../../adr/0002_execution_modes_local_remote_hybrid.md) — Why local/remote/hybrid exist

