## ADR 0002: Execution modes (local, remote, hybrid)

### Status
Accepted (2025-12-11)

### Context
Agents/workflows must run in multiple deployment topologies:
- thin clients (mobile/web) calling a backend LLM gateway
- backend orchestration calling GPU inference fleets
- local/dev mode (everything on one machine)

AbstractCore already provides two compatible boundaries:
- in-process python API (`create_llm(...).generate(...)`)
- HTTP server boundary (`/v1/chat/completions`)

### Decision
AbstractRuntime supports three execution modes:

- **Local**: in-process AbstractCore LLM + local tool execution
- **Remote**: HTTP to AbstractCore server; tools default to passthrough (untrusted)
- **Hybrid**: remote LLM + local tool execution

### Consequences
- Thin-mode clients can run the workflow logic while delegating inference to a server.
- Remote mode supports AbstractCore per-request `base_url` routing (dynamic endpoint selection).
- Tool execution can be gated by trust/sandbox policy outside the router.

### See Also
- Implementation: [`backlog/completed/005_abstractcore_integration.md`](../backlog/completed/005_abstractcore_integration.md)
- Integration guide: [`integrations/abstractcore.md`](../integrations/abstractcore.md)
- Code: `src/abstractruntime/integrations/abstractcore/factory.py`

