## ADR 0001: Layered coupling with AbstractCore

### Status
Accepted (2025-12-11)

### Context
AbstractRuntime needs to reuse AbstractCore capabilities (LLM calls, tool execution, structured logging), but we must keep the **durable kernel** stable and dependency-light.

Mixing AbstractCore imports into the kernel creates:
- brittle coupling to AbstractCore internals
- larger runtime footprint for processes that only need bookkeeping (stores/ledger)
- harder evolution of the durable state machine

### Decision
- The kernel (`abstractruntime.core`, `abstractruntime.storage`, `abstractruntime.identity`) must **not** import AbstractCore.
- AbstractCore integration is an explicit opt-in module:
  - `abstractruntime.integrations.abstractcore`

### Consequences
- Kernel remains stable and reusable across topologies.
- AbstractCore can evolve; only the integration module needs updates.
- We still support heavy reuse of AbstractCore in real deployments (local/remote/hybrid execution modes).

### See Also
- Implementation: [`backlog/completed/005_abstractcore_integration.md`](../backlog/completed/005_abstractcore_integration.md)
- Code: `src/abstractruntime/integrations/abstractcore/`

