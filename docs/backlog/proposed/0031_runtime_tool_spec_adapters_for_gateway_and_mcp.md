# Proposed: Runtime tool-spec adapters for Gateway and MCP consumers

## Metadata
- Created: 2026-05-21
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None unless Runtime later promotes a package-level tool-spec API outside the AbstractCore integration package.

## Context

Runtime still derives JSON-safe tool specs from public AbstractCore tool
definitions in:

- `src/abstractruntime/integrations/abstractcore/default_tools.py`
- `src/abstractruntime/integrations/abstractcore/mcp_worker.py`

Gateway already consumes Runtime-owned default tool specs for production
surfaces. The remaining pressure is narrower:

- one Gateway-side contract test still imports `abstractcore.tools.core.ToolDefinition`
- Runtime still calls `ToolDefinition.from_function(...)` directly for some
  host/MCP/toolset paths

That makes tool-spec adapters useful, but lower priority than the remaining
comms/Telegram boundary cleanup tracked in planned item `0030`.

## Problem or opportunity

If Runtime wants hosts and tests to avoid importing Core tool-definition types
entirely, Runtime may eventually need a small public adapter surface for:

- deriving JSON-safe tool specs from callables
- deriving MCP-compatible tool entries
- normalizing the small Runtime-owned presentation quirks layered on top

This is not yet proven to be urgent enough to block the narrower Gateway
cleanup work.

## Proposed direction

Keep this as a follow-up until phase-1 `0030` adoption lands.

If promoted, the likely shape is a small integration-scoped helper module under
`abstractruntime.integrations.abstractcore`, or a narrow `abstractruntime.tools`
surface if the API becomes generally useful outside the AbstractCore
integration.

The first promotion target would be:

- `default_tools.py`
- `mcp_worker.py`
- Gateway contract tests or renderer paths that still want spec derivation

## Why it might matter

- removes the last low-level `ToolDefinition` dependency from Runtime-owned host
  surfaces
- lets Gateway tests validate Runtime-owned tool specs without importing Core
  tool types directly
- gives MCP worker conversion a Runtime-owned public seam if more hosts need it

## Promotion criteria

Promote only if one of these becomes true:

- `0030` comms/Telegram adoption still leaves material Gateway/Core coupling
  because tool-spec consumers remain blocked on `ToolDefinition`
- Runtime starts duplicating tool-spec normalization logic in more than one
  place
- another host besides Gateway needs a public Runtime-owned tool-spec adapter
  surface

## Non-goals

- Do not redesign durable `TOOL_CALLS`.
- Do not replace `abstractcore.tools.ToolCall` everywhere in one step.
- Do not block comms/Telegram boundary cleanup on this work.

## Guidance for future agents

Treat this as optional cleanup until a real host consumer proves it is worth
promoting. Runtime can continue using thin wrappers over public Core tool
definition helpers in the meantime.
