# Deprecated: Agent Runtime Convenience Constructor

## Metadata
- Created: 2026-05-20
- Status: Deprecated
- Completed: N/A
- Deprecated: 2026-05-21

## Context

The old `015_agent_integration_improvements` item no longer fits as planned runtime work.

Most of the important integration correctness work is already done:

- agent tool execution can go through durable `TOOL_CALLS`
- runtime tool execution is host-configured through `ToolExecutor`
- agent state persistence is not blocked on runtime internals

The main remaining idea is much smaller: a convenience constructor such as
`create_agent_runtime(...)` that would wrap `create_local_runtime(...)` plus
`MappingToolExecutor.from_tools(...)` for simple local setups.

## Current code reality

- `src/abstractruntime/integrations/abstractcore/factory.py`
  - already exposes the generic Runtime constructor layer:
    - `create_local_runtime(...)`
    - `create_remote_runtime(...)`
    - `create_hybrid_runtime(...)`
- `../abstractagent/src/abstractagent/agents/react.py`
  - already wraps `create_local_runtime(...)` +
    `MappingToolExecutor.from_tools(...)`
- `../abstractagent/src/abstractagent/agents/codeact.py`
  - does the same for CodeAct
- `../abstractagent/src/abstractagent/agents/memact.py`
  - does the same for MemAct
- `../abstractflow/abstractflow/visual/executor.py`
  - assembles its own Runtime/tool wiring with different host-specific policy

The duplication exists, but it is concentrated in higher-level packages that
already own agent or host semantics.

## Why it is not planned now

- It is an ergonomics improvement, not a runtime correctness gap.
- The API shape depends on cross-repo expectations from `abstractagent`.
- This repo should stay careful about adding convenience APIs that quietly pull in higher-level agent assumptions.

## Possible future direction

- keep the helper explicitly local-only
- accept direct tool functions and build the `MappingToolExecutor`
- avoid storing callables in durable run state
- document clearly that advanced approval/passthrough/delegated tool setups should still use the lower-level factories directly

## Guidance

Re-promote this only if maintainers still want a tiny convenience API after the higher-priority runtime boundary work lands.

## Deprecation report

Deprecated: 2026-05-21

Reason:

- Runtime already has the correct generic constructor layer.
- The exact convenience pattern is agent-shaped rather than runtime-shaped.
- The real duplication is currently in `abstractagent`, not across Runtime
  consumers generally.

What replaced it:

- Nothing new in Runtime. Higher-level packages should keep owning their own
  factories or factor a shared helper in their own package.

Why it should not be built here:

- A public `create_agent_runtime(...)` would encode agent-level assumptions into
  the lower-level Runtime package.
- Sibling packages do not share one uniform shape anyway: some use
  approval/passthrough executors or other host policy that a tiny Runtime
  convenience wrapper would not capture cleanly.

If this idea returns later, it should start in `abstractagent` as a package-
owned helper rather than a new public Runtime API.
