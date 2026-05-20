# Proposed: Agent Runtime Convenience Constructor

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## Context

The old `015_agent_integration_improvements` item no longer fits as planned runtime work.

Most of the important integration correctness work is already done:

- agent tool execution can go through durable `TOOL_CALLS`
- runtime tool execution is host-configured through `ToolExecutor`
- agent state persistence is not blocked on runtime internals

The main remaining idea is much smaller: a convenience constructor such as
`create_agent_runtime(...)` that would wrap `create_local_runtime(...)` plus
`MappingToolExecutor.from_tools(...)` for simple local setups.

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
