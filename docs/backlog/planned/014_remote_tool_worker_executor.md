## 014_remote_tool_worker_executor (planned)

**Status**: Planned
**Priority**: Low
**Depends on**: 005_abstractcore_integration (completed)

---

## Goal

Decide whether Runtime still needs a generic non-MCP remote tool-job client beyond the delegated MCP support that already ships today.

---

## Current code reality

- `WaitReason.JOB` already exists.
- The `TOOL_CALLS` effect handler already turns delegated executor results into durable waits.
- `DelegatingMcpToolExecutor` already provides a real job-style delegation path.
- `McpToolExecutor` and `abstractruntime-mcp-worker` already cover a concrete remote-worker story.

So the broad "add remote worker execution" item is no longer accurate. The remaining question is narrower: should Runtime also ship a generic HTTP tool-job client for deployments that do not want MCP?

---

## Problem

Some thin-client or centralized-execution deployments may still want:

- a simple HTTP job queue instead of MCP
- runtime-owned packaging of tool batches into a documented remote job contract

But this only matters if MCP delegation is not sufficient for real adopters.

---

## Planned scope

1. Document the existing delegated MCP path as the preferred shipped solution.
2. Evaluate whether a generic non-MCP job protocol still has real consumers.
3. If yes, add a small `RemoteToolExecutor` / client abstraction that:
   - submits tool batches to a remote service
   - returns a durable `JOB` wait payload
   - stays compatible with existing host resume flows
4. Keep the contract intentionally minimal and host-agnostic.

---

## Acceptance criteria

- [ ] The backlog clearly distinguishes existing MCP delegation from any still-missing generic worker client.
- [ ] If implemented, a generic executor produces durable `JOB` waits without keeping in-flight tool work in RAM.
- [ ] Documentation explains when to use passthrough, delegated MCP, or a generic HTTP job path.

---

## Validation

1. Documentation review against the current MCP worker and delegated executor behavior.
2. If a generic client is added, unit tests with a stub transport covering submit, wait details, and resume semantics.

---

## Non-goals

- replacing the existing MCP path
- mandatory worker infrastructure
- cluster scheduling or leasing semantics

---

## Priority note

Keep this low priority unless a concrete non-MCP deployment requirement appears. The durable job-wait foundation is already in place.
