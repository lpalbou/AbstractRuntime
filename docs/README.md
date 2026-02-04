# Documentation

This folder contains **user-facing docs** (how to use AbstractRuntime) and **maintainer docs** (ADRs/backlog).

## Start here

- `../README.md` — install + quick start
- `getting-started.md` — first steps (recommended)
- `architecture.md` — how the runtime is structured (with diagrams)
- `proposal.md` — design goals and scope boundaries

## Guides

- `manual_testing.md` — manual smoke tests and how to run the test suite
- `integrations/abstractcore.md` — wiring `LLM_CALL` / `TOOL_CALLS` via AbstractCore
- `tools-comms.md` — enabling the optional comms toolset (email/WhatsApp/Telegram)

## Features (reference)

- `evidence.md` — artifact-backed evidence capture for external-boundary tools
- `mcp-worker.md` — MCP worker CLI (`abstractruntime-mcp-worker`)
- `snapshots.md` — snapshot/bookmark model and stores
- `provenance.md` — tamper-evident hash-chained ledger
- `limits.md` — runtime-aware `_limits` namespace and APIs
- `workflow-bundles.md` — `.flow` bundle format (VisualFlow distribution)

## Maintainers

- `adr/README.md` — architectural decisions (why)
- `backlog/README.md` — implemented and planned work items (what/how)
