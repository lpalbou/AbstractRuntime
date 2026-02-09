# Documentation

This folder contains **user-facing docs** (how to use AbstractRuntime) and **maintainer docs** (ADRs/backlog).

If you are new: read `getting-started.md` → `api.md` → `architecture.md`.

## Ecosystem

AbstractRuntime is part of the wider AbstractFramework ecosystem:
- AbstractFramework umbrella: [lpalbou/AbstractFramework](https://github.com/lpalbou/AbstractFramework)
- AbstractCore (LLM + tools): [lpalbou/abstractcore](https://github.com/lpalbou/abstractcore)

In this repo, the AbstractCore wiring lives under `src/abstractruntime/integrations/abstractcore/*` and is documented in `integrations/abstractcore.md`.

## Start here

- `../README.md` — install + quick start
- `getting-started.md` — first steps (recommended)
- `api.md` — public API surface (imports + pointers)
- `architecture.md` — how the runtime is structured (with diagrams)
- `proposal.md` — design goals and scope boundaries

## Guides

- `faq.md` — common questions (recommended)
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

- `../CHANGELOG.md` — release notes
- `../CONTRIBUTING.md` — how to build/test and submit changes
- `../SECURITY.md` — responsible vulnerability reporting
- `../ACKNOWLEDGMENTS.md` — credits
- `../ROADMAP.md` — prioritized next steps
- `adr/README.md` — architectural decisions (why)
- `backlog/README.md` — implemented and planned work items (what/how)
