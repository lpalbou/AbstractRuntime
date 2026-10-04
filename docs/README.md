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
- `troubleshooting.md` — symptom-oriented setup, runtime, and integration fixes
- `proposal.md` — design goals and scope boundaries

## Guides

- `faq.md` — common questions (recommended)
- `manual_testing.md` — manual smoke tests and how to run the test suite
- `artifacts.md` — Runtime artifact identity, descriptors, provenance, catalog search, and access stats
- `integrations/abstractcore.md` — wiring `LLM_CALL` / `TOOL_CALLS`, the `config_facade` models/engines/host-jobs passthroughs for hosts, the `structured_facade` response_format check for hosts, cached sessions with per-session prompt-cache listing/clearing, host memory snapshots, model-residency listings, locks and model switching, live token streaming, workspace-scoped tools, durable bloc prompt-cache control, media inputs, generated media outputs, video progress events, and tool approval waits via AbstractCore
- `tools-comms.md` — enabling the optional comms toolset (email/WhatsApp/Telegram)
- `email.md` — email for each user: run-scoped account binding and the host resolver, sending without asking (self and pre-authorised recipients), the durable event inbox and mail feeder, the `email.received@1` trigger, and the send-email action
- `tool-approval.md` — tool risk tiers, the run-policy rank ceiling, and per-call refiners (the `send_email` recipient rule: self and pre-authorised recipients)
- `api.md#workflowbundles-flow-and-visualflow-distribution` — VisualFlow compiler APIs, media nodes, and document nodes (`read_pdf` / `write_pdf` / `write_docx`)
- `api.md#sessions-and-history` — session history replay: one window, the most recent 50,000 tokens of whole turns (`HISTORY_REPLAY_MAX_TOKENS`), returned as a `ReplayedHistory` whose `.report` records what was replayed and dropped
- `api.md#run-history-bundle-export-portable-replay-artifact` — run history bundles, including `resolved_actions` summaries for cross-client capability replay

## Features (reference)

- `automations.md` — automations: a workflow run on a schedule or on request as a durable controller run; triggers (`schedule@1`, `manual@1`, `email.received@1`, entry-point sources), commands, independent or growing context, discussions forked at any occurrence (own writable workspace, the automation's workspace mounted read-only), tool approval, typed waits, notifications and retries, and the storage guarantees (create-if-absent, per-run lock, turn roots, session kinds, one writer process per store)
- `entity-runtime.md` — per-entity runtimes for summoned entities: homes, the one-writer lease, act-only diary privacy (`$act_only` refs), durable visit waits with deadlines
- `evidence.md` — artifact-backed evidence capture for external-boundary tools
- `mcp-worker.md` — MCP worker CLI (`abstractruntime-mcp-worker`)
- `snapshots.md` — snapshot/bookmark model and stores
- `provenance.md` — tamper-evident hash-chained ledger
- `limits.md` — runtime-aware `_limits` namespace and APIs
- `workflow-bundles.md` — `.flow` bundle format, VisualFlow distribution, and multi-entry fan-in metadata

## Maintainers

- `../CHANGELOG.md` — release notes
- `../CODE_OF_CONDUCT.md` — contributor conduct expectations
- `../CONTRIBUTING.md` — how to build/test and submit changes
- `../SECURITY.md` — responsible vulnerability reporting
- `../ACKNOWLEDGMENTS.md` — credits
- `../ROADMAP.md` — current status and longer-term direction
- `adr/README.md` — architectural decisions (why)
- `backlog/README.md` — implemented and planned work items (what/how)
