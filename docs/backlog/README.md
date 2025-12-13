# AbstractRuntime Backlog

This directory contains the organized backlog for AbstractRuntime development.

## Structure

```
backlog/
├── completed/     # Implemented and shipped items
├── planned/       # Items scheduled for future implementation
├── deprecated/    # Legacy/superseded backlog items (historical reference)
└── README.md      # This file
```

## Completed Items

| ID | Title | Description |
|----|-------|-------------|
| 001 | Runtime Kernel | Core models, spec, runtime loop (start/tick/resume) |
| 002 | Persistence and Ledger | RunStore, LedgerStore interfaces + in-memory + JSON backends |
| 003 | Wait Primitives | wait_event, wait_until, ask_user effect handlers |
| 004 | Scheduler Driver | Built-in scheduler with WorkflowRegistry, auto-resume, event ingestion |
| 005 | AbstractCore Integration | LLM_CALL, TOOL_CALLS handlers + local/remote/hybrid modes |
| 006 | Snapshots/Bookmarks | Named checkpoints with search |
| 007 | Provenance Hash Chain | Tamper-evident ledger with HashChainedLedgerStore |
| 012 | RunStore Query + Scheduler Support | QueryableRunStore protocol + list_runs + list_due_wait_until |

## Planned Items

| ID | Title | Priority | Dependencies |
|----|-------|----------|--------------|
| 011 | Subworkflow Support | High | - |
| 010 | Examples and Composition | High | - |
| 009 | Artifact Store | Medium | - |
| 013 | Effect Retries and Idempotency | Medium | - |
| 008 | Signatures and Keys | Low | 007 |
| 014 | Remote Tool Worker Executor | Low | - |

## Plan Reference

See `deprecated/abstractruntime_docs_final_02a7373b.plan.md` for the documentation finalization plan that guided the initial implementation.

