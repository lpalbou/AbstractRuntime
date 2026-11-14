# 1214: Project stable diary-resolution verdicts across replay

> Package: AbstractMemory, AbstractRuntime, AbstractGateway, AbstractEntity

## Metadata
- Created: 2026-11-14
- Status: Planned — in progress
- Completed: N/A
- Source: operator release recovery, root backlog 1209

## ADR status
- Governing ADRs: framework ADR 0050/0051 release gates; existing package contracts.
- ADR impact: implementation correction within existing contracts.

## Current code reality

The diary handler returned the resolution verdict only after graph projection. Projection lacked verdict/target fields, and retry validation against later entries could change a previous success into already-resolved.

## Scope and expected outcome

Use the original chain entry and its preceding prefix on replay; derive one resolution descriptor for projection/result. Preserve private target/title rules. On target metadata failure complete the chain act with labelled fallback and defer projection so retry repairs it. Raise Memory floor to 0.5.2.

Deferred governance, spoken-language, Core reasoning/usage, composer redesign and breaking TUI migration remain outside this item.

## Validation

48 real-home targeted tests pass, including four privacy combinations, immutable replay, future target refusal, duplicate resolution, target-read/projection failure repair. Independent Runtime review approved. No historical projection backfill.

## Release acceptance

Code, tests and coredoc reach remote main; all exact-commit gates green; changed artifacts published in dependency order. Root/site publication remains held for operator GO.
