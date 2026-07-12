# Proposed: Extract the memory-span handlers from the Runtime class

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None (structure only)

## Context
`core/runtime.py` is ~4,975 lines; roughly 1,700 of them are
span/compaction domain logic inside the kernel class:
`_handle_memory_query/tag/compact/note/rehydrate` + render helpers
(runtime.py:3213-4975). A `memory/` package already exists
(active_memory.py, compaction.py, memact_composer.py).

## Current code reality
The handlers are self-contained methods registered like any effect handler;
they also carry the cross-run memory-owner write race (planned 0050 item 3
— fix the RACE there regardless of this move).

## Problem or opportunity
The kernel class carries a second domain; after extraction, runtime.py at
~3,200 lines is an actual kernel (tick, waits, resume, control,
subworkflow) — the size that was RULED acceptable to keep whole.

## Proposed direction
Move to `memory/span_effects.py` as handler factories taking a narrow
context (run_store, artifact_store, scope resolver), registered in
`_register_builtin_handlers` exactly like the integrations handlers.

## Why it might matter
Halves the kernel's surface for reviewers; groups span logic with its
siblings; zero behavior change expected (tests target effect behavior).

## Promotion criteria
After planned 0045/0046/0050 land (do not shuffle code the concurrency wave
is actively editing); or bundled into a 1.0 structure wave with 0049's
pattern.

## Validation ideas
Full suite green; effect-handler registration parity pin; no import cycles
(memory/ must not import core beyond models/spec, verify).

## Non-goals
Do NOT split tick/resume/waits out of the kernel (explicitly ruled
against); do not change span semantics.

## Guidance for future agents
Sequence AFTER the concurrency items; the move is mechanical only if the
owner-write serialization (0050) already landed.
