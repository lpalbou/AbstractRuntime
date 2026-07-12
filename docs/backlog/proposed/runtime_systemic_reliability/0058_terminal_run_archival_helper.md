# Proposed: Terminal-run archival helper (explicit, marker-recorded)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None (extends the existing retention posture in planned 025;
  archival stays an explicit operator act, never automatic)

## Context
Nothing prunes by default (correct: append-only is the audit and identity
posture). Deletable store protocols exist; retention policy is
host-improvised (planned 025 is the retention/purge contract item).

## Current code reality (verified 2026-07-12)
- The JSON run store's own docstring warns to archive/prune before ~10k
  files (json_files.py:283-290); list_runs scans are linear in TOTAL files
  including terminal ones.
- 50 residents' subruns + a 1s scheduler poll + the gateway's multiple
  scans per 250ms all pay that linear cost.
- A data-root purge under a live sidecar already happened once (operator
  cleanup by hand — the class this helper prevents).

## Problem or opportunity
The directory cliff arrives with fleet scale, and the absence of a paved
archival path invites ad-hoc `rm` under live processes.

## Proposed direction
One runtime-owned mechanic: `archive_terminal_runs(before=..., dest=...)` —
moves run+ledger files of TERMINAL runs to an archive subdir (artifacts
stay), host-invoked, marker/record-emitting so the act is auditable.
Scheduler/list scans then see only live runs.

## Why it might matter
Prevents the ~10k-file cliff AND the purge-under-live-instance class with
one safe verb operators can actually reach.

## Promotion criteria
Fleet deployments approaching thousands of terminal runs per store, or
planned 025's wave opening (fold this into it rather than shipping
separately if timing aligns).

## Validation ideas
Archive → list_runs excludes archived; ledgers readable from the archive
path; idempotent re-run; refuses non-terminal runs.

## Non-goals
No automatic/scheduled pruning; no artifact deletion; no in-place ledger
compaction (ruled against).

## Guidance for future agents
Coordinate with planned 025 (retention contract) — this is its mechanical
half, not a competing policy.
