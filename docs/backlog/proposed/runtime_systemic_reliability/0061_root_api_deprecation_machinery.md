# Proposed: Root-API deprecation machinery + public/internal split

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: the public-surface definition should land in docs on
  promotion (what is public = the root list + documented module paths;
  underscore-prefixed is fair game).

## Context
The root `__init__.py` exports ~120 names and grows with each wave — often
deliberately (the phase/tool-grant block exists so the GATEWAY imports from
the root), which makes the root a cross-repo contract.

## Current code reality (verified 2026-07-12)
- No `__getattr__` shim, no `warnings.warn` deprecation pattern anywhere at
  the root.
- Hand-written migration shims exist with death dates ("DIES BEFORE
  RELEASE": identity/lease.py; LEGACY_PHASE_ALIASES; overlay key aliases)
  — release is approaching and their deaths need scheduling.

## Problem or opportunity
The structure items (0049 split, 0059/0060 moves) are only safe to stage
incrementally if renamed/moved names can be served with a warning instead
of breaking consumers.

## Proposed direction
- A ~10-line module `__getattr__` serving renamed/moved names with a
  DeprecationWarning (one table: old name → new location).
- A one-page public-vs-internal policy.
- Schedule the existing shims' removals (lease shim, phase aliases, overlay
  aliases) against the release calendar.

## Why it might matter
It is the enabling mechanism for every structural move in this track, and
it is what makes 1.0's compatibility promise cheap to keep.

## Promotion criteria
BEFORE starting 0049's split (its facade wants this table), or when the
release wave opens.

## Validation ideas
Pin: importing a moved name fires exactly one DeprecationWarning and
returns the right object; the public list is import-smoke-tested.

## Non-goals
No breaking removals in the same wave that introduces the machinery.

## Guidance for future agents
Keep the table small and dated; a deprecation shim without a removal date
is how permanent aliases are born (the drift-pin lesson).
