# Proposed: Schema-version stamps + quarantine loading (the compat contract)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: the split-runner compat rule should land in
  docs/architecture.md on promotion.

## Context
Version-skew incidents are a recorded class (stale serving processes, the
diary_type clamp copies, legacy phase/overlay aliases). Run files and
StepRecords carry no schema version (core/models.py:164-300; artifacts DO —
artifacts.py:39).

## Current code reality (line-verified 2026-07-12)
Unknown enum value on load today:
- JSON store RAISES OUT OF THE SCAN (json_files.py:209/217 sit outside the
  try at 201-205) — one v-next run file kills every list_runs() and
  scheduler poll for the WHOLE directory.
- SQLite silently returns None (sqlite.py:370-373) — a zombie run that
  never ticks again, no log.
Two stores, two failure modes, both wrong.

## Problem or opportunity
A split-runner deployment (API process + runner process, already real
gateway-side) upgrading one side first WILL eventually write a value the
other side cannot parse; today that takes down directory scans or silently
strands runs.

## Proposed direction
- Write `schema: "run.v1"` on save (absence tolerated as v1).
- Unknown status/wait-reason/schema on load → PER-RUN QUARANTINE: skip the
  run, loud counter (0054) + warning naming the file and the unknown value,
  a listable quarantine surface — never a scan crash, never a silent zombie.
- Document the rule: v-next reads v-prev = yes; v-prev meeting v-next =
  refuses that run loudly, everything else keeps working.

## Why it might matter
The blast radius today is every run in a directory — the worst failure
shape in the storage layer, triggered by routine upgrades.

## Promotion criteria
Any of: a real split-runner version-skew incident; a planned enum/schema
change to RunState/StepRecord; the 1.0 wave opening.

## Validation ideas
Pins: a run file with an unknown status → list_runs returns the others +
quarantine counter; SQLite same; round-trip with schema stamp absent/present.

## Non-goals
No migration framework; no bidirectional compat promises beyond the stated
rule.

## Guidance for future agents
Fix the JSON-store scan-crash placement (the two lines outside the try)
even if the full item stays proposed — that half is a bug, not a feature.
