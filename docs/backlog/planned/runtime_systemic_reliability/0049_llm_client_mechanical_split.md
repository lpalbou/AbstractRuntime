# Planned: Mechanical split of llm_client.py (11.8k lines → llm/ subpackage + facade)

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Priority: P0 (merge-hazard + review-blindness risk; splittable today, harder every week)

## ADR status
- Governing ADRs: None
- ADR impact: None (structure only; public paths preserved)

## Context
`integrations/abstractcore/llm_client.py` is 11,825 lines and is where every
growth wave lands: this session alone it absorbed volatile markers,
on_token, streaming think-block parity, and structured-output stream gating
— four unrelated features in one file. At this size every edit is a
conflict magnet and reviews skim.

## Current code reality (band map verified 2026-07-12)
Clean internal seams already exist:
- grounding/tz/country/envelope: lines ~377-1066 (~690 lines, STDLIB-ONLY);
- subprocess media runners: ~1531-1879;
- prompt-cache export/import: ~1997-2192;
- blocs + KV artifacts: ~2192-2816;
- model residency: ~2816-3763 (~950);
- generated-artifact normalization: ~3763-4823 (~1,060);
- streaming/think-block normalization: ~4823-5045;
- `LocalAbstractCoreLLMClient` (~1,900), `MultiLocalAbstractCoreLLMClient`
  (~1,400), `RemoteAbstractCoreLLMClient` (~2,940), 35-method protocol.
LAYERING BONUS: `core/runtime.py:562` imports four grounding functions from
this file — a core→integrations INVERSION. The functions are stdlib-only;
moving them DOWN (e.g. `core/grounding.py`) kills the inversion as part of
the split.

## Problem
Six-plus responsibilities in one filename; the growth pattern guarantees it
gets worse; one real layering inversion hides inside it.

## What we want to do
New package `integrations/abstractcore/llm/`:
`grounding.py` (or move to core), `residency.py`, `blocs.py`,
`prompt_cache.py`, `media_artifacts.py`, `normalize.py`, `protocol.py`,
`local_client.py`, `multilocal_client.py`, `remote_client.py`.
`llm_client.py` becomes a PURE RE-EXPORT FACADE so every existing import —
including private names other repos/tests import — keeps working. No
signature changes, no behavior changes, ONE commit gated by the full suite.

## Why
Stops the monolith while it is still mechanically splittable; ends the
"everything lands here" accretion; removes the core→integrations inversion.

## Requirements
- Zero behavior change (the 1,145-test suite is the gate).
- The facade re-exports EVERYTHING currently importable, private names
  included (grep consumers in gateway/agent/code trees first).
- The grounding move updates `core/runtime.py:562` to the new home.

## Suggested implementation
Band-by-band cut in one working session; run the suite after each band
lands in the new module with the facade updated.

## Scope
The split + facade + core grounding import fix + a consumers grep.

## Non-goals
- Do NOT merge the three client classes (genuinely different transports —
  explicitly ruled against).
- Do NOT redesign the 35-method protocol here (discovery-passthrough
  folding is proposed 0060).

## Dependencies and related tasks
- 0044 (band map), proposed 0060 (protocol shrink), the memory-span
  extraction (proposed 0059) follows the same pattern for core/runtime.py.

## Expected outcomes
No module in the package over ~3,000 lines except core/runtime.py (whose
size is accepted by ruling); future llm features land in their band's file.

## Validation
- Full suite green; gateway + agent + code suites green against the facade
  (coordinate a same-day check).
- `python -c` import matrix over the old paths.

## Progress checklist
- [ ] Consumer import grep (gateway/agent/code/flow trees)
- [ ] Bands moved, facade complete
- [ ] core/runtime.py grounding import re-pointed
- [ ] Cross-repo import smoke
- [ ] Fable5 adversary folded
