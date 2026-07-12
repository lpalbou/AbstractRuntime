# Proposed: VisualFlow compiler decomposition (visual_to_flow + the two mega-factories)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Second-largest merge-hazard site after llm_client (planned 0049), and the
one where a closure-capture bug becomes a silent cross-run leak — that
class was already paid for once (the `_static_node_outputs` reset,
executor.py:83-86).

## Current code reality (verified 2026-07-12)
- `visual/executor.py:60` opens `visual_to_flow` — ONE function ending
  ~line 4,208, containing most of the file's 166 nested defs.
- `compiler.py` holds `_create_visual_agent_effect_handler` (915-2301,
  ~1,380 lines) and `_sync_effect_results_to_node_outputs` (2493-3482).
- Closure-captured mutable state makes units untestable except end-to-end;
  any two features touching visual nodes collide in one diff region.

## Problem or opportunity
Every visual-node feature edits the same few thousand lines; reviews skim;
captured-state bugs are the recorded failure class.

## Proposed direction
An explicit `CompileContext` (the values the closures capture: data-edge
map, node outputs, pin defaults) + per-node-type handler builders in a
registry of small functions. BAND-BY-BAND, never big-bang — the visualflow
test population is large and is the gate.

## Why it might matter
Testable units, parallel-editable node types, and structural prevention of
the capture-leak class.

## Promotion criteria
The next sizable visualflow feature wave (do the decomposition as its first
band), or two merge conflicts in this region within a week.

## Validation ideas
Full visualflow suite per band; a capture-isolation pin (two compiled flows
share zero mutable state).

## Non-goals
No node-semantics changes; no new compiler IR.

## Guidance for future agents
Highest-risk item in the track — respect the band-by-band rule and land
each band with the suite green before starting the next.
