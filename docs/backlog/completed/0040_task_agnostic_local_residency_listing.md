# 0040 Task-Agnostic Local Residency Listing

## Metadata
- Created: 2026-05-24
- Status: Completed
- Completed: 2026-05-24
- Origin: direct bug fix from AbstractFlow model residency UI investigation

## Goal

Make Runtime's local model-residency list operation match the Gateway/Core
control-plane contract: omitting `task` lists resident models across tasks,
while an explicit `task` remains a filter.

## What Shipped

- Local and MultiLocal Runtime residency listing now treat an omitted task as
  task-agnostic instead of silently defaulting to `text_generation`.
- Local capability-backed residency rows are normalized with verified
  `provider_resident` / `provider_residency_verified` fields when the
  capability plugin reports `loaded` or `resident`.
- AbstractFlow's loaded-model table now honors the canonical `loaded` /
  `resident` state used by Core capability records, while still respecting
  explicit negative provider-residency fields.
- Regression tests cover local and MultiLocal TTS rows appearing in an
  unfiltered residency list after a capability-backed load.

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `tests/test_model_residency_control_plane.py`
- AbstractFlow `web/frontend/src/components/ModelResidencyPanel.tsx`

## Validation

Focused validation run on 2026-05-24:

- `python -m py_compile abstractruntime/src/abstractruntime/integrations/abstractcore/llm_client.py`
- `PYTHONPATH=abstractruntime/src pytest -q abstractruntime/tests/test_model_residency_control_plane.py -q`
- `npm run build` from `abstractflow/web/frontend`
- `git diff --check` for the touched Runtime and Flow files

## Completion Report

### Summary

The Omnivoice warmup path was already returning a true resident runtime. The
row disappeared because local Runtime listing with no task returned only text
rows, and Flow then hid capability rows that used `loaded=true` /
`state=resident` instead of text-provider-specific `provider_loaded`.

The fix is at the control-plane normalization boundary, not an Omnivoice
special case.
