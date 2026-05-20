# 022 Model Residency Control Plane

## Metadata
- Created: 2026-05-19
- Status: Completed
- Completed: 2026-05-19
- Origin: moved from `docs/backlog/proposed/2026-05-19_model_residency_control_plane.md` during backlog housekeeping on 2026-05-20

## Goal

Add a Runtime-owned, replay-safe control plane for operational model residency without turning live residency state into
durable workflow truth.

## What Shipped

- Added `EffectType.MODEL_RESIDENCY` in the Runtime core model set.
- Added model-residency methods on local, multi-local, and remote AbstractCore clients:
  - `list_model_residency(...)`
  - `load_model_residency(...)`
  - `unload_model_residency(...)`
- Added the AbstractCore residency effect handler and registered it through normal AbstractCore effect wiring.
- Reused the normalized root-level Core URL joining logic for prompt-cache and model-residency endpoints.
- Kept local media residency truthful: one-shot local media execution reports unsupported instead of pretending that
  media models can stay warm.
- Added VisualFlow lowering for the Runtime `model_residency` effect shape.

## Current Code Pointers

- `src/abstractruntime/core/models.py`
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `src/abstractruntime/integrations/abstractcore/effect_handlers.py`
- `src/abstractruntime/integrations/abstractcore/factory.py`
- `src/abstractruntime/visualflow_compiler/compiler.py`
- `src/abstractruntime/visualflow_compiler/adapters/effect_adapter.py`
- `src/abstractruntime/visualflow_compiler/visual/executor.py`
- `tests/test_model_residency_control_plane.py`

## Validation

Current focused validation run on 2026-05-20:

- `pytest -q tests/test_model_residency_control_plane.py`
- Result: `10 passed in 0.20s`

Broader adjacent validation run on 2026-05-20:

- `pytest -q tests/test_packaging_extras.py tests/test_runtime_install_boundary.py tests/test_remote_llm_client.py tests/test_prompt_cache_modules.py tests/test_session_attachments_registry_and_open_tool.py tests/test_workflow_bundle_registry.py tests/test_model_residency_control_plane.py`
- Result: `51 passed, 3 skipped in 0.83s`

## Completion report

### Date

2026-05-20

### Summary

The residency control plane is implemented and tested in the current tree, but the backlog note remained under
`proposed/`. This move records the actual outcome: Runtime now owns a first-class residency effect and a matching
AbstractCore control contract, while still treating residency as operational state rather than durable business truth.

### Behavior changes

- Workflows can request `MODEL_RESIDENCY` through the Runtime effect system.
- Remote AbstractCore clients can call `/acore/models/*` through the normalized control-plane path.
- Local and multi-local clients are honest about what they can keep warm: text-generation clients only.
- Replay uses the recorded residency result snapshot instead of recalling live Core state.

### Tests

- `tests/test_model_residency_control_plane.py`

### Docs

- This completion supersedes the stranded proposal copy.
- The public host-facing follow-on for operator access is now completed separately in
  `docs/backlog/completed/019_runtime_host_facade_for_core_operator_surfaces.md`.

### Residual risks

- Local media residency remains intentionally unsupported; that is correct, but it means hosts must render optional
  warmup steps honestly.
- The Runtime-side truthfulness and durable host-run surfaces are now completed separately in
  `docs/backlog/completed/023_truthful_local_media_residency_boundaries.md` and
  `docs/backlog/completed/024_runtime_owned_run_scoped_media_execution.md`, but external hosts still need to adopt
  those surfaces.

### Backlog/code drift found

- The proposal text itself already described implemented behavior and even marked itself `Implemented`, but it had not
  been moved to `completed/`.

### Follow-ups

- Preserve the replay rule: residency history reuses recorded results and does not become live polling by accident.
