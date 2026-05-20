# 023 Truthful Local Media Residency Boundaries

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-20
- Origin: moved from `docs/backlog/proposed/2026-05-20_truthful_local_media_residency_boundaries.md`

## Goal

Keep local media residency honest and machine-readable so Runtime does not imply reusable warmup where the execution
mode cannot actually reuse the previous state.

## What Shipped

- Local unsupported media residency payloads now include:
  - `execution_mode="local_one_shot_subprocess"`
  - `requires_long_lived_server=true`
  - `config_hint` pointing at `ABSTRACTCORE_SERVER_BASE_URL`
- Optional residency failures (`required=false`) now complete durably with:
  - `status_hint="warning"`
  - `degraded=true`
- Local media normalization now separates orchestration identity from actual media backend identity:
  - `runtime_provider` / `runtime_model`
  - `media_provider` / `media_model`
- Local subprocess metadata no longer stamps the runtime chat model into generic media identity fields.

## Current Code Pointers

- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `src/abstractruntime/integrations/abstractcore/media_subprocess.py`
- `src/abstractruntime/integrations/abstractcore/effect_handlers.py`
- `tests/test_model_residency_control_plane.py`
- `tests/test_multimodal_abstractcore_integration.py`

## Validation

Focused validation run on 2026-05-20:

- `pytest -q tests/test_abstractcore_run_facade.py tests/test_model_residency_control_plane.py tests/test_multimodal_abstractcore_integration.py tests/test_abstractcore_host_facade.py`
- Result: `61 passed in 0.27s`

## Completion report

### Date

2026-05-20

### Summary

The original proposal was accurate and is now implemented in the Runtime integration layer. Runtime no longer reports
unsupported local media residency as an unclassified soft failure, and local media-only outputs no longer blur the
runtime orchestration model with the actual media backend model.

### Behavior changes

- Local `MODEL_RESIDENCY` responses for `image_generation`, `tts`, and `stt` now explain why warmup is unsupported.
- Optional local media warmup remains non-fatal but is explicitly marked as a warning/degraded result.
- Media-only responses expose both runtime-side and media-backend identity fields so upper layers can render honest
  badges and diagnostics.

### Tests

- `tests/test_model_residency_control_plane.py`
- `tests/test_multimodal_abstractcore_integration.py`

### Docs

- `docs/integrations/abstractcore.md`
- `docs/api.md`
- `docs/faq.md`

### Residual risks

- Consumers outside this repo still need to switch their rendering logic to prefer `media_provider` / `media_model`
  over legacy assumptions about top-level metadata.
- This item does not migrate host route code by itself; it only makes Runtime results truthful once Runtime executes
  the work.

### Follow-ups

- Keep host migrations focused on using Runtime-owned execution paths instead of synthesizing media history after the
  fact.
