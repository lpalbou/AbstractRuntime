# 0043 Runtime vision adapter and batch surface

**Status**: Completed  
**Completed**: 2026-06-13  
**Priority**: High  
**Depends on**: 024_runtime_owned_run_scoped_media_execution (completed), 0042_core_vision_upscale_and_parameter_surface (completed)  
**Related ADRs**: 0004_runtime_owns_run_scoped_media_execution_truth

---

## Goal

Surface the latest AbstractCore and AbstractVision media contract through
AbstractRuntime without duplicating package-owned capability logic.

## Scope

- Extend Runtime discovery to proxy installed compatible vision adapters from
  AbstractCore.
- Forward the latest vision request fields through local and remote Runtime
  paths:
  - task-specific routes: `text_to_image`, `image_to_image`, `image_upscale`,
    `text_to_video`, `image_to_video`
  - ordered `lora_adapters`
  - batch controls `count` / `n` and `seeds`
  - video controls `guidance_2` and `flow_shift`
- Ensure local subprocess isolation covers the same image/video task surface as
  the in-process Runtime path when the subprocess path is selected.
- Keep Runtime thin: capability truth, adapter compatibility, and model-task
  support remain owned by AbstractCore / AbstractVision.

## Acceptance criteria

- [x] Runtime discovery facade exposes vision adapter inventory with
  model/task/provider filters.
- [x] Local and remote Runtime media execution preserve `lora_adapters`, batch
  controls, and route-specific video parameters.
- [x] Local subprocess execution supports task-compatible image edit/upscale and
  preserves progress/output semantics.
- [x] Focused Runtime contract tests cover discovery, request forwarding, and
  subprocess task routing.

## Implementation summary

- Added `list_vision_adapters(...)` to the public discovery facade and local
  discovery helpers.
- Extended local and remote Runtime/Core media forwarding for `count` / `n`,
  `seeds`, ordered `lora_adapters`, and video `flow_shift`.
- Widened local subprocess media execution to cover image edit, image upscale,
  text-to-video batch, and image-to-video batch paths while keeping the durable
  artifact/result contract unchanged.
- Extended VisualFlow media-node lowering so authored workflows can pass the new
  request fields without Gateway or higher clients rebuilding lower-package
  semantics.

## Validation

1. `PYTHONPATH=src:../abstractcore:../abstractvision/src pytest -q tests/test_abstractcore_discovery_facade.py tests/test_multimodal_abstractcore_integration.py tests/test_visualflow_media_nodes.py`
2. Result at completion time: `102 passed`

## Follow-up notes

- Gateway can now proxy adapter discovery and batch/LoRA media fields without
  rebuilding compatibility logic.
- Runtime still treats adapter/model compatibility as lower-package truth; any
  future provider-specific adapter catalog expansion belongs in AbstractCore and
  AbstractVision first.
