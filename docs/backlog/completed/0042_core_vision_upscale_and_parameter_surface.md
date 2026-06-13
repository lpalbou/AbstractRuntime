# 0042 Core Vision Upscale and Parameter Surface

## Metadata
- Created: 2026-06-07
- Completed: 2026-06-07
- Status: Completed
- Owner: AbstractRuntime
- Origin: follow-up to AbstractCore and AbstractVision releases that added richer Vision/Core models, image upscaling, and generation controls.

## ADR Status
- Governing ADRs:
  - `docs/adr/0004_runtime_owns_run_scoped_media_execution_truth.md`
  - `docs/adr/0005_runtime_owns_abstractcore_host_discovery_queries.md`
  - `docs/adr/0007_runtime_relays_core_owned_model_residency_truth.md`
- ADR impact: none. This work applies the existing Runtime-owned execution and discovery boundary to the newly exposed Core/Vision surface.

## Current Code Reality

- Runtime already exposes durable run-facade helpers for image generation, image edit, text-to-video, image-to-video, TTS, music, and STT.
- Runtime discovery already relays Core/Vision provider-model catalogs, but its local Vision task allowlist only accepts `text_to_image`, `image_to_image`, `text_to_video`, and `image_to_video`.
- Remote generated-media routing handles Core image generation/edit and video generation/edit endpoints, including async video progress, but it has no `image_upscale` route.
- VisualFlow media-node lowering forwards common image/video parameters, but not the newer Core/Vision controls such as `guidance_2` or image upscaler parameters.
- Runtime model residency support reports media tasks, but does not yet include image upscaling.

## Architecture Review

Alternatives considered:

1. **Duplicate Core's model and task registry in Runtime.**
   - Rejected because Core owns model capabilities and provider residency truth. Runtime should relay and route, not become a second registry.
2. **Let Gateway call Core's new upscaler endpoint directly.**
   - Rejected for run-scoped media work because ADR 0004 requires Runtime-owned child-run history and artifact provenance.
3. **Add a generic untyped Core pass-through media endpoint.**
   - Deferred. It would be flexible, but it would make thin-client contracts weaker and harder to validate.
4. **Extend Runtime's existing media abstractions with `image_upscale` and pass through named Core/Vision controls.**
   - Recommended. It keeps the boundary simple, preserves child-run durability, and avoids registry duplication.

## Goal

Make Runtime support the richer AbstractCore/Vision media surface without guessing model truth:

- relay `image_upscale` in Vision discovery;
- execute image upscaling through Runtime-owned child runs and Core remote/local integration paths;
- pass through new generation controls such as `guidance_2` and upscaler controls;
- expose truthful residency/task support for image upscaling;
- keep progress callbacks as ledger `abstract.progress` events and never persist callbacks.

## Acceptance Criteria

- `AbstractCoreRunFacade` exposes a durable `upscale_image(...)` helper.
- Remote Core execution routes `output={"modality":"image","task":"image_upscale"}` to `/v1/images/upscale` or `/{provider}/v1/images/upscale`.
- Local Core execution can route an upscaling output spec through the existing multimodal provider path.
- Runtime discovery accepts `task=image_upscale` for Vision provider-model catalogs and cached model filters.
- Runtime VisualFlow compilation/lowering recognizes an `upscale_image` / `image_upscale` node shape and forwards upscaler parameters.
- `guidance_2` is forwarded for image/video generation and edit paths where supplied.
- Model-residency task reporting includes `image_upscale`.
- Focused tests prove request payloads, artifact storage, discovery filters, and progress callback behavior.

## Review Checklist

- No direct Gateway/Core bypass is introduced.
- No model capability or provider residency truth is inferred in Runtime.
- Generated media outputs remain artifact-backed and run-scoped.
- Callback objects are not serialized into run vars, effect payloads, or artifacts.
- Existing image/video/music/STT tests remain green.

## Implementation Summary

- Added `image_upscale` to Runtime's Core Vision discovery task allowlist and model-residency capability descriptors.
- Added `AbstractCoreRunFacade.upscale_image(...)`, preserving parent/child run ownership and artifact provenance.
- Routed remote `output={"modality":"image","task":"image_upscale"}` calls to AbstractCore Server image upscaling endpoints, including the async job/progress path when a progress callback is active.
- Forwarded newer Core/Vision parameters including `guidance_2` for image/video generation/edit paths and upscaler controls for image upscaling.
- Added VisualFlow compiler/executor recognition for `upscale_image` / `image_upscale` nodes.
- Updated Runtime docs, FAQ, troubleshooting, and AI-readable indexes for the new Core floor and media surface.

## Validation

- `PYTHONPATH=src:../abstractcore python -m compileall -q src/abstractruntime/integrations/abstractcore src/abstractruntime/visualflow_compiler src/abstractruntime/core/runtime.py`
- `PYTHONPATH=src:../abstractcore pytest -q tests/test_packaging_extras.py tests/test_visualflow_media_nodes.py tests/test_model_residency_control_plane.py::test_model_residency_capabilities_describe_core_backed_truth_by_task tests/test_abstractcore_run_facade.py::test_run_facade_upscale_image_creates_durable_child_run_with_media tests/test_multimodal_abstractcore_integration.py::test_remote_image_upscale_uses_abstractcore_images_upscale_endpoint_and_stores_artifact tests/test_multimodal_abstractcore_integration.py::test_remote_image_upscale_with_progress_uses_core_job_endpoint tests/test_abstractcore_discovery_facade.py::test_local_vision_provider_models_accept_image_upscale_task_from_cache`
  - Result: `23 passed in 0.29s`
