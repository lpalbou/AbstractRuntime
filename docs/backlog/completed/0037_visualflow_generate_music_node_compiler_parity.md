# 0037 VisualFlow Generate Music Node Compiler Parity

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21
- Origin: moved from `docs/backlog/proposed/0037_visualflow_generate_music_node_compiler_parity.md`

## ADR status
- Governing ADRs: `docs/adr/0004_runtime_owns_run_scoped_media_execution_truth.md`, `docs/adr/0005_runtime_owns_abstractcore_host_discovery_queries.md`
- ADR impact: None

## Context
Runtime already exposes AbstractCore-backed music discovery and durable run-scoped generation through completed item `029_runtime_music_generation_and_discovery_via_abstractcore.md`. Gateway and hosts can already adopt that surface through the Runtime-owned contracts.

AbstractFlow now ships a `generate_music` authoring node. Runtime must recognize and compile that node type so VisualFlow JSON can be persisted in source shape (no client-side lowering shims) and executed durably by Runtime.

## Original code reality
- `src/abstractruntime/integrations/abstractcore/run_facade.py` exposed `generate_music(...)`.
- `src/abstractruntime/integrations/abstractcore/llm_client.py` supported output specs with `modality="music"` and `task="music_generation"`.
- `src/abstractruntime/visualflow_compiler/visual/executor.py` handled `generate_image`, `generate_voice`, `transcribe_audio`, and `listen_voice`, but did not recognize `generate_music`.
- `src/abstractruntime/visualflow_compiler/compiler.py` treated `llm_call`, `generate_image`, `generate_voice`, and `transcribe_audio` as LLM-call-backed effects, but did not include `generate_music` in effect routing or output mapping.
- `tests/test_visualflow_media_nodes.py` had compiler coverage for image/voice/transcription/listen, but no `generate_music` cases.

## What shipped
- Added first-class VisualFlow lowering support for `nodeType="generate_music"` in Runtime:
  - compiles into an `EffectType.LLM_CALL` pending effect with output selector `{modality: "music", task: "music_generation"}`
  - accepts `prompt`, `music_provider`, `music_model`, `lyrics`, `duration_s`, `format`, `seed`, `num_inference_steps`, `guidance_scale`, `instrumental`, `enhance_prompt`, `auto_lyrics`, `structure_prompt`, `text_planner_mode`, and `extra` from pins or `effectConfig` (`music_backend` is a legacy alias and is rejected; use `music_provider`)
  - keeps runtime LLM `provider`/`model` separate from music provider/model selectors (no legacy fallback)
- Extended VisualFlow compiler routing so `generate_music` is treated as an LLM-call-backed effect node (like `generate_image` / `generate_voice` / `transcribe_audio`).
- Extended VisualFlow effect-result mapping so completed `generate_music` nodes populate:
  - `music_artifact` (primary) plus `audio_artifact` alias for UI compatibility
  - `artifact_ref`, `artifact_id`, `content_type`, `outputs`, `meta`, `raw`, `success`
- Added focused compiler tests beside `tests/test_visualflow_media_nodes.py`.

## Current code pointers
- `src/abstractruntime/visualflow_compiler/visual/executor.py`
- `src/abstractruntime/visualflow_compiler/compiler.py`
- `tests/test_visualflow_media_nodes.py`

## Validation
Focused validation run on 2026-05-21:

- `pytest -q tests/test_visualflow_media_nodes.py`
- Result: `9 passed in 0.05s`
- `pytest -q tests/test_multimodal_abstractcore_integration.py tests/test_abstractcore_run_facade.py`
- Result: `44 passed in 0.17s`

## Completion report

### Date

2026-05-21

### Summary

Runtime VisualFlow compilation now supports `generate_music` as a first-class node type with the same compiler parity as `generate_image` and `generate_voice`. Hosts can persist and replay VisualFlow graphs without client-side lowering shims.

### Backward compatibility

Existing flows that already persist lowered `llm_call` nodes with `output.modality="music"` continue to work unchanged.

### Boundary notes

This change keeps the package boundary clean: Runtime still integrates only with AbstractCore contracts and does not import `abstractmusic` directly.

## Follow-up report

### Date

2026-05-22

### Summary

Runtime expanded the `generate_music` VisualFlow lowering to pass through the richer AbstractMusic/Core controls now
visible in higher layers: `vocal_language`, `negative_prompt`, `sample_rate`, `bpm`, `keyscale`, `timesignature`,
`composition_plan`, `positive_styles`, `negative_styles`, and `planning`.

The compiler result sync now has focused coverage proving completed `generate_music` nodes populate `music_artifact`,
`audio_artifact`, `artifact_ref`, `artifact_id`, `content_type`, `outputs`, `meta`, `raw`, and `success`.

Adjacent media parity work also added first-class `edit_image` / `image_to_image` VisualFlow nodes. Those lower to
`LLM_CALL` with `{modality:"image", task:"image_edit"}` plus source/mask media roles, keeping image editing on the same
Runtime/Core output-selector contract instead of requiring host-side lowering.

### Validation

- `PYTHONPATH=src pytest -q tests/test_visualflow_media_nodes.py tests/test_visualflow_llm_call_multimodal_output.py`
