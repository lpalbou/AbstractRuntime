# 0035 Model Residency Provider Truth For Local HTTP Clients

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21
- Origin: moved from `docs/backlog/proposed/0035_model_residency_provider_truth_for_local_http_clients.md`

## ADR status
- Governing ADRs:
  - `docs/adr/0007_runtime_relays_core_owned_model_residency_truth.md`
  - `../abstractcore/docs/adr/0008-provider-owned-model-residency-truth.md`
- ADR impact: Created Runtime ADR 0007 and Core ADR 0008 because model residency truth is a cross-repo boundary rule, not a one-off implementation detail.

## Context
AbstractFlow's Model Residency panel reads Gateway `/api/gateway/models/loaded`, which delegates to Runtime's AbstractCore host facade. For local text-generation clients, Runtime can report the default configured provider/model as resident because a local provider client exists in the Runtime process.

That is not equivalent to provider-side residency for providers whose model memory lives outside the Runtime process.
AbstractCore owns the provider implementation and any provider-specific loaded-instance truth; Runtime must not query
those provider APIs directly.

## Original code reality
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
  - `_local_residency_record(...)` always returns `resident: true`, `loaded: true`, and `state: "loaded"`.
  - `LocalAbstractCoreLLMClient.list_model_residency(...)` returns the active configured text-generation model as loaded.
  - `MultiLocalAbstractCoreLLMClient.list_model_residency(...)` returns all cached local clients as loaded.
  - `MultiLocalAbstractCoreLLMClient.unload_model_residency(...)` intentionally refuses to unload the default client and returns the warning `The default local text-generation client remains resident for runtime services.`
- Observed failure mode on 2026-05-21:
  - Flow proxy `GET /api/gateway/models/loaded` returned `provider: lmstudio`, `model: qwen/qwen3.5-35b-a3b`, `resident: true`, `loaded: true`, `state: loaded`, `default: true`, `source: abstractruntime.local`.
  - An external operator check of LM Studio's own model state showed no loaded model instances for the available models.
  - The operator saw a "Loaded Models" row and an unload button, but LM Studio did not show the model as loaded and unload was not actionable.

## Problem or opportunity
Runtime currently conflates at least three states:

- configured default provider/model;
- cached in-process Runtime provider client;
- provider-side resident model loaded into the backing server or native runtime.

For LM Studio and similar OpenAI-compatible HTTP providers, this makes the operator surface misleading. A default configured model can look loaded even when no model is resident in LM Studio.

## Proposed direction
Extend Runtime's model residency contract so provider-side truth is separated from Runtime configuration/cache state.

Possible shape:

- Keep `runtime_id`, `provider`, `model`, and `default` for Runtime identity/configuration.
- Add explicit fields such as `provider_resident`, `provider_residency_verified`, `provider_instance_ids`, and `cache_state`.
- Treat `resident` / `loaded` as true only when the model is actually provider-resident, or rename the Runtime-cache state so consumers cannot confuse it with provider residency.
- For providers with native loaded-instance APIs, expose that truth through an AbstractCore-owned public contract before
  Runtime consumes it.
- When AbstractCore does not expose provider-side residency truth, report an unverified cache/config state instead of
  asserting provider residency.
- Do not use AbstractCore provider metadata, provider names, `base_url`, default ports, model catalogs, or cached
  Runtime clients as loaded-state evidence.
- Make default-client unload semantics explicit: either disable unload for configuration-only rows or delegate provider
  unload through an AbstractCore-owned contract while preserving the Runtime default config.

## Why it might matter
Operator UIs, Gateway routes, and workflow control-plane steps need truthful residency semantics. If Runtime reports a configured LM Studio default as loaded, users infer model memory is occupied and expect unload to free it. That is not necessarily true.

This also affects future automated model lifecycle decisions: incorrect residency can lead to skipped warmups, confusing no-op unloads, or false memory-pressure explanations.

## Promotion criteria
Promote to `planned/` when one of these is true:

- Flow/Gateway operators need reliable loaded-model state for LM Studio or other HTTP-backed local providers.
- A Runtime consumer needs to distinguish configured defaults from resident provider models.
- AbstractCore exposes or stabilizes provider-side loaded-instance APIs for the relevant providers.

## Validation ideas
- Add Runtime unit tests for `MultiLocalAbstractCoreLLMClient` where the underlying provider client is cached but
  AbstractCore exposes no provider-residency truth.
- Add a test where an AbstractCore-owned generic residency contract reports a loaded instance and Runtime marks provider
  residency true.
- Add a default-client unload test proving the response explains configuration/cache state separately from provider unload.
- Add Gateway contract passthrough coverage once Runtime response fields are stable.
- Add Flow frontend coverage only after Runtime/Gateway fields exist.

## Non-goals
- Do not make Flow infer provider residency by scraping provider-specific endpoints directly.
- Do not make Gateway own provider-specific loaded-model semantics; Gateway should relay Runtime's host-facade truth.
- Do not remove Runtime default provider/model configuration.
- Do not force all providers to implement native loaded-instance probes before improving the contract for providers that already can.

## Guidance for future agents
Re-check AbstractCore provider APIs before implementation. If provider-side residency remains private or unavailable,
keep the new Runtime response fields conservative and explicit rather than overclaiming. Coordinate any response-shape
changes with Gateway and Flow contract consumers.

## Completion report

### Date

2026-05-21

### Summary

Runtime now separates Runtime client cache/configuration from provider-side residency for local text-generation clients.
Runtime does not contact provider-specific HTTP APIs directly, does not maintain its own provider-name taxonomy, and
does not infer residency from AbstractCore metadata, `base_url`, ports, catalogs, or cached client existence.

Runtime consumes only AbstractCore-owned residency truth. In local mode that means the concrete AbstractCore provider
object's public `get_model_residency(...)` contract. If that contract is absent, fails, or returns unverified truth,
Runtime reports the cached client as `runtime_cached=true` but `loaded=false` and `resident=false`.

AbstractCore was updated in the same pass so text providers have a public residency hook. The default Core hook is
unknown/fail-closed, MLX and Hugging Face report in-process loaded state, LM Studio reports loaded instances from inside
the LM Studio Core provider, and Core server `/acore/models/*` uses the same provider-owned truth.

### Behavior changes

- `_local_residency_record(...)` now emits explicit cache and provider-truth fields:
  - `runtime_cached`
  - `cache_state`
  - `provider_residency_verified`
  - `provider_resident`
  - `provider_residency_source`
- A cached Runtime client without verified AbstractCore provider residency reports
  `state="provider_residency_unknown"`, `resident=false`, and `loaded=false`.
- Verified AbstractCore provider residency reports `state="provider_loaded"` or `state="provider_not_loaded"`.
- `loaded_new` no longer means "a Runtime client object was created." Runtime exposes `runtime_cache_loaded_new` for
  that cache event and only sets `loaded_new=true` when Core verifies provider residency after creating the cache.
- `unloaded` no longer means "a Runtime client object was evicted." Runtime exposes `runtime_cache_unloaded` for that
  cache event. Local Runtime does not call provider-specific unload methods directly.
- Local and multi-local unload paths preserve default Runtime-client behavior and do not call provider-specific unload
  methods directly from Runtime.

### Files and symbols touched

- `src/abstractruntime/integrations/abstractcore/llm_client.py`
  - `_local_residency_record`
  - `LocalAbstractCoreLLMClient.list_model_residency`
  - `LocalAbstractCoreLLMClient.load_model_residency`
  - `LocalAbstractCoreLLMClient.unload_model_residency`
  - `MultiLocalAbstractCoreLLMClient.list_model_residency`
  - `MultiLocalAbstractCoreLLMClient.load_model_residency`
  - `MultiLocalAbstractCoreLLMClient.unload_model_residency`
- `tests/test_model_residency_control_plane.py`
- `docs/backlog/README.md`
- `docs/adr/0007_runtime_relays_core_owned_model_residency_truth.md`
- `../abstractcore/abstractcore/core/interface.py`
- `../abstractcore/abstractcore/providers/lmstudio_provider.py`
- `../abstractcore/abstractcore/providers/mlx_provider.py`
- `../abstractcore/abstractcore/providers/huggingface_provider.py`
- `../abstractcore/abstractcore/server/app.py`
- `../abstractcore/docs/adr/0008-provider-owned-model-residency-truth.md`

### Validation

- `pytest -q tests/test_model_residency_control_plane.py`
  - Result: `15 passed in 0.14s`
- `python -m compileall -q src/abstractruntime/integrations/abstractcore/llm_client.py tests/test_model_residency_control_plane.py`
  - Result: passed
- Adjacent validation:
  - `pytest -q tests/test_abstractcore_host_facade.py tests/test_remote_llm_client.py`
  - Result: `25 passed in 0.14s`
- Cross-repo AbstractCore validation:
  - `pytest -q tests/providers/test_lmstudio_unload_model_native_rest_unit.py tests/server/test_server_loaded_runtime_control_plane.py tests/server/test_server_model_residency_control_plane.py`
  - Result: `17 passed in 0.28s`
  - `python -m compileall -q abstractcore/core/interface.py abstractcore/providers/mlx_provider.py abstractcore/providers/huggingface_provider.py abstractcore/providers/lmstudio_provider.py abstractcore/server/app.py tests/providers/test_lmstudio_unload_model_native_rest_unit.py tests/server/test_server_loaded_runtime_control_plane.py`
  - Result: passed

### AbstractCore decision

AbstractCore is the only layer that may inspect provider-native residency state. Runtime intentionally does not inspect
LM Studio's native REST API or any other provider-specific loaded-instance API. Positive provider-side truth for
providers whose model memory is outside Runtime now comes through Core provider code and public Core response fields.

### Residual risks

- Providers whose model memory is outside Runtime remain unverified in local Runtime mode until their AbstractCore
  provider implements positive provider-residency truth.
- Runtime now returns cache/config rows with `loaded=false`; hosts should continue to render those as cache/config state,
  not as loaded provider models.

### Follow-ups

- Extend additional AbstractCore providers behind `get_model_residency(...)` as needed. Runtime must keep consuming the
  generic Core contract rather than probing provider APIs itself.

## Follow-up report

### Date

2026-05-22

### Summary

Runtime tightened the completed residency work after Core added more explicit load/preload/unload and prompt-cache
surfaces. Runtime now exposes `get_model_residency_capabilities(...)` through the AbstractCore host facade so Gateway and
higher-level apps can ask Runtime which residency tasks are relayable before rendering controls.

`MODEL_RESIDENCY` load effects now require Core-derived proof that the requested model is loaded. If Core returns success
without `loaded/resident/provider_resident=true`, Runtime marks the load result `ok=false`, `status_hint="warning"`, and
`degraded=true`; required loads fail the step instead of silently continuing on unverified truth.

Multi-local unload responses also stop carrying stale provider-resident fields after Runtime evicts a cached client. The
post-unload runtime record is not provider-residency proof; it is an unloaded/not-cached Runtime cache state.

### Validation

- `PYTHONPATH=src pytest -q tests/test_model_residency_control_plane.py`
- Covered verified loaded, verified not-loaded, unknown provider residency, unsupported media residency tasks, capability
  descriptors, and stale unload-record prevention.
