# Local Media Residency Bridge To Core Residency

## Metadata
- Created: 2026-05-21
- Status: Proposed

## ADR status
- Governing ADRs:
  - `docs/adr/0004_runtime_owns_run_scoped_media_execution_truth.md`
  - `docs/adr/0007_runtime_relays_core_owned_model_residency_truth.md`
  - `../abstractcore/docs/adr/0008-provider-owned-model-residency-truth.md`
- ADR impact: None for this proposal after correction. Any implementation must preserve the accepted rule that provider
  communication and residency truth are Core-owned.

## Summary

AbstractCore now owns task-aware residency control for `text_generation`, `image_generation`, `tts`, and `stt`.
Runtime should not make independent residency claims; it should ask the configured AbstractCore backend what is loaded
now and relay that live, ephemeral truth to Gateway/Flow.

This proposal is only about the remaining local Runtime gap. It is not a Core-server-pool or UUID-affinity design.
Future multi-Core routing and per-server residency affinity is tracked separately in
`0038_core_server_pool_residency_affinity.md`.

Observed live through AbstractFlow/Gateway:

- `POST /api/gateway/models/load`
- payload: `{"task":"image_generation","provider":"huggingface","model":"..."}`
- response: `ok:false`, `supported:false`, `code:"model_residency_unsupported"`
- Runtime message: `MultiLocalAbstractCoreLLMClient can keep text-generation clients warm only. Media warmup requires a long-lived remote AbstractCore server.`

## Current Code Reality

- Gateway forwards `/api/gateway/models/load` to Runtime host facade
  `load_model_residency`; Flow is not bypassing Gateway.
- `RemoteAbstractCoreLLMClient.load_model_residency(...)` forwards supported media
  residency to `/acore/models/load`, where AbstractCore server owns the provider/backend truth.
- Runtime exposes `get_model_residency_capabilities(...)` on the host facade and client layer. Local mode reports
  text-generation residency support only; remote mode reports Core-server relay support for text/image/TTS/STT and
  marks `music_generation` unsupported until Core exposes that model-residency task.
- `MultiLocalAbstractCoreLLMClient.load_model_residency(...)` short-circuits any
  non-text task with `model_residency_unsupported`, `status_hint="warning"`, and `degraded=true`.
- Local Runtime image generation normally uses
  `media_subprocess.py` with `execution_mode="local_one_shot_subprocess"`, so a
  parent-process warm cache would not currently be reused by the worker.
- Runtime remote image edits now route through AbstractCore Server `/v1/images/edits` and provider-scoped
  `/{provider}/v1/images/edits`; this is execution support, not residency support.
- Lower-layer packages provide local load/preload/unload primitives, but Runtime must not call provider packages
  directly for residency truth. Any local media residency implementation must be exposed through an AbstractCore
  contract or facade that Runtime can relay.
- AbstractCore local vision cache/catalog helpers describe available or cached model assets; they must not be treated as
  loaded/resident model truth unless Core's residency contract says so.
- Runtime does not currently model a pool of AbstractCore servers. In remote mode it stores one configured
  `server_base_url` and relays control/execution calls to that endpoint. Loaded state is ephemeral by design and should
  be checked live rather than persisted across Core restarts.

## Problem

The architectural abstraction exists below Runtime, but Runtime's local media
execution mode is not fully connected to Core-owned media residency. The result is contradictory behavior:

- Gateway discovery can advertise media residency support that local Runtime cannot actually honor.
- Runtime local execution rejects the same media residency request by design unless a shared Core-owned backend exists.
- Flow can only show an unsupported ledger result even though Core can report live media residency in a compatible
  long-lived process.
- Remote Core-server mode should not be treated as blocked by this proposal: Runtime can simply ask that configured
  Core server which models are loaded right now.

## Goals

- Make local Runtime media residency genuinely usable by relaying Core-owned live residency truth, or make Gateway
  discovery truthfully advertise that only configured Core-server mode supports the task.
- Keep Flow thin: no direct AbstractCore or AbstractVision imports from Flow.
- Preserve ledger truth: workflow warmup must remain a `model_residency` effect,
  not a hidden operator call.
- Preserve provider truth: Runtime must never infer media loaded state from catalogs, default configuration, process
  cache, or direct provider/native API calls.
- Preserve ephemeral semantics: a loaded model is a live Core state, not durable configuration, and Runtime should check
  it live after restarts instead of trying to recover stale loaded-state records.

## Proposed Direction

Choose one explicit runtime mode and make all three surfaces agree:

1. Remote Core server mode: keep current behavior, but Gateway should use Runtime's
   `get_model_residency_capabilities(...)` plus actual `/acore/models/*` responses
   before advertising media residency. Runtime relays `/acore/models/*` truth from that configured server.
2. Local Core-facade mode: if Runtime instantiates or receives a Core-owned local facade that exposes media residency
   and also performs the media generation, Runtime may relay that Core contract for `image_generation`, `tts`, and
   `stt`.
3. Local one-shot mode: if generated media still runs through a fresh subprocess or any path that cannot share Core's
   live resident state, Runtime should keep local media residency unsupported.

This is not about guaranteeing a loaded model forever. It is about avoiding a false claim at the moment Runtime answers:
Runtime should answer from Core's current state, and after a Core restart that answer may legitimately become empty.

## Acceptance Criteria

- A local Runtime deployment that advertises `supports.image_generation=true`
  can successfully load, list, use, and unload a Hugging Face/Diffusers image
  model through `model_residency` with loaded state supplied by Core-owned
  residency truth.
- Equivalent task-specific truth is available for `tts` and `stt` before Runtime advertises those local tasks as
  supported.
- `music_generation` residency remains unsupported until AbstractCore exposes an equivalent residency contract for that
  task.
- `model_residency` load/list/unload and generated-image execution share the
  same Core-owned facade, server, or persistent worker when local media residency is
  enabled.
- If local media residency is not enabled, Gateway discovery reports media
  residency unsupported rather than relying on Flow to discover it after a run.
- Loaded state is checked live from Core; Runtime does not persist and replay loaded-state claims across AbstractCore
  restarts.
- Tests cover both local unsupported/truthful mode and local supported/warm-cache
  mode.

## Non-goals

- Do not make Runtime query AbstractVision, Diffusers, provider servers, or provider-native APIs directly for residency
  truth.
- Do not treat local model catalogs, downloaded weights, or cache directories as loaded/resident model state.
- Do not advertise local media residency until generation and residency controls share the same Core-owned backend.
- Do not solve multi-Core server pools, `core_uuid -> server` routing, or pool-wide warmup here; see
  `0038_core_server_pool_residency_affinity.md`.

## Validation

- Runtime unit tests for `MultiLocalAbstractCoreLLMClient.load_model_residency`
  with `task=image_generation`, `tts`, `stt`, and `music_generation`, including unsupported truth for tasks without a
  Core residency contract.
- Integration test proving a model loaded by `model_residency` is reused by a
  subsequent local generated-image `LLM_CALL`.
- Gateway contract test proving support flags match Runtime behavior for local
  and remote modes.
- Runtime capability tests proving unsupported local media tasks are visible before hosts decide which warmup controls to show.
- Restart/liveness check proving Runtime asks Core for current loaded state instead of replaying a stale loaded result.
