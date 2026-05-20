# API reference

This document summarizes the **public Python API** of AbstractRuntime and points to the **source of truth in code**.

Public exports live in `src/abstractruntime/__init__.py`. If you are unsure what is supported for external use, start there.

Stability guideline:
- Prefer imports from `abstractruntime` (package root) and `abstractruntime.storage`.
- Deep imports from `abstractruntime.core.*` / `abstractruntime.storage.*` are fine for advanced use, but treat them as lower-stability unless they are explicitly documented/re-exported.

## Recommended imports

Core kernel:

```python
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
```

Storage helpers (common stores):

```python
from abstractruntime.storage import (
    InMemoryLedgerStore,
    InMemoryRunStore,
    JsonFileRunStore,
    JsonlLedgerStore,
)
```

Scheduler convenience wrapper:

```python
from abstractruntime import create_scheduled_runtime
```

Optional integration (requires `abstractruntime[abstractcore]`):

```python
from abstractruntime.integrations.abstractcore import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
    create_local_runtime,
)
```

See also: `getting-started.md` (end-to-end runnable examples).

## Core types (durable workflow semantics)

Implementation: `src/abstractruntime/core/models.py`, `src/abstractruntime/core/spec.py`.

- `WorkflowSpec`: in-memory workflow graph (`workflow_id`, `entry_node`, `nodes`).
- `StepPlan`: node return value (what happens next): `effect`, `next_node`, or `complete_output`.
- `Effect` / `EffectType`: durable side-effect request protocol (the runtime mediates execution).
- `RunState` / `RunStatus`: durable checkpoint for a run, persisted by a `RunStore`.
- `WaitState` / `WaitReason`: durable pause metadata for `WAIT_*` / `ASK_USER` / passthrough tool waits.

Durability invariant: `RunState.vars` must remain JSON-serializable (`src/abstractruntime/core/models.py`). For large payloads use artifacts/offloading (`src/abstractruntime/storage/artifacts.py`, `src/abstractruntime/storage/offloading.py`).

## Runtime (start / tick / resume)

Implementation: `src/abstractruntime/core/runtime.py`.

- `Runtime.start(workflow, vars=..., actor_id=..., session_id=...) -> run_id`
  - creates and persists a new `RunState`
- `Runtime.tick(workflow, run_id, max_steps=...) -> RunState`
  - executes node handlers and effects until the run becomes `WAITING`, `COMPLETED`, `FAILED`, or `CANCELLED`
- `Runtime.resume(workflow, run_id, wait_key, payload, max_steps=...) -> RunState`
  - validates the `wait_key`, writes `payload` to `WaitState.result_key` (if set), and continues from `WaitState.resume_to_node`
- `Runtime.get_state(run_id) -> RunState` and `Runtime.get_ledger(run_id) -> list[dict]`
  - host-facing read APIs for checkpoints and the append-only ledger

For the execution model (ledger records, effect outcomes, waits), see `architecture.md`.

## Scheduler convenience API

Implementation: `src/abstractruntime/scheduler/*`.

Use `create_scheduled_runtime()` for a zero-config wrapper that bundles `Runtime` + an in-process polling `Scheduler`:
- `ScheduledRuntime.run(workflow, vars=..., actor_id=..., max_steps=...) -> (run_id, state)` (`src/abstractruntime/scheduler/convenience.py`)
- `ScheduledRuntime.respond(run_id, payload) -> RunState` (resumes a waiting run using its stored `wait_key`)
- `ScheduledRuntime.stop()` (stops the scheduler thread/loop)

For time-based waits, the scheduler polls due runs via `QueryableRunStore.list_due_wait_until(...)` (`src/abstractruntime/storage/base.py`, `src/abstractruntime/scheduler/scheduler.py`).

## Storage layer (durability backends)

Interfaces: `RunStore`, `LedgerStore`, and `QueryableRunStore` are defined in `src/abstractruntime/storage/base.py`.

Included backends:
- In-memory (tests/dev): `InMemoryRunStore`, `InMemoryLedgerStore` (`src/abstractruntime/storage/in_memory.py`)
- Filesystem:
  - checkpoints: `JsonFileRunStore` (`src/abstractruntime/storage/json_files.py`)
  - append-only ledger: `JsonlLedgerStore` (`src/abstractruntime/storage/json_files.py`)
- SQLite:
  - `SqliteRunStore`, `SqliteLedgerStore` (`src/abstractruntime/storage/sqlite.py`)

Notes:
- `abstractruntime.storage` intentionally exports only the most common store types. SQLite types are available via:
  - `from abstractruntime import SqliteRunStore, SqliteLedgerStore`, or
  - `from abstractruntime.storage.sqlite import SqliteRunStore, SqliteLedgerStore`

Common decorators:
- `ObservableLedgerStore` for subscriptions (`src/abstractruntime/storage/observable.py`)
- `HashChainedLedgerStore` + `verify_ledger_chain(...)` for tamper-evidence (`src/abstractruntime/storage/ledger_chain.py`)
- `OffloadingRunStore` / `OffloadingLedgerStore` to store large values by artifact reference (`src/abstractruntime/storage/offloading.py`)

## Commands (durable control-plane inbox)

AbstractRuntime ships append-only, idempotent **command inbox** primitives designed for gateways/workers that must accept retries safely:
- models + interfaces: `CommandRecord`, `CommandStore`, `CommandCursorStore` (`src/abstractruntime/storage/commands.py`)
- backends: in-memory + JSONL (`src/abstractruntime/storage/commands.py`), SQLite (`src/abstractruntime/storage/sqlite.py`)

These APIs are exported at the package root (see `src/abstractruntime/__init__.py`).

## Artifacts (store by reference)

Implementation: `src/abstractruntime/storage/artifacts.py`.

Key types:
- `ArtifactStore` (interface), `InMemoryArtifactStore`, `FileArtifactStore`
- helpers: `artifact_ref(...)`, `resolve_artifact(...)`, `is_artifact_ref(...)`

Artifacts are used by:
- offloading wrappers (`src/abstractruntime/storage/offloading.py`)
- evidence capture (`docs/evidence.md`, `src/abstractruntime/evidence/recorder.py`)
- AbstractCore media integration: input artifact refs can be materialized for LLM calls, and generated image/voice/music/audio outputs are stored as artifact refs

## Snapshots / bookmarks

Implementation: `src/abstractruntime/storage/snapshots.py`.

- `SnapshotStore` interface + `InMemorySnapshotStore`, `JsonSnapshotStore`
- `Snapshot` model (a named bookmark of run state)

Docs: `snapshots.md`.

## Effect policies (retries + idempotency)

Implementation: `src/abstractruntime/core/policy.py`.

- `EffectPolicy` protocol and implementations: `DefaultEffectPolicy`, `RetryPolicy`, `NoRetryPolicy`
- `compute_idempotency_key(...)` helper

Docs: `architecture.md` (reliability section).

## WorkflowBundles (`.flow`) and VisualFlow distribution

Implementation:
- bundles: `src/abstractruntime/workflow_bundle/*`
- compiler: `src/abstractruntime/visualflow_compiler/*`

VisualFlow compiler helpers are available from `abstractruntime.visualflow_compiler`:
- `load_visualflow_json(...)` normalizes VisualFlow JSON into the stdlib model.
- `visual_to_flow(...)` lowers VisualFlow into the internal Flow IR.
- `compile_visualflow(...)` and `compile_visualflow_tree(...)` compile VisualFlow JSON into executable `WorkflowSpec` objects.

Public bundle APIs are exported from `src/abstractruntime/workflow_bundle/__init__.py` and re-exported in `src/abstractruntime/__init__.py`:
- open: `open_workflow_bundle(...)`
- registry: `WorkflowBundleRegistry`
- pack/unpack: `pack_workflow_bundle(...)`, `unpack_workflow_bundle(...)`

Docs: `workflow-bundles.md`.

## Run history bundle export (portable replay artifact)

Implementation: `src/abstractruntime/history_bundle.py`.

- `export_run_history_bundle(...)`
- `persist_workflow_snapshot(...)`

This produces a portable record of a run’s state + ledger + artifacts suitable for debugging/review.

## Optional integrations

### AbstractCore (LLM + tools)

Requires: `pip install "abstractruntime[abstractcore]"` (AbstractCore 2.13.24 or newer).

Implementation: `src/abstractruntime/integrations/abstractcore/*`.

Entry points:
- `create_local_runtime(...)`, `create_remote_runtime(...)`, `create_hybrid_runtime(...)` (`src/abstractruntime/integrations/abstractcore/factory.py`)
- public discovery facade: `AbstractCoreDiscoveryFacade`, `get_abstractcore_discovery_facade(...)` (`src/abstractruntime/integrations/abstractcore/discovery_facade.py`)
- public host facade: `AbstractCoreHostFacade`, `get_abstractcore_host_facade(...)` (`src/abstractruntime/integrations/abstractcore/host_facade.py`)
- public durable run facade: `AbstractCoreRunFacade`, `get_abstractcore_run_facade(...)` (`src/abstractruntime/integrations/abstractcore/run_facade.py`)
- effect handler wiring: `build_effect_handlers(...)` (`src/abstractruntime/integrations/abstractcore/effect_handlers.py`)
- tool executors: `MappingToolExecutor`, `AbstractCoreToolExecutor`, `PassthroughToolExecutor`, `ApprovalToolExecutor`, `ToolApprovalPolicy` (`src/abstractruntime/integrations/abstractcore/tool_executor.py`)
- discovery-facade delegation is implemented by the configured AbstractCore LLM clients in `src/abstractruntime/integrations/abstractcore/llm_client.py` (`list_providers`, `list_provider_models`, `get_voice_catalog`, `list_tts_models`, `list_stt_models`, `list_music_providers`, `list_music_models`, `list_vision_provider_models`, `list_cached_vision_models`)
- host-facade delegation is implemented by the configured AbstractCore LLM clients in `src/abstractruntime/integrations/abstractcore/llm_client.py` (`get_prompt_cache_capabilities`, `get_prompt_cache_stats`, `prompt_cache_set`, `prompt_cache_update`, `prompt_cache_fork`, `prompt_cache_clear`, `prompt_cache_prepare_modules`, `upsert_text_bloc`, `get_bloc_record`, `list_blocs`, `get_bloc_kv_manifest`, `ensure_bloc_kv_artifact`, `load_bloc_kv_artifact`, `list_bloc_kv_artifacts`, `delete_bloc_kv_artifact`, `prune_bloc_kv_artifacts`, `delete_bloc`, `list_model_residency`, `load_model_residency`, `unload_model_residency`)
- run-facade helpers create durable child runs for existing runs (`execute_llm_call`, `generate_image`, `generate_voice`, `generate_music`, `transcribe_audio`)

`LLM_CALL` payloads are JSON-safe effect payloads. Common fields:
- `prompt`, `messages`, `system_prompt`, and convenience `text`
- `media`: a media path, artifact ref (`{"$artifact": "..."}` or `{"artifact_id": "..."}`), media dict, or list of those
- `output`: AbstractCore output selector; top-level `outputs` is accepted as a runtime alias
- `params`: provider/model routing, generation controls, prompt-cache keys or `prompt_cache_binding`, structured-output schema options, and tracing metadata

Multimodal support:
- install `abstractruntime[multimodal]` for common AbstractCore media, vision, voice, audio, and music dependencies
- local clients call AbstractCore's unified `generate(..., media=..., output=...)`
- remote and hybrid clients support AbstractCore Server chat media content arrays plus image generation, speech, music generation, and transcription endpoints; pass an output-specific `model` for remote media provider routing, otherwise the server endpoint can use its configured capability default
- remote transcription requires one audio media item that resolves to a local file path or artifact-backed temporary file
- generated image/voice/music/audio bytes require a runtime `ArtifactStore`; the result contains `artifact_id` / `artifact_ref` instead of inline bytes
- media-only normalized results expose `runtime_provider` / `runtime_model` separately from `media_provider` / `media_model`
- optional local media residency failures complete with `status_hint="warning"` and `degraded=true`, while unsupported local media warmup also reports `execution_mode="local_one_shot_subprocess"` and `requires_long_lived_server=true`
- Gateway/hosts remain responsible for explicit Core server URLs, Core server auth headers, provider/model defaults, selected Core/capability install profiles, and translation of Gateway-owned env/config into explicit Runtime inputs; Runtime persists only JSON-safe routing metadata and artifact refs

Prompt cache / cached sessions:
- LLM clients expose cache control methods listed above for host-side preparation and inspection
- `LLM_CALL.params.prompt_cache_key` selects a cache key for a call; runtime can also derive a session-scoped key from `run.vars["_runtime"]["prompt_cache"]` or the Runtime-owned `ABSTRACTRUNTIME_PROMPT_CACHE` process default
- `LLM_CALL.params.prompt_cache_binding` is the durable exact-reuse input for bloc-backed prompt caching; if a binding includes `key`, Runtime adopts it as the effective prompt-cache key and refuses mismatches before provider execution
- `get_abstractcore_host_facade(...)` also exposes durable bloc helpers (`upsert_text_bloc`, `get_bloc_record`, `list_blocs`, `get_bloc_kv_manifest`, `ensure_bloc_kv_artifact`, `load_bloc_kv_artifact`, `list_bloc_kv_artifacts`, `delete_bloc_kv_artifact`, `prune_bloc_kv_artifacts`, `delete_bloc`)
- local Runtime owns the bloc root policy: `~/.abstractruntime/blocs` by default, `<base_dir>/blocs` for `create_local_file_runtime(...)`, and explicit `bloc_root_dir=...` overrides when needed
- provider cache/session handles are not durable runtime state and should not be stored in `RunState.vars`

Attachment registration limits:
- `TOOL_CALLS.payload.max_attachment_bytes`, `run.vars["_runtime"]["max_attachment_bytes"]`, or `ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES` bound the bytes Runtime stores when local `read_file` outputs are captured as session attachments

Docs: `integrations/abstractcore.md`.

### AbstractMemory bridge (KG effects)

Implementation: `src/abstractruntime/integrations/abstractmemory/effect_handlers.py`.

This provides handlers for `MEMORY_KG_*` effects (opt-in wiring layer).

## Utilities (host UX)

- Rendering helpers: `abstractruntime.rendering.stringify_json(...)` and `abstractruntime.rendering.render_agent_trace_markdown(...)` (`src/abstractruntime/rendering/*`)
- Active-context helpers (what is sent to the LLM): `ActiveContextPolicy`, `TimeRange` (`src/abstractruntime/memory/active_context.py`, exports in `src/abstractruntime/memory/__init__.py`)

## See also

- `../README.md` — install + quick start
- `getting-started.md` — first durable workflow
- `architecture.md` — component map + durability invariants
- `faq.md` — common questions and gotchas
- `integrations/abstractcore.md` — `LLM_CALL` / `TOOL_CALLS` wiring
