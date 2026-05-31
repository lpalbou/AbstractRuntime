# FAQ

## What is AbstractRuntime (in one sentence)?

AbstractRuntime is a **durable workflow runtime**: it runs workflow graphs as a persisted state machine with explicit waits (pause → resume) and an append-only execution ledger.  
Code: `src/abstractruntime/core/runtime.py`, `src/abstractruntime/core/models.py`.

## Is AbstractRuntime an agent framework?

No. AbstractRuntime is the **execution substrate**. Agent logic (ReAct/CodeAct loops, prompt policies, etc.) is built *on top* of it.  
Docs: `proposal.md`. Code: `src/abstractruntime/core/*`.

## How does AbstractRuntime relate to AbstractCore / AbstractFramework?

AbstractRuntime is the **durable execution kernel**. In the AbstractFramework ecosystem, it is commonly paired with:
- **AbstractCore** for LLM + tool execution (`EffectType.LLM_CALL`, `EffectType.TOOL_CALLS`)  
  Code: `src/abstractruntime/integrations/abstractcore/*`. Repo: [lpalbou/abstractcore](https://github.com/lpalbou/abstractcore)

AbstractFramework umbrella: [lpalbou/AbstractFramework](https://github.com/lpalbou/AbstractFramework)

## Where is the public API documented?

- API guide: `api.md`
- Canonical export list: `src/abstractruntime/__init__.py`

## How do pause/resume work?

- A node returns a `StepPlan` with an `Effect` (e.g. `ASK_USER`, `WAIT_UNTIL`, `WAIT_EVENT`).
- The runtime persists a `WaitState` into `RunState.waiting` and returns `status=waiting`.
- You resume by calling `Runtime.resume(...)` (or `ScheduledRuntime.respond(...)`) with the matching `wait_key`.

Docs: `getting-started.md`, `architecture.md`. Code: `src/abstractruntime/core/runtime.py` (`tick`, `resume`) and `src/abstractruntime/core/models.py` (`WaitState`).

## Does time-based waiting (`WAIT_UNTIL`) progress automatically?

Only if **something drives the runtime**:
- `Runtime.tick(...)` will auto-unblock a due `WAIT_UNTIL` run *when called*.
- The built-in `Scheduler` provides a driver loop that polls due waits and ticks runs.

Docs: `getting-started.md`, `architecture.md`. Code: `src/abstractruntime/core/runtime.py` (`tick`), `src/abstractruntime/scheduler/scheduler.py`.

## How do I resume a waiting run?

- If you have the `WorkflowSpec`: call `Runtime.resume(workflow=..., run_id=..., wait_key=..., payload=...)`.
- If you use `create_scheduled_runtime()`: call `sr.respond(run_id, payload)` (it uses `state.waiting.wait_key`).

Docs: `getting-started.md`. Code: `src/abstractruntime/core/runtime.py`, `src/abstractruntime/scheduler/convenience.py`.

## Why is my `ASK_USER` answer a dict?

`Runtime.resume(..., payload=...)` always takes a **dict** payload. If the wait has a `result_key`, the runtime stores that dict into `RunState.vars` at `result_key`.  
Code: `src/abstractruntime/core/runtime.py` (`Runtime.resume`) and `src/abstractruntime/core/models.py` (`WaitState.result_key`).

Common pattern:
- resume with `{"text": "..."}` (host-side)
- read `run.vars["my_result_key"]["text"]` (node-side)

## What storage backends are included?

AbstractRuntime includes:
- in-memory: `InMemoryRunStore`, `InMemoryLedgerStore`
- filesystem: `JsonFileRunStore` (checkpoints), `JsonlLedgerStore` (append-only JSONL ledger)
- SQLite: `SqliteRunStore`, `SqliteLedgerStore`

Docs: `architecture.md`. Code: `src/abstractruntime/storage/*`.

## What must be JSON-serializable (and why)?

Everything stored in `RunState.vars` must be JSON-serializable because it is persisted as durable state.  
Code: `src/abstractruntime/core/models.py` (`RunState`) and store implementations under `src/abstractruntime/storage/`.

For large values, use:
- `ArtifactStore` references (`src/abstractruntime/storage/artifacts.py`)
- offloading wrappers (`OffloadingRunStore`, `OffloadingLedgerStore`) (`src/abstractruntime/storage/offloading.py`)

Docs: `architecture.md`.

## How do I run LLM calls and tools?

LLM and tool execution are wired via the **AbstractCore integration**:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/*`.

## Can `LLM_CALL` analyze images, audio, or files?

Yes, when the configured AbstractCore provider/model supports the media. Pass `payload.media` as a path, a media dict, an artifact ref such as `{"$artifact": "..."}`, or a list of those. The runtime keeps the effect payload JSON-safe and materializes artifact refs into temporary provider-ready files for the call.

Common remote-light media/vision/audio/music dependencies are included in the base `abstractruntime` install. Use `abstractruntime[apple]` or `abstractruntime[gpu]` only when this host should execute local inferencer stacks.
Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/effect_handlers.py`, `src/abstractruntime/integrations/abstractcore/llm_client.py`.

## How do I generate images, video, voice/audio, or music?

Use `LLM_CALL` with AbstractCore's `output` selector:

```python
{"text": "A red cube on a white table", "output": {"modality": "image", "format": "png"}}
{"text": "A logo reveal", "output": {"modality": "video", "task": "text_to_video", "provider": "mlx-gen", "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "format": "mp4"}}
{"text": "Hello from Runtime", "output": {"modality": "voice", "voice": "alloy", "format": "wav"}}
{"text": "Warm lo-fi piano with brushed drums", "output": {"modality": "music", "provider": "acemusic", "model": "ace-step", "format": "wav"}}
```

Generated bytes require a runtime `ArtifactStore`. The durable result contains `artifact_id` / `artifact_ref`, not inline binary data. Remote and hybrid runtimes support common AbstractCore Server endpoints for image generation, image edits, text-to-video, image-to-video, speech, music generation, transcription, and chat media. Local runtimes can use richer AbstractCore capability plugins for voice cloning, reference-guided generation, local text-to-music, and local video generation when those AbstractCore capabilities are installed.

## Does AbstractRuntime implement image, voice, music, or video engines?

No. AbstractRuntime provides the durable graph runner, checkpoint/ledger model, waits, and artifact boundary. AbstractCore provides the LLM/media generation and analysis capabilities. Image, video, voice, transcription, and music all flow through the same JSON-safe `output` selector plus artifact-backed result shape; Runtime does not implement provider engines itself.

## Where should cached session or prompt-cache state live?

Store stable cache selectors or cache configuration in runtime-visible JSON. There are two main tracks:

- best-effort session reuse: `payload.params.prompt_cache_key`, `run.vars["_runtime"]["prompt_cache"]`, or the Runtime-owned `ABSTRACTRUNTIME_PROMPT_CACHE`
- durable exact reuse: `payload.params.prompt_cache_binding` from a previously loaded bloc/KV artifact

If a binding includes `key`, Runtime uses it as the effective prompt-cache key and does not derive a competing session key. Do not store provider session objects, cache handles, clients, or warm-cache state in `RunState.vars`. AbstractCore clients/servers own those objects, and runtime correctness should still hold when a cache is cold.

Gateway-specific prompt-cache environment variables should be consumed by Gateway and passed to Runtime explicitly; Runtime does not read the Gateway env namespace directly.

Hosts can inspect, prepare, and now clean up caches through `abstractruntime.integrations.abstractcore.get_abstractcore_host_facade(runtime)`, which exposes the normal prompt-cache/model-residency controls plus durable bloc helpers such as `upsert_text_bloc(...)`, `ensure_bloc_kv_artifact(...)`, `load_bloc_kv_artifact(...)`, `list_bloc_kv_artifacts(...)`, `delete_bloc_kv_artifact(...)`, and `delete_bloc(...)` without depending on the private runtime attachment directly.
Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/host_facade.py`, `src/abstractruntime/integrations/abstractcore/llm_client.py`.

## Can a host still export or import local provider prompt caches?

Yes, but treat that as **host-local operator tooling**, not the main durable
workflow memory model.

Use the Runtime host facade:
- `list_prompt_cache_exports(...)`
- `prompt_cache_export(...)`
- `prompt_cache_import(...)`

Important limits:
- this surface is **local-only**; remote and hybrid runtimes return
  `prompt_cache_local_only`
- Runtime owns the export root policy:
  - `~/.abstractruntime/prompt_cache_exports` by default
  - `<base_dir>/prompt_cache_exports` for `create_local_file_runtime(...)`
- exports are partitioned per provider/model, so the same logical export name
  can coexist cleanly across different local backends

For durable replay-safe workflow reuse, prefer `prompt_cache_binding` from
durable bloc/KV artifacts instead of host-local provider cache exports.
Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/host_facade.py`, `src/abstractruntime/integrations/abstractcore/llm_client.py`.

## Does Runtime duplicate durable bloc text? How do per-model caches relate to it?

For local runtimes, Runtime owns the bloc root and stores one durable **text snapshot** per SHA256 within that root. That bloc is the source of truth. The provider/model cache is a **derived artifact** under that bloc, not a second independent memory model.

So the intended shape is:
- one text/file bloc per content hash inside one Runtime bloc root
- zero or more derived cache artifacts, one per provider/model pair

That means the same bloc text can back several model-specific caches, but those caches are intentionally separate because provider/model-native KV formats are not portable.
Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/llm_client.py`, `../abstractcore/abstractcore/core/file_blocs.py`.

## Can I delete a specific durable bloc or prune old bloc caches?

Yes.

Use the Runtime host facade:
- `list_blocs(...)`
- `list_bloc_kv_artifacts(...)`
- `delete_bloc_kv_artifact(...)`
- `prune_bloc_kv_artifacts(...)`
- `delete_bloc(...)`

The important safety flags are:
- `dry_run=True` to preview the affected artifact or bloc set
- `clear_loaded=True` to clear matching live prompt-cache keys before deletion when Runtime can see that live state
- `force=True` only when you intentionally want to bypass the live-binding safety check

The important scope distinction is:
- `delete_bloc_kv_artifact(...)`: delete one provider/model artifact, keep the durable text bloc
- `delete_bloc(...)`: delete the durable text bloc itself and, by default, all derived KV artifacts under it

## Where should a host get provider / voice / music / vision catalogs from?

From Runtime. Use `abstractruntime.integrations.abstractcore.get_abstractcore_discovery_facade(runtime)` for
provider discovery, provider models, model capability lookup, voice/TTS/STT catalogs, music provider/model catalogs,
vision provider catalogs, and cached vision model snapshots.

These are snapshot/query reads, not durable `LLM_CALL` effects, so replay should use the recorded snapshot rather than
re-querying the current machine or server and pretending the answer is unchanged.
Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/discovery_facade.py`, `src/abstractruntime/integrations/abstractcore/discovery_queries.py`, `src/abstractruntime/integrations/abstractcore/llm_client.py`.

## Should Gateway or another host import AbstractCore comms or Telegram helpers directly?

No. For the remaining host/operator paths, use Runtime's public wrappers instead:

- `get_abstractcore_host_facade(runtime).list_email_accounts(...)`
- `...list_emails(...)`
- `...read_email(...)`
- `...send_email(...)`
- `abstractruntime.integrations.abstractcore.list_email_accounts(...)`
- `...list_emails(...)`
- `...read_email(...)`
- `...send_email(...)`
- `abstractruntime.integrations.abstractcore.telegram_facade.bootstrap_telegram_auth_from_env(...)`
- `...get_global_telegram_client(...)`
- `...stop_global_telegram_client()`
- `...send_telegram_message(...)`

Important nuance: the read/bootstrap wrappers are still **host-local**. They do not proxy through a remote Core
server, and they do not write durable Runtime history on their own. They exist so hosts can depend
on Runtime as the package boundary instead of importing `abstractcore.tools.comms_tools`,
`abstractcore.tools.telegram_tdlib`, or `abstractcore.tools.telegram_tools` directly.

For outbound sends that belong to a run, use the durable run facade instead:

- `get_abstractcore_run_facade(runtime).send_email(...)`
- `get_abstractcore_run_facade(runtime).send_telegram_message(...)`
- `get_abstractcore_run_facade(runtime).resume_tool_calls(...)` when an approval-gated or passthrough tool child run needs to continue

Those create child runs, record the send request and outcome in the ledger, and replay should show
the recorded result rather than resending the external message.

## Should a host execute image / TTS / music / STT directly for an existing run?

No. If the work is run-scoped and should become part of durable run history, the host should ask Runtime to execute it. Use `abstractruntime.integrations.abstractcore.get_abstractcore_run_facade(runtime)` and create a child run with `generate_image(...)`, `edit_image(...)`, `generate_voice(...)`, `generate_music(...)`, `transcribe_audio(...)`, or the lower-level `execute_llm_call(...)`.

That keeps the ledger, artifacts, and replay surface Runtime-authored instead of synthesizing history after host-side work already happened.
Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/run_facade.py`.

## Why can local media residency return `ok:false` without failing the run?

Because local media warmup is not always a meaningful reusable state. In particular, local image generation may execute through a one-shot subprocess isolation boundary, so a prior warmup cannot be reused by the next request. Runtime therefore reports unsupported local media residency explicitly instead of pretending success.

For optional residency (`required=false`), the effect still completes durably but includes `status_hint="warning"` and `degraded=true`. Unsupported local media responses also report `requires_long_lived_server=true` and a `config_hint` that points at `ABSTRACTCORE_SERVER_BASE_URL`; image generation additionally reports `execution_mode="local_one_shot_subprocess"`.
Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/effect_handlers.py`, `src/abstractruntime/integrations/abstractcore/llm_client.py`.

## What are “local / remote / hybrid” execution modes?

They refer to where LLM and tools execute:
- **Local**: in-process LLM + local tool execution
- **Remote**: HTTP to an AbstractCore server + tools typically passthrough
- **Hybrid**: remote LLM + local tools

`create_local_runtime(...)` currently uses `MultiLocalAbstractCoreLLMClient` under the hood. That client is still
local-only: it can keep multiple in-process `(provider, model)` local clients warm and route between them per request,
but it does not switch between local and remote AbstractCore backends. If you want remote model execution, use
`create_remote_runtime(...)` or `create_hybrid_runtime(...)`.

Docs: `integrations/abstractcore.md`, `../docs/adr/0002_execution_modes_local_remote_hybrid.md`. Code: `src/abstractruntime/integrations/abstractcore/factory.py`.

## What does passthrough tool mode mean?

In passthrough mode, tool calls are **not executed** in-process:
- the `TOOL_CALLS` handler returns `WAITING` with tool call details
- an external worker/operator executes the tools
- the host resumes the run with the tool results

Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/tool_executor.py` (`PassthroughToolExecutor`).

## How do I require approval before tools run?

Use `ApprovalToolExecutor` around a trusted local executor. Safe read-only/default bridge tools can execute immediately; write, command, email/WhatsApp, and unknown tools produce a durable approval wait. Resume with `{"approved": true}` to run the pending calls or `{"approved": false, "reason": "..."}` to return structured tool errors.

Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/tool_executor.py`.

## How should provider API keys be passed to a remote AbstractCore server?

Use `Authorization: Bearer <server-key>` for AbstractCore server authentication. If a request needs a per-request upstream provider key, pass `params.provider_api_key` (or legacy `params.api_key`) in the runtime payload; Runtime converts it to the `X-AbstractCore-Provider-API-Key` header. Current AbstractCore servers reject provider keys in query strings or JSON bodies for security.

Docs: `integrations/abstractcore.md`. Code: `src/abstractruntime/integrations/abstractcore/llm_client.py`.

## Does AbstractRuntime retry effects (LLM/tools)? Is it idempotent?

Retry and idempotency are controlled via `EffectPolicy`:
- idempotency keys are used to reuse prior completed results after restarts
- retry behavior is configurable (e.g. `RetryPolicy`)

Docs: `architecture.md`. Code: `src/abstractruntime/core/policy.py`, `src/abstractruntime/core/runtime.py` (effect execution + reuse).

## Is the ledger tamper-proof?

No. The built-in provenance feature is **tamper-evident** (hash chain), not signature-backed non-forgeability.

Docs: `provenance.md`. Code: `src/abstractruntime/storage/ledger_chain.py`.

## How do I stream progress updates?

If your `LedgerStore` supports subscriptions (or is wrapped with `ObservableLedgerStore`), you can subscribe in-process:
- `Runtime.subscribe_ledger(callback, run_id=...)`

Long-running generated media uses the same ledger stream. Runtime converts provider progress callbacks into `EMIT_EVENT` ledger records named `abstract.progress` with JSON-safe payloads such as `phase`, `step`, `total_steps`, `frame`, `total_frames`, and `progress`.

Docs: `architecture.md`. Code: `src/abstractruntime/core/runtime.py` (`subscribe_ledger`), `src/abstractruntime/storage/observable.py`.

## What is “evidence capture”?

Evidence capture records durable, artifact-backed evidence for selected external-boundary tools:
- `web_search`, `fetch_url`, `execute_command`

It runs best-effort after successful `TOOL_CALLS` and requires an `ArtifactStore`.  
Docs: `evidence.md`. Code: `src/abstractruntime/evidence/recorder.py`, `src/abstractruntime/core/runtime.py` (`_maybe_record_tool_evidence`, `list_evidence`, `load_evidence`).

## What are snapshots and are they safe to restore?

Snapshots are named bookmarks of run state. Restoring a snapshot is a host-level operation (load + write back into your RunStore).  
Safety depends on whether workflow code/spec has changed since the snapshot was taken.

Docs: `snapshots.md`. Code: `src/abstractruntime/storage/snapshots.py`.

## How do WorkflowBundles (`.flow`) relate to `WorkflowSpec`?

`WorkflowSpec` is an in-memory graph of Python callables (not portable). WorkflowBundles (`.flow`) distribute **VisualFlow JSON** plus a manifest; hosts compile VisualFlow JSON into `WorkflowSpec` using the VisualFlow compiler.

Docs: `workflow-bundles.md`, `architecture.md`. Code: `src/abstractruntime/workflow_bundle/*`, `src/abstractruntime/visualflow_compiler/*`.

## How do I run the MCP worker?

Use the `abstractruntime-mcp-worker` CLI from the base Runtime install and select toolsets explicitly.

Docs: `mcp-worker.md`. Code: `src/abstractruntime/integrations/abstractcore/mcp_worker.py`.

## Where should I look for runnable examples?

- `../examples/README.md` (runnable scripts)
- `manual_testing.md` (smoke tests)
