# AbstractCore integration

This integration wires AbstractRuntime effects to AbstractCore so workflows can execute:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

Implementation pointers (this repo):
- factories: `src/abstractruntime/integrations/abstractcore/factory.py`
- effect handlers: `src/abstractruntime/integrations/abstractcore/effect_handlers.py`
- tool executors: `src/abstractruntime/integrations/abstractcore/tool_executor.py`
- default toolsets (incl. comms gating): `src/abstractruntime/integrations/abstractcore/default_tools.py`

## Install

```bash
pip install abstractruntime
```

The base install includes AbstractCore 2.14.0 or newer. That is the supported baseline for the current server auth split (`Authorization` for server auth, `X-AbstractCore-Provider-API-Key` for provider overrides), generated-media contracts, image upscaling, capability catalog, prompt-cache control-plane endpoints (including session attribution via `/acore/prompt_cache/key_meta`), host memory snapshots, the host-wide loaded-model sweep used by residency listings, durable bloc prompt-cache helpers, bindings and lifecycle operations, task-aware model residency for text/image/video/TTS/STT, current tool catalog, AbstractCore's public output-selector contract, async/sync text-generation output-selector parity, video generation endpoints, the public local vision-cache catalog helper used by Runtime discovery, vision adapter discovery plus batch/LoRA media controls, and the released shared workspace/file-filter utility surface used by Runtime packaging and integration checks.

The base install also includes the remote-light media/capability plugins needed
for AbstractCore's multimodal `generate(..., output=...)` path. Local
image/video/voice/music generation still depends on configured AbstractCore
capability backends and hardware profiles:

```bash
pip install "abstractruntime[apple]"
pip install "abstractruntime[gpu]"
```

With `abstractmusic>=0.1.12`, the base music integration includes the lightweight remote ACE Music backend without local model-runtime extras. The MCP worker entrypoint is included in the base Runtime install.

## Execution modes

The factories implement three execution modes (ADR-0002):
- **Local**: in-process AbstractCore providers + local tool execution
- **Remote**: HTTP to an AbstractCore server (`/v1/chat/completions`) + tool passthrough
- **Hybrid**: remote LLM + local tool execution

Local mode currently uses `MultiLocalAbstractCoreLLMClient` as the built-in LLM router. Despite the name, it is not a
local+remote combo client: it routes among multiple in-process local `(provider, model)` clients and keeps them warm in
the current process. Remote model execution is a separate topology exposed through `create_remote_runtime(...)` and
`create_hybrid_runtime(...)`.

Factory functions (exported from `abstractruntime.integrations.abstractcore`):
- `create_local_runtime(...)`
- `create_remote_runtime(...)`
- `create_hybrid_runtime(...)`

Runtime stays explicit at the boundary: Gateway/hosts construct these clients with the Core server URL, Core server auth headers, provider/model defaults, retry policy, tool executor, and artifact store they intend to use. Runtime does not read `ABSTRACTGATEWAY_*` environment variables directly and does not reinterpret Gateway bearer tokens as Core server tokens or provider keys. Gateway-owned config should be consumed by Gateway, then passed to Runtime through explicit run state, effect payloads, constructor arguments, or Runtime-owned environment variables.

## Minimal LLM workflow

```python
from abstractruntime import Effect, EffectType, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore import create_local_runtime


def ask_model(run, ctx):
    return StepPlan(
        node_id="ask_model",
        effect=Effect(
            type=EffectType.LLM_CALL,
            payload={
                "prompt": "Answer in one sentence: what is durable workflow state?",
                "params": {"temperature": 0.0, "max_tokens": 128},
            },
            result_key="llm",
        ),
        next_node="done",
    )


def done(run, ctx):
    llm = run.vars.get("llm") or {}
    return StepPlan(node_id="done", complete_output={"answer": llm.get("content")})


workflow = WorkflowSpec(
    workflow_id="abstractcore_llm_demo",
    entry_node="ask_model",
    nodes={"ask_model": ask_model, "done": done},
)

rt = create_local_runtime(provider="ollama", model="qwen3:4b")
run_id = rt.start(workflow=workflow)
state = rt.tick(workflow=workflow, run_id=run_id)
print(state.output)
```

## `LLM_CALL` payload (recommended shape)

`Effect(type=EffectType.LLM_CALL, payload=...)`

```json
{
  "prompt": "...",
  "request": {
    "text": "...",
    "messages": [{"role": "user", "content": "..."}],
    "media": ["path/or/artifact-ref"]
  },
  "text": "optional text alias, useful for TTS",
  "messages": [{"role": "user", "content": "..."}],
  "system_prompt": "...",
  "media": ["path/or/artifact-ref"],
  "output": {"modality": "text|image|video|voice|sound|music", "task": "optional"},
  "tools": [{"name": "...", "description": "...", "parameters": {...}}],
  "params": {
    "temperature": 0.0,
    "max_tokens": 256,
    "base_url": null
  }
}
```

Notes:
- `request` is the lower-level Core semantic request shape. Legacy top-level `prompt`, `text`,
  `messages`, and `media` fields still work and normalize to the same Core request contract.
- Remote mode supports per-request dynamic routing by forwarding `params.base_url` to the AbstractCore server request body (`src/abstractruntime/integrations/abstractcore/llm_client.py`).
- Remote mode sends per-request provider key overrides from `params.api_key` / `params.provider_api_key` as `X-AbstractCore-Provider-API-Key` headers. Server/master auth should be supplied separately through the client's configured headers, usually `Authorization: Bearer <ABSTRACTCORE_SERVER_API_KEY>`.
- Local mode treats `base_url` and provider API keys as provider-construction concerns. `MultiLocalAbstractCoreLLMClient` can construct a per-call client when a host injects `params.base_url` plus `params.api_key` or `params.provider_api_key` (for example from a Gateway provider endpoint profile), then strips those fields before calling the provider.
- `media` accepts one item or a list. Durable artifact refs such as `{"$artifact": "...", "filename": "speech.wav"}` are materialized to temporary files for AbstractCore and never stored as raw bytes in `RunState`.
- `output` may be top-level or inside `params`; top-level `outputs` is accepted as a runtime alias for AbstractCore's `output`.
- `output.tags`, when present, are merged into the generated artifact metadata. Runtime metadata such as `run_id` and `tags` is used by AbstractRuntime's ArtifactStore boundary and is not forwarded as provider-specific generation kwargs.
- Host-supplied run defaults such as `run.vars["_runtime"]["provider"]` and `run.vars["_runtime"]["model"]` are persisted as JSON-safe routing metadata; provider clients, auth objects, downloaded model handles, and server sessions are not durable runtime state.

## Execution controls and local concurrency

For text/chat calls, pass `params.thinking` as a boolean or reasoning level and
`params.speculation` as `False`, `True`, or a Core speculation object, for example
`{"mode": "native_mtp", "num_draft_tokens": 3, "require_acceleration": true}`.
Explicit `False` overrides the loaded provider's default; omission leaves that
default available. Core owns support validation and actual inference behavior.
For native MLX, configure the head and `mlx_batching=True` when constructing/loading
the provider; a per-call parameter alone does not enable an unloaded head/scheduler.

Local Runtime clients allow concurrent submission only when the loaded MLX instance
explicitly advertises safe scheduling. Ordinary instances retain serialization
through completion, including streamed iteration. Scheduler admission is not a
promise that every request batches: MTP uses fixed cohorts, and separate prefix-cache
managers remain separate scheduling groups. Token callbacks on one client can
interleave; use request results for attribution.

Remote text/chat clients forward these controls and expose Core's returned
`execution`, `speculation`, `performance`, and `prompt_cache` fields in result
`metadata`, without replacing Runtime-owned request/trace provenance. Remote chat
still returns an aggregated response. These controls require a compatible Core
backend; they do not implement HF/GGUF native MTP by themselves.

Set `_runtime.speculation` for a run-wide preference. Subworkflows, Agent loops,
delegated children and structured-output follow-up calls inherit it unless an
explicit call/node/child setting overrides it. Missing or `None` inherits; `False`
means Off and survives every boundary. Dictionaries are copied between scopes.

Core owns the configured default at `input.text.options.speculation` in capability
routes (`output.text` is the same route). Fresh Core configurations use depth 2
where native MTP is supported; existing configurations are preserved. Applications
should leave the control unset to follow that default. Scoped Core configuration
is supplied before local provider construction, so cached head preparation uses
the correct policy. Remote discovery queries the actual Core execution host and
does not substitute local model-registry assumptions.

## Runtime grounding

AbstractRuntime records per-call grounding as structured response metadata under `metadata.runtime_grounding`. The current fields include local datetime, timezone when detectable, country, source, whether prompt injection occurred, and an optional user identity when supplied by trace metadata or local environment.

For text/chat LLM calls only, the same grounding is rendered into the current user turn as a tagged runtime envelope:

```text
<runtime_metadata>{"country":"FR","local_datetime":"2026-05-13T18:00:00+02:00"}</runtime_metadata>
hello
```

This makes time/location/user context visible to the LLM without mutating the durable human message into a natural-language prefix. If a model echoes the runtime-owned envelope, AbstractRuntime removes that envelope from user-facing response text while preserving `metadata.runtime_grounding` for audit.

Direct media requests, including image generation, TTS, and transcription, do not receive prompt-injected grounding. They still receive trace headers/tags for observability and artifact ownership, but TTS `input` and image prompts remain the literal text supplied by the workflow.

## Multimodal generation

AbstractRuntime forwards AbstractCore's unified `generate(..., output=...)` selector and normalizes multimodal responses into JSON-safe, artifact-backed results.

Generate an image:

```python
Effect(
    type=EffectType.LLM_CALL,
    payload={
        "prompt": "A red ceramic mug on a white table.",
        "output": {"modality": "image", "format": "png", "width": 1024, "height": 1024},
    },
    result_key="image_result",
)
```

Generate speech:

```python
Effect(
    type=EffectType.LLM_CALL,
    payload={
        "text": "Hello from AbstractRuntime.",
        "output": {"modality": "voice", "voice": "coral", "format": "wav"},
    },
    result_key="speech_result",
)
```

Generate music:

```python
Effect(
    type=EffectType.LLM_CALL,
    payload={
        "text": "Warm lo-fi piano with brushed drums.",
        "output": {"modality": "music", "provider": "acemusic", "model": "ace-step", "format": "wav"},
    },
    result_key="music_result",
)
```

Generate video:

```python
Effect(
    type=EffectType.LLM_CALL,
    payload={
        "prompt": "Glowing data streams converge into a geometric logo.",
        "output": {
            "modality": "video",
            "task": "text_to_video",
            "provider": "mlx-gen",
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "format": "mp4",
            "num_frames": 41,
            "fps": 24,
            "steps": 10,
        },
    },
    result_key="video_result",
)
```

Image to video:

```python
Effect(
    type=EffectType.LLM_CALL,
    payload={
        "prompt": "Add a slow camera orbit.",
        "media": {"$artifact": "source_image_artifact_id", "type": "image", "role": "source"},
        "output": {
            "modality": "video",
            "task": "image_to_video",
            "provider": "mlx-gen",
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "format": "mp4",
        },
    },
    result_key="video_result",
)
```

Transcribe/analyze audio:

```python
Effect(
    type=EffectType.LLM_CALL,
    payload={
        "media": {"$artifact": "audio_artifact_id", "filename": "speech.wav"},
        "output": "text",
    },
    result_key="transcript",
)
```

Generated binary media requires a runtime `ArtifactStore` and is stored there. The persisted result contains artifact references:

```json
{
  "outputs": {
    "image": [
      {
        "modality": "image",
        "task": "image_generation",
        "artifact_id": "...",
        "artifact_ref": {"$artifact": "...", "content_type": "image/png"}
      }
    ]
  }
}
```

LLM-call results that flow through the Core request/output path now also expose a replay-safe
`metadata._runtime_resolved_action` summary. `history_bundle` collects those into top-level
`resolved_actions`, so another client can replay which capability family, task, normalized
request/output summary, and effective route actually ran.

Media-only normalized results now distinguish orchestration identity from the actual media backend:

- `runtime_provider` / `runtime_model`: the runtime-side orchestration identity, when relevant
- `media_provider` / `media_model`: the actual image/video/voice/music backend identity surfaced from the generated output

For local one-shot subprocess image generation, runtime metadata also records `execution_mode="local_one_shot_subprocess"`.

Long-running generated media may expose provider progress callbacks. Runtime offers a transient `on_progress` callback during `LLM_CALL` execution and persists each callback as an `EMIT_EVENT` ledger record named `abstract.progress`. The callback itself is never stored in the effect payload or run vars.

**It is not carried in the payload either.** `Effect.payload` is JSON to every consumer downstream (ledger, SSE stream, host handlers that `json.dumps` it), so the callback travels BESIDE the effect: `Runtime._execute_effect_with_retry` installs it on the `contextvars.ContextVar` in `abstractruntime/core/progress_channel.py` for the duration of the handler call, and `make_llm_call_handler` reads it there into its own params dict — which is what becomes provider kwargs. An explicit `params["on_progress"]` from the caller always wins; an effect with no progress channel is offered `None`, never the previous effect's callback. A handler that hands work to a bare thread must carry the callback object (it does — `params` holds it), not re-read the ContextVar.

### Text phase feedback (prefill vs generation)

The callback is injected for **every** `LLM_CALL`, not only generated-media ones. AbstractCore decides per PROVIDER whether it has an honest signal (`BaseProvider.supports_text_progress_events()`, default `False`); a provider that has none never calls back, and that run's ledger gains no progress records at all. Providers that do (today: every MLX lane) report the prefill→generation boundary, so a client can replace "Thinking…" with what the call is actually doing.

Text phase payloads carry `kind: "llm"` — the discriminator that separates them from the media shapes on the same event name — plus `phase` (`prefill` | `generate` | `complete`), `prompt_tokens`, `cached_tokens`, `fed_tokens`, `generated_tokens`, `ttft_s`, `tokens_per_second` and `elapsed_s`, on top of the runtime identity (`run_id`, `workflow_id`, `node_id`, `step_id`, `idempotency_key`, `attempt`) every progress payload gets. Keys with no measured value are absent, never zero.

One real record, from a `basic-agent` chat turn on `mlx/Qwen3.5-4B-4bit`:

```json
{"run_id": "6ef69750-…", "node_id": "reason", "status": "completed",
 "effect": {"type": "emit_event", "payload": {"name": "abstract.progress", "scope": "run",
   "payload": {"run_id": "6ef69750-…", "workflow_id": "visual_react_agent_basic-agent_0_0_4_81795ea9_node-2",
               "node_id": "reason", "step_id": "1cdb2c55-…", "attempt": 1,
               "kind": "llm", "phase": "prefill", "event_index": 0, "elapsed_s": 0.045,
               "provider": "mlx", "model": "mlx-community/Qwen3.5-4B-4bit",
               "prompt_tokens": 808, "cached_tokens": 690, "fed_tokens": 118,
               "generated_tokens": 0}}},
 "idempotency_key": "system:progress:1cdb2c55-…:c9c572d067aa"}
```

**Cost.** AbstractCore, not Runtime, bounds the volume: `prefill`, the first-token event and the terminal `complete` always fire, cadence events no closer than 0.5 s apart, for the whole call, never capped by count (ADR-0026). A 3.7 s answer produced 9 records. Runtime appends each already-completed with a uuid suffix, so no progress tick re-parses the ledger.

**Scope.** These records are `scope: "run"` and land in the ledger of the run that made the call. In a chat bundle the `llm_call` runs inside the Agent node's SUBWORKFLOW, so a client sees them only once it follows child ledgers — the same requirement that already applies to the `abstract.status` "Thinking…" event (`result.wait.details.sub_run_id` on the parent's waiting record).

Remote runtimes support chat media by sending OpenAI-compatible data URL content arrays to AbstractCore Server. They also support image generation (`/v1/images/generations`), image edits (`/v1/images/edits` or `/{provider}/v1/images/edits`), image upscaling (`/v1/images/upscale` or `/{provider}/v1/images/upscale`), text-to-video (`/v1/videos/generations`), image-to-video (`/v1/videos/edits` or `/{provider}/v1/videos/edits`), TTS (`/v1/audio/speech`), music generation (`/v1/audio/music`), and STT (`/v1/audio/transcriptions`) with the same artifact-backed result shape. The Runtime/Core request surface now forwards task-specific media controls including `count`/`n`, `seeds`, ordered `lora_adapters`, and video `flow_shift`. Remote media endpoint calls do not inherit the chat model by default; pass an output-specific `model` only when you want a remote provider/model instead of the server's configured capability default. Remote STT requires exactly one audio media item that resolves to a local file path or artifact-backed temporary file. Remote image edits, image upscaling, and image-to-video require one source image media item resolving to a local path or artifact-backed temporary file. For voice clone/register or reference-guided TTS, use local execution so AbstractCore can use its in-process capability dispatcher. Runtime does not import `abstractmusic` directly; local music support comes through the configured AbstractCore capability stack.

Remote multimodal generation currently supports one `output` selector per `LLM_CALL`. Hybrid runtimes use the same remote LLM/media path as remote mode while executing tools locally. Local runtimes can use AbstractCore's in-process multimodal dispatcher for richer capability plugin behavior.

Local media residency is intentionally explicit when unsupported. `MODEL_RESIDENCY` results for local `image_generation`, `image_upscale`, `video_generation`, `text_to_video`, `image_to_video`, `tts`, `stt`, and `music_generation` return:

- `code="model_residency_unsupported"`
- `requires_long_lived_server=true`
- `config_hint` pointing to `ABSTRACTCORE_SERVER_BASE_URL`

Image/video-generation residency responses also include `execution_mode="local_one_shot_subprocess"` because local generated media can be isolated into one-shot workers unless a long-lived Core server owns the media backend.

When the workflow marks residency as optional (`required=false`), the effect still completes durably but includes `status_hint="warning"` and `degraded=true` so hosts can render the no-op honestly.

Remote auth example:

```python
from abstractruntime.integrations.abstractcore import create_remote_runtime

rt = create_remote_runtime(
    server_base_url="http://127.0.0.1:8000",
    model="openai/gpt-4o-mini",
    headers={"Authorization": "Bearer server-master-key"},
)
```

Then pass a per-request upstream provider key through `params.provider_api_key` only when the AbstractCore server is acting as a provider proxy for that request:

```python
payload = {
    "prompt": "Summarize this in one sentence.",
    "params": {
        "provider_api_key": "sk-provider-key",
        "base_url": "http://127.0.0.1:1234/v1",
    },
}
```

## `TOOL_CALLS` payload

```json
{
  "tool_calls": [
    {
      "name": "tool_name",
      "arguments": {"x": 1},
      "call_id": "optional (provider id)",
      "runtime_call_id": "optional (stable; runtime-generated)"
    }
  ],
  "allowed_tools": ["optional allowlist (order-insensitive)"]
}
```

Notes:
- `runtime_call_id` is generated/normalized by the runtime for durability (`src/abstractruntime/core/runtime.py`).
- In remote/passthrough mode, a host/worker boundary can use `runtime_call_id` as an idempotency key.

## Tool execution modes

Tool execution is controlled by the configured `ToolExecutor` (`src/abstractruntime/integrations/abstractcore/tool_executor.py`):

- **Executed (trusted local)**: use `MappingToolExecutor` (recommended) or `AbstractCoreToolExecutor`.
- **Passthrough (untrusted/server/edge)**: use `PassthroughToolExecutor`.
  - The `TOOL_CALLS` handler returns a durable `WAITING` run state.
  - The host executes the tool calls externally and resumes the run with results (`Runtime.resume(...)` / `Scheduler.resume_event(...)`).
- **Approval-gated local execution**: wrap a trusted executor with `ApprovalToolExecutor`.
  - Safe read-only/default bridge tools can run immediately.
  - Riskier or unknown tools return a durable `approval_required` wait.
  - A thin client can resume with `{"approved": true}` to execute the approved calls in-runtime, or `{"approved": false, "reason": "..."}` to return structured tool errors.

Approval example:

```python
from abstractruntime.integrations.abstractcore import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
    create_local_runtime,
)


def write_file(*, path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"path": path, "bytes": len(content.encode("utf-8"))}


tools = ApprovalToolExecutor(
    delegate=MappingToolExecutor({"write_file": write_file}),
    policy=ToolApprovalPolicy(),
)
rt = create_local_runtime(provider="ollama", model="qwen3:4b", tool_executor=tools)
```

## Prompt-cache control plane and durable blocs

AbstractRuntime's AbstractCore integration now exposes a public host-control facade for prompt-cache, durable bloc/KV prompt-cache operations, and model-residency operations:

- `get_abstractcore_host_facade(runtime)`
- `AbstractCoreHostFacade`
- `get_prompt_cache_capabilities(...)`
- `get_prompt_cache_stats(...)`
- `prompt_cache_set(...)`
- `prompt_cache_update(...)`
- `prompt_cache_fork(...)`
- `prompt_cache_clear(...)`
- `prompt_cache_prepare_modules(...)`
- `list_prompt_cache_exports(...)`
- `prompt_cache_export(...)`
- `prompt_cache_import(...)`
- `upsert_text_bloc(...)`
- `get_bloc_record(...)`
- `list_blocs(...)`
- `get_bloc_kv_manifest(...)`
- `ensure_bloc_kv_artifact(...)`
- `load_bloc_kv_artifact(...)`
- `list_bloc_kv_artifacts(...)`
- `delete_bloc_kv_artifact(...)`
- `prune_bloc_kv_artifacts(...)`
- `delete_bloc(...)`
- `get_model_residency_capabilities(...)`
- `list_model_residency(...)`
- `load_model_residency(...)`
- `unload_model_residency(...)`
- `lock_model_residency(...)`
- `unlock_model_residency(...)`
- `get_context_estimate(...)`
- `get_memory_snapshot(...)`
- `list_session_prompt_caches(...)`
- `clear_session_prompt_caches(...)`

Behavior by execution mode:

- **Local** (`MultiLocalAbstractCoreLLMClient` / `LocalAbstractCoreLLMClient`): delegates to the in-process AbstractCore provider and normalizes responses into the same JSON-safe shape used by the endpoint.
- **Remote / Hybrid** (`RemoteAbstractCoreLLMClient`): proxies `/acore/prompt_cache/*`, `/acore/models/*`, and `/acore/memory` on the configured AbstractCore server.
  - When the remote target is the multi-provider AbstractCore server proxy rather than a direct AbstractEndpoint, callers can forward upstream `base_url` through these prompt-cache methods. Per-request provider key overrides supplied as `api_key` / `provider_api_key` are converted to `X-AbstractCore-Provider-API-Key` headers, not request bodies or query strings.
  - For durable bloc/KV methods, `base_url` takes precedence over local loaded-runtime selectors. Runtime omits `provider`, `model`, and `runtime_id` when `base_url` is supplied so Core takes the upstream endpoint branch cleanly.

Contract notes:

- Capability discovery is explicit: callers can branch on `capabilities.mode` (`none`, `keyed`, `local_control_plane`) and `supports_*` flags.
- Unsupported operations return structured payloads with `supported=false`, `operation`, `code`, and `capabilities`.
- When a provider reports `mode=local_control_plane` (for example MLX, or GGUF models whose llama.cpp chat format has an exact cached renderer), the runtime can maintain a compartmentalized `system | tools | history` cache path automatically.
- When a provider reports `mode=keyed`, the runtime still forwards stable `prompt_cache_key`s but skips module preparation/fork/update orchestration.
- This surface is intentionally host-oriented; the runtime effect handlers still only use prompt caching during LLM execution, but gateway/CLI hosts can now manage prompt caches and durable bloc/KV artifacts through the public facade instead of reaching through to provider internals.
- Automatic per-session prompt-cache keys are enabled by `run.vars["_runtime"]["prompt_cache"]`, `LLM_CALL.params.prompt_cache_key`, or the Runtime-owned `ABSTRACTRUNTIME_PROMPT_CACHE` process default. Gateway-specific prompt-cache env vars should be translated by Gateway into `_runtime.prompt_cache`.
- Durable exact reuse uses `LLM_CALL.params.prompt_cache_binding`. If a binding includes `key`, Runtime adopts it as the effective cache key, rejects mismatches before provider execution, and skips auto-derived session-key injection for that call.
- Automatic prompt-cache key derivation is text/chat-only. Non-text output selectors such as image, voice, music, and transcription may carry an explicit `prompt_cache_binding`, but Runtime does not derive a session cache key for them.
- Local Runtime owns the bloc store root policy:
  - default local root: `~/.abstractruntime/blocs`
  - default file-runtime root: `<base_dir>/blocs`
  - explicit `bloc_root_dir=...` overrides are allowed when hosts need a different root
- The three prompt-cache tracks are distinct:
  - session prompt cache: best-effort volatile reuse
  - durable bloc prompt cache: exact reuse through bloc/KV/binding
  - host-local prompt-cache export/import admin: optional operator tooling
    around live local provider cache state, separate from durable workflow
    memory
- `get_memory_snapshot(...)`, `list_session_prompt_caches(...)`, `clear_session_prompt_caches(...)`, `lock_model_residency(...)`, `unlock_model_residency(...)`, and `get_context_estimate(...)` are optional in the LLM-client contract. A configured client that does not implement one still binds to the facade; the facade answers `{"ok": false, "supported": false, "operation": ...}` for that call instead of failing at construction time.
- `lock_model_residency(...)`, `unlock_model_residency(...)`, and `get_context_estimate(...)` accept an optional payload mapping and/or keyword arguments. Keyword arguments merge over a copy of the payload and win on key conflicts; the first positional argument is only ever the payload mapping, and a non-mapping positional raises `TypeError`.

Host-side prompt-cache example:

```python
from abstractruntime.integrations.abstractcore import (
    create_local_runtime,
    get_abstractcore_host_facade,
)

rt = create_local_runtime(provider="mlx", model="mlx-community/Qwen3-4B-4bit")
facade = get_abstractcore_host_facade(rt)

caps = facade.get_prompt_cache_capabilities()
if caps.get("capabilities", {}).get("supports_prepare_modules"):
    facade.prompt_cache_prepare_modules(
        namespace="assistant",
        modules=[
            {"module_id": "system", "system_prompt": "You are concise."},
            {"module_id": "tools", "tools": [{"name": "read_file", "parameters": {"type": "object"}}]},
        ],
    )
```

Host-side durable bloc example:

```python
from abstractruntime.integrations.abstractcore import (
    create_local_file_runtime,
    get_abstractcore_host_facade,
)

rt = create_local_file_runtime(
    base_dir="./runtime-data",
    provider="mlx",
    model="mlx-community/Qwen3-4B-4bit",
)
facade = get_abstractcore_host_facade(rt)

record = facade.upsert_text_bloc(
    path="assistant/system.txt",
    content="Long-lived system prompt or memory text",
)
artifact = facade.ensure_bloc_kv_artifact(
    provider="mlx",
    model="mlx-community/Qwen3-4B-4bit",
    sha256=record["sha256"],
)
loaded = facade.load_bloc_kv_artifact(
    provider="mlx",
    model="mlx-community/Qwen3-4B-4bit",
    sha256=record["sha256"],
)

binding = loaded["artifact"]["prompt_cache_binding"]
```

Host-local prompt-cache export/import example:

```python
saved = facade.prompt_cache_export(
    name="orbit-cache",
    key="sess:orbit",
    q8=True,
)
listed = facade.list_prompt_cache_exports()
loaded_cache = facade.prompt_cache_import(
    name="orbit-cache",
    key="loaded:orbit",
    clear_existing=True,
)
```

Host-local export/import contract:

- This surface is **local-only**. Remote and hybrid runtimes return structured
  `prompt_cache_local_only` payloads instead of proxying host filesystem state
  through Core Server.
- Runtime owns the export root policy:
  - default local root: `~/.abstractruntime/prompt_cache_exports`
  - default file-runtime root: `<base_dir>/prompt_cache_exports`
  - explicit `prompt_cache_export_root_dir=...` overrides are allowed when a
    host needs a different local catalog root
- Exports stay partitioned by provider/model under that root, so the same
  logical export name can coexist safely across different local backends.
- This is a **secondary operator/admin feature**, not the primary durable app
  contract. For replay-safe exact reuse inside workflows, prefer
  `prompt_cache_binding` from durable bloc/KV artifacts instead.

Host-side durable bloc lifecycle example:

```python
records = facade.list_blocs()
artifacts = facade.list_bloc_kv_artifacts(bloc_id=record["record"]["bloc_id"])

# Preview a safe delete first.
preview = facade.delete_bloc_kv_artifact(
    bloc_id=record["record"]["bloc_id"],
    artifact_path=artifacts["artifacts"][0]["artifact_path"],
    dry_run=True,
)

# Remove one derived KV artifact but keep the durable text bloc.
facade.delete_bloc_kv_artifact(
    bloc_id=record["record"]["bloc_id"],
    artifact_path=artifacts["artifacts"][0]["artifact_path"],
    clear_loaded=True,
)

# Remove the whole bloc and all derived artifacts under it.
facade.delete_bloc(
    bloc_id=record["record"]["bloc_id"],
    clear_loaded=True,
)
```

Then use the binding in a normal runtime `LLM_CALL`:

```python
Effect(
    type=EffectType.LLM_CALL,
    payload={
        "prompt": "Use the durable cached prefix.",
        "params": {"prompt_cache_binding": binding},
    },
    result_key="llm",
)
```

### Storage semantics

- For **local** and **local-file** runtimes, `upsert_text_bloc(...)` persists one durable text snapshot under the Runtime-owned bloc root. Runtime chooses the root (`~/.abstractruntime/blocs` by default, or `<base_dir>/blocs` for `create_local_file_runtime(...)`), while AbstractCore's `FileBlocStore` defines the on-disk layout under that root.
- Within one bloc root, the durable source of truth is **content-addressed by SHA256**. Re-upserting the same text/file hash reuses or updates the same bloc record; it does not intentionally create several independent bloc copies under that same root.
- Deduplication is therefore **per bloc root**, not global across every Runtime instance. If several runtimes should share one durable bloc store, point them at the same `bloc_root_dir`. Separate roots intentionally isolate storage and can hold separate copies of the same text.
- The durable text bloc and the provider/model cache are different layers:
  - one bloc: durable extracted text plus metadata
  - zero or more derived KV artifacts: one per `(provider, model)` pair, stored under that bloc's `kv/` area
- Derived KV artifacts are **not portable** across providers or models. The same text bloc can legitimately have several provider/model-native artifacts, but each artifact remains tied to one provider/backend/model rendering path.
- `prompt_cache_binding` is a request-time proof that a specific runtime cache key still points at the exact loaded bloc artifact. It is not the durable text itself.
- For **remote** and **hybrid** runtimes using `base_url`, Runtime does not create its own local bloc copy; it proxies the bloc/KV operation to the configured AbstractCore server or upstream endpoint, and that remote side owns the store.

### Lifecycle operations

- Runtime now exposes public host methods for:
  - listing durable bloc records
  - listing provider/model KV artifacts under those blocs
  - deleting one derived KV artifact while keeping the bloc text
  - pruning matching KV artifacts by filter
  - deleting one durable bloc and, by default, its derived KV artifacts
- Safety behavior mirrors the public AbstractCore contract:
  - `dry_run=True` previews the delete/prune result without mutating storage
  - `clear_loaded=True` clears matching live prompt-cache keys before deletion when the relevant provider/model is resident in the current runtime or the remote Core server
  - `force=True` bypasses that safety check and should be treated as an explicit operator choice
- `delete_bloc_kv_artifact(...)` deletes exactly one artifact. If the selector matches several provider/model artifacts, Runtime returns a structured error rather than guessing.
- `delete_bloc(...)` removes the durable text bloc itself. By default it also removes derived KV artifacts under that bloc; pass `delete_kv=False` only if you intentionally want to leave those artifacts behind.

### Session prompt-cache attribution and lifecycle

When Runtime derives the session-scoped prompt-cache key for a text/chat `LLM_CALL`, the LLM client stamps session attribution — `session_id`, `run_id`, `workflow_id`, `node_id`, and `namespace` — into the cache entry's metadata after each generate that used the derived key. Local clients stamp through the provider's public key-meta contract; remote clients post `POST /acore/prompt_cache/key_meta` to the configured AbstractCore server, and skip the stamp when the server does not expose that route. Stamping is best-effort and never affects the LLM call result.

Caller-owned keys are deliberately never stamped: an explicit `LLM_CALL.params.prompt_cache_key`, a `prompt_cache_binding` key, or a `_runtime.prompt_cache.key` override may be shared across sessions, so session-scoped clearing never touches them.

Hosts inspect and manage session caches through the facade:

- `list_session_prompt_caches(session_id=None)` returns `{"ok": true, "caches": [...]}` with one row per live cache key: `key`, `provider`, `model`, `runtime_id`, `session_id`, `token_count`, `bytes`, `created_at_s`, `last_used_at_s`, and the raw `meta`. Fields the backend does not report are `null` (`last_used_at_s` currently is). Without a filter, all live keys are listed; keys without attribution carry `session_id: null`.
- `clear_session_prompt_caches(session_id)` enumerates the session's keys and clears them one by one. It returns `{"ok": true, "cleared": [...], "count": <successes>}` with a per-row `cleared` flag; per-key failures are reported in the rows rather than raised. A `session_id` is required.
- `get_memory_snapshot()` returns the Core-owned host memory snapshot: `ram` (total/available/used), `process` (`rss_bytes`), and `device` (backend, allocated/total/free bytes; `null` where the backend has no query). To confirm that an unload released memory, compare `device.allocated_bytes`; process RSS is not a reliable release signal because freed device buffers can return to the process heap.

Session caches outlive runs by design: completing a run does not clear the session's caches, so later runs in the same session keep their warm prefixes. Caches go away when a host clears them explicitly, when the owning model is unloaded (`unload_model_residency` clears the provider's cache stores and Runtime's client-side mirrors), or when the provider evicts them.

```python
snapshot = facade.get_memory_snapshot()
caches = facade.list_session_prompt_caches(session_id="support-session-1")
result = facade.clear_session_prompt_caches("support-session-1")
```

## Models, engines and host jobs (config facade)

AbstractCore 2.14.0 implements the model browser, the local-engine installer and
a host job registry. `abstractruntime.integrations.abstractcore.config_facade`
passes them through unchanged, so a host such as AbstractGateway serves the same
payloads as `abstractcore models|engines … --json` and the `/acore/*` routes
without importing AbstractCore itself.

| Function | Returns |
|---|---|
| `host_profile(refresh=False)` | `host_profile_v1`: OS, accelerator, memory ceiling, free disk per model store |
| `engine_inventory(probe=False)` | `engines_status_v1`: Ollama, LM Studio, MLX, llama.cpp, vLLM, Hugging Face; installed, running, install plan |
| `engine_status(engine_id, probe=False)` | one engine row |
| `engine_install_plan(engine_id)` | the fixed install command (`argv`), method, download URL and notes for this host |
| `engine_download_url(engine_id)` | the vendor download page |
| `engine_install(engine_id, dry_run=False, force=False, allow=None)` | a `host_job_v1` job of kind `engine_install` |
| `model_catalog(q=None, engine=None, fits_only=False, hub=False)` | `model_catalog_v1`: downloadable models with presence and a fit verdict |
| `list_installed_models(provider=None)` | `models_installed_v1`: installed models per engine, with sizes |
| `model_delete_blockers(provider, artifact)` | what would stop a delete (reads only) |
| `delete_model_artifact(provider, artifact, dry_run=False, force=False)` | a `host_job_v1` job of kind `delete` |
| `start_model_download_job(provider, artifact, dry_run=False)` | a `host_job_v1` job of kind `download` (single-flight per artifact) |
| `host_jobs_list(kind=None, status=None)` | `{"schema": "host_jobs_v1", "jobs": [...]}`, newest first, including jobs started by other processes on the host |
| `host_job(job_id)` / `host_job_cancel(job_id)` | one job, or `None` when the id is unknown |
| `console_fragment(kind)` | AbstractCore's embeddable web screen (`models` or `engines`): `{"html", "js", "css"}` |
| `models_engines_support()` | `{"available", "abstractcore_version", "required", "missing"}`; never raises |

Job-starting calls run in the background and return the queued job; a dry run
(and `run_inline=True`) finishes on the calling thread and returns the finished
job.

Errors a host can map to HTTP statuses:

- `AbstractCoreTooOld` (a `NotImplementedError`): the installed AbstractCore
  predates these modules. The message names the installed version and the
  upgrade command. Map it to 501.
- `RuntimeError`: AbstractCore is not installed. Map it to 503.
- `HostActionRefused`: a refusal with `status_code` and `payload()`. Examples:
  403 `not_allowed` (engine installs disabled by the host's policy), 404
  `not_found` / `unknown_engine`, 409 `busy` (an engine install is already
  running), 409 `refused` with `delete_blockers` (the model is loaded or shares
  a cache; pass `force=True` to override).

```python
from abstractruntime.integrations.abstractcore import config_facade as core

core.engine_inventory(probe=True)["engines"][0]["install"]["argv"]  # e.g. ["brew", "install", "ollama"]
job = core.engine_install("ollama", dry_run=True)                    # finished job, nothing runs
try:
    core.delete_model_artifact("ollama", "qwen3:8b", dry_run=True)
except core.HostActionRefused as refused:
    print(refused.status_code, refused.payload())
```

## Model residency listings

`list_model_residency(...)` relays Core-owned residency truth (ADR-0007) and, for text-generation listings, includes the whole host:

- Local and multi-local runtimes merge AbstractCore's host-wide loaded-model sweep into the listing, so models resident on host-local provider servers (for example Ollama or LM Studio) appear even when they were not loaded through this runtime. Sweep-only rows carry `source: "provider_server"` and no `task` label — Runtime relays what the sweep reports and does not invent task assignments. Remote runtimes receive the same host-wide view from the Core server.
- Rows for models this runtime loaded win deduplication against sweep rows and absorb the sweep's `size_bytes` / `size_vram_bytes` when they lack their own.
- Provider-reported size extras are normalized across local and remote listings: `size` is surfaced as `size_bytes` and `size_vram` as `size_vram_bytes`, with the original fields kept.
- Local text rows carry AbstractCore's registry-declared `modalities` (capability route keys such as `input.text`, `input.image`, `output.text`). When the registry has no entry for the model, the field is omitted rather than guessed. When the provider instance reports its vision lane unusable, `input.image` is removed and `modalities_note: "vision_unusable"` records the divergence.
- Local rows also carry the serving host's identity (`host_id`, `host_name`) from AbstractCore's host-info utility; rows already carrying another host's attribution — for example sweep rows — are not overwritten. Remote listings relay the Core server's rows verbatim, including the server's host identity; Runtime never re-stamps them with the client's identity.
- Rows report lock truth: rows for models this runtime manages carry `locked` and `lockable: true`, while sweep-only provider-server rows carry `lockable: false` because this runtime cannot enforce a lock on them. Lock state is runtime-owned — provider claims cannot supply or override `locked`, `lockable`, or `locked_at`. `pinned` is a truthful alias of `locked` (same value, Core parity), never the default-identity flag: `default` alone marks the client's default pair, so a configured capability default is never presented as pinned or loaded.

## Model residency locks and context estimates

The host facade and all execution modes expose model-residency lock controls and a context-fit estimate relay:

- `lock_model_residency(payload=None, **kwargs)`
- `unlock_model_residency(payload=None, **kwargs)`
- `get_context_estimate(payload=None, **kwargs)`

Each accepts an optional payload mapping and/or keyword arguments; keyword arguments win on conflicts. Locks apply to text-generation runtimes only — a request naming another task returns a structured `model_residency_unsupported` payload instead of being relayed onto a text runtime.

**Lock rule (all execution modes): lock requires provider-verified residency.** A lock is a promise the model stays in memory, so it only applies to a model the provider verifies as resident (`provider_resident: true`). A warm client or configured default alone is configuration, not memory — locking a non-resident pair refuses with `{"ok": false, "error": "model_not_resident", ...}`; load the model first (load with `lock: true`) to lock it at load time. Unlock never requires residency, so a locked-but-since-evicted pair can always be released.

Lock behavior by execution mode:

- **Local / multi-local**: the lock is enforced client-side per `(provider, model)` pair. `unload_model_residency` refuses a locked pair with a soft `{"ok": false, "error": "model_locked", ...}` payload — never an exception. Passing `force=true` unloads the pair, and the lock is released only after the provider unload succeeds; a failed forced unload leaves the pair resident and locked. For Ollama, lock and unlock also apply the server-side `keep_alive` knob best-effort (`-1` on lock, the `"5m"` default on unlock) and report the outcome under `provider_side` (`supported` / `applied` / optional `detail`); the client-side flag remains the enforcement truth for every provider, so a knob failure does not fail the lock. Because the knob rides Ollama's native load request, unlocking a pair whose model was since evicted skips the keep-alive restore (`applied: false` with a detail) — unlock never loads a model back as a side effect, and the residency-required lock rule prevents the lock-side equivalent.
- **Remote / hybrid**: `lock_model_residency` and `unlock_model_residency` relay `POST /acore/models/lock` and `POST /acore/models/unlock` on the configured AbstractCore server, which owns the lock (and enforces the same residency-required rule). When the server refuses to unload a locked runtime with HTTP 409, `unload_model_residency` converts that refusal into the same structured `{"ok": false, "error": "model_locked", "status_code": 409, ...}` payload instead of raising; a lock refused for a non-resident model (HTTP 409) is likewise converted to `{"ok": false, "error": "model_not_resident", "status_code": 409, ...}`. `force` rides the unload request body only when true. Transport or server errors from these residency operations report `supported: true`; `supported: false` is reserved for a client that does not implement the operation at all.

Request selectors:

- Address the runtime with explicit `provider` / `model` fields or with a `runtime_id`. Local clients accept `local:text_generation:<provider>:<model>` runtime ids; a runtime id that does not address a local text runtime returns a not-found payload rather than acting on a runtime the caller did not name. A request with no selector at all applies to the client's own default identity.
- Locking requires the pair to be warm in the local client or pool AND provider-verified resident (the lock rule above). Unlocking additionally reaches a locked pair whose pooled client is no longer warm, so a lock can always be released.

When a multi-local pool re-points its default provider/model, locked pairs are exempt from the pool eviction so their in-process weights stay resident behind the lock. The one exception is a locked pair that becomes the new default identity itself: its pooled client is rebuilt with the new connection settings and its lock is cleared (logged as a warning); re-lock to pin the rebuilt client.

`get_context_estimate(...)` relays AbstractCore's analytical context-fit estimator. Local and multi-local clients call the estimator in-process, defaulting `provider` / `model` to the client's identity when omitted; remote clients relay `GET /acore/models/context_estimate` with `provider`, `model`, and an optional integer-coerced `context_length`. The estimator response is relayed verbatim — including the tri-state `fits_weights` / `fits_requested_context` split, `predicted_max_context` (the context that fits beside the weights), and `budget_bytes` (real-ceiling budget; basis and reserve stated in `notes`); it is advisory only — no Runtime load path gates on it. When the local estimator utility is unavailable, the call degrades to `{"ok": false, "supported": false, "operation": "context_estimate", ...}`.

Example:

```python
facade = get_abstractcore_host_facade(rt)

locked = facade.lock_model_residency(provider="ollama", model="qwen3:4b")
refused = facade.unload_model_residency(provider="ollama", model="qwen3:4b")  # model_locked payload
unloaded = facade.unload_model_residency(provider="ollama", model="qwen3:4b", force=True)
fit = facade.get_context_estimate(provider="ollama", model="qwen3:4b", context_length=32768)
```

### `MODEL_RESIDENCY` effect operations

The `MODEL_RESIDENCY` effect supports the operations `list_loaded`, `load`, `unload`, `lock`, and `unlock`, so workflows can author residency control durably:

```json
{"operation": "lock", "provider": "ollama", "model": "qwen3:4b"}
{"operation": "unlock", "runtime_id": "local:text_generation:ollama:qwen3:4b"}
{"operation": "unload", "provider": "ollama", "model": "qwen3:4b", "force": true, "required": false}
```

`lock` / `unlock` accept the same selector fields as the client methods (`task`, `provider`, `model`, `runtime_id`, plus `base_url` / `timeout_s` / provider-key overrides for remote relays). On `unload`, `force` is forwarded only when it is authored in the effect payload. All residency operations keep soft-fail semantics: unless the payload sets `required: true`, a refusal such as `model_locked` completes the step with `status_hint: "warning"` and `degraded: true` instead of failing the run.

## Host-local comms and Telegram wrappers

Runtime also exposes the remaining Gateway-facing host/operator wrappers for
email and Telegram:

- `get_abstractcore_host_facade(runtime)` now includes:
  - `list_email_accounts(...)`
  - `list_emails(...)`
  - `read_email(...)`
  - `send_email(...)`
- `abstractruntime.integrations.abstractcore.comms_facade` also exposes:
  - `list_email_accounts(...)`
  - `list_emails(...)`
  - `read_email(...)`
  - `send_email(...)`
- `abstractruntime.integrations.abstractcore.telegram_facade` exposes:
  - `TelegramTdlibNotAvailable`
  - `bootstrap_telegram_auth_from_env(...)`
  - `get_global_telegram_client(start=False)`
  - `stop_global_telegram_client()`
  - `send_telegram_message(...)`

Contract notes:

- These are **host-local** wrappers over current public AbstractCore tool
  modules. They do not proxy through the remote AbstractCore server.
- The host facade email methods and the standalone `comms_facade` functions use
  the same Runtime-owned email wrapper layer; choose whichever is more natural
  for the host surface you are building.
- They are intentionally **nondurable**. They do not write Runtime run history
  on their own.
- Direct `send_email(...)` on the host facade and direct
  `telegram_facade.send_telegram_message(...)` are for operator-owned
  host-local flows only. If the outbound send belongs to a workflow/run, prefer
  the durable run facade methods shown below.
- Even for **remote** and **hybrid** runtimes, they still use the current host
  process env/config, local TDLib installation, and the host's own outbound
  network access.
- The Telegram global client is process-wide, not runtime-instance scoped.

Host-side operator example:

```python
from abstractruntime.integrations.abstractcore import (
    create_local_runtime,
    get_abstractcore_host_facade,
)
from abstractruntime.integrations.abstractcore.telegram_facade import (
    TelegramTdlibNotAvailable,
    bootstrap_telegram_auth_from_env,
    send_telegram_message,
)

rt = create_local_runtime(provider="ollama", model="qwen3:4b")
facade = get_abstractcore_host_facade(rt)

accounts = facade.list_email_accounts()
sent = facade.send_email(
    ["ops@example.com"],
    "Runtime status",
    body_text="All green.",
)

try:
    bootstrap = bootstrap_telegram_auth_from_env(timeout_s=30)
except TelegramTdlibNotAvailable:
    bootstrap = {"success": False, "error": "TDLib is not installed on this host."}

notify = send_telegram_message(chat_id=123456, text="Runtime check complete.")
```

## Discovery snapshots

AbstractRuntime's AbstractCore integration also exposes a public host discovery facade for snapshot/query reads:

- `get_abstractcore_discovery_facade(runtime)`
- `AbstractCoreDiscoveryFacade`
- `list_providers(...)`
- `list_provider_models(...)`
- `get_model_capabilities(...)`
- `get_voice_catalog(...)`
- `list_tts_models(...)`
- `list_stt_models(...)`
- `list_music_providers(...)`
- `list_music_models(...)`
- `list_vision_provider_models(...)`
- `list_cached_vision_models(...)`

Behavior by execution mode:

- **Local** (`MultiLocalAbstractCoreLLMClient` / `LocalAbstractCoreLLMClient`): uses public AbstractCore registries,
  capability facades, and local vision cache inspection to return JSON-safe snapshot payloads.
- **Remote / Hybrid** (`RemoteAbstractCoreLLMClient`): proxies `/providers`, `/v1/models`, `/v1/audio/*`, and
  `/v1/vision/*` on the configured AbstractCore server. Per-request provider key overrides supplied as `api_key` /
  `provider_api_key` become `X-AbstractCore-Provider-API-Key` headers.

`list_provider_models(provider, ...)` accepts the legacy `input_type` and
`output_type` filters plus Core's precise `capability_route` filter. Local mode
normalizes route filters before calling AbstractCore's provider registry; remote
mode forwards them to `/v1/models?capability_route=...`:

```python
models = facade.list_provider_models(
    "lmstudio",
    capability_route=["input.image", "output.text"],
)
embeddings = facade.list_provider_models("lmstudio", capability_route="embedding.text")
```

Contract notes:

- This surface is query-oriented. It does not create durable Runtime history on its own.
- Hosts should still ask Runtime for these reads instead of rebuilding Core catalog logic or importing Core server
  helpers directly.
- Model capability lookup is static metadata, not a live server probe. Replay should treat it as a recorded snapshot,
  not as a query to re-run.
- `list_cached_vision_models(...)` may still depend on the current local machine state. It is a Runtime-owned snapshot
  query, not durable run truth.
- Remote discovery methods accept `timeout_s=...` through facade kwargs. Local discovery remains synchronous helper
  code; async hosts should offload it to a worker thread if they do not want to block their event loop.

Host-side discovery example:

```python
from abstractruntime.integrations.abstractcore import (
    create_remote_runtime,
    get_abstractcore_discovery_facade,
)

rt = create_remote_runtime(
    server_base_url="http://127.0.0.1:8000",
    model="openai/gpt-4o-mini",
    headers={"Authorization": "Bearer server-master-key"},
)
facade = get_abstractcore_discovery_facade(rt)

providers = facade.list_providers(include_models=False)
voices = facade.get_voice_catalog(provider="openai", providers_only=True)
music = facade.list_music_providers(task="text_to_music")
vision = facade.list_vision_provider_models(task="text_to_image", providers_only=True)
upscalers = facade.list_vision_provider_models(task="image_upscale")
adapters = facade.list_vision_adapters(
    task="text_to_video",
    model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
)
```

## Durable run-scoped media and comms execution

Hosts sometimes need to trigger image/TTS/music/STT work or outbound comms sends for an existing run. That work should still execute through Runtime so the child run ledger, artifact ownership, and replay surface remain Runtime-authored.

Public durable entry points:

- `get_abstractcore_run_facade(runtime)`
- `AbstractCoreRunFacade`
- `execute_llm_call(...)`
- `execute_tool_calls(...)`
- `resume_tool_calls(...)`
- `generate_image(...)`
- `edit_image(...)`
- `upscale_image(...)`
- `generate_video(...)`
- `image_to_video(...)`
- `generate_voice(...)`
- `stream_voice(...)`
- `generate_music(...)`
- `transcribe_audio(...)`
- `send_email(...)`
- `send_telegram_message(...)`

These helpers create child runs under an existing parent run and execute the real
`LLM_CALL`, `TOOL_CALLS`, or stream finalization through Runtime rather than
doing external work in host/controller code. `stream_voice(...)` yields TTS
stream events for progressive playback and completes the child run with the
final audio artifact when the stream succeeds.

Example:

```python
from abstractruntime.integrations.abstractcore import (
    create_local_runtime,
    get_abstractcore_run_facade,
)

rt = create_local_runtime(provider="mlx", model="qwen-chat")
facade = get_abstractcore_run_facade(rt)

child = facade.generate_image(
    "existing-parent-run-id",
    prompt="A red mug on a white table.",
    output={
        "provider": "mlx-gen",
        "model": "AbstractFramework/qwen-image-2512-8bit",
        "format": "png",
        "count": 2,
        "seeds": [101, 102],
        "lora_adapters": [
            {"id": "pixel-art", "scale": 0.7},
            {"id": "cool-grade", "scale": 0.2},
        ],
    },
)

assert child.status.value == "completed"
result = child.output["result"]
print(child.run_id, result["media_model"], result["outputs"]["image"][0]["artifact_id"])
```

Image upscaling uses the same durable child-run boundary and Core-owned `image_upscale` selector:

```python
child = facade.upscale_image(
    "existing-parent-run-id",
    media={"$artifact": "source-image-artifact-id", "type": "image"},
    output={
        "provider": "mlx-gen",
        "format": "png",
        "scale": 2,
        "resolution": 1024,
    },
)

print(child.run_id, child.output["result"]["outputs"]["image"][0]["artifact_id"])
```

For video, use the same child-run boundary:

```python
child = facade.generate_video(
    "existing-parent-run-id",
    prompt="Glowing data streams converge into a geometric logo.",
    output={
        "provider": "mlx-gen",
        "model": "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
        "format": "mp4",
        "num_frames": 41,
        "count": 2,
        "seeds": [401, 402],
        "flow_shift": 3.0,
        "lora_adapters": [{"id": "documentary-motion", "scale": 0.6}],
    },
)

print(child.run_id, child.output["result"]["outputs"]["video"][0]["artifact_id"])
```

Outbound comms sends that belong to a run should use the same durable child-run surface:

```python
email_child = facade.send_email(
    "existing-parent-run-id",
    to=["ops@example.com"],
    subject="Workflow alert",
    body_text="The workflow completed.",
)

telegram_child = facade.send_telegram_message(
    "existing-parent-run-id",
    chat_id=123456,
    text="Workflow completed.",
)
```

Contract notes for durable comms sends:

- Runtime records the send request and the send outcome in the child run ledger.
- Replay should show the recorded result; it should **not** resend the external email or Telegram message.
- Local and hybrid runtimes usually execute those sends immediately when the configured tool executor can run them.
- Remote runtimes may still enter a durable tool wait if the configured tool executor is passthrough/delegated or approval-gated. That wait/resume path is still Runtime-authored truth.
- To resume a waiting durable comms/tool child run through the same public boundary, use `get_abstractcore_run_facade(runtime).resume_tool_calls(child_run_id, payload=...)`.

## Attachment registration limits

When local `read_file` tool outputs are captured as session attachments, Runtime bounds the file bytes it stores. The limit is resolved in this order:

- `TOOL_CALLS.payload.max_attachment_bytes`
- `run.vars["_runtime"]["max_attachment_bytes"]`
- `ABSTRACTRUNTIME_MAX_ATTACHMENT_BYTES`
- the default of 25 MiB

Gateway-specific attachment env vars should be translated by Gateway into one of the explicit Runtime inputs above.

## Default toolsets (incl. comms)

`default_tools.get_default_toolsets()` provides a host-side convenience catalog of common tools:
- file/web/system tools
- optional comms tools behind env-var gating (`docs/tools-comms.md`)

This is useful when building a `MappingToolExecutor` quickly.

## See also

- `../architecture.md` — effect handler boundaries and durability invariants
- `../tools-comms.md` — enabling email/WhatsApp/Telegram tools
- `../adr/0002_execution_modes_local_remote_hybrid.md` — rationale for local/remote/hybrid
