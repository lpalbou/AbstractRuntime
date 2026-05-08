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
pip install "abstractruntime[abstractcore]"
```

This extra installs AbstractCore 2.13.11 or newer. That is the supported baseline for the current server auth split (`Authorization` for server auth, `X-AbstractCore-Provider-API-Key` for provider overrides), generated-media contracts, capability catalog, prompt-cache control-plane endpoints, current tool catalog, AbstractCore's public output-selector contract, and async/sync text-generation output-selector parity.

For AbstractCore's multimodal `generate(..., output=...)` path, use the newer baseline and optional media packages:

```bash
pip install "abstractruntime[multimodal]"
```

This installs `abstractcore[media,openai,vision,voice,audio]>=2.13.11`. Local image/voice generation still depends on the configured AbstractCore capability backends (for example AbstractVision and AbstractVoice, or OpenAI/OpenAI-compatible remote engines).

The MCP worker entrypoint uses the `mcp-worker` extra:

```bash
pip install "abstractruntime[mcp-worker]"
```

## Execution modes

The factories implement three execution modes (ADR-0002):
- **Local**: in-process AbstractCore providers + local tool execution
- **Remote**: HTTP to an AbstractCore server (`/v1/chat/completions`) + tool passthrough
- **Hybrid**: remote LLM + local tool execution

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
  "text": "optional text alias, useful for TTS",
  "messages": [{"role": "user", "content": "..."}],
  "system_prompt": "...",
  "media": ["path/or/artifact-ref"],
  "output": {"modality": "text|image|voice", "task": "optional"},
  "tools": [{"name": "...", "description": "...", "parameters": {...}}],
  "params": {
    "temperature": 0.0,
    "max_tokens": 256,
    "base_url": null
  }
}
```

Notes:
- Remote mode supports per-request dynamic routing by forwarding `params.base_url` to the AbstractCore server request body (`src/abstractruntime/integrations/abstractcore/llm_client.py`).
- Remote mode sends per-request provider key overrides from `params.api_key` / `params.provider_api_key` as `X-AbstractCore-Provider-API-Key` headers. Server/master auth should be supplied separately through the client's configured headers, usually `Authorization: Bearer <ABSTRACTCORE_SERVER_API_KEY>`.
- Local mode treats `base_url` as a provider construction concern; the local client intentionally strips `params.base_url`.
- `media` accepts one item or a list. Durable artifact refs such as `{"$artifact": "...", "filename": "speech.wav"}` are materialized to temporary files for AbstractCore and never stored as raw bytes in `RunState`.
- `output` may be top-level or inside `params`; top-level `outputs` is accepted as a runtime alias for AbstractCore's `output`.
- `output.tags`, when present, are merged into the generated artifact metadata. Runtime metadata such as `run_id` and `tags` is used by AbstractRuntime's ArtifactStore boundary and is not forwarded as provider-specific generation kwargs.
- Host-supplied run defaults such as `run.vars["_runtime"]["provider"]` and `run.vars["_runtime"]["model"]` are persisted as JSON-safe routing metadata; provider clients, auth objects, downloaded model handles, and server sessions are not durable runtime state.

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

Remote runtimes support chat media by sending OpenAI-compatible data URL content arrays to AbstractCore Server. They also support image generation (`/v1/images/generations`), TTS (`/v1/audio/speech`), and STT (`/v1/audio/transcriptions`) with the same artifact-backed result shape. Remote media endpoint calls do not inherit the chat model by default; pass an output-specific `model` only when you want a remote provider/model instead of the server's configured capability default. Remote STT requires exactly one audio media item that resolves to a local file path or artifact-backed temporary file. For image edits, input-media image generation, voice clone/register, or reference-guided TTS, use local execution so AbstractCore can use its in-process capability dispatcher.

Remote multimodal generation currently supports one `output` selector per `LLM_CALL`. Hybrid runtimes use the same remote LLM/media path as remote mode while executing tools locally. Local runtimes can use AbstractCore's in-process multimodal dispatcher for richer capability plugin behavior.

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

## Prompt-cache control plane

AbstractRuntime's AbstractCore LLM clients now expose a unified prompt-cache control-plane surface for host code:

- `get_prompt_cache_capabilities(...)`
- `get_prompt_cache_stats(...)`
- `prompt_cache_set(...)`
- `prompt_cache_update(...)`
- `prompt_cache_fork(...)`
- `prompt_cache_clear(...)`
- `prompt_cache_prepare_modules(...)`

Behavior by execution mode:

- **Local** (`MultiLocalAbstractCoreLLMClient` / `LocalAbstractCoreLLMClient`): delegates to the in-process AbstractCore provider and normalizes responses into the same JSON-safe shape used by the endpoint.
- **Remote / Hybrid** (`RemoteAbstractCoreLLMClient`): proxies `/acore/prompt_cache/*` on the configured AbstractCore server.
  - When the remote target is the multi-provider AbstractCore server proxy rather than a direct AbstractEndpoint, callers can forward upstream `base_url` through these prompt-cache methods. Per-request provider key overrides supplied as `api_key` / `provider_api_key` are converted to `X-AbstractCore-Provider-API-Key` headers, not request bodies or query strings.

Contract notes:

- Capability discovery is explicit: callers can branch on `capabilities.mode` (`none`, `keyed`, `local_control_plane`) and `supports_*` flags.
- Unsupported operations return structured payloads with `supported=false`, `operation`, `code`, and `capabilities`.
- When a provider reports `mode=local_control_plane` (for example MLX, or GGUF models whose llama.cpp chat format has an exact cached renderer), the runtime can maintain a compartmentalized `system | tools | history` cache path automatically.
- When a provider reports `mode=keyed`, the runtime still forwards stable `prompt_cache_key`s but skips module preparation/fork/update orchestration.
- This surface is intentionally host-oriented; the runtime effect handlers still only use prompt caching during LLM execution, but gateway/CLI hosts can now manage prompt caches without reaching through to provider internals.
- Automatic per-session prompt-cache keys are enabled by `run.vars["_runtime"]["prompt_cache"]`, `LLM_CALL.params.prompt_cache_key`, or the Runtime-owned `ABSTRACTRUNTIME_PROMPT_CACHE` process default. Gateway-specific prompt-cache env vars should be translated by Gateway into `_runtime.prompt_cache`.

Host-side prompt-cache example:

```python
rt = create_local_runtime(provider="mlx", model="mlx-community/Qwen3-4B-4bit")
client = getattr(rt, "_abstractcore_llm_client", None)

caps = client.get_prompt_cache_capabilities() if client is not None else {}
if caps.get("capabilities", {}).get("supports_prepare_modules"):
    client.prompt_cache_prepare_modules(
        namespace="assistant",
        modules=[
            {"module_id": "system", "system_prompt": "You are concise."},
            {"module_id": "tools", "tools": [{"name": "read_file", "parameters": {"type": "object"}}]},
        ],
    )
```

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
