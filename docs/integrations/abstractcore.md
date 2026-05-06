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

This extra installs AbstractCore 2.13.4 or newer. That is the supported baseline for the current server auth split (`Authorization` for server auth, `X-AbstractCore-Provider-API-Key` for provider overrides), prompt-cache control-plane endpoints, and current tool catalog.

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
  "messages": [{"role": "user", "content": "..."}],
  "system_prompt": "...",
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

## Default toolsets (incl. comms)

`default_tools.get_default_toolsets()` provides a host-side convenience catalog of common tools:
- file/web/system tools
- optional comms tools behind env-var gating (`docs/tools-comms.md`)

This is useful when building a `MappingToolExecutor` quickly.

## See also

- `../architecture.md` — effect handler boundaries and durability invariants
- `../tools-comms.md` — enabling email/WhatsApp/Telegram tools
- `../adr/0002_execution_modes_local_remote_hybrid.md` — rationale for local/remote/hybrid
