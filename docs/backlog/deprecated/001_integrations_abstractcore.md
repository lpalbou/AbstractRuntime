## Todo 1 — AbstractCore integration module (layered coupling)

### Goal
Implement `abstractruntime.integrations.abstractcore` so `abstractruntime` can execute:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

…either:
- **locally** (in-process AbstractCore)
- **remotely** (HTTP to AbstractCore server)
- **hybrid** (remote LLM + local tools)

### Non-goals
- Do not move tool execution out of AbstractCore.
- Do not add “agent” logic here.
- Do not store non-JSON-safe objects in `RunState.vars`.

---

### Architectural constraints (from our discussions)

- **Durability**: everything written into `RunState.vars` must be JSON-friendly.
- **Layered coupling**:
  - `abstractruntime.core` must not import AbstractCore.
  - `abstractruntime.integrations.abstractcore` may import AbstractCore.
- **Security**:
  - Remote mode should default to **tool pass-through**, not tool execution.

---

### Target file layout (in the `abstractruntime` repo)

Create:

- `src/abstractruntime/integrations/abstractcore/__init__.py`
- `src/abstractruntime/integrations/abstractcore/logging.py`
- `src/abstractruntime/integrations/abstractcore/llm_client.py`
- `src/abstractruntime/integrations/abstractcore/tool_executor.py`
- `src/abstractruntime/integrations/abstractcore/effect_handlers.py`
- `src/abstractruntime/integrations/abstractcore/factory.py`

Keep `src/abstractruntime/integrations/__init__.py` **empty** (do not import AbstractCore by default).

---

### Effect payload schemas (stable + mode-agnostic)

#### `EffectType.LLM_CALL`
`Effect.payload` should be a JSON dict shaped like:

```json
{
  "prompt": "...",
  "messages": [{"role": "user", "content": "..."}],
  "system_prompt": "...",
  "tools": [{"name": "...", "description": "...", "parameters": {...}}],
  "params": {
    "base_url": null,
    "temperature": 0,
    "max_tokens": 256,
    "response_model": null
  }
}
```

Rules:
- `prompt` is required.
- `messages`, `system_prompt`, `tools` optional.
- `params` is an “escape hatch” for provider/server kwargs.
- `params.base_url` is optional routing for OpenAI-compatible backends:
  - In **remote mode**, this should be sent as top-level `"base_url"` to AbstractCore server `/v1/chat/completions` (since AbstractCore supports per-request base_url).
  - In **local mode**, base_url is typically a provider constructor arg; per-request override would require creating a new provider instance (avoid unless you explicitly need it).
- For v0.1, recommend `stream=false` (streaming complicates durable step semantics).

#### `EffectType.TOOL_CALLS`

```json
{
  "tool_calls": [
    {"name": "tool_name", "arguments": {"x": 1}, "call_id": "optional"}
  ]
}
```

Rules:
- A tool call is name + JSON args.
- Tool results must be normalized into JSON-safe dicts.

---

### Core interfaces (OOP)

Define small interfaces inside the integration module.

#### LLM client interface

```python
from typing import Any, Dict, List, Optional, Protocol

class AbstractCoreLLMClient(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return JSON-safe dict: {content, tool_calls, usage, model, raw?}."""
```

#### Tool executor interface

```python
from typing import Any, Dict, List, Protocol

class ToolExecutor(Protocol):
    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return JSON-safe dict with results or a passthrough directive."""
```

---

### Local LLM client (in-process AbstractCore)

`LocalAbstractCoreLLMClient` responsibilities:
- Hold a configured AbstractCore provider instance: `create_llm(provider, model, **kwargs)`.
- Call `llm.generate(...)`.
- Normalize `GenerateResponse` (or structured output) into a JSON dict.

Normalization rule of thumb:
- Always return:
  - `content: str | None`
  - `tool_calls: list | None`
  - `usage: dict | None`
  - `model: str | None`
  - `finish_reason: str | None`
- Only include `raw_response` if explicitly requested (it may not be JSON-serializable).

**Important**: do not store the provider object into `RunState`.
Base URL note:
- If you need dynamic per-request `base_url` routing (multi-tenant / dynamic endpoint selection), prefer **remote mode** (AbstractCore server supports `base_url` as a request field), or create separate configured runtimes/providers per tenant.

---

### Remote LLM client (HTTP to AbstractCore server)

`RemoteAbstractCoreLLMClient` responsibilities:
- POST to `POST {server_base_url}/v1/chat/completions`
- Pass through `model`, `messages`, `temperature`, etc. as expected by AbstractCore’s server.
- If `params.base_url` is provided, forward it as top-level `base_url` (AbstractCore server supports per-request routing for OpenAI-compatible endpoints).

Design for testability (no live network):
- Inject a `request_sender` callable or an `http_client` object into the client.

Example seam:

```python
class RequestSender(Protocol):
    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Dict[str, Any]: ...
```

In production you implement it with `httpx`.
In tests you implement it with a stub that records calls and returns deterministic JSON.

---

### Tool execution options

#### `AbstractCoreToolExecutor` (local trusted mode)
- Converts input dict tool calls into AbstractCore `ToolCall` dataclasses.
- Calls AbstractCore tool registry execution (`abstractcore.tools.registry.execute_tools`).
- Normalizes results into JSON dicts:

```json
{
  "mode": "executed",
  "results": [
    {"call_id": "...", "success": true, "output": {"...": "..."}, "error": null}
  ]
}
```

#### `PassthroughToolExecutor` (remote/untrusted mode)
- Does **not** execute.
- Returns:

```json
{
  "mode": "passthrough",
  "tool_calls": [...]
}
```

Recommended runtime behavior:
- If `mode == passthrough`, the `EffectType.TOOL_CALLS` handler should **return WAITING** with a `WaitState(reason=EVENT)` and a stable `wait_key`.
  - This cleanly pauses the run until an external host executes tools and resumes with results.

---

### Effect handlers

Create a small adapter that converts effects into `EffectOutcome`.

- `LLM_CALL` handler:
  - Extract payload
  - Call LLM client
  - Return `EffectOutcome.completed(result=normalized_llm_result)`

- `TOOL_CALLS` handler:
  - Extract tool_calls
  - Call tool executor
  - If executed → `completed(result=...)`
  - If passthrough → `waiting(wait=WaitState(...))`

---

### Runtime factories

Provide constructors that wire everything consistently.

- `create_local_runtime(...)`:
  - Local LLM client
  - Local tool executor

- `create_remote_runtime(...)`:
  - Remote LLM client
  - Passthrough tool executor (default)

- `create_hybrid_runtime(...)`:
  - Remote LLM client
  - Local tool executor

The factory should return a configured `Runtime` with:
- stores (in-memory or file stores)
- `effect_handlers={EffectType.LLM_CALL: ..., EffectType.TOOL_CALLS: ...}`

---

### Logging adapter

Implement `get_logger(name)` wrapper:
- Prefer `abstractcore.utils.structured_logging.get_logger` for consistent structured logs.
- Fallback to stdlib `logging.getLogger` if AbstractCore isn’t installed.

---

### Deliverable checklist

- [ ] Integration package files exist and are not imported by default
- [ ] LLM_CALL + TOOL_CALLS effect handlers implemented
- [ ] Local/remote/hybrid factories provided
- [ ] Tool execution supports executed + passthrough + wait
- [ ] All results stored into `RunState.vars` are JSON-safe


