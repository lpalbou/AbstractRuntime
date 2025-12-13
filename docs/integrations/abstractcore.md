## AbstractCore integration

This integration lets AbstractRuntime execute:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

via AbstractCore either:
- **locally** (in-process)
- **remotely** (HTTP to AbstractCore server)
- **hybrid** (remote LLM + local tools)

Implementation lives in:
- `src/abstractruntime/integrations/abstractcore/`

---

### Layered coupling (why it matters)

- The kernel (`abstractruntime.core`, `abstractruntime.storage`, `abstractruntime.identity`) stays dependency-light.
- This module is the explicit opt-in to importing `abstractcore`.

This keeps the durable state machine stable even if AbstractCore internals change.

---

### LLM_CALL payload schema

`Effect(type=EffectType.LLM_CALL, payload=...)`

Recommended payload shape:

```json
{
  "prompt": "...",
  "messages": [{"role": "user", "content": "..."}],
  "system_prompt": "...",
  "tools": [{"name": "...", "description": "...", "parameters": {...}}],
  "params": {
    "temperature": 0,
    "max_tokens": 256,
    "base_url": null
  }
}
```

Notes:
- **Remote mode** supports `params.base_url` by forwarding it as top-level `base_url` to AbstractCore server `/v1/chat/completions`.
- **Local mode** treats `base_url` as a provider construction concern (do not expect per-request override unless you build separate runtimes/providers).

---

### TOOL_CALLS payload schema

```json
{
  "tool_calls": [
    {"name": "tool_name", "arguments": {"x": 1}, "call_id": "optional"}
  ]
}
```

---

### Tool execution modes

#### Executed (trusted local)
- Uses AbstractCore global tool registry.
- Returns:

```json
{
  "mode": "executed",
  "results": [
    {"call_id": "...", "success": true, "output": 123, "error": null}
  ]
}
```

#### Passthrough (untrusted / server / edge)
- Does not execute tools.
- The handler returns a **WAITING** run state.
- The `WaitState.details` includes the tool calls.

The host must:
1) execute the tool calls externally
2) call `runtime.resume(..., payload=<tool_results>)`

---

### Convenience factories

- `create_local_runtime(provider, model, ...)`
- `create_remote_runtime(server_base_url, model, ...)`
- `create_hybrid_runtime(server_base_url, model, ...)`

See `src/abstractruntime/integrations/abstractcore/factory.py`.

