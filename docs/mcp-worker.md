# MCP worker (`abstractruntime-mcp-worker`)

AbstractRuntime ships an optional MCP worker that exposes AbstractCore toolsets over MCP (JSON-RPC) via:
- stdio (default)
- HTTP (optional)

Entry point:
- CLI script: `abstractruntime-mcp-worker` (`pyproject.toml`)
- implementation: `src/abstractruntime/integrations/abstractcore/mcp_worker.py`

## Install

```bash
pip install "abstractruntime[mcp-worker]"
```

## Run (stdio)

Choose toolsets explicitly (comma-separated):

```bash
abstractruntime-mcp-worker --toolsets files,web,system
```

Toolsets come from `get_default_toolsets()` (`src/abstractruntime/integrations/abstractcore/default_tools.py`). If comms tools are enabled, you can also expose `comms` (`docs/tools-comms.md`).

## Run (HTTP)

```bash
abstractruntime-mcp-worker --transport http --toolsets files,system --host 127.0.0.1 --port 8765
```

For anything beyond localhost, enable auth:

```bash
export ABSTRACT_WORKER_TOKEN="..."
abstractruntime-mcp-worker --transport http --toolsets files,system --http-require-auth
```

Optional origin allowlist (when clients send an `Origin` header):

```bash
abstractruntime-mcp-worker --transport http --toolsets files --http-allow-origin http://localhost:3000
```

## Security notes

- Exposing `system` tools can execute commands; treat the worker as privileged.
- Prefer stdio transport over an authenticated channel (e.g., SSH) when possible.

## See also

- `integrations/abstractcore.md` — tool executors and default toolsets

