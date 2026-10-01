"""AbstractCore's MCP clients, reached through AbstractRuntime (the gateway's import boundary).

The gateway may not import `abstractcore` directly (tests/test_gateway_import_boundary.py). Everything
it needs to test a registered MCP server and to call that server's tools from a run lives here:

- `open_mcp_client`: a connected-on-demand client for one server (stdio or Streamable HTTP). A stdio
  server starts with exactly the environment given (`inherit_env=False`), so no gateway secret leaks
  into a third-party process.
- `mcp_tool_spec`: one server tool as the spec agents see, named `mcp::<server>::<tool>`.
- `call_mcp_tool`: one `tools/call`, mapped to the runtime's tool result convention
  `(success, output, error)` (the same mapping `McpToolExecutor` uses).
- the client error classes, for callers that turn failures into sentences.

Requires the AbstractCore MCP clients with `initialize()` and `inherit_env` (abstractcore round 3).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from abstractcore.mcp import (  # noqa: F401 - re-exported names
    MCP_TOOL_PREFIX,
    McpClient,
    McpServerInfo,
    McpStdioClient,
    mcp_tool_to_abstractcore_tool_spec,
    namespaced_tool_name,
    parse_namespaced_tool_name,
)
from abstractcore.mcp.client import McpError, McpHttpError, McpProtocolError, McpRpcError  # noqa: F401


def is_mcp_tool_name(name: Any) -> bool:
    """True for a namespaced MCP tool name (`mcp::<server>::<tool>`)."""
    return parse_namespaced_tool_name(str(name or "")) is not None


def open_mcp_client(
    *,
    transport: str,
    command: str = "",
    args: Sequence[str] = (),
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    url: str = "",
    headers: Optional[Mapping[str, str]] = None,
    timeout_s: Optional[float] = 30.0,
    client_name: str = "abstractgateway",
) -> Any:
    """A client for one MCP server. stdio starts the process now (raises FileNotFoundError and
    friends when it cannot start); http connects on the first request. Call `initialize()` first."""
    if transport == "stdio":
        if not str(command or "").strip():
            raise ValueError("a stdio MCP server needs a command")
        return McpStdioClient(
            command=[str(command)] + [str(a) for a in args],
            cwd=cwd or None,
            env=dict(env or {}),
            inherit_env=False,
            timeout_s=timeout_s,
            client_name=client_name,
        )
    if transport == "http":
        if not str(url or "").strip():
            raise ValueError("an http MCP server needs a URL")
        return McpClient(url=str(url), headers=dict(headers or {}), timeout_s=timeout_s, client_name=client_name)
    raise ValueError(f"unknown MCP transport {transport!r} (stdio or http)")


def close_mcp_client(client: Any) -> None:
    """Close the client; a stdio server still running afterwards is killed."""
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass
    proc = getattr(client, "_proc", None)
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass


def mcp_tool_spec(*, server_id: str, tool: Dict[str, Any], transport: str) -> Dict[str, Any]:
    """The agent-facing spec of one MCP tool ({name: mcp::<server>::<tool>, description, parameters,
    tags, origin}). `tool` is a tools/list entry ({name, description, inputSchema}). The origin carries
    the server id and transport only — never a URL, header or command."""
    server = McpServerInfo(server_id=server_id, url=f"{transport}://{server_id}", transport=transport)
    spec = mcp_tool_to_abstractcore_tool_spec(tool, server=server)
    origin = dict(spec.get("origin") or {})
    origin.pop("url", None)
    spec["origin"] = origin
    return spec


def call_mcp_tool(client: Any, *, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any, Optional[str]]:
    """One tools/call → (success, output, error). Raises the client's errors (the caller words them)."""
    from .tool_executor import _mcp_result_to_error, _mcp_result_to_output

    result = client.call_tool(name=tool_name, arguments=dict(arguments or {}))
    err = _mcp_result_to_error(result)
    if err is not None:
        return False, None, err
    return True, _mcp_result_to_output(result), None


__all__ = [
    "MCP_TOOL_PREFIX",
    "McpError",
    "McpHttpError",
    "McpProtocolError",
    "McpRpcError",
    "call_mcp_tool",
    "close_mcp_client",
    "is_mcp_tool_name",
    "mcp_tool_spec",
    "namespaced_tool_name",
    "open_mcp_client",
    "parse_namespaced_tool_name",
]
