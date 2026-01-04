from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .default_tools import get_default_toolsets
from .tool_executor import MappingToolExecutor


def _ansi(enabled: bool, code: str) -> str:
    if not enabled:
        return ""
    return f"\033[{code}m"


def _truncate(text: str, *, limit: int) -> str:
    s = "" if text is None else str(text)
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + "…"


def _log_worker_event(tag: str, message: str) -> None:
    """Write a human-friendly worker log line to stderr (never stdout)."""
    enabled = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    ts = time.strftime("%H:%M:%S")
    tag_norm = str(tag or "").strip() or "WORKER"

    if tag_norm == "RECEIVING_COMMANDS":
        color = _ansi(enabled, "38;5;39")  # blue
    elif tag_norm == "RETURNING_RESULTS":
        color = _ansi(enabled, "38;5;208")  # orange
    else:
        color = _ansi(enabled, "2")  # dim
    reset = _ansi(enabled, "0")

    try:
        sys.stderr.write(f"{color}[{tag_norm}]{reset} {ts} {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _preview_json(value: Any, *, limit: int) -> str:
    try:
        return _truncate(json.dumps(value, ensure_ascii=False, sort_keys=True), limit=limit)
    except Exception:
        return _truncate(str(value), limit=limit)


def _summarize_jsonrpc_request(req: Any, *, limit: int) -> str:
    if not isinstance(req, dict):
        return f"rpc invalid type={type(req).__name__} preview={_preview_json(req, limit=limit)}"

    req_id = req.get("id")
    method = str(req.get("method") or "").strip() or "<missing>"

    summary = f"rpc method={method} id={_truncate(str(req_id), limit=64)}"
    params = req.get("params")
    params_obj = params if isinstance(params, dict) else {}
    if method == "tools/call":
        name = str(params_obj.get("name") or "").strip() or "<missing>"
        arguments = params_obj.get("arguments")
        args = dict(arguments) if isinstance(arguments, dict) else {}
        summary += f" name={name} args={_preview_json(args, limit=limit)}"
        return summary

    if method == "tools/list":
        return summary

    if method == "initialize":
        proto = str(params_obj.get("protocolVersion") or "").strip()
        if proto:
            summary += f" protocolVersion={_truncate(proto, limit=64)}"
        return summary

    return summary


def _summarize_jsonrpc_response(resp: Any, *, limit: int) -> str:
    if resp is None:
        return "rpc notification(no response)"
    if not isinstance(resp, dict):
        return f"rpc invalid_response type={type(resp).__name__} preview={_preview_json(resp, limit=limit)}"

    if isinstance(resp.get("error"), dict):
        err = resp["error"]
        code = err.get("code")
        msg = str(err.get("message") or "").strip()
        return f"rpc_error code={code} message={_truncate(msg, limit=limit)}"

    result = resp.get("result")
    if isinstance(result, dict) and isinstance(result.get("tools"), list):
        tools = result.get("tools") or []
        names = [
            str(t.get("name") or "").strip()
            for t in tools
            if isinstance(t, dict) and isinstance(t.get("name"), str) and str(t.get("name") or "").strip()
        ]
        preview = ", ".join(names[:10])
        suffix = f" names={_truncate(preview, limit=limit)}" if preview else ""
        return f"rpc_result tools_count={len(tools)}{suffix}"

    if isinstance(result, dict) and isinstance(result.get("protocolVersion"), str):
        proto = str(result.get("protocolVersion") or "").strip()
        if proto:
            return f"rpc_result protocolVersion={_truncate(proto, limit=limit)}"

    if isinstance(result, dict) and isinstance(result.get("content"), list):
        is_error = result.get("isError")
        content = result.get("content") or []
        first = content[0] if isinstance(content, list) and content else None
        if isinstance(first, dict) and first.get("type") == "text":
            text = str(first.get("text") or "")
            return f"rpc_result isError={bool(is_error)} preview={_truncate(text, limit=limit)}"
        return f"rpc_result isError={bool(is_error)}"

    return f"rpc_result preview={_preview_json(result, limit=limit)}"


def _summarize_http_request(environ: Dict[str, Any], *, limit: int) -> str:
    method = str(environ.get("REQUEST_METHOD") or "").strip() or "?"
    path = str(environ.get("PATH_INFO") or "/") or "/"
    query = str(environ.get("QUERY_STRING") or "").strip()
    if query:
        path = f"{path}?{query}"

    remote = str(environ.get("REMOTE_ADDR") or "").strip()
    port = str(environ.get("REMOTE_PORT") or "").strip()
    if remote and port:
        remote = f"{remote}:{port}"

    origin = str(environ.get("HTTP_ORIGIN") or "").strip()
    user_agent = str(environ.get("HTTP_USER_AGENT") or "").strip()
    content_length = str(environ.get("CONTENT_LENGTH") or "").strip()

    auth_present = bool(
        str(environ.get("HTTP_AUTHORIZATION") or "").strip() or str(environ.get("HTTP_X_ABSTRACT_WORKER_TOKEN") or "").strip()
    )

    parts = [f"http {method} {path}"]
    if remote:
        parts.append(f"from={_truncate(remote, limit=128)}")
    if origin:
        parts.append(f"origin={_truncate(origin, limit=128)}")
    parts.append(f"auth={'present' if auth_present else 'missing'}")
    if content_length:
        parts.append(f"content_length={_truncate(content_length, limit=32)}")
    if user_agent:
        parts.append(f"ua={_truncate(user_agent, limit=128)}")

    return _truncate(" ".join(parts), limit=limit)


def _runtime_version() -> Optional[str]:
    for name in ("AbstractRuntime", "abstractruntime"):
        try:
            return metadata.version(name)
        except Exception:
            continue
    return None


def _jsonrpc_error(req_id: Any, *, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": int(code), "message": str(message)}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _jsonrpc_result(req_id: Any, *, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _tool_callable_name(func: Callable[..., Any]) -> str:
    tool_def = getattr(func, "_tool_definition", None)
    if tool_def is not None:
        name = getattr(tool_def, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = getattr(func, "__name__", "")
    return str(name or "").strip()


def _tool_callable_definition(func: Callable[..., Any]) -> Any:
    tool_def = getattr(func, "_tool_definition", None)
    if tool_def is not None:
        return tool_def

    from abstractcore.tools.core import ToolDefinition

    return ToolDefinition.from_function(func)


def _tool_def_to_mcp_entry(tool_def: Any) -> Dict[str, Any]:
    name = str(getattr(tool_def, "name", "") or "").strip()
    if not name:
        raise ValueError("ToolDefinition missing name")

    description = str(getattr(tool_def, "description", "") or "").strip()
    if not description:
        description = f"Tool '{name}'"

    params = getattr(tool_def, "parameters", None)
    if not isinstance(params, dict):
        params = {}

    props: Dict[str, Any] = {}
    required: List[str] = []
    for k, schema in params.items():
        if not isinstance(k, str) or not k.strip():
            continue
        meta = dict(schema) if isinstance(schema, dict) else {}
        props[k] = meta
        if isinstance(schema, dict) and "default" not in schema:
            required.append(k)

    input_schema: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        input_schema["required"] = required

    return {"name": name, "description": description, "inputSchema": input_schema}


def _format_tool_result_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


@dataclass(frozen=True)
class McpWorkerState:
    tools: List[Dict[str, Any]]
    executor: MappingToolExecutor
    http_require_auth: bool = False
    http_auth_token: Optional[str] = None
    http_allowed_origins: Tuple[str, ...] = ()


def build_worker_state(*, toolsets: Sequence[str]) -> McpWorkerState:
    desired = [str(t).strip() for t in (toolsets or []) if str(t).strip()]
    available = get_default_toolsets()

    selected: List[Callable[..., Any]] = []
    for tid in desired:
        spec = available.get(tid)
        if not isinstance(spec, dict):
            continue
        for tool in spec.get("tools", []):
            if callable(tool):
                selected.append(tool)

    tool_map: Dict[str, Callable[..., Any]] = {}
    for tool in selected:
        name = _tool_callable_name(tool)
        if not name:
            continue
        tool_map[name] = tool

    if not tool_map:
        raise ValueError(
            "No tools selected.\n\n"
            f"Requested toolsets: {desired or ['(none)']}\n"
            f"Available toolsets: {sorted(available.keys())}"
        )

    tools: List[Dict[str, Any]] = []
    for tool in tool_map.values():
        try:
            tools.append(_tool_def_to_mcp_entry(_tool_callable_definition(tool)))
        except Exception:
            continue

    tools.sort(key=lambda t: str(t.get("name") or ""))
    return McpWorkerState(tools=tools, executor=MappingToolExecutor(tool_map))


def handle_mcp_request(*, req: Dict[str, Any], state: McpWorkerState) -> Optional[Dict[str, Any]]:
    if not isinstance(req, dict):
        return _jsonrpc_error(None, code=-32600, message="Invalid Request: must be an object")

    if req.get("jsonrpc") != "2.0":
        return _jsonrpc_error(req.get("id"), code=-32600, message="Invalid Request: jsonrpc must be '2.0'")

    method = req.get("method")
    if not isinstance(method, str) or not method.strip():
        return _jsonrpc_error(req.get("id"), code=-32600, message="Invalid Request: missing method")

    req_id = req.get("id")
    if req_id is None:
        # Notification: no response.
        return None

    params = req.get("params")
    params_obj = params if isinstance(params, dict) else {}

    if method == "initialize":
        client_req_version = params_obj.get("protocolVersion")
        proto = str(client_req_version or "2025-11-25")
        server_info: Dict[str, Any] = {"name": "abstractruntime-mcp-worker"}
        ver = _runtime_version()
        if ver:
            server_info["version"] = ver
        return _jsonrpc_result(
            req_id,
            result={
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": server_info,
            },
        )

    if method == "tools/list":
        return _jsonrpc_result(req_id, result={"tools": list(state.tools)})

    if method == "tools/call":
        name = str(params_obj.get("name") or "").strip()
        if not name:
            return _jsonrpc_error(req_id, code=-32602, message="Invalid params: missing tool name")
        arguments = params_obj.get("arguments")
        args = dict(arguments) if isinstance(arguments, dict) else {}
        executed = state.executor.execute(tool_calls=[{"name": name, "arguments": args, "call_id": "mcp"}])
        results = executed.get("results") if isinstance(executed, dict) else None
        first = results[0] if isinstance(results, list) and results else None
        if not isinstance(first, dict):
            return _jsonrpc_error(req_id, code=-32000, message="Tool execution failed (malformed result)")

        if first.get("success") is True:
            text = _format_tool_result_text(first.get("output"))
            return _jsonrpc_result(
                req_id,
                result={"content": [{"type": "text", "text": text}], "isError": False},
            )

        err = str(first.get("error") or "Tool execution failed").strip()
        return _jsonrpc_result(
            req_id,
            result={"content": [{"type": "text", "text": err}], "isError": True},
        )

    return _jsonrpc_error(req_id, code=-32601, message=f"Method not found: {method}")


def serve_stdio(*, state: McpWorkerState) -> None:
    for line in sys.stdin:
        text = (line or "").strip()
        if not text:
            continue
        started = time.perf_counter()
        try:
            req = json.loads(text)
        except Exception:
            _log_worker_event("RECEIVING_COMMANDS", f"stdio parse_error body={_truncate(text, limit=500)}")
            _log_worker_event("RETURNING_RESULTS", "stdio ok=false error=parse_error")
            continue
        _log_worker_event("RECEIVING_COMMANDS", f"stdio {_summarize_jsonrpc_request(req, limit=500)}")
        if not isinstance(req, dict):
            _log_worker_event("RETURNING_RESULTS", "stdio ok=false error=invalid_request_type")
            continue
        resp = handle_mcp_request(req=req, state=state)
        elapsed_s = time.perf_counter() - started
        ok = not (isinstance(resp, dict) and isinstance(resp.get("error"), dict))
        _log_worker_event(
            "RETURNING_RESULTS",
            f"stdio ok={str(ok).lower()} elapsed_s={elapsed_s:.2f} {_summarize_jsonrpc_response(resp, limit=500)}",
        )
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def build_wsgi_app(*, state: McpWorkerState):
    def _parse_auth(environ: Dict[str, Any]) -> Optional[str]:
        raw = str(environ.get("HTTP_AUTHORIZATION") or "").strip()
        if raw.lower().startswith("bearer "):
            token = raw[len("bearer ") :].strip()
            return token or None
        # Allow a simple custom header for non-Bearer clients.
        alt = str(environ.get("HTTP_X_ABSTRACT_WORKER_TOKEN") or "").strip()
        return alt or None

    def _check_security(environ: Dict[str, Any]) -> Optional[Tuple[str, List[Tuple[str, str]], Dict[str, Any]]]:
        origin = str(environ.get("HTTP_ORIGIN") or "").strip()
        if origin:
            allowed = set(state.http_allowed_origins or ())
            if not allowed or origin not in allowed:
                return (
                    "403 Forbidden",
                    [("Content-Type", "application/json")],
                    _jsonrpc_error(None, code=-32000, message="Forbidden: invalid Origin"),
                )

        if state.http_require_auth:
            expected = str(state.http_auth_token or "").strip()
            if not expected:
                return (
                    "500 Internal Server Error",
                    [("Content-Type", "application/json")],
                    _jsonrpc_error(None, code=-32000, message="Server misconfigured: auth enabled but no token set"),
                )
            provided = _parse_auth(environ)
            if provided != expected:
                headers = [("Content-Type", "application/json"), ("WWW-Authenticate", "Bearer")]
                return ("401 Unauthorized", headers, _jsonrpc_error(None, code=-32001, message="Unauthorized"))

        return None

    def _app(environ: Dict[str, Any], start_response):
        started = time.perf_counter()
        sec = _check_security(environ)
        if sec is not None:
            status, headers, payload = sec
            status_code = int(str(status).split(" ", 1)[0]) if status else 500
            base = _summarize_http_request(environ, limit=500)
            _log_worker_event("RECEIVING_COMMANDS", base)
            elapsed_s = time.perf_counter() - started
            _log_worker_event(
                "RETURNING_RESULTS",
                f"http status={status_code} elapsed_s={elapsed_s:.2f} {_summarize_jsonrpc_response(payload, limit=500)}",
            )
            start_response(status, headers)
            return [json.dumps(payload).encode("utf-8")]

        method = environ.get("REQUEST_METHOD")
        if method != "POST":
            base = _summarize_http_request(environ, limit=500)
            _log_worker_event("RECEIVING_COMMANDS", base)
            elapsed_s = time.perf_counter() - started
            _log_worker_event("RETURNING_RESULTS", f"http status=405 elapsed_s={elapsed_s:.2f} body=method not allowed")
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
            return [b"method not allowed"]

        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except Exception:
            length = 0
        body = environ.get("wsgi.input").read(length) if environ.get("wsgi.input") else b""
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception:
            base = _summarize_http_request(environ, limit=500)
            _log_worker_event("RECEIVING_COMMANDS", f"{base} body={_truncate(body.decode('utf-8', errors='replace'), limit=500)}")
            elapsed_s = time.perf_counter() - started
            _log_worker_event("RETURNING_RESULTS", f"http status=400 elapsed_s={elapsed_s:.2f} rpc_error parse_error")
            start_response("400 Bad Request", [("Content-Type", "application/json")])
            return [json.dumps(_jsonrpc_error(None, code=-32700, message="Parse error")).encode("utf-8")]

        base = _summarize_http_request(environ, limit=500)
        summary = _summarize_jsonrpc_request(req, limit=500)
        _log_worker_event("RECEIVING_COMMANDS", f"{base} {summary}")

        resp = handle_mcp_request(req=req if isinstance(req, dict) else {}, state=state)
        if resp is None:
            resp = _jsonrpc_error(None, code=-32600, message="Notification not supported over HTTP")

        elapsed_s = time.perf_counter() - started
        _log_worker_event(
            "RETURNING_RESULTS",
            f"http status=200 elapsed_s={elapsed_s:.2f} {_summarize_jsonrpc_response(resp, limit=500)}",
        )
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps(resp).encode("utf-8")]

    return _app


def _parse_toolsets(value: str) -> List[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]

def _parse_allowed_origins(values: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for v in values or []:
        for part in str(v or "").split(","):
            s = part.strip()
            if s:
                out.append(s)
    # stable
    return tuple(dict.fromkeys(out))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="abstractruntime-mcp-worker", add_help=True)
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP transport to serve (default: stdio).",
    )
    parser.add_argument(
        "--toolsets",
        default="",
        help="Comma-separated toolsets to expose (required). Options: files, web, system.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host (http transport only).")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port (http transport only).")
    parser.add_argument(
        "--http-require-auth",
        action="store_true",
        help="Require HTTP Authorization (Bearer token). Recommended when binding beyond localhost.",
    )
    parser.add_argument(
        "--http-auth-token",
        default="",
        help="Shared secret token for HTTP auth (prefer env var for stdio/ssh; see --http-auth-token-env).",
    )
    parser.add_argument(
        "--http-auth-token-env",
        default="ABSTRACT_WORKER_TOKEN",
        help="Environment variable to read the HTTP auth token from when --http-require-auth is set (default: ABSTRACT_WORKER_TOKEN).",
    )
    parser.add_argument(
        "--http-allow-origin",
        action="append",
        default=[],
        help="Allowed Origin header value(s). If an Origin header is present and not allowed, the server returns 403.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        base_state = build_worker_state(toolsets=_parse_toolsets(str(args.toolsets or "")))
    except Exception as e:
        sys.stderr.write(
            "Failed to start MCP worker.\n\n"
            f"Error: {e}\n\n"
            "Tips:\n"
            "- Ensure AbstractCore is installed (and, for web tools, `abstractcore[tools]`).\n"
            "- Choose toolsets explicitly (e.g. `--toolsets files` or `--toolsets files,system`).\n"
        )
        return 1

    http_require_auth = bool(args.http_require_auth)
    token = str(args.http_auth_token or "").strip()
    token_env = str(args.http_auth_token_env or "").strip() or "ABSTRACT_WORKER_TOKEN"
    if http_require_auth and not token:
        token = str(os.environ.get(token_env) or "").strip()
    if http_require_auth and not token:
        sys.stderr.write(
            "Failed to start MCP worker.\n\n"
            "Error: --http-require-auth is set but no auth token was provided.\n\n"
            "Provide one of:\n"
            "- --http-auth-token <token>\n"
            f"- environment variable {token_env}=<token>\n"
        )
        return 1

    allowed_origins = _parse_allowed_origins(list(args.http_allow_origin or []))

    state = McpWorkerState(
        tools=base_state.tools,
        executor=base_state.executor,
        http_require_auth=http_require_auth,
        http_auth_token=token or None,
        http_allowed_origins=allowed_origins,
    )

    if args.transport == "http":
        from socketserver import ThreadingMixIn
        from wsgiref.simple_server import WSGIServer, make_server

        app = build_wsgi_app(state=state)

        class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
            daemon_threads = True

        with make_server(str(args.host), int(args.port), app, server_class=_ThreadingWSGIServer) as httpd:
            httpd.serve_forever()
        return 0

    serve_stdio(state=state)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
