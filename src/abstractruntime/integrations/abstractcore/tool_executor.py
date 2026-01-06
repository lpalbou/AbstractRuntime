"""abstractruntime.integrations.abstractcore.tool_executor

Tool execution adapters.

- `AbstractCoreToolExecutor`: executes tool calls in-process using AbstractCore's
  global tool registry.
- `PassthroughToolExecutor`: does not execute; returns tool calls to the host.

The runtime can use passthrough mode for untrusted environments (server/edge) and
pause until the host resumes with the tool results.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import inspect
import json
import re
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .logging import get_logger

logger = get_logger(__name__)


class ToolExecutor(Protocol):
    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]: ...


def _normalize_timeout_s(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    # Contract: non-positive values are treated as "unlimited".
    return None if f <= 0 else f


def _call_with_timeout(func: Callable[[], Any], *, timeout_s: Optional[float]) -> tuple[bool, Any, Optional[str]]:
    """Execute a callable with a best-effort timeout.

    Important limitation (Python semantics): we cannot forcibly stop a running function
    without process isolation. On timeout we return an error, but the underlying callable
    may still finish later (daemon thread).
    """
    timeout_s = _normalize_timeout_s(timeout_s)
    if timeout_s is None:
        try:
            return True, func(), None
        except Exception as e:
            return False, None, str(e)

    result: Dict[str, Any] = {"done": False, "ok": False, "value": None, "error": None}

    def _runner() -> None:
        try:
            result["value"] = func()
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)
            result["ok"] = False
        finally:
            result["done"] = True

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout_s)

    if not result.get("done", False):
        return False, None, f"Tool execution timed out after {timeout_s}s"
    if result.get("ok", False):
        return True, result.get("value"), None
    return False, None, str(result.get("error") or "Tool execution failed")


class MappingToolExecutor:
    """Executes tool calls using an explicit {tool_name -> callable} mapping.

    This is the recommended durable execution path: the mapping is held by the
    host/runtime process and is never persisted inside RunState.
    """

    def __init__(self, tool_map: Dict[str, Callable[..., Any]], *, timeout_s: Optional[float] = None):
        self._tool_map = dict(tool_map)
        self._timeout_s = _normalize_timeout_s(timeout_s)

    @classmethod
    def from_tools(cls, tools: Sequence[Callable[..., Any]], *, timeout_s: Optional[float] = None) -> "MappingToolExecutor":
        tool_map: Dict[str, Callable[..., Any]] = {}
        for t in tools:
            tool_def = getattr(t, "_tool_definition", None)
            if tool_def is not None:
                name = str(getattr(tool_def, "name", "") or "")
                func = getattr(tool_def, "function", None) or t
            else:
                name = str(getattr(t, "__name__", "") or "")
                func = t

            if not name:
                raise ValueError("Tool is missing a name")
            if not callable(func):
                raise ValueError(f"Tool '{name}' is not callable")
            if name in tool_map:
                raise ValueError(f"Duplicate tool name '{name}'")

            tool_map[name] = func

        return cls(tool_map, timeout_s=timeout_s)

    def set_timeout_s(self, timeout_s: Optional[float]) -> None:
        self._timeout_s = _normalize_timeout_s(timeout_s)

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []

        def _loads_dict_like(value: Any) -> Optional[Dict[str, Any]]:
            if value is None:
                return None
            if isinstance(value, dict):
                return dict(value)
            if not isinstance(value, str):
                return None
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None

        def _unwrap_wrapper_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Unwrap common wrapper shapes like {"name":..., "arguments":{...}}.

            Some models emit tool kwargs wrapped inside an "arguments" object and may
            mistakenly place real kwargs alongside wrapper fields. We unwrap and merge
            (inner args take precedence).
            """
            current: Dict[str, Any] = dict(kwargs or {})
            wrapper_keys = {"name", "arguments", "call_id", "id"}
            for _ in range(4):
                inner = current.get("arguments")
                inner_dict = _loads_dict_like(inner)
                if not isinstance(inner_dict, dict):
                    break
                extras = {k: v for k, v in current.items() if k not in wrapper_keys}
                merged = dict(inner_dict)
                for k, v in extras.items():
                    merged.setdefault(k, v)
                current = merged
            return current

        def _filter_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Best-effort filtering of unexpected kwargs for callables without **kwargs."""
            try:
                sig = inspect.signature(func)
            except Exception:
                return kwargs

            params = list(sig.parameters.values())
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return kwargs

            allowed = {
                p.name
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            return {k: v for k, v in kwargs.items() if k in allowed}

        def _normalize_key(key: str) -> str:
            # Lowercase and remove common separators so `file_path`, `filePath`,
            # `file-path`, `file path` all normalize to the same token.
            return re.sub(r"[\s_\-]+", "", str(key or "").strip().lower())

        _SYNONYM_ALIASES: Dict[str, List[str]] = {
            # Common semantic drift across many tools
            "path": ["file_path", "directory_path", "path"],
            # Common CLI/media naming drift
            "filename": ["file_path"],
            "filepath": ["file_path"],
            "dir": ["directory_path", "path"],
            "directory": ["directory_path", "path"],
            "folder": ["directory_path", "path"],
            "query": ["pattern", "query"],
            "regex": ["pattern", "regex"],
            # Range drift (used by multiple tools)
            "start": ["start_line", "start"],
            "end": ["end_line", "end"],
            "startlineoneindexed": ["start_line"],
            "endlineoneindexedinclusive": ["end_line"],
        }

        def _canonicalize_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Best-effort canonicalization of kwarg names.

            Strategy:
            - Unwrap common wrapper shapes (nested `arguments`)
            - Map keys by normalized form (case + separators)
            - Apply a small, tool-agnostic synonym table (path/query/start/end)
            - Finally, filter unexpected kwargs for callables without **kwargs
            """
            if not isinstance(kwargs, dict) or not kwargs:
                return {}

            # 1) Unwrap wrapper shapes early.
            current = _unwrap_wrapper_args(kwargs)

            try:
                sig = inspect.signature(func)
            except Exception:
                return current

            params = list(sig.parameters.values())
            allowed_names = {
                p.name
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            norm_to_param = { _normalize_key(n): n for n in allowed_names }

            out: Dict[str, Any] = dict(current)

            # 2) Normalized (morphological) key mapping.
            for k in list(out.keys()):
                if k in allowed_names:
                    continue
                nk = _normalize_key(k)
                target = norm_to_param.get(nk)
                if target and target not in out:
                    out[target] = out.pop(k)

            # 3) Synonym mapping (semantic).
            for k in list(out.keys()):
                if k in allowed_names:
                    continue
                nk = _normalize_key(k)
                candidates = _SYNONYM_ALIASES.get(nk, [])
                for cand in candidates:
                    if cand in allowed_names and cand not in out:
                        out[cand] = out.pop(k)
                        break

            # 4) Filter unexpected kwargs when callable doesn't accept **kwargs.
            return _filter_kwargs(func, out)

        def _error_from_output(value: Any) -> Optional[str]:
            """Detect tool failures reported as string outputs (instead of exceptions)."""
            # Structured tool outputs may explicitly report failure without raising.
            # Only treat as error when the tool declares failure.
            if isinstance(value, dict):
                success = value.get("success")
                ok = value.get("ok")
                if success is False or ok is False:
                    err = value.get("error") or value.get("message") or "Tool reported failure"
                    text = str(err).strip()
                    return text or "Tool reported failure"
                return None
            if not isinstance(value, str):
                return None
            text = value.strip()
            if not text:
                return None
            if text.startswith("Error:"):
                cleaned = text[len("Error:") :].strip()
                return cleaned or text
            if text.startswith(("❌", "🚫", "⏰")):
                cleaned = text.lstrip("❌🚫⏰").strip()
                if cleaned.startswith("Error:"):
                    cleaned = cleaned[len("Error:") :].strip()
                return cleaned or text
            return None

        def _append_result(*, call_id: str, name: str, output: Any) -> None:
            error = _error_from_output(output)
            if error is not None:
                # Preserve structured outputs for provenance/evidence. For string-only error outputs
                # (the historical convention), keep output empty and store the message in `error`.
                output_json = None if isinstance(output, str) else _jsonable(output)
                results.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "success": False,
                        "output": output_json,
                        "error": error,
                    }
                )
                return

            results.append(
                {
                    "call_id": call_id,
                    "name": name,
                    "success": True,
                    "output": _jsonable(output),
                    "error": None,
                }
            )

        for tc in tool_calls:
            name = str(tc.get("name", "") or "")
            raw_arguments = tc.get("arguments") or {}
            arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else (_loads_dict_like(raw_arguments) or {})
            call_id = str(tc.get("call_id") or "")

            func = self._tool_map.get(name)
            if func is None:
                results.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "success": False,
                        "output": None,
                        "error": f"Tool '{name}' not found",
                    }
                )
                continue

            arguments = _canonicalize_kwargs(func, arguments)

            def _invoke() -> Any:
                try:
                    return func(**arguments)
                except TypeError:
                    # Retry once with sanitized kwargs for common wrapper/extra-arg failures.
                    filtered = _canonicalize_kwargs(func, arguments)
                    if filtered != arguments:
                        return func(**filtered)
                    raise

            ok, output, err = _call_with_timeout(_invoke, timeout_s=self._timeout_s)
            if ok:
                _append_result(call_id=call_id, name=name, output=output)
            else:
                results.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "success": False,
                        "output": None,
                        "error": str(err or "Tool execution failed"),
                    }
                )

        return {"mode": "executed", "results": results}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())

    return str(value)


class AbstractCoreToolExecutor:
    """Executes tool calls using AbstractCore's global tool registry."""

    def __init__(self, *, timeout_s: Optional[float] = None):
        self._timeout_s = _normalize_timeout_s(timeout_s)

    def set_timeout_s(self, timeout_s: Optional[float]) -> None:
        self._timeout_s = _normalize_timeout_s(timeout_s)

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        from abstractcore.tools.core import ToolCall
        from abstractcore.tools.registry import execute_tool

        calls = [
            ToolCall(
                name=str(tc.get("name")),
                arguments=dict(tc.get("arguments") or {}),
                call_id=tc.get("call_id"),
            )
            for tc in tool_calls
        ]

        normalized = []
        for call in calls:
            ok, out, err = _call_with_timeout(lambda c=call: execute_tool(c), timeout_s=self._timeout_s)
            if ok:
                r = out
                normalized.append(
                    {
                        "call_id": getattr(r, "call_id", "") if r is not None else "",
                        "name": getattr(call, "name", ""),
                        "success": bool(getattr(r, "success", False)) if r is not None else True,
                        "output": _jsonable(getattr(r, "output", None)) if r is not None else None,
                        "error": getattr(r, "error", None) if r is not None else None,
                    }
                )
                continue

            normalized.append(
                {
                    "call_id": str(getattr(call, "call_id", "") or ""),
                    "name": getattr(call, "name", ""),
                    "success": False,
                    "output": None,
                    "error": str(err or "Tool execution failed"),
                }
            )

        return {"mode": "executed", "results": normalized}


class PassthroughToolExecutor:
    """Returns tool calls unchanged without executing them."""

    def __init__(self, *, mode: str = "passthrough"):
        self._mode = mode

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"mode": self._mode, "tool_calls": _jsonable(tool_calls)}


def _mcp_result_to_output(result: Any) -> Any:
    if not isinstance(result, dict):
        return _jsonable(result)

    content = result.get("content")
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        if texts:
            joined = "\n".join(texts).strip()
            if joined:
                try:
                    return _jsonable(json.loads(joined))
                except Exception:
                    return joined

    return _jsonable(result)


def _mcp_result_to_error(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    output = _mcp_result_to_output(result)

    # MCP-native error flag.
    if result.get("isError") is True:
        if isinstance(output, str) and output.strip():
            return output.strip()
        return "MCP tool call reported error"

    # Some real MCP servers return error strings inside content while leaving `isError=false`.
    # Match the local executor's convention for string error outputs.
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return None
        if text.startswith("Error:"):
            cleaned = text[len("Error:") :].strip()
            return cleaned or text
        if text.startswith(("❌", "🚫", "⏰")):
            cleaned = text.lstrip("❌🚫⏰").strip()
            if cleaned.startswith("Error:"):
                cleaned = cleaned[len("Error:") :].strip()
            return cleaned or text
        if text.lower().startswith("traceback"):
            return text
    return None


class McpToolExecutor:
    """Executes tool calls remotely via an MCP server (Streamable HTTP / JSON-RPC)."""

    def __init__(
        self,
        *,
        server_id: str,
        mcp_url: str,
        timeout_s: Optional[float] = 30.0,
        mcp_client: Optional[Any] = None,
    ):
        self._server_id = str(server_id or "").strip()
        if not self._server_id:
            raise ValueError("McpToolExecutor requires a non-empty server_id")
        self._mcp_url = str(mcp_url or "").strip()
        if not self._mcp_url:
            raise ValueError("McpToolExecutor requires a non-empty mcp_url")
        self._timeout_s = _normalize_timeout_s(timeout_s)
        self._mcp_client = mcp_client

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        from abstractcore.mcp import McpClient, parse_namespaced_tool_name

        results: List[Dict[str, Any]] = []
        client = self._mcp_client or McpClient(url=self._mcp_url, timeout_s=self._timeout_s)
        close_client = self._mcp_client is None
        try:
            for tc in tool_calls:
                name = str(tc.get("name", "") or "")
                call_id = str(tc.get("call_id") or "")
                raw_arguments = tc.get("arguments") or {}
                arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else {}

                remote_name = name
                parsed = parse_namespaced_tool_name(name)
                if parsed is not None:
                    server_id, tool_name = parsed
                    if server_id != self._server_id:
                        results.append(
                            {
                                "call_id": call_id,
                                "name": name,
                                "success": False,
                                "output": None,
                                "error": f"MCP tool '{name}' targets server '{server_id}', expected '{self._server_id}'",
                            }
                        )
                        continue
                    remote_name = tool_name

                try:
                    mcp_result = client.call_tool(name=remote_name, arguments=arguments)
                    err = _mcp_result_to_error(mcp_result)
                    if err is not None:
                        results.append(
                            {
                                "call_id": call_id,
                                "name": name,
                                "success": False,
                                "output": None,
                                "error": err,
                            }
                        )
                        continue
                    results.append(
                        {
                            "call_id": call_id,
                            "name": name,
                            "success": True,
                            "output": _mcp_result_to_output(mcp_result),
                            "error": None,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "call_id": call_id,
                            "name": name,
                            "success": False,
                            "output": None,
                            "error": str(e),
                        }
                    )

        finally:
            if close_client:
                try:
                    client.close()
                except Exception:
                    pass

        return {"mode": "executed", "results": results}


class DelegatingMcpToolExecutor:
    """Delegates tool calls to an MCP server by returning a durable JOB wait payload.

    This executor does not execute tools directly; it packages the tool calls plus
    MCP endpoint metadata into a `WAITING` state so an external worker can execute
    them and resume the run with results.
    """

    def __init__(
        self,
        *,
        server_id: str,
        mcp_url: str,
        transport: str = "streamable_http",
        wait_key_factory: Optional[Callable[[], str]] = None,
    ):
        self._server_id = str(server_id or "").strip()
        if not self._server_id:
            raise ValueError("DelegatingMcpToolExecutor requires a non-empty server_id")
        self._mcp_url = str(mcp_url or "").strip()
        if not self._mcp_url:
            raise ValueError("DelegatingMcpToolExecutor requires a non-empty mcp_url")
        self._transport = str(transport or "").strip() or "streamable_http"
        self._wait_key_factory = wait_key_factory or (lambda: f"mcp_job:{uuid.uuid4().hex}")

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "mode": "delegated",
            "wait_reason": "job",
            "wait_key": self._wait_key_factory(),
            "tool_calls": _jsonable(tool_calls),
            "details": {
                "protocol": "mcp",
                "transport": self._transport,
                "url": self._mcp_url,
                "server_id": self._server_id,
                "tool_name_prefix": f"mcp::{self._server_id}::",
            },
        }
