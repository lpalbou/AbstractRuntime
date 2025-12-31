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
import threading
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

        def _error_from_output(value: Any) -> Optional[str]:
            """Detect tool failures reported as string outputs (instead of exceptions)."""
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
                results.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "success": False,
                        "output": None,
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
            arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else {}
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

            def _invoke() -> Any:
                try:
                    return func(**arguments)
                except TypeError:
                    # Retry once with sanitized kwargs for common wrapper/extra-arg failures.
                    unwrapped = _unwrap_wrapper_args(arguments)
                    filtered = _filter_kwargs(func, unwrapped)
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
