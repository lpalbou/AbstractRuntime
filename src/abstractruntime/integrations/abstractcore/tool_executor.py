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
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .logging import get_logger

logger = get_logger(__name__)


class ToolExecutor(Protocol):
    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]: ...


class MappingToolExecutor:
    """Executes tool calls using an explicit {tool_name -> callable} mapping.

    This is the recommended durable execution path: the mapping is held by the
    host/runtime process and is never persisted inside RunState.
    """

    def __init__(self, tool_map: Dict[str, Callable[..., Any]]):
        self._tool_map = dict(tool_map)

    @classmethod
    def from_tools(cls, tools: Sequence[Callable[..., Any]]) -> "MappingToolExecutor":
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

        return cls(tool_map)

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []

        for tc in tool_calls:
            name = str(tc.get("name", "") or "")
            arguments = dict(tc.get("arguments") or {})
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

            try:
                output = func(**arguments)
                results.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "success": True,
                        "output": _jsonable(output),
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

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        from abstractcore.tools.core import ToolCall
        from abstractcore.tools.registry import execute_tools

        calls = [
            ToolCall(
                name=str(tc.get("name")),
                arguments=dict(tc.get("arguments") or {}),
                call_id=tc.get("call_id"),
            )
            for tc in tool_calls
        ]

        results = execute_tools(calls)
        normalized = []
        for call, r in zip(calls, results):
            normalized.append(
                {
                    "call_id": getattr(r, "call_id", ""),
                    "name": getattr(call, "name", ""),
                    "success": bool(getattr(r, "success", False)),
                    "output": _jsonable(getattr(r, "output", None)),
                    "error": getattr(r, "error", None),
                }
            )

        return {"mode": "executed", "results": normalized}


class PassthroughToolExecutor:
    """Returns tool calls unchanged without executing them."""

    def __init__(self, *, mode: str = "passthrough"):
        self._mode = mode

    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"mode": self._mode, "tool_calls": _jsonable(tool_calls)}
