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
from typing import Any, Dict, List, Optional, Protocol

from .logging import get_logger

logger = get_logger(__name__)


class ToolExecutor(Protocol):
    def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]: ...


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
        for r in results:
            normalized.append(
                {
                    "call_id": getattr(r, "call_id", ""),
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

