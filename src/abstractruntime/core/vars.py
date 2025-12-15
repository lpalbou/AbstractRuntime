"""RunState.vars namespacing helpers.

AbstractRuntime treats `RunState.vars` as JSON-serializable user/workflow state.
To avoid key collisions and to clarify ownership, we use a simple convention:

- `context`: user-facing context (task, conversation, inputs)
- `scratchpad`: agent/workflow working memory (iteration counters, plans)
- `_runtime`: runtime/host-managed metadata (tool specs, inbox, etc.)
- `_temp`: ephemeral step-to-step values (llm_response, tool_results, etc.)

This is a convention, not a strict schema; helpers here are intentionally small.
"""

from __future__ import annotations

from typing import Any, Dict

CONTEXT = "context"
SCRATCHPAD = "scratchpad"
RUNTIME = "_runtime"
TEMP = "_temp"


def ensure_namespaces(vars: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the four canonical namespaces exist and are dicts."""
    for key in (CONTEXT, SCRATCHPAD, RUNTIME, TEMP):
        current = vars.get(key)
        if not isinstance(current, dict):
            vars[key] = {}
    return vars


def get_namespace(vars: Dict[str, Any], key: str) -> Dict[str, Any]:
    ensure_namespaces(vars)
    return vars[key]  # type: ignore[return-value]


def get_context(vars: Dict[str, Any]) -> Dict[str, Any]:
    return get_namespace(vars, CONTEXT)


def get_scratchpad(vars: Dict[str, Any]) -> Dict[str, Any]:
    return get_namespace(vars, SCRATCHPAD)


def get_runtime(vars: Dict[str, Any]) -> Dict[str, Any]:
    return get_namespace(vars, RUNTIME)


def get_temp(vars: Dict[str, Any]) -> Dict[str, Any]:
    return get_namespace(vars, TEMP)


def clear_temp(vars: Dict[str, Any]) -> None:
    get_temp(vars).clear()

