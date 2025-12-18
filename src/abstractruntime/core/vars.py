"""RunState.vars namespacing helpers.

AbstractRuntime treats `RunState.vars` as JSON-serializable user/workflow state.
To avoid key collisions and to clarify ownership, we use a simple convention:

- `context`: user-facing context (task, conversation, inputs)
- `scratchpad`: agent/workflow working memory (iteration counters, plans)
- `_runtime`: runtime/host-managed metadata (tool specs, inbox, etc.)
- `_temp`: ephemeral step-to-step values (llm_response, tool_results, etc.)
- `_limits`: runtime resource limits (max_iterations, max_tokens, etc.)

This is a convention, not a strict schema; helpers here are intentionally small.
"""

from __future__ import annotations

from typing import Any, Dict

CONTEXT = "context"
SCRATCHPAD = "scratchpad"
RUNTIME = "_runtime"
TEMP = "_temp"
LIMITS = "_limits"  # Canonical storage for runtime resource limits


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


def get_limits(vars: Dict[str, Any]) -> Dict[str, Any]:
    """Get the _limits namespace, creating with defaults if missing."""
    if LIMITS not in vars or not isinstance(vars.get(LIMITS), dict):
        vars[LIMITS] = _default_limits()
    return vars[LIMITS]  # type: ignore[return-value]


def ensure_limits(vars: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure _limits namespace exists with defaults.

    This is the canonical location for runtime resource limits:
    - max_iterations / current_iteration: Iteration control
    - max_tokens / estimated_tokens_used: Token/context window management
    - max_history_messages: Conversation history limit (-1 = unlimited)
    - warn_*_pct: Warning thresholds for proactive notifications

    Returns:
        The _limits dict (mutable reference into vars)
    """
    return get_limits(vars)


def _default_limits() -> Dict[str, Any]:
    """Return default limits dict."""
    return {
        "max_iterations": 25,
        "current_iteration": 0,
        "max_tokens": 32768,
        "max_output_tokens": None,
        "max_history_messages": -1,
        "estimated_tokens_used": 0,
        "warn_iterations_pct": 80,
        "warn_tokens_pct": 80,
    }

