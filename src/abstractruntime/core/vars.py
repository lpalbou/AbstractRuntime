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
NODE_TRACES = "node_traces"  # _runtime namespace key for per-node execution traces


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


def get_node_traces(vars: Dict[str, Any]) -> Dict[str, Any]:
    """Return the runtime-owned per-node trace mapping.

    Stored under `run.vars["_runtime"]["node_traces"]`.
    This is intended for host UX/debugging and for exposing traces to higher layers.
    """
    runtime_ns = get_runtime(vars)
    traces = runtime_ns.get(NODE_TRACES)
    if not isinstance(traces, dict):
        traces = {}
        runtime_ns[NODE_TRACES] = traces
    return traces


def get_node_trace(vars: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Return a single node trace object (always a dict)."""
    traces = get_node_traces(vars)
    trace = traces.get(node_id)
    if isinstance(trace, dict):
        return trace
    return {"node_id": node_id, "steps": []}


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


def parse_vars_path(path: str) -> list[Any]:
    """Parse a path for inspecting `RunState.vars`.

    Supports:
      - dot paths: "scratchpad.research.sources[0].title"
      - JSON pointer-ish paths: "/scratchpad/research/sources/0/title"
    """
    import re

    raw = str(path or "").strip()
    if not raw:
        return []

    tokens: list[Any] = []

    if raw.startswith("/"):
        for part in [p for p in raw.split("/") if p]:
            part = part.replace("~1", "/").replace("~0", "~")
            if part.isdigit():
                tokens.append(int(part))
            else:
                tokens.append(part)
        return tokens

    for part in [p for p in raw.split(".") if p]:
        # Allow list indexing as a bare segment: `foo.0.bar`
        if "[" not in part and part.isdigit():
            tokens.append(int(part))
            continue

        # Split `foo[0][1]` into ["foo", 0, 1]
        for m in re.finditer(r"([^\[\]]+)|\[(\d+)\]", part):
            key = m.group(1)
            idx = m.group(2)
            if key is not None:
                tokens.append(key)
            elif idx is not None:
                tokens.append(int(idx))

    return tokens


def resolve_vars_path(root: Any, tokens: list[Any]) -> Any:
    """Resolve tokens against nested dict/list structures."""
    cur: Any = root
    at: list[str] = []

    for tok in tokens:
        if isinstance(tok, int):
            if not isinstance(cur, list):
                where = ".".join([p for p in at if p]) or "(root)"
                raise ValueError(f"Expected list at {where} but found {type(cur).__name__}")
            if tok < 0 or tok >= len(cur):
                where = ".".join([p for p in at if p]) or "(root)"
                raise ValueError(f"Index {tok} out of range at {where} (len={len(cur)})")
            cur = cur[tok]
            at.append(str(tok))
            continue

        key = str(tok)
        if not isinstance(cur, dict):
            where = ".".join([p for p in at if p]) or "(root)"
            raise ValueError(f"Expected object at {where} but found {type(cur).__name__}")
        if key not in cur:
            where = ".".join([p for p in at if p]) or "(root)"
            raise ValueError(f"Missing key '{key}' at {where}")
        cur = cur[key]
        at.append(key)

    return cur
