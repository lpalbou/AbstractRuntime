"""Built-in function handlers for visual nodes.

These are intentionally pure and JSON-friendly so visual workflows can run in
any host that can compile the VisualFlow JSON to a WorkflowSpec.
"""

from __future__ import annotations

import ast
from datetime import datetime
import json
import locale
import os
import random
from typing import Any, Callable, Dict, List, Optional

from abstractruntime.rendering import render_agent_trace_markdown as runtime_render_agent_trace_markdown
from abstractruntime.rendering import stringify_json as runtime_stringify_json


def get_builtin_handler(node_type: str) -> Optional[Callable[[Any], Any]]:
    """Get a built-in handler function for a node type."""
    return BUILTIN_HANDLERS.get(node_type)


def _path_tokens(path: str) -> list[Any]:
    """Parse a dotted/bracket path into tokens.

    Supported:
    - `a.b.c`
    - `a[0].b`

    Returns tokens as str keys and int indices.
    """
    import re

    p = str(path or "").strip()
    if not p:
        return []
    token_re = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")
    out: list[Any] = []
    for m in token_re.finditer(p):
        key = m.group(1)
        if key is not None:
            k = key.strip()
            if k:
                out.append(k)
            continue
        idx = m.group(2)
        if idx is not None:
            try:
                out.append(int(idx))
            except Exception:
                continue
    return out


def _get_path(value: Any, path: str) -> Any:
    """Best-effort nested lookup (dict keys + list indices)."""
    tokens = _path_tokens(path)
    if not tokens:
        return None
    current: Any = value
    for tok in tokens:
        if isinstance(current, dict) and isinstance(tok, str):
            current = current.get(tok)
            continue
        if isinstance(current, list):
            idx: Optional[int] = None
            if isinstance(tok, int):
                idx = tok
            elif isinstance(tok, str) and tok.isdigit():
                idx = int(tok)
            if idx is None:
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


# Math operations
def math_add(inputs: Dict[str, Any]) -> float:
    """Add two numbers."""
    return float(inputs.get("a", 0)) + float(inputs.get("b", 0))


def math_subtract(inputs: Dict[str, Any]) -> float:
    """Subtract b from a."""
    return float(inputs.get("a", 0)) - float(inputs.get("b", 0))


def math_multiply(inputs: Dict[str, Any]) -> float:
    """Multiply two numbers."""
    return float(inputs.get("a", 0)) * float(inputs.get("b", 0))


def math_divide(inputs: Dict[str, Any]) -> float:
    """Divide a by b."""
    b = float(inputs.get("b", 1))
    if b == 0:
        raise ValueError("Division by zero")
    return float(inputs.get("a", 0)) / b


def math_modulo(inputs: Dict[str, Any]) -> float:
    """Get remainder of a divided by b."""
    b = float(inputs.get("b", 1))
    if b == 0:
        raise ValueError("Modulo by zero")
    return float(inputs.get("a", 0)) % b


def math_power(inputs: Dict[str, Any]) -> float:
    """Raise base to exponent power."""
    return float(inputs.get("base", 0)) ** float(inputs.get("exp", 1))


def math_abs(inputs: Dict[str, Any]) -> float:
    """Get absolute value."""
    return abs(float(inputs.get("value", 0)))


def math_round(inputs: Dict[str, Any]) -> float:
    """Round to specified decimal places."""
    value = float(inputs.get("value", 0))
    decimals = int(inputs.get("decimals", 0))
    return round(value, decimals)


def math_random_int(inputs: Dict[str, Any]) -> int:
    """Random integer in [min, max] (inclusive)."""
    raw_min = inputs.get("min", 0)
    raw_max = inputs.get("max", 100)
    try:
        min_v = int(float(raw_min))
    except Exception:
        min_v = 0
    try:
        max_v = int(float(raw_max))
    except Exception:
        max_v = 100

    if max_v < min_v:
        min_v, max_v = max_v, min_v
    return random.randint(min_v, max_v)


def math_random_float(inputs: Dict[str, Any]) -> float:
    """Random float in [min, max]."""
    raw_min = inputs.get("min", 0)
    raw_max = inputs.get("max", 1)
    try:
        min_v = float(raw_min)
    except Exception:
        min_v = 0.0
    try:
        max_v = float(raw_max)
    except Exception:
        max_v = 1.0

    if max_v < min_v:
        min_v, max_v = max_v, min_v
    if max_v == min_v:
        return float(min_v)
    return min_v + (max_v - min_v) * random.random()


# String operations
def string_concat(inputs: Dict[str, Any]) -> str:
    """Concatenate two strings."""
    return str(inputs.get("a", "")) + str(inputs.get("b", ""))


def string_split(inputs: Dict[str, Any]) -> List[str]:
    """Split a string by a delimiter (defaults are tuned for real-world workflow usage).

    Notes:
    - Visual workflows often use human-edited / LLM-generated text where trailing
      delimiters are common (e.g. "A@@B@@"). A strict `str.split` would produce an
      empty last element and create a spurious downstream loop iteration.
    - We therefore support optional normalization flags with sensible defaults:
      - `trim` (default True): strip whitespace around parts
      - `drop_empty` (default True): drop empty parts after trimming
    - Delimiters may be entered as escape sequences (e.g. "\\n") from the UI.
    """

    raw_text = inputs.get("text", "")
    text = "" if raw_text is None else str(raw_text)

    raw_delim = inputs.get("delimiter", ",")
    delimiter = "" if raw_delim is None else str(raw_delim)
    delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    trim = bool(inputs.get("trim", True))
    drop_empty = bool(inputs.get("drop_empty", True))

    # Avoid ValueError from Python's `split("")` and keep behavior predictable.
    if delimiter == "":
        parts = [text] if text else []
    else:
        raw_maxsplit = inputs.get("maxsplit")
        maxsplit: Optional[int] = None
        if raw_maxsplit is not None:
            try:
                maxsplit = int(raw_maxsplit)
            except Exception:
                maxsplit = None
        if maxsplit is not None and maxsplit >= 0:
            parts = text.split(delimiter, maxsplit)
        else:
            parts = text.split(delimiter)

    if trim:
        parts = [p.strip() for p in parts]

    if drop_empty:
        parts = [p for p in parts if p != ""]

    return parts


def string_join(inputs: Dict[str, Any]) -> str:
    """Join array items with delimiter."""
    items = inputs.get("items")
    # Visual workflows frequently pass optional pins; treat `null` as empty.
    if items is None:
        items_list: list[Any] = []
    elif isinstance(items, list):
        items_list = items
    elif isinstance(items, tuple):
        items_list = list(items)
    else:
        # Defensive: if a non-array leaks in, treat it as a single element instead of
        # iterating characters (strings) or keys (dicts).
        items_list = [items]

    delimiter = str(inputs.get("delimiter", ","))
    # UI often stores escape sequences (e.g. "\\n") in JSON.
    delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    return delimiter.join("" if item is None else str(item) for item in items_list)


def string_format(inputs: Dict[str, Any]) -> str:
    """Format string with values."""
    template = str(inputs.get("template", ""))
    values = inputs.get("values", {})
    if isinstance(values, dict):
        return template.format(**values)
    return template


def string_uppercase(inputs: Dict[str, Any]) -> str:
    """Convert to uppercase."""
    return str(inputs.get("text", "")).upper()


def string_lowercase(inputs: Dict[str, Any]) -> str:
    """Convert to lowercase."""
    return str(inputs.get("text", "")).lower()


def string_trim(inputs: Dict[str, Any]) -> str:
    """Trim whitespace."""
    return str(inputs.get("text", "")).strip()


def string_substring(inputs: Dict[str, Any]) -> str:
    """Get substring."""
    text = str(inputs.get("text", ""))
    start = int(inputs.get("start", 0))
    end = inputs.get("end")
    if end is not None:
        return text[start : int(end)]
    return text[start:]


def string_length(inputs: Dict[str, Any]) -> int:
    """Get string length."""
    return len(str(inputs.get("text", "")))


def string_contains(inputs: Dict[str, Any]) -> bool:
    """Return True if `text` contains `pattern` (substring match)."""
    pattern_raw = inputs.get("pattern")
    if pattern_raw is None:
        return False
    pattern = str(pattern_raw)
    # Avoid surprising behavior where "" is "contained" in every string.
    if pattern == "":
        return False

    text_raw = inputs.get("text")
    text = "" if text_raw is None else str(text_raw)
    return pattern in text


def string_replace(inputs: Dict[str, Any]) -> str:
    """Replace `pattern` in `text` with `replacement`.

    mode:
    - "first": replace the first occurrence
    - "all" (default): replace all occurrences
    """
    text_raw = inputs.get("text")
    text = "" if text_raw is None else str(text_raw)

    pattern_raw = inputs.get("pattern")
    if pattern_raw is None:
        return text
    pattern = str(pattern_raw)
    if pattern == "":
        return text

    replacement_raw = inputs.get("replacement")
    replacement = "" if replacement_raw is None else str(replacement_raw)

    mode_raw = inputs.get("mode")
    mode = str(mode_raw).strip().lower() if mode_raw is not None else "all"

    if mode in {"first", "once", "1"}:
        return text.replace(pattern, replacement, 1)
    if mode in {"all", "*", "global"}:
        return text.replace(pattern, replacement)

    # Best-effort support for numeric counts (e.g. mode="2").
    try:
        n = int(mode_raw)  # type: ignore[arg-type]
        if n < 0:
            return text.replace(pattern, replacement)
        return text.replace(pattern, replacement, n)
    except Exception:
        return text.replace(pattern, replacement)


def string_template(inputs: Dict[str, Any]) -> str:
    """Render a template with placeholders like `{{path.to.value}}`.

    Supported filters:
    - `| json`            -> json.dumps(value)
    - `| join(", ")`      -> join array values with delimiter
    - `| trim` / `| lower` / `| upper`
    """
    import re

    template = str(inputs.get("template", "") or "")
    vars_raw = inputs.get("vars")
    vars_obj = vars_raw if isinstance(vars_raw, dict) else {}

    pat = re.compile(r"\{\{\s*(.*?)\s*\}\}")

    def _apply_filters(value: Any, filters: list[str]) -> Any:
        cur = value
        for f in filters:
            f = f.strip()
            if not f:
                continue
            if f == "json":
                cur = json.dumps(cur, ensure_ascii=False, sort_keys=True)
                continue
            if f.startswith("join"):
                m = re.match(r"join\((.*)\)$", f)
                delim = ", "
                if m:
                    raw = m.group(1).strip()
                    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                        raw = raw[1:-1]
                    delim = raw
                if isinstance(cur, list):
                    cur = delim.join("" if x is None else str(x) for x in cur)
                else:
                    cur = "" if cur is None else str(cur)
                continue
            if f == "trim":
                cur = ("" if cur is None else str(cur)).strip()
                continue
            if f == "lower":
                cur = ("" if cur is None else str(cur)).lower()
                continue
            if f == "upper":
                cur = ("" if cur is None else str(cur)).upper()
                continue
            # Unknown filters are ignored (best-effort, stable).
        return cur

    def _render_expr(expr: str) -> str:
        parts = [p.strip() for p in str(expr or "").split("|")]
        path = parts[0] if parts else ""
        filters = parts[1:] if len(parts) > 1 else []
        value = _get_path(vars_obj, path)
        if value is None:
            return ""
        value = _apply_filters(value, filters)
        return "" if value is None else str(value)

    return pat.sub(lambda m: _render_expr(m.group(1)), template)


# Control flow helpers (these return decision values, not execution control)
def control_compare(inputs: Dict[str, Any]) -> bool:
    """Compare two values."""
    a = inputs.get("a")
    b = inputs.get("b")
    op = str(inputs.get("op", "=="))

    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        try:
            return a < b
        except Exception:
            return False
    if op == "<=":
        try:
            return a <= b
        except Exception:
            return False
    if op == ">":
        try:
            return a > b
        except Exception:
            return False
    if op == ">=":
        try:
            return a >= b
        except Exception:
            return False
    raise ValueError(f"Unknown comparison operator: {op}")


def control_not(inputs: Dict[str, Any]) -> bool:
    """Logical NOT."""
    return not bool(inputs.get("value", False))


def control_and(inputs: Dict[str, Any]) -> bool:
    """Logical AND."""
    return bool(inputs.get("a", False)) and bool(inputs.get("b", False))


def control_or(inputs: Dict[str, Any]) -> bool:
    """Logical OR."""
    return bool(inputs.get("a", False)) or bool(inputs.get("b", False))


def control_coalesce(inputs: Dict[str, Any]) -> Any:
    """Return the first non-None input in pin order.

    Pin order is injected by the visual executor as `_pin_order` based on the node's
    input pin list, so selection is deterministic and matches the visual layout.
    """
    order = inputs.get("_pin_order")
    pin_order: list[str] = []
    if isinstance(order, list):
        for x in order:
            if isinstance(x, str) and x:
                pin_order.append(x)
    if not pin_order:
        pin_order = ["a", "b"]

    for pid in pin_order:
        if pid not in inputs:
            continue
        v = inputs.get(pid)
        if v is not None:
            return v
    return None


# Data operations
def data_get(inputs: Dict[str, Any]) -> Any:
    """Get property from object."""
    obj = inputs.get("object", {})
    key = str(inputs.get("key", ""))
    default = inputs.get("default")

    value = _get_path(obj, key)
    if value is None:
        return {"value": default}
    return {"value": value}


def data_set(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Set property on object (returns new object)."""
    obj = dict(inputs.get("object", {}))
    key = str(inputs.get("key", ""))
    value = inputs.get("value")

    # Support dot notation
    parts = key.split(".")
    current = obj
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value
    return obj


def data_merge(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two objects."""
    a = dict(inputs.get("a", {}))
    b = dict(inputs.get("b", {}))
    return {**a, **b}


def data_make_object(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a flat JSON object from the provided inputs.

    VisualFlow's executor injects helper keys like `_pin_order` / `_literalValue`;
    this node ignores keys starting with "_" to keep output stable.
    """
    out: Dict[str, Any] = {}
    for k, v in (inputs or {}).items():
        if not isinstance(k, str):
            continue
        if k.startswith("_"):
            continue
        out[k] = v
    return out


def data_make_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a context object {task, messages, ...extra}."""
    extra = inputs.get("extra")
    out: Dict[str, Any] = dict(extra) if isinstance(extra, dict) else {}

    task = inputs.get("task")
    if task is None:
        out["task"] = ""
    elif isinstance(task, str):
        out["task"] = task
    else:
        out["task"] = str(task)

    messages = inputs.get("messages")
    if isinstance(messages, list):
        out["messages"] = list(messages)
    elif isinstance(messages, tuple):
        out["messages"] = list(messages)
    elif messages is None:
        out["messages"] = []
    else:
        out["messages"] = [messages]

    return {"context": out}


def data_add_message(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a canonical message object for context.messages.

    Shape:
    {
      "role": "...",
      "content": "...",
      "timestamp": "<utc iso>",
      "metadata": {"message_id": "msg_<hex>"}
    }
    """
    role_raw = inputs.get("role")
    role = str(role_raw) if role_raw is not None else "user"

    content_raw = inputs.get("content")
    content = str(content_raw) if content_raw is not None else ""

    try:
        from datetime import timezone

        timestamp = datetime.now(timezone.utc).isoformat()
    except Exception:
        timestamp = datetime.utcnow().isoformat() + "Z"

    import uuid

    return {
        "message": {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "metadata": {"message_id": f"msg_{uuid.uuid4().hex}"},
        }
    }


def data_make_meta(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a host-facing meta envelope (portable, no strict schema enforcement)."""
    extra = inputs.get("extra")
    out: Dict[str, Any] = dict(extra) if isinstance(extra, dict) else {}

    schema = inputs.get("schema")
    schema_str = schema.strip() if isinstance(schema, str) and schema.strip() else "abstractcode.agent.v1.meta"

    version = inputs.get("version")
    version_int = 1
    try:
        if isinstance(version, bool):
            version_int = 1
        elif isinstance(version, (int, float)):
            version_int = int(version)
        elif isinstance(version, str) and version.strip():
            version_int = int(float(version.strip()))
    except Exception:
        version_int = 1

    out["schema"] = schema_str
    out["version"] = version_int

    provider = inputs.get("provider")
    if isinstance(provider, str) and provider.strip():
        out["provider"] = provider.strip()
    elif provider is not None and str(provider).strip():
        out["provider"] = str(provider).strip()

    model = inputs.get("model")
    if isinstance(model, str) and model.strip():
        out["model"] = model.strip()
    elif model is not None and str(model).strip():
        out["model"] = str(model).strip()

    usage = inputs.get("usage")
    if isinstance(usage, dict):
        out["usage"] = dict(usage)

    trace_id = inputs.get("trace_id")
    trace_id_str = trace_id.strip() if isinstance(trace_id, str) and trace_id.strip() else ""
    if trace_id_str:
        trace = out.get("trace")
        trace_obj: Dict[str, Any] = dict(trace) if isinstance(trace, dict) else {}
        trace_obj["trace_id"] = trace_id_str
        out["trace"] = trace_obj

    warnings = inputs.get("warnings")
    if isinstance(warnings, list):
        out["warnings"] = [str(w) for w in warnings if str(w).strip()]
    elif isinstance(warnings, tuple):
        out["warnings"] = [str(w) for w in list(warnings) if str(w).strip()]
    elif isinstance(warnings, str) and warnings.strip():
        out["warnings"] = [warnings.strip()]

    debug = inputs.get("debug")
    if isinstance(debug, dict):
        out["debug"] = dict(debug)
    elif debug is not None:
        out["debug"] = debug

    return {"meta": out}


def data_make_scratchpad(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a scratchpad/trace envelope (commonly from Agent outputs)."""
    extra = inputs.get("extra")
    out: Dict[str, Any] = dict(extra) if isinstance(extra, dict) else {}

    sub_run_id = inputs.get("sub_run_id")
    if isinstance(sub_run_id, str) and sub_run_id.strip():
        out["sub_run_id"] = sub_run_id.strip()

    workflow_id = inputs.get("workflow_id")
    if isinstance(workflow_id, str) and workflow_id.strip():
        out["workflow_id"] = workflow_id.strip()

    node_traces = inputs.get("node_traces")
    out["node_traces"] = dict(node_traces) if isinstance(node_traces, dict) else {}

    steps = inputs.get("steps")
    if isinstance(steps, list):
        out["steps"] = list(steps)
    elif isinstance(steps, tuple):
        out["steps"] = list(steps)
    elif steps is None:
        out["steps"] = []
    else:
        out["steps"] = [steps]

    return {"scratchpad": out}


def data_make_raw_result(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a normalized raw LLM result envelope (best-effort)."""
    extra = inputs.get("extra")
    out: Dict[str, Any] = dict(extra) if isinstance(extra, dict) else {}

    # Reserved keys: always set from pins (override extra).
    out["content"] = inputs.get("content")
    out["data"] = inputs.get("data")

    tool_calls = inputs.get("tool_calls")
    if isinstance(tool_calls, list):
        out["tool_calls"] = list(tool_calls)
    elif isinstance(tool_calls, tuple):
        out["tool_calls"] = list(tool_calls)
    elif tool_calls is None:
        out["tool_calls"] = []
    else:
        out["tool_calls"] = [tool_calls]

    out["usage"] = inputs.get("usage")

    model = inputs.get("model")
    out["model"] = model.strip() if isinstance(model, str) else model

    finish_reason = inputs.get("finish_reason")
    out["finish_reason"] = finish_reason.strip() if isinstance(finish_reason, str) else finish_reason

    metadata = inputs.get("metadata")
    out["metadata"] = dict(metadata) if isinstance(metadata, dict) else metadata

    trace_id = inputs.get("trace_id")
    out["trace_id"] = trace_id.strip() if isinstance(trace_id, str) else trace_id

    return {"raw_result": out}


def data_get_element(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Get an array element by index with an optional default.

    - Supports list/tuple as arrays; other inputs are treated as empty.
    - Index is coerced to int (floats truncated).
    - Negative indices are supported (Python-style).
    """
    arr_raw = inputs.get("array")
    if isinstance(arr_raw, list):
        arr = arr_raw
    elif isinstance(arr_raw, tuple):
        arr = list(arr_raw)
    else:
        arr = []

    default = inputs.get("default")
    raw_index = inputs.get("index", 0)
    try:
        if isinstance(raw_index, bool):
            idx = 0
        elif isinstance(raw_index, (int, float)):
            idx = int(raw_index)
        elif isinstance(raw_index, str) and raw_index.strip():
            idx = int(float(raw_index.strip()))
        else:
            idx = 0
    except Exception:
        idx = 0

    if idx < 0:
        idx = len(arr) + idx

    if 0 <= idx < len(arr):
        return {"result": arr[idx], "found": True}
    return {"result": default, "found": False}


def data_get_random_element(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Pick a random element from an array with an optional default."""
    arr_raw = inputs.get("array")
    if isinstance(arr_raw, list):
        arr = arr_raw
    elif isinstance(arr_raw, tuple):
        arr = list(arr_raw)
    else:
        arr = []

    default = inputs.get("default")
    if not arr:
        return {"result": default, "found": False}

    # Use global RNG; tests only assert membership/non-empty, so this stays non-flaky.
    return {"result": random.choice(arr), "found": True}


def data_array_map(inputs: Dict[str, Any]) -> List[Any]:
    """Map array items (extract property from each)."""
    items = inputs.get("items", [])
    key = str(inputs.get("key", ""))

    result: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            result.append(item.get(key))
        else:
            result.append(item)
    return result


def data_array_filter(inputs: Dict[str, Any]) -> List[Any]:
    """Filter array by condition."""
    items = inputs.get("items", [])
    key = str(inputs.get("key", ""))
    value = inputs.get("value")

    result: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            if item.get(key) == value:
                result.append(item)
        elif item == value:
            result.append(item)
    return result


def data_array_length(inputs: Dict[str, Any]) -> int:
    """Return array length (0 if not an array)."""
    items = inputs.get("array")
    if isinstance(items, list):
        return len(items)
    if isinstance(items, tuple):
        return len(list(items))
    return 0


def data_has_tools(inputs: Dict[str, Any]) -> bool:
    """Convenience: return True if the input array has at least one element.

    Intended for checking LLM `tool_calls` arrays, but works with any array-like input.
    Non-array / null inputs are treated as empty.
    """
    items = inputs.get("array")
    if isinstance(items, list):
        return len(items) > 0
    if isinstance(items, tuple):
        return len(items) > 0
    return False


def data_array_append(inputs: Dict[str, Any]) -> List[Any]:
    """Append an item to an array (returns a new array)."""
    items = inputs.get("array")
    item = inputs.get("item")
    out: list[Any]
    if isinstance(items, list):
        out = list(items)
    elif isinstance(items, tuple):
        out = list(items)
    elif items is None:
        out = []
    else:
        out = [items]
    out.append(item)
    return out


def data_array_dedup(inputs: Dict[str, Any]) -> List[Any]:
    """Stable-order dedup for arrays.

    If `key` is provided (string path), dedup objects by that path value.
    """
    items = inputs.get("array")
    if not isinstance(items, list):
        if isinstance(items, tuple):
            items = list(items)
        else:
            return []

    key = inputs.get("key")
    key_path = str(key or "").strip()

    def _fingerprint(v: Any) -> str:
        if v is None or isinstance(v, (bool, int, float, str)):
            return f"{type(v).__name__}:{v}"
        try:
            return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return str(v)

    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        if key_path:
            k = _get_path(item, key_path)
            fp = _fingerprint(k) if k is not None else _fingerprint(item)
        else:
            fp = _fingerprint(item)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(item)
    return out


def system_datetime(_: Dict[str, Any]) -> Dict[str, Any]:
    """Return current system date/time and best-effort locale metadata.

    All values are JSON-serializable and stable-keyed.
    """
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0

    tzname = now.tzname() or ""

    # Avoid deprecated locale.getdefaultlocale() in Python 3.12+.
    lang = os.environ.get("LC_ALL") or os.environ.get("LANG") or os.environ.get("LC_CTYPE") or ""
    env_locale = lang.split(".", 1)[0] if lang else ""

    loc = locale.getlocale()[0] or env_locale

    return {
        "iso": now.isoformat(),
        "timezone": tzname,
        "utc_offset_minutes": offset_minutes,
        "locale": loc or "",
    }


def data_parse_json(inputs: Dict[str, Any]) -> Any:
    """Parse JSON (or JSON-ish) text into a JSON-serializable Python value.

    Primary use-case: turn an LLM string response into an object/array that can be
    fed into `Break Object` (dynamic pins) or other data nodes.

    Behavior:
    - If the input is already a dict/list, returns it unchanged (idempotent).
    - Tries strict `json.loads` first.
    - If that fails, tries to extract the first JSON object/array substring and parse it.
    - As a last resort, tries `ast.literal_eval` to handle Python-style dicts/lists
      (common in LLM output), then converts to JSON-friendly types.
    - If the parsed value is a scalar, wraps it as `{ "value": <scalar> }` by default,
      so `Break Object` can still expose it.
    """

    def _strip_code_fence(text: str) -> str:
        s = text.strip()
        if not s.startswith("```"):
            return s
        # Opening fence line can be ```json / ```js etc; drop it.
        nl = s.find("\n")
        if nl == -1:
            return s.strip("`").strip()
        body = s[nl + 1 :]
        end = body.rfind("```")
        if end != -1:
            body = body[:end]
        return body.strip()

    def _jsonify(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): _jsonify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_jsonify(v) for v in value]
        if isinstance(value, tuple):
            return [_jsonify(v) for v in value]
        return str(value)

    raw = inputs.get("text")
    if isinstance(raw, (dict, list)):
        parsed: Any = raw
    else:
        if raw is None:
            raise ValueError("parse_json requires a non-empty 'text' input.")
        text = _strip_code_fence(str(raw))
        if not text.strip():
            raise ValueError("parse_json requires a non-empty 'text' input.")

        parsed = None
        text_stripped = text.strip()

        try:
            parsed = json.loads(text_stripped)
        except Exception:
            # Best-effort: find and parse the first JSON object/array substring.
            decoder = json.JSONDecoder()
            starts: list[int] = []
            for i, ch in enumerate(text_stripped):
                if ch in "{[":
                    starts.append(i)
                if len(starts) >= 64:
                    break
            for i in starts:
                try:
                    parsed, _end = decoder.raw_decode(text_stripped[i:])
                    break
                except Exception:
                    continue

        if parsed is None:
            # Last resort: tolerate Python-literal dict/list output.
            try:
                parsed = ast.literal_eval(text_stripped)
            except Exception as e:
                raise ValueError(f"Invalid JSON: {e}") from e

    parsed = _jsonify(parsed)

    wrap_scalar = bool(inputs.get("wrap_scalar", True))
    if wrap_scalar and not isinstance(parsed, (dict, list)):
        return {"value": parsed}
    return parsed


def data_stringify_json(inputs: Dict[str, Any]) -> str:
    """Render a JSON-like value into a string (runtime-owned implementation).

    The core stringify logic lives in AbstractRuntime so multiple hosts can reuse it.

    Supported inputs (backward compatible):
    - `value`: JSON value (dict/list/scalar) OR a JSON-ish string.
    - `mode`: none | beautify | minified
    - Legacy: `indent` (<=0 => minified; >0 => beautify with that indent)
    - Legacy: `sort_keys` (bool)
    """

    value = inputs.get("value")

    raw_mode = inputs.get("mode")
    mode = str(raw_mode).strip().lower() if isinstance(raw_mode, str) else ""

    raw_indent = inputs.get("indent")
    indent_n: Optional[int] = None
    if raw_indent is not None:
        try:
            indent_n = int(raw_indent)
        except Exception:
            indent_n = None

    raw_sort_keys = inputs.get("sort_keys")
    sort_keys = bool(raw_sort_keys) if isinstance(raw_sort_keys, bool) else False

    # If mode not provided, infer from legacy indent.
    if not mode:
        if indent_n is not None and indent_n <= 0:
            mode = "minified"
        elif indent_n is not None and indent_n > 0:
            mode = "beautify"
        else:
            mode = "beautify"

    return runtime_stringify_json(
        value,
        mode=mode,
        beautify_indent=indent_n if isinstance(indent_n, int) and indent_n > 0 else 2,
        sort_keys=sort_keys,
        parse_strings=True,
    )


def data_agent_trace_report(inputs: Dict[str, Any]) -> str:
    """Render an agent scratchpad (runtime-owned node traces) into Markdown."""
    scratchpad = inputs.get("scratchpad")
    return runtime_render_agent_trace_markdown(scratchpad)


# Literal value handlers - return configured constant values
def literal_string(inputs: Dict[str, Any]) -> str:
    """Return string literal value."""
    return str(inputs.get("_literalValue", ""))


def literal_number(inputs: Dict[str, Any]) -> float:
    """Return number literal value."""
    value = inputs.get("_literalValue", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def literal_boolean(inputs: Dict[str, Any]) -> bool:
    """Return boolean literal value."""
    return bool(inputs.get("_literalValue", False))


def literal_json(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON literal value."""
    value = inputs.get("_literalValue", {})
    if isinstance(value, (dict, list)):
        return value  # type: ignore[return-value]
    return {}


def literal_array(inputs: Dict[str, Any]) -> List[Any]:
    """Return array literal value."""
    value = inputs.get("_literalValue", [])
    if isinstance(value, list):
        return value
    return []


def tools_allowlist(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a workflow-scope tool allowlist as a named output.

    The visual editor stores the selected tools as a JSON array of strings in the
    node's `literalValue`. The executor injects it as `_literalValue`.
    """
    value = inputs.get("_literalValue", [])
    if not isinstance(value, list):
        return {"tools": []}
    out: list[str] = []
    for x in value:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    # Preserve order; remove duplicates.
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return {"tools": uniq}


# Handler registry
BUILTIN_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    # Math
    "add": math_add,
    "subtract": math_subtract,
    "multiply": math_multiply,
    "divide": math_divide,
    "modulo": math_modulo,
    "power": math_power,
    "abs": math_abs,
    "round": math_round,
    "random_int": math_random_int,
    "random_float": math_random_float,
    # String
    "concat": string_concat,
    "split": string_split,
    "join": string_join,
    "format": string_format,
    "string_template": string_template,
    "uppercase": string_uppercase,
    "lowercase": string_lowercase,
    "trim": string_trim,
    "substring": string_substring,
    "length": string_length,
    "contains": string_contains,
    "replace": string_replace,
    # Control
    "compare": control_compare,
    "not": control_not,
    "and": control_and,
    "or": control_or,
    "coalesce": control_coalesce,
    # Data
    "get": data_get,
    "set": data_set,
    "merge": data_merge,
    "make_object": data_make_object,
    "make_context": data_make_context,
    "add_message": data_add_message,
    "make_meta": data_make_meta,
    "make_scratchpad": data_make_scratchpad,
    "make_raw_result": data_make_raw_result,
    "get_element": data_get_element,
    "get_random_element": data_get_random_element,
    "array_map": data_array_map,
    "array_filter": data_array_filter,
    "array_length": data_array_length,
    "has_tools": data_has_tools,
    "array_append": data_array_append,
    "array_dedup": data_array_dedup,
    "parse_json": data_parse_json,
    "stringify_json": data_stringify_json,
    "agent_trace_report": data_agent_trace_report,
    "system_datetime": system_datetime,
    # Literals
    "literal_string": literal_string,
    "literal_number": literal_number,
    "literal_boolean": literal_boolean,
    "literal_json": literal_json,
    "literal_array": literal_array,
    "tools_allowlist": tools_allowlist,
}
