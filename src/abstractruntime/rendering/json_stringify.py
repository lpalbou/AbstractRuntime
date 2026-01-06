"""JSON stringify utilities.

Why this exists in AbstractRuntime:
- Many hosts need consistent "JSON → string" semantics (UI preview, reports, prompts).
- Keeping the core logic in runtime avoids host-specific divergence (layering, ADR-0001).

This intentionally stays dependency-light (stdlib only).
"""

from __future__ import annotations

import ast
import json
from enum import Enum
from typing import Any, Optional


class JsonStringifyMode(str, Enum):
    """Formatting mode for JSON stringification."""

    NONE = "none"  # default json.dumps formatting (single line, spaces after separators)
    BEAUTIFY = "beautify"  # multi-line, indented
    MINIFIED = "minified"  # condensed separators (no spaces)


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
    """Convert a value into JSON-serializable types (best-effort)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    return str(value)


def _parse_jsonish_maybe(text: str) -> Optional[Any]:
    """Best-effort parse of JSON-ish strings.

    Accepts:
    - strict JSON
    - JSON embedded in a larger string (extract first object/array substring)
    - Python-literal dict/list (common LLM output), via ast.literal_eval
    """
    s = _strip_code_fence(text)
    if not s:
        return None
    s = s.strip()
    if not s:
        return None

    try:
        return json.loads(s)
    except Exception:
        pass

    # Best-effort: parse the first JSON object/array substring.
    decoder = json.JSONDecoder()
    starts: list[int] = []
    for i, ch in enumerate(s):
        if ch in "{[":
            starts.append(i)
        if len(starts) >= 64:
            break
    for i in starts:
        try:
            parsed, _end = decoder.raw_decode(s[i:])
            return parsed
        except Exception:
            continue

    # Last resort: tolerate Python-literal dict/list output.
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def stringify_json(
    value: Any,
    *,
    mode: str | JsonStringifyMode = JsonStringifyMode.BEAUTIFY,
    beautify_indent: int = 2,
    sort_keys: bool = False,
    parse_strings: bool = True,
) -> str:
    """Render a JSON-like value into a string.

    Args:
        value: Any JSON-like value (dict/list/scalar). If `parse_strings=True`, a string
            that contains JSON (or JSON-ish text) is parsed and then rendered.
        mode: none | beautify | minified.
        beautify_indent: Indentation width for beautify mode.
        sort_keys: When true, sort object keys for deterministic output.
        parse_strings: When true, attempt to parse JSON-ish strings before rendering.
    """
    mode_value = mode.value if isinstance(mode, JsonStringifyMode) else str(mode or "").strip().lower()
    if mode_value not in {m.value for m in JsonStringifyMode}:
        mode_value = JsonStringifyMode.BEAUTIFY.value

    if parse_strings and isinstance(value, str) and value.strip():
        parsed = _parse_jsonish_maybe(value)
        if parsed is not None:
            value = parsed

    safe = _jsonify(value)

    if mode_value == JsonStringifyMode.MINIFIED.value:
        return json.dumps(safe, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))

    if mode_value == JsonStringifyMode.NONE.value:
        return json.dumps(safe, ensure_ascii=False, sort_keys=sort_keys)

    indent = beautify_indent if isinstance(beautify_indent, int) else 2
    if indent < 0:
        indent = 2
    return json.dumps(safe, ensure_ascii=False, sort_keys=sort_keys, indent=indent)


