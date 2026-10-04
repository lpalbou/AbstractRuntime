"""Runtime-owned AbstractCore structured-output facade for hosts.

Hosts (e.g. AbstractGateway's OpenAI-compatible endpoint) should import this
module instead of reaching into `abstractcore.structured.json_schema` directly.
This keeps the host -> Runtime -> AbstractCore boundary clean: the host never
imports `abstractcore` itself.

Scope:
- `parse_response_format`: check an OpenAI `response_format` object and return
  `(kind, schema, name)`; kind is "text", "json_object" or "json_schema"
- `ResponseFormatError`: the error it raises for an unusable `response_format`
  (AbstractCore's own class, so `except ResponseFormatError` catches exactly
  what Core raises; `.param` names the offending field)

Non-goals:
- driving structured generation (AbstractCore's server does that per provider)

Design mirrors `config_facade.py`: AbstractCore is imported lazily, so importing
this module stays light and a missing dependency gives an actionable error.
AbstractCore >= 2.25.0 provides `abstractcore.structured.json_schema`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = ["ResponseFormatError", "parse_response_format"]


def _json_schema_module():
    try:
        from abstractcore.structured import json_schema
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "AbstractCore structured-output support is unavailable. Install a Runtime "
            "environment with AbstractCore >= 2.25.0 (the `integrations/abstractcore` opt-in)."
        ) from exc
    return json_schema


def parse_response_format(raw: Any) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """(kind, schema, name) for a response_format; raises ResponseFormatError."""
    return _json_schema_module().parse_response_format(raw)


def __getattr__(name: str) -> Any:
    # `ResponseFormatError` is AbstractCore's class itself (not a copy), resolved
    # on first use so importing this module does not import AbstractCore.
    if name == "ResponseFormatError":
        return _json_schema_module().ResponseFormatError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
