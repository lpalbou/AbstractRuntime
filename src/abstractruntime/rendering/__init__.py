"""Rendering utilities (JSON-safe) for host UX and workflow nodes.

This module intentionally lives in AbstractRuntime so multiple hosts (AbstractFlow,
AbstractCode, future runners) can reuse the same rendering logic without duplicating
semantics in higher layers.
"""

from .agent_trace_report import render_agent_trace_markdown
from .json_stringify import JsonStringifyMode, stringify_json

__all__ = [
    "JsonStringifyMode",
    "render_agent_trace_markdown",
    "stringify_json",
]


