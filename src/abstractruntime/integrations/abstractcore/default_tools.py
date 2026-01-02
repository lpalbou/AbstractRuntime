"""Default toolsets for AbstractRuntime's AbstractCore integration.

This module provides a *host-side* convenience list of common, safe(ish) tools
that can be wired into a Runtime via MappingToolExecutor.

Design notes:
- We keep the runtime kernel dependency-light; this lives under
  `integrations/abstractcore/` which is the explicit opt-in to AbstractCore.
- Tool callables are never persisted in RunState; only ToolSpecs (dicts) are.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence


ToolCallable = Callable[..., Any]


def _tool_name(func: ToolCallable) -> str:
    tool_def = getattr(func, "_tool_definition", None)
    if tool_def is not None:
        name = getattr(tool_def, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = getattr(func, "__name__", "")
    return str(name or "").strip()


def _tool_spec(func: ToolCallable) -> Dict[str, Any]:
    tool_def = getattr(func, "_tool_definition", None)
    if tool_def is not None and hasattr(tool_def, "to_dict"):
        return dict(tool_def.to_dict())

    from abstractcore.tools.core import ToolDefinition

    return dict(ToolDefinition.from_function(func).to_dict())


def get_default_toolsets() -> Dict[str, Dict[str, Any]]:
    """Return default toolsets {id -> {label, tools:[callables]}}."""
    from abstractcore.tools.common_tools import (
        list_files,
        read_file,
        search_files,
        analyze_code,
        write_file,
        edit_file,
        web_search,
        fetch_url,
        execute_command,
    )

    return {
        "files": {
            "id": "files",
            "label": "Files",
            "tools": [list_files, search_files, analyze_code, read_file, write_file, edit_file],
        },
        "web": {
            "id": "web",
            "label": "Web",
            "tools": [web_search, fetch_url],
        },
        "system": {
            "id": "system",
            "label": "System",
            "tools": [execute_command],
        },
    }


def get_default_tools() -> List[ToolCallable]:
    """Return the flattened list of all default tool callables."""
    toolsets = get_default_toolsets()
    out: list[ToolCallable] = []
    seen: set[str] = set()
    for spec in toolsets.values():
        for tool in spec.get("tools", []):
            if not callable(tool):
                continue
            name = _tool_name(tool)
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(tool)
    return out


def list_default_tool_specs() -> List[Dict[str, Any]]:
    """Return ToolSpecs for UI and LLM payloads (JSON-safe)."""
    toolsets = get_default_toolsets()
    toolset_by_name: Dict[str, str] = {}
    for tid, spec in toolsets.items():
        for tool in spec.get("tools", []):
            if callable(tool):
                name = _tool_name(tool)
                if name:
                    toolset_by_name[name] = tid

    out: list[Dict[str, Any]] = []
    for tool in get_default_tools():
        spec = _tool_spec(tool)
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        spec["toolset"] = toolset_by_name.get(name) or "other"
        out.append(spec)

    # Stable ordering: toolset then name
    out.sort(key=lambda s: (str(s.get("toolset") or ""), str(s.get("name") or "")))
    return out


def build_default_tool_map() -> Dict[str, ToolCallable]:
    """Return {tool_name -> callable} for MappingToolExecutor."""
    tool_map: Dict[str, ToolCallable] = {}
    for tool in get_default_tools():
        name = _tool_name(tool)
        if not name:
            continue
        tool_map[name] = tool
    return tool_map


def filter_tool_specs(tool_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Return ToolSpecs for the requested tool names (order preserved)."""
    available = {str(s.get("name")): s for s in list_default_tool_specs() if isinstance(s.get("name"), str)}
    out: list[Dict[str, Any]] = []
    for name in tool_names:
        spec = available.get(name)
        if spec is not None:
            out.append(spec)
    return out
