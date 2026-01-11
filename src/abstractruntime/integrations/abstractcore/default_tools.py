"""Default toolsets for AbstractRuntime's AbstractCore integration.

This module provides a *host-side* convenience list of common, safe(ish) tools
that can be wired into a Runtime via MappingToolExecutor.

Design notes:
- We keep the runtime kernel dependency-light; this lives under
  `integrations/abstractcore/` which is the explicit opt-in to AbstractCore.
- Tool callables are never persisted in RunState; only ToolSpecs (dicts) are.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Sequence


ToolCallable = Callable[..., Any]

_COMMS_ENABLE_ENV_VARS = (
    "ABSTRACT_ENABLE_COMMS_TOOLS",
    "ABSTRACT_ENABLE_EMAIL_TOOLS",
    "ABSTRACT_ENABLE_WHATSAPP_TOOLS",
    "ABSTRACT_ENABLE_TELEGRAM_TOOLS",
)


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def comms_tools_enabled() -> bool:
    """Return True when the host explicitly opts into comms tools via env."""
    return any(_env_flag(k) for k in _COMMS_ENABLE_ENV_VARS)


def email_tools_enabled() -> bool:
    return _env_flag("ABSTRACT_ENABLE_COMMS_TOOLS") or _env_flag("ABSTRACT_ENABLE_EMAIL_TOOLS")


def whatsapp_tools_enabled() -> bool:
    return _env_flag("ABSTRACT_ENABLE_COMMS_TOOLS") or _env_flag("ABSTRACT_ENABLE_WHATSAPP_TOOLS")


def telegram_tools_enabled() -> bool:
    return _env_flag("ABSTRACT_ENABLE_COMMS_TOOLS") or _env_flag("ABSTRACT_ENABLE_TELEGRAM_TOOLS")


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

    toolsets: Dict[str, Dict[str, Any]] = {
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

    if comms_tools_enabled():
        comms: list[ToolCallable] = []
        if email_tools_enabled():
            from abstractcore.tools.comms_tools import list_emails, read_email, send_email

            comms.extend([send_email, list_emails, read_email])
        if whatsapp_tools_enabled():
            from abstractcore.tools.comms_tools import list_whatsapp_messages, read_whatsapp_message, send_whatsapp_message

            comms.extend([send_whatsapp_message, list_whatsapp_messages, read_whatsapp_message])
        if telegram_tools_enabled():
            from abstractcore.tools.telegram_tools import send_telegram_artifact, send_telegram_message

            comms.extend([send_telegram_message, send_telegram_artifact])

        if comms:
            toolsets["comms"] = {
                "id": "comms",
                "label": "Comms",
                "tools": comms,
            }

    return toolsets


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
