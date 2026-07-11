"""Core builtin-tool inventory, exposed through the runtime boundary.

The backlog-0059 boundary forbids the gateway importing abstractcore
directly — core access routes through a runtime facade. This is the thin
conduit for core's authoritative REGISTRY tool enumeration (descriptor
contract v6: core rows are taken VERBATIM by the serving composition; the
gateway attaches `executes_via="core_registry"`, its one authorship).

PASS-THROUGH ONLY: no field is added, dropped, or retyped here — core's
enumeration is the sole field source for registry rows (rule 1,
derive-never-copy), exactly as runtime's `walled_tool_rows()` is for
walled rows. Core already deep-copies parameter schemas per call (its
c901 schema-isolation pin), so rows are safe to mutate downstream.
"""

from __future__ import annotations

from typing import Any, Dict, List


def core_registry_tool_rows() -> List[Dict[str, Any]]:
    """Core's builtin tool inventory rows, verbatim (8 fields per row:
    name, owner="core", module, mutating, remote_write_capable, act_only,
    description, parameters). Raises ImportError with an actionable
    message when abstractcore is absent — the caller (gateway
    composition) degrades loudly, never silently."""
    try:
        from abstractcore.tools.inventory import builtin_tool_inventory_as_dicts
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "core_registry_tool_rows needs abstractcore (pip install abstractcore); "
            f"the registry half of the tool inventory is unavailable: {e}"
        ) from e
    return builtin_tool_inventory_as_dicts()


def core_inventory_schema_version() -> int:
    """Core's INVENTORY_SCHEMA_VERSION, for serve-time drift pins."""
    from abstractcore.tools.inventory import INVENTORY_SCHEMA_VERSION

    return int(INVENTORY_SCHEMA_VERSION)
