"""The core-inventory facade (gateway c924 ask: the backlog-0059 boundary
routes core access through runtime — this is the thin conduit for core's
registry tool rows; pass-through, never a copy)."""

from __future__ import annotations

import pytest

pytest.importorskip("abstractcore")

from abstractruntime.integrations.abstractcore.tool_inventory_facade import (  # noqa: E402
    core_inventory_schema_version,
    core_registry_tool_rows,
)


def test_pass_through_of_cores_enumeration() -> None:
    """Rows arrive VERBATIM from core's enumeration — same count, same
    field sets, same names in the same order (rule 1: the facade adds,
    drops, retypes nothing)."""
    from abstractcore.tools.inventory import builtin_tool_inventory_as_dicts

    ours = core_registry_tool_rows()
    theirs = builtin_tool_inventory_as_dicts()
    assert [r["name"] for r in ours] == [r["name"] for r in theirs]
    assert all(set(a) == set(b) for a, b in zip(ours, theirs))
    assert all(r["owner"] == "core" for r in ours)
    # The colliding names arrive with CORE's shapes (two-containment truth:
    # the serving layer renders these beside runtime's walled twins).
    names = {r["name"] for r in ours}
    assert {"web_search", "fetch_url"} <= names


def test_rows_are_isolated_copies() -> None:
    """Core's schema-isolation pin holds THROUGH the facade: scribbling on
    a served row never poisons the next call."""
    row = next(r for r in core_registry_tool_rows() if r["name"] == "web_search")
    row["description"] = "SCRIBBLED"
    params = row.get("parameters")
    if isinstance(params, dict):
        params["SCRIBBLE"] = True
    fresh = next(r for r in core_registry_tool_rows() if r["name"] == "web_search")
    assert fresh["description"] != "SCRIBBLED"
    assert "SCRIBBLE" not in (fresh.get("parameters") or {})


def test_schema_version_is_served() -> None:
    assert core_inventory_schema_version() >= 1
