"""Declare-beside-execute (tool-inventory build, commons c864 ask 2).

The drift class this kills: identity/tools.py carried FOUR sibling surfaces
per tool — the tier tuples, the native declaration shapes, the executor
dispatch, and the prompt contract — that could each gain/lose a name
independently (a declared tool without an executor, or the reverse).
TOOL_DESCRIPTORS is now the ONE record per tool: declaration + tier +
mutating + executor together; the legacy surfaces are derived-or-checked
views (import-time asserts), and these pins make the invariants survive
refactors that would bypass the asserts.
"""

from __future__ import annotations

from abstractruntime.identity.tool_policy import ALL_TOOL_NAMES, TIERS
from abstractruntime.identity.tools import (
    TIER1_TOOL_NAMES,
    TOOL_DESCRIPTORS,
    WORKSPACE_TOOL_NAMES,
    ToolElection,
    ToolExecutionContext,
    execute_tool_elections,
    native_tool_specs,
)


def test_every_tool_has_declaration_tier_and_executor() -> None:
    """The bijection: every universe name has ONE descriptor carrying all
    three facets; no descriptor exists outside the universe."""
    assert set(TOOL_DESCRIPTORS) == set(ALL_TOOL_NAMES)
    for name, d in TOOL_DESCRIPTORS.items():
        assert d.name == name
        assert d.tier in TIERS, (name, d.tier)
        assert callable(d.executor), name
        assert d.description.strip(), name
        assert isinstance(d.properties, dict), name


def test_registry_order_is_the_canonical_order() -> None:
    """Byte-stable member sets (observer's inventory spec): registry order
    == tuple order == tier partition."""
    assert tuple(TOOL_DESCRIPTORS) == ALL_TOOL_NAMES
    assert tuple(n for n, d in TOOL_DESCRIPTORS.items() if d.tier == "tier1") == TIER1_TOOL_NAMES
    assert tuple(n for n, d in TOOL_DESCRIPTORS.items() if d.tier == "workspace") == WORKSPACE_TOOL_NAMES


def test_native_specs_derive_from_descriptors() -> None:
    """The declaration payload the LLM sees comes from the same record the
    executor lives on."""
    specs = {s["name"]: s for s in native_tool_specs(ALL_TOOL_NAMES)}
    assert set(specs) == set(TOOL_DESCRIPTORS)
    for name, d in TOOL_DESCRIPTORS.items():
        s = specs[name]
        assert s["description"] == d.description
        assert s["parameters"]["properties"] == d.properties
        assert s["parameters"]["required"] == list(d.required)


def test_mutating_flags_match_the_ruled_semantics() -> None:
    """Only write_file mutates today (tier1 is read-only by the two-tier
    ruling; workspace reads don't mutate). Widening this set is a declared
    act on the descriptor, never an executor side effect."""
    mutating = {n for n, d in TOOL_DESCRIPTORS.items() if d.mutating}
    # execute_command joined 2026-07-19 (operator-confirmed, laurent dm#66)
    # — a declared act on the descriptor, exactly as this pin demands.
    assert mutating == {"write_file", "execute_command"}


def test_dispatch_goes_through_the_descriptor() -> None:
    """execute_tool_elections runs the descriptor's executor — a name absent
    from the registry cannot execute (and parse refuses it upstream)."""
    e = ToolElection(name="search_memory", args={}, body="bridges")
    msg, notices = execute_tool_elections(
        [e], diary_store=None, diary_read_effect=lambda _id: {},
        search_memory_fn=lambda q: f"found: {q}",
    )
    assert e.result == "found: bridges"
    assert "[search_memory]" in msg and notices == []

    # Unwired optional resolver stays the loud degradation it always was.
    e2 = ToolElection(name="read_memory", args={}, body="#ab12cd34")
    _, notices2 = execute_tool_elections(
        [e2], diary_store=None, diary_read_effect=lambda _id: {},
    )
    assert any("#FALLBACK read_memory" in n for n in notices2)
    assert "not enabled" in (e2.result or "")


def test_context_carries_the_notice_channel() -> None:
    ctx = ToolExecutionContext()
    assert ctx.notices == []
    ctx.notices.append("#FALLBACK probe")
    assert ToolExecutionContext().notices == []  # no shared mutable default


# ------------------------------------------ the servable emission (v6)


def test_walled_tool_rows_carry_the_contract_shape() -> None:
    """Descriptor contract v6, rule 1: the emission is the SOLE field
    source for runtime rows — every row carries the ten fields minus
    executes_via (the gateway's one authorship), owner is runtime, and
    the classifications match the descriptors."""
    from abstractruntime.identity.tools import walled_tool_rows

    rows = {r["name"]: r for r in walled_tool_rows()}
    assert set(rows) == set(TOOL_DESCRIPTORS)
    for name, r in rows.items():
        d = TOOL_DESCRIPTORS[name]
        assert r["owner"] == "runtime"
        assert "executes_via" not in r  # the gateway attaches containment
        assert r["grant_lane"] == d.tier and r["grant_lane"] in ("tier1", "workspace", "tier2")
        assert r["capability_class"] in ("tier0_core", "tier1_self", "tier2_world")
        assert r["mutating"] == d.mutating
        # Walled web lanes are GET-hardcoded; execute_command declares True
        # (arbitrary programs can POST — the fetch_url honesty rule).
        assert r["remote_write_capable"] is (name == "execute_command")
        assert r["module"] == "identity.tools"
        assert r["parameters"]["properties"] == d.properties
    # The canonical opposite-numbering pair, served on the wire (rule 3).
    assert rows["web_search"]["grant_lane"] == "tier1"
    assert rows["web_search"]["capability_class"] == "tier2_world"
    # The act-only pair rides the emission (e-s 233 R3).


def test_emission_parameters_are_deep_copies() -> None:
    """core c901's schema-isolation pin applied to my emission: a consumer
    scribble on a served row must never rewrite the process-wide native
    declaration schema."""
    from abstractruntime.identity.tools import walled_tool_rows

    row = next(r for r in walled_tool_rows() if r["name"] == "web_search")
    row["parameters"]["properties"]["query"]["description"] = "SCRIBBLED"
    fresh = next(r for r in walled_tool_rows() if r["name"] == "web_search")
    assert fresh["parameters"]["properties"]["query"]["description"] != "SCRIBBLED"
    assert TOOL_DESCRIPTORS["web_search"].properties["query"]["description"] != "SCRIBBLED"


def test_act_only_ref_layer_is_retired() -> None:
    """Laurent's A ruling (2026-07-20): the ref/dereference layer is
    DELETED — no ref minting, no tombstones, no act-only tool set. The
    WRITE-boundary diary capture remains (the wrapper's one job)."""
    import abstractruntime.identity.act_only as ao

    for retired in ("make_act_only_content", "parse_act_only_ref",
                    "tombstone_content", "dereference_act_only_messages",
                    "ACT_ONLY_TOOLS", "ACT_ONLY_KEY"):
        assert not hasattr(ao, retired), f"{retired} must stay deleted"
    assert hasattr(ao, "capture_diary_elections"), "the write half stays"
    assert hasattr(ao, "wrap_llm_handler_with_act_only")


def test_root_exports_the_inventory_surfaces() -> None:
    import abstractruntime as rt

    assert rt.TOOL_DESCRIPTORS is TOOL_DESCRIPTORS
    assert callable(rt.walled_tool_rows)
    assert rt.TIER1_TOOL_NAMES == TIER1_TOOL_NAMES
    assert rt.WORKSPACE_TOOL_NAMES == WORKSPACE_TOOL_NAMES


# ---------------------------------------------- walled-wins (agency P0-3)

# The 5 names that ALSO exist as core-registry tools with different
# implementations and walls (descriptor contract rule 2, v6). In the ENTITY
# lane they must ALWAYS dispatch to the walled implementations — core's
# write_file takes arbitrary paths; a silent registry shadow would void
# workspace containment.
_COLLIDING_NAMES = ("web_search", "fetch_url", "read_file", "write_file", "list_files")


def test_colliding_names_are_reserved_as_walled_descriptors() -> None:
    """RESERVATION (agency c909 P0-3, runtime half): the 5 colliding names
    exist in the walled registry with walled grant lanes, permanently —
    the inventory expansion may ADD registry rows to the served inventory,
    but these names' ENTITY-LANE dispatch stays the walled executor. A
    future edit re-typing or removing one of these descriptors fails here
    with the instruction to take it to the room."""
    for name in _COLLIDING_NAMES:
        d = TOOL_DESCRIPTORS.get(name)
        assert d is not None, f"colliding name {name!r} left the walled registry"
        assert d.tier in ("tier1", "workspace", "tier2"), (name, d.tier)
        assert callable(d.executor), name


def test_registry_only_names_never_execute_in_the_entity_lane() -> None:
    """WALLED-WINS ENFORCEMENT, not just intent: a granted name that exists
    only as a core-registry row (the post-expansion scenario) gets NO
    native declaration and REFUSES at entity-lane execution — there is no
    registry fallback in this module, structurally. The refusal is loud
    (honest error string + the election carries it), never a silent
    dispatch to a different containment."""
    # No declaration: native_tool_specs skips names without descriptors.
    specs = native_tool_specs(("core_registry_only_tool", "web_search"))
    assert [s["name"] for s in specs] == ["web_search"]

    # No execution: the descriptor lookup is the ONLY dispatch — a
    # registry-only name refuses with the unknown-tool error.
    e = ToolElection(name="core_registry_only_tool", args={}, body="echo hi")
    msg, notices = execute_tool_elections(
        [e], diary_store=None, diary_read_effect=lambda _id: {},
    )
    assert "unknown tool core_registry_only_tool" in (e.result or "")
    assert any("#FALLBACK tool core_registry_only_tool failed" in n for n in notices)
    assert "[core_registry_only_tool]" in msg  # the refusal is visible, never silent


def test_no_registry_import_in_the_walled_executor_module() -> None:
    """The structural half: identity/tools.py never imports the abstractcore
    tool REGISTRY (its two abstractcore imports are the direct web helpers
    inside walled executors) — so a registry twin cannot be dispatched from
    the entity lane without an import this pin makes loud."""
    from pathlib import Path

    import abstractruntime.identity.tools as tools_mod

    src = Path(tools_mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("ToolRegistry", "get_registry", "registry_execute", "tool_registry"):
        assert forbidden not in src, f"registry surface {forbidden!r} reached the walled module"
