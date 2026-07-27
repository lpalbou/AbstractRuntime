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


def capability_tool_facts(capability: str) -> Dict[str, Dict[str, Any]]:
    """Per-tool risk facts for a CAPABILITY plugin's tools (gateway c4899
    facade ask 1) — re-exports core's `capability_tool_facts` through the
    one import boundary (the gateway's backlog-0059 pin forbids its catalog
    importing abstractcore directly; every core surface it reads rides this
    facade module). Consumer: the discovery facts-join — without it, camera
    rows derive unvetted/destroy on discovery. Same error posture as
    core_registry_tool_rows: absent abstractcore raises an ACTIONABLE
    ImportError (the caller degrades loudly, never silently); a capability
    with no declared facts returns {} (core's accessor contract)."""
    try:
        from abstractcore.capabilities import capability_tool_facts as _core_facts
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "capability_tool_facts needs abstractcore (pip install abstractcore); "
            f"capability fact rows are unavailable: {e}"
        ) from e
    out = _core_facts(str(capability))
    return dict(out) if isinstance(out, dict) else {}


# The versioned fact->risk mapping (tool-tiers cycle-3; laurent's examples
# as PRESET DATA, core c4512 max-wins shape). Core HOSTS the canonical fn
# when its schema-v3 wave lands (import-never-copy switches this seed to
# their export); gateway+runtime OWN the content either way. Version rides
# every served row so grants pin tier-at-grant and upward moves re-ask.
RISK_MAPPING_VERSION = "runtime-seed-1"  # served when core's hosted fold is absent

_RISK_FACT_KEYS = ("destructive_capable", "comms_send", "captures_environment",
                   "standing_effect", "mutating", "remote_write_capable")


_SEED_RANK_WORDS = {1: "observe", 2: "act", 3: "outreach", 4: "destroy"}


def derive_risk_assessment(row: Dict[str, Any]) -> Dict[str, Any]:
    """The full wire assessment for one row (semantics c4589 shape, gateway
    c4592 vote): risk_tier = the band WORD (identity), risk_rank = the
    INTEGER (ordinal - ceilings compare against rank), risk_presentation =
    the render word (differs from the band only for factless rows:
    gate at destroy rank, render "unvetted" - never an overclaim)."""
    present = [k for k in _RISK_FACT_KEYS if k in row]
    try:
        from abstractcore.tools.risk_facts import derive_risk

        facts = {k: bool(row.get(k, False)) for k in _RISK_FACT_KEYS} if present else None
        return derive_risk(facts).to_dict()
    except ImportError:
        pass  # version skew: the seed governs, labeled by its version string
    rank = _seed_rank(row, present)
    word = _SEED_RANK_WORDS[rank]
    presentation = "unvetted" if not present else word
    return {"risk_tier": word, "risk_rank": rank,
            "risk_presentation": presentation,
            "risk_mapping_version": RISK_MAPPING_VERSION}


def _seed_rank(row: Dict[str, Any], present: list) -> int:
    if not present:
        return 4
    if row.get("destructive_capable"):
        return 4
    if row.get("comms_send") or row.get("captures_environment") or row.get("standing_effect"):
        return 3
    if row.get("mutating") or row.get("remote_write_capable"):
        return 2
    return 1


def derive_risk_tier(row: Dict[str, Any]) -> int:
    """Max-wins over declared facts (laurent's ladder as data) - INTEGER
    RANK view (1..4). IMPORT-NEVER-COPY: when core's hosted fold exists
    (abstractcore.tools.risk_facts.derive_risk, shipped c4577 - the ONE
    mapping gateway+runtime own the content of), this delegates to it and
    the seed below serves only version-skewed cores. Semantics identical
    by construction (both encode laurent's examples; core's adds the
    band-neutral facts which do not move the rank).

    FULLY FACTLESS rows derive 4 fail-closed - unvetted is the top of the
    ladder, never the bottom. A row that declares SOME facts reads absent
    ones as false (declared rows are vetted rows)."""
    return int(derive_risk_assessment(row)["risk_rank"])


def _active_mapping_version() -> Any:
    try:
        from abstractcore.tools.risk_facts import RISK_MAPPING_VERSION as core_version

        return core_version
    except ImportError:
        return RISK_MAPPING_VERSION


def annotate_tool_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stamp runtime-OWNED policy fields onto inventory rows (in place,
    returns the same list): `tier` + `approval_default` — the discovery
    fields laurent's accepted-tier UX needs (c4336/c4342: a thin client
    cannot honor "accept tier N = never ask" without SEEING the tier).

    This deliberately lives BESIDE the verbatim pass-throughs, not inside
    them (their contracts forbid field authorship): the verbatim functions
    stay the sole field source for their OWNERS' fields; these two fields
    are RUNTIME's authorship (the tier vocabulary + the approval fold are
    runtime authorities), applied at composition exactly like the
    gateway's `executes_via`.

    Field semantics:
    - tier: walled rows carry their DECLARED `capability_class` verbatim.
      Registry rows are `tier2_world` BY THE RULED DEFINITION (2026-07-06:
      tier = the boundary crossed, "read-only world tools are tier2" —
      every core-registry tool crosses the world boundary; this is the
      ruling applied, never a per-name heuristic).
    - approval_default: auto|ask from runtime's ONE approval fold
      (`default_approval_policy_sets` — base safe/mutating partition +
      capability-served partitions). Names in neither set default to ask
      (the executor's default-deny posture, mirrored honestly).
    """
    from .default_tools import default_approval_policy_sets

    auto, require = default_approval_policy_sets()
    for row in rows:
        if not isinstance(row, dict):
            continue
        declared = str(row.get("capability_class") or "").strip()
        row["tier"] = declared if declared else "tier2_world"
        name = str(row.get("name") or "")
        if name in auto:
            row["approval_default"] = "auto"
        elif name in require:
            row["approval_default"] = "ask"
        else:
            row["approval_default"] = "ask"  # default-deny, stated honestly
        # RISK AXIS (cycle-3 build; the boundary-echo `tier` above stays
        # frozen-deprecated - c4413 P0, gateway-accepted c4416): the
        # operator-facing severity, derived never declared.
        # WIRE SHAPE (semantics c4589, gateway vote c4592, flipped pre-
        # bounce with zero consumers): risk_tier = band WORD, risk_rank =
        # INTEGER, risk_presentation = render word (unvetted for factless
        # rows - never rendered "destroy" without declared facts).
        incoming_refiner = row.get("risk_refiner")  # core-declared, per-tool
        row.update(derive_risk_assessment(row))
        # derive_risk over FACTS never sets a refiner (it is a row property
        # core wires onto specific tools, dm#244) - preserve the incoming
        # one so the serving path keeps send_email's send_email_recipient@v1.
        if incoming_refiner:
            row["risk_refiner"] = incoming_refiner
        # GRANTABILITY IS A PLANE QUESTION (memory's synthesis, adopted
        # c4485): life-plane infrastructure carries honest severity words
        # but is NOT grantable - its exposure is channel-law. Everything
        # else is grant-governed.
        row["grantable"] = declared not in ("tier0_core", "tier1_self")
        # OUTREACH CARVE-OUT DECLARED ON THE ROW (converged-contract finding
        # 3, c5028 R4): comms/hub tools serve approval_default="auto" by the
        # 2026-02-21 ruling (unattended runs must not stall on their own
        # comms) while their risk band is outreach(3)+ — under an explicit
        # permission_mode the LADDER derives from RANK, so the server side
        # was always consistent; the served field alone misled facts-
        # trusting clients in the transitional window. The carve-out is now
        # SAID on the row: auto here is a static-fold ruling, not a
        # rank-derivation — clients deriving from rank band stay correct,
        # clients reading approval_default see why it disagrees with the
        # band.
        try:
            if row.get("approval_default") == "auto" and int(row.get("risk_rank") or 0) >= 3:
                row["approval_carveout"] = "static-fold-comms-auto-2026-02-21"
        except (TypeError, ValueError):
            pass
    return rows


def core_inventory_schema_version() -> int:
    """Core's INVENTORY_SCHEMA_VERSION, for serve-time drift pins."""
    from abstractcore.tools.inventory import INVENTORY_SCHEMA_VERSION

    return int(INVENTORY_SCHEMA_VERSION)


def permission_mode_ladder() -> Dict[str, Any]:
    """Core's served permission-mode ladder, re-exported through the one
    import boundary (gateway c5065 [ladderfacade]; backlog-0059 forbids the
    gateway importing abstractcore directly — same path as
    capability_tool_facts). Returns {modes, semantics, max_auto_rank:
    {mode: int}} — the words + copy thin clients must RENDER, never invent
    (core C2, c5036). Symbol names per core's c5069 correction (the shipped
    surface is permission_mode_*, not ln_*). ImportError posture mirrors
    core_registry_tool_rows: actionable, loud, never silent."""
    try:
        from abstractcore.tools.risk_facts import (
            PERMISSION_MODE_SEMANTICS,
            PERMISSION_MODES,
            permission_mode_max_auto_rank,
        )
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "permission_mode_ladder needs abstractcore >= the C2 ship (c5036); "
            f"the posture vocabulary is unavailable: {e}"
        ) from e
    return {
        "modes": list(PERMISSION_MODES),
        "semantics": dict(PERMISSION_MODE_SEMANTICS),
        "max_auto_rank": {m: int(permission_mode_max_auto_rank(m)) for m in PERMISSION_MODES},
    }


def permission_mode_auto_approves(
    mode: str, *, risk_rank: int, model_controlled_destination: bool
) -> bool:
    """The ONE served auto-vs-ask decision (core c5061), re-exported: rank
    ceiling + the mcd belt fused so a consumer literally cannot forget the
    belt (an mcd tool at act(2) is under the write ceiling yet ASKS below
    full-auto). Fail-closed at the source (unknown mode = read-only
    ceiling; unrankable never autos)."""
    try:
        from abstractcore.tools.risk_facts import (
            permission_mode_auto_approves as _core_decide,
        )
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "permission_mode_auto_approves needs abstractcore >= the c5061 ship; "
            f"the fused ceiling+mcd helper is unavailable: {e}"
        ) from e
    return bool(
        _core_decide(
            mode, risk_rank=risk_rank,
            model_controlled_destination=model_controlled_destination,
        )
    )
