"""Per-phase tool policy — which tools an entity holds in each mode of life.

The maintainer's two-tier ruling (a2a 0007, restated 2026-07-08): tier-1
cognition tools (read-only — the six TIER1_TOOL_NAMES in tools.py:
web_search, fetch_url, diary_list, diary_read, read_memory, search_memory)
are safe for a persistent agent 24/7; the workspace tools write inside the
home's walls. Grants are per PHASE of life:

- "visit"    — a summoned conversation (chat driver, gateway chat drawer)
- "resident" — the entity's own time (the 24/7 life loop)
- "sleep"    — consolidation passes (no EXECUTION path consumes this grant
               today — display endpoints resolve it for the matrix; the
               dream pass honors it when it grows tool use)

The policy lives IN THE HOME (`<home>/tool_policy.yaml`) — operator config
beside spark.yaml, never a code constant. Missing file = the defaults
below — THE MAINTAINER'S RULED DEFAULTS (2026-07-11 12:37, set from the
workspace matrix): visit and resident hold the FULL set (an entity has its
hands by default; narrowing is the operator's explicit act, via the matrix
or this file); sleep holds the read-only exploration set MINUS the diary —
his rationale verbatim: "the entity can't act/change the environment while
sleeping, but it can recall or search information; it won't be in its
diary, but i believe it will be somewhere in the runtime ledger...
(unconscious)". The sleeping mind explores (web, memory, its own files)
but does not act (no writes) and does not open its elected/conscious
surface (no diary tools).

File shape (either style per phase; `tools` wins when both are present —
the sleep example shows the RULED DEFAULT spelled out; `tools: []` would
be an explicit ZERO grant, not the default):

    visit:
      tiers: [tier1]              # tier names
      add: [write_file]           # extra tool names on top of the tiers
      remove: [web_search]        # denied tool names (final word)
    resident:
      tools: [diary_list, diary_read, read_memory, list_files, read_file]
    sleep:
      tools: [web_search, fetch_url, read_memory, search_memory, read_file, list_files]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tools import TIER1_TOOL_NAMES, WORKSPACE_TOOL_NAMES

__all__ = [
    "ALL_TOOL_NAMES",
    "PHASES",
    "POLICY_FILENAME",
    "SLEEP_DEFAULT_TOOL_NAMES",
    "TIERS",
    "ToolGrant",
    "read_policy_file",
    "resolve_tool_grant",
    "write_policy_file",
]

PHASES = ("visit", "resident", "sleep")
TIERS: Dict[str, Tuple[str, ...]] = {
    "tier1": TIER1_TOOL_NAMES,
    "workspace": WORKSPACE_TOOL_NAMES,
}
ALL_TOOL_NAMES: Tuple[str, ...] = TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES
POLICY_FILENAME = "tool_policy.yaml"

# The sleep default (maintainer ruling 2026-07-11): read-only exploration
# without the diary — recall/search + web + own-file READS; no writes (a
# sleeping entity does not change the environment), no diary tools (the
# elected/conscious surface stays closed; sleep exploration lives in the
# ledger — the unconscious — unless the sleep pass forms records from it).
# Canonical ALL_TOOL_NAMES order.
#
# NOTE for the future sleep-pass tool build: read_file/list_files are
# WORKSPACE_TOOL_NAMES members, so this grant reports workspace_enabled=True
# even though write_file is absent — copying the visit pattern
# (WorkspaceRoot when workspace_enabled) is CORRECT here: reads work,
# write_file refuses at parse because it is not granted. Do not "fix" the
# coincidence by gating the workspace off.
#
# MIGRATION NOTE (defaults change, 2026-07-11): "the file wins" is
# PER-PHASE — a home with a partial policy file (say only `sleep:`) gains
# the new defaults on every phase the file does not name. Ruled behavior:
# unnamed phases follow the framework default as it evolves.
SLEEP_DEFAULT_TOOL_NAMES: Tuple[str, ...] = (
    "web_search", "fetch_url", "read_memory", "search_memory", "read_file", "list_files",
)


@dataclass(frozen=True)
class ToolGrant:
    """The resolved answer for one phase: which tools, and why."""

    tools: Tuple[str, ...]
    source: str  # "default" | "policy-file"
    notes: Tuple[str, ...] = ()

    @property
    def workspace_enabled(self) -> bool:
        return any(t in self.tools for t in WORKSPACE_TOOL_NAMES)


def _default_tools(phase: str, *, enable_workspace: bool) -> Tuple[str, ...]:
    # MAINTAINER'S RULED DEFAULTS (2026-07-11 12:37): visit and resident
    # hold the FULL set — an entity has its hands by default; narrowing is
    # the operator's explicit act (matrix / policy file). The historical
    # per-session `enable_workspace` gate no longer SUBTRACTS from the
    # default visit grant (the kwarg stays accepted: the policy file and
    # explicit grants are the authority, and callers still pass it).
    if phase == "sleep":
        return SLEEP_DEFAULT_TOOL_NAMES
    return TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES


def read_policy_file(home_dir: Path) -> Optional[Dict[str, Any]]:
    """The raw policy mapping, or None when the home carries none.
    A malformed file reads as None WITH the parse error left to the caller's
    resolve (works-or-loud: resolve_tool_grant notes it)."""
    path = Path(home_dir) / POLICY_FILENAME
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def write_policy_file(home_dir: Path, policy: Dict[str, Optional[Sequence[str]]]) -> Path:
    """Persist per-phase tool lists (the UI's write shape) by MERGING into
    the existing file: only phases NAMED in `policy` change; a phase mapped
    to None DELETES its entry (revert to the evolving framework defaults);
    a file left with no phases is removed (absence = defaults, honestly).

    WHY MERGE (adversary find, 2026-07-11): the old whole-document replace
    meant any matrix save wrote back the RESOLVED state of every phase —
    materializing the day's defaults as "the operator's word" forever.
    Live consequence: every real home carried a frozen `sleep: []` from
    pre-ruling saves, so the ruled sleep default was dead on all of them.

    Unknown phases and unknown tool names refuse loudly — a policy that
    silently grants nothing is how an entity loses its hands without
    anyone deciding that."""
    import yaml

    existing = read_policy_file(home_dir)
    merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    for phase, tools in (policy or {}).items():
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r} (known: {list(PHASES)})")
        if tools is None:
            merged.pop(phase, None)  # revert this phase to the defaults
            continue
        names = [str(t).strip() for t in (tools or []) if str(t).strip()]
        unknown = sorted(set(names) - set(ALL_TOOL_NAMES))
        if unknown:
            raise ValueError(f"unknown tool name(s) {unknown} (known: {list(ALL_TOOL_NAMES)})")
        merged[phase] = {"tools": names}
    path = Path(home_dir) / POLICY_FILENAME
    if not merged:
        path.unlink(missing_ok=True)
        return path
    from ..utils.atomic_files import atomic_write_text

    # Atomic: a crash mid-write must never leave a torn policy that
    # silently resolves as "no operator word" (adversary P2, 2026-07-11).
    atomic_write_text(path, yaml.safe_dump(merged, sort_keys=True))
    return path


def resolve_tool_grant(
    home_dir: Path, phase: str, *, enable_workspace: bool = False
) -> ToolGrant:
    """The tools an entity holds in `phase`, per the home's policy file.

    Missing file → THE RULED DEFAULTS (maintainer 2026-07-11: visit +
    resident = the full set, sleep = read-only exploration minus the
    diary). File present → the file is the operator's word for the phases
    it NAMES: `tools` (exact list) wins over `tiers`+`add`−`remove`;
    unnamed phases follow the defaults. Unknown names are dropped WITH a
    note, never silently."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r} (known: {list(PHASES)})")
    notes: List[str] = []
    try:
        raw = read_policy_file(home_dir)
    except Exception as e:  # noqa: BLE001 - a broken file must not block a summon
        return ToolGrant(
            tools=_default_tools(phase, enable_workspace=enable_workspace),
            source="default",
            notes=(f"#FALLBACK tool_policy.yaml unreadable ({e}); phase {phase!r} runs on defaults",),
        )
    if raw is None or phase not in raw or not isinstance(raw.get(phase), dict):
        # A phase the file NAMES but malforms (e.g. `visit:` left null in a
        # hand edit) falls to defaults LOUDLY — under the ruled full-set
        # defaults, silence here would be a silent WIDEN, not a narrow.
        if raw is not None and phase in raw:
            notes.append(
                f"#FALLBACK tool_policy.yaml names phase {phase!r} but its spec is not a "
                "mapping; the ruled defaults apply"
            )
        return ToolGrant(
            tools=_default_tools(phase, enable_workspace=enable_workspace),
            source="default",
            notes=tuple(notes),
        )
    spec = raw[phase]
    known = set(ALL_TOOL_NAMES)

    def _names(value: Any) -> List[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(spec.get("tools"), (list, tuple)):
        wanted = _names(spec.get("tools"))
    else:
        wanted = []
        for tier in _names(spec.get("tiers")):
            if tier in TIERS:
                wanted.extend(TIERS[tier])
            else:
                notes.append(f"#FALLBACK unknown tier {tier!r} in tool_policy.yaml ignored")
        wanted.extend(_names(spec.get("add")))
        removed = set(_names(spec.get("remove")))
        wanted = [t for t in wanted if t not in removed]
    unknown = sorted({t for t in wanted if t not in known})
    if unknown:
        notes.append(f"#FALLBACK unknown tool name(s) {unknown} in tool_policy.yaml ignored")
    ordered = tuple(t for t in ALL_TOOL_NAMES if t in set(wanted))  # canonical order, deduped
    return ToolGrant(tools=ordered, source="policy-file", notes=tuple(notes))
