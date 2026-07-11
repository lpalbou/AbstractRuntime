"""Per-phase tool policy — which tools an entity holds in each mode of life.

The maintainer's two-tier ruling (a2a 0007, restated 2026-07-08): tier-1
cognition tools (read-only — the six TIER1_TOOL_NAMES in tools.py:
web_search, fetch_url, diary_list, diary_read, read_memory, search_memory)
are safe for a persistent agent 24/7; the workspace tools write inside the
home's walls. Grants are per PHASE of life — the FOUR ruled keys
(laurent's single-human-words ruling, c786 2026-07-11 20:30 — "like for
us human: visit/work/personal/sleep"):

- "visit"    — turns driven by a verified human/operator visitor
- "work"     — pursuing operator-GIVEN tasks; ticks until done, then sleeps
- "personal" — no given tasks: the entity's own time, self-directed
               exploration (the life loop). OFF BY DEFAULT at the door
               (personal_grant — gateway lane); the TOOL default here is
               the full set per laurent's Q1 ruling (c684): the brake is
               the grant, never handlessness.
- "sleep"    — consolidation/dream window. FIRST-CLASS and operator-
               widenable (laurent c653); no EXECUTION path consumes this
               grant yet — display endpoints resolve it for the matrix;
               the dream pass honors it when it grows tool use.

DEFINITION (semantics c794, the one adjacency named in both files):
`personal_grant` = the OPERATOR'S AUTHORIZATION for the personal phase's
loop (may-it-run: mode disabled|timer|until_revoked — a config-object
section, gateway lane, never a key in this file); distinct from the
ToolGrant this module resolves (which tools a session holds once a phase
RUNS). The brake is the personal_grant, the hands are the ToolGrant.

LEGACY SPELLINGS (migration window, dies before release — the lease-shim
policy): "tasked"→work and "own_time"→personal were the pre-ruling
consensus names; "resident" was own_time's own predecessor and maps
transitively to personal. All resolve paths accept them LOUDLY (arg +
file section map to the ruled key with a #FALLBACK note; writes normalize
keys on disk) so no home file or caller is bricked mid-migration and an
operator's narrow legacy grant is NEVER silently widened to the ruled
default (adversary F7, gateway-verified).

The policy lives IN THE HOME (`<home>/tool_policy.yaml`) — operator config
beside spark.yaml, never a code constant. Missing file = the defaults
below — THE MAINTAINER'S RULED DEFAULTS (12:37 matrix ruling + Q1 c684):
visit, work and personal hold the FULL set (an entity has its hands by
default; narrowing is the operator's explicit act, via the matrix or this
file); sleep holds the read-only exploration set MINUS the diary — his
rationale verbatim: "the entity can't act/change the environment while
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
    personal:
      tools: [diary_list, diary_read, read_memory, list_files, read_file]
    sleep:
      tools: [web_search, fetch_url, read_memory, search_memory, read_file, list_files]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tools import TIER1_TOOL_NAMES, WORKSPACE_TOOL_NAMES

__all__ = [
    "ALL_TOOL_NAMES",
    "LEGACY_PHASE_ALIASES",
    "PHASES",
    "PHASE_PERSONAL",
    "PHASE_SLEEP",
    "PHASE_VISIT",
    "PHASE_WORK",
    "POLICY_FILENAME",
    "SLEEP_DEFAULT_TOOL_NAMES",
    "TIERS",
    "ToolGrant",
    "canonical_phase",
    "read_policy_file",
    "resolve_tool_grant",
    "write_policy_file",
]

# The four ruled phase keys (laurent c786: single human words, "like for
# us human: visit/work/personal/sleep") — named constants so callers never
# re-type the strings (the two-vocabularies drift that motivated F7).
PHASE_VISIT = "visit"
PHASE_WORK = "work"
PHASE_PERSONAL = "personal"
PHASE_SLEEP = "sleep"
PHASES = (PHASE_VISIT, PHASE_WORK, PHASE_PERSONAL, PHASE_SLEEP)

# Migration-window aliases (old spelling -> ruled key). DIES BEFORE RELEASE:
# once gateway's literals flip and existing home files have been written
# once (writes normalize keys), this map empties and the old spellings
# become unknown again — the removal flips the alias tests to expects-raise.
# History of the spellings: "resident" (pre-consensus) -> "own_time"
# (consensus c607) -> "personal" (laurent's human-words ruling c786);
# "tasked" (c607) -> "work" (c786). All map to the RULED keys directly —
# transitive chains resolve in one hop.
LEGACY_PHASE_ALIASES: Dict[str, str] = {
    "resident": PHASE_PERSONAL,
    "own_time": PHASE_PERSONAL,
    "tasked": PHASE_WORK,
}
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
    # MAINTAINER'S RULED DEFAULTS (12:37 matrix ruling; the personal phase
    # confirmed full-set by Q1, c684): visit, work and personal hold the FULL set —
    # an entity has its hands by default; narrowing is the operator's
    # explicit act (matrix / policy file). The historical per-session
    # `enable_workspace` gate no longer SUBTRACTS from the default visit
    # grant (the kwarg stays accepted: the policy file and explicit grants
    # are the authority, and callers still pass it).
    if phase == PHASE_SLEEP:
        return SLEEP_DEFAULT_TOOL_NAMES
    return TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES


def _normalize_phase(phase: str, notes: Optional[List[str]] = None, *, context: str) -> str:
    """One gate for phase spellings: ruled keys pass, legacy aliases map
    LOUDLY (note appended when a notes list is given), anything else raises
    — refusing code owns the set."""
    # Lowercase on the ARG axis so the two entry points agree (adversary
    # find 9: ChatSession lowercases, the direct resolver did not — a
    # "Resident" arg aliased through one door and raised through the
    # other). FILE keys stay exact-match: YAML is case-sensitive.
    p = str(phase or "").strip().lower()
    if p in PHASES:
        return p
    if p in LEGACY_PHASE_ALIASES:
        target = LEGACY_PHASE_ALIASES[p]
        if notes is not None:
            notes.append(
                f"#FALLBACK legacy phase spelling {p!r} ({context}) maps to "
                f"{target!r}; the alias dies before release"
            )
        return target
    raise ValueError(
        f"unknown phase {p!r} (known: {list(PHASES)}; "
        f"legacy aliases: {sorted(LEGACY_PHASE_ALIASES)})"
    )


def canonical_phase(phase: str) -> str:
    """The ruled spelling for `phase`, or a raise for unknowns — the PUBLIC
    normalizer for consumers that persist or SIGN phase strings (semantics
    c700 V5: entity-stamp-v2 must sign the canonical phase only; an alias
    inside the MAC basis is a verify-time chain-break). One gate, no second
    copy of the alias map."""
    return _normalize_phase(phase, context="canonical_phase")


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
    anyone deciding that. Legacy phase spellings (LEGACY_PHASE_ALIASES) are
    accepted with a warning and NORMALIZED ON WRITE: the file always lands
    with the ruled keys, so home files converge to the new vocabulary
    without an operator act (N7 migration contract, c672)."""
    import yaml

    existing = read_policy_file(home_dir)
    merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}

    # Warnings are collected and emitted only after the payload validates:
    # a write that raises on an unknown tool name must not have already
    # claimed a normalization that never landed (adversary find 8).
    pending_warnings: List[str] = []

    # Migrate legacy keys already AT REST in the file (e.g. a pre-rename
    # `resident:` section) so any write converges the whole file. An
    # explicit ruled key wins over its legacy twin when both are present.
    for legacy, target in LEGACY_PHASE_ALIASES.items():
        if legacy in merged:
            legacy_spec = merged.pop(legacy)
            if target in merged:
                # Accurate attribution (adversary F2): the surviving section
                # may be the file's own ruled key OR an earlier-resolved
                # legacy twin that already migrated — name the mechanism,
                # not a "ruled key" the file may never have carried.
                pending_warnings.append(
                    f"tool_policy.yaml carries {legacy!r} but {target!r} is already "
                    f"resolved (from the file or an earlier legacy key); "
                    f"{legacy!r} is dropped"
                )
            else:
                merged[target] = legacy_spec
                pending_warnings.append(
                    f"#FALLBACK tool_policy.yaml key {legacy!r} normalized to "
                    f"{target!r} on write; the alias dies before release"
                )

    # The ruled-key-wins precedence must hold WITHIN one payload too
    # (adversary find 1): {"personal": [...], "own_time": [...]} in either
    # order must land the explicit ruled key's word, never the legacy
    # twin's — insertion order must not decide.
    ruled_in_payload = {str(p) for p in (policy or {}) if str(p) in PHASES}
    for phase, tools in (policy or {}).items():
        canonical = _normalize_phase(phase, context="write_policy_file")
        if canonical != phase:
            if canonical in ruled_in_payload:
                pending_warnings.append(
                    f"write names both {phase!r} and {canonical!r}; the ruled key "
                    f"{canonical!r} wins and the legacy-spelled entry is ignored"
                )
                continue
            pending_warnings.append(
                f"#FALLBACK legacy phase spelling {phase!r} written as "
                f"{canonical!r}; the alias dies before release"
            )
        if tools is None:
            merged.pop(canonical, None)  # revert this phase to the defaults
            continue
        names = [str(t).strip() for t in (tools or []) if str(t).strip()]
        unknown = sorted(set(names) - set(ALL_TOOL_NAMES))
        if unknown:
            raise ValueError(f"unknown tool name(s) {unknown} (known: {list(ALL_TOOL_NAMES)})")
        merged[canonical] = {"tools": names}

    for msg in pending_warnings:
        warnings.warn(msg, stacklevel=2)
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

    Missing file → THE RULED DEFAULTS (12:37 matrix ruling + Q1 c684:
    visit + work + personal = the full set, sleep = read-only
    exploration minus the diary). File present → the file is the
    operator's word for the phases it NAMES: `tools` (exact list) wins
    over `tiers`+`add`−`remove`; unnamed phases follow the defaults.
    Unknown names are dropped WITH a note, never silently.

    Legacy spellings map loudly on BOTH axes (migration window): a caller
    passing phase="own_time" (or "resident"/"tasked") resolves the ruled
    key's grant with a #FALLBACK note, and a policy FILE still carrying a
    legacy section is honored under the ruled key with a note naming the
    file — the operator's
    narrow grant survives the rename, never a silent widen (F7)."""
    notes: List[str] = []
    phase = _normalize_phase(phase, notes, context="resolve_tool_grant arg")
    try:
        raw = read_policy_file(home_dir)
    except Exception as e:  # noqa: BLE001 - a broken file must not block a summon
        return ToolGrant(
            tools=_default_tools(phase, enable_workspace=enable_workspace),
            source="default",
            notes=tuple(notes)
            + (f"#FALLBACK tool_policy.yaml unreadable ({e}); phase {phase!r} runs on defaults",),
        )

    # FILE-SECTION SHIM: the ruled key wins; a legacy-keyed section (an
    # operator's pre-rename word, e.g. `resident:`) is honored for its
    # target phase with a note naming the file. Also surface any unknown
    # keys the file carries — a typo'd section silently granting nothing
    # is the same class of quiet loss.
    section_key: Optional[str] = None
    if isinstance(raw, dict):
        # The ruled key wins WHEN WELL-FORMED; a malformed ruled section
        # (e.g. `personal:` left null by a half-finished hand edit) must
        # not shadow an intact narrow legacy section — that would widen
        # the operator's last intact word to the full default (adversary
        # find 4, the F7 class again).
        # Candidate keys for this phase, precedence order: the RULED key
        # first, then its legacy spellings in map order. The first INTACT
        # (mapping-shaped) candidate wins; every malformed candidate that
        # appears in the file is NAMED (find 4: a null section from a
        # half-finished hand edit must never silently shadow or vanish —
        # the class applies to every spelling of the phase).
        candidates = [phase] + [
            legacy for legacy, target in LEGACY_PHASE_ALIASES.items() if target == phase
        ]
        for key in candidates:
            if key not in raw:
                continue
            if not isinstance(raw.get(key), dict):
                notes.append(
                    f"#FALLBACK tool_policy.yaml names phase {key!r} but its spec is "
                    "not a mapping; ignored"
                )
                continue
            if section_key is None:
                section_key = key
                if key != phase:
                    notes.append(
                        f"#FALLBACK tool_policy.yaml at {Path(home_dir) / POLICY_FILENAME} "
                        f"names legacy phase {key!r}; honored as {phase!r} — "
                        "rewrite the key (any policy save normalizes it)"
                    )
            else:
                # An INTACT losing twin is named too (adversary F1: the
                # read path was silent where the write path warns — the
                # operator's first signal that a shadowed section will be
                # dropped must not be the save that drops it).
                notes.append(
                    f"#FALLBACK tool_policy.yaml also names {key!r}; shadowed by "
                    f"{section_key!r} (a policy save will drop it)"
                )
        known_keys = set(PHASES) | set(LEGACY_PHASE_ALIASES)
        unknown_keys = sorted(str(k) for k in raw if k not in known_keys)
        if unknown_keys:
            notes.append(
                f"#FALLBACK tool_policy.yaml names unknown phase key(s) {unknown_keys}; ignored"
            )

    if raw is None or section_key is None:
        # No intact section for this phase in any spelling → the ruled
        # defaults apply. Loudness is already handled per candidate above
        # (a named-but-malformed section noted "not a mapping; ignored" —
        # under full-set defaults silence would be a silent WIDEN); the
        # candidates loop only ever assigns section_key to an INTACT
        # mapping, so no malformed-section re-check is needed here
        # (adversary F3: the old third clause was unreachable dead code).
        return ToolGrant(
            tools=_default_tools(phase, enable_workspace=enable_workspace),
            source="default",
            notes=tuple(notes),
        )
    spec = raw[section_key]
    known = set(ALL_TOOL_NAMES)

    def _names(value: Any) -> List[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(spec.get("tools"), (list, tuple)):
        wanted = _names(spec.get("tools"))
    else:
        if "tools" in spec:
            # A scalar `tools: diary_list` silently resolving to deny-all
            # is "the entity loses its hands without anyone deciding"
            # (adversary find 7) — fall through to tiers, but say so.
            notes.append(
                f"#FALLBACK tool_policy.yaml phase {section_key!r} has a non-list "
                "`tools` value; ignored (use a YAML list)"
            )
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
