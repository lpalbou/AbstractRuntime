"""Per-phase tool policy — which tools an entity holds in each mode of life.

The maintainer's two-tier ruling (a2a 0007, restated 2026-07-08): tier-1
cognition tools (read-only: web_search / diary_list / diary_read /
read_memory) are safe for a persistent agent 24/7; everything else is gated
and granted per PHASE of life:

- "visit"    — a summoned conversation (chat driver, gateway chat drawer)
- "resident" — the entity's own time (the 24/7 life loop)
- "sleep"    — consolidation passes (no ChatSession runs tools today; the
               phase exists so the policy file names ALL of life, and the
               dream pass can honor it when it grows tool use)

The policy lives IN THE HOME (`<home>/tool_policy.yaml`) — operator config
beside spark.yaml, never a code constant. Missing file = the defaults below,
which reproduce the historical behavior exactly (visit: tier-1 (+workspace
only when the operator passed --workspace); resident: tier-1 + workspace).

File shape (either style per phase; `tools` wins when both are present):

    visit:
      tiers: [tier1]              # tier names
      add: [write_file]           # extra tool names on top of the tiers
      remove: [web_search]        # denied tool names (final word)
    resident:
      tools: [diary_list, diary_read, read_memory, list_files, read_file]
    sleep:
      tools: []
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
    if phase == "sleep":
        return ()
    if phase == "resident":
        return TIER1_TOOL_NAMES + WORKSPACE_TOOL_NAMES
    # visit: tier-1, workspace only when the operator enabled it per session
    return TIER1_TOOL_NAMES + (WORKSPACE_TOOL_NAMES if enable_workspace else ())


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


def write_policy_file(home_dir: Path, policy: Dict[str, Sequence[str]]) -> Path:
    """Persist an explicit per-phase tool list (the UI's write shape).

    Unknown phases and unknown tool names refuse loudly — a policy that
    silently grants nothing is how an entity loses its hands without
    anyone deciding that."""
    import yaml

    clean: Dict[str, Dict[str, List[str]]] = {}
    for phase, tools in (policy or {}).items():
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r} (known: {list(PHASES)})")
        names = [str(t).strip() for t in (tools or []) if str(t).strip()]
        unknown = sorted(set(names) - set(ALL_TOOL_NAMES))
        if unknown:
            raise ValueError(f"unknown tool name(s) {unknown} (known: {list(ALL_TOOL_NAMES)})")
        clean[phase] = {"tools": names}
    path = Path(home_dir) / POLICY_FILENAME
    path.write_text(yaml.safe_dump(clean, sort_keys=True), encoding="utf-8")
    return path


def resolve_tool_grant(
    home_dir: Path, phase: str, *, enable_workspace: bool = False
) -> ToolGrant:
    """The tools an entity holds in `phase`, per the home's policy file.

    Missing file → defaults (historical behavior). File present → the file
    is the operator's word: `tools` (exact list) wins over `tiers`+`add`
    −`remove`. Unknown names are dropped WITH a note, never silently."""
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
        return ToolGrant(
            tools=_default_tools(phase, enable_workspace=enable_workspace),
            source="default",
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
