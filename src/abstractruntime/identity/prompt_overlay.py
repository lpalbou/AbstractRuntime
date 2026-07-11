"""Operator prompt overlay — the editable layer of an entity's system prompt.

The system prompt of a summoned entity composes from layers with different
owners (maintainer ask, 2026-07-11: "a tab/badge for system prompt that we
could rewrite"):

- IDENTITY PRELUDE — rendered from the engrammed core + diary + standing.
  NEVER operator-editable prose: identity evolves by the entity's own acts
  (spark v1-for-life; hash drift refuses a summon). Rewriting it in a text
  box would be editing the entity behind its back.
- CONVERSATION CONTRACT — behavioral text (how memories arrive, how to
  diary). Code default; operator may REWRITE it per entity.
- VISIT / OWN-TIME PARAGRAPHS — life framing per phase. Code defaults;
  operator may REWRITE them per entity.
- TOOLS CONTRACT — derived from the resolved tool grant (fence syntax,
  the exact granted names). Structural: its content must match what the
  executor actually accepts, so it is never free-text editable — the tools
  tab is its editor.
- OPERATOR NOTES — a pure addendum appended last, attributed honestly
  ("from your operator"), because words in the head should not pretend to
  be the entity's own.

The overlay lives IN THE HOME (`<home>/system_prompt.yaml`) — operator
config beside spark.yaml / substrate.yaml / tool_policy.yaml. Missing file
= code defaults, byte-identical to the pre-overlay composition. Sessions
read it at summon time (snapshot-at-summon, same semantics as the tool
policy: the NEXT summon obeys an edit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "OVERLAY_FILENAME",
    "OVERLAY_KEYS",
    "read_prompt_overlay",
    "write_prompt_overlay",
]

OVERLAY_FILENAME = "system_prompt.yaml"

# key -> which built-in text it replaces (or where it lands).
# conversation: CONTRACT_PARAGRAPH (chat.py)
# visit:        VISIT_OWN_TIME_PARAGRAPH (chat.py; visit-phase sessions)
# own_time:     OWN_TIME_CONTRACT (life.py; own-time sessions)
# operator:     appended LAST as attributed standing instructions
OVERLAY_KEYS = ("conversation", "visit", "own_time", "operator")

# A rewritten layer competes with recall for the context window, and no
# budget accounting sees it (the prelude has its own refusal; the overlay
# needs one too). ~16k chars ≈ 4k tokens — an order of magnitude above the
# built-in paragraphs, far below a paste accident.
MAX_LAYER_CHARS = 16_000

_HEADER_COMMENT = """# Operator prompt overlay for this entity (system_prompt.yaml).
#
# Each key REPLACES one built-in layer of the system prompt (empty/absent
# key = the built-in default). `operator` is different: it is appended at
# the end, attributed as standing instructions from the operator.
#
#   conversation: the conversation contract (memories framing, diary offer)
#   visit:        the visit-phase life paragraph (own time continues)
#   own_time:     the own-time contract (own-time/loop sessions)
#   operator:     standing operator instructions, appended last
#
# The identity prelude and the tools contract are NOT here by design:
# identity evolves by the entity's own acts, and the tools text must match
# the actual grant (edit tools in tool_policy.yaml / the tools tab).
"""


def read_prompt_overlay(home_dir: Path) -> Dict[str, str]:
    """The operator's overlay, or {} when the home carries none.

    Unknown keys are IGNORED on read with no error (the write side refuses
    them; a hand-edited file with extras must not brick a summon). Values
    are coerced to stripped strings; empties drop out.
    """
    path = Path(home_dir) / OVERLAY_FILENAME
    if not path.exists():
        return {}
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        # A malformed file must not block a summon; the caller's notes
        # surface it (works-or-loud handled at the composition site).
        return {"#error": "unreadable"}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for key in OVERLAY_KEYS:
        raw = data.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            out[key] = text
    return out


def write_prompt_overlay(home_dir: Path, overlay: Mapping[str, Any]) -> Path:
    """Persist the operator's overlay (the UI's write shape).

    Unknown keys refuse loudly — a typo'd key silently changing nothing is
    how an operator believes a prompt changed when it did not. Empty values
    are dropped (= revert that layer to the built-in default); an overlay
    with no surviving keys DELETES the file, so absence stays the honest
    "all defaults" state.
    """
    unknown = sorted(set(map(str, (overlay or {}).keys())) - set(OVERLAY_KEYS))
    if unknown:
        raise ValueError(f"unknown overlay key(s) {unknown} (known: {list(OVERLAY_KEYS)})")
    clean: Dict[str, str] = {}
    for key in OVERLAY_KEYS:
        raw = (overlay or {}).get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if len(text) > MAX_LAYER_CHARS:
            raise ValueError(
                f"overlay layer {key!r} is {len(text)} chars (cap {MAX_LAYER_CHARS}) — "
                "a head this large starves recall of the context window"
            )
        if text:
            clean[key] = text
    path = Path(home_dir) / OVERLAY_FILENAME
    if not clean:
        path.unlink(missing_ok=True)
        return path
    import yaml

    from ..utils.atomic_files import atomic_write_text

    body = yaml.safe_dump(clean, sort_keys=True, allow_unicode=True, width=88)
    # Atomic: a torn overlay would read as malformed and silently drop the
    # operator's word to defaults (adversary P2, 2026-07-11).
    atomic_write_text(path, _HEADER_COMMENT + body)
    return path


def overlay_note(overlay: Mapping[str, str]) -> Optional[str]:
    """One observability line for session logs when an overlay is active."""
    if not overlay:
        return None
    if "#error" in overlay:
        return "#FALLBACK system_prompt.yaml unreadable; built-in prompt defaults used"
    keys = [k for k in OVERLAY_KEYS if k in overlay]
    return "operator prompt overlay active: " + ", ".join(keys) if keys else None
