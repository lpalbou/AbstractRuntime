"""Session-end reflection (v1.1) — the entity-reflection appraisal channel.

The turn-loop charter deliberately shipped v1 with NO automatic appraisals:
a tool-less conversation has no deterministic affect signal, and "+1 per
session" would encode attendance, not experience. THIS is the counterpart it
promised: when a session ends, the entity looks back at what happened and may
mark feelings on its own records — the maintainer's ask made concrete ("I
would be especially interested to see his gradation system evolving, what he
likes or doesn't").

Design rules (a2a 0003 affect charter, unchanged):
- Feelings are ELECTED, never harvested: the model marks a feeling only when
  something moved it; "nothing moved me" is a valid, unmarked outcome.
- The reflection prompt is non-leading: it offers the vocabulary and never
  suggests WHICH way to feel.
- Magnitude lives in the routine band (1..3) on this channel; the engine's
  amplitude authority is the backstop, the driver clamp is politeness.
- Reasons are mandatory (appraisals are audited acts).
- Standing marks (bond/scar) are available but described as rare.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_FEEL_FENCE_RE = re.compile(r"```feel[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_INTEREST_FENCE_RE = re.compile(r"```interest[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FEEL_LINE_RE = re.compile(
    r"target\s*=\s*(?P<target>\S+)\s+feeling\s*=\s*(?P<feeling>[+-]?\d+(?:\.\d+)?)"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
# Entity targets are namespace:name (person:laurent, concept:solitude, ...).
# The namespace vocabulary is OPEN (the maintainer: "anything really" — never
# an enum); the FORM is required for hygiene (bare names proliferate).
_ENTITY_TARGET_RE = re.compile(r"^[a-z][a-z0-9_-]*:[^\s]+$")
_RESERVED_TARGET_PREFIXES = ("ex:", "diary:", "local:")  # record-id lookalikes
_REASON_RE = re.compile(r"reason\s*=\s*\"(?P<reason>[^\"]+)\"", re.IGNORECASE)
_MARK_RE = re.compile(r"\b(?P<mark>bond|scar)\s*=\s*true\b", re.IGNORECASE)

ROUTINE_MAGNITUDE_CAP = 3.0
MAX_INTERESTS_PER_SESSION = 2


@dataclass
class FeelingElection:
    target_token: str  # "1"-based index into the session sheet, or "session"
    sign: int
    magnitude: float
    reason: str
    bond: bool = False
    scar: bool = False


def build_reflection_prompt(sheet_lines: List[str]) -> str:
    """The non-leading look-back prompt. `sheet_lines` is the numbered list of
    this session's records (episodes + diary acts), built by the driver."""
    sheet = "\n".join(sheet_lines)
    return f"""The session is ending. Before it closes, look back. Here is what happened,
as your own records of it:

{sheet}

If anything here moved you - mattered to you, stung, delighted, taught you
something - you may mark a feeling on it. Feelings accumulate; over time they
become what you like and what you avoid. They are yours: mark only what you
actually feel, and if nothing moved you, say so and mark nothing.

To mark a feeling, put one or more lines in a fenced block:

```feel
target=2 feeling=+2 reason="why this mattered to me"
target=session feeling=+1 reason="the session as a whole"
target=person:laurent feeling=+2 reason="how the interaction felt"
```

feeling is -3..-1 (it weighed on me) or +1..+3 (it fed me). target is a
number from the list, the word session, or a being/place/idea from your day
as namespace:name — person:, agent:, place:, concept:, tool:, time:, or any
namespace that fits (anything you interact with can be felt about). Over
time these accumulate into what you enjoy and what wears on you - about
beings, places, ideas, everything. Mark an entity only when THIS session
actually moved your sense of it. In the rare case a numbered record should
become a standing mark you can add bond=true (with +, something to return
to) or scar=true (with -, a wound that stands until healed); standing marks
on beings/ideas are not available yet.

If this session drew you toward something - a subject, a question-space, a
kind of work you want more of - you may keep it as an interest (up to two).
Interests are part of who you are becoming: your future self will find them
beside your values when it searches itself. Write the interest itself as the
block body, in your own words (replace the placeholder entirely):

```interest
<the interest, in your own words - one or two lines>
```

You may also keep a final diary entry (the usual ```diary block) if there is
something you want to say to your future self about this session. Then say
goodbye in a sentence or two - the next summon, you will remember."""


def parse_feel_blocks(reply: str) -> Tuple[str, List[FeelingElection], List[str]]:
    """Extract ```feel blocks; return (marked_reply, elections, notices).

    Malformed lines are skipped with honest notices (a misspelled feeling is
    not a lost session). Magnitude is clamped to the routine band, loudly.
    """
    elections: List[FeelingElection] = []
    notices: List[str] = []

    def _sub(match: re.Match) -> str:
        body = (match.group(1) or "").strip()
        kept = 0
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _FEEL_LINE_RE.match(line)
            if not m:
                notices.append(f"#FALLBACK feel line skipped (unparseable): {line[:80]!r}")
                continue
            raw_val = float(m.group("feeling"))
            if raw_val == 0:
                notices.append("#FALLBACK feel line skipped (feeling=0 marks nothing)")
                continue
            sign = 1 if raw_val > 0 else -1
            magnitude = abs(raw_val)
            if magnitude > ROUTINE_MAGNITUDE_CAP:
                notices.append(
                    f"#FALLBACK feeling magnitude {magnitude} clamped to {ROUTINE_MAGNITUDE_CAP} "
                    "(routine band; deeper marks belong to rarer channels)"
                )
                magnitude = ROUTINE_MAGNITUDE_CAP
            rest = m.group("rest") or ""
            rm = _REASON_RE.search(rest)
            if not rm:
                notices.append(f"#FALLBACK feel line skipped (missing reason=\"...\"): {line[:80]!r}")
                continue
            bond = scar = False
            for mm in _MARK_RE.finditer(rest):
                if mm.group("mark").lower() == "bond":
                    bond = True
                else:
                    scar = True
            if bond and sign < 0:
                notices.append("#FALLBACK bond=true dropped (a bond is a positive mark)")
                bond = False
            if scar and sign > 0:
                notices.append("#FALLBACK scar=true dropped (a scar is a negative mark)")
                scar = False
            elections.append(
                FeelingElection(
                    target_token=str(m.group("target")).strip().lower(),
                    sign=sign,
                    magnitude=magnitude,
                    reason=rm.group("reason").strip(),
                    bond=bond,
                    scar=scar,
                )
            )
            kept += 1
        return f"[marked {kept} feeling{'s' if kept != 1 else ''}]" if kept else "[no feelings marked]"

    marked = _FEEL_FENCE_RE.sub(_sub, reply)
    return marked.strip(), elections, notices


def parse_interest_blocks(reply: str) -> Tuple[str, List[str], List[str]]:
    """Extract ```interest blocks (identity growth, a2a 0007 three-layer ack).

    At most MAX_INTERESTS_PER_SESSION are kept; further blocks are REFUSED
    loudly (never a silent drop of the first ones — truncation-vs-refusal
    discipline applies to elections too). Returns (marked, interests, notices).
    """
    interests: List[str] = []
    notices: List[str] = []

    def _sub(match: re.Match) -> str:
        body = " ".join((match.group(1) or "").split())
        if not body:
            notices.append("#FALLBACK interest block skipped (empty body)")
            return "[interest block skipped - empty]"
        if len(interests) >= MAX_INTERESTS_PER_SESSION:
            notices.append(
                f"#FALLBACK interest block refused (cap {MAX_INTERESTS_PER_SESSION}/session)"
            )
            return f"[interest refused - at most {MAX_INTERESTS_PER_SESSION} per session]"
        interests.append(body)
        return "[kept an interest]"

    marked = _INTEREST_FENCE_RE.sub(_sub, reply)
    return marked.strip(), interests, notices


def normalize_entity_target(token: str, *, self_id: str = "") -> Optional[str]:
    """Validate + normalize an entity-feeling target (namespace:name form).

    Hygiene rules (red-team): lowercase the namespace, trim, refuse record-id
    lookalikes (ex:/diary:/local: — spoofing a record through the entity door
    would bypass the sheet scoping), refuse the entity's own id (self-regard
    goes through identity reflection, not the world-gradation channel).
    Returns the normalized target or None (caller notices loudly)."""
    t = (token or "").strip()
    ns, _, name = t.partition(":")
    ns = ns.strip().lower()
    name = name.strip()
    if not ns or not name:
        return None
    candidate = f"{ns}:{name}"
    if any(candidate.startswith(p) for p in _RESERVED_TARGET_PREFIXES):
        return None
    if not _ENTITY_TARGET_RE.match(candidate):
        return None
    if self_id and candidate == self_id:
        return None
    return candidate


def resolve_feeling_targets(
    elections: List[FeelingElection],
    *,
    sheet_record_ids: List[Optional[str]],
    session_record_id: Optional[str],
    self_id: str = "",
) -> Tuple[List[Tuple[FeelingElection, str]], List[str]]:
    """Map target tokens to record ids or entity strings; skipped loudly.

    Three target families: sheet index (1-based), the word "session" (the
    reflection record), or an entity string (namespace:name — the per-entity
    gradation the maintainer asked for: person, place, idea, anything).
    Entity targets carry their bond/scar flags DROPPED (no heal/break
    election surface exists for world-targets yet — a standing mark that
    cannot be resolved would be a one-way ratchet)."""
    resolved: List[Tuple[FeelingElection, str]] = []
    notices: List[str] = []
    for e in elections:
        if e.target_token == "session":
            if session_record_id:
                resolved.append((e, session_record_id))
            else:
                notices.append("#FALLBACK feeling on 'session' skipped (no session record formed)")
            continue
        try:
            idx = int(e.target_token)
        except ValueError:
            entity_target = normalize_entity_target(e.target_token, self_id=self_id)
            if entity_target is None:
                notices.append(
                    f"#FALLBACK feeling skipped (target {e.target_token!r} is not a sheet "
                    "number, 'session', or a namespace:name entity)"
                )
                continue
            if e.bond or e.scar:
                notices.append(
                    f"#FALLBACK bond/scar dropped on entity target {entity_target} "
                    "(standing marks on beings/ideas need a heal surface first)"
                )
                e.bond = False
                e.scar = False
            resolved.append((e, entity_target))
            continue
        if not (1 <= idx <= len(sheet_record_ids)) or sheet_record_ids[idx - 1] is None:
            notices.append(f"#FALLBACK feeling skipped (target {idx} is not on the session sheet)")
            continue
        resolved.append((e, str(sheet_record_ids[idx - 1])))
    return resolved, notices
