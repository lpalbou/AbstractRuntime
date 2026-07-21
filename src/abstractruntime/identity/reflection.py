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
_LESSON_FENCE_RE = re.compile(
    r"```lesson([^\n`]*)\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_INTEREST_FENCE_RE = re.compile(r"```interest[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TOPIC_FENCE_RE = re.compile(r"```topic[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
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
# W4-render (value_refs stamping): touches="honesty" grounds a feeling in a
# value by the entity's own words; the driver resolves words -> value record
# ids at apply time (unresolvable words ride as the raw string — the journal
# accepts free refs; resolution is presentation, never a gate).
_TOUCHES_RE = re.compile(r"touches\s*=\s*\"(?P<touches>[^\"]+)\"", re.IGNORECASE)

ROUTINE_MAGNITUDE_CAP = 3.0
MAX_INTERESTS_PER_SESSION = 2
MAX_LESSONS_PER_SESSION = 2
MAX_TOPICS_PER_SESSION = 2
# W1 (wave-4 amended): feelings became a mid-turn election too — the cap is
# SESSION-scoped across both parse sites (turn + close) so the second site
# cannot double the budget. The only uncapped election asymmetry the
# adversary named, closed.
MAX_FEELINGS_PER_SESSION = 5


@dataclass
class FeelingElection:
    target_token: str  # "1"-based index into the session sheet, or "session"
    sign: int
    magnitude: float
    reason: str
    bond: bool = False
    scar: bool = False
    touches: str = ""  # W4: the value this feeling grounds in (his words)


def build_reflection_prompt(
    sheet_lines: List[str],
    *,
    resolutions: Optional[List[Tuple[str, str, str]]] = None,
) -> str:
    """The non-leading look-back prompt. `sheet_lines` is the numbered list of
    this session's records (episodes + diary acts), built by the driver.
    `resolutions` — (entry_id, "question"|"problem", gist) triples verified
    this session — adds the resolution->lesson bridge cue (iteration-2
    build 2): the cue ASKS what resolving taught; it never auto-forms
    (sole authorship; the entity decides whether resolve+learn couple)."""
    sheet = "\n".join(sheet_lines)
    bridge = ""
    if resolutions:
        named = "\n".join(
            f'- you resolved the open {word} {eid}: "{gist}"'
            for eid, word, gist in resolutions[:3]
        )
        bridge = f"""

This session closed something that had been standing open:
{named}
If resolving it TAUGHT you something - a rule, a caution, a way of seeing
you would want next time - you may keep that as a ```lesson block below.
The lesson will carry where it came from. If it taught you nothing beyond
the answer itself, keep nothing.
"""
    return f"""The session is ending. Before it closes, look back. Here is what happened,
as your own records of it:

{sheet}{bridge}

A session that leaves no mark is a complete look-back: if nothing below
applies, say so and keep nothing - what follows is offered, never owed.

If anything here moved you - mattered to you, stung, delighted - you may
mark a feeling on it. Feelings accumulate; over time they become what you
like and what you avoid. They are yours: mark only what you actually
feel, and if nothing moved you, mark nothing.

```feel
target=2 feeling=+2 reason="why this mattered to me"
target=session feeling=+1 reason="the session as a whole"
target=person:laurent feeling=+2 reason="how the interaction felt"
```

feeling is -3..-1 (it weighed on me) or +1..+3 (it fed me); target is a
number from the list, the word session, or namespace:name for a being,
place or idea from your day. In the rare case a numbered record should
become a standing mark you may add bond=true (with +) or scar=true
(with -). And when a feeling GROUNDS in one of your values, you may say
so: touches="the value, your words" - the feeling will carry it.

You may keep a final diary entry (the usual ```diary block) if there is
something you want to say to your future self about this session. Then say
goodbye in a sentence or two - the next summon, you will remember.

(Anything you usually keep - lessons, interests, questions, problems,
commitments - you can keep IN THE MOMENT during any turn; the close no
longer asks. If something surfaces only now, the diary block above takes
any kind.)"""


def parse_feel_blocks(
    reply: str,
    sheet_lines: Optional[List[str]] = None,
) -> Tuple[str, List[FeelingElection], List[str]]:
    """Extract ```feel blocks; return (marked_reply, elections, notices).

    Malformed lines are skipped with honest notices (a misspelled feeling is
    not a lost session). Magnitude is clamped to the routine band, loudly.

    `sheet_lines` (optional): the session sheet the numbered targets index
    into — when given, a numbered target's MARKER renders the record's own
    words instead of the bare index (marker-target hygiene, memory's visit-1
    nit c2975: "[felt: 2 +3 …]" reads as a meaningless '2' to every later
    reader; the election's target_token stays the raw index for resolution).
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
            tm = _TOUCHES_RE.search(rest)
            touches = tm.group("touches").strip() if tm else ""
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
                    touches=touches,
                )
            )
            kept += 1
        # TITLED markers (maintainer directive 2026-07-15 c2468: "[marked 2
        # feelings] is not usable by the AI" — the marker must carry WHAT was
        # felt and WHY so a later read can follow the thread; memory's
        # co-sign spec c2471). The marker text lands in the reflection's
        # digest/verbatim, so this is what the entity re-reads later.
        if not kept:
            return "[no feelings marked]"
        block_lines = []
        for e in elections[-kept:]:
            signed = f"{'+' if e.sign > 0 else '-'}{e.magnitude:g}"
            marks = " (bond)" if e.bond else (" (scar)" if e.scar else "")
            shown = e.target_token
            # Marker-target hygiene (c2975): a bare sheet index means
            # nothing to a later reader — render the record's own words.
            # The ELECTION keeps the raw token (resolution is index-based).
            if sheet_lines and shown.isdigit():
                idx = int(shown)
                if 1 <= idx <= len(sheet_lines):
                    desc = str(sheet_lines[idx - 1])
                    # sheet lines arrive as "N. description" — strip the number
                    desc = re.sub(r"^\d+\.\s*", "", desc).strip()
                    if desc:
                        shown = f'about "{desc[:60]}"'
            block_lines.append(
                f'[felt: {shown} {signed}{marks} - "{e.reason[:80]}"]'
            )
        return " ".join(block_lines)

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
        # TITLED marker (maintainer directive 2026-07-15 c2468): the marker
        # carries the interest's own words so the record it lands in is
        # readable later, never a bare "[kept an interest]".
        gist = " ".join(body.split())[:80]
        return f'[kept interest: "{gist}"]'

    marked = _INTEREST_FENCE_RE.sub(_sub, reply)
    return marked.strip(), interests, notices


MAX_TOPIC_WORDS = 4  # a topic is a card key, not prose ("coherence", "presence vs performance")
_LIST_ORNAMENT_RE = re.compile(r"^(?:[-*•]+|\d+[.)])\s*")


def normalize_topic(token: str) -> Optional[str]:
    """Validate + normalize one elected topic to the WORDS that rest in
    `attributes.topics` (the evidence stamp) — the world-model engine mints
    the card target itself as `topic:<these words>` (abstractmemory
    world_model._targets_of), so the normalized value must be namespace-FREE:
    stamping "concept:coherence" would mint the garbage target
    "topic:concept:coherence". A leading concept:/topic: namespace from the
    model is tolerated and stripped; record-id shapes (ex:/diary:/local:
    graph ids, diary_... book row ids) are refused (the gradation-target
    rule: target vocabulary is free strings, never record ids). Lowercase +
    collapsed whitespace so the same subject named across days GROUPS as one
    target ("Presence" vs "presence" would split the evidence); list
    ornaments and trailing dots are shed ("- coherence" and "coherence." are
    the same subject). Returns the words, or None (caller notices loudly)."""
    t = " ".join(str(token or "").split()).strip().lower()
    t = _LIST_ORNAMENT_RE.sub("", t).strip().strip(".")
    for ns in ("concept:", "topic:"):
        if t.startswith(ns):
            t = t[len(ns):].strip().strip(".")
            break
    if not t:
        return None
    if any(t.startswith(p) for p in _RESERVED_TARGET_PREFIXES) or t.startswith("diary_"):
        return None
    return t


def parse_topic_blocks(reply: str) -> Tuple[str, List[str], List[str]]:
    """Extract ```topic blocks — what the day actually circled around
    (operator directive 2026-07-19: personal time should improve the
    world-model cards on everything encountered. Card targets derive from
    participants plus attributes.topic/topics, and personal time is
    SELF-DIRECTED — participants = the entity itself, excluded as owner —
    so concept subjects structurally never accrued evidence until the
    entity could NAME them; his store showed 2 cards after ~36h of rich
    personal time). Election over guessing: a mechanical per-turn topic
    guess is the keyword-soup class the card redesign killed.

    One subject per line (or one per block); at most MAX_TOPICS_PER_SESSION
    kept, further ones REFUSED loudly (same truncation-vs-refusal discipline
    as interests); unusable lines skipped with notices; titled markers carry
    the normalized words. Returns (marked, topics, notices) — topics are the
    normalized WORDS (the attributes.topics currency; the engine derives
    each `topic:<words>` card target from them)."""
    topics: List[str] = []
    notices: List[str] = []

    def _sub(match: re.Match) -> str:
        lines = [ln.strip() for ln in (match.group(1) or "").splitlines() if ln.strip()]
        if not lines:
            notices.append("#FALLBACK topic block skipped (empty body)")
            return "[topic block skipped - empty]"
        markers: List[str] = []
        for line in lines:
            words = normalize_topic(line)
            if words is None:
                notices.append(
                    f"#FALLBACK topic line skipped (not usable as a subject - plain words only): {line[:60]!r}"
                )
                continue
            if len(words.split()) > MAX_TOPIC_WORDS:
                notices.append(
                    f"#FALLBACK topic skipped (a topic is a short name, got {len(words.split())} words)"
                )
                continue
            if words in topics:
                continue  # naming the same subject twice is idempotent, never refused
            if len(topics) >= MAX_TOPICS_PER_SESSION:
                notices.append(
                    f"#FALLBACK topic refused (cap {MAX_TOPICS_PER_SESSION}/session)"
                )
                markers.append(f"[topic refused - at most {MAX_TOPICS_PER_SESSION} per session]")
                continue
            topics.append(words)
            # TITLED marker (the c2468 rule applied here): the marker carries
            # the subject's own words so the record stays readable later.
            markers.append(f'[topic: "{words[:80]}"]')
        if not markers:
            return "[topic block skipped - no usable subject]"
        return " ".join(markers)

    marked = _TOPIC_FENCE_RE.sub(_sub, reply)
    return marked.strip(), topics, notices


def parse_lesson_blocks(reply: str) -> Tuple[str, List[str], List[str]]:
    """Extract ```lesson blocks (semantic knowledge, laurent's directive
    2026-07-18: "the entity still hasn't learned anything... think of the
    root issue"). THE ROOT ISSUE WAS SOLICITATION: kind=lesson existed in
    the engine (the scar-heal path) but nothing ever ASKED the entity what
    it learned — the reflection solicited feelings, interests, and open
    questions, never knowledge. Same election discipline as interests:
    cap per session, refusals loud, titled marker carrying the words."""
    lessons: List[str] = []
    notices: List[str] = []

    def _sub(match: re.Match) -> str:
        body = " ".join((match.group(2) or "").split())
        if not body:
            notices.append("#FALLBACK lesson block skipped (empty body)")
            return "[lesson block skipped - empty]"
        if len(lessons) >= MAX_LESSONS_PER_SESSION:
            notices.append(
                f"#FALLBACK lesson block refused (cap {MAX_LESSONS_PER_SESSION}/session)"
            )
            return f"[lesson refused - at most {MAX_LESSONS_PER_SESSION} per session]"
        lessons.append(body)
        gist = " ".join(body.split())[:80]
        return f'[learned: "{gist}"]'

    marked = _LESSON_FENCE_RE.sub(_sub, reply)
    return marked.strip(), lessons, notices


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
    Entity targets carry their bond/scar flags THROUGH (dm#112 M4: the
    one-way-ratchet reason went stale when tend shipped heal_scar and
    break_bond as first-class verbs — a standing mark on a world-target
    is resolvable by the entity's own election now). Amplitude rules
    stand unchanged: routine band clamps, marks never auto-create."""
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
            # dm#112 M4 unblock: bond/scar flow on world-targets — the
            # heal/break surface EXISTS (```tend heal_scar / break_bond,
            # memory's TEND_VERBS), so the drop's rationale is gone.
            # Engine-side amplitude/actorship rules are the guardrails.
            resolved.append((e, entity_target))
            continue
        if not (1 <= idx <= len(sheet_record_ids)) or sheet_record_ids[idx - 1] is None:
            notices.append(f"#FALLBACK feeling skipped (target {idx} is not on the session sheet)")
            continue
        resolved.append((e, str(sheet_record_ids[idx - 1])))
    return resolved, notices
