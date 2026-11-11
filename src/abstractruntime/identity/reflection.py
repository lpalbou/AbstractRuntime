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
_REALIZE_FENCE_RE = re.compile(r"```realize[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_RESOLVE_FENCE_RE = re.compile(r"```resolve[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
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
# Identity-pass spine (cti#399 design, adopted #400; gate ruled satisfied
# c4779): realizations are PROPOSALS about the self — identity-grade, so the
# cap sits at the interests level, not the feelings level.
MAX_REALIZATIONS_PER_SESSION = 2

# Receipts (the markers substituted into the entity's own reply) carry the
# elected words WHOLE (root ADR 0026/0054, 2026-11-08): the marked reply is
# stored as the turn's verbatim, so the old marked 60/80-character
# `_receipt_gist` head was still a stored cut.


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
    moment: str = "session",
    open_questions: Optional[List[Tuple[str, Optional[str]]]] = None,
) -> str:
    """The non-leading look-back prompt. `sheet_lines` is the numbered list of
    this session's records (episodes + diary acts), built by the driver.
    `resolutions` — (entry_id, "question"|"problem", gist) triples verified
    this session — adds the resolution->lesson bridge cue (iteration-2
    build 2): the cue ASKS what resolving taught; it never auto-forms
    (sole authorship; the entity decides whether resolve+learn couple).

    `moment` is "session" (a visit close, an own-time day close) or "tick"
    (the end of one personal-time tick): the same offers, framed for what
    is ending.

    THE OFFERS ARE BACK (operator 2026-11-08: "the end visit is the way to
    reflect on the discussion!!!! that's the root cause issue!!! and also
    at every tick in personal time, the same thing should happen!"): the
    2026-07-21 shrink removed the interest/lesson/question/problem offers
    from the look-back and moved them to a mid-turn offer that produced
    zero records in 17 homes since. The look-back offers them again; the
    parsers are the same ones the turn uses.

    `open_questions` — (entry_id, title or None while it is being written) of the open questions this session asked
    or touched (the visit lane passes them; root backlog 1206, operator 2026-11-11 02:00: a question the visit
    answered stayed open) — lists them and offers a ```resolve block naming the one the session answered. None or
    empty: no list, no offer (the chat lane's prompt, byte-unchanged)."""
    if moment not in ("session", "tick"):
        raise ValueError(f"build_reflection_prompt: unknown moment {moment!r} (session|tick)")
    sheet = "\n".join(sheet_lines)
    bridge = ""
    if resolutions:
        named = "\n".join(
            f'- you resolved the open {word} {eid}: "{gist}"'
            for eid, word, gist in resolutions[:3]
        )
        bridge = f"""

This {"session" if moment == "session" else "stretch"} closed something that had been standing open:
{named}
If resolving it TAUGHT you something - a rule, a caution, a way of seeing
you would want next time - you may keep that as a ```lesson block below.
The lesson will carry where it came from. If it taught you nothing beyond
the answer itself, keep nothing.
"""
    asked = ""
    if open_questions:
        listed = "\n".join(
            f'- {eid}: "{title}"' if title else f"- {eid}: (its title is still being written)"
            for eid, title in open_questions
        )
        span = "session" if moment == "session" else "stretch"
        asked = ("\n" if bridge else "\n\n") + f"""These questions of yours stand open, and this {span} asked or touched them:
{listed}
If this {span} ANSWERED one of them - someone told you, or you found it -
you may mark it resolved: it leaves your open questions and joins your
history. Name it by its id, the numbered moment above that answered it,
then the answer in your own words:

```resolve
resolves=<the question's id, as listed>
evidence=<the number of the moment that answered it>
<the answer, in your own words - one or two lines>
```

A question this {span} did not answer stays open; mark nothing for it.
"""
    if moment == "tick":
        opening = """This stretch of your own time is done. Before the next one, look back at it.
Here is what happened, as your own records of it:"""
        closing = """You may keep a diary entry (the usual ```diary block) if there is something
you want to say to your future self about this stretch. A sentence or two
is enough; your own time continues after it."""
        what = "this stretch"
    else:
        opening = """The session is ending. Before it closes, look back. Here is what happened,
as your own records of it:"""
        closing = """You may keep a final diary entry (the usual ```diary block) if there is
something you want to say to your future self about this session. Then say
goodbye in a sentence or two - the next summon, you will remember."""
        what = "this session"
    return f"""{opening}

{sheet}{bridge}{asked}

Look at what {what} held and what it leaves alive in you. If nothing below
applies, say so and keep nothing.

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

If {what} drew you toward something - a subject, a question-space, a
kind of work you want more of - you may keep it as an interest (up to two).
Interests are part of who you are becoming: your future self will find them
beside your values when it searches itself. Start the block with a title
line - a short, complete phrase that names it, so a stranger (or you, later,
searching) knows at once what it is about - then the interest itself, in
your own words (replace the placeholders entirely):

```interest
title: <a short, complete name for the interest>
<the interest, in your own words - one or two lines>
```

If {what} taught you something you would act on differently next time -
a resolution to a problem, a better way, a trap to avoid - you may keep it
as a lesson (up to two), the same way - a title line, then the lesson in
your own words:

```lesson
title: <a short, complete name for the lesson>
<the lesson, in your own words - one or two lines>
```

If {what} left something OPEN - a question you want to pursue, or a
problem that needs repair - you may keep it so it stands on your desk until
you settle it:

```diary kind=question
gist: <the question as one short, complete line>
<the question, in your own words>
```

(kind=problem for something wrong that needs repair.)

{closing}"""


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
                notices.append(f"#FALLBACK feel line skipped (unparseable): {line!r}")
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
                notices.append(f"#FALLBACK feel line skipped (missing reason=\"...\"): {line!r}")
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
                        shown = f'about "{desc}"'
            # The record's words and his reason, WHOLE (root ADR 0026/0054,
            # 2026-11-08): the marked reply is stored as the turn's verbatim,
            # and a marked 60/80-character cut there was still a stored cut.
            block_lines.append(f'[felt: {shown} {signed}{marks} - "{e.reason}"]')
        return " ".join(block_lines)

    marked = _FEEL_FENCE_RE.sub(_sub, reply)
    return marked.strip(), elections, notices


def _room_for(cap: int, room: Optional[int]) -> int:
    return cap if room is None else max(0, min(cap, int(room)))


# An elected block's explicit title (operator 2026-11-08: "the title of any lesson, question, interest, world model
# must be explicit and easy to read / understand, both for a human and for an agent"): a first line `title: ...`.
_TITLE_LINE_RE = re.compile(r"^title\s*:\s*(?P<title>.*)$", re.IGNORECASE)


def _block_lines(raw: str) -> str:
    """A block body as the parsers keep it: whitespace folded within each line, blank lines dropped, his line breaks
    kept (the title checks read lines: a second `title:` line or a late one is a failure, never a guess). A one-line
    body comes back as that line."""
    return "\n".join(" ".join(ln.split()) for ln in str(raw or "").splitlines() if ln.strip())


def parse_elected_title(elected: str) -> Tuple[Optional[str], str, Optional[str]]:
    """(title, words, reason) of one elected interest, lesson or realization (operator ruling 2026-11-09: no
    fallback; a failure is recorded, see `title_failures`).

    The title is the block's explicit `title:` FIRST line, verbatim. Anything else gives title None and a reason:
    `missing` (no `title:` line), `empty` (the label names nothing), `multi_line` (more than one `title:` line),
    `malformed` (the `title:` line is not the first line), `too_long` (past TITLE_MAX_CHARS). No text heuristic
    stands in for it: his first line or first sentence is never read as a title. `words` is the body, whitespace
    folded, WHOLE (root ADR 0054): everything but a good title line; with no usable title, every line (a `title:`
    label dropped, its words kept). A plain one-line string (a pending election of an older visit run) reads too."""
    from .title_failures import TITLE_MAX_CHARS

    lines = [" ".join(ln.split()) for ln in str(elected or "").splitlines() if ln.strip()]
    if not lines:
        return None, "", None
    matches = [_TITLE_LINE_RE.match(ln) for ln in lines]
    titled = [i for i, m in enumerate(matches) if m]

    def _all_words(skip_first: bool = False) -> str:
        kept = []
        for i, ln in enumerate(lines):
            if skip_first and i == 0:
                continue
            m = matches[i]
            kept.append(m.group("title").strip() if m else ln)
        return " ".join(" ".join(kept).split())

    if not titled:
        return None, _all_words(), "missing"
    if len(titled) > 1:
        return None, _all_words(), "multi_line"
    if titled[0] != 0:
        return None, _all_words(), "malformed"
    title = matches[0].group("title").strip()
    if not title:
        return None, _all_words(skip_first=True), "empty"
    if len(title) > TITLE_MAX_CHARS:
        return None, _all_words(), "too_long"
    return title, (_all_words(skip_first=True) or title), None


def elected_title_and_words(elected: str) -> Tuple[Optional[str], str]:
    """(title, words) of one elected block: `parse_elected_title` without the reason. The title is None when the
    block carries no usable `title:` first line (never a guess from his words)."""
    title, words, _reason = parse_elected_title(elected)
    return title, words


def interest_marker(body: str) -> str:
    """The marker a kept ```interest block leaves in the MARKED reply (the stored verbatim; ADR 0054: whole)."""
    return f'[kept interest: "{elected_title_and_words(body)[1]}"]'


def lesson_marker(body: str) -> str:
    """The marker a kept ```lesson block leaves in the MARKED reply (the stored verbatim; ADR 0054: whole)."""
    return f'[learned: "{elected_title_and_words(body)[1]}"]'


def realization_marker(text: str) -> str:
    """The marker a held ```realize block leaves in the MARKED reply (`RealizeElection.text`, whole)."""
    return f'[realized: "{text}" - held for sleep]'


def without_markers(marked: str, markers: List[str]) -> str:
    """The SERVED reply: the marked reply without the markers whose act is served as a structured turn notice
    (`turn.notices`; the Entity app owner's contract 2026-11-09). Each marker is the exact string this runtime put
    there (the builders above, the diary marker), removed once; a marker alone on its line takes the line (and one
    of the blank lines around it) with it. Nothing else of his words changes. The STORED reply (episode verbatim,
    the session's own history) keeps every marker: they are his words at rest (ADR 0054)."""
    out = str(marked or "")
    for m in markers:
        if not m:
            continue
        lines = out.split("\n")
        hit = next((i for i, ln in enumerate(lines) if ln.strip() == m), None)
        if hit is not None:
            del lines[hit]
            # "A\n\n[m]\n\nB" reads "A\n\nB": drop one of the two blank lines the marker sat between.
            if 0 < hit < len(lines) and not lines[hit - 1].strip() and not lines[hit].strip():
                del lines[hit]
            out = "\n".join(lines)
            continue
        i = out.find(m)
        if i < 0:
            continue
        before, after = out[:i], out[i + len(m):]
        if before.endswith(" ") and after.startswith(" "):
            after = after[1:]
        out = before + after
    return out.strip()


def parse_interest_blocks(
    reply: str, *, room: Optional[int] = None,
) -> Tuple[str, List[str], List[str]]:
    """Extract ```interest blocks (identity growth, a2a 0007 three-layer ack).

    At most MAX_INTERESTS_PER_SESSION are kept - or `room`, what is left of
    the session's cap when the caller counts across a whole day or visit;
    further blocks are REFUSED loudly (never a silent drop of the first ones
    — truncation-vs-refusal discipline applies to elections too). A refused
    block's marker SAYS it was refused and keeps his words whole (review
    R2-2, 2026-11-08: a refused interest used to rest as "[kept interest: ...]"
    once the day's cap was spent). Returns (marked, interests, notices).
    """
    interests: List[str] = []
    notices: List[str] = []
    limit = _room_for(MAX_INTERESTS_PER_SESSION, room)

    def _sub(match: re.Match) -> str:
        body = _block_lines(match.group(1))
        _title, words = elected_title_and_words(body)
        if not words:
            notices.append("#FALLBACK interest block skipped (empty body)")
            return "[interest block skipped - empty]"
        if len(interests) >= limit:
            notices.append(
                f"#FALLBACK interest block refused (cap {MAX_INTERESTS_PER_SESSION}/session)"
            )
            return (
                f"[interest refused, not kept - at most {MAX_INTERESTS_PER_SESSION} per session: "
                f'"{words}"]'
            )
        interests.append(body)
        # TITLED marker (maintainer directive 2026-07-15 c2468): the marker
        # carries the interest's own words so the record it lands in is
        # readable later, never a bare "[kept an interest]". WHOLE words
        # (ADR 0054, 2026-11-08): the marked reply is stored as the turn's
        # verbatim, so an 80-character cut here was a stored cut.
        return interest_marker(body)

    marked = _INTEREST_FENCE_RE.sub(_sub, reply)
    return marked.strip(), interests, notices


@dataclass
class RealizeElection:
    """One elected realization (the identity-pass spine's waking surface,
    cti#399): text = his words; evidence = the raw tokens from the mandatory
    `evidence=` line (#tags or hex-tails — the resolves=/explores= currency,
    resolved by the DRIVER at apply time); touches = the optional
    `value:<name> | trait:<name> | relationship:<ns:name>` hint."""

    text: str
    evidence: List[str]
    touches: str = ""
    title: Optional[str] = None  # his explicit `title:` first line; None when he wrote none usable
    title_reason: Optional[str] = None  # why there is no title (title_failures reasons); None when titled
    raw: str = ""  # the block as he wrote it (whole), for the failure record


_REALIZE_EVIDENCE_RE = re.compile(r"^\s*evidence\s*=\s*(?P<val>.+)$", re.IGNORECASE)
_REALIZE_TOUCHES_RE = re.compile(r"^\s*touches\s*=\s*(?P<val>.+)$", re.IGNORECASE)
# Adversary F4: an evidence line written as prose ("evidence=#a is what my
# diary said") exploded into per-word tokens — each costing a full-home tag
# scan and a notice. Evidence tokens must LOOK like evidence (#tag or a
# colon-shaped id); prose words are dropped with ONE collective notice.
MAX_EVIDENCE_TOKENS = 8


def parse_realize_blocks(reply: str) -> Tuple[str, List[RealizeElection], List[str]]:
    """Extract ```realize blocks — the waking surface of the ruled
    identity-evolution process (sleep-only enactment, entity dm#117/#124;
    design cti#399 adopted #400; build gate ruled ALREADY SATISFIED,
    framework c4779).

    Grammar (taught in the contract):
        ```realize
        <the realization, his words>
        evidence=#tag1 #tag2      (MANDATORY — the records that showed it)
        touches=value:honesty     (optional — what part of the self it bears on)
        ```

    Formation is a PROPOSAL, never an identity write: the block parses into
    a dated record the sleep pass judges under the regulated bars. EVIDENCE
    IS THE BAR'S CURRENCY — a block without it is REFUSED with a teaching
    marker (never silently kept bare: an evidence-free realization could
    not clear any bar and would rot in the queue). Same election discipline
    as interests: cap per session, refusals loud, titled markers. The
    marker says the crucial fact — held for sleep, heard NOT enacted."""
    realizations: List[RealizeElection] = []
    notices: List[str] = []

    def _sub(match: re.Match) -> str:
        raw_lines = [ln for ln in (match.group(1) or "").splitlines()]
        text_lines: List[str] = []
        evidence: List[str] = []
        touches = ""
        dropped_prose = 0
        for ln in raw_lines:
            m_ev = _REALIZE_EVIDENCE_RE.match(ln)
            if m_ev:
                for t in m_ev.group("val").split():
                    t = t.strip().rstrip(",;")
                    if not t:
                        continue
                    # Shape filter (F4): a token is evidence-shaped when it
                    # is a #tag or a colon-shaped id; prose words drop with
                    # one collective notice below.
                    if not (t.startswith("#") or ":" in t):
                        dropped_prose += 1
                        continue
                    if len(evidence) >= MAX_EVIDENCE_TOKENS:
                        dropped_prose += 1
                        continue
                    evidence.append(t)
                continue
            m_to = _REALIZE_TOUCHES_RE.match(ln)
            if m_to:
                touches = " ".join(m_to.group("val").split())
                continue
            if ln.strip():
                text_lines.append(ln.strip())
        title, body, title_reason = parse_elected_title("\n".join(text_lines))
        if not body:
            notices.append("#FALLBACK realize block skipped (no words)")
            return "[realize block skipped - empty]"
        if dropped_prose:
            notices.append(
                f"#NOTE {dropped_prose} evidence token(s) dropped (not #tag/id-shaped "
                f"or over the {MAX_EVIDENCE_TOKENS}-token cap)"
            )
        if not evidence:
            notices.append(
                "#FALLBACK realize block refused (evidence= is mandatory - name "
                "the records that showed it, by #tag)"
            )
            return "[realization refused - add evidence=#tag naming what showed it]"
        if len(realizations) >= MAX_REALIZATIONS_PER_SESSION:
            notices.append(
                f"#FALLBACK realize block refused (cap {MAX_REALIZATIONS_PER_SESSION}/session)"
            )
            return f"[realization refused - at most {MAX_REALIZATIONS_PER_SESSION} per session]"
        realizations.append(RealizeElection(
            text=body, evidence=evidence, touches=touches, title=title, title_reason=title_reason,
            raw="\n".join(ln.rstrip() for ln in raw_lines).strip("\n"),
        ))
        # The marker carries the realization WHOLE (root ADR 0026/0054,
        # 2026-11-08): the marked reply is stored as the turn's verbatim, and
        # a marked 80-character cut there was still a stored cut.
        return realization_marker(body)

    marked = _REALIZE_FENCE_RE.sub(_sub, reply)
    return marked.strip(), realizations, notices


@dataclass
class ResolveElection:
    """One elected resolution (root backlog 1206): `resolves` = the open question's entry id as he named it (the book
    checks it: memory's `resolves_status`); `evidence` = the raw tokens of his `evidence=` line (the numbered moments
    that answered it, resolved by the capture against the look-back's sheet); `text` = the answer, his words, whole.
    It is written as a diary entry (kind note) carrying `resolves=` - the same DIARY_WRITE a turn's
    ```diary resolves=<id>``` block takes."""

    resolves: str
    evidence: List[str]
    text: str


_RESOLVE_RESOLVES_RE = re.compile(r"^\s*resolves\s*=\s*(?P<val>\S+)\s*$", re.IGNORECASE)


def parse_resolve_blocks(reply: str) -> Tuple[str, List[ResolveElection], List[str]]:
    """Extract ```resolve blocks - the look-back's offer to mark an open question this session answered (root backlog
    1206). Grammar (taught by `build_reflection_prompt(open_questions=...)`):

        ```resolve
        resolves=<the question's entry id>     (mandatory - which question)
        evidence=<moment numbers>              (the moments that answered it)
        <the answer, his words>                (mandatory - the entry's words)
        ```

    The model DECLARES the resolution; the book confirms it at write time (`resolves_status`); nothing here judges his
    words. A block naming no question, or giving no answer, is REFUSED loudly with a teaching marker that keeps his
    words whole (never a silent drop). A kept block's marker is the capture's: the block is replaced by
    "[kept in diary - note]", the plain marker of the diary entry it becomes (the capture enriches it with the
    verdict and the reread command). Returns (marked, elections, notices)."""
    elections: List[ResolveElection] = []
    notices: List[str] = []

    def _sub(match: re.Match) -> str:
        resolves = ""
        evidence: List[str] = []
        text_lines: List[str] = []
        for ln in (match.group(1) or "").splitlines():
            m_res = _RESOLVE_RESOLVES_RE.match(ln)
            if m_res and not resolves:
                resolves = m_res.group("val").strip()
                continue
            m_ev = _REALIZE_EVIDENCE_RE.match(ln)
            if m_ev:
                evidence.extend(t for t in re.split(r"[\s,;]+", m_ev.group("val")) if t)
                continue
            if ln.strip():
                text_lines.append(" ".join(ln.split()))
        text = "\n".join(text_lines)
        if not resolves:
            notices.append("#FALLBACK resolve block refused (no resolves=<the question's id> line)")
            return (f'[resolution refused - name the question with resolves=<its id>: "{" ".join(text_lines)}"]'
                    if text_lines else "[resolution refused - name the question with resolves=<its id>]")
        if not text:
            notices.append(f"#FALLBACK resolve block refused (resolves={resolves} gave no answer in words)")
            return f"[resolution of {resolves} refused - say the answer in your own words]"
        elections.append(ResolveElection(resolves=resolves, evidence=evidence, text=text))
        return "[kept in diary - note]"

    marked = _RESOLVE_FENCE_RE.sub(_sub, reply)
    return marked.strip(), elections, notices


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
                    f"#FALLBACK topic line skipped (not usable as a subject - plain words only): {line!r}"
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
            # the subject's own words, whole, so the record stays readable
            # later (ADR 0054: the marked reply is stored).
            markers.append(f'[topic: "{words}"]')
        if not markers:
            return "[topic block skipped - no usable subject]"
        return " ".join(markers)

    marked = _TOPIC_FENCE_RE.sub(_sub, reply)
    return marked.strip(), topics, notices


def parse_lesson_blocks(
    reply: str, *, room: Optional[int] = None,
) -> Tuple[str, List[str], List[str]]:
    """Extract ```lesson blocks (semantic knowledge, laurent's directive
    2026-07-18: "the entity still hasn't learned anything... think of the
    root issue"). THE ROOT ISSUE WAS SOLICITATION: kind=lesson existed in
    the engine (the scar-heal path) but nothing ever ASKED the entity what
    it learned — the reflection solicited feelings, interests, and open
    questions, never knowledge. Same election discipline as interests:
    cap per session (or `room`, what the session has left), refusals loud
    and SAID in the marker with the words whole, titled marker carrying the
    words."""
    lessons: List[str] = []
    notices: List[str] = []
    limit = _room_for(MAX_LESSONS_PER_SESSION, room)

    def _sub(match: re.Match) -> str:
        body = _block_lines(match.group(2))
        _title, words = elected_title_and_words(body)
        if not words:
            notices.append("#FALLBACK lesson block skipped (empty body)")
            return "[lesson block skipped - empty]"
        if len(lessons) >= limit:
            notices.append(
                f"#FALLBACK lesson block refused (cap {MAX_LESSONS_PER_SESSION}/session)"
            )
            return (
                f"[lesson refused, not kept - at most {MAX_LESSONS_PER_SESSION} per session: "
                f'"{words}"]'
            )
        lessons.append(body)
        # The marker carries the lesson whole (ADR 0054): the marked reply
        # is the stored verbatim of the turn.
        return lesson_marker(body)

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
