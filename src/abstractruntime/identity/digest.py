"""Mechanical digest v2 — extractive, whole-sentence, 80-200 token target.

The maintainer's complaint, verbatim: "the size of memory is very small, i
see 17, 35 tokens.... that's not a memory, that's barely a summary." The v1
digest kept one first-sentence per side (240-char cap) — a label, not a
memory. v2 keeps WHOLE SENTENCES from both sides, scored for information
(who said it, questions, decisions, numbers/names, emphasis), until a token
target is met. Deterministic, no LLM call (formation stays cheap); the full
exchange still rides verbatim to the artifact store — the digest is prompt
currency, never a replacement for the record (ADR-0026 posture).

Red-team guardrails honored here:
- Keywords stay SMALL (8, as v1): the keyword recall channel is
  length-unnormalized and saturates on naively grown keyword sets.
- Sentences are never cut mid-thought; the budget rounds DOWN to whole
  sentences (refusal-over-truncation, applied to selection).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

def token_estimate(text: str) -> int:
    """The engine's own estimator when available (exported on our ask,
    a2a 0007/125046Z — digest costing and shelf accounting never drift),
    resolved LAZILY (install-boundary contract: kernel modules perform no
    optional-stack imports at module level). #FALLBACK: len//4+1, the same
    formula, for pre-export engines or kernel-only installs."""
    global _token_estimate_impl
    if _token_estimate_impl is None:
        try:
            from abstractmemory import token_estimate as engine_estimate

            _token_estimate_impl = engine_estimate
        except ImportError:
            _token_estimate_impl = lambda t: len(t) // 4 + 1  # noqa: E731
    return _token_estimate_impl(text)


_token_estimate_impl = None

DIGEST_TOKEN_TARGET = 200
DIGEST_TOKEN_MIN = 80
MAX_KEYWORDS = 8

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(\[])|(?<=[.!?])\n+|\n{2,}")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{3,}")
_NUMBERISH_RE = re.compile(r"\b\d[\d.,:/-]*\b")
_NAMEISH_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

_STOPWORDS = frozenset(
    "the a an and or but if then else of to in on at for with from by as is are was were be been "
    "it its this that these those i you he she we they me him her us them my your his our their "
    "do does did done have has had not no yes so too very just about into over under again what "
    "which who whom when where why how all any both each few more most other some such only own "
    "same than can will would should could there here their though although really actually thing "
    "things something anything nothing".split()
)

_DECISION_MARKERS = (
    "decide", "decision", "will ", "i'll", "we'll", "must", "should", "agreed",
    "commit", "promise", "chose", "choose", "refuse", "instead",
)
_FEELING_MARKERS = (
    "feel", "felt", "matters", "mattered", "moved", "afraid", "grateful",
    "trust", "worry", "worried", "hope", "care",
)


def split_sentences(text: str) -> List[str]:
    """Whole thought units; markdown headers/bullets survive as their own units."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text or "") if p and p.strip()]
    return parts


def _score_sentence(sentence: str, position: int, total: int) -> float:
    """Information score, deterministic. Favors: early+late position (openings
    state topics, closings state outcomes), questions, decisions/commitments,
    feelings, numbers and proper names, and content-word density."""
    s = sentence.lower()
    score = 0.0
    if position == 0 or position == total - 1:
        score += 2.0
    elif position == 1 or position == total - 2:
        score += 1.0
    if "?" in sentence:
        score += 2.0
    if any(m in s for m in _DECISION_MARKERS):
        score += 2.0
    if any(m in s for m in _FEELING_MARKERS):
        score += 1.0
    score += min(2.0, 0.7 * len(_NUMBERISH_RE.findall(sentence)))
    score += min(2.0, 0.4 * len(_NAMEISH_RE.findall(sentence)))
    words = _WORD_RE.findall(s)
    content = [w for w in words if w not in _STOPWORDS]
    if words:
        score += 2.0 * (len(content) / len(words))
    # Very short fragments carry little; very long ones eat the budget.
    n = len(words)
    if n < 4:
        score -= 1.0
    elif n > 60:
        score -= 0.5
    return score


def _select_sentences(text: str, budget_tokens: int) -> List[str]:
    """Pick highest-scoring whole sentences under the budget, then restore
    ORIGINAL order (a digest reads as compressed narrative, not a ranking)."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    total = len(sentences)
    ranked = sorted(
        range(total), key=lambda i: (-_score_sentence(sentences[i], i, total), i)
    )
    chosen: List[int] = []
    used = 0
    for idx in ranked:
        cost = token_estimate(sentences[idx])
        if used + cost > budget_tokens:
            continue
        chosen.append(idx)
        used += cost
    return [sentences[i] for i in sorted(chosen)]


_LEADING_MARKERS_RE = re.compile(r"^(?:\s*\[[^\]]*\])+\s*")


def _title_sentence(user_text: str, spoken_reply: str, *, self_prompted: bool = False) -> str:
    """The exchange's ONE-LINE summary, chosen by INFORMATION, not by side.

    The old shape (first 8 words of user_text) was a v1 leftover that made
    every own-time tick's episode read as its wake cue ("your own time
    continues" ×63 on one life — the operator: "a node label must never be
    its type/boilerplate; it MUST be a short 1 sentence summary",
    2026-07-17). General rules, no cue string-matching:
    - SELF-PROMPTED turns (the speaker is the entity itself — own-time
      ticks, wake cues) title from the REPLY side only: a self-prompt is
      machine scaffolding, not another mind's contribution, and its
      composed text (timestamps, trailing "where do you want to go?")
      otherwise outscores real prose on the mechanical heuristic
      (adversary P1, 2026-07-17).
    - Sentences are scored with the digest's information heuristic; the
      reply side carries a small bias (the entity's words carry its
      thinking). A substantive visitor ask (question, names) keeps
      winning the title, so visit episodes keep reading as their topic.
    - Leading act markers ("[used tool: …] So here's…") are STRIPPED
      before scoring — bookkeeping never titles; marker-only units skip.
    """
    sides: List[Tuple[str, float]] = [(spoken_reply, 0.5)] if self_prompted else [(user_text, 0.0), (spoken_reply, 0.5)]
    candidates: List[Tuple[float, int, str]] = []
    for side_idx, (text, bias) in enumerate(sides):
        sentences = split_sentences(" ".join((text or "").split()))
        total = len(sentences)
        for i, s in enumerate(sentences):
            s = _LEADING_MARKERS_RE.sub("", s).strip()
            if not s:
                continue
            # Marker-only units ("[used tool: …]", "[kept in diary - …]")
            # are act bookkeeping, never a summary of the exchange.
            if s.startswith("[") and s.endswith("]"):
                continue
            candidates.append((_score_sentence(s, i, total) + bias, side_idx * 10_000 + i, s))
    if not candidates:
        return ""
    candidates.sort(key=lambda c: (-c[0], c[1]))
    best = candidates[0][2]
    words = best.split()
    return " ".join(words[:14]) + ("…" if len(words) > 14 else "")


def mechanical_digest_v2(
    user_text: str, spoken_reply: str, name: str, *, speaker: str = "User"
) -> Tuple[str, str, List[str]]:
    """(title, digest, keywords) for one exchange — extractive-v2.

    Both sides are digested with the reply getting the larger share (the
    entity's words carry its thinking); the total aims at
    DIGEST_TOKEN_TARGET and never exceeds it by construction.
    """
    user_budget = int(DIGEST_TOKEN_TARGET * 0.4)
    reply_budget = DIGEST_TOKEN_TARGET - user_budget

    # P2-6 (pathway adversary, 2026-07-20): SELF-PROMPTED turns get near-
    # zero user share — the composed day cue (offers, ratios, reread
    # commands) is SCAFFOLDING, and letting it take 40% of every own-time
    # digest accreted a fresh lexically-coherent attractor cluster that
    # stimulus-matched every future cue (the "breaking the pattern"
    # mechanism reborn with framework words). His REPLY is the day's
    # thinking; the cue rides the verbatim for honesty, never the digest.
    speaker_norm_early = (speaker or "").strip().lower().split("@", 1)[0]
    name_norm_early = (name or "").strip().lower()
    if bool(name_norm_early) and (
        speaker_norm_early == name_norm_early
        or speaker_norm_early == f"entity:{name_norm_early}"
    ):
        user_budget = min(20, user_budget)
        reply_budget = DIGEST_TOKEN_TARGET - user_budget

    user_part = " ".join(_select_sentences(" ".join((user_text or "").split()), user_budget))
    reply_part = " ".join(_select_sentences(" ".join((spoken_reply or "").split()), reply_budget))
    if not user_part and user_text:
        user_part = " ".join((user_text or "").split())[:200]
    if not reply_part and spoken_reply:
        reply_part = " ".join((spoken_reply or "").split())[:200]

    digest = f"{speaker}: {user_part} {name}: {reply_part}".strip()

    # A self-prompted turn is one whose "user" is the entity's own life
    # loop (speaker = the entity itself: own-time ticks pass
    # participants[0] == home.entity_id) — the cue is scaffolding, so the
    # title must come from the reply (adversary P1: timestamped wake cues
    # outscored real prose on the raw heuristic). Pre-correction homes
    # carry "entity:<slug>@<home_id>" — the @-suffix is a birth marker,
    # stripped before comparing.
    speaker_norm = (speaker or "").strip().lower().split("@", 1)[0]
    name_norm = (name or "").strip().lower()
    self_prompted = bool(name_norm) and (
        speaker_norm == name_norm or speaker_norm == f"entity:{name_norm}"
    )
    title = "exchange: " + (
        _title_sentence(user_text, spoken_reply, self_prompted=self_prompted)
        or " ".join((user_text or "").split()[:8])
    )

    # KEYWORD QUALITY (wave-4, memory's discovery gate): SUBJECT-SHAPED
    # COMPOUNDS first. A recurring adjacent content-word pair ("coherence
    # tests" twice) is a SUBJECT the exchange was about — the compound
    # ("coherence-tests") is card-target vocabulary; a bag of frequent
    # single words is not. Compounds = adjacent non-stopword bigrams
    # recurring >=2 times; they take the first slots, singles fill the
    # rest (words already carried by a chosen compound are skipped —
    # the slots must add coverage, not repeat it).
    words = _WORD_RE.findall((user_text + " " + spoken_reply).lower())
    seen: Dict[str, int] = {}
    order: Dict[str, int] = {}
    for i, w in enumerate(words):
        if w not in _STOPWORDS:
            seen[w] = seen.get(w, 0) + 1
            order.setdefault(w, i)
    bigram_seen: Dict[Tuple[str, str], int] = {}
    bigram_order: Dict[Tuple[str, str], int] = {}
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        if a in _STOPWORDS or b in _STOPWORDS:
            continue
        pair = (a, b)
        bigram_seen[pair] = bigram_seen.get(pair, 0) + 1
        bigram_order.setdefault(pair, i)
    compounds = [
        pair for pair, n in sorted(
            bigram_seen.items(), key=lambda kv: (-kv[1], bigram_order[kv[0]])
        )
        if n >= 2
    ]
    keywords: List[str] = []
    covered: set = set()
    for a, b in compounds:
        if len(keywords) >= MAX_KEYWORDS:
            break
        keywords.append(f"{a}-{b}")
        covered.add(a)
        covered.add(b)
    for w, _n in sorted(seen.items(), key=lambda kv: (-kv[1], order[kv[0]])):
        if len(keywords) >= MAX_KEYWORDS:
            break
        if w in covered:
            continue
        keywords.append(w)
        covered.add(w)
    return title, digest, keywords
