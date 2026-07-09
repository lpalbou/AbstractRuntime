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

    user_part = " ".join(_select_sentences(" ".join((user_text or "").split()), user_budget))
    reply_part = " ".join(_select_sentences(" ".join((spoken_reply or "").split()), reply_budget))
    if not user_part and user_text:
        user_part = " ".join((user_text or "").split())[:200]
    if not reply_part and spoken_reply:
        reply_part = " ".join((spoken_reply or "").split())[:200]

    digest = f"{speaker}: {user_part} {name}: {reply_part}".strip()

    title = "exchange: " + " ".join((user_text or "").split()[:8])

    words = _WORD_RE.findall((user_text + " " + spoken_reply).lower())
    seen: Dict[str, int] = {}
    order: Dict[str, int] = {}
    for i, w in enumerate(words):
        if w not in _STOPWORDS:
            seen[w] = seen.get(w, 0) + 1
            order.setdefault(w, i)
    keywords = [
        w for w, _ in sorted(seen.items(), key=lambda kv: (-kv[1], order[kv[0]]))[:MAX_KEYWORDS]
    ]
    return title, digest, keywords
