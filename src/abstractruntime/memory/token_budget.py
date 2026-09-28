"""Token-budget helpers for active context selection.

These utilities are intentionally best-effort and dependency-light:
- Prefer AbstractCore's TokenUtils when available (better model-aware heuristics).
- Fall back to simple character-based estimation otherwise.

They exist to keep VisualFlow workflows bounded by an explicit `max_input_tokens`
budget (ADR-0008) even when the underlying model supports much larger contexts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Metadata marker stamped on the notice message this module inserts when it
# drops history. Consumers (transcripts, UI, tests) can find the cut by key
# instead of by matching prose.
TRIM_NOTICE_KIND = "input_budget_trim"


def estimate_tokens(text: str, *, model: Optional[str] = None) -> int:
    """Best-effort token estimation for a string."""
    s = str(text or "")
    if not s:
        return 0
    try:
        from abstractcore.utils.token_utils import TokenUtils

        return int(TokenUtils.estimate_tokens(s, model))
    except Exception:
        # Conservative fallback: ~4 chars per token.
        return max(1, int(len(s) / 4))


# Flat estimate for one non-text content part (an image, a file) — AbstractCore's
# own fallback per image (`VLMTokenCalculator` base_tokens). The real cost depends
# on the model and the image size; what matters here is that it is a bounded
# per-part cost, not the length of the part's base64 payload.
MEDIA_PART_TOKEN_ESTIMATE = 512


def estimate_message_tokens(message: Dict[str, Any], *, model: Optional[str] = None) -> int:
    """Estimate tokens for a chat message dict (role+content).

    A content-part list (OpenAI-style `[{"type": "text", ...}, {"type":
    "image_url", ...}]`) counts the text of its text parts, plus
    `MEDIA_PART_TOKEN_ESTIMATE` for each other part. Before 0.7.1 the list
    was stringified, so an inline image counted as its base64 text (~50,000
    tokens for a 150 KB image) and pushed every older turn out of the window.
    """
    if not isinstance(message, dict):
        return 0
    role = str(message.get("role") or "").strip()
    raw = message.get("content")
    media_parts = 0
    if isinstance(raw, list):
        texts: List[str] = []
        for part in raw:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and "text" in str(part.get("type") or "text"):
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
            else:
                media_parts += 1
        content = "\n".join(texts)
    else:
        content = "" if raw is None else str(raw)
    # Include a small role prefix so token estimation reflects chat formatting overhead.
    text = f"{role}: {content}" if role else content
    return estimate_tokens(text, model=model) + media_parts * MEDIA_PART_TOKEN_ESTIMATE


def trim_messages_to_max_input_tokens(
    messages: Iterable[Any],
    *,
    max_input_tokens: int,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Trim oldest non-system messages until the estimated budget fits.

    Rules:
    - Preserve all system messages.
    - Preserve the most recent non-system message (typically the user prompt).
    - Drop oldest non-system messages first.
    - Tool results travel with the call that asked for them: every
      `role="tool"` message is kept or dropped together with the message
      before it (the assistant tool call, or the previous tool result of the
      same call batch). An orphan tool result — its call dropped — is a
      provider-hostile request (OpenAI-style APIs reject it), and a call
      without its results leaves the model waiting on answers it never gets.

    LOUDNESS (adversarial budget audit 2026-08-02, ADR-0026 §1): the budget
    itself is legitimate — it only runs when a caller sets a POSITIVE
    `max_input_tokens` (unset/`-1` disables it, and that is the default). What
    was not legitimate is the way it fired: whole turns simply disappeared, so
    the model was handed a conversation with a missing beginning and no way to
    know. Dropping messages for budget reasons is lossy truncation by ADR-0026's
    own list, and "no truncation may occur quietly". When this drops anything it
    now inserts a labeled notice naming the budget that caused it, and logs a
    warning attributable to this module. Returning the messages untouched
    (budget unset, or nothing dropped) inserts nothing.
    """
    try:
        budget = int(max_input_tokens)
    except Exception:
        return [dict(m) for m in messages if isinstance(m, dict)]

    if budget <= 0:
        return [dict(m) for m in messages if isinstance(m, dict)]

    typed: List[Dict[str, Any]] = [dict(m) for m in messages if isinstance(m, dict)]
    system_messages = [m for m in typed if m.get("role") == "system"]
    non_system = [m for m in typed if m.get("role") != "system"]

    if not non_system:
        return system_messages

    # Pre-compute per-message token estimates (cheaper than re-tokenizing whole text repeatedly).
    sys_tokens = sum(estimate_message_tokens(m, model=model) for m in system_messages)
    non_tokens = [estimate_message_tokens(m, model=model) for m in non_system]

    # Units kept or dropped whole: a message plus the tool results that follow it.
    units: List[List[int]] = []
    for i, m in enumerate(non_system):
        if m.get("role") == "tool" and units:
            units[-1].append(i)
        else:
            units.append([i])

    def _unit_tokens(unit: List[int]) -> int:
        return sum(non_tokens[i] for i in unit)

    # Always keep the final unit (it holds the final non-system message).
    kept_units: List[List[int]] = [units[-1]]
    total = sys_tokens + _unit_tokens(units[-1])

    # If we're already over budget, we still return system + the last unit.
    for unit in reversed(units[:-1]):
        tok = _unit_tokens(unit)
        if total + tok > budget:
            break
        kept_units.append(unit)
        total += tok

    first_kept = kept_units[-1][0]
    kept = non_system[first_kept:]

    dropped = first_kept
    if dropped <= 0:
        return system_messages + kept

    #[WARNING:TRUNCATION] caller-declared input budget dropped whole messages — never silently
    dropped_tokens = sum(non_tokens[:first_kept])
    logger.warning(
        "abstractruntime.memory.token_budget: dropped %d oldest message(s) "
        "(~%d estimated tokens) to fit max_input_tokens=%d",
        dropped,
        dropped_tokens,
        budget,
    )
    notice = {
        "role": "system",
        "content": (
            f"#TRUNCATION: {dropped} earlier message(s) (~{dropped_tokens} estimated tokens) were "
            f"dropped from this request to fit max_input_tokens={budget} "
            "(abstractruntime.memory.token_budget). The conversation above is INCOMPLETE — "
            "do not assume the missing turns never happened; ask or re-read if they matter."
        ),
        "metadata": {"kind": TRIM_NOTICE_KIND, "dropped_messages": dropped, "max_input_tokens": budget},
    }
    return system_messages + [notice] + kept

