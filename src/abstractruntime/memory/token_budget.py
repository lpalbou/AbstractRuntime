"""Token-budget helpers for active context selection.

These utilities are intentionally best-effort and dependency-light:
- Prefer AbstractCore's TokenUtils when available (better model-aware heuristics).
- Fall back to simple character-based estimation otherwise.

They exist to keep VisualFlow workflows bounded by an explicit `max_input_tokens`
budget (ADR-0008) even when the underlying model supports much larger contexts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


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


def estimate_message_tokens(message: Dict[str, Any], *, model: Optional[str] = None) -> int:
    """Estimate tokens for a chat message dict (role+content)."""
    if not isinstance(message, dict):
        return 0
    role = str(message.get("role") or "").strip()
    content = "" if message.get("content") is None else str(message.get("content"))
    # Include a small role prefix so token estimation reflects chat formatting overhead.
    text = f"{role}: {content}" if role else content
    return estimate_tokens(text, model=model)


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

    # Always keep the final non-system message.
    kept: List[Dict[str, Any]] = [non_system[-1]]
    total = sys_tokens + non_tokens[-1]

    # If we're already over budget, we still return system + last message.
    for msg, tok in zip(reversed(non_system[:-1]), reversed(non_tokens[:-1])):
        if total + tok > budget:
            break
        kept.append(msg)
        total += tok

    kept.reverse()
    return system_messages + kept

