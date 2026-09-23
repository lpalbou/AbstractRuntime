"""Durable turn grounding — stamp the runtime envelope ONCE, into the transcript.

WHY THIS MODULE EXISTS (mission A, 2026-09-22)

The runtime grounds every LLM turn with a machine-owned envelope carrying the
local date/time:

    <runtime_metadata>{"display":"[2026-09-22 03:32:27]","local_datetime":"…"}</runtime_metadata>

Until now that envelope was injected at the PAYLOAD boundary, on every call, and
never stored. The durable transcript therefore held `identify yourself` while the
model was actually sent
`<runtime_metadata>{…}</runtime_metadata>\\nidentify yourself`. One turn later the
same message was re-rendered from the transcript WITHOUT the envelope, so turn
N+1's prompt diverged from turn N's at the first byte of the previous user
message — and a prefix cache (prompt cache, KV reuse, mlx-vlm APC) can restore
nothing past a divergence. Measured on the MLX native lane (4B pair, 8-turn chat,
`untracked/missionA/replay_runtime.py`): 185-388 tokens re-prefilled on EVERY
turn, growing with the conversation, where byte-stability costs tens.

The invariant this module enforces: **the bytes a user turn is SENT with are the
bytes that turn is STORED with.** An adapter that owns a durable transcript calls
`stamp_user_turn_grounding` at its reason boundary — once per message, before the
payload is built — and every later pass (`_normalize_turn_grounding` in the
runtime ledger pass and again in the LLM client pass) keeps what it finds.

Scope note: this stamps the DURABLE message. Hosts that keep their own transcript
(abstractassistant) must replay the stamped content verbatim for the invariant to
hold across turns; the gateway/runtime path does that through
`abstractruntime.session_history`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "SYNTHETIC_MESSAGE_KEY",
    "SYNTHETIC_TOOL_RESULT",
    "message_is_synthetic",
    "strip_synthetic_markers",
    "runtime_grounding_envelope",
    "message_carries_grounding_envelope",
    "transcript_carries_grounding_envelope",
    "stamp_user_turn_grounding",
    "strip_turn_grounding",
]


# ---------------------------------------------------------------------------
# CARRIERS (mission A3, 2026-09-22)
#
# A payload-boundary repair can synthesize a `user`-role message that no human
# ever wrote: abstractagent's `sanitize_transcript_messages` folds a tool result
# whose call id nothing announced into `[unpaired tool result]: …`, because a
# strict provider 400s on an orphan `tool` message. That carrier is RE-DERIVED
# from the durable transcript on every iteration of a tool loop, so anything
# written into it is written into a message that will be rebuilt without it one
# iteration later. Stamping one is therefore the worst possible place to put the
# grounding envelope, and it is exactly what happened on the operator's live run
# 081d8daa (2026-09-22 10:48): the carrier is the LAST message of the payload, so
# the grounding pass read the transcript as "chat shape" and injected the
# envelope at its head; the next iteration rebuilt that carrier WITHOUT the
# envelope and stamped the newer one instead. Consecutive prompts then diverged
# by exactly 118 chars (one envelope) at message index 13 of 21 — thousands of
# tokens before the end of the prompt, far outside the checkpoint window — and
# every iteration from the third on was COLD.
#
# The rule is STRUCTURAL, never a pattern match on the carrier's prose: the
# producer marks what it synthesized and the grounding pass skips it.
# ---------------------------------------------------------------------------

# ONE definition, in the module that also strips the marker before the wire
# (`_strip_synthetic_message_markers`, alongside the older `volatile` marker).
# Re-exported here because THIS is the public module adapters import.
from .integrations.abstractcore.llm_client import (  # noqa: E402,PLC0415
    SYNTHETIC_MESSAGE_KEY,
    SYNTHETIC_TOOL_RESULT,
    _message_is_synthetic_carrier as _is_synthetic,
    _strip_synthetic_message_markers as _strip_markers,
)


def message_is_synthetic(message: Any) -> bool:
    """True when this message was SYNTHESIZED by a payload-boundary repair.

    Such a message is not a durable turn: it is re-derived from the durable
    transcript every time a payload is built, so it can never be the anchor for
    anything that must survive to the next iteration.
    """
    return _is_synthetic(message)


def strip_synthetic_markers(messages: Any) -> Any:
    """Drop the internal marker just before the wire (providers reject extras)."""
    return _strip_markers(messages)


def _grounding_api():
    """The envelope builders live with the rest of the grounding contract.

    Imported lazily and deliberately un-hedged: the caller is an adapter that
    already depends on the AbstractCore integration, and a silent no-op here
    would restore exactly the invisible-divergence bug this module exists to
    close.
    """
    from .integrations.abstractcore.llm_client import (  # noqa: PLC0415
        _content_carries_grounding_envelope,
        _mark_grounding_prompt_injected,
        _runtime_grounding_metadata,
        _runtime_grounding_prompt_envelope,
        _strip_runtime_grounding_prefix,
    )

    return (
        _content_carries_grounding_envelope,
        _mark_grounding_prompt_injected,
        _runtime_grounding_metadata,
        _runtime_grounding_prompt_envelope,
        _strip_runtime_grounding_prefix,
    )


def runtime_grounding_envelope(grounding: Optional[Dict[str, Any]] = None) -> str:
    """The `<runtime_metadata>…</runtime_metadata>` envelope for NOW (or `grounding`)."""
    (_carries, _mark, _metadata, _envelope, _strip) = _grounding_api()
    payload = grounding if isinstance(grounding, dict) and grounding else _mark(_metadata(), True)
    return _envelope(payload)


def strip_turn_grounding(text: Any) -> str:
    """The human-authored text of a (possibly stamped) turn, envelope removed.

    For IDENTITY comparisons — "is this durable message the same turn as this
    task string?" — not for display (`history_bundle` owns that).
    """
    (_carries, _mark, _metadata, _envelope, _strip) = _grounding_api()
    if not isinstance(text, str):
        return ""
    return _strip(text)


def message_carries_grounding_envelope(message: Any) -> bool:
    """True when this message's content already begins with its own envelope."""
    (_carries, _mark, _metadata, _envelope, _strip) = _grounding_api()
    if isinstance(message, dict):
        return bool(_carries(message.get("content")))
    return bool(_carries(message))


def transcript_carries_grounding_envelope(messages: Any) -> bool:
    """True when ANY non-synthetic user message of the transcript is stamped.

    The tool-loop question is "has this turn been grounded?", and the turn's own
    message is not the last user message once a payload repair has appended a
    carrier — it is the durable task (or the operator interjection) several
    messages back. Checking only the last one made the grounding pass append a
    fresh envelope on every iteration; checking ALL of them is what "stamp once"
    actually means (mission A3).
    """
    from .integrations.abstractcore.llm_client import (  # noqa: PLC0415
        _transcript_carries_grounding_envelope,
    )

    return bool(_transcript_carries_grounding_envelope(messages))


def stamp_user_turn_grounding(
    messages: Optional[List[Dict[str, Any]]],
    *,
    grounding: Optional[Dict[str, Any]] = None,
) -> bool:
    """Stamp the grounding envelope into the LAST user message, IN PLACE, once.

    Returns True when a stamp was written (the caller's durable transcript was
    mutated), False when nothing was needed — no user message, a non-text
    content shape, or a message that already carries its own envelope.

    Idempotent by construction: a message that already carries a head envelope is
    left byte-for-byte alone, so calling this on every iteration of a tool loop
    stamps the turn exactly once, at the iteration that first sends it.

    CARRIERS ARE REFUSED (mission A3): a message a payload-boundary repair
    synthesized (`SYNTHETIC_MESSAGE_KEY`) is not a durable turn — it is rebuilt
    from the transcript on the next iteration, so a stamp written into it is lost
    and re-written one message later, which is precisely the divergence that made
    every tool-loop iteration cold. The stamp lands on the newest message a HUMAN
    (or the operator-guidance drain) actually put in the transcript.
    """
    if not isinstance(messages, list) or not messages:
        return False

    (_carries, _mark, _metadata, _envelope, _strip) = _grounding_api()

    idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict) or str(m.get("role") or "").strip().lower() != "user":
            continue
        if message_is_synthetic(m):
            continue
        idx = i
        break
    if idx is None:
        return False

    target = messages[idx]
    content = target.get("content")
    if not isinstance(content, str) or not content.strip():
        # Content-part lists and media turns are the payload boundary's business;
        # the durable transcript lane this serves is plain text.
        return False
    if _carries(content):
        return False

    payload = grounding if isinstance(grounding, dict) and grounding else _mark(_metadata(), True)
    cleaned = _strip(content)
    envelope = _envelope(payload)
    target["content"] = f"{envelope}\n{cleaned}" if cleaned else envelope
    return True
