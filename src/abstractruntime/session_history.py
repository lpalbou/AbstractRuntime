"""Durable session conversation history (read side).

The run store is the single durable source of a session's conversation:
each thin-client turn is a root run carrying the user prompt in its input
vars and the assistant answer in its output. This module reconstructs that
conversation as plain chat messages so hosts (AbstractGateway) can seed new
runs of the same session server-side — no client-side transcript authority,
and no second transcript store that could drift from the runs.

Contract (agora channel `durable-sessions`, v1):
- Only COMPLETED root runs contribute, and only when both a user prompt and
  an assistant answer could be extracted (a failed run never spoke — its
  half-turn is invisible to replay; revisit if reviewers rule otherwise).
- Internal (`__*`) and scheduled workflows never contribute.
- Messages are chronological; the newest `max_messages` are kept, under a
  cumulative `max_total_chars` budget (drop-oldest whole turns) so replay
  can never blow past a small model's context window — the agent lane
  deliberately disables downstream input trimming, so this budget is the
  only guard.
- Over-long message content is cut with a labeled `#TRUNCATION` marker —
  never silently.
- The reconstruction rides `_best_effort_session_turns` (history bundles),
  so what a replayed model sees is exactly what thin clients already
  display as session history — except for the runtime grounding envelope,
  which replay must carry VERBATIM where display strips it (see the
  `prompt_verbatim` note at the fold below; mission A, 2026-09-22).

Known v1 limits (documented, not silent): turns are dropped whole, never
split. The original limit here — "$artifact-offloaded answers extract as
empty and their turn is skipped" — was the deepest server link in the
2026-07-23 401-incident chain (code-tui c4978 R2) and is CLOSED two ways:
the offloader now reduces largest-children-first so answers stay inline by
size (R1), and root-replaced outputs from the existing corpus resolve
boundedly at extraction (history_bundle._resolve_offloaded_output — offload-
minted refs only, size-capped, answer extraction only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .history_bundle import _best_effort_session_turns

__all__ = [
    "SESSION_TURN_KIND",
    "session_chat_messages",
]

# Metadata marker carried by every replayed message so downstream consumers
# (context folds, transcripts, ledgers) can tell replayed history from the
# live turn.
SESSION_TURN_KIND = "session_turn"

_DEFAULT_MAX_MESSAGES = 40
_DEFAULT_MAX_CHARS_PER_MESSAGE = 8000
# Cumulative budget across ALL replayed messages (~6k tokens at 4 chars/tk):
# small local models must survive replay even when every turn is long.
_DEFAULT_MAX_TOTAL_CHARS = 24000


def _truncate_labeled(text: str, *, max_chars: int, run_id: str) -> str:
    """Cut over-long content with an explicit, labeled marker (house rule:
    truncation is never silent).

    #[WARNING:TRUNCATION] caller-supplied `max_chars_per_message` (ADR-0026 §4):
    the bound is a parameter with a documented default, the cut names its size
    and the run holding the full text, and the run itself is never mutated.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"\n[#TRUNCATION: replay cut at {max_chars} chars; full text in run {run_id}]"
    return text[:max_chars].rstrip() + marker


def session_chat_messages(
    *,
    run_store: Any,
    ledger_store: Any = None,
    artifact_store: Any = None,
    session_id: str,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
    max_chars_per_message: int = _DEFAULT_MAX_CHARS_PER_MESSAGE,
    max_total_chars: int = _DEFAULT_MAX_TOTAL_CHARS,
    until_ms: Optional[int] = None,
    exclude_run_ids: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Reconstruct a session's prior conversation as chat messages.

    Returns `[{"role": "user"|"assistant", "content": str, "metadata":
    {"kind": "session_turn", "run_id": ..., "ts": ...}}, ...]` in
    chronological order, capped to the newest `max_messages` under a
    cumulative `max_total_chars` budget (whole turns dropped oldest-first).
    Strictly alternating user/assistant PAIRS: a turn missing either side is
    skipped whole (runtime review A3 — dangling user messages are
    provider-hostile). Consequence: a turn needs 2 message slots, so
    `max_messages=1` replays nothing.

    Cost contract (runtime review A1): O(turns) run loads plus at most one
    ledger read per ANSWERLESS run (flow-end fallback); never stats or
    artifact walks. Passing `ledger_store` improves answer recall — some
    flow-style runs carry their answer only in the ledger's flow-end record
    (runtime review A2); with `ledger_store=None` those turns are skipped.

    Pure read: nothing is written, nothing decays. Failures in individual
    turns are skipped (a corrupt run must not take the whole replay down);
    a broken store surfaces as the exception the caller must handle.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return []

    max_msgs = int(max_messages) if isinstance(max_messages, int) else _DEFAULT_MAX_MESSAGES
    if max_msgs <= 0:
        return []
    max_chars = (
        int(max_chars_per_message)
        if isinstance(max_chars_per_message, int) and int(max_chars_per_message) > 0
        else _DEFAULT_MAX_CHARS_PER_MESSAGE
    )
    total_budget = (
        int(max_total_chars)
        if isinstance(max_total_chars, int) and int(max_total_chars) > 0
        else _DEFAULT_MAX_TOTAL_CHARS
    )
    excluded = {str(r).strip() for r in (exclude_run_ids or []) if str(r or "").strip()}

    # Turn budget: each turn yields at most 2 messages, so fetching
    # ceil(max_msgs / 2) newest turns is sufficient even before skips;
    # over-fetch a little so skipped turns (failed, empty) don't starve
    # the window.
    turn_limit = max(4, (max_msgs + 1) // 2 + 8)

    turns = _best_effort_session_turns(
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        session_id=sid,
        limit=turn_limit,
        until_ms=until_ms,
        include_stats=False,
        include_artifacts=False,
    )

    # Build whole turns first: caps below must drop turns atomically — a
    # leading assistant message with its user half cut off would
    # misattribute the reply to the wrong question.
    turn_pairs: List[List[Dict[str, Any]]] = []
    for turn in turns or []:
        if not isinstance(turn, dict):
            continue
        rid = str(turn.get("run_id") or "").strip()
        if rid and rid in excluded:
            continue
        kind = str(turn.get("kind") or "").strip().lower()
        if kind in {"internal", "scheduled"}:
            # Belt on top of the reconstruction's own filtering: scheduled
            # wrapper runs are not conversation (audit finding #4).
            continue
        status = str(turn.get("status") or "").strip().lower()
        if status != "completed":
            # v1: a run that never completed never answered — its half-turn
            # is invisible to replay (contract question (a), conservative).
            continue
        prompt = str(turn.get("prompt") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if not prompt or not answer:
            continue
        # BYTE-EXACT REPLAY (mission A, 2026-09-22). `prompt` is the DISPLAY form —
        # the runtime's `<runtime_metadata>` envelope stripped off so UIs show what
        # the human typed. Replaying that form re-sends a user turn with different
        # bytes than it was sent with, and turn N's prompt then stops being a byte-
        # prefix of turn N+1's, which is the one precondition every prefix cache
        # (provider prompt cache, KV reuse, mlx-vlm APC) needs. `prompt_verbatim`
        # carries the stored bytes when the turn was stamped; absent, `prompt` is
        # already the exact bytes.
        verbatim = str(turn.get("prompt_verbatim") or "").strip()
        if verbatim:
            prompt = verbatim
        ts = str(turn.get("created_at") or turn.get("updated_at") or "").strip()
        meta: Dict[str, Any] = {"kind": SESSION_TURN_KIND, "run_id": rid}
        if ts:
            meta["ts"] = ts
        turn_pairs.append(
            [
                {
                    "role": "user",
                    "content": _truncate_labeled(prompt, max_chars=max_chars, run_id=rid),
                    "metadata": dict(meta),
                },
                {
                    "role": "assistant",
                    "content": _truncate_labeled(answer, max_chars=max_chars, run_id=rid),
                    "metadata": dict(meta),
                },
            ]
        )

    # Newest-first fold under both caps, then restore chronology.
    kept: List[List[Dict[str, Any]]] = []
    kept_messages = 0
    kept_chars = 0
    dropped_by = ""
    for pair in reversed(turn_pairs):
        pair_chars = sum(len(str(m.get("content") or "")) for m in pair)
        if kept_messages + len(pair) > max_msgs:
            dropped_by = f"max_messages={max_msgs}"
            break
        if kept and kept_chars + pair_chars > total_budget:
            # Always keep at least the newest turn, even when it alone
            # exceeds the budget: replaying nothing would be worse than
            # replaying one long turn (per-message caps already bound it).
            dropped_by = f"max_total_chars={total_budget}"
            break
        kept.append(pair)
        kept_messages += len(pair)
        kept_chars += pair_chars

    messages: List[Dict[str, Any]] = []
    for pair in reversed(kept):
        messages.extend(pair)

    dropped_turns = len(turn_pairs) - len(kept)
    if dropped_turns > 0 and messages:
        #[WARNING:TRUNCATION] whole turns dropped by the replay budget — stated, never silent
        #
        # Per-MESSAGE cuts were already labeled (`_truncate_labeled`); dropping
        # a whole TURN was not, so a replayed model received a conversation
        # whose beginning had been deleted and read it as the whole session
        # (ADR-0026 §1: "no truncation may occur quietly", and §1's attribution
        # rule — the marker names the budget that caused the cut).
        #
        # Carried as a PREFIX on the oldest surviving user message rather than
        # as an extra message: this function's contract is strict user/assistant
        # PAIRS (a dangling half-turn is provider-hostile), and the message
        # count is what the `max_messages` window means. A prefix is visible to
        # the model and to any transcript without breaking either.
        head = messages[0]
        head["metadata"] = {
            **(head.get("metadata") if isinstance(head.get("metadata"), dict) else {}),
            "replay_truncated": True,
            "dropped_turns": dropped_turns,
            "dropped_by": dropped_by,
        }
        head["content"] = (
            f"[#TRUNCATION: {dropped_turns} earlier turn(s) of this session were dropped from replay "
            f"by {dropped_by} (abstractruntime.session_history); this history starts mid-conversation]\n"
            f"{head.get('content') or ''}"
        )
    return messages
