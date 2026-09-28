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
- THE HISTORY WINDOW (operator ruling 2026-09-28, ADR-0026 §3 — a budget met
  by SELECTION, never by cutting): replay keeps the most recent turns up to
  `HISTORY_REPLAY_MAX_TOKENS` (50,000) estimated tokens, whole messages,
  newest first. No message is ever cut, no message count is capped, and a
  turn is never split (user + answer stay together: a dangling half-turn is
  provider-hostile). Tokens are estimated by the framework's one estimator,
  `memory.token_budget.estimate_message_tokens`. The window is the ONLY
  bound this module applies; everything else in the model's context is the
  model's to use (its full context window).
- The window is observable: the returned `ReplayedHistory` carries a
  `report` (replayed/dropped messages and tokens, the budget, the estimator)
  that hosts record in the run (`vars._runtime.session_history`), and when
  older turns were dropped the oldest surviving message starts with a
  labeled `#TRUNCATION` notice (ADR-0026 §1: never silent).
- The reconstruction rides `_best_effort_session_turns` (history bundles),
  so what a replayed model sees is exactly what thin clients already
  display as session history — except for the runtime grounding envelope,
  which replay must carry VERBATIM where display strips it (see the
  `prompt_verbatim` note at the fold below; mission A, 2026-09-22).

Offloaded answers: the original limit here — "$artifact-offloaded answers
extract as empty and their turn is skipped" — was the deepest server link in
the 2026-07-23 401-incident chain (code-tui c4978 R2) and is CLOSED two ways:
the offloader now reduces largest-children-first so answers stay inline by
size (R1), and root-replaced outputs from the existing corpus resolve
boundedly at extraction (history_bundle._resolve_offloaded_output — offload-
minted refs only, size-capped, answer extraction only).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .history_bundle import _best_effort_session_turns
from .memory.token_budget import estimate_message_tokens

__all__ = [
    "HISTORY_REPLAY_MAX_TOKENS",
    "HISTORY_WINDOW_POLICY",
    "RETIRED_REPLAY_CAP_INPUTS",
    "ReplayedHistory",
    "SESSION_TURN_KIND",
    "SessionHistoryError",
    "announce_dropped",
    "discussion_seed_messages",
    "fold_history_window",
    "session_chat_messages",
    "window_transcript",
]

logger = logging.getLogger(__name__)


class SessionHistoryError(RuntimeError):
    """A strict history read could not produce the session's history.

    Raised only with `strict=True` (automation context preparation and
    discussion seeding, automations contract C3): starting those without their
    context would silently change what the model sees, so they fail instead.
    """

    reason_code = "history_unavailable"

# Metadata marker carried by every replayed message so downstream consumers
# (context folds, transcripts, ledgers) can tell replayed history from the
# live turn.
SESSION_TURN_KIND = "session_turn"

# THE history window (operator ruling 2026-09-28, verbatim: "by default,
# models must be allowed to use their full context if needed. for history, we
# can limit it at the last 50k tokens."). One rule for every replay path
# (gateway session seeding, automation growing mode, discussion seeds).
HISTORY_REPLAY_MAX_TOKENS = 50_000
HISTORY_WINDOW_POLICY = "most_recent_whole_turns"
_TOKEN_ESTIMATOR = "abstractruntime.memory.token_budget.estimate_message_tokens"

# The replay caps retired by the history window (runtime <= 0.6). Hosts built
# against 0.6 (AbstractGateway 0.6.0) still pass them; refusing them raised a
# TypeError those hosts swallowed, so the session replayed NO history. They are
# accepted, IGNORED (the window is the only bound — no count or char cap comes
# back), logged, and named in `report["ignored_inputs"]`.
RETIRED_REPLAY_CAP_INPUTS = ("max_messages", "max_total_chars", "max_chars_per_message")

# First fetch of the newest turns; doubled while the window still has room
# and older turns remain (a turn is 2 messages, so 32 turns is already more
# than most windows hold).
_FIRST_TURN_FETCH = 32


class ReplayedHistory(list):
    """The replayed messages (a plain list of chat messages) plus `report`.

    `report` is the window's receipt, recorded by hosts in the run
    (`vars._runtime.session_history`):
    `{policy, max_tokens, token_estimator, replayed_messages, replayed_tokens,
    dropped_messages, dropped_tokens, dropped_counts_complete,
    oversize_turn_kept}`. `dropped_counts_complete` is False when the read
    stopped at the window without walking the rest of a long session (cost
    contract A1): the dropped counts then cover only the turns read, and more,
    older turns were dropped too. `oversize_turn_kept` is True when the newest
    turn alone exceeds the budget and was kept whole (see
    `fold_history_window`).
    """

    report: Dict[str, Any]

    def __init__(self, messages: Sequence[Dict[str, Any]] = (), *, report: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(messages)
        self.report = dict(report) if isinstance(report, dict) else window_report(max_tokens=HISTORY_REPLAY_MAX_TOKENS)


def window_report(
    *,
    max_tokens: int,
    replayed_messages: int = 0,
    replayed_tokens: int = 0,
    dropped_messages: int = 0,
    dropped_tokens: int = 0,
    dropped_counts_complete: bool = True,
    oversize_turn_kept: bool = False,
) -> Dict[str, Any]:
    return {
        "policy": HISTORY_WINDOW_POLICY,
        "max_tokens": int(max_tokens),
        "token_estimator": _TOKEN_ESTIMATOR,
        "replayed_messages": int(replayed_messages),
        "replayed_tokens": int(replayed_tokens),
        "dropped_messages": int(dropped_messages),
        "dropped_tokens": int(dropped_tokens),
        "dropped_counts_complete": bool(dropped_counts_complete),
        "oversize_turn_kept": bool(oversize_turn_kept),
    }


def _validated_max_tokens(max_tokens: Any) -> int:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"history window max_tokens must be a positive int, got {max_tokens!r}")
    return int(max_tokens)


def fold_history_window(
    pairs: Sequence[Sequence[Dict[str, Any]]],
    *,
    max_tokens: int = HISTORY_REPLAY_MAX_TOKENS,
    older_unread: bool = False,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """Keep the newest whole turns that fit `max_tokens`; return (kept, report).

    `pairs` are chronological turns (each a list of whole messages, normally a
    user/assistant pair). The fold walks newest-first and stops at the first
    turn that does not fit — the window is contiguous: skipping one large turn
    to keep older ones would hand the model a conversation with a hole in it.
    Content is never cut (ADR-0026 §3: budgets are met by selection).

    A total of exactly `max_tokens` fits. THE ONE EXCEPTION to the budget: when
    the NEWEST turn alone is larger than `max_tokens` it is kept whole and
    alone (`oversize_turn_kept`). Cutting it would be lossy truncation of the
    most relevant turn (ADR-0026 §2/§3 forbid that on this path), and dropping
    it would replay nothing — the model would answer a follow-up to its own
    previous reply without that reply. The window exists to select history,
    not to impose a context limit: models may use their full context, and a
    turn the model's context cannot hold fails loudly at the provider instead
    of silently disappearing here.

    `older_unread=True` says the caller stopped reading before the session's
    beginning; if the window is full, the dropped counts are then a lower
    bound (`dropped_counts_complete=False`).
    """
    budget = _validated_max_tokens(max_tokens)
    sized = [(list(pair), sum(estimate_message_tokens(m) for m in pair)) for pair in pairs]
    kept: List[List[Dict[str, Any]]] = []
    kept_tokens = 0
    oversize = False
    stop = len(sized)
    for i in range(len(sized) - 1, -1, -1):
        pair, tokens = sized[i]
        if kept_tokens + tokens > budget:
            if kept:
                break
            # The newest turn alone exceeds the window: kept whole (see docstring).
            oversize = True
            kept.append(pair)
            kept_tokens += tokens
            stop = i
            break
        kept.append(pair)
        kept_tokens += tokens
        stop = i
    kept.reverse()
    dropped = sized[:stop]
    report = window_report(
        max_tokens=budget,
        replayed_messages=sum(len(p) for p in kept),
        replayed_tokens=kept_tokens,
        dropped_messages=sum(len(p) for p, _t in dropped),
        dropped_tokens=sum(t for _p, t in dropped),
        dropped_counts_complete=not (older_unread and bool(dropped)),
        oversize_turn_kept=oversize,
    )
    return kept, report


def announce_dropped(messages: List[Dict[str, Any]], report: Dict[str, Any], *, stamp_metadata: bool = True) -> None:
    """Prefix the oldest kept message with the window's labeled drop notice (in place).

    `messages` are the kept messages, oldest first; `report` is the window's
    receipt from `fold_history_window`. Nothing happens when the report
    dropped nothing. Hosts that fold history themselves (e.g. client-sent
    messages) call this so their notice is the same as session replay's.

    #[WARNING:TRUNCATION] whole turns dropped by the history window — stated, never silent

    Dropping older turns is lossy by ADR-0026's own list ("trimming message
    history"); unannounced, a replayed model reads a conversation whose
    beginning was deleted as the whole session (§1: "no truncation may occur
    quietly", attributed to the component and the budget that caused it).
    Carried as a PREFIX on the oldest surviving user message rather than as an
    extra message: the contract is strict user/assistant PAIRS. The notice is
    not counted against the window (it is one line).

    `stamp_metadata=False` leaves the message's keys as they were (plain
    role/content transcripts that go straight to a model client); the notice
    in the content is the same.
    """
    if not messages or int(report.get("dropped_messages") or 0) <= 0:
        return
    more = "" if report.get("dropped_counts_complete") else " (and older turns not counted)"
    head = messages[0]
    if stamp_metadata:
        head["metadata"] = {
            **(head.get("metadata") if isinstance(head.get("metadata"), dict) else {}),
            "replay_truncated": True,
            "history_window": dict(report),
        }
    notice = (
        f"[#TRUNCATION: {report['dropped_messages']} earlier message(s) of this session "
        f"(~{report['dropped_tokens']} tokens){more} were dropped from replay by the history window "
        f"(the most recent {report['max_tokens']} tokens, whole turns; abstractruntime.session_history); "
        f"this history starts mid-conversation]"
    )
    # A stamped turn keeps its `<runtime_metadata>` envelope at the head (the
    # payload boundary would otherwise stack a second one in front).
    from .turn_grounding import split_head_grounding  # lazy: the grounding helpers load AbstractCore

    envelope, rest = split_head_grounding(head.get("content") or "")
    if envelope:
        sep = "" if envelope.endswith("\n") else "\n"
        head["content"] = f"{envelope}{sep}{notice}\n{rest.lstrip(chr(10))}"
    else:
        head["content"] = f"{notice}\n{rest}"


# The pre-0.7.0 private name, kept for callers that imported it.
_announce_dropped = announce_dropped


def window_transcript(
    messages: Sequence[Dict[str, Any]],
    *,
    max_tokens: int = HISTORY_REPLAY_MAX_TOKENS,
    current_turn_start: Optional[int] = None,
) -> ReplayedHistory:
    """THE history window over a transcript a host keeps in memory or in run vars.

    For conversations that are not replayed from the run store — the entity
    chat driver's and the entity visit's own user/assistant history — so they
    select history by the same rule as session replay: `fold_history_window`
    over whole turns (a turn is a user message and every message that follows
    it up to the next user message), newest first, up to `max_tokens`, with
    the labeled `#TRUNCATION` notice on the oldest kept message when older
    turns were dropped. Nothing is cut and no count is capped. The input is
    not mutated; kept messages are copies with the keys they came with. The
    window's receipt is `.report`, for the host to record in its run.

    `current_turn_start` is the index (in `messages`) of the message that
    opened the turn in progress. Everything from there on is ONE turn — the
    newest, always kept whole (`oversize_turn_kept` when it alone exceeds the
    window) — however many user-role messages a loop adds inside it (an
    `ask_user` answer, operator guidance): splitting there would drop the
    turn's own question and tool results. Only the messages before it are
    grouped at user messages. Out of range raises ValueError.
    """
    items = list(messages)
    if current_turn_start is None:
        older, current = items, []
    else:
        if isinstance(current_turn_start, bool) or not isinstance(current_turn_start, int) or not (
            0 <= current_turn_start <= len(items)
        ):
            raise ValueError(
                f"current_turn_start must be an index into the {len(items)} messages, got {current_turn_start!r}"
            )
        older, current = items[:current_turn_start], items[current_turn_start:]
    turns: List[List[Dict[str, Any]]] = []
    for message in older:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user" or not turns:
            turns.append([])
        turns[-1].append(dict(message))
    current_turn = [dict(m) for m in current if isinstance(m, dict)]
    if current_turn:
        turns.append(current_turn)
    kept, report = fold_history_window(turns, max_tokens=max_tokens)
    out = [m for turn in kept for m in turn]
    announce_dropped(out, report, stamp_metadata=False)
    return ReplayedHistory(out, report=report)


def session_chat_messages(
    *,
    run_store: Any,
    ledger_store: Any = None,
    artifact_store: Any = None,
    session_id: str,
    max_tokens: int = HISTORY_REPLAY_MAX_TOKENS,
    until_ms: Optional[int] = None,
    exclude_run_ids: Optional[Any] = None,
    automation_id: Optional[str] = None,
    through_occurrence: Optional[int] = None,
    strict: bool = False,
    max_messages: Optional[int] = None,
    max_total_chars: Optional[int] = None,
    max_chars_per_message: Optional[int] = None,
) -> ReplayedHistory:
    """Reconstruct a session's prior conversation as chat messages.

    Turns come from `session_turns.select_session_turns` (so an automation's
    occurrences are turns); `automation_id` / `through_occurrence` bound them
    (e.g. a discussion seeded "through occurrence N"). In a DISCUSSION session
    (runs carrying `vars._meta.discussion`) the discussion root's
    `seed_messages` are prepended as the oldest history, dropped first under
    the window.

    `strict=True` (automation context preparation, discussion seeding) raises
    `SessionHistoryError` instead of degrading: a store without a run index,
    a discussion whose root or seed is missing, or a seed offloaded to an
    artifact that cannot be resolved. Non-strict reads keep the historical
    best-effort behavior.

    Returns a `ReplayedHistory` — a list of `{"role": "user"|"assistant",
    "content": str, "metadata": {"kind": "session_turn", "run_id": ...,
    "ts": ...}}` in chronological order, the newest whole turns that fit
    `max_tokens` (`fold_history_window`) — with the window's receipt in
    `.report`. Strictly alternating user/assistant PAIRS: a turn missing
    either side is skipped whole (runtime review A3 — dangling user messages
    are provider-hostile). Content is replayed exactly as stored, never cut.

    Cost contract (runtime review A1): O(turns read) run loads plus at most one
    ledger read per ANSWERLESS run (flow-end fallback); never stats or
    artifact walks. Turns are read newest-first in doubling batches and the
    read stops as soon as the window is full, so a long session costs about
    what its window holds. Passing `ledger_store` improves answer recall —
    some flow-style runs carry their answer only in the ledger's flow-end
    record (runtime review A2); with `ledger_store=None` those turns are
    skipped.

    `max_messages`, `max_total_chars` and `max_chars_per_message` are the
    retired replay caps (`RETIRED_REPLAY_CAP_INPUTS`): accepted so a host built
    against runtime 0.6 still gets its history, and IGNORED — the window above
    is the only bound. Any that is passed is logged as a warning and named in
    `report["ignored_inputs"]`.

    Pure read: nothing is written, nothing decays. Failures in individual
    turns are skipped (a corrupt run must not take the whole replay down);
    a broken store surfaces as the exception the caller must handle.
    """
    ignored = _ignored_retired_inputs(
        max_tokens,
        max_messages=max_messages, max_total_chars=max_total_chars, max_chars_per_message=max_chars_per_message
    )
    history = _session_chat_messages(
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        session_id=session_id,
        max_tokens=max_tokens,
        until_ms=until_ms,
        exclude_run_ids=exclude_run_ids,
        automation_id=automation_id,
        through_occurrence=through_occurrence,
        strict=strict,
    )
    if ignored:
        history.report["ignored_inputs"] = ignored
    return history


def _ignored_retired_inputs(max_tokens: Any, **passed: Any) -> Dict[str, Any]:
    """The retired cap inputs a caller passed (name -> value), logged once per call."""
    ignored = {name: value for name, value in passed.items() if value is not None}
    if ignored:
        logger.warning(
            "abstractruntime.session_history: ignoring retired replay cap input(s) %s — the history "
            "window (the most recent %s tokens of whole turns) is the only bound; update the caller",
            ", ".join(f"{name}={value!r}" for name, value in ignored.items()),
            max_tokens,
        )
    return ignored


def _session_chat_messages(
    *,
    run_store: Any,
    ledger_store: Any,
    artifact_store: Any,
    session_id: str,
    max_tokens: int,
    until_ms: Optional[int],
    exclude_run_ids: Optional[Any],
    automation_id: Optional[str],
    through_occurrence: Optional[int],
    strict: bool,
) -> ReplayedHistory:
    budget = _validated_max_tokens(max_tokens)
    sid = str(session_id or "").strip()
    if not sid:
        return ReplayedHistory(report=window_report(max_tokens=budget))
    excluded = {str(r).strip() for r in (exclude_run_ids or []) if str(r or "").strip()}

    if strict and not callable(getattr(run_store, "list_run_index", None)):
        raise SessionHistoryError(
            f"run store {type(run_store).__name__} has no run index; strict history needs one"
        )

    from .session_turns import OccurrenceNotInSession

    seed_pairs = _seed_pairs(
        discussion_seed_messages(run_store=run_store, artifact_store=artifact_store, session_id=sid, strict=strict),
        strict=strict,
    )

    fetch = _FIRST_TURN_FETCH
    while True:
        try:
            turns = _best_effort_session_turns(
                run_store=run_store,
                ledger_store=ledger_store,
                artifact_store=artifact_store,
                session_id=sid,
                limit=fetch,
                until_ms=until_ms,
                include_stats=False,
                include_artifacts=False,
                automation_id=automation_id,
                through_occurrence=through_occurrence,
            )
        except OccurrenceNotInSession as exc:
            # "History through occurrence N" without N has no correct answer.
            if strict:
                raise SessionHistoryError(str(exc)) from exc
            return ReplayedHistory(report=window_report(max_tokens=budget))
        older_unread = len(turns or []) >= fetch
        # The discussion seed is the OLDEST history: prepended, dropped first —
        # and only once the session's own turns were all read.
        pairs = _turn_pairs(turns, excluded=excluded)
        if not older_unread:
            pairs = seed_pairs + pairs
        kept, report = fold_history_window(pairs, max_tokens=budget, older_unread=older_unread)
        if not older_unread or len(kept) < len(pairs):
            # Either everything was read, or the window filled before the
            # oldest turn read — reading further cannot change what is kept.
            break
        fetch *= 2

    messages: List[Dict[str, Any]] = [m for pair in kept for m in pair]
    announce_dropped(messages, report)
    return ReplayedHistory(messages, report=report)


def _turn_pairs(turns: Any, *, excluded: set) -> List[List[Dict[str, Any]]]:
    """Whole user/assistant pairs from session turns, chronological.

    Built whole first: the window drops turns atomically — a leading assistant
    message with its user half cut off would misattribute the reply to the
    wrong question.
    """
    turn_pairs: List[List[Dict[str, Any]]] = []
    for turn in turns or []:
        if not isinstance(turn, dict):
            continue
        rid = str(turn.get("run_id") or "").strip()
        if rid and rid in excluded:
            continue
        # Which runs are turns is decided in ONE place,
        # `session_turns.select_session_turns` (internal runs and legacy
        # scheduled wrappers excluded; automation occurrences included).
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
                {"role": "user", "content": prompt, "metadata": dict(meta)},
                {"role": "assistant", "content": answer, "metadata": dict(meta)},
            ]
        )
    return turn_pairs


def _resolve_offloaded(value: Any, *, artifact_store: Any) -> Any:
    from .core.runtime import _resolve_artifact_backed_value

    return _resolve_artifact_backed_value(value, artifact_store=artifact_store)


def _contains_artifact_ref(value: Any, *, depth: int = 0) -> bool:
    from .storage.artifacts import is_artifact_ref

    if depth > 12:
        return False
    if is_artifact_ref(value):
        return True
    if isinstance(value, dict):
        return any(_contains_artifact_ref(v, depth=depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(_contains_artifact_ref(v, depth=depth + 1) for v in value)
    return False


def discussion_seed_messages(
    *, run_store: Any, artifact_store: Any = None, session_id: str, strict: bool = False
) -> List[Dict[str, Any]]:
    """The seed messages of a discussion session, or [] for any other session.

    A discussion session's runs carry `vars._meta.discussion` (index role
    `discussion`); its root run (`discussion_root_run_id`) holds the
    `seed_messages` — the automation's history through occurrence N — once.
    An offloaded seed is resolved through the artifact store. Missing or
    unresolvable seeds raise `SessionHistoryError` when `strict`, else yield [].
    """
    from .core.run_attribution import SessionAttributionError, resolve_discussion_root, store_session_kinds

    sid = str(session_id or "").strip()
    if not sid or not callable(getattr(run_store, "list_run_index", None)):
        return []
    if "discussion" not in store_session_kinds(run_store, sid):
        return []

    def _fail(message: str) -> List[Dict[str, Any]]:
        if strict:
            raise SessionHistoryError(f"discussion session {sid}: {message}")
        return []

    try:
        root, root_discussion = resolve_discussion_root(run_store, sid)
    except SessionAttributionError as exc:
        # Never seed from an unvalidated (possibly foreign) root, strict or not.
        return _fail(str(exc))
    root_id = str(root.run_id)
    seed = _resolve_offloaded(root_discussion.get("seed_messages"), artifact_store=artifact_store)
    if _contains_artifact_ref(seed):
        return _fail(f"seed_messages of {root_id} are offloaded and cannot be resolved")
    if not isinstance(seed, list) or not all(
        isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        for m in seed
    ):
        return _fail(f"seed_messages of {root_id} are malformed")
    return [dict(m) for m in seed]


def _seed_pairs(seed: List[Dict[str, Any]], *, strict: bool) -> List[List[Dict[str, Any]]]:
    """Seed messages as whole user/assistant pairs (the fold drops turns atomically)."""
    pairs: List[List[Dict[str, Any]]] = []
    for i in range(0, len(seed) - 1, 2):
        user, assistant = seed[i], seed[i + 1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            if strict:
                raise SessionHistoryError("discussion seed_messages are not user/assistant pairs")
            continue
        pair = []
        for m in (user, assistant):
            meta = dict(m.get("metadata")) if isinstance(m.get("metadata"), dict) else {}
            meta["discussion_seed"] = True
            pair.append({"role": m["role"], "content": str(m["content"]), "metadata": meta})
        pairs.append(pair)
    if len(seed) % 2 and strict:
        raise SessionHistoryError("discussion seed_messages end with an unpaired message")
    return pairs
