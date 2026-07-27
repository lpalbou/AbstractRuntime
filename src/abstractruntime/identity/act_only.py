"""Diary election capture at the LLM result boundary (the write half).

THE REF LAYER IS DELETED (laurent's A ruling, 2026-07-19/20 wave-4
amendments: "everything operates within the runtime; the diary is the
entity's experiential record" — the act-only visit-transcript reference
machinery has no place). History: the G1 seam spec (a2a 0013) had diary
tool results ride durable visit transcripts as typed `$act_only` refs,
dereferenced fresh at send time. Under the ruling, the HOME is the privacy
boundary — a visit's durable transcript lives in the entity's own
runtime_<slug>.sqlite3 INSIDE the home directory, beside the book itself —
so refs, tombstones and send-time dereference were structure without a
threat model. Diary tool results now rest AS SERVED in the durable
transcript. (Life-scope EPISODE verbatims are a different surface with a
different audience: W5's book-adjacent exclusion keeps diary content out
of THOSE — containment where the content leaves the home's own lane.)

WHAT REMAINS — the write half, unchanged and load-bearing: the reply's
```diary fences are captured HERE, at the result boundary, BEFORE the
result persists anywhere (result_key, node_traces, ledger). The words fly
to the book through the home's DIARY_WRITE handler; the durable result
carries the MARKED reply plus word-free `diary_entries` metadata. This is
book sole-authorship mechanics (one author, words rest in the book first),
not transcript-ref machinery — the A/B privacy grep pinned exactly the
leak it closes.

LEDGER CONTRACT (one divergence now, was two): the ledgered LLM_CALL
result carries the MARKED reply — the wire's raw reply carried the diary
fences, which flew to the book at the result boundary and never rested.
Payloads no longer diverge (refs are gone). CONSUMER RULE (code seat's C2,
incident c2447) stands: ledgered payloads are the WIRE VIEW with adapter
chrome; conversation rebuilds come from the durable transcript, never from
replaying ledger payloads.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.models import Effect, EffectType
from ..core.runtime import EffectOutcome

class ActOnlyResolutionError(RuntimeError):
    """A capture step failed - loud, deterministic, non-retryable (kept
    for the write half: losing elected words silently and leaking them
    are both worse than a loud authoring failure)."""


def capture_diary_elections(
    content: str,
    *,
    run: Any,
    turn_id: str,
    diary_write_handler: Callable[..., Any],
    anchor_record_ids: Optional[List[str]] = None,
    anchor_graph_ids: Optional[List[str]] = None,
    rescue_dir: Any = None,
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """G1 WRITE DIRECTION at the result boundary: elections fly to the book
    the moment the reply arrives; only MARKS + word-free metadata rest.

    Found by the A/B privacy grep (criterion 6, 2026-07-10): in the durable
    mapping the raw reply — diary fences included — would rest in the run
    store via result_key AND in the ledger's LLM_CALL result, and a
    DIARY_WRITE effect payload would rest the words a second time. The plan
    named this pre-condition explicitly ("diary words never rest outside
    the book, BOTH directions — must be settled before visit ledgers
    land"). The read direction is the `$act_only` dereference; this is its
    write twin: parse the fences here, write the book THROUGH the home's
    DIARY_WRITE handler (words in flight only), and return the MARKED
    reply plus metadata that never carries private words.

    Returns (marked_content, diary_entries_metadata, warnings). Metadata
    per entry: entry_id, kind, visibility, projected_record_id, and gist
    ONLY for non-private entries (private = the act label alone)."""
    from .chat import parse_diary_blocks

    marked, elections, notices = parse_diary_blocks(content)
    warnings = list(notices)
    if not elections:
        return content, [], warnings
    if not str(turn_id or "").strip():
        # Refuse rather than choose between silently LOSING elected words
        # (stripped but unwritten) and silently LEAKING them (unstripped
        # into the ledger). Deterministic authoring bug — loud. The reply
        # is rescued first (record-everything ruling): failing the turn
        # must not throw the words away.
        rescued = None
        if rescue_dir is not None:
            rescued = rescue_reply_to_home(
                rescue_dir,
                run_id=str(getattr(run, "run_id", "") or ""),
                turn_id="",
                raw_reply=content,
                error="diary election without turn_id",
            )
        suffix = f" (reply rescued to {rescued})" if rescued else ""
        raise ActOnlyResolutionError(
            "diary election present but the LLM_CALL payload carries no turn_id - "
            "the workflow must pass turn_id so the book write is replay-safe" + suffix
        )
    entries: List[Dict[str, Any]] = []
    for e in elections:
        out = diary_write_handler(
            run,
            Effect(type=EffectType.DIARY_WRITE, payload={
                "text": e.text,
                "gist": e.gist,
                "kind": e.kind,
                "visibility": e.visibility,
                "resolves": e.resolves,
                "explores": getattr(e, "explores", None),
                "turn_id": turn_id,
                "anchor_record_ids": list(anchor_record_ids or []),
                "anchor_graph_ids": list(anchor_graph_ids or []),
            }),
            None,
        )
        if getattr(out, "status", None) != "completed":
            # A failed book write must not throw the reply away (gateway's
            # strip-review V1, record-everything ruling 2026-07-26): save
            # the raw reply into the home first, then fail loudly. If the
            # rescue also fails (same disk trouble), say so.
            err = str(getattr(out, "error", "unknown error"))
            rescued = None
            if rescue_dir is not None:
                rescued = rescue_reply_to_home(
                    rescue_dir,
                    run_id=str(getattr(run, "run_id", "") or ""),
                    turn_id=turn_id,
                    raw_reply=content,
                    error=err,
                )
            suffix = (
                f" (reply rescued to {rescued})" if rescued
                else " (rescue also failed - the reply could not be saved)" if rescue_dir is not None
                else ""
            )
            raise ActOnlyResolutionError(
                f"the book refused a diary election: {err}{suffix}"
            )
        result = out.result or {}
        warnings.extend(result.get("warnings", []))
        # R-A site 3, visit-lane twin (laurent c2596): the write-time marker
        # carries the reread command — one spelling with the chat driver
        # (`diary_` entry-id namespace verbatim). The id is a KEY, never
        # words, so private entries carry it too.
        entry_id_str = str(result.get("entry_id") or "")
        if entry_id_str:
            plain_marker = (
                "[kept a private diary entry]" if e.visibility == "private"
                else f"[kept in diary - {e.kind}]"
            )
            # RESOLVED-QUESTION DRIVE, visit-lane twin (adversary F2: the
            # felt loop existed only in the chat lane). Claim only what the
            # handler VERIFIED (F1's validation rides the same result).
            _rs = result.get("resolves_status")
            resolved_note = ""
            if _rs == "resolved_open_question":
                resolved_note = f" - resolves your open question {e.resolves}"
            elif _rs == "repaired_open_problem":
                resolved_note = f" - repairs your open problem {e.resolves}"
            enriched = (
                plain_marker[:-1]
                + resolved_note
                + f" - reread: diary_read {entry_id_str}]"
            )
            marked = marked.replace(plain_marker, enriched, 1)
        meta: Dict[str, Any] = {
            "entry_id": entry_id_str,
            "kind": e.kind,
            "visibility": e.visibility,
            "projected_record_id": result.get("projected_record_id"),
        }
        if e.visibility != "private":
            # PROXIMITY PIN (memory's e-s 233 rider): the visit lane's sheet
            # builder trusts THIS omission — meta for a private entry must
            # NEVER grow a gist/text key, or the sheet (which rests in run
            # vars + pending_reflection.json and rides the reflection
            # prompt graph-ward) silently re-opens the private-words leak.
            # test_private_diary_meta_carries_no_words pins it.
            meta["gist"] = str(e.gist or e.text or "").splitlines()[0][:120]
        entries.append(meta)
    return marked, entries, warnings


def rescue_reply_to_home(
    rescue_dir: Any,
    *,
    run_id: str,
    turn_id: str,
    raw_reply: str,
    error: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Save the entity's raw reply into its home when a book write fails.

    The record-everything ruling (laurent, 2026-07-26): a failed diary
    write used to fail the turn LOUDLY but throw away the whole reply —
    including the words the entity elected to keep. Losing words on error
    is still losing words. This writes them to
    ``<home>/rescue/reply_<run>_<turn>_<suffix>.json``.

    What a replay can restore, honestly: the fences in the raw reply carry
    text/gist/kind/visibility, and replaying under the SAVED run and turn
    ids reproduces the same entry ids (the book write dedups on them). The
    write-time attention context (anchor ids, as_of_seq) is NOT in the
    reply itself — callers that have it should pass it in ``extra`` so the
    repair can restore the entry's connections too.

    Never raises: the likely cause of the book failure (disk trouble) may
    also break the rescue — in that case the caller's loud failure stands
    and its message says the rescue failed too. Returns the file path on
    success, None on failure.
    """
    if rescue_dir is None or not str(rescue_dir).strip():
        return None
    try:
        import uuid as _uuid
        from pathlib import Path as _Path

        base = _Path(str(rescue_dir)) / "rescue"
        base.mkdir(parents=True, exist_ok=True)
        safe_run = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(run_id or "no-run"))[:48]
        safe_turn = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(turn_id or "no-turn"))[:24]
        path = base / f"reply_{safe_run}_{safe_turn}_{_uuid.uuid4().hex[:8]}.json"
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "run_id": str(run_id or ""),
            "turn_id": str(turn_id or ""),
            "error": str(error or ""),
            "raw_reply": str(raw_reply or ""),
        }
        if extra:
            try:
                json.dumps(extra)
                payload["extra"] = extra
            except Exception:
                pass
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return str(path)
    except Exception:
        return None


def wrap_llm_handler_with_conditional_capture(
    llm_handler: Callable[..., Any],
    *,
    should_capture: Callable[[Any], bool],
    diary_write_for_run: Callable[[Any], Optional[Callable[..., Any]]],
    rescue_dir_for_run: Optional[Callable[[Any], Any]] = None,
) -> Callable[..., Any]:
    """Stamp-conditional G1 capture for SHARED runtimes (flow c5342 R1 — the
    Mira diary leak: the flow-brain lane ran LLM_CALL on the base runtime
    where the capture wrap never existed, so elected fences passed through
    unparsed and PRIVATE words rested in ledgers and reached the visitor's
    screen).

    One LLM_CALL handler on a shared runtime serves both stamped entity runs
    and plain workflow runs; the capture boundary must follow the RUN:

    - ``should_capture(run)`` decides per dispatch (gateway wires its
      verified-stamp resolver; an unstamped run gets the BYTE-IDENTICAL
      passthrough — same handler object, same outcome).
    - ``diary_write_for_run(run)`` resolves the stamped run's HOME
      DIARY_WRITE handler (per home, per run — a shared runtime cannot bind
      one book at composition time).

    Exceptions from either callable RAISE (loud): the predicate is the
    gateway's own stamp walk, designed to answer None/False for plain runs —
    a raising predicate is a wiring bug, and failing the effect is strictly
    better than either leaking (silent passthrough on a stamped run) or
    misrouting diary words into the wrong book. Exported so the gateway
    COMPOSES the capture boundary, never hand-rolls it (c5342's ask)."""

    def wrapped(run: Any, effect: Effect, default_next_node: Any = None) -> Any:
        if not should_capture(run):
            return llm_handler(run, effect, default_next_node)
        diary_write = diary_write_for_run(run)
        if diary_write is None:
            raise RuntimeError(
                "conditional G1 capture: should_capture(run) is True but "
                "diary_write_for_run(run) resolved no DIARY_WRITE handler - "
                "a stamped run without its book would leak elected words; refusing loudly"
            )
        # Per-run rescue location (a shared runtime serves many homes): a
        # failed book write saves the reply into THIS run's home before
        # failing (record-everything ruling). Absent = no rescue, failure
        # message stays honest about it. A RAISING resolver degrades to no
        # rescue instead of failing the turn — the rescue is a safety net,
        # and a net must never be the thing that drops the reply.
        rescue_dir = None
        if rescue_dir_for_run is not None:
            try:
                rescue_dir = rescue_dir_for_run(run)
            except Exception:
                print("#FALLBACK conditional capture: rescue_dir_for_run raised; continuing without rescue")
                rescue_dir = None
        return wrap_llm_handler_with_act_only(
            llm_handler, diary_write_handler=diary_write, rescue_dir=rescue_dir
        )(run, effect, default_next_node)

    return wrapped


def wrap_llm_handler_with_act_only(
    llm_handler: Callable[..., Any],
    *,
    diary_write_handler: Optional[Callable[..., Any]] = None,
    rescue_dir: Any = None,
    **_retired: Any,
) -> Callable[..., Any]:
    """Wrap a host LLM_CALL handler with the WRITE-boundary capture: the
    reply's ```diary fences are captured before the result persists —
    words fly to the book, the durable result keeps the MARKED reply +
    word-free metadata. Elections with NO turn_id in the payload fail
    LOUD and non-retryable (authoring bug; silent loss and leak are both
    worse).

    The READ half (send-time ref dereference) is DELETED per laurent's A
    ruling — see the module docstring. Retired kwargs
    (diary_read_handler, diary_list_resolver) are accepted and ignored so
    older composition code fails soft with a loud warning, not an import
    crash; new code should stop passing them."""
    if _retired:
        # The docstring promises a LOUD warning; make it true (adversary C5).
        print(
            "#FALLBACK wrap_llm_handler_with_act_only ignoring retired kwargs "
            f"{sorted(_retired)} (the read-half was deleted, A ruling); stop passing them"
        )

    def wrapped(run: Any, effect: Effect, default_next_node: Any = None) -> Any:
        payload = effect.payload or {}
        outcome = llm_handler(run, effect, default_next_node)
        warnings: List[str] = []
        if (
            diary_write_handler is not None
            and getattr(outcome, "status", None) == "completed"
            and isinstance(outcome.result, dict)
            and isinstance(outcome.result.get("content"), str)
            # CASE-INSENSITIVE gate (adversary find 3): the fence parser is
            # IGNORECASE, so a ```Diary fence missing a lowercase substring
            # check would skip capture and rest the raw private words in
            # run vars + ledger while silently losing the book write.
            and "```diary" in outcome.result["content"].lower()
        ):
            try:
                marked, entries, capture_warnings = capture_diary_elections(
                    outcome.result["content"],
                    run=run,
                    turn_id=str(payload.get("turn_id") or ""),
                    diary_write_handler=diary_write_handler,
                    anchor_record_ids=payload.get("anchor_record_ids"),
                    anchor_graph_ids=payload.get("anchor_graph_ids"),
                    rescue_dir=rescue_dir,
                )
            except ActOnlyResolutionError as e:
                # Losing elected words silently or leaking them are both
                # worse than a loud deterministic failure (authoring bug).
                return EffectOutcome.failed(
                    f"diary election capture failed: {e}", retryable=False
                )
            new_result = {**outcome.result, "content": marked}
            if entries:
                new_result["diary_entries"] = entries
            warnings.extend(capture_warnings)
            # SIBLING-KEY hygiene (wave-4 P0, AMENDED by the simplicity
            # audit + laurent's Q2 ruling "verbatim is verbatim",
            # 2026-07-19): raw_response is DROPPED whenever the reply
            # carried a diary fence — it is the provider wire envelope
            # duplicating the unmarked reply byte-for-byte; dropping it
            # loses no thought. `reasoning` is DELIBERATELY LEFT VERBATIM:
            # the earlier fence-strip rewrote the entity's own thought at
            # rest and died under the ruling (diary is his experiential
            # notes, not a privacy regime over his reasoning; his thoughts
            # must always be readable by him).
            if marked != outcome.result.get("content"):
                if new_result.get("raw_response") is not None:
                    new_result["raw_response"] = None
                    warnings.append(
                        "#NOTE raw_response dropped (carried the unmarked reply; "
                        "diary words rest only in the book)"
                    )
            outcome = EffectOutcome.completed(new_result)

        if warnings and getattr(outcome, "status", None) == "completed" and isinstance(outcome.result, dict):
            outcome = EffectOutcome.completed({**outcome.result, "act_only_warnings": list(warnings)})
        return outcome

    return wrapped
