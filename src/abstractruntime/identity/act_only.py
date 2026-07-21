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
        # into the ledger). Deterministic authoring bug — loud.
        raise ActOnlyResolutionError(
            "diary election present but the LLM_CALL payload carries no turn_id - "
            "the workflow must pass turn_id so the book write is replay-safe"
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
            raise ActOnlyResolutionError(
                f"the book refused a diary election: {getattr(out, 'error', 'unknown error')}"
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


def wrap_llm_handler_with_act_only(
    llm_handler: Callable[..., Any],
    *,
    diary_write_handler: Optional[Callable[..., Any]] = None,
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
