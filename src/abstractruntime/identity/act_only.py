"""Act-only content: references at rest, words only in flight (G1).

Frozen seam spec (a2a/threads/0013-visit-seam-spec, v2 072711Z + freeze
addendum 073846Z; ruled two-of-two with memory, co-signed by agency/agent/
gateway/core): when an entity reads its own diary mid-visit, the words may
never REST outside the book — not in `run.vars`, not in ledger records, not
in rebuilt prompts. The durable transcript carries a typed REFERENCE; the
words are dereferenced fresh from the book at SEND time, into the wire copy
only.

The four frozen mechanics this module implements:

1. THE REF RIDES CONTENT AS EXACT JSON (agent fact 1): the durable tool
   message's `content` is one JSON object with a single top-level
   `"$act_only"` key. Detection is PARSE, never regex: `json.loads`, then
   the lone-key check. Anything else — prose, arrays, extra keys — is not
   a ref.
2. TOOL MESSAGES ONLY: refs are honored on `role == "tool"` messages, which
   only the host's observe step appends. A visitor pasting `$act_only`-
   looking JSON into their message stays INERT TEXT — payloads are never
   authoritative; a ref must not be a capability a visitor can type.
3. IN-PLACE SUBSTITUTION ON A WIRE COPY (agent fact 2): resolution replaces
   the tool message's content STRING and touches nothing else — role,
   tool_call_id, position fixed (message identity is load-bearing for the
   orphan-repair and alternation disciplines). The ORIGINAL payload object
   is never mutated: the ledger keeps the ref (contrast: the grounding
   injector mutates pre-ledger BY DESIGN so the ledger shows what was sent;
   act-only inverts that deliberately — the ledger must NOT show the words;
   two documented boundary behaviors).
4. FAILURE IS LOUD, DETERMINISTIC, AND SURVIVABLE (amended 2026-07-10 on
   agent's wedge finding, 0013/080636Z): refs are DURABLE — a transcript
   message survives every later turn — and a failed effect is a TERMINAL
   run, so "fail the effect" turned one bad historical ref into a
   permanently dead visit (every resume re-failed identically; the visit
   could never speak again). Against an APPEND-ONLY book a well-authored
   ref can never dangle, so an unresolvable ref is always a BUG
   (malformed authoring, cross-home composition, book damage) — and the
   failure-direction principle (memory's own G1 argument) applies: a bug
   must surface loudly without killing the life-session. An unresolvable
   ref therefore resolves to a LABELED TOMBSTONE
   (`[act-only content unavailable: …]`) and the call carries a loud
   `#FALLBACK` warning into the result (ledger- and vars-visible). The
   original guarantees hold unchanged: never silent, never a retry burn
   (deterministic), never raw ref JSON on the wire, and the words never
   leak (a tombstone names the entry id and failure class — exactly what
   the old error string exposed).

Resolution AUTHORITY comes from composition, not from this module: the
resolver is the run's own DIARY_READ handler — raw at the home-direct
driver, stamp-verified behind the gateway door (`install_entity_routing`),
so "under the run's verified stamp" holds by construction wherever the
door wired the handlers.

THE LEDGER CONTRACT, both divergences in one place (memory's co-sign note,
a2a 0014/094344Z): the run ledger records LLM_CALL payloads and results as
the RUNTIME held them, which differs from the provider wire in exactly two
documented, deterministic ways — READ: act-only refs appear UNRESOLVED in
the ledgered payload (the wire carried the dereferenced words; media
`$artifact` refs behave identically); WRITE: the ledgered result carries
the MARKED reply (the wire's raw reply carried the diary fences, which
flew to the book at the result boundary and never rested). Reconstruction
re-resolves refs through doors that enforce authority; nothing else about
the payload/result diverges. The non-private gist fallback (first line of
the entry text, ≤120 chars, when the author gave no gist) is a DECISION on
the record — gist-grade under the operator-audience ruling — not an
accident of truncation.

CONSUMER RULE (code seat's C2, incident c2447): ledgered LLM_CALL payloads
are the WIRE VIEW — they legitimately contain adapter-authored prompt
chrome (loop-iteration tails, plan renders, retry nudges merged into
message copies), and volatile markers are stripped before capture, so the
chrome is NOT labeled in the record. Ledger payloads are OBSERVABILITY;
conversation rebuilds must come from the durable transcript
(context.messages / _visit.history), never from replaying ledger payloads
into prompts — a ledger-payload history consumer would re-inject chrome
the live lanes have since suppressed.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.models import Effect, EffectType
from ..core.runtime import EffectOutcome

ACT_ONLY_KEY = "$act_only"

# The act-only tool set — DERIVED from the descriptor registry (one source:
# each ToolDescriptor declares `act_only`; the wire flag and the resolver
# dispatch can never disagree). Widening the set is a privacy-lane ruling
# expressed ON THE DESCRIPTOR, never an edit here; an unknown tool in a ref
# still fails loudly rather than guessing a resolver.
#
# diary_list JOINED 2026-07-11 (memory's e-s 233 R3 ruling, adversary-
# broken claim: a private entry's GIST is part of the private words, and
# _run_diary_list emits every entry's gist — private included — so one
# granted diary_list call in a durable visit RESTED private gists in the
# run store/ledger). Its ref carries tool+args (no entry_id — the listing
# is re-run fresh at send time), the tool+args generalization memory's
# ruling named.
from .tools import TOOL_DESCRIPTORS as _TOOL_DESCRIPTORS

ACT_ONLY_TOOLS = tuple(n for n, d in _TOOL_DESCRIPTORS.items() if d.act_only)
assert ACT_ONLY_TOOLS == ("diary_list", "diary_read"), (
    "act-only derivation drifted from the ruled set (e-s 233 R3); a widening "
    f"must be a privacy-lane ruling: {ACT_ONLY_TOOLS}"
)


class ActOnlyResolutionError(RuntimeError):
    """A ref did not resolve — loud, deterministic, non-retryable."""


def make_act_only_content(
    *,
    tool: str,
    entry_id: str = "",
    reason: str = "",
    gist: str = "",
    args: Optional[Dict[str, Any]] = None,
) -> str:
    """The canonical ref JSON for a durable tool message's content.

    Two authoring shapes, one envelope (the tool+args generalization,
    e-s 233 R3): entry-addressed tools (diary_read) carry `entry_id`;
    re-run tools (diary_list) carry `args` (JSON-safe, word-free — the
    listing is re-executed fresh at send time, never stored). `gist` must
    be the BOUNDED one-line gist the diary door authored — never an
    excerpt (memory's freeze bound; the door is the author, no caller may
    slice words into it)."""
    ref: Dict[str, Any] = {"tool": str(tool)}
    if entry_id:
        ref["entry_id"] = str(entry_id)
    if args is not None:
        ref["args"] = dict(args)
    if reason:
        ref["reason"] = str(reason)
    if gist:
        ref["gist"] = str(gist)
    return json.dumps({ACT_ONLY_KEY: ref}, ensure_ascii=False)


def parse_act_only_ref(content: Any) -> Optional[Dict[str, Any]]:
    """Parse-not-regex detection: exactly one JSON object whose ONLY
    top-level key is `$act_only` with a dict value. Everything else is
    ordinary content (returns None)."""
    if not isinstance(content, str) or ACT_ONLY_KEY not in content:
        return None  # cheap pre-check; the substring alone never decides
    try:
        data = json.loads(content)
    except Exception:  # noqa: BLE001 - not JSON = not a ref
        return None
    if not isinstance(data, dict) or set(data.keys()) != {ACT_ONLY_KEY}:
        return None
    ref = data[ACT_ONLY_KEY]
    return ref if isinstance(ref, dict) else None


def tombstone_content(ref: Dict[str, Any], failure: str) -> str:
    """The survivable substitution for a ref that cannot resolve — labeled,
    deterministic, word-free. Names the address (entry id for read-shaped
    refs, the word-free args for re-run-shaped refs) and failure class —
    the same facts the old fatal error exposed — so the model AND the
    operator see honestly that content is missing; the words stay in the
    book."""
    tool = str(ref.get("tool") or "").strip() or "<no tool>"
    entry_id = str(ref.get("entry_id") or "").strip()
    if not entry_id and isinstance(ref.get("args"), dict):
        entry_id = json.dumps(ref["args"], ensure_ascii=False, sort_keys=True)
    entry_id = entry_id or "<no address>"
    return (
        f"[act-only content unavailable: {tool} {entry_id} - {failure}; "
        "the words remain in the book - re-elect the read if you need them]"
    )


def dereference_act_only_messages(
    messages: List[Any],
    *,
    read_entry: Callable[[Dict[str, Any]], str],
    warnings: Optional[List[str]] = None,
) -> Tuple[List[Any], int]:
    """Return (wire_copy, substituted_count). The input list and its
    messages are never mutated; non-ref messages pass through by reference.
    Refs on non-tool roles stay inert text (mechanic 2).

    `read_entry(ref)` returns the resolved words or raises
    ActOnlyResolutionError. A failed resolution SUBSTITUTES A LABELED
    TOMBSTONE instead of raising out (mechanic 4, the wedge amendment):
    refs are durable, so a fatal failure here would re-fail every later
    call in the run — one bad historical ref must degrade one message,
    never kill the life-session. Each failure appends a loud `#FALLBACK`
    line to `warnings` (caller-supplied accumulator; the wrapper rides it
    into the effect result so the ledger shows the degradation)."""
    out: List[Any] = []
    substituted = 0
    for msg in messages:
        if not isinstance(msg, dict) or str(msg.get("role") or "") != "tool":
            out.append(msg)
            continue
        ref = parse_act_only_ref(msg.get("content"))
        if ref is None:
            out.append(msg)
            continue
        tool = str(ref.get("tool") or "")
        try:
            if tool not in ACT_ONLY_TOOLS:
                raise ActOnlyResolutionError(
                    f"unknown act-only tool {tool!r} (known: {', '.join(ACT_ONLY_TOOLS)}) - "
                    "refusing to guess a resolver"
                )
            content = read_entry(ref)
        except ActOnlyResolutionError as e:
            content = tombstone_content(ref, str(e))
            if warnings is not None:
                warnings.append(f"#FALLBACK act-only ref did not resolve: {e}")
        except Exception as e:  # noqa: BLE001 - adversary find 2 (2026-07-11):
            # a RAISED resolver/store error (sqlite failure, resolver bug)
            # must get the same survivability as a structured refusal —
            # letting it escape fails the LLM effect and terminal-FAILs the
            # visit, the exact wedge class mechanic 4 exists to prevent.
            content = tombstone_content(ref, f"resolver error: {e}")
            if warnings is not None:
                warnings.append(f"#FALLBACK act-only resolver raised: {e}")
        substituted += 1
        # In-place substitution on the copy: content only; identity untouched.
        out.append({**msg, "content": content})
    return out, substituted


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
        meta: Dict[str, Any] = {
            "entry_id": str(result.get("entry_id") or ""),
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
    diary_read_handler: Callable[..., Any],
    diary_write_handler: Optional[Callable[..., Any]] = None,
    diary_list_resolver: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Callable[..., Any]:
    """Wrap a host LLM_CALL handler with BOTH G1 directions: send-time
    dereference (reads) and result-boundary election capture (writes).

    The wrapper is a pass-through (same effect object, zero copies) when no
    ref is present; with refs it builds a NEW effect whose payload carries
    the resolved wire messages — the original effect (what the ledger and
    run state hold) keeps the refs. The resolver is the run's DIARY_READ
    handler, so door postures (raw vs stamp-verified) apply unchanged.

    Unresolvable refs substitute a LABELED TOMBSTONE (never the raw ref
    JSON, never the run's death — the wedge amendment): the call proceeds
    with the degradation visible in the wire text AND a `#FALLBACK`
    warning list on the effect result (`act_only_warnings`), so ledgers
    and probes show it loudly while the visit stays alive.

    WRITE DIRECTION (when `diary_write_handler` is wired): the reply's
    ```diary fences are captured HERE, at the result boundary, BEFORE the
    result persists anywhere — the words fly to the book through the
    home's DIARY_WRITE handler and the durable result carries the MARKED
    reply plus word-free `diary_entries` metadata (entry_id, kind,
    visibility, projected_record_id; gist for non-private only). Without
    this, the raw reply would rest in run.vars (result_key) and the
    ledger's LLM_CALL result — the A/B privacy grep caught exactly that.
    The payload keys `turn_id` / `anchor_record_ids` / `anchor_graph_ids`
    (word-free) parameterize the book write; elections with NO turn_id in
    the payload fail LOUD and non-retryable (a workflow authoring bug —
    losing elected words silently and leaking them are both worse)."""

    def wrapped(run: Any, effect: Effect, default_next_node: Any = None) -> Any:
        payload = effect.payload or {}
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return llm_handler(run, effect, default_next_node)

        def read_entry(ref: Dict[str, Any]) -> str:
            # Dispatch by ref shape (e-s 233 R3 generalization): diary_read
            # dereferences ONE entry through the run's DIARY_READ handler;
            # diary_list RE-RUNS the listing fresh at send time through the
            # host-wired resolver (the listing — private gists included for
            # the entity's own eyes — exists only in the wire copy).
            tool = str(ref.get("tool") or "")
            if tool == "diary_list":
                if diary_list_resolver is None:
                    raise ActOnlyResolutionError(
                        "diary_list ref present but no diary_list_resolver is wired "
                        "on this runtime (host composition gap)"
                    )
                listing = str(diary_list_resolver(ref) or "")
                if not listing.strip():
                    raise ActOnlyResolutionError("diary_list resolved to no content")
                return f"[diary_list - resolved from the book at send time]\n{listing}"
            entry_id = str(ref.get("entry_id") or "").strip()
            if not entry_id:
                raise ActOnlyResolutionError("act-only ref carries no entry_id")
            out = diary_read_handler(
                run, Effect(type=EffectType.DIARY_READ, payload={"entry_id": entry_id}), None
            )
            if getattr(out, "status", None) != "completed":
                raise ActOnlyResolutionError(
                    f"the book refused ref {entry_id!r}: {getattr(out, 'error', 'unknown error')}"
                )
            text = str((out.result or {}).get("text") or "")
            if not text.strip():
                raise ActOnlyResolutionError(f"entry {entry_id!r} resolved to no words")
            # Deterministic v1 resolved form: an act-frame header + the words.
            return f"[diary_read {entry_id} - resolved from the book at send time]\n{text}"

        warnings: List[str] = []
        wire_messages, substituted = dereference_act_only_messages(
            messages, read_entry=read_entry, warnings=warnings
        )
        if substituted == 0:
            wire_effect = effect
        else:
            wire_effect = Effect(
                type=effect.type,
                payload={**payload, "messages": wire_messages},
                result_key=effect.result_key,
            )
        outcome = llm_handler(run, wire_effect, default_next_node)

        # WRITE DIRECTION: capture diary elections from the reply BEFORE the
        # result persists (result_key/ledger). Words fly to the book now;
        # only the marked reply + word-free metadata rest.
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
            outcome = EffectOutcome.completed(new_result)

        if warnings and getattr(outcome, "status", None) == "completed" and isinstance(outcome.result, dict):
            # Ride the degradation into the DURABLE result (ledger + vars):
            # a tombstoned turn must be loud everywhere, not just in prose.
            outcome = EffectOutcome.completed({**outcome.result, "act_only_warnings": list(warnings)})
        return outcome

    return wrapped
