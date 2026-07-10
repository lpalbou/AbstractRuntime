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
4. FAILURE IS LOUD AND DETERMINISTIC (agency line 1): a ref that does not
   resolve fails the effect, classified NON-RETRYABLE (the same ref
   resolves the same way against an append-only book — retrying burns
   attempts for an identical failure). Never a silent gist substitution;
   never unresolved ref JSON shipped as degraded payload.

Resolution AUTHORITY comes from composition, not from this module: the
resolver is the run's own DIARY_READ handler — raw at the home-direct
driver, stamp-verified behind the gateway door (`install_entity_routing`),
so "under the run's verified stamp" holds by construction wherever the
door wired the handlers.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.models import Effect, EffectType
from ..core.runtime import EffectOutcome

ACT_ONLY_KEY = "$act_only"

# v1's one act-only tool. Widening this set is a spec change (the attribute
# is declared on the tool contract, core's `act_only` field); an unknown
# tool in a ref fails loudly rather than guessing a resolver.
ACT_ONLY_TOOLS = ("diary_read",)


class ActOnlyResolutionError(RuntimeError):
    """A ref did not resolve — loud, deterministic, non-retryable."""


def make_act_only_content(
    *, tool: str, entry_id: str, reason: str = "", gist: str = ""
) -> str:
    """The canonical ref JSON for a durable tool message's content.

    `gist` must be the BOUNDED one-line gist the diary door authored —
    never an excerpt (memory's freeze bound; the door is the author, no
    caller may slice words into it)."""
    ref: Dict[str, Any] = {"tool": str(tool), "entry_id": str(entry_id)}
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


def dereference_act_only_messages(
    messages: List[Any], *, read_entry: Callable[[Dict[str, Any]], str]
) -> Tuple[List[Any], int]:
    """Return (wire_copy, resolved_count). The input list and its messages
    are never mutated; non-ref messages pass through by reference. Refs on
    non-tool roles stay inert text (mechanic 2). `read_entry(ref)` returns
    the resolved words or raises ActOnlyResolutionError."""
    out: List[Any] = []
    resolved = 0
    for msg in messages:
        if not isinstance(msg, dict) or str(msg.get("role") or "") != "tool":
            out.append(msg)
            continue
        ref = parse_act_only_ref(msg.get("content"))
        if ref is None:
            out.append(msg)
            continue
        tool = str(ref.get("tool") or "")
        if tool not in ACT_ONLY_TOOLS:
            raise ActOnlyResolutionError(
                f"unknown act-only tool {tool!r} in ref (known: {', '.join(ACT_ONLY_TOOLS)}) - "
                "refusing to guess a resolver"
            )
        words = read_entry(ref)
        resolved += 1
        # In-place substitution on the copy: content only; identity untouched.
        out.append({**msg, "content": words})
    return out, resolved


def wrap_llm_handler_with_act_only(
    llm_handler: Callable[..., Any], *, diary_read_handler: Callable[..., Any]
) -> Callable[..., Any]:
    """Wrap a host LLM_CALL handler with send-time dereference.

    The wrapper is a pass-through (same effect object, zero copies) when no
    ref is present; with refs it builds a NEW effect whose payload carries
    the resolved wire messages — the original effect (what the ledger and
    run state hold) keeps the refs. The resolver is the run's DIARY_READ
    handler, so door postures (raw vs stamp-verified) apply unchanged."""

    def wrapped(run: Any, effect: Effect, default_next_node: Any = None) -> Any:
        payload = effect.payload or {}
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return llm_handler(run, effect, default_next_node)

        def read_entry(ref: Dict[str, Any]) -> str:
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

        try:
            wire_messages, resolved = dereference_act_only_messages(messages, read_entry=read_entry)
        except ActOnlyResolutionError as e:
            # Loud + deterministic: same ref, same failure — never retried,
            # never degraded into sending the raw ref JSON to the provider.
            return EffectOutcome.failed(f"act-only dereference failed: {e}", retryable=False)
        if resolved == 0:
            return llm_handler(run, effect, default_next_node)
        wire_effect = Effect(
            type=effect.type,
            payload={**payload, "messages": wire_messages},
            result_key=effect.result_key,
        )
        return llm_handler(run, wire_effect, default_next_node)

    return wrapped
