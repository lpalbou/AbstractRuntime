"""abstractruntime.storage.ledger_slim

Terminal-record slimming for the append-only ledger (backlog 0067-M).

The ledger persists the same conversation-sized bytes several times per
effectful step: the STARTED record embeds the full effect payload, the
terminal record (COMPLETED / WAITING / FAILED) is the same record
re-appended with the payload still attached, and an LLM result carries
observability copies of the request (`metadata._provider_request`,
`metadata._runtime_observability.llm_generate_kwargs`). Measured on a
120-message conversation this triplication costs ~540 KB of ledger per
LLM effect and makes ledger growth O(turns^2) in bytes.

This module removes the DUPLICATE copies only — never the originals:

- `slim_terminal_effect(...)`: on the terminal append, any top-level
  payload field whose compact JSON exceeds the threshold is replaced by a
  `$slim` marker naming the STARTED record (same `step_id`) that already
  holds the bytes. Sub-threshold fields stay inline so ledger consumers
  (UIs, stats) keep reading small payloads (tool args, flags) directly.
- `slim_result_metadata(...)`: the two known observability copies inside
  an LLM result are replaced by markers ONLY when they are provably
  reconstructable from the STARTED payload — byte-verified at slim time
  (sha256 over compact JSON). Anything that differs from the effect
  payload (decorated wire messages, grounding envelopes) is kept
  verbatim: the `_provider_request` capture is the "durable bytes equal
  sent bytes" lane and is never approximated.

Resolution is deterministic and verified: markers rebuild from the
STARTED record and check the recorded sha256 — a mismatch keeps the
marker in place (loud, never a silently-wrong payload).

Two hard-won rules from the 2026-07-14 adversary pass:

- SLIMMING IS ANCHORED TO STARTED-TIME DIGESTS
  (`capture_started_payload_digests`, taken immediately before the
  STARTED append): non-LLM records hold the payload BY REFERENCE, so a
  handler mutating an oversized field in place between the appends would
  otherwise mint a marker whose bytes exist NOWHERE (the sha could never
  match the STARTED reconstruction). A field that diverged from its
  digest keeps its verbatim mutated bytes — slimming drops duplicates,
  never information.
- REHYDRATION IS TARGETED, NEVER WHOLE-TREE
  (`resolve_result_metadata_markers` on the crash-replay path): a tool
  result can echo a slimmed record as DATA, and a whole-tree resolve
  would "resolve" the echo — diverging replayed vars from the live path.
  Only the paths this module writes are rehydrated; `resolve_slim_tree`
  stays exported as a generic tool for host consumers who know their
  trees carry no marker-shaped data.

Copy discipline: slimming builds SPINE COPIES (new dicts along the
changed paths, values by reference). The live result object that flows
into `run.vars[result_key]` is never mutated — the run-state copy keeps
full fidelity for consumers like `/llm --verbatim`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

SLIM_MARKER_KEY = "$slim"
SLIM_MARKER_VERSION = 1

# Per-field compact-JSON byte gate. Conversation payloads (the O(turns^2)
# term) are far above it; tool args / flags / small prompts stay inline.
SLIM_FIELD_THRESHOLD_BYTES = 4096

# Lower gate for the KNOWN conversation-shaped fields. The replay-integrity
# audit (2026-07-25, code-tui incident) measured a 3,918-byte system prompt —
# 178B UNDER the general threshold — duplicating on every one of 32 calls in
# a session. A marker costs ~250B, so deduplicating these named fields is
# profitable well below 4096; arbitrary fields keep the conservative gate.
SLIM_CONVERSATION_FIELD_FLOOR_BYTES = 512
_CONVERSATION_FIELDS = frozenset({"messages", "system_prompt", "prompt"})


def _field_threshold(field: Optional[str]) -> int:
    if field is not None and str(field) in _CONVERSATION_FIELDS:
        return SLIM_CONVERSATION_FIELD_FLOOR_BYTES
    return SLIM_FIELD_THRESHOLD_BYTES


# Appendix bounds (appendix-aware dedup, 2026-07-25): extras rest VERBATIM
# inside the marker, so they must stay few and small — the measured case is
# ONE ~350B system message appended after payload build. The dedup must also
# stay profitable: total appendix bytes may never exceed half the value.
SLIM_APPENDIX_MAX_ITEMS = 8
SLIM_APPENDIX_MAX_ITEM_BYTES = 4096

# Marker kinds:
# - "started_payload_field": value is byte-identical to
#   `started.effect.payload[<field>]` of the record named by `step_id`.
# - "started_messages_layout": value is the provider message list rebuilt
#   from the STARTED payload as
#   [system message?] + payload.messages + [user prompt message?]
#   in the order given by `layout` (subset of ["system","messages","prompt"]).
_KIND_FIELD = "started_payload_field"
_KIND_LAYOUT = "started_messages_layout"

# Result-metadata copies eligible for same-name field dedup against the
# effect payload (the `_runtime_observability` capture mirrors the
# handler's generate kwargs, which come FROM the payload).
_OBSERVABILITY_DEDUP_FIELDS = ("messages", "system_prompt", "tools", "media", "prompt")


def _compact(value: Any) -> Optional[str]:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    except Exception:
        return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_slim_marker(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get(SLIM_MARKER_KEY), dict)


def _make_marker(
    *,
    kind: str,
    step_id: str,
    compact_text: str,
    field: Optional[str] = None,
    layout: Optional[List[str]] = None,
    appendix: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "v": SLIM_MARKER_VERSION,
        "kind": kind,
        "step_id": str(step_id),
        "sha256": _sha256(compact_text),
        "bytes": len(compact_text.encode("utf-8")),
    }
    if field is not None:
        body["field"] = str(field)
    if layout is not None:
        body["layout"] = list(layout)
    if appendix:
        # Extras carried VERBATIM (positions + items): the reconstruction is
        # base-from-STARTED with each item inserted at its recorded index.
        # The sha over the FULL value keeps correctness structural — a bad
        # appendix can only fail resolution, never fabricate bytes.
        body["appendix"] = [dict(a) for a in appendix]
    return {SLIM_MARKER_KEY: body}


def _subsequence_extras(value: List[Any], base: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """Match `base` as an ordered subsequence of `value`; return the extras.

    Returns [{"at": <index in value>, "item": <verbatim element>}] for every
    element of `value` that is not part of the match, or None when `base` is
    not a subsequence. Elements compare by compact-JSON bytes. Greedy
    first-match is sufficient: any successful alignment reconstructs to the
    same byte sequence, and the marker's sha over the full value verifies
    the reconstruction regardless of which alignment was recorded.

    This is the appendix-aware dedup core (replay-integrity audit,
    2026-07-25): an adapter appending ONE ~350B message AFTER the payload is
    built used to defeat byte-identity for the whole ~250KB metadata copy —
    0-of-59 dedup hits on the incident bundles.
    """
    base_compact = [_compact(b) for b in base]
    if any(c is None for c in base_compact):
        return None
    extras: List[Dict[str, Any]] = []
    bi = 0
    for vi, item in enumerate(value):
        item_compact = _compact(item)
        if item_compact is None:
            return None
        if bi < len(base_compact) and item_compact == base_compact[bi]:
            bi += 1
            continue
        if len(extras) >= SLIM_APPENDIX_MAX_ITEMS:
            return None
        if len(item_compact.encode("utf-8")) > SLIM_APPENDIX_MAX_ITEM_BYTES:
            return None
        extras.append({"at": vi, "item": item})
    if bi != len(base_compact):
        return None
    return extras


def capture_started_payload_digests(effect: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Digest the oversized payload fields as the STARTED record persists them.

    Called immediately before the STARTED append; the returned
    {field: sha256-of-compact-json} map is the byte truth the terminal
    slimming verifies against (2026-07-14 adversary P1-1: for non-LLM
    effects the record holds the payload BY REFERENCE and a handler that
    mutates an oversized field in place between the appends would have made
    an unverified marker permanently unresolvable — with the execution-time
    bytes existing NOWHERE). Only fields above the threshold are digested
    (sub-threshold fields never slim, so their digests would be waste).
    """
    digests: Dict[str, str] = {}
    if not isinstance(effect, dict):
        return digests
    payload = effect.get("payload")
    if not isinstance(payload, dict):
        return digests
    for key, value in payload.items():
        if is_slim_marker(value) or value is None or isinstance(value, (bool, int, float)):
            continue
        compact_text = _compact(value)
        if compact_text is None:
            continue
        # Layout-relevant small fields are digested at ANY size: the result-
        # metadata dedup reconstructs provider messages from them, so their
        # freshness must be verifiable even when they are tiny.
        if len(compact_text.encode("utf-8")) > SLIM_FIELD_THRESHOLD_BYTES or key in ("system_prompt", "prompt", "messages"):
            digests[str(key)] = _sha256(compact_text)
    return digests


def _payload_field_unchanged(payload: Dict[str, Any], field: str, started_digests: Optional[Dict[str, str]]) -> bool:
    """True when the payload field's bytes still match the STARTED digest."""
    if not started_digests or field not in started_digests:
        return False
    compact_text = _compact(payload.get(field))
    return compact_text is not None and _sha256(compact_text) == started_digests[field]


def slim_terminal_effect(
    effect: Optional[Dict[str, Any]],
    *,
    step_id: str,
    started_digests: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a slimmed copy of a terminal record's effect dict.

    Top-level payload fields above the threshold are replaced by markers
    referencing the STARTED record (same `step_id`, appended before
    execution by construction) — but ONLY when the field's bytes still
    match the STARTED-time digest (`started_digests` from
    `capture_started_payload_digests`). A field mutated in place during
    execution keeps its VERBATIM mutated bytes (the pre-slimming truth) —
    slimming may drop duplicates, never information. When no digest map is
    supplied (host-driven appends outside the retry loop), nothing slims.

    Returns the ORIGINAL object when nothing qualifies, a spine copy
    otherwise — the caller may assign it to the record without touching the
    live `Effect.payload`.
    """
    if not isinstance(effect, dict):
        return effect
    if not started_digests:
        return effect
    payload = effect.get("payload")
    if not isinstance(payload, dict) or not payload:
        return effect

    slimmed_payload: Optional[Dict[str, Any]] = None
    for key, value in payload.items():
        if is_slim_marker(value):
            continue
        if value is None or isinstance(value, (bool, int, float)):
            continue
        expected_sha = started_digests.get(str(key))
        if expected_sha is None:
            continue
        compact_text = _compact(value)
        if compact_text is None or len(compact_text.encode("utf-8")) <= _field_threshold(str(key)):
            continue
        if _sha256(compact_text) != expected_sha:
            # Mutated between the appends: the terminal record is the only
            # holder of the execution-time bytes — keep them.
            continue
        if slimmed_payload is None:
            slimmed_payload = dict(payload)
        slimmed_payload[key] = _make_marker(
            kind=_KIND_FIELD, step_id=step_id, compact_text=compact_text, field=str(key)
        )

    if slimmed_payload is None:
        return effect
    out = dict(effect)
    out["payload"] = slimmed_payload
    return out


def _layout_reconstruction(payload: Dict[str, Any], layout: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Rebuild a provider message list from a STARTED effect payload.

    Mirrors the local client's fabricated `_provider_request` builder:
    optional system message, the payload messages (dict-copied), and an
    optional trailing user prompt message.
    """
    out: List[Dict[str, Any]] = []
    for part in layout:
        if part == "system":
            sp = payload.get("system_prompt")
            if not isinstance(sp, str) or not sp:
                return None
            out.append({"role": "system", "content": sp})
        elif part == "messages":
            msgs = payload.get("messages")
            if not isinstance(msgs, list):
                return None
            out.extend([dict(m) for m in msgs if isinstance(m, dict)])
        elif part == "prompt":
            pr = payload.get("prompt")
            if not isinstance(pr, str) or not pr:
                return None
            out.append({"role": "user", "content": pr})
        else:
            return None
    return out


_LAYOUT_PART_FIELDS = {"system": "system_prompt", "messages": "messages", "prompt": "prompt"}


def _try_dedup_value(
    value: Any,
    *,
    payload: Dict[str, Any],
    step_id: str,
    started_digests: Optional[Dict[str, str]] = None,
    same_name_field: Optional[str] = None,
    allow_layout: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return a marker for `value` when it is provably reconstructable, else None.

    Provably = byte-identical to the payload NOW *and* the payload field(s)
    byte-identical to what the STARTED record persisted (`started_digests`)
    — an in-place payload mutation during execution otherwise made a marker
    whose sha can never match the STARTED reconstruction, losing the
    result's copy (adversary P1-1, metadata half).

    Appendix-aware (2026-07-25): when exact byte-identity fails but the
    reference is an ordered SUBSEQUENCE of the value (an adapter inserted a
    few small messages after the payload was built), the marker carries the
    extras verbatim with their positions — dedup still drops only the bytes
    the STARTED record provably holds.
    """
    compact_text = _compact(value)
    if compact_text is None or len(compact_text.encode("utf-8")) <= _field_threshold(same_name_field):
        return None
    value_bytes = len(compact_text.encode("utf-8"))

    def _appendix_ok(extras: List[Dict[str, Any]]) -> bool:
        if not extras:
            return True
        appendix_compact = _compact(extras)
        if appendix_compact is None:
            return False
        # Profitability floor: never mint a marker that mostly re-carries
        # the value it claims to dedup.
        return len(appendix_compact.encode("utf-8")) <= value_bytes // 2

    # Match order prefers reconstructions that carry the FEWEST verbatim
    # bytes in the marker: exact same-name, exact layout, layout+appendix,
    # same-name+appendix last (a provider list is [system?]+messages+[prompt?]
    # — the layout kind rebuilds system/prompt from STARTED for free, so the
    # same-name path must never pre-empt it by carrying them as appendix).
    same_name_ref: Any = None
    if same_name_field is not None and _payload_field_unchanged(payload, same_name_field, started_digests):
        same_name_ref = payload.get(same_name_field)
        ref_text = _compact(same_name_ref)
        if ref_text is not None and ref_text == compact_text:
            return _make_marker(kind=_KIND_FIELD, step_id=step_id, compact_text=compact_text, field=same_name_field)

    layout_appendix: Optional[Dict[str, Any]] = None
    if allow_layout and isinstance(value, list):
        for layout in (["system", "messages", "prompt"], ["system", "messages"], ["messages", "prompt"], ["messages"], ["system", "prompt"], ["prompt"]):
            parts_fresh = all(
                _payload_field_unchanged(payload, _LAYOUT_PART_FIELDS[part], started_digests)
                for part in layout
            )
            if not parts_fresh:
                continue
            rebuilt = _layout_reconstruction(payload, layout)
            if rebuilt is None:
                continue
            rebuilt_text = _compact(rebuilt)
            if rebuilt_text is not None and rebuilt_text == compact_text:
                return _make_marker(kind=_KIND_LAYOUT, step_id=step_id, compact_text=compact_text, layout=layout)
            # Exact miss: remember the FIRST (longest-layout) appendix match,
            # but keep scanning — an exact match on a later layout wins.
            if layout_appendix is None and rebuilt:
                extras = _subsequence_extras(value, rebuilt)
                if extras is not None and extras and _appendix_ok(extras):
                    layout_appendix = _make_marker(
                        kind=_KIND_LAYOUT,
                        step_id=step_id,
                        compact_text=compact_text,
                        layout=layout,
                        appendix=extras,
                    )
    if layout_appendix is not None:
        return layout_appendix

    if same_name_ref is not None and isinstance(value, list) and isinstance(same_name_ref, list) and same_name_ref:
        extras = _subsequence_extras(value, same_name_ref)
        if extras is not None and extras and _appendix_ok(extras):
            return _make_marker(
                kind=_KIND_FIELD,
                step_id=step_id,
                compact_text=compact_text,
                field=same_name_field,
                appendix=extras,
            )
    return None


def slim_result_metadata(
    result: Optional[Dict[str, Any]],
    *,
    effect_payload: Optional[Dict[str, Any]],
    step_id: str,
    started_digests: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a ledger copy of `result` with duplicate request captures deduped.

    Only the two known observability paths are considered:
    - `metadata._runtime_observability.llm_generate_kwargs.<field>` for
      fields byte-identical to the same-named effect payload field.
    - `metadata._provider_request.payload.messages` when byte-identical to
      the payload messages OR to the documented local reconstruction
      (system + messages + prompt). Decorated wire bytes never match and
      stay verbatim — B3 fidelity by construction.

    Returns the ORIGINAL object when nothing deduped; otherwise a spine
    copy (the caller assigns it to the StepRecord only — never to vars).
    """
    if not isinstance(result, dict) or not isinstance(effect_payload, dict):
        return result
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return result

    new_metadata: Optional[Dict[str, Any]] = None

    def _metadata_copy() -> Dict[str, Any]:
        nonlocal new_metadata
        if new_metadata is None:
            new_metadata = dict(metadata)
        return new_metadata

    obs = metadata.get("_runtime_observability")
    kwargs = obs.get("llm_generate_kwargs") if isinstance(obs, dict) else None
    if isinstance(kwargs, dict):
        new_kwargs: Optional[Dict[str, Any]] = None
        for field in _OBSERVABILITY_DEDUP_FIELDS:
            if field not in kwargs or is_slim_marker(kwargs.get(field)):
                continue
            marker = _try_dedup_value(
                kwargs.get(field),
                payload=effect_payload,
                step_id=step_id,
                started_digests=started_digests,
                same_name_field=field,
            )
            if marker is None:
                continue
            if new_kwargs is None:
                new_kwargs = dict(kwargs)
            new_kwargs[field] = marker
        if new_kwargs is not None:
            new_obs = dict(obs)  # type: ignore[arg-type]
            new_obs["llm_generate_kwargs"] = new_kwargs
            _metadata_copy()["_runtime_observability"] = new_obs

    preq = metadata.get("_provider_request")
    preq_payload = preq.get("payload") if isinstance(preq, dict) else None
    if isinstance(preq_payload, dict):
        msgs = preq_payload.get("messages")
        if msgs is not None and not is_slim_marker(msgs):
            marker = _try_dedup_value(
                msgs,
                payload=effect_payload,
                step_id=step_id,
                started_digests=started_digests,
                same_name_field="messages",
                allow_layout=True,
            )
            if marker is not None:
                new_preq_payload = dict(preq_payload)
                new_preq_payload["messages"] = marker
                new_preq = dict(preq)  # type: ignore[arg-type]
                new_preq["payload"] = new_preq_payload
                _metadata_copy()["_provider_request"] = new_preq

    if new_metadata is None:
        return result
    out = dict(result)
    out["metadata"] = new_metadata
    return out


# ---------------------------------------------------------------------------
# Resolution (readers / crash-replay rehydration)
# ---------------------------------------------------------------------------


def _status_str(raw: Any) -> str:
    """Ledger dicts carry status as a string (JSONL/SQLite) or as the
    StepStatus enum (in-memory asdict); normalize to the value string."""
    value = getattr(raw, "value", None)
    if isinstance(value, str):
        return value
    return raw if isinstance(raw, str) else str(raw or "")


def build_started_payload_index(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map step_id -> STARTED effect payload for marker resolution."""
    index: Dict[str, Dict[str, Any]] = {}
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        if _status_str(rec.get("status")) != "started":
            continue
        step_id = str(rec.get("step_id") or "")
        eff = rec.get("effect")
        payload = eff.get("payload") if isinstance(eff, dict) else None
        if step_id and isinstance(payload, dict):
            # First STARTED wins; retries create fresh step_ids so
            # collisions do not occur in well-formed ledgers.
            index.setdefault(step_id, payload)
    return index


def resolve_slim_value(value: Any, index: Dict[str, Dict[str, Any]]) -> Any:
    """Resolve one `$slim` marker against a STARTED-payload index.

    Verified: the rebuilt value must hash to the recorded sha256; on any
    mismatch or missing source the marker is returned unchanged so a
    consumer never sees a silently-wrong payload.
    """
    if not is_slim_marker(value):
        return value
    body = value[SLIM_MARKER_KEY]
    step_id = str(body.get("step_id") or "")
    payload = index.get(step_id)
    if not isinstance(payload, dict):
        return value

    kind = str(body.get("kind") or "")
    rebuilt: Any = None
    if kind == _KIND_FIELD:
        field = str(body.get("field") or "")
        if field not in payload:
            return value
        rebuilt = payload.get(field)
    elif kind == _KIND_LAYOUT:
        layout = body.get("layout")
        if not isinstance(layout, list):
            return value
        rebuilt = _layout_reconstruction(payload, [str(p) for p in layout])
        if rebuilt is None:
            return value
    else:
        return value

    appendix = body.get("appendix")
    if appendix:
        # Appendix-aware markers: insert the verbatim extras at their
        # recorded positions (ascending — each `at` is the index in the
        # FINAL list, so in-order insertion lands every item exactly).
        if not isinstance(appendix, list) or not isinstance(rebuilt, list):
            return value
        merged = [dict(m) if isinstance(m, dict) else m for m in rebuilt]
        try:
            for entry in sorted(appendix, key=lambda e: int(e.get("at", -1))):
                at = int(entry.get("at"))
                if at < 0 or at > len(merged):
                    return value
                merged.insert(at, entry.get("item"))
        except Exception:
            return value
        rebuilt = merged

    compact_text = _compact(rebuilt)
    if compact_text is None or _sha256(compact_text) != str(body.get("sha256") or ""):
        return value
    return rebuilt


def result_metadata_has_markers(result: Any) -> bool:
    """Marker check restricted to the two paths the runtime itself dedups.

    Crash-replay rehydration must be TARGETED (2026-07-14 adversary P2-2):
    a tool result can echo a slimmed ledger record as DATA (the runtime-
    explore/introspection class), and a whole-tree resolve would "resolve"
    those echoes — making the replayed vars diverge from the live path by
    exactly the bytes the echo carried. Only the paths this module writes
    are ever rehydrated; marker-shaped data anywhere else is data.
    """
    if not isinstance(result, dict):
        return False
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return False
    obs = metadata.get("_runtime_observability")
    kwargs = obs.get("llm_generate_kwargs") if isinstance(obs, dict) else None
    if isinstance(kwargs, dict) and any(is_slim_marker(kwargs.get(f)) for f in _OBSERVABILITY_DEDUP_FIELDS):
        return True
    preq = metadata.get("_provider_request")
    preq_payload = preq.get("payload") if isinstance(preq, dict) else None
    return isinstance(preq_payload, dict) and is_slim_marker(preq_payload.get("messages"))


def resolve_result_metadata_markers(result: Any, index: Dict[str, Dict[str, Any]]) -> Any:
    """Rehydrate ONLY the runtime-written metadata paths (spine copies).

    The inverse of `slim_result_metadata`, path-for-path. Unresolvable
    markers (missing STARTED, sha mismatch) stay in place — loud, never a
    silently-wrong payload. Everything outside the two paths — including
    marker-shaped data a tool echoed — is returned untouched.
    """
    if not result_metadata_has_markers(result):
        return result
    metadata = result["metadata"]
    new_metadata = dict(metadata)
    changed = False

    obs = metadata.get("_runtime_observability")
    kwargs = obs.get("llm_generate_kwargs") if isinstance(obs, dict) else None
    if isinstance(kwargs, dict):
        new_kwargs: Optional[Dict[str, Any]] = None
        for field in _OBSERVABILITY_DEDUP_FIELDS:
            value = kwargs.get(field)
            if not is_slim_marker(value):
                continue
            resolved = resolve_slim_value(value, index)
            if resolved is not value:
                if new_kwargs is None:
                    new_kwargs = dict(kwargs)
                new_kwargs[field] = resolved
        if new_kwargs is not None:
            new_obs = dict(obs)  # type: ignore[arg-type]
            new_obs["llm_generate_kwargs"] = new_kwargs
            new_metadata["_runtime_observability"] = new_obs
            changed = True

    preq = metadata.get("_provider_request")
    preq_payload = preq.get("payload") if isinstance(preq, dict) else None
    if isinstance(preq_payload, dict) and is_slim_marker(preq_payload.get("messages")):
        resolved = resolve_slim_value(preq_payload["messages"], index)
        if resolved is not preq_payload["messages"]:
            new_preq_payload = dict(preq_payload)
            new_preq_payload["messages"] = resolved
            new_preq = dict(preq)  # type: ignore[arg-type]
            new_preq["payload"] = new_preq_payload
            new_metadata["_provider_request"] = new_preq
            changed = True

    if not changed:
        return result
    out = dict(result)
    out["metadata"] = new_metadata
    return out


def contains_slim_marker(value: Any, *, _depth: int = 0) -> bool:
    """Cheap detection walk used to skip resolution entirely on the common path."""
    if _depth > 12:
        return False
    if is_slim_marker(value):
        return True
    if isinstance(value, dict):
        return any(contains_slim_marker(v, _depth=_depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(contains_slim_marker(v, _depth=_depth + 1) for v in value)
    return False


def resolve_slim_tree(value: Any, index: Dict[str, Dict[str, Any]], *, _depth: int = 0) -> Any:
    """Recursively resolve `$slim` markers, returning copies only where needed."""
    if _depth > 12:
        return value
    if is_slim_marker(value):
        return resolve_slim_value(value, index)
    if isinstance(value, dict):
        out: Optional[Dict[str, Any]] = None
        for k, v in value.items():
            nv = resolve_slim_tree(v, index, _depth=_depth + 1)
            if nv is not v:
                if out is None:
                    out = dict(value)
                out[k] = nv
        return out if out is not None else value
    if isinstance(value, list):
        out_list: Optional[List[Any]] = None
        for i, v in enumerate(value):
            nv = resolve_slim_tree(v, index, _depth=_depth + 1)
            if nv is not v:
                if out_list is None:
                    out_list = list(value)
                out_list[i] = nv
        return out_list if out_list is not None else value
    return value
