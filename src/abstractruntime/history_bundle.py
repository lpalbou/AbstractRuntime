"""abstractruntime.history_bundle

Runtime-owned, versioned run history export (RunHistoryBundle).

Design goals:
- Client-agnostic: any host UI can render from the same durable contract.
- Reproducible: include a workflow snapshot reference (ArtifactStore-backed).
- JSON-safe: keep payloads serializable; offload oversized leaves to ArtifactStore when possible.

This module is intentionally dependency-light (stdlib + abstractruntime stores/models).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .core.models import RunState
from .storage.artifacts import ArtifactMetadata, ArtifactStore
from .storage.ledger_slim import build_started_payload_index, is_slim_marker, resolve_slim_value
from .storage.offloading import DEFAULT_MAX_INLINE_BYTES, offload_large_values

RUN_HISTORY_BUNDLE_VERSION_V1 = 1
RUN_HISTORY_BUNDLE_ARTIFACT_LIMIT = 500

_RUNTIME_METADATA_ENVELOPE_RE = re.compile(
    r"^\s*<runtime_metadata>\s*(.*?)\s*</runtime_metadata>\s*",
    re.IGNORECASE | re.DOTALL,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enum_str(raw: Any) -> str:
    if raw is None:
        return ""
    # Enum subclasses of `str` (e.g. StepStatus) should be treated as their underlying value.
    if isinstance(raw, str):
        return raw
    v = getattr(raw, "value", None)
    if isinstance(v, str):
        return v
    return str(raw)


def _json_dumps_canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _split_runtime_metadata_envelope(text: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not isinstance(text, str):
        return ("", None)
    raw = text.strip()
    if not raw:
        return ("", None)
    match = _RUNTIME_METADATA_ENVELOPE_RE.match(raw)
    if not match:
        return (raw, None)
    metadata: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(match.group(1).strip())
        if isinstance(parsed, dict):
            metadata = dict(parsed)
    except Exception:
        metadata = None
    return (raw[match.end() :].strip(), metadata)


def _extract_user_prompt_from_input(raw: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not isinstance(raw, dict):
        return ("", None)
    input_data = raw.get("input_data") if isinstance(raw.get("input_data"), dict) else raw

    candidates = [
        input_data.get("prompt"),
        input_data.get("message"),
        input_data.get("task"),
    ]
    ctx = input_data.get("context") if isinstance(input_data.get("context"), dict) else None
    if isinstance(ctx, dict):
        candidates.extend([ctx.get("task"), ctx.get("message")])

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return _split_runtime_metadata_envelope(c)

    msgs = ctx.get("messages") if isinstance(ctx, dict) else None
    if isinstance(msgs, list):
        # `context.messages` is a conversation history. The replay prompt for a
        # root run is the latest user turn, not the first user turn in the session.
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            if role != "user":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                prompt, runtime_metadata = _split_runtime_metadata_envelope(content)
                msg_meta = m.get("runtime_metadata") if isinstance(m.get("runtime_metadata"), dict) else None
                if msg_meta is None:
                    meta0 = m.get("metadata") if isinstance(m.get("metadata"), dict) else None
                    msg_meta = meta0.get("runtime_metadata") if isinstance(meta0, dict) and isinstance(meta0.get("runtime_metadata"), dict) else None
                if isinstance(msg_meta, dict):
                    merged = dict(runtime_metadata or {})
                    merged.update({str(k): v for k, v in msg_meta.items() if v is not None and str(v).strip()})
                    runtime_metadata = merged or None
                return (prompt, runtime_metadata)
    return ("", None)


def _verbatim_message_lanes(raw: Any, output: Any) -> List[Any]:
    """Every place a turn's own durable transcript can be found on ITS run.

    Two shapes, both already loaded — this costs no extra I/O:

    - `vars.context.messages`: a run whose loop owns the transcript directly
      (a react/codeact/memact workflow started with the turn's context).
    - `output.scratchpad.messages` / `output.messages`: the BUNDLE shape. The
      gateway's basic-agent flow runs its react loop in a SUBRUN, so the root's
      `vars.context.messages` holds only the seeded history and the current
      turn never appears there — but the agent node folds the loop's durable
      transcript back into the root's output, and that copy carries the turn as
      it was sent.
    """
    lanes: List[Any] = []
    if isinstance(raw, dict):
        input_data = raw.get("input_data") if isinstance(raw.get("input_data"), dict) else raw
        ctx = input_data.get("context") if isinstance(input_data.get("context"), dict) else None
        if isinstance(ctx, dict) and isinstance(ctx.get("messages"), list):
            lanes.append(ctx["messages"])
    if isinstance(output, dict):
        scratch = output.get("scratchpad")
        if isinstance(scratch, dict) and isinstance(scratch.get("messages"), list):
            lanes.append(scratch["messages"])
        if isinstance(output.get("messages"), list):
            lanes.append(output["messages"])
    return lanes


def _extract_user_prompt_verbatim_from_input(raw: Any, *, display_prompt: str, output: Any = None) -> str:
    """The turn's user message exactly as it was SENT, envelope included.

    `_extract_user_prompt_from_input` above is the DISPLAY channel: it strips the
    runtime's `<runtime_metadata>` envelope so UIs show what the human typed. The
    REPLAY channel (`abstractruntime.session_history`) needs the opposite — the
    exact bytes, because turn N's prompt must stay an exact byte-prefix of turn
    N+1's for any prefix cache to restore it (mission A, 2026-09-22). Replaying a
    stripped copy of a stamped message is what made every conversational turn
    re-prefill from the previous user message onward.

    Conservative by construction: the durable `context.messages` entry is returned
    only when stripping its envelope yields EXACTLY the display prompt this turn
    already resolved to — so this can never substitute a different message (an
    `ask_user` reply, an operator-guidance interjection) for the turn's question.
    Returns "" when there is no such message, and the caller falls back to the
    display prompt.
    """
    wanted = str(display_prompt or "").strip()
    if not wanted:
        return ""
    for msgs in _verbatim_message_lanes(raw, output):
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            if str(m.get("role") or "").strip() != "user":
                continue
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            stripped, _meta = _split_runtime_metadata_envelope(content)
            if stripped.strip() != wanted:
                continue
            # Already envelope-free: the display prompt IS the bytes.
            return content if content.strip() != wanted else ""
    return ""


def _extract_context_attachments_from_input(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    input_data = raw.get("input_data") if isinstance(raw.get("input_data"), dict) else raw
    ctx = input_data.get("context") if isinstance(input_data.get("context"), dict) else None
    atts = ctx.get("attachments") if isinstance(ctx, dict) else None
    if not isinstance(atts, list):
        return []
    out: List[Dict[str, Any]] = []
    for a in atts:
        if isinstance(a, dict):
            out.append(dict(a))
    return out


def _parse_iso_ms(raw: Any) -> Optional[int]:
    s = str(raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = f"{s[:-1]}+00:00"
    try:
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except Exception:
        return None


def _parse_usage_summary(value: Any) -> Optional[Dict[str, int]]:
    if not isinstance(value, dict):
        return None
    v = value
    in_tok = v.get("input_tokens")
    if in_tok is None:
        in_tok = v.get("prompt_tokens")
    if in_tok is None:
        in_tok = v.get("prompt")
    if in_tok is None:
        in_tok = v.get("input")
    if in_tok is None:
        in_tok = v.get("in")

    out_tok = v.get("output_tokens")
    if out_tok is None:
        out_tok = v.get("completion_tokens")
    if out_tok is None:
        out_tok = v.get("completion")
    if out_tok is None:
        out_tok = v.get("output")
    if out_tok is None:
        out_tok = v.get("out")

    total_tok = v.get("total_tokens")
    if total_tok is None:
        total_tok = v.get("total")

    try:
        in_i = int(in_tok) if in_tok is not None and not isinstance(in_tok, bool) else 0
    except Exception:
        in_i = 0
    try:
        out_i = int(out_tok) if out_tok is not None and not isinstance(out_tok, bool) else 0
    except Exception:
        out_i = 0
    try:
        total_i = int(total_tok) if total_tok is not None and not isinstance(total_tok, bool) else in_i + out_i
    except Exception:
        total_i = in_i + out_i

    if in_i <= 0 and out_i <= 0 and total_i <= 0:
        return None
    return {
        "input_tokens": max(0, int(in_i)),
        "output_tokens": max(0, int(out_i)),
        "total_tokens": max(0, int(total_i)),
    }


def _extract_usage_from_ledger_record(rec: Dict[str, Any]) -> Optional[Dict[str, int]]:
    result = rec.get("result")
    if not isinstance(result, dict):
        return None
    usage = result.get("usage") or result.get("token_usage") or result.get("tokens")
    if not isinstance(usage, dict):
        output = result.get("output")
        if isinstance(output, dict):
            usage = output.get("usage") or output.get("token_usage") or output.get("tokens")
    if not isinstance(usage, dict):
        return None
    return _parse_usage_summary(usage)


def _extract_resolved_action_from_ledger_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(rec, dict):
        return None
    result = rec.get("result")
    if not isinstance(result, dict):
        return None
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return None
    action = metadata.get("_runtime_resolved_action")
    if not isinstance(action, dict):
        return None
    return dict(action)


def _collect_resolved_actions(ledgers: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run_id, ledger in (ledgers or {}).items():
        items = ledger.get("items") if isinstance(ledger, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            rec = item.get("record") if isinstance(item, dict) else None
            action = _extract_resolved_action_from_ledger_record(rec)
            if action is None:
                continue
            enriched = dict(action)
            enriched.setdefault("run_id", str(run_id or ""))
            out.append(enriched)
    return out


def _extract_repl_stats_from_ledger(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    llm_calls = 0
    tool_calls = 0
    usage_sum = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    min_ms: Optional[int] = None
    max_ms: Optional[int] = None

    records_list = [rec for rec in (records or []) if isinstance(rec, dict)]
    # Terminal records may carry `$slim` markers instead of oversized payload
    # fields (0067-M); the STARTED records in the same list hold the bytes.
    started_index: Optional[Dict[str, Dict[str, Any]]] = None

    def _resolved(value: Any) -> Any:
        nonlocal started_index
        if not is_slim_marker(value):
            return value
        if started_index is None:
            started_index = build_started_payload_index(records_list)
        return resolve_slim_value(value, started_index)

    for rec in records_list:
        st = _enum_str(rec.get("status")).strip()

        ms_start = _parse_iso_ms(rec.get("started_at"))
        ms_end = _parse_iso_ms(rec.get("ended_at"))
        ms = ms_end if ms_end is not None else ms_start
        if ms is not None:
            min_ms = ms if min_ms is None else min(min_ms, ms)
            max_ms = ms if max_ms is None else max(max_ms, ms)

        eff = rec.get("effect") if isinstance(rec.get("effect"), dict) else None
        eff_type = str((eff or {}).get("type") or "").strip()
        if eff_type == "llm_call" and st == "completed":
            llm_calls += 1
            usage = _extract_usage_from_ledger_record(rec)
            if usage:
                usage_sum["input_tokens"] += int(usage.get("input_tokens") or 0)
                usage_sum["output_tokens"] += int(usage.get("output_tokens") or 0)
                usage_sum["total_tokens"] += int(usage.get("total_tokens") or 0)

        if eff_type == "tool_calls" and st == "completed":
            payload = eff.get("payload") if isinstance(eff, dict) and isinstance(eff.get("payload"), dict) else None
            calls = _resolved(payload.get("tool_calls")) if isinstance(payload, dict) else None
            if isinstance(calls, list):
                tool_calls += len([c for c in calls if isinstance(c, dict) or c is not None])

    duration_ms = int(max(0, (max_ms - min_ms))) if min_ms is not None and max_ms is not None else 0
    tok_s: Optional[float] = None
    if duration_ms > 0 and usage_sum.get("total_tokens", 0) > 0:
        tok_s = float(usage_sum["total_tokens"]) / (float(duration_ms) / 1000.0)
    return {
        "duration_ms": duration_ms,
        "llm_calls": int(llm_calls),
        "tool_calls": int(tool_calls),
        "usage": usage_sum,
        "tok_s": tok_s,
    }


def _extract_flow_end_output_from_ledger(records: List[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Best-effort extract the final assistant response from ledger records.

    Mirrors AbstractCode Web's `extract_flow_end_output` heuristics.
    Returns (response_text, meta_obj_or_none).
    """

    def _pick_textish(v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        if v is None:
            return ""
        if isinstance(v, (int, float, bool)):
            return str(v)
        return ""

    records_list = [rec for rec in (records or []) if isinstance(rec, dict)]
    started_index: Optional[Dict[str, Dict[str, Any]]] = None

    def _resolved(value: Any) -> Any:
        # `$slim` markers on terminal records (0067-M) resolve against the
        # STARTED records already present in the same list.
        nonlocal started_index
        if not is_slim_marker(value):
            return value
        if started_index is None:
            started_index = build_started_payload_index(records_list)
        return resolve_slim_value(value, started_index)

    for rec in reversed(records_list):
        status = _enum_str(rec.get("status")).strip()
        eff = rec.get("effect") if isinstance(rec.get("effect"), dict) else None
        eff_type = str((eff or {}).get("type") or "").strip()

        # answer_user: common in chat-like flows.
        if status == "completed" and eff_type == "answer_user":
            res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            msg = res.get("message")
            if msg is None and isinstance(eff, dict):
                payload = eff.get("payload") if isinstance(eff.get("payload"), dict) else {}
                msg = _resolved(payload.get("message")) or _resolved(payload.get("text")) or _resolved(payload.get("content"))
            text = _pick_textish(msg)
            if text:
                return (text, None)

        # output node: record.result.output.{answer/response/message/...}
        result = rec.get("result") if isinstance(rec.get("result"), dict) else None
        out0 = result.get("output") if isinstance(result, dict) else None
        if isinstance(out0, str):
            s = out0.strip()
            if s:
                return (s, None)
        if isinstance(out0, dict):
            msg = (
                _pick_textish(out0.get("answer"))
                or _pick_textish(out0.get("response"))
                or _pick_textish(out0.get("message"))
                or _pick_textish(out0.get("text"))
                or _pick_textish(out0.get("content"))
            )
            if msg:
                meta = out0.get("meta") if isinstance(out0.get("meta"), dict) else None
                return (msg, dict(meta) if isinstance(meta, dict) else None)

        # Terminal resume completion record (runtime may append an output envelope).
        if status == "completed" and isinstance(result, dict):
            out_res = result.get("output") if isinstance(result.get("output"), dict) else None
            if isinstance(out_res, dict):
                msg = (
                    _pick_textish(out_res.get("answer"))
                    or _pick_textish(out_res.get("response"))
                    or _pick_textish(out_res.get("message"))
                    or _pick_textish(out_res.get("text"))
                    or _pick_textish(out_res.get("content"))
                )
                if msg:
                    return (msg, None)

    return ("", None)


def persist_workflow_snapshot(
    *,
    run_store: Any,
    artifact_store: ArtifactStore,
    run_id: str,
    workflow_id: str,
    snapshot: Dict[str, Any],
    format: str,
) -> Dict[str, Any]:
    """Persist a workflow snapshot for a run and store a small ref in run.vars.

    Returns the stored ref dict (JSON-safe), which is also written to:
      run.vars["_runtime"]["workflow_snapshot"].
    """

    rid = str(run_id or "").strip()
    wid = str(workflow_id or "").strip()
    fmt = str(format or "").strip() or "unknown"
    if not rid:
        raise ValueError("run_id is required")
    if not wid:
        raise ValueError("workflow_id is required")
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a dict")

    run: Optional[RunState]
    try:
        run = run_store.load(rid)
    except Exception as e:
        raise RuntimeError(f"Failed to load run '{rid}': {e}") from e
    if run is None:
        raise KeyError(f"Run '{rid}' not found")

    # Idempotency: if a snapshot ref already exists, keep it.
    vars_obj = getattr(run, "vars", None)
    if not isinstance(vars_obj, dict):
        vars_obj = {}
        run.vars = vars_obj  # type: ignore[assignment]

    runtime_ns = vars_obj.get("_runtime")
    if not isinstance(runtime_ns, dict):
        runtime_ns = {}
        vars_obj["_runtime"] = runtime_ns

    existing = runtime_ns.get("workflow_snapshot")
    if isinstance(existing, dict) and str(existing.get("artifact_id") or "").strip():
        return dict(existing)

    content = _json_dumps_canonical(snapshot)
    sha = _sha256_hex(content)

    tags = {
        "kind": "workflow_snapshot",
        "workflow_id": wid,
        "format": fmt,
        "sha256": sha,
    }
    meta = artifact_store.store_json(snapshot, run_id=rid, tags=tags)
    artifact_id = str(getattr(meta, "artifact_id", "") or "").strip()
    if not artifact_id:
        raise RuntimeError("ArtifactStore returned empty artifact_id for workflow snapshot")

    ref: Dict[str, Any] = {
        "workflow_id": wid,
        "format": fmt,
        "sha256": sha,
        "artifact_id": artifact_id,
        "created_at": _utc_now_iso(),
    }
    runtime_ns["workflow_snapshot"] = ref
    try:
        run_store.save(run)
    except Exception as e:
        raise RuntimeError(f"Failed to persist workflow snapshot ref to run '{rid}': {e}") from e
    return dict(ref)


def _list_descendant_run_ids(
    *,
    run_store: Any,
    root_run_id: str,
    limit: int = 5000,
    warnings_sink: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Return descendant run ids (BFS) when the RunStore supports list_children().

    Per-node discovery failures are survived (a bad subtree must not kill
    the export) but REPORTED into `warnings_sink` when the caller supplies
    one — a silently missing subtree is the class the replay-integrity
    audit (2026-07-25) named: the bundle looked complete while whole subrun
    ledgers were absent.
    """
    out: List[str] = []
    list_children = getattr(run_store, "list_children", None)
    if not callable(list_children):
        return out
    queue: List[str] = [root_run_id]
    seen: set[str] = set()
    while queue and len(out) < limit:
        cur = str(queue.pop(0) or "").strip()
        if not cur or cur in seen:
            continue
        seen.add(cur)
        try:
            kids = list_children(parent_run_id=cur) or []
        except Exception as e:
            kids = []
            if warnings_sink is not None:
                warnings_sink.append(
                    {
                        "code": "subtree_discovery_failed",
                        "run_id": cur,
                        "detail": f"list_children failed; descendants of this run are missing from the bundle: {e}",
                    }
                )
        for c in kids:
            cid = getattr(c, "run_id", None)
            cid2 = str(cid or "").strip()
            if not cid2 or cid2 in seen:
                continue
            out.append(cid2)
            queue.append(cid2)
    if len(out) >= limit and queue and warnings_sink is not None:
        warnings_sink.append(
            {
                "code": "subtree_truncated",
                "run_id": str(root_run_id),
                "detail": f"descendant discovery stopped at the {limit}-run cap; deeper subruns are missing from the bundle",
            }
        )
    return out


def _replay_safe(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded JSON-safe value for replay metadata.

    Artifact descriptors are already redacted by Runtime, but replay bundles should stay
    compact because they are fetched with ledgers and session context.
    """

    if depth > 5:
        return "#TRUNCATION: replay metadata depth limit"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:2000] + "#TRUNCATION" if len(value) > 2000 else value
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [_replay_safe(item, depth=depth + 1) for item in items[:40]]
        if len(items) > 40:
            out.append({"truncated_items": len(items) - 40})
        return out
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, raw in list(value.items())[:80]:
            if key is None:
                continue
            out[str(key)] = _replay_safe(raw, depth=depth + 1)
        if len(value) > 80:
            out["truncated_keys"] = len(value) - 80
        return out
    return str(value)


def _artifact_replay_summary(meta: ArtifactMetadata) -> Dict[str, Any]:
    descriptor = meta.descriptor.to_dict() if getattr(meta, "descriptor", None) is not None else {}
    access = meta.access.to_dict() if getattr(meta, "access", None) is not None else {}
    return {
        "artifact_id": str(getattr(meta, "artifact_id", "") or ""),
        "blob_id": getattr(meta, "blob_id", None),
        "run_id": getattr(meta, "run_id", None),
        "content_type": str(getattr(meta, "content_type", "") or ""),
        "size_bytes": int(getattr(meta, "size_bytes", 0) or 0),
        "created_at": getattr(meta, "created_at", None),
        "tags": _replay_safe(getattr(meta, "tags", None) or {}),
        "descriptor": _replay_safe(descriptor),
        "metadata": _replay_safe(getattr(meta, "metadata", None) or {}),
        "access": _replay_safe(access),
    }


def _list_replay_artifacts_for_run(
    *,
    artifact_store: Optional[ArtifactStore],
    run_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if artifact_store is None:
        return []
    rid = str(run_id or "").strip()
    if not rid:
        return []
    try:
        metas = artifact_store.list_by_run(rid)
    except Exception:
        metas = []
    if not isinstance(metas, list):
        return []
    metas0 = [m for m in metas if isinstance(m, ArtifactMetadata)]
    metas0.sort(key=lambda m: (str(m.created_at or ""), str(m.artifact_id or "")))
    max_items = max(0, int(limit or 0))
    if max_items > 0:
        metas0 = metas0[:max_items]
    return [_artifact_replay_summary(m) for m in metas0]


def _list_replay_artifacts_for_runs(
    *,
    artifact_store: Optional[ArtifactStore],
    run_ids: Iterable[str],
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if artifact_store is None:
        return []
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    max_items = max(0, int(limit or 0))
    for rid in run_ids:
        for item in _list_replay_artifacts_for_run(artifact_store=artifact_store, run_id=rid, limit=max_items or RUN_HISTORY_BUNDLE_ARTIFACT_LIMIT):
            aid = str(item.get("artifact_id") or "").strip()
            if not aid or aid in seen:
                continue
            seen.add(aid)
            out.append(item)
            if max_items > 0 and len(out) >= max_items:
                return out
    return out


# Bounded answer-resolve cap (c4978 R2): offloaded outputs are >=256KB by
# construction; the incident doc was 445KB. 4MB covers scratchpad-heavy
# outputs while keeping the resolve a bounded read, never an artifact walk.
_OFFLOADED_ANSWER_RESOLVE_MAX_BYTES = 4 * 1024 * 1024


def _best_effort_session_turns(
    *,
    run_store: Any,
    ledger_store: Any,
    artifact_store: Optional[ArtifactStore],
    session_id: str,
    limit: int,
    until_ms: Optional[int] = None,
    include_stats: bool = True,
    include_artifacts: bool = True,
) -> List[Dict[str, Any]]:
    """Best-effort session turn list (root runs only).

    This is a pragmatic bridge for thin clients (AbstractCode Web/mobile) until a more
    explicit session history contract exists.

    `include_stats`/`include_artifacts` exist for the session-replay read path
    (`abstractruntime.session_history`), which needs only prompt+answer per turn:
    stats and artifact listings scan every descendant ledger and are wasted work
    on that hot path. Defaults keep the history-bundle behavior unchanged.
    """

    def _classify_turn(*, workflow_id: str, vars_obj: Any) -> str:
        wid = str(workflow_id or "")
        # Internal = the runtime's RESERVED dunder ids (__session_memory__,
        # __global_memory__): both start AND end with `__`. A bare
        # startswith("__") also swallowed every tenant-catalog workflow
        # (`__catalog__v2__...@ver:flow`) — which is how gateway thin clients
        # run — silently hiding ALL their turns from session views and the
        # durable session replay (live-proof finding, 2026-07-16).
        if wid.startswith("__") and wid.endswith("__"):
            return "internal"
        if wid.startswith("scheduled:"):
            return "scheduled"
        if isinstance(vars_obj, dict):
            meta = vars_obj.get("_meta")
            if isinstance(meta, dict) and isinstance(meta.get("schedule"), dict):
                return "scheduled"
            ctx0 = vars_obj.get("context")
            if isinstance(ctx0, dict) and isinstance(ctx0.get("messages"), list):
                return "chat"
        return "run"

    def _resolve_offloaded_output(out: Any) -> Optional[Dict[str, Any]]:
        """Bounded resolve of a ROOT-REPLACED run output (code-tui c4978 R2,
        the 401-incident chain's deepest server link): a terminal output that
        crossed the inline cap was replaced WHOLESALE by an `$artifact` ref,
        so its turn extracted an empty answer and session replay silently
        forgot exactly the turns that most need replaying — the client then
        carried transcripts the server should have owned. Resolution is
        BOUNDED three ways: only refs the run-output offloader itself minted
        (tags.source=run_output_offload — handler-authored artifact currency
        stays refs, the ledger-rehydrate discipline), only up to
        _OFFLOADED_ANSWER_RESOLVE_MAX_BYTES, and the parsed doc is used for
        ANSWER extraction only — never stored, never returned whole to
        callers."""
        from .storage.artifacts import get_artifact_id, is_artifact_ref

        if artifact_store is None or not is_artifact_ref(out):
            return None
        try:
            aid = get_artifact_id(out)
            meta = artifact_store.get_metadata(aid)
            tags = dict(getattr(meta, "tags", None) or {}) if meta is not None else {}
            if str(tags.get("source") or "") != "run_output_offload":
                return None
            size = int(getattr(meta, "size_bytes", 0) or 0)
            if size > _OFFLOADED_ANSWER_RESOLVE_MAX_BYTES:
                return None
            artifact = artifact_store.load(aid)
            doc = artifact.as_json() if artifact is not None else None
            return doc if isinstance(doc, dict) else None
        except Exception:  # noqa: BLE001 - a broken artifact keeps the v1 skip, never raises
            return None

    def _extract_answer_from_run_output(run: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        out = getattr(run, "output", None)
        resolved = _resolve_offloaded_output(out)
        if resolved is not None:
            out = resolved
        if not isinstance(out, dict):
            return ("", None)

        # Common envelopes used by AbstractCode agent flows.
        candidates = [
            out.get("response"),
            out.get("answer"),
            out.get("message"),
            out.get("text"),
            out.get("content"),
        ]
        # Nested results are sometimes stored under `result` or `output`.
        nested = out.get("result") if isinstance(out.get("result"), dict) else None
        if nested is None:
            nested = out.get("output") if isinstance(out.get("output"), dict) else None
        if isinstance(nested, dict):
            candidates.extend(
                [
                    nested.get("response"),
                    nested.get("answer"),
                    nested.get("message"),
                    nested.get("text"),
                    nested.get("content"),
                ]
            )

        answer = ""
        for c in candidates:
            if isinstance(c, str) and c.strip():
                answer = c.strip()
                break

        meta0 = out.get("meta") if isinstance(out.get("meta"), dict) else None
        if meta0 is None and isinstance(nested, dict):
            meta0 = nested.get("meta") if isinstance(nested.get("meta"), dict) else None
        meta = dict(meta0) if isinstance(meta0, dict) else None
        return (answer, meta)

    sid = str(session_id or "").strip()
    if not sid:
        return []
    list_runs = getattr(run_store, "list_runs", None)

    roots: List[RunState] = []

    # Prefer the lightweight run index when available. It is the only current path
    # that can query by session without scanning unrelated runs.
    list_run_index = getattr(run_store, "list_run_index", None)
    if callable(list_run_index):
        try:
            rows = list_run_index(session_id=sid, root_only=True, limit=max(1000, int(limit) * 5))
        except Exception:
            rows = []
        # Load only a bounded newest-first window of full RunStates: the
        # result is sliced to the newest `limit` turns anyway, and loading
        # every root of a long-lived session made each session-replay read
        # O(session length) in full JSON parses (audit finding #3). The 3x
        # over-fetch absorbs rows that classify out (internal/scheduled).
        load_window = max(int(limit) * 3, int(limit) + 8)
        for row in (rows or [])[:load_window]:
            if not isinstance(row, dict):
                continue
            rid0 = str(row.get("run_id") or "").strip()
            if not rid0:
                continue
            try:
                loaded = run_store.load(rid0)
            except Exception:
                loaded = None
            if loaded is None:
                continue
            roots.append(loaded)

    if not roots and callable(list_runs):
        # Compatibility fallback for older stores.
        try:
            candidates = list_runs(limit=max(1000, int(limit) * 5))
        except Exception:
            candidates = []

        for r in candidates or []:
            try:
                rid = str(getattr(r, "run_id", "") or "").strip()
                if not rid:
                    continue
                if str(getattr(r, "session_id", "") or "").strip() != sid:
                    continue
                if str(getattr(r, "parent_run_id", "") or "").strip():
                    continue
                roots.append(r)
            except Exception:
                continue

    if not roots:
        return []

    roots0: List[RunState] = []
    seen_roots: set[str] = set()
    for r in roots:
        try:
            rid = str(getattr(r, "run_id", "") or "").strip()
            if not rid or rid in seen_roots:
                continue
            seen_roots.add(rid)
            if str(getattr(r, "session_id", "") or "").strip() != sid:
                continue
            if str(getattr(r, "parent_run_id", "") or "").strip():
                continue
            vars_obj = getattr(r, "vars", None)
            wid = str(getattr(r, "workflow_id", "") or "")
            kind = _classify_turn(workflow_id=wid, vars_obj=vars_obj)
            if kind == "internal":
                continue
            roots0.append(r)
        except Exception:
            continue
    roots = roots0

    # Prefer chat-like turns when present (avoid scheduled wrapper runs polluting chat replay).
    if roots:
        chat_roots: List[RunState] = []
        for r in roots:
            wid = str(getattr(r, "workflow_id", "") or "")
            vars_obj = getattr(r, "vars", None)
            if _classify_turn(workflow_id=wid, vars_obj=vars_obj) == "chat":
                chat_roots.append(r)
        if chat_roots:
            roots = chat_roots

    if until_ms is not None:
        bounded_roots: List[RunState] = []
        for r in roots:
            created_ms = _parse_iso_ms(getattr(r, "created_at", None))
            if created_ms is None:
                created_ms = _parse_iso_ms(getattr(r, "updated_at", None))
            if created_ms is None or created_ms <= until_ms:
                bounded_roots.append(r)
        roots = bounded_roots

    def _ts_key(r: Any) -> tuple:
        # (parsed ms, raw ISO string): _parse_iso_ms rounds to milliseconds,
        # and a stable ascending sort would otherwise keep the newest-first
        # index order WITHIN a tie — reversing same-millisecond turns. The
        # raw ISO string preserves sub-ms precision as the tiebreak.
        for k in ("created_at", "updated_at"):
            raw = getattr(r, k, None)
            ms = _parse_iso_ms(raw)
            if ms is not None:
                return (float(ms), str(raw or ""))
        return (0.0, "")

    roots.sort(key=_ts_key)
    roots = roots[-int(limit) :] if limit > 0 else roots

    ledger_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _ledger_for(run_id: str) -> List[Dict[str, Any]]:
        rid2 = str(run_id or "").strip()
        if not rid2:
            return []
        cached = ledger_cache.get(rid2)
        if isinstance(cached, list):
            return cached
        try:
            raw = ledger_store.list(rid2) if hasattr(ledger_store, "list") else []
        except Exception:
            raw = []
        records = [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
        ledger_cache[rid2] = records
        return records

    out: List[Dict[str, Any]] = []
    for r in roots:
        rid = str(getattr(r, "run_id", "") or "").strip()
        vars_obj = getattr(r, "vars", None)
        input_data = dict(vars_obj) if isinstance(vars_obj, dict) else {}
        wid = str(getattr(r, "workflow_id", "") or "").strip()
        kind = _classify_turn(workflow_id=wid, vars_obj=vars_obj)
        prompt, prompt_metadata = _extract_user_prompt_from_input(input_data)
        # Replay channel (see `_extract_user_prompt_verbatim_from_input`): the exact
        # bytes this turn was sent with. Display consumers keep reading `prompt`.
        prompt_verbatim = _extract_user_prompt_verbatim_from_input(
            input_data, display_prompt=prompt, output=getattr(r, "output", None)
        )
        attachments = _extract_context_attachments_from_input(input_data)
        status = getattr(getattr(r, "status", None), "value", None) or str(getattr(r, "status", "") or "")
        created_at = str(getattr(r, "created_at", "") or "").strip() or None
        updated_at = str(getattr(r, "updated_at", "") or "").strip() or None

        answer, answer_meta = _extract_answer_from_run_output(r)
        if not answer:
            try:
                ledger = _ledger_for(rid)
                if ledger:
                    answer, answer_meta = _extract_flow_end_output_from_ledger(ledger)
            except Exception:
                answer = ""
                answer_meta = None

        stats: Optional[Dict[str, Any]] = None
        run_ids = [rid]
        if include_stats:
            try:
                run_ids.extend(_list_descendant_run_ids(run_store=run_store, root_run_id=rid))
                all_records: List[Dict[str, Any]] = []
                for rid2 in run_ids:
                    all_records.extend(_ledger_for(rid2))
                stats = _extract_repl_stats_from_ledger(all_records)
            except Exception:
                run_ids = [rid]
                stats = None

        artifacts: List[Dict[str, Any]] = []
        if include_artifacts:
            artifacts = _list_replay_artifacts_for_runs(
                artifact_store=artifact_store,
                run_ids=run_ids,
                limit=RUN_HISTORY_BUNDLE_ARTIFACT_LIMIT,
            )

        out.append(
            {
                "run_id": rid,
                "workflow_id": wid or None,
                "kind": kind,
                "status": str(status),
                "created_at": created_at,
                "updated_at": updated_at,
                "prompt": prompt or None,
                "prompt_verbatim": prompt_verbatim or None,
                "prompt_metadata": prompt_metadata,
                "attachments": attachments,
                "answer": answer or None,
                "answer_meta": answer_meta,
                "stats": stats,
                "artifacts": artifacts,
            }
        )
    return out


def _omitted_marker(*, field: str, value: Any) -> Dict[str, Any]:
    """Structured replacement for a field dropped by a bundle profile.

    Never a silent hole: the marker names the profile, the field, and the
    dropped byte count so a consumer (and the operator) can see exactly what
    a projection removed and fetch the full bundle when they need it.
    """
    try:
        nbytes = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except Exception:
        nbytes = None
    return {"$omitted": {"profile": "replay", "field": field, "bytes": nbytes}}


# Request-side payload fields a transcript fold never reads (replay-integrity
# audit, 2026-07-25: STARTED payload.messages alone was 4.1MB of a 14.3MB
# single-turn bundle). The profile DROPS whole fields with markers — never
# truncates (truncation violates the ADR bar the operator set).
_REPLAY_DROP_PAYLOAD_FIELDS = ("messages", "system_prompt", "prompt")
# The two observability metadata paths (5.95MB in the same bundle) — request
# captures, not transcript content.
_REPLAY_DROP_METADATA_FIELDS = ("_runtime_observability", "_provider_request")


def _project_record_for_replay(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Spine-copy projection of one ledger record for the replay profile.

    Drops request-side payload fields and the observability metadata copies,
    each replaced by a structured `$omitted` marker. The input record is
    NEVER mutated (ledger stores may cache and share record objects — a
    consumer mutation must not poison the store's copy).
    """
    out = rec
    eff = rec.get("effect")
    payload = eff.get("payload") if isinstance(eff, dict) else None
    if isinstance(payload, dict) and any(f in payload for f in _REPLAY_DROP_PAYLOAD_FIELDS):
        new_payload = dict(payload)
        for f in _REPLAY_DROP_PAYLOAD_FIELDS:
            if f in new_payload:
                new_payload[f] = _omitted_marker(field=f"payload.{f}", value=new_payload[f])
        new_eff = dict(eff)
        new_eff["payload"] = new_payload
        out = dict(out) if out is rec else out
        out["effect"] = new_eff
    result = rec.get("result")
    metadata = result.get("metadata") if isinstance(result, dict) else None
    if isinstance(metadata, dict) and any(f in metadata for f in _REPLAY_DROP_METADATA_FIELDS):
        new_metadata = dict(metadata)
        for f in _REPLAY_DROP_METADATA_FIELDS:
            if f in new_metadata:
                new_metadata[f] = _omitted_marker(field=f"result.metadata.{f}", value=new_metadata[f])
        new_result = dict(result)
        new_result["metadata"] = new_metadata
        out = dict(out) if out is rec else out
        out["result"] = new_result
    return out


def export_run_history_bundle(
    *,
    run_id: str,
    run_store: Any,
    ledger_store: Any,
    artifact_store: Optional[ArtifactStore] = None,
    include_subruns: bool = True,
    include_session: bool = False,
    session_turn_limit: int = 200,
    ledger_mode: str = "tail",  # "tail" | "full"
    ledger_max_items: int = 2000,
    detail: str = "full",  # "full" | "replay"
) -> Dict[str, Any]:
    """Export a versioned RunHistoryBundle dict (v1).

    Notes:
    - This function is pure export (no network); gateway hosts should expose it as an endpoint.
    - Payload is JSON-safe; when ArtifactStore is available, very large leaves are offloaded.
    - `detail="replay"` is a labeled PROJECTION for transcript folds: it drops
      request-side payload fields and the observability metadata copies (each
      replaced by a structured `$omitted` marker) and skips the timeline. The
      exact bundle stays the default; the projection is opt-in per fetch.
    - `warnings` (in-band): every degradation the export survives — subtree
      discovery failure, ledger read failure, torn rows, tail-window
      truncation — is reported in the bundle instead of silently omitted
      (operator ruling, 2026-07-25: a bundle that cannot be complete must
      say so).
    """

    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    profile = str(detail or "full").strip().lower()
    if profile not in ("full", "replay"):
        raise ValueError(f"detail must be 'full' or 'replay', got {detail!r}")

    warnings: List[Dict[str, Any]] = []

    run: Optional[RunState]
    try:
        run = run_store.load(rid)
    except Exception as e:
        raise RuntimeError(f"Failed to load run '{rid}': {e}") from e
    if run is None:
        raise KeyError(f"Run '{rid}' not found")

    # Collect run tree ids (root + descendants).
    run_ids: List[str] = [rid]
    if include_subruns:
        try:
            run_ids.extend(
                _list_descendant_run_ids(run_store=run_store, root_run_id=rid, warnings_sink=warnings)
            )
        except Exception as e:
            warnings.append(
                {
                    "code": "subtree_discovery_failed",
                    "run_id": rid,
                    "detail": f"descendant run discovery failed; bundle covers the root run only: {e}",
                }
            )

    # Snapshot ref (best-effort, stored under run.vars._runtime.workflow_snapshot).
    snapshot_ref = None
    try:
        vars_obj = getattr(run, "vars", None)
        runtime_ns = vars_obj.get("_runtime") if isinstance(vars_obj, dict) else None
        ws = runtime_ns.get("workflow_snapshot") if isinstance(runtime_ns, dict) else None
        if isinstance(ws, dict) and str(ws.get("artifact_id") or "").strip():
            snapshot_ref = dict(ws)
    except Exception:
        snapshot_ref = None

    ledgers: Dict[str, Any] = {}
    timeline: List[Dict[str, Any]] = []

    def _append_timeline_items(*, run_id2: str, items_with_cursor: List[Dict[str, Any]]) -> None:
        for it in items_with_cursor:
            cursor = it.get("cursor")
            rec = it.get("record")
            if not isinstance(rec, dict):
                continue
            node_id = str(rec.get("node_id") or "").strip() or None
            status = _enum_str(rec.get("status")).strip() or None
            eff = rec.get("effect") if isinstance(rec.get("effect"), dict) else None
            eff_type = str((eff or {}).get("type") or "").strip() or None
            started_at = rec.get("started_at")
            ended_at = rec.get("ended_at") or rec.get("started_at")
            duration_ms = None
            try:
                if started_at and ended_at:
                    s = datetime.fromisoformat(str(started_at))
                    e = datetime.fromisoformat(str(ended_at))
                    duration_ms = max(0.0, (e - s).total_seconds() * 1000.0)
            except Exception:
                duration_ms = None
            timeline.append(
                {
                    "run_id": run_id2,
                    "cursor": cursor,
                    "node_id": node_id,
                    "status": status,
                    "effect_type": eff_type,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_ms": round(float(duration_ms), 2) if isinstance(duration_ms, (int, float)) else None,
                }
            )

    for rid2 in run_ids:
        try:
            raw = ledger_store.list(rid2) if hasattr(ledger_store, "list") else []
        except Exception as e:
            raw = []
            warnings.append(
                {
                    "code": "ledger_read_failed",
                    "run_id": rid2,
                    "detail": f"ledger read failed; this run's records are missing from the bundle: {e}",
                }
            )
        records = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
        total = len(records)
        torn = (len(raw) - total) if isinstance(raw, list) else 0
        if torn > 0:
            warnings.append(
                {
                    "code": "torn_rows_skipped",
                    "run_id": rid2,
                    "detail": f"{torn} non-record ledger row(s) skipped (recovered/torn lines)",
                    "count": torn,
                }
            )

        mode = str(ledger_mode or "tail").strip().lower()
        max_items_raw = int(ledger_max_items)
        if max_items_raw <= 0 and mode == "tail":
            mode = "full"
        max_items = max_items_raw if max_items_raw > 0 else 2000

        if mode != "full":
            # Tail mode by default (bounded, good for UI). Keep absolute cursor indices.
            if total > max_items:
                start_idx = total - max_items
                window = records[start_idx:]
                cursor_start = start_idx + 1
                warnings.append(
                    {
                        "code": "ledger_tail_window",
                        "run_id": rid2,
                        "detail": (
                            f"tail window: {len(window)} of {total} records included "
                            f"(cursors {start_idx + 1}..{total}); fetch ledger_mode=full for the rest. "
                            "NOTE: $slim markers whose STARTED record fell before the window "
                            "cannot resolve client-side."
                        ),
                        "total": total,
                        "window": len(window),
                    }
                )
            else:
                window = records
                cursor_start = 1
        else:
            window = records
            cursor_start = 1

        items_with_cursor: List[Dict[str, Any]] = []
        for i, rec in enumerate(window):
            rec_out = _project_record_for_replay(rec) if profile == "replay" else rec
            items_with_cursor.append({"cursor": cursor_start + i, "record": rec_out})

        ledgers[rid2] = {
            "run_id": rid2,
            "total": int(total),
            "cursor_start": int(cursor_start),
            "cursor_end": int(cursor_start + len(window) - 1) if window else int(cursor_start - 1),
            "items": items_with_cursor,
            "artifacts": _list_replay_artifacts_for_run(artifact_store=artifact_store, run_id=rid2, limit=RUN_HISTORY_BUNDLE_ARTIFACT_LIMIT),
        }
        if profile != "replay":
            # The replay profile skips the timeline entirely (audit: 0.45MB
            # per incident bundle, never read by a transcript fold).
            _append_timeline_items(run_id2=rid2, items_with_cursor=items_with_cursor)

    # Session section (best-effort, bounded).
    session_section = None
    if include_session:
        sid = str(getattr(run, "session_id", "") or "").strip()
        if sid:
            session_until_ms = _parse_iso_ms(getattr(run, "created_at", None))
            if session_until_ms is None:
                session_until_ms = _parse_iso_ms(getattr(run, "updated_at", None))
            session_section = {
                "session_id": sid,
                "turns": _best_effort_session_turns(
                    run_store=run_store,
                    ledger_store=ledger_store,
                    artifact_store=artifact_store,
                    session_id=sid,
                    limit=max(1, int(session_turn_limit) if int(session_turn_limit) > 0 else 200),
                    until_ms=session_until_ms,
                ),
            }

    # Filtered input_data (exclude private namespaces). This mirrors gateway's behavior but is runtime-owned.
    vars_obj = getattr(run, "vars", None)
    input_data = dict(vars_obj) if isinstance(vars_obj, dict) else {}
    filtered_input_data = {k: v for k, v in input_data.items() if isinstance(k, str) and not k.startswith("_")}

    # Offload oversized leaves when possible (keeps HTTP payload bounded).
    if artifact_store is not None:
        try:
            filtered_input_data = offload_large_values(
                filtered_input_data,
                artifact_store=artifact_store,
                run_id=rid,
                max_inline_bytes=DEFAULT_MAX_INLINE_BYTES,
                base_tags={"source": "history_bundle", "kind": "input_data"},
                root_path="input_data",
                allow_root_replace=False,
            )
        except Exception as e:
            warnings.append(
                {
                    "code": "input_data_offload_failed",
                    "run_id": rid,
                    "detail": f"oversized input_data leaves could not offload to the artifact store (kept inline): {e}",
                }
            )

    # Final bundle.
    bundle: Dict[str, Any] = {
        "version": RUN_HISTORY_BUNDLE_VERSION_V1,
        "generated_at": _utc_now_iso(),
        "detail": profile,
        "root_run_id": rid,
        "run": {
            "run_id": str(getattr(run, "run_id", "") or ""),
            "workflow_id": str(getattr(run, "workflow_id", "") or ""),
            "status": getattr(getattr(run, "status", None), "value", None) or str(getattr(run, "status", "") or ""),
            "current_node": str(getattr(run, "current_node", "") or ""),
            "created_at": getattr(run, "created_at", None),
            "updated_at": getattr(run, "updated_at", None),
            "actor_id": getattr(run, "actor_id", None),
            "session_id": getattr(run, "session_id", None),
            "parent_run_id": getattr(run, "parent_run_id", None),
            "error": getattr(run, "error", None),
            "waiting": getattr(run, "waiting", None).__dict__ if getattr(run, "waiting", None) is not None else None,
        },
        "workflow_snapshot": snapshot_ref,
        "input_data": filtered_input_data,
        "ledgers": ledgers,
        "timeline": timeline,
        "resolved_actions": _collect_resolved_actions(ledgers),
        "session": session_section,
        "warnings": warnings,
    }
    return bundle
