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
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .core.models import RunState
from .storage.artifacts import ArtifactStore
from .storage.offloading import DEFAULT_MAX_INLINE_BYTES, offload_large_values

RUN_HISTORY_BUNDLE_VERSION_V1 = 1


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


def _extract_user_prompt_from_input(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
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
            return c.strip()

    msgs = ctx.get("messages") if isinstance(ctx, dict) else None
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            if role != "user":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
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

    for rec in reversed(records or []):
        if not isinstance(rec, dict):
            continue

        status = _enum_str(rec.get("status")).strip()
        eff = rec.get("effect") if isinstance(rec.get("effect"), dict) else None
        eff_type = str((eff or {}).get("type") or "").strip()

        # answer_user: common in chat-like flows.
        if status == "completed" and eff_type == "answer_user":
            res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            msg = res.get("message")
            if msg is None and isinstance(eff, dict):
                payload = eff.get("payload") if isinstance(eff.get("payload"), dict) else {}
                msg = payload.get("message") or payload.get("text") or payload.get("content")
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


def _list_descendant_run_ids(*, run_store: Any, root_run_id: str, limit: int = 5000) -> List[str]:
    """Return descendant run ids (BFS) when the RunStore supports list_children()."""
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
        except Exception:
            kids = []
        for c in kids:
            cid = getattr(c, "run_id", None)
            cid2 = str(cid or "").strip()
            if not cid2 or cid2 in seen:
                continue
            out.append(cid2)
            queue.append(cid2)
    return out


def _best_effort_session_turns(
    *,
    run_store: Any,
    ledger_store: Any,
    session_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Best-effort session turn list (root runs only).

    This is a pragmatic bridge for thin clients (AbstractCode Web/mobile) until a more
    explicit session history contract exists.
    """

    def _classify_turn(*, workflow_id: str, vars_obj: Any) -> str:
        wid = str(workflow_id or "")
        if wid.startswith("__"):
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

    def _extract_answer_from_run_output(run: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        out = getattr(run, "output", None)
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
    if not callable(list_runs):
        return []

    # RunStore has no session_id index today; we scan a bounded window and filter.
    try:
        candidates = list_runs(limit=max(1000, int(limit) * 5))
    except Exception:
        candidates = []

    roots: List[RunState] = []
    for r in candidates or []:
        try:
            rid = str(getattr(r, "run_id", "") or "").strip()
            if not rid:
                continue
            if str(getattr(r, "session_id", "") or "").strip() != sid:
                continue
            if str(getattr(r, "parent_run_id", "") or "").strip():
                continue
            vars_obj = getattr(r, "vars", None)
            wid = str(getattr(r, "workflow_id", "") or "")
            kind = _classify_turn(workflow_id=wid, vars_obj=vars_obj)
            if kind == "internal":
                continue
            roots.append(r)
        except Exception:
            continue

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

    def _ts_key(r: Any) -> float:
        for k in ("created_at", "updated_at"):
            try:
                v = getattr(r, k, None)
                if isinstance(v, str) and v.strip():
                    return float(datetime.fromisoformat(v).timestamp())
            except Exception:
                continue
        return 0.0

    roots.sort(key=_ts_key)
    roots = roots[-int(limit) :] if limit > 0 else roots

    out: List[Dict[str, Any]] = []
    for r in roots:
        rid = str(getattr(r, "run_id", "") or "").strip()
        vars_obj = getattr(r, "vars", None)
        input_data = dict(vars_obj) if isinstance(vars_obj, dict) else {}
        wid = str(getattr(r, "workflow_id", "") or "").strip()
        kind = _classify_turn(workflow_id=wid, vars_obj=vars_obj)
        prompt = _extract_user_prompt_from_input(input_data)
        attachments = _extract_context_attachments_from_input(input_data)
        status = getattr(getattr(r, "status", None), "value", None) or str(getattr(r, "status", "") or "")
        created_at = str(getattr(r, "created_at", "") or "").strip() or None
        updated_at = str(getattr(r, "updated_at", "") or "").strip() or None

        answer, answer_meta = _extract_answer_from_run_output(r)
        if not answer:
            try:
                ledger = ledger_store.list(rid) if hasattr(ledger_store, "list") else []
                if isinstance(ledger, list):
                    answer, answer_meta = _extract_flow_end_output_from_ledger([x for x in ledger if isinstance(x, dict)])
            except Exception:
                answer = ""
                answer_meta = None

        out.append(
            {
                "run_id": rid,
                "workflow_id": wid or None,
                "kind": kind,
                "status": str(status),
                "created_at": created_at,
                "updated_at": updated_at,
                "prompt": prompt or None,
                "attachments": attachments,
                "answer": answer or None,
                "answer_meta": answer_meta,
            }
        )
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
) -> Dict[str, Any]:
    """Export a versioned RunHistoryBundle dict (v1).

    Notes:
    - This function is pure export (no network); gateway hosts should expose it as an endpoint.
    - Payload is JSON-safe; when ArtifactStore is available, very large leaves are offloaded.
    """

    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

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
        run_ids.extend(_list_descendant_run_ids(run_store=run_store, root_run_id=rid))

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
        except Exception:
            raw = []
        records = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
        total = len(records)

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
            else:
                window = records
                cursor_start = 1
        else:
            window = records
            cursor_start = 1

        items_with_cursor: List[Dict[str, Any]] = []
        for i, rec in enumerate(window):
            items_with_cursor.append({"cursor": cursor_start + i, "record": rec})

        ledgers[rid2] = {
            "run_id": rid2,
            "total": int(total),
            "cursor_start": int(cursor_start),
            "cursor_end": int(cursor_start + len(window) - 1) if window else int(cursor_start - 1),
            "items": items_with_cursor,
        }
        _append_timeline_items(run_id2=rid2, items_with_cursor=items_with_cursor)

    # Session section (best-effort, bounded).
    session_section = None
    if include_session:
        sid = str(getattr(run, "session_id", "") or "").strip()
        if sid:
            session_section = {
                "session_id": sid,
                "turns": _best_effort_session_turns(
                    run_store=run_store,
                    ledger_store=ledger_store,
                    session_id=sid,
                    limit=max(1, int(session_turn_limit) if int(session_turn_limit) > 0 else 200),
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
        except Exception:
            pass

    # Final bundle.
    bundle: Dict[str, Any] = {
        "version": RUN_HISTORY_BUNDLE_VERSION_V1,
        "generated_at": _utc_now_iso(),
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
        "session": session_section,
    }
    return bundle
