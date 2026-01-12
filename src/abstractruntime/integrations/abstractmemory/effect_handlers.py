from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Callable, Dict, Iterable, Optional

from ...core.models import Effect, EffectType, RunState
from ...core.runtime import EffectHandler, EffectOutcome
from ...storage.base import RunStore

_DEFAULT_GLOBAL_MEMORY_RUN_ID = "global_memory"
_DEFAULT_SESSION_MEMORY_RUN_PREFIX = "session_memory_"
_SAFE_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _global_memory_owner_id() -> str:
    rid = os.environ.get("ABSTRACTRUNTIME_GLOBAL_MEMORY_RUN_ID")
    rid = str(rid or "").strip()
    if rid and _SAFE_RUN_ID_PATTERN.match(rid):
        return rid
    return _DEFAULT_GLOBAL_MEMORY_RUN_ID


def _session_memory_owner_id(session_id: str) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    if _SAFE_RUN_ID_PATTERN.match(sid):
        rid = f"{_DEFAULT_SESSION_MEMORY_RUN_PREFIX}{sid}"
        if _SAFE_RUN_ID_PATTERN.match(rid):
            return rid
    digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:32]
    return f"{_DEFAULT_SESSION_MEMORY_RUN_PREFIX}sha_{digest}"


def _resolve_run_tree_root_run_id(run: RunState, *, run_store: RunStore) -> str:
    cur = run
    seen: set[str] = set()
    while True:
        parent_id = getattr(cur, "parent_run_id", None)
        if not isinstance(parent_id, str) or not parent_id.strip():
            return str(getattr(cur, "run_id", "") or "")
        pid = parent_id.strip()
        if pid in seen:
            return str(getattr(cur, "run_id", "") or "")
        seen.add(pid)
        parent = run_store.load(pid)
        if parent is None:
            return str(getattr(cur, "run_id", "") or "")
        cur = parent


def resolve_scope_owner_id(run: RunState, *, scope: str, run_store: RunStore) -> str:
    s = str(scope or "").strip().lower() or "run"
    if s == "run":
        return str(getattr(run, "run_id", "") or "")
    if s == "session":
        sid = getattr(run, "session_id", None)
        if isinstance(sid, str) and sid.strip():
            return _session_memory_owner_id(sid.strip())
        return _resolve_run_tree_root_run_id(run, run_store=run_store)
    if s == "global":
        return _global_memory_owner_id()
    raise ValueError(f"Unknown memory scope: {scope}")


def _import_abstractmemory():
    try:
        from abstractmemory import TripleAssertion, TripleQuery  # type: ignore

        return TripleAssertion, TripleQuery
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "AbstractMemory is not available. Install and ensure it is importable (e.g. `pip install -e abstractmemory`)."
        ) from e


def build_memory_kg_effect_handlers(
    *,
    store: Any,
    run_store: RunStore,
    now_iso: Callable[[], str],
) -> Dict[EffectType, EffectHandler]:
    """Build effect handlers for `memory_kg_*` effects.

    These are host-provided handlers that bridge VisualFlow nodes to AbstractMemory stores.
    """
    TripleAssertion, TripleQuery = _import_abstractmemory()

    # Avoid `isinstance` checks against Protocols; use duck-typing instead.
    if not callable(getattr(store, "add", None)) or not callable(getattr(store, "query", None)):
        raise TypeError("store must provide add(...) and query(...) methods (abstractmemory TripleStore contract)")

    def _normalize_assertions(raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            return [dict(raw)]
        if isinstance(raw, list):
            out: list[dict[str, Any]] = []
            for x in raw:
                if isinstance(x, dict):
                    out.append(dict(x))
            return out
        return []

    def _handle_assert(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = dict(effect.payload or {})

        raw_assertions = payload.get("assertions")
        if raw_assertions is None:
            raw_assertions = payload.get("triples")
        if raw_assertions is None:
            raw_assertions = payload.get("items")
        if raw_assertions is None:
            return EffectOutcome.failed("MEMORY_KG_ASSERT requires payload.assertions (list[object])")

        # Empty assertion lists are a valid no-op (e.g. extractor found no facts).
        # This should not fail the entire run/workflow.
        if isinstance(raw_assertions, list) and len(raw_assertions) == 0:
            return EffectOutcome.completed({"ok": True, "count": 0, "assertion_ids": [], "skipped": True})

        assertions = _normalize_assertions(raw_assertions)
        if not assertions:
            return EffectOutcome.failed("MEMORY_KG_ASSERT requires payload.assertions (list[object])")

        scope_default = str(payload.get("scope") or "run").strip().lower() or "run"
        owner_default = payload.get("owner_id")
        owner_default = str(owner_default).strip() if isinstance(owner_default, str) and owner_default.strip() else None
        span_id_default = payload.get("span_id")
        span_id_default = str(span_id_default).strip() if isinstance(span_id_default, str) and span_id_default.strip() else None

        observed_at = now_iso()
        out_rows: list[Any] = []
        for a in assertions:
            try:
                merged = dict(a)
                if "scope" not in merged:
                    merged["scope"] = scope_default
                if "owner_id" not in merged:
                    merged["owner_id"] = owner_default or resolve_scope_owner_id(run, scope=merged.get("scope") or scope_default, run_store=run_store)
                if "observed_at" not in merged:
                    merged["observed_at"] = observed_at
                provenance = merged.get("provenance")
                prov2: dict[str, Any] = dict(provenance) if isinstance(provenance, dict) else {}
                if span_id_default and "span_id" not in prov2:
                    prov2["span_id"] = span_id_default
                prov2.setdefault("writer_run_id", str(getattr(run, "run_id", "") or ""))
                prov2.setdefault("writer_workflow_id", str(getattr(run, "workflow_id", "") or ""))
                merged["provenance"] = prov2
                out_rows.append(TripleAssertion.from_dict(merged))
            except Exception:
                continue

        if not out_rows:
            return EffectOutcome.failed("MEMORY_KG_ASSERT contained no valid assertions")

        try:
            ids = store.add(out_rows)
        except Exception as e:
            return EffectOutcome.failed(f"MEMORY_KG_ASSERT store.add failed: {e}")

        return EffectOutcome.completed({"ok": True, "count": len(ids), "assertion_ids": ids})

    def _handle_query(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = dict(effect.payload or {})

        scope_raw = payload.get("scope")
        scope = str(scope_raw or "run").strip().lower() or "run"
        owner_id_raw = payload.get("owner_id")
        owner_id = str(owner_id_raw).strip() if isinstance(owner_id_raw, str) and owner_id_raw.strip() else None

        if scope not in {"run", "session", "global", "all"}:
            return EffectOutcome.completed({"ok": False, "count": 0, "items": [], "error": f"Unknown memory scope: {scope}"})

        def _one_query(*, scope_label: str, owner_id2: str) -> list[Any]:
            q = TripleQuery(
                subject=str(payload.get("subject")).strip() if isinstance(payload.get("subject"), str) else None,
                predicate=str(payload.get("predicate")).strip() if isinstance(payload.get("predicate"), str) else None,
                object=str(payload.get("object")).strip() if isinstance(payload.get("object"), str) else None,
                scope=scope_label,
                owner_id=owner_id2,
                since=str(payload.get("since")).strip() if isinstance(payload.get("since"), str) else None,
                until=str(payload.get("until")).strip() if isinstance(payload.get("until"), str) else None,
                active_at=str(payload.get("active_at")).strip() if isinstance(payload.get("active_at"), str) else None,
                query_text=str(payload.get("query_text")).strip() if isinstance(payload.get("query_text"), str) else None,
                limit=int(payload.get("limit") or 100),
                order=str(payload.get("order") or "desc"),
            )
            return store.query(q)

        results: list[Any] = []
        errors: list[str] = []
        if scope == "all":
            owners: list[tuple[str, str]] = []
            try:
                owners.append(("run", resolve_scope_owner_id(run, scope="run", run_store=run_store)))
                owners.append(("session", resolve_scope_owner_id(run, scope="session", run_store=run_store)))
                owners.append(("global", resolve_scope_owner_id(run, scope="global", run_store=run_store)))
            except Exception as e:
                errors.append(str(e))
            for label, oid in owners:
                try:
                    results.extend(_one_query(scope_label=label, owner_id2=oid))
                except Exception as e:
                    errors.append(f"{label}: {e}")
        else:
            try:
                owner = owner_id or resolve_scope_owner_id(run, scope=scope, run_store=run_store)
                results.extend(_one_query(scope_label=scope, owner_id2=owner))
            except Exception as e:
                errors.append(str(e))

        # Normalize output to JSON-safe dicts.
        out_items: list[dict[str, Any]] = []
        for a in results:
            to_dict = getattr(a, "to_dict", None)
            if callable(to_dict):
                d = to_dict()
                if isinstance(d, dict):
                    out_items.append(d)

        # Sort by observed_at desc (stable default).
        out_items.sort(key=lambda d: str(d.get("observed_at") or ""), reverse=True)
        limit = int(payload.get("limit") or 100)
        limit = max(1, min(limit, 10_000))
        out_items = out_items[:limit]

        if not out_items and errors:
            return EffectOutcome.completed(
                {
                    "ok": False,
                    "count": 0,
                    "items": [],
                    "error": " | ".join([e for e in errors if str(e).strip()]),
                }
            )

        result: dict[str, Any] = {"ok": True, "count": len(out_items), "items": out_items}
        if errors:
            result["warnings"] = [e for e in errors if str(e).strip()]
        return EffectOutcome.completed(result)

    return {
        EffectType.MEMORY_KG_ASSERT: _handle_assert,
        EffectType.MEMORY_KG_QUERY: _handle_query,
    }
