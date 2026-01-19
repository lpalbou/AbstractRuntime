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

_ALLOWED_PREDICATE_IDS: set[str] | None = None


def _allowed_predicate_ids() -> set[str]:
    global _ALLOWED_PREDICATE_IDS
    if _ALLOWED_PREDICATE_IDS is not None:
        return _ALLOWED_PREDICATE_IDS
    try:
        from abstractsemantics import load_semantics_registry  # type: ignore
    except Exception as e:  # pragma: no cover
        # Dev convenience (monorepo):
        # `abstractsemantics` uses src-layout (abstractsemantics/src). When running from source
        # without editable installs, add that path if present.
        try:
            import sys
            from pathlib import Path

            here = Path(__file__).resolve()
            for parent in here.parents:
                candidate = parent / "abstractsemantics" / "src"
                if candidate.is_dir():
                    sys.path.insert(0, str(candidate))
                    break
            from abstractsemantics import load_semantics_registry  # type: ignore
        except Exception:
            raise RuntimeError(
                "Semantics registry is required for MEMORY_KG_ASSERT validation. "
                "Install/enable `abstractsemantics` (or disable validation explicitly in a future mode)."
            ) from e
    reg = load_semantics_registry()
    ids = getattr(reg, "predicate_ids", None)
    allowed = ids() if callable(ids) else set()
    if not isinstance(allowed, set) or not allowed:
        raise RuntimeError("Semantics registry loaded but returned no predicate ids")
    # Canonical predicate ids are compared case-insensitively to avoid accidental casing drift.
    _ALLOWED_PREDICATE_IDS = {str(x).strip().lower() for x in allowed if isinstance(x, str) and x.strip()}
    return _ALLOWED_PREDICATE_IDS


_PREDICATE_ALIAS_MAP: dict[str, str] = {
    # Common schema.org-ish synonyms → canonical minimal semantics (v4).
    #
    # Rationale: extraction models often use `schema:description` / `schema:creator` by default,
    # but the framework's canonical set is deliberately small and uses `dcterms:*` for metadata.
    "schema:description": "dcterms:description",
    "schema:creator": "dcterms:creator",
    # Awareness is a synonym for the canonical `schema:knowsAbout` predicate.
    "schema:awareness": "schema:knowsabout",
    # Structural / membership-y variants seen in practice.
    "schema:hasparent": "dcterms:ispartof",
    "schema:hasmember": "dcterms:haspart",
    # Identity / reference-ish variants (normalize to the canonical set).
    "schema:recognizedas": "skos:closematch",
    "schema:hasmemorysource": "dcterms:references",
    # Namespace drift + common typos.
    "schema:haspart": "dcterms:haspart",
    "schema:ispartof": "dcterms:ispartof",
    "dcterms:has_part": "dcterms:haspart",
    "dcterms:is_part_of": "dcterms:ispartof",
}


def _normalize_predicate_id(raw: Any) -> str:
    value = raw if isinstance(raw, str) else str(raw or "")
    value2 = value.strip()
    if not value2:
        return ""
    key = value2.lower()
    return _PREDICATE_ALIAS_MAP.get(key, value2)


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

    def _store_warning() -> Optional[str]:
        name = store.__class__.__name__
        if name == "InMemoryTripleStore":
            return (
                "AbstractMemory KG store is running in-memory (process-local, non-durable). "
                "Install `AbstractMemory[lancedb]` (or `lancedb`) for persistence across server restarts."
            )
        return None

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

        # Apply predicate alias normalization before validation so common synonyms are accepted.
        for a in assertions:
            if not isinstance(a, dict):
                continue
            if "predicate" in a:
                a["predicate"] = _normalize_predicate_id(a.get("predicate"))

        allow_custom = bool(payload.get("allow_custom_predicates") or payload.get("allow_custom"))
        allowed_predicates = _allowed_predicate_ids()
        invalid_predicates: list[str] = []
        for a in assertions:
            pred = a.get("predicate") if isinstance(a, dict) else None
            pred = pred if isinstance(pred, str) else ""
            pred2 = pred.strip()
            if not pred2:
                invalid_predicates.append("<missing>")
                continue
            pred_norm = pred2.lower()
            if pred_norm in allowed_predicates:
                continue
            if allow_custom and pred_norm.startswith("ex:"):
                continue
            invalid_predicates.append(pred2)

        if invalid_predicates:
            uniq = []
            seen: set[str] = set()
            for p in invalid_predicates:
                if p in seen:
                    continue
                uniq.append(p)
                seen.add(p)
            preview = ", ".join(uniq[:12])
            suffix = " …" if len(uniq) > 12 else ""
            return EffectOutcome.failed(
                "MEMORY_KG_ASSERT rejected unknown predicates. "
                f"Got: {preview}{suffix}. "
                "Update the extractor to use allowed semantics (or set allow_custom_predicates=true for ex:* predicates)."
            )

        scope_default = str(payload.get("scope") or "run").strip().lower() or "run"
        owner_default = payload.get("owner_id")
        owner_default = str(owner_default).strip() if isinstance(owner_default, str) and owner_default.strip() else None
        span_id_default = payload.get("span_id")
        span_id_default = str(span_id_default).strip() if isinstance(span_id_default, str) and span_id_default.strip() else None

        observed_at = now_iso()
        out_rows: list[Any] = []
        parse_errors: list[str] = []
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
            except Exception as e:
                parse_errors.append(str(e))

        if parse_errors:
            preview = " | ".join([e for e in parse_errors[:5] if str(e).strip()])
            suffix = " …" if len(parse_errors) > 5 else ""
            return EffectOutcome.failed(f"MEMORY_KG_ASSERT contained invalid assertions: {preview}{suffix}")

        if not out_rows:
            return EffectOutcome.failed("MEMORY_KG_ASSERT contained no valid assertions")

        try:
            ids = store.add(out_rows)
        except Exception as e:
            return EffectOutcome.failed(f"MEMORY_KG_ASSERT store.add failed: {e}")

        result: dict[str, Any] = {"ok": True, "count": len(ids), "assertion_ids": ids}
        warn = _store_warning()
        if warn:
            result["warnings"] = [warn]
        return EffectOutcome.completed(result)

    def _handle_query(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = dict(effect.payload or {})

        recall_level_raw = payload.get("recall_level")
        if recall_level_raw is None:
            recall_level_raw = payload.get("recallLevel")
        try:
            from abstractruntime.memory.recall_levels import parse_recall_level, policy_for

            recall_level = parse_recall_level(recall_level_raw)
        except Exception as e:
            return EffectOutcome.failed(str(e))

        recall_warnings: list[str] = []
        recall_effort: dict[str, Any] = {}

        scope_raw = payload.get("scope")
        scope = str(scope_raw or "run").strip().lower() or "run"
        owner_id_raw = payload.get("owner_id")
        owner_id = str(owner_id_raw).strip() if isinstance(owner_id_raw, str) and owner_id_raw.strip() else None

        if scope not in {"run", "session", "global", "all"}:
            return EffectOutcome.completed({"ok": False, "count": 0, "items": [], "error": f"Unknown memory scope: {scope}"})

        query_text_raw = payload.get("query_text")
        is_semantic = isinstance(query_text_raw, str) and query_text_raw.strip().lower() != ""

        # Apply recall budgets when explicitly requested.
        limit_value: int = 100
        min_score_value: Optional[float] = None
        budget_value: Optional[int] = None
        if recall_level is not None:
            pol = policy_for(recall_level)

            raw_limit = payload.get("limit")
            if raw_limit is None:
                limit_value = pol.kg.limit_default
            else:
                try:
                    limit_value = int(raw_limit)
                except Exception:
                    limit_value = pol.kg.limit_default
            if limit_value < 1:
                limit_value = 1
            if limit_value > pol.kg.limit_max:
                recall_warnings.append(
                    f"recall_level={recall_level.value}: clamped limit from {limit_value} to {pol.kg.limit_max}"
                )
                limit_value = pol.kg.limit_max

            if is_semantic:
                raw_ms = payload.get("min_score")
                if raw_ms is None:
                    min_score_value = float(pol.kg.min_score_default)
                else:
                    try:
                        min_score_value = float(raw_ms)
                    except Exception:
                        min_score_value = float(pol.kg.min_score_default)
                if not (min_score_value == min_score_value):  # NaN
                    min_score_value = float(pol.kg.min_score_default)
                if min_score_value < pol.kg.min_score_floor:
                    recall_warnings.append(
                        f"recall_level={recall_level.value}: raised min_score from {min_score_value} to {pol.kg.min_score_floor}"
                    )
                    min_score_value = float(pol.kg.min_score_floor)
            else:
                min_score_value = None

            raw_budget = payload.get("max_input_tokens")
            if raw_budget is None:
                raw_budget = payload.get("max_in_tokens")
            if raw_budget is None:
                budget_value = pol.kg.max_input_tokens_default
            else:
                try:
                    bf = float(raw_budget)
                except Exception:
                    bf = None
                if bf is None or not (bf == bf):  # NaN
                    budget_value = pol.kg.max_input_tokens_default
                elif bf == 0:
                    # Explicitly disable packetization/Active Memory packing.
                    budget_value = 0
                elif bf < 1:
                    budget_value = pol.kg.max_input_tokens_default
                else:
                    budget_value = int(bf)

            if budget_value > 0 and budget_value > pol.kg.max_input_tokens_max:
                recall_warnings.append(
                    f"recall_level={recall_level.value}: clamped max_input_tokens from {budget_value} to {pol.kg.max_input_tokens_max}"
                )
                budget_value = pol.kg.max_input_tokens_max

            recall_effort = {
                "recall_level": recall_level.value,
                "applied": {
                    "limit": int(limit_value),
                    "min_score": float(min_score_value) if min_score_value is not None else None,
                    "max_input_tokens": int(budget_value) if budget_value is not None else None,
                },
            }
        else:
            # Backward-compatible defaults (no policy).
            try:
                limit_value = int(payload.get("limit") or 100)
            except Exception:
                limit_value = 100
            limit_value = max(1, min(limit_value, 10_000))
            raw_ms = payload.get("min_score")
            if raw_ms is not None:
                try:
                    v = float(raw_ms)
                except Exception:
                    v = None
                if v is not None and (v == v):
                    min_score_value = v

            raw_budget = payload.get("max_input_tokens")
            if raw_budget is None:
                raw_budget = payload.get("max_in_tokens")
            if raw_budget is not None and not isinstance(raw_budget, bool):
                try:
                    b = int(float(raw_budget))
                except Exception:
                    b = None
                if isinstance(b, int) and b > 0:
                    budget_value = b

        def _one_query(*, scope_label: str, owner_id2: str) -> list[Any]:
            q = TripleQuery(
                subject=str(payload.get("subject")).strip() if isinstance(payload.get("subject"), str) else None,
                predicate=_normalize_predicate_id(payload.get("predicate")) if payload.get("predicate") is not None else None,
                object=str(payload.get("object")).strip() if isinstance(payload.get("object"), str) else None,
                scope=scope_label,
                owner_id=owner_id2,
                since=str(payload.get("since")).strip() if isinstance(payload.get("since"), str) else None,
                until=str(payload.get("until")).strip() if isinstance(payload.get("until"), str) else None,
                active_at=str(payload.get("active_at")).strip() if isinstance(payload.get("active_at"), str) else None,
                query_text=str(payload.get("query_text")).strip() if isinstance(payload.get("query_text"), str) else None,
                limit=int(limit_value),
                order=str(payload.get("order") or "desc"),
                min_score=min_score_value,
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

        # Ordering:
        # - Pattern queries (no query_text/query_vector): observed_at (asc/desc)
        # - Semantic queries: preserve similarity ranking via `_retrieval.score` (desc),
        #   tie-break by observed_at desc for stability.
        query_text_raw = payload.get("query_text")
        is_semantic = isinstance(query_text_raw, str) and query_text_raw.strip().lower() != ""

        def _observed_at_key(d: dict[str, Any]) -> str:
            return str(d.get("observed_at") or "")

        if is_semantic:
            def _score(d: dict[str, Any]) -> float:
                attrs = d.get("attributes")
                if isinstance(attrs, dict):
                    ret = attrs.get("_retrieval")
                    if isinstance(ret, dict):
                        s = ret.get("score")
                        if isinstance(s, (int, float)):
                            return float(s)
                return float("-inf")

            out_items.sort(key=lambda d: (_score(d), _observed_at_key(d)), reverse=True)
        else:
            order = str(payload.get("order") or "desc").strip().lower()
            reverse = order != "asc"
            out_items.sort(key=_observed_at_key, reverse=reverse)
        out_items = out_items[: max(1, min(int(limit_value), 10_000))]

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
        if recall_effort:
            result["effort"] = recall_effort

        # Optional: packetize + pack into an LLM-friendly Active Memory block.
        #
        # This is used by `ltm-ai-kg-map-to-active` and chat-like flows that inject
        # KG recall into the system prompt without blowing up token budgets.
        budget = budget_value
        if isinstance(budget, int) and budget > 0:
            try:
                model_name = payload.get("model")
                model_name = str(model_name).strip() if isinstance(model_name, str) and model_name.strip() else None

                from abstractruntime.memory.kg_packets import packetize_assertions, pack_active_memory_text

                packets_all = packetize_assertions(out_items)
                active_text, kept_packets, est_tokens, dropped = pack_active_memory_text(
                    packets_all,
                    scope=scope,
                    max_input_tokens=int(budget),
                    model=model_name,
                    include_scores=bool(is_semantic),
                )

                result["packets_version"] = 0
                result["packets"] = kept_packets
                result["packed_count"] = len(kept_packets)
                result["active_memory_text"] = active_text
                result["estimated_tokens"] = est_tokens
                result["dropped"] = dropped
            except Exception as e:
                return EffectOutcome.failed(f"MEMORY_KG_QUERY packetize failed: {e}")

        warn = _store_warning()
        if warn:
            warnings = result.get("warnings")
            if isinstance(warnings, list):
                warnings.append(warn)
            else:
                result["warnings"] = [warn]
        if errors:
            cur = result.get("warnings")
            merged: list[str] = []
            if isinstance(cur, list):
                merged.extend([str(x) for x in cur if str(x).strip()])
            merged.extend([e for e in errors if str(e).strip()])
            result["warnings"] = merged
        if recall_warnings:
            cur = result.get("warnings")
            merged: list[str] = []
            if isinstance(cur, list):
                merged.extend([str(x) for x in cur if str(x).strip()])
            merged.extend([w for w in recall_warnings if str(w).strip()])
            result["warnings"] = merged
        return EffectOutcome.completed(result)

    return {
        EffectType.MEMORY_KG_ASSERT: _handle_assert,
        EffectType.MEMORY_KG_QUERY: _handle_query,
    }
