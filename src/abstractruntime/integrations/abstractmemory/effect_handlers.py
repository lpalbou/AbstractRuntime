from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional

from ...core.models import Effect, EffectType, RunState
from ...core.runtime import EffectHandler, EffectOutcome
from ...storage.base import RunStore
from .memory_facade import MemoryFacade

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
        raise RuntimeError(
            "Semantics registry is required for MEMORY_KG_ASSERT validation. "
            "Install `abstractsemantics` into the same environment as the runtime/gateway (e.g. `pip install -e ./abstractsemantics`)."
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


_EX_PREFIX_RE = re.compile(r"^ex:", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DASH_RUN_RE = re.compile(r"-{2,}")


def _slugify_kebab(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    folded = (
        unicodedata.normalize("NFKD", raw)
        .encode("ascii", "ignore")
        .decode("ascii", errors="ignore")
        .strip()
    )
    lowered = folded.lower()
    # Replace any run of non-alphanumerics with a dash.
    dashed = _NON_ALNUM_RE.sub("-", lowered)
    dashed = _DASH_RUN_RE.sub("-", dashed).strip("-")
    return dashed


def _normalize_ex_curie(raw: Any) -> Any:
    """Normalize `ex:` instance identifiers to `ex:{kind}-{kebab-case}` formatting.

    Notes:
    - We intentionally do not attempt semantic entity resolution here (synonyms, honorific stripping, etc).
      This is a formatting normalization layer to reduce accidental drift (spaces/underscores/punctuation).
    - Returns the original value when normalization cannot be applied safely.
    """
    if not isinstance(raw, str):
        return raw
    value = raw.strip()
    if not value or not _EX_PREFIX_RE.match(value):
        return raw
    local = value.split(":", 1)[1].strip()
    slug = _slugify_kebab(local)
    if not slug:
        return raw
    return f"ex:{slug}"


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


def derived_owner_ids(run: RunState, *, run_store: RunStore) -> set[str]:
    """The owner partitions a run may write: its own (`run`), its session's (`session`), the plane's (`global`).
    A payload `owner_id` outside this set is another partition (another session, another run) and is refused: a
    workflow names WHICH of its own partitions, never whose (architect B1; the full fix, owner = the principal, is
    an operator decision, ADR 0009)."""
    out: set[str] = set()
    for scope in ("run", "session", "global"):
        try:
            owner = resolve_scope_owner_id(run, scope=scope, run_store=run_store)
        except Exception:  # noqa: BLE001 - an unresolvable partition is simply not one this run may write
            continue
        if owner:
            out.add(owner)
    return out


def _import_abstractmemory():
    try:
        from abstractmemory import TripleAssertion, TripleQuery  # type: ignore

        return TripleAssertion, TripleQuery
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "AbstractMemory is not available. Install and ensure it is importable (e.g. `pip install -e abstractmemory`)."
        ) from e


# MEMORY_REVISE payload keys (docs/integrations/abstractmemory.md). `type` / `resume_to_node` are VisualFlow
# transport keys the compiler adds to every pending effect; anything else is refused, never silently dropped.
_REVISE_KEYS = frozenset({
    "ops", "scope", "owner_id", "cause", "revision_id", "occurred_at", "summary", "artifact_ids", "provenance",
})
_REVISE_TRANSPORT_KEYS = frozenset({"type", "resume_to_node"})
# The caller's part of a revision's provenance; `run`/`step` refs are runtime-owned (stamped from the step context).
_REVISE_PROVENANCE_KEYS = frozenset({"sources", "derived_from", "actor", "reader", "reason", "extra"})
_RUNTIME_OWNED_REF_KINDS = frozenset({"run", "step"})


def _memory_graph_for(store: Any, journal: Any) -> tuple[Any, Optional[str]]:
    """(MemoryGraph, None) when `store` is a graph home, else (None, why). Built ONCE per builder call: opening a
    graph creates its tables in the store's file."""
    from abstractmemory import InMemoryTripleStore, SQLiteTripleStore  # type: ignore
    from abstractmemory.graph import MemoryGraph  # type: ignore

    if isinstance(store, (SQLiteTripleStore, InMemoryTripleStore)):
        return MemoryGraph(store, journal), None
    return None, (
        f"MEMORY_REVISE needs a graph home (SQLiteTripleStore, or InMemoryTripleStore for tests); this host's KG store "
        f"is {type(store).__name__}, which has no revision log yet (AbstractMemory backlog 0057: one SQLite home per "
        "plane)"
    )


def build_memory_kg_effect_handlers(
    *,
    store: Any,
    run_store: RunStore,
    now_iso: Callable[[], str],
    journal: Any = None,
    runtime_id: Optional[str] = None,
    artifact_store: Any = None,
) -> Dict[EffectType, EffectHandler]:
    """Build effect handlers for `memory_kg_*` effects and `MEMORY_REVISE`.

    These are host-provided handlers that bridge VisualFlow nodes to AbstractMemory stores.
    `MEMORY_REVISE` applies one revision through `abstractmemory.graph.MemoryGraph` over `store` (a graph home:
    SQLiteTripleStore / InMemoryTripleStore; any other store gets a failed outcome naming AbstractMemory backlog
    0057). `journal` receives the `ClosureRecord` of each `close` op (None: the revision and the history row still
    carry the close; ranked retrieval folds closures only from a journal). `runtime_id` (the plane's or the home's)
    is written on every SourceRef the handlers stamp. `artifact_store` (the plane's) lets `MEMORY_REVISE` check that
    the artifacts a revision names exist; without one, a revision naming artifacts is refused (it could not be
    grounded honestly).
    """
    TripleAssertion, TripleQuery = _import_abstractmemory()
    memory_graph, graph_unavailable = _memory_graph_for(store, journal)

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
        attributes_defaults_raw = payload.get("attributes_defaults")
        attributes_defaults = dict(attributes_defaults_raw) if isinstance(attributes_defaults_raw, dict) else {}

        observed_at = now_iso()
        out_rows: list[Any] = []
        parse_errors: list[str] = []
        # Ledger provenance (ADR 0009): the MEMORY_KG_ASSERT step that wrote each assertion; `span_id` -> artifact.
        facade = MemoryFacade(run_store=run_store, runtime_id=runtime_id)
        writer_run_id = str(getattr(run, "run_id", "") or "")
        for a in assertions:
            try:
                merged = dict(a)
                if "subject" in merged:
                    merged["subject"] = _normalize_ex_curie(merged.get("subject"))
                if "object" in merged:
                    merged["object"] = _normalize_ex_curie(merged.get("object"))
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
                prov2.setdefault("writer_run_id", writer_run_id)
                prov2.setdefault("writer_workflow_id", str(getattr(run, "workflow_id", "") or ""))
                merged["provenance"] = facade.stamp_current_step(prov2, run_id=writer_run_id)
                if attributes_defaults:
                    attrs = merged.get("attributes")
                    attrs2: dict[str, Any] = dict(attrs) if isinstance(attrs, dict) else {}
                    for k, v in attributes_defaults.items():
                        if k not in attrs2:
                            attrs2[k] = v
                    merged["attributes"] = attrs2
                out_rows.append(TripleAssertion.from_dict(merged))
            except Exception as e:
                parse_errors.append(str(e))

        foreign_owners = sorted({str(a.owner_id) for a in out_rows} - derived_owner_ids(run, run_store=run_store))
        if foreign_owners:
            return EffectOutcome.failed(
                f"MEMORY_KG_ASSERT: owner_id {foreign_owners} is not a partition this run may write "
                f"(its run, its session, or the global owner); nothing was written"
            )

        if parse_errors:
            preview = " | ".join([e for e in parse_errors[:5] if str(e).strip()])
            suffix = " …" if len(parse_errors) > 5 else ""
            return EffectOutcome.failed(f"MEMORY_KG_ASSERT contained invalid assertions: {preview}{suffix}")

        if not out_rows:
            return EffectOutcome.failed("MEMORY_KG_ASSERT contained no valid assertions")

        dedupe_raw = payload.get("dedupe")
        if dedupe_raw is None:
            dedupe_raw = payload.get("deduplicate")
        dedupe = True
        if isinstance(dedupe_raw, bool):
            dedupe = dedupe_raw
        elif isinstance(dedupe_raw, str) and dedupe_raw.strip():
            dedupe = dedupe_raw.strip().lower() not in {"0", "false", "no", "off"}

        skipped_duplicates = 0
        # A duplicate row is not written, so the NEW sources of this assertion (its step, its span) are not recorded
        # on the existing row: say so in the outcome instead of dropping them silently. Recording a second source
        # on an existing memory is the graph's job (a revision that adopts the row), a follow-up of backlog 0853.
        skipped_rows: list[dict[str, Any]] = []
        to_insert = out_rows
        if dedupe:
            to_insert = []
            for a in out_rows:
                try:
                    existing = store.query(
                        TripleQuery(
                            subject=a.subject,
                            predicate=a.predicate,
                            object=a.object,
                            scope=a.scope,
                            owner_id=a.owner_id,
                            limit=1,
                            order="desc",
                        )
                    )
                except Exception:
                    # Dedupe is best-effort; if the store can't query, fall back to insert.
                    existing = []
                if existing:
                    skipped_duplicates += 1
                    prov_new = a.provenance if isinstance(a.provenance, dict) else {}
                    skipped_rows.append({
                        "subject": a.subject, "predicate": a.predicate, "object": a.object,
                        "scope": a.scope, "owner_id": a.owner_id,
                        "existing_assertion_id": getattr(existing[0], "assertion_id", None),
                        "lost_sources": list(prov_new.get("sources") or []),
                    })
                    continue
                to_insert.append(a)

        ids: list[str] = []
        if to_insert:
            try:
                ids = store.add(to_insert)
            except Exception as e:
                return EffectOutcome.failed(f"MEMORY_KG_ASSERT store.add failed: {e}")

        result: dict[str, Any] = {
            "ok": True,
            "count": len(ids),
            "assertion_ids": ids,
            "count_attempted": len(out_rows),
            "skipped_duplicates": skipped_duplicates,
        }
        if skipped_rows:
            result["skipped_duplicate_rows"] = skipped_rows
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
            subject_raw = str(payload.get("subject")).strip() if isinstance(payload.get("subject"), str) else None
            object_raw = str(payload.get("object")).strip() if isinstance(payload.get("object"), str) else None
            q = TripleQuery(
                subject=_normalize_ex_curie(subject_raw) if isinstance(subject_raw, str) else subject_raw,
                predicate=_normalize_predicate_id(payload.get("predicate")) if payload.get("predicate") is not None else None,
                object=_normalize_ex_curie(object_raw) if isinstance(object_raw, str) else object_raw,
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

    def _handle_resolve(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """Resolve candidate entity ids by label (+ optional rdf:type filter)."""
        del default_next_node
        payload = dict(effect.payload or {})

        label_raw = payload.get("label")
        if label_raw is None:
            label_raw = payload.get("query")
        if label_raw is None:
            label_raw = payload.get("query_text")

        label = str(label_raw or "").strip()
        if not label:
            return EffectOutcome.completed({"ok": False, "count": 0, "candidates": [], "error": "label is required"})

        expected_type_raw = payload.get("expected_type")
        if expected_type_raw is None:
            expected_type_raw = payload.get("expectedType")
        if expected_type_raw is None:
            expected_type_raw = payload.get("type")

        expected_type = str(expected_type_raw or "").strip().lower() if expected_type_raw is not None else ""
        expected_type = expected_type if expected_type else ""

        recall_level_raw = payload.get("recall_level")
        if recall_level_raw is None:
            recall_level_raw = payload.get("recallLevel")
        try:
            from abstractruntime.memory.recall_levels import parse_recall_level

            recall_level = parse_recall_level(recall_level_raw)
        except Exception as e:
            return EffectOutcome.failed(str(e))

        scope_raw = payload.get("scope")
        scope = str(scope_raw or "run").strip().lower() or "run"
        owner_id_raw = payload.get("owner_id")
        owner_id = str(owner_id_raw).strip() if isinstance(owner_id_raw, str) and owner_id_raw.strip() else None

        if scope not in {"run", "session", "global", "all"}:
            return EffectOutcome.completed({"ok": False, "count": 0, "candidates": [], "error": f"Unknown memory scope: {scope}"})

        def _normalize_label(value: str) -> str:
            # Collapse whitespace + lowercase to match extractor normalization.
            return " ".join(str(value or "").split()).strip().lower()

        label_norm = _normalize_label(label)

        # Budgets / behavior flags.
        max_candidates_raw = payload.get("max_candidates")
        if max_candidates_raw is None:
            max_candidates_raw = payload.get("maxCandidates")
        if max_candidates_raw is None:
            max_candidates_raw = payload.get("limit")

        default_max = 5
        semantic_mode: str = "fallback"
        if recall_level is not None:
            if recall_level.value == "urgent":
                default_max = 3
                semantic_mode = "none"
            elif recall_level.value == "deep":
                default_max = 10
                semantic_mode = "always"
            else:
                default_max = 5
                semantic_mode = "fallback"

        try:
            max_candidates = int(float(max_candidates_raw)) if max_candidates_raw is not None else int(default_max)
        except Exception:
            max_candidates = int(default_max)
        if max_candidates < 1:
            max_candidates = 1
        if max_candidates > 50:
            max_candidates = 50

        include_semantic_raw = payload.get("include_semantic")
        if include_semantic_raw is None:
            include_semantic_raw = payload.get("includeSemantic")
        include_semantic = None
        if include_semantic_raw is not None:
            include_semantic = bool(include_semantic_raw) if isinstance(include_semantic_raw, bool) else None
        if include_semantic is not None:
            semantic_mode = "fallback" if include_semantic else "none"

        min_score_raw = payload.get("min_score")
        if min_score_raw is None:
            min_score_raw = payload.get("minScore")
        min_score_value: Optional[float] = None
        if min_score_raw is not None and not isinstance(min_score_raw, bool):
            try:
                min_score_value = float(min_score_raw)
            except Exception:
                min_score_value = None

        label_predicates = [
            "schema:name",
            "skos:preflabel",
            "skos:altlabel",
            "dcterms:title",
            "dcterms:identifier",
        ]

        warnings: list[str] = []

        def _score_from_dict(d: dict[str, Any]) -> Optional[float]:
            attrs = d.get("attributes")
            if isinstance(attrs, dict):
                ret = attrs.get("_retrieval")
                if isinstance(ret, dict):
                    s = ret.get("score")
                    if isinstance(s, (int, float)):
                        return float(s)
            return None

        def _one_query(
            *,
            scope_label: str,
            owner_id2: str,
            predicate_id: str,
            object_value: Optional[str] = None,
            query_text: Optional[str] = None,
        ) -> list[dict[str, Any]]:
            q = TripleQuery(
                subject=None,
                predicate=predicate_id,
                object=object_value,
                scope=scope_label,
                owner_id=owner_id2,
                query_text=query_text,
                limit=max(10, min(max_candidates * 10, 200)),
                order="desc",
                min_score=min_score_value,
            )
            rows = store.query(q)
            out: list[dict[str, Any]] = []
            for a in rows:
                to_dict = getattr(a, "to_dict", None)
                if callable(to_dict):
                    d = to_dict()
                    if isinstance(d, dict):
                        out.append(d)
            return out

        owners: list[tuple[str, str]] = []
        if scope == "all":
            try:
                owners.append(("run", resolve_scope_owner_id(run, scope="run", run_store=run_store)))
                owners.append(("session", resolve_scope_owner_id(run, scope="session", run_store=run_store)))
                owners.append(("global", resolve_scope_owner_id(run, scope="global", run_store=run_store)))
            except Exception as e:
                warnings.append(str(e))
        else:
            try:
                owners.append((scope, owner_id or resolve_scope_owner_id(run, scope=scope, run_store=run_store)))
            except Exception as e:
                warnings.append(str(e))

        exact_hits: list[dict[str, Any]] = []
        for scope_label, owner2 in owners:
            for pid in label_predicates:
                try:
                    exact_hits.extend(_one_query(scope_label=scope_label, owner_id2=owner2, predicate_id=pid, object_value=label_norm))
                except Exception as e:
                    warnings.append(f"{scope_label}.{pid}: {e}")

        semantic_hits: list[dict[str, Any]] = []
        if semantic_mode != "none":
            allow = semantic_mode == "always" or (semantic_mode == "fallback" and not exact_hits)
            if allow:
                for scope_label, owner2 in owners:
                    for pid in label_predicates:
                        try:
                            semantic_hits.extend(_one_query(scope_label=scope_label, owner_id2=owner2, predicate_id=pid, query_text=label_norm))
                        except Exception as e:
                            warnings.append(f"semantic {scope_label}.{pid}: {e}")

        # Candidate aggregation.
        cand_by_key: dict[tuple[str, Optional[str], str], dict[str, Any]] = {}
        for d in list(exact_hits) + list(semantic_hits):
            subj = d.get("subject")
            if not isinstance(subj, str) or not subj.strip().lower().startswith("ex:"):
                continue
            scope_val = str(d.get("scope") or "").strip().lower() or "run"
            owner_val = d.get("owner_id") if isinstance(d.get("owner_id"), str) and d.get("owner_id").strip() else None
            key = (scope_val, owner_val, subj.strip())

            score = _score_from_dict(d)
            observed_at = str(d.get("observed_at") or "")
            obj_val = d.get("object")
            label_val = obj_val if isinstance(obj_val, str) and obj_val.strip() else label_norm

            c = cand_by_key.get(key)
            if c is None:
                c = {
                    "id": key[2],
                    "label": label_val,
                    "scope": key[0],
                    "owner_id": key[1],
                    "score": score if score is not None else (1.0 if d in exact_hits else None),
                    "_observed_at": observed_at,
                    "_evidence": [d],
                }
                cand_by_key[key] = c
            else:
                if isinstance(label_val, str) and label_val.strip() and not str(c.get("label") or "").strip():
                    c["label"] = label_val
                cur_score = c.get("score")
                if score is not None and (cur_score is None or (isinstance(cur_score, (int, float)) and score > float(cur_score))):
                    c["score"] = float(score)
                if observed_at and observed_at > str(c.get("_observed_at") or ""):
                    c["_observed_at"] = observed_at
                ev = c.get("_evidence")
                if isinstance(ev, list) and len(ev) < 3:
                    ev.append(d)

        candidates = list(cand_by_key.values())
        candidates.sort(
            key=lambda c: (
                float(c.get("score")) if isinstance(c.get("score"), (int, float)) else -1.0,
                str(c.get("_observed_at") or ""),
            ),
            reverse=True,
        )

        # Bound the number of rdf:type lookups (and the final output).
        candidates = candidates[: max_candidates * 2]

        out: list[dict[str, Any]] = []
        for c in candidates:
            cid = c.get("id")
            if not isinstance(cid, str) or not cid.strip():
                continue
            scope_val = str(c.get("scope") or "run").strip().lower() or "run"
            owner_val = c.get("owner_id") if isinstance(c.get("owner_id"), str) and c.get("owner_id").strip() else None

            type_rows: list[dict[str, Any]] = []
            try:
                q = TripleQuery(
                    subject=cid.strip(),
                    predicate="rdf:type",
                    object=None,
                    scope=scope_val,
                    owner_id=owner_val,
                    limit=50,
                    order="desc",
                )
                for a in store.query(q):
                    to_dict = getattr(a, "to_dict", None)
                    if callable(to_dict):
                        d = to_dict()
                        if isinstance(d, dict):
                            type_rows.append(d)
            except Exception as e:
                warnings.append(f"type {scope_val}.{cid}: {e}")

            types: list[str] = []
            for d in type_rows:
                obj = d.get("object")
                if isinstance(obj, str) and obj.strip():
                    types.append(obj.strip().lower())
            types = list(dict.fromkeys(types))  # stable unique

            if expected_type:
                if expected_type not in types:
                    continue

            out.append(
                {
                    "id": cid.strip(),
                    "label": str(c.get("label") or label_norm),
                    "types": types,
                    "scope": scope_val,
                    "owner_id": owner_val,
                    "score": c.get("score"),
                    "evidence": c.get("_evidence"),
                }
            )
            if len(out) >= max_candidates:
                break

        result: dict[str, Any] = {"ok": True, "count": len(out), "candidates": out, "raw": {"label": label_norm, "expected_type": expected_type or None}}
        if warnings:
            result["warnings"] = [w for w in warnings if str(w).strip()]
        return EffectOutcome.completed(result)

    def _handle_revise(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """One revision of the mind, as a workflow step (ADR 0009; AbstractMemory ADR 0003).

        The payload is a Revision (contract §3) WITHOUT the runtime-owned part of its provenance: `scope` defaults
        to "run" and `owner_id` resolves like the KG lane; `cause` defaults to "ingest"; `occurred_at` to now;
        `revision_id` to a hash of (node, scope, owner, cause, ops, caller sources, derived_from, artifact_ids,
        occurred_at) so a retried attempt or a re-run is a no-op (a re-observation) and a new source is a new
        revision. The runtime
        stamps the step ref of THIS effect (and an artifact ref per `artifact_ids`); the caller keeps
        `derived_from` / `actor` / `reader` / `reason` / `extra` and may name artifact / message / event /
        diary_entry sources, never a `run` or `step` ref. An ungrounded revision (no source, no derived_from, e.g.
        a handler called outside any run) is refused by the graph (`ProvenanceRequired`) as a failed outcome.
        """
        del default_next_node
        if memory_graph is None:
            return EffectOutcome.failed(str(graph_unavailable))
        payload = dict(effect.payload or {})
        if "recorded_at" in payload:
            # Record time is when THIS mind learned it: the store's clock, never a caller's claim (a backdated
            # recorded_at would rewrite the revision log's history). Observation time is `occurred_at`.
            return EffectOutcome.failed(
                "MEMORY_REVISE: recorded_at is the store's record time and cannot be set by a workflow; "
                "pass the observation time as occurred_at"
            )
        unknown = sorted(set(payload) - _REVISE_KEYS - _REVISE_TRANSPORT_KEYS)
        if unknown:
            return EffectOutcome.failed(
                f"MEMORY_REVISE: unknown payload keys {unknown} (known: {sorted(_REVISE_KEYS)})"
            )
        ops = payload.get("ops")
        if not isinstance(ops, list) or not ops or not all(isinstance(o, dict) for o in ops):
            return EffectOutcome.failed("MEMORY_REVISE requires payload.ops: a non-empty list of {op, payload}")

        raw_prov = payload.get("provenance")
        if raw_prov is None:
            raw_prov = {}
        if not isinstance(raw_prov, dict):
            return EffectOutcome.failed("MEMORY_REVISE payload.provenance is an object")
        prov_unknown = sorted(set(raw_prov) - _REVISE_PROVENANCE_KEYS)
        if prov_unknown:
            return EffectOutcome.failed(
                f"MEMORY_REVISE provenance keys {prov_unknown} are outside the graph shape "
                f"{sorted(_REVISE_PROVENANCE_KEYS)} (anything else belongs in provenance.extra)"
            )
        caller_sources = raw_prov.get("sources") or []
        if not isinstance(caller_sources, list) or not all(isinstance(r, dict) for r in caller_sources):
            return EffectOutcome.failed("MEMORY_REVISE provenance.sources is a list of SourceRef objects")
        # Validate (and normalize: kind is case- and space-insensitive) through AbstractMemory's own SourceRef FIRST,
        # then compare the normalized kind: "STEP" or " step " is a step ref too and is refused like one.
        from abstractmemory.graph import SourceRef  # type: ignore

        try:
            parsed_sources = [SourceRef.from_dict(r) for r in caller_sources]
        except ValueError as e:  # InvalidSourceRef
            return EffectOutcome.failed(f"MEMORY_REVISE provenance.sources holds an invalid SourceRef: {e}")
        caller_sources = [r.to_dict() for r in parsed_sources]
        owned = sorted({r.kind for r in parsed_sources} & _RUNTIME_OWNED_REF_KINDS)
        if owned:
            return EffectOutcome.failed(
                f"MEMORY_REVISE: {owned} refs are runtime-owned (the runtime stamps the executing step); "
                "name artifacts with payload.artifact_ids"
            )
        # A caller ref may only attach this revision to the run executing it: a foreign `run_id` would index the
        # memory under another run (`derived_from_source({"kind": "run", ...})` answers by run id).
        this_run = str(getattr(run, "run_id", "") or "")
        foreign = sorted({r.run_id for r in parsed_sources if r.run_id and r.run_id != this_run})
        if foreign:
            return EffectOutcome.failed(
                f"MEMORY_REVISE: provenance.sources name run(s) {foreign}; a caller ref's run_id must be this run "
                f"({this_run!r}) or absent"
            )
        artifact_ids = payload.get("artifact_ids") or []
        if isinstance(artifact_ids, str):
            artifact_ids = [artifact_ids]
        if not isinstance(artifact_ids, list) or not all(isinstance(a, str) and a.strip() for a in artifact_ids):
            return EffectOutcome.failed("MEMORY_REVISE payload.artifact_ids is a list of artifact id strings")
        named_artifacts = sorted({a.strip() for a in artifact_ids} | {
            str(r.artifact_id) for r in parsed_sources if r.kind == "artifact"})
        if named_artifacts:
            # An artifact grounds a revision only if it exists: never accept a name we cannot check.
            if artifact_store is None:
                return EffectOutcome.failed(
                    "MEMORY_REVISE names artifacts but this host gave the handler no artifact_store to check them "
                    "(build_memory_kg_effect_handlers(..., artifact_store=)); nothing was written"
                )
            missing = [a for a in named_artifacts if not artifact_store.exists(a)]
            if missing:
                return EffectOutcome.failed(f"MEMORY_REVISE names artifacts that do not exist: {missing}")

        scope = str(payload.get("scope") or "run").strip().lower() or "run"
        owner_raw = payload.get("owner_id")
        if isinstance(owner_raw, str) and owner_raw.strip():
            owner_id = owner_raw.strip()
            if owner_id not in derived_owner_ids(run, run_store=run_store):
                return EffectOutcome.failed(
                    f"MEMORY_REVISE: owner_id {owner_id!r} is not a partition this run may write (its run, its "
                    "session, or the global owner)"
                )
        else:
            try:
                owner_id = resolve_scope_owner_id(run, scope=scope, run_store=run_store)
            except Exception as e:
                return EffectOutcome.failed(
                    f"MEMORY_REVISE could not resolve owner_id for scope {scope!r} ({e}); pass payload.owner_id"
                )
        cause = str(payload.get("cause") or "ingest").strip().lower()
        run_id = str(getattr(run, "run_id", "") or "")
        revision_id = payload.get("revision_id")
        if not (isinstance(revision_id, str) and revision_id.strip()):
            # Everything the caller said, so a retried attempt of the SAME revision is a no-op while the same ops
            # with a new source (another artifact, another derivation, another observation time) are a new revision.
            # Neither the run nor the runtime's step ref is in the basis: a retry has a new STARTED row, and the same
            # workflow run again with the same ops re-observes ONE memory instead of duplicating it (AbstractMemory's
            # re-observation rule records the new run's step on it).
            basis = json.dumps(
                [str(getattr(run, "current_node", "") or ""), scope, owner_id, cause, ops,
                 caller_sources, raw_prov.get("derived_from") or [], sorted(a.strip() for a in artifact_ids),
                 payload.get("occurred_at")],
                sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
            )
            revision_id = "rev_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]

        try:
            from abstractmemory.graph import Revision  # type: ignore

            stamped = facade.stamp_current_step(
                {
                    "sources": list(caller_sources),
                    "derived_from": raw_prov.get("derived_from") or [],
                    "actor": raw_prov.get("actor") or "runtime",
                    "reader": raw_prov.get("reader"),
                    "reason": raw_prov.get("reason"),
                    "extra": raw_prov.get("extra") or {},
                },
                run_id=run_id,
                artifact_ids=artifact_ids,
            )
            rev: Dict[str, Any] = {
                "revision_id": revision_id.strip(),
                "scope": scope,
                "owner_id": owner_id,
                "cause": cause,
                "occurred_at": str(payload.get("occurred_at") or "").strip() or now_iso(),
                "ops": ops,
                "provenance": stamped,
                "summary": payload.get("summary"),
            }
            result = facade.apply_revision(Revision.from_dict(rev))
        except ValueError as e:  # ProvenanceRequired, InvalidSourceRef and malformed ops are all ValueErrors
            return EffectOutcome.failed(f"MEMORY_REVISE refused ({type(e).__name__}): {e}")

        out: Dict[str, Any] = {"ok": True, **result, "scope": scope, "owner_id": owner_id,
                               "sources": stamped["sources"]}
        if not result.get("applied"):
            # The revision was already in the log (a retried attempt after a crash, or a re-run): nothing this attempt
            # stamped was written. Report what the MEMORY holds (its ids from the log, their sources), never this
            # attempt's step ref as if it were the memory's source.
            memory_ids = list(dict.fromkeys([*result.get("assertion_ids", []), *result.get("closed", []),
                                             *result.get("narrowed", [])]))
            held: List[Dict[str, Any]] = []
            for mid in memory_ids:
                for src in facade.sources_of(mid, transitive=False):
                    if src not in held:
                        held.append(src)
            out["sources"] = held
            out["replayed"] = True
            out["attempt_sources"] = stamped["sources"]
        return EffectOutcome.completed(out)

    facade = MemoryFacade(graph=memory_graph, run_store=run_store, runtime_id=runtime_id)

    return {
        EffectType.MEMORY_KG_ASSERT: _handle_assert,
        EffectType.MEMORY_KG_QUERY: _handle_query,
        EffectType.MEMORY_KG_RESOLVE: _handle_resolve,
        EffectType.MEMORY_REVISE: _handle_revise,
    }
