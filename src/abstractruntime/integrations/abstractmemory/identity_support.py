"""AbstractMemory-touching support for the identity kernel (diary + prelude).

Placement rationale (install-boundary contract): `abstractruntime.identity`
is a KERNEL package and must not import optional capability stacks — the
diary chain and the prelude layout are pure kernel, but the graph projection
(MemoryRecordInput) and the identity-row read (TripleQuery) belong to the
abstractmemory integration. The kernel modules import THIS module lazily at
call time, so an abstractmemory-free install can still use the chain-only
diary and gets an actionable error only when a graph-touching feature is
actually exercised.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

__all__ = ["folded_self_records", "identity_rows", "project_diary_entry", "spark_hash"]


def spark_hash(spark: Any) -> str:
    """One hash definition, memory-owned (a2a 0003 ask 2 — shipped)."""
    from abstractmemory import canonical_spark_hash  # type: ignore

    return canonical_spark_hash(spark)


def folded_self_records(
    memory_system: Any, *, scope: str, entity_id: str, spark_version: Optional[int] = None
) -> List[Any]:
    """The folded identity core (a2a 0003 ask 3 — shipped memory-side):
    prompt-active AND closure-folded, identity kinds only, ordered kind rank
    -> precedence -> record id. A retracted value never renders."""
    return list(
        memory_system.self_records(scope=scope, owner_id=entity_id, spark_version=spark_version)
    )


def _import_record_input():
    try:
        from abstractmemory.records import MemoryRecordInput  # type: ignore

        return MemoryRecordInput
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "AbstractMemory is not available for the diary projection. Install it "
            "(e.g. `pip install -e abstractmemory`) or construct the diary handlers "
            "with memory_system=None for chain-only operation."
        ) from e


def identity_rows(memory_system: Any, *, scope: str, entity_id: str) -> List[Any]:
    """All digest assertions in the entity's identity scope (layer-1 read)."""
    from abstractmemory import TripleQuery  # type: ignore

    return list(
        memory_system.query(
            TripleQuery(scope=scope, owner_id=entity_id, predicate="dcterms:abstract", limit=0)
        )
    )


def project_diary_entry(
    memory_system: Any,
    *,
    entity_id: str,
    entry_id: str,
    kind: str,
    visibility: str,
    gist: Optional[str],
    written_at: str,
    turn_id: str,
    origin: Dict[str, Any],
    anchor_graph_ids: Optional[List[str]] = None,
    resolves: Optional[str] = None,
) -> Tuple[Optional[str], List[str]]:
    """Project the MEMORY OF THE ACT of a diary write into the graph.

    Semantics (a2a 0003, maintainer round 3): the graph involuntarily records
    that the act happened — "I wrote about X in my diary" — never a second
    copy of the prose. `attributes.entry_id` is the book address; progressive
    disclosure fetches the verbatim through DIARY_READ.

    Connectivity (maintainer, 2026-07-07: "the diary is connected to none
    other memory and that is not ok"): the projection now writes
    `written_amid` edges to the GRAPH ids the entity was attending to at
    write time — UNIFORMLY, private entries included. The policy analysis:
    the edge reveals only the act-frame ("written while attending to X"),
    which is exactly what the act-only projection already discloses; the
    words stay in the book. `written_amid` must stay OUT of the dream pass's
    component-defining relations (memory's allowlist guard covers this — it
    lists authored relations explicitly and written_amid is not one).

    Returns (record_id, warnings); failure degrades, never blocks the chain.
    """
    warnings: List[str] = []
    if memory_system is None:
        return None, warnings
    if not callable(getattr(memory_system, "remember_many", None)):
        warnings.append("#FALLBACK diary projection skipped: memory_system lacks remember_many")
        return None, warnings

    private = visibility == "private"
    # Memory's closed diary_type vocabulary; the open kind stays in diary_kind.
    # "question" and "problem" are first-class (maintainer rounds 4+6: open
    # questions, problems, and ideas are autonomy drivers — curiosity,
    # wrongness, direction — the heartbeat's need-check queries all three).
    diary_type = kind if kind in ("note", "idea", "commitment", "reflection", "question", "problem") else "note"
    if diary_type != kind and not private:
        warnings.append(
            f"#FALLBACK diary kind {kind!r} is outside memory's diary_type vocabulary; "
            f"projected as diary_type='note' (verbatim kind kept in diary_kind)"
        )

    written_date = str(written_at or "")[:10]
    if private:
        title = f"Diary entry (private) — {written_date}"
        digest = "Wrote a private diary entry."
        attributes: Dict[str, Any] = {"entry_id": entry_id, "written_at": written_at, "private": True}
        provenance: Dict[str, Any] = {"source": "diary-projection", "entry_id": entry_id}
    else:
        title = f"Diary entry ({kind}) — {written_date}"
        digest = (gist or "").strip() or f"Wrote a diary entry ({kind}) at {written_at}; no gist elected."
        attributes = {
            "entry_id": entry_id,
            "diary_type": diary_type,
            "diary_kind": kind,
            "written_at": written_at,
        }
        if resolves:
            # Entity-elected resolution (a2a 0009): the card's resolved-
            # questions read joins question entries on this attribute.
            attributes["resolves"] = resolves
        provenance = {"source": "diary-projection", "entry_id": entry_id}
        provenance.update({k: v for k, v in origin.items() if v is not None})

    edges = tuple(
        ("written_amid", gid)
        for gid in (anchor_graph_ids or [])
        if isinstance(gid, str) and gid.strip()
    )

    try:
        MemoryRecordInput = _import_record_input()
        record_ids = memory_system.remember_many(
            [
                MemoryRecordInput(
                    kind="diary",
                    title=title,
                    digest=digest,
                    keywords=(),
                    payload_ref=None,
                    attributes=attributes,
                    provenance=provenance,
                    edges=edges,
                )
            ],
            scope="diary",
            owner_id=entity_id,
            idempotency_key=f"diary:{entry_id}",
            turn_id=turn_id,
        )
        return (str(record_ids[0]) if record_ids else None), warnings
    except Exception as e:
        warnings.append(f"#FALLBACK diary projection failed (chain entry preserved): {e}")
        return None, warnings
