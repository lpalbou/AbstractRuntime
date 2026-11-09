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



def _diary_types() -> frozenset:
    """Memory's closed diary_type set, imported live so every widening
    reaches this clamp for free (the two-copies drift is the recorded
    gotcha class). Fallback snapshot only for version skew — an installed
    abstractmemory too old to export DIARY_TYPES also refuses unknown
    types at write, so the snapshot deliberately EXCLUDES lesson (clamping
    to note there is correct: the store would refuse the projection)."""
    try:
        from abstractmemory import DIARY_TYPES  # type: ignore

        return frozenset(DIARY_TYPES)
    except Exception:  # noqa: BLE001 - version skew degrades, never breaks
        return frozenset(
            {"note", "idea", "commitment", "reflection", "question", "problem"}
        )


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
    text: Optional[str] = None,
    written_at: str,
    turn_id: str,
    origin: Dict[str, Any],
    anchor_graph_ids: Optional[List[str]] = None,
    resolves: Optional[str] = None,
    explores: Optional[str] = None,
    digest_method: Optional[str] = None,
    entry_record_hash: Optional[str] = None,
    home_dir: Optional[Any] = None,
    authored: Optional[Dict[str, Any]] = None,
    authoring_failure: Optional[Dict[str, Any]] = None,
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
    pending_fields: List[str] = []  # a public entry's fields its authoring could not write (see below)
    # Memory's closed diary_type vocabulary; the open kind stays in diary_kind.
    # "question" and "problem" are first-class (maintainer rounds 4+6: open
    # questions, problems, and ideas are autonomy drivers — curiosity,
    # wrongness, direction — the heartbeat's need-check queries all three).
    # IMPORTED, not copied (semantics ruling decision:diary-type-lesson-widening,
    # 2026-07-19): memory exports DIARY_TYPES for exactly this — the hardcoded
    # tuple here was the drift-class root (Ephemeral's elected kind=lesson
    # clamped to note while 20 machine-formed lessons stood beside it). The
    # local tuple survives only as the version-skew fallback.
    diary_type = kind if kind in _diary_types() else "note"
    if diary_type != kind and not private:
        warnings.append(
            f"#FALLBACK diary kind {kind!r} is outside memory's diary_type vocabulary; "
            f"projected as diary_type='note' (verbatim kind kept in diary_kind)"
        )

    written_date = str(written_at or "")[:10]
    if private:
        # UNIFORM-TITLE FIX, private half (flow's long-life adversary, commons
        # c5208 ask 1): same-day private projections shared one exact title,
        # so title-identity consumers (duplicate-group maintenance, dream
        # composition) read them as copies — flags self-amplified. Words are
        # FORBIDDEN in every graph field for private entries, so the
        # disambiguator is the entry id's opaque tail — already carried in
        # attributes.entry_id, zero new disclosure.
        _tail = str(entry_id or "")[-6:]
        title = f"Diary entry (private) — {written_date}" + (f" [{_tail}]" if _tail else "")
        digest = "Wrote a private diary entry."
        attributes: Dict[str, Any] = {"entry_id": entry_id, "written_at": written_at, "private": True}
        if resolves:
            # A resolution is a KEY, never words (same argument as entry_id,
            # which the private branch already carries) — dropping it made
            # book-derived and graph-derived resolution counts disagree for
            # private resolves (adversary F4, 2026-07-18): the entity-card
            # fold joins on this attribute and kept showing the question
            # open while the day-open ratio (book-read) counted it resolved.
            attributes["resolves"] = resolves
        if explores:
            # Same key-not-words argument as resolves/entry_id: the drive
            # ratio's fold (cognition_health) joins on this attribute.
            attributes["explores"] = explores
        provenance: Dict[str, Any] = {"source": "diary-projection", "entry_id": entry_id}
    else:
        # AUTHORED AT FORMATION (operator 2026-11-09; Codex review 2 #10): no template, slug or cut. A diary record
        # is his own act, so its SUMMARY is his words, never authored (memory refuses a pending summary on it): his
        # elected gist, else his first line - a line is structure, kept whole (root ADR 0054). Its TITLE is his gist
        # when he elected one, else authored by the summaries capability BEFORE the record forms (`authored`, passed
        # by the lane that wrote the entry: ChatSession's DIARY_WRITE, the visit's close step). Only a recorded
        # authoring failure forms the title pending - with its failure line (written below, once the record has its
        # id); a lane that passed no authored title is that failure, said as such.
        # A writer-declared MECHANICAL entry (digest_method: a deterministic close note) carries machine words, never
        # his elected title: its title is authored like a gistless entry's.
        his_title = None if digest_method else ((gist or "").strip() or None)
        got = authored if isinstance(authored, dict) else {}
        authored_title = str(got.get("title") or "").strip() or None
        title = his_title or authored_title
        pending_fields = ["title"] if title is None else []
        digest = (gist or "").strip() or (str(text or "").strip().splitlines() or [""])[0].strip()
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
        if explores:
            attributes["explores"] = explores
        authored_fields = ["title"] if (authored_title and not his_title) else []
        if authored_fields:
            attributes["metadata_authored"] = {"fields": authored_fields, "capability": "summaries",
                                               **{k: got[k] for k in ("reader", "source") if got.get(k)}}
        provenance = {"source": "diary-projection", "entry_id": entry_id}
        provenance.update({k: v for k, v in origin.items() if v is not None})

    # Writer-declared authorship metadata (c5270 P2-2, jointly ruled with
    # memory): projections of machine-worded entries carry the writer's
    # digest_method so memory's machine-authorship bridge guard reads
    # projections and formed records through ONE key. Both branches — the
    # attribute is authorship metadata, never content.
    if digest_method:
        attributes["digest_method"] = digest_method

    edges = tuple(
        ("written_amid", gid)
        for gid in (anchor_graph_ids or [])
        if isinstance(gid, str) and gid.strip()
    )

    try:
        # Ledger provenance (ADR 0011): the diary entry (with its chain) is the ground; a public projection also
        # names the DIARY_WRITE step of the writing run when it runs inside one. A private projection names the
        # entry only: it carries no origin by design, and the chain entry itself holds it.
        from ...identity.diary import diary_chain_id
        from .memory_facade import MemoryFacade

        entry_ref = {"kind": "diary_entry", "entry_id": entry_id, "run_id": diary_chain_id(entity_id)}
        if entry_record_hash:  # the chain's own hash of the entry: what the projection vouches the entry said
            entry_ref["digest"] = entry_record_hash
        provenance = {**provenance, "sources": [entry_ref]}
        writer_run = None if private else origin.get("run_id")
        provenance = MemoryFacade().stamp_current_step(provenance, run_id=writer_run)
        MemoryRecordInput = _import_record_input()
        record_ids = memory_system.remember_many(
            [
                MemoryRecordInput(
                    kind="diary",
                    title=title,
                    **({"pending": tuple(pending_fields)} if (not private and pending_fields) else {}),
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
        record_id = str(record_ids[0]) if record_ids else None
    except Exception as e:
        warnings.append(f"#FALLBACK diary projection failed (chain entry preserved): {e}")
        return None, warnings
    if not private and pending_fields:
        from ...identity.title_failures import record_authoring_failure

        why = dict(authoring_failure or {}) if isinstance(authoring_failure, dict) else {}
        if not why:
            why = {"reason": "unavailable", "error_class": None,
                   "error_message": "the lane that wrote this entry passed no authored title or summary"}
        logged = record_authoring_failure(
            home_dir, entity=entity_id, record_id=record_id, kind="diary", raw_block=str(text or ""),
            template_id="diary.projection", provider=why.get("provider"), model=why.get("model"),
            reason=str(why.get("reason") or "unavailable"),
            missing="title",
            extra={"entry_id": entry_id, "capability": "summaries",
                   **{k: why[k] for k in ("error_class", "error_message", "error") if why.get(k)}},
        )
        if logged.get("logged"):
            warnings.append(str(logged["logged"]))
    return record_id, warnings
