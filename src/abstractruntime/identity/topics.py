"""The entity's TOPICS: its memory grouped by subject (memory communities) - the runtime half.

The operator's rulings (2026-11-10): topics "must run server side (runtime) with eventual consistency, and in
particular during sleep/maintenance phase... so that it becomes part of the knowledge of the entity. it should also
help the entity to have its knowledge organized by topic, to know what it knows, and let it delve deeper into
subparts of its mind/memory"; "AbstractMemory should be the one owning those graph operations, but it could be called
indeed by runtime".

So AbstractMemory owns the partition, the `community` records (each with an authored title and summary, versioned,
with its layout) and the verbs. This module holds no graph code; it is the runtime's use of those verbs:

- THE NIGHT'S STAGE, `sleep_communities` (run by `life.build_consolidator` after the sleep's metadata repair and
  before its closing voice): memory embeds the records that have no vector yet (`embed_missing`), then
  `communities_pass` forms the night's version in one guarded transaction. Titles and summaries are authored by the
  summaries capability (the night's `author`, the one that titles every record the night forms), bounded by what the
  night's authoring budget left (`budget`, in model calls). It is FENCED: `cancelled=fence.moved` on both verbs, the
  author checked before every call, and memory's commit guard refuses any write after the operator's Stop. A stopped
  night records nothing for its topics.

The memory verbs are imported by name where they are used. A memory without them is a loud `#MISSING_VERB` line and
a receipt that says so - never a second implementation here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

__all__ = [
    "STAGE_VERBS",
    "sleep_communities",
]

STAGE_VERBS = "abstractmemory.embed_missing / communities_pass"
_STOPPED = "the operator's Stop"


# ------------------------------------------------------------------------------------------------ the night's stage


def sleep_communities(ms: Any, *, entity_id: str, scopes: Sequence[Tuple[str, str]],
                      author: Callable[[str, str, Any], Any], fence: Any, budget: Optional[int],
                      out: Callable[[str], None], maintenance_kind: Optional[str] = None) -> Dict[str, Any]:
    """THE NIGHT'S TOPICS STAGE. Never raises; returns the receipt the night's dict carries under `communities`.

    1. `embed_missing(ms, scopes=, owner_id=, record_ids=None, cancelled=fence.moved)`: memory embeds every open
       record of these scopes that has no vector yet, FIRST, so the pass's nearest-neighbour edges see them. Its
       counts ride the receipt (`embedding: {embedded, skipped, failed}`); what was not embedded is a night note
       (by memory's reason codes), never a failure.
    2. `communities_pass(ms, scopes=, owner_id=, author=, cancelled=fence.moved, budget=)`: memory's version of the
       night, one guarded transaction. The author is fenced here too (`fence.check` before every call: no model call
       starts after the operator's Stop, whatever client the host built). Memory's receipt is returned as memory
       gave it (`version_seq`, `communities`, `new`, `kept`, `renamed`, `dissolved`, `authored`, `pending`,
       `failed`, `authoring.failures`, `unvectored_count`, `unplaced_count`, `cancelled`, `skipped`, ...), plus
       `embedding` and `budget`.
    3. A version written: `communities_version` (its `version_seq`) joins tonight's completed-maintenance detail
       (`maintenance_kind`: the memq_maintenance row tonight's `sleep_pass` completed, "night" or "cycle"; None =
       it completed none). Then one own_time.log line - or, when memory wrote no version (`skipped`: unchanged,
       empty, nothing vectored, no numpy), one line naming why.

    The operator's Stop at any point - the fence moved, memory's check before its write said `cancelled`, or its
    commit guard refused the transaction: the receipt carries `stopped`, and nothing more is written or logged (no
    line, no maintenance note; the night's one Stop line is the consolidator's). The caller writes the
    authoring-failure lines from `authoring.failures` (`log_pass_authoring_failures`)."""
    from .chat import TurnCancelled

    try:
        from abstractmemory.communities import communities_pass, embed_missing
    except ImportError as e:
        out(f"#MISSING_VERB {STAGE_VERBS} ({e}): no topics formed this sleep - the stage runs once memory ships them")
        return {"missing_verb": STAGE_VERBS}
    if fence.moved():
        return {"stopped": _STOPPED}

    embedding: Dict[str, Any]
    note = ""  # what was not embedded: said with the receipt's line (a stopped night says nothing)
    try:
        embedded = embed_missing(ms, scopes=list(scopes), owner_id=entity_id, record_ids=None, cancelled=fence.moved)
    except TurnCancelled as stopped:  # memory's commit guard refused a vector: the operator's Stop
        return {"stopped": f"{_STOPPED} ({stopped.where})"}
    except Exception as e:  # noqa: BLE001 - the pass still runs: vector-less records are placed by their neighbours
        embedding = {"error": f"{type(e).__name__}: {e}"}
        out(f"#FALLBACK the topics step could not embed the memories without a vector ({embedding['error']}); "
            "the pass places them by their neighbours")
    else:
        embedding = {"embedded": len(embedded["embedded"]), "skipped": len(embedded["skipped"]),
                     "failed": len(embedded["failed"])}
        note = _embedding_note(embedded)
    if fence.moved():
        return {"stopped": _STOPPED, "embedding": embedding}

    def fenced_author(kind: str, body: str, context: Any) -> Any:
        fence.check("the topics step's authoring")  # memory records a refusal as the author raising; nothing lands
        return author(kind, body, context)

    try:
        receipt = dict(communities_pass(ms, scopes=list(scopes), owner_id=entity_id, author=fenced_author,
                                        cancelled=fence.moved, budget=budget))
    except TurnCancelled as stopped:  # memory's commit guard refused the version: rolled back, nothing landed
        return {"stopped": f"{_STOPPED} ({stopped.where})", "version_seq": None, "embedding": embedding}
    except Exception as e:  # noqa: BLE001 - the night's other work stands
        error = f"{type(e).__name__}: {e}"
        if note:
            out(note)
        out(f"#FALLBACK the topics step did not run ({error})")
        return {"error": error, "embedding": embedding}
    receipt["embedding"] = embedding
    receipt["budget"] = budget
    try:
        return _after_the_pass(ms, scopes, receipt, maintenance_kind=maintenance_kind, fence=fence, note=note,
                               out=out)
    except TurnCancelled as stopped:  # the Stop landed after the version: the version stands, the note does not
        receipt["stopped"] = f"{_STOPPED} ({stopped.where})"
        return receipt
    except Exception as e:  # noqa: BLE001 - a receipt this runtime cannot read is said, never a lost night
        receipt["error"] = f"the receipt could not be read ({type(e).__name__}: {e})"
        out(f"#FALLBACK the topics step ran but its receipt could not be read ({type(e).__name__}: {e}); memory's "
            f"receipt keys: {sorted(str(k) for k in receipt)}")
        return receipt


def _embedding_note(embedded: Dict[str, Any]) -> str:
    """One line for what embed_missing did not embed, by memory's reason codes (`no_embedder`,
    `embedder_unavailable: <error>`, `dimension_mismatch`, ...: the code is what precedes the first colon)."""
    reasons: Dict[str, int] = {}
    why = embedded.get("reasons") or {}
    for rid in embedded["skipped"]:
        code = str(why.get(rid) or "skipped").split(":", 1)[0]
        reasons[code] = reasons.get(code, 0) + 1
    for f in embedded["failed"]:
        code = str(f.get("reason") or "failed").split(":", 1)[0]
        reasons[code] = reasons.get(code, 0) + 1
    if not reasons:
        return ""
    n = sum(reasons.values())
    by = ", ".join(f"{count} {code}" for code, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
    return f"(topics: {n} memories were not embedded this sleep - {by}; the pass places them by their neighbours)"


def _after_the_pass(ms: Any, scopes: Sequence[Tuple[str, str]], receipt: Dict[str, Any], *,
                    maintenance_kind: Optional[str], fence: Any, note: str, out: Callable[[str], None]) -> Dict[str, Any]:
    """What a returned pass leaves: nothing after a Stop; else the maintenance note, then the night's lines."""
    if receipt["cancelled"] is not None or fence.moved():
        receipt["stopped"] = _STOPPED
        return receipt
    version_seq = receipt["version_seq"]
    line = _receipt_line(receipt)  # read whole BEFORE anything is written: a receipt it cannot read writes nothing
    if version_seq is not None and maintenance_kind:
        _note_version(ms, scopes, maintenance_kind, version_seq)
    if note:
        out(note)
    out(line)
    return receipt


def _note_version(ms: Any, scopes: Sequence[Tuple[str, str]], kind: str, version_seq: int) -> None:
    """`communities_version` into tonight's completed-maintenance row (memq_maintenance.detail, memory's queue verbs):
    the row `sleep_pass` just wrote is re-recorded with the same completion time and journal seq, its detail
    extended. Fenced like every memory write (the home's commit guard)."""
    row = ms.tasks.last_maintenance(list(scopes), kind=kind)
    if row is None:
        return
    ms.tasks.record_maintenance(list(scopes), kind=kind, completed_at=row["completed_at"], seq=int(row["seq"]),
                                detail={**dict(row["detail"] or {}), "communities_version": int(version_seq)})


def _receipt_line(r: Dict[str, Any]) -> str:
    e = r["embedding"]
    first = f"{e['embedded']} embedded first" if "embedded" in e else "embedding failed"
    vectors = (f"{first}, {int(r['unvectored_count'])} without a vector, {int(r['unplaced_count'])} unplaced")
    if r["version_seq"] is None:
        return f"(topics: no new version - {r['skipped']}; {int(r['communities'])} topics; {vectors})"
    return (f"(topics: version {int(r['version_seq'])} - {int(r['communities'])} topics: {int(r['new'])} new, "
            f"{int(r['kept'])} kept, {int(r['dissolved'])} dissolved, {int(r['renamed'])} retitled, "
            f"{int(r['moved_out'])} memories moved; titles {int(r['authored'])} authored, {int(r['titles_kept'])} "
            f"kept, {int(r['pending'])} pending, {int(r['failed'])} failed; {vectors})")
