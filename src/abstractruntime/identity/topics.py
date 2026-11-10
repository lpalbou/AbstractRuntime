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
  night's authoring budget left (`budget`, in model calls). It is FENCED: `cancelled=fence.moved` on both verbs, and
  memory's commit guard refuses any write after the operator's Stop. A stopped night records nothing for its topics.

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

    1. `embed_missing(ms, scopes, owner_id, record_ids=None, cancelled=fence.moved)`: memory embeds every record of
       these scopes that has no vector yet, FIRST, so the pass's nearest-neighbour edges see them. Its counts ride the
       receipt (`embedding`); what could not be embedded is a night note, never a failure.
    2. `communities_pass(ms, scopes, owner_id, author, cancelled=fence.moved, budget)`: memory's version of the night
       (one guarded transaction). Its receipt is returned as memory gave it, plus `embedding` and `budget`.
    3. When a version stands: `communities_version` joins tonight's maintenance detail (`maintenance_kind`: the
       memq_maintenance row tonight's `sleep_pass` completed, "night" or "cycle"; None = none was), and one
       own_time.log line says what formed.

    The operator's Stop at any point (the fence moved, or memory's commit guard refused): the receipt says
    `cancelled`, and nothing more is written or logged - no line, no maintenance note (the night's one Stop line is
    the consolidator's). `cancelled_before_write` is memory's own word that its version did not land."""
    from .chat import TurnCancelled

    try:
        from abstractmemory import communities_pass, embed_missing
    except ImportError as e:
        out(f"#MISSING_VERB {STAGE_VERBS} ({e}): no topics formed this sleep - the stage runs once memory ships them")
        return {"missing_verb": STAGE_VERBS}
    if fence.moved():
        return {"cancelled": _STOPPED}

    embedding: Dict[str, Any]
    note = ""  # what could not be embedded: said with the receipt's line (a stopped night says nothing)
    try:
        embedded = embed_missing(ms, scopes=list(scopes), owner_id=entity_id, record_ids=None, cancelled=fence.moved)
    except TurnCancelled as stopped:
        return {"cancelled": f"{_STOPPED} ({stopped.where})"}
    except Exception as e:  # noqa: BLE001 - the pass still runs: vector-less records are placed by their neighbours
        embedding = {"error": f"{type(e).__name__}: {e}"}
        out(f"#FALLBACK the topics step could not embed the memories without a vector ({embedding['error']}); "
            "the pass places them by their neighbours")
    else:
        embedding = {"embedded": len(embedded["embedded"]), "skipped": len(embedded["skipped"]),
                     "failed": len(embedded["failed"])}
        if embedded["failed"]:
            reasons = sorted({str(f.get("reason") or "unknown") for f in embedded["failed"]})
            note = (f"(topics: {len(embedded['failed'])} memories could not be embedded this sleep - "
                    f"{'; '.join(reasons[:3])}; the pass places them by their neighbours)")
    if fence.moved():
        return {"cancelled": _STOPPED, "embedding": embedding}

    try:
        receipt = dict(communities_pass(ms, scopes=list(scopes), owner_id=entity_id, author=author,
                                        cancelled=fence.moved, budget=budget))
    except TurnCancelled as stopped:  # memory's commit guard refused the version: rolled back, nothing landed
        return {"cancelled": f"{_STOPPED} ({stopped.where})", "cancelled_before_write": True, "embedding": embedding}
    except Exception as e:  # noqa: BLE001 - the night's other work stands
        error = f"{type(e).__name__}: {e}"
        if note:
            out(note)
        out(f"#FALLBACK the topics step did not run ({error})")
        return {"error": error, "embedding": embedding}
    receipt["embedding"] = embedding
    receipt["budget"] = budget
    if receipt["cancelled_before_write"] or fence.moved():
        receipt["cancelled"] = _STOPPED
        return receipt

    version = receipt["version"]
    if version is not None and maintenance_kind:
        try:
            _note_version(ms, scopes, maintenance_kind, version)
        except TurnCancelled as stopped:  # the Stop landed after the version: the version stands, the note does not
            receipt["cancelled"] = f"{_STOPPED} ({stopped.where})"
            return receipt
    if note:
        out(note)
    out(_receipt_line(receipt))
    authoring = receipt["authoring"]
    if int(authoring["failed"]) and not (authoring.get("failures") or []):
        # The failure lines are written from memory's `authoring.failures` (formation_authoring's receipt): a
        # count with no records would leave failures without their lines - said, never silent.
        out(f"#FALLBACK topics: memory counted {int(authoring['failed'])} failed title(s) but returned no failure "
            "records - their authoring-failure lines are missing")
    return receipt


def _note_version(ms: Any, scopes: Sequence[Tuple[str, str]], kind: str, version: Any) -> None:
    """`communities_version` into tonight's completed-maintenance row (memq_maintenance.detail, memory's queue verbs):
    the row `sleep_pass` just wrote is re-recorded with the same completion time and journal seq, its detail
    extended. Fenced like every memory write (the home's commit guard)."""
    row = ms.tasks.last_maintenance(list(scopes), kind=kind)
    if row is None:
        return
    ms.tasks.record_maintenance(list(scopes), kind=kind, completed_at=row["completed_at"], seq=int(row["seq"]),
                                detail={**dict(row["detail"] or {}), "communities_version": version})


def _receipt_line(r: Dict[str, Any]) -> str:
    a = r["authoring"]
    e = r["embedding"]
    first = f"{e['embedded']} embedded first" if "embedded" in e else "embedding failed"
    version = f"version {r['version']}" if r["version"] is not None else "no new version"
    return (f"(topics: {version} - {r['communities']} topics, {r['members_placed']} memories placed "
            f"({r['attached']} attached, {r['merged']} merged); titles {a['authored']} authored, {a['pending']} "
            f"pending, {a['failed']} failed; {first}, {len(r['unvectored'])} still without a vector)")
