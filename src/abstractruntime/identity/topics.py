"""The entity's TOPICS: its memory grouped by subject (memory communities) - the runtime half.

The operator's rulings (2026-11-10): topics "must run server side (runtime) with eventual consistency, and in
particular during sleep/maintenance phase... so that it becomes part of the knowledge of the entity. it should also
help the entity to have its knowledge organized by topic, to know what it knows, and let it delve deeper into
subparts of its mind/memory"; "AbstractMemory should be the one owning those graph operations, but it could be called
indeed by runtime".

So AbstractMemory owns the partition, the `community` records (each with an authored title and summary, versioned,
with its layout) and the verbs. This module holds no graph code; it is the runtime's two uses of those verbs:

- THE NIGHT'S STAGE, `sleep_communities` (run by `life.build_consolidator` after the sleep's metadata repair and
  before its closing voice): memory embeds the records that have no vector yet (`embed_missing`), then
  `communities_pass` forms the night's version in one guarded transaction. Titles and summaries are authored by the
  summaries capability (the night's `author`, the one that titles every record the night forms), bounded by what the
  night's authoring budget left (`budget`, in model calls). It is FENCED: `cancelled=fence.moved` on both verbs, and
  memory's commit guard refuses any write after the operator's Stop. A stopped night records nothing for its topics.
- THE ENTITY'S TOOL, `topics_text` (the tier-1 `topics` tool, through `HomeMemoryReader.topics`): the list, one
  topic's memories, the topic closest to some words. It reads what memory serves (`community_index.communities`),
  never a cache of its own: the list is that one read; a drill-down adds one read of his digests (the members' titles);
  meaning ranks through the store's vector query (memory's VectorMatrix).
- `entity_communities(ms, scopes=, owner_id=)`: THE read the tool makes, and the one a host's route can make.

The memory verbs are imported by name where they are used. A memory without them is a loud `#MISSING_VERB` line and
a receipt that says so at night, and a failed tool call (its `#FALLBACK` notice) by day - never a second
implementation here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PAYLOAD_VERSION",
    "STAGE_VERBS",
    "entity_communities",
    "sleep_communities",
    "topics_text",
]

STAGE_VERBS = "abstractmemory.embed_missing / communities_pass"
# The communities payload this runtime reads (memory's `communities()`, COMMUNITIES-DESIGN §3: "version": 2).
PAYLOAD_VERSION = 2
_STOPPED = "the operator's Stop"

# The tool's bounds: the list is a page, never the whole graph.
TOPICS_SHOWN = 40
MEMBERS_SHOWN = 20
RANKED_SHOWN = 12
# Meaning ranks this many of a life's topic members for `topics <words>` (the store's vector query, one embedding).
RANK_POOL = 60


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


# ------------------------------------------------------------------------------------------------------ the read


def entity_communities(ms: Any, *, scopes: Sequence[Tuple[str, str]], owner_id: str) -> Dict[str, Any]:
    """Memory's communities payload for one entity (`abstractmemory.community_index.communities`: the current
    version, with the members placed since it folded in, and each topic's `facts`), checked for the version this
    runtime reads. Raises ImportError when memory has no such verb, ValueError on another payload version."""
    from abstractmemory.community_index import communities

    payload = communities(ms, scopes=list(scopes), owner_id=owner_id)
    if payload.get("version") != PAYLOAD_VERSION:
        raise ValueError(f"memory served communities payload version {payload.get('version')!r}; this runtime reads "
                         f"version {PAYLOAD_VERSION}")
    return payload


# ------------------------------------------------------------------------------------------------------ the tool


def topics_text(reader: Any, body: str) -> str:
    """The `topics` tool over one home (`reader`: a HomeMemoryReader - its ladder, its #tag map, its digest reads).

    - empty body: the list - title, one-line summary, size and newest memory per topic, by size;
    - `#tag` (a topic's, as the list shows it): that topic's memories by title, newest first; a MEMORY's #tag: the
      topic it belongs to, the same way; `#tag some words`: that topic's memories ranked by meaning against the words;
    - words: the topic closest to them by meaning, its closest memories first.

    Every memory shown carries its #tag, registered for read_memory. Pure read: nothing is deposited."""
    from .tools import sanitize_tool_surface

    ladder = [(str(s), str(o)) for s, o in reader.ladder]
    payload = entity_communities(reader.home.ms, scopes=ladder, owner_id=str(reader.home.entity_id))
    topics = list(payload["communities"])
    if not topics:
        if payload["first_sleep_pending"]:
            return ("You have no topics yet: your memories are grouped by subject while you sleep, and the first "
                    "grouping happens during your next sleep.")
        return "Your memory holds no topics: there is not yet enough in it to group by subject."
    view = _TopicsView(reader, topics)
    words = " ".join((body or "").split())
    if not words:
        return view.listing()
    head, _, rest = words.partition(" ")
    if head.startswith("#"):
        tag = head.lstrip("#").strip().lower()
        topic = view.by_tag.get(tag)
        note = ""
        if topic is None:
            gid = reader.tag_map.get(tag) or reader.find_tag_in_home(tag)[0]
            topic = view.topic_of.get(gid) if gid else None
            if topic is None:
                return (f"No topic or memory with tag #{sanitize_tool_surface(tag, 20)} in your topics.\n\n"
                        + view.listing())
            note = f"#{tag} is one of the memories of this topic."
        return view.topic(topic, rest.strip(), note=note)
    return view.closest(words)


class _TopicsView:
    """One call's view: the served topics and their tags; the digest of every memory they hold is read (one read of
    his ladder) only when an answer shows members."""

    def __init__(self, reader: Any, topics: List[Dict[str, Any]]) -> None:
        from .memory_reader import memory_tag

        self.reader = reader
        # A topic's #tag is its stable KEY's tail (memory keeps a community's key across versions; its record id is
        # the version's), so a tag he noted stays his topic's after a sleep.
        self.topics = sorted(topics, key=lambda c: (-int(c["size"]), str(c["key"])))
        self.tag = {str(c["key"]): memory_tag(str(c["key"])) for c in self.topics}
        self.by_tag = {self.tag[str(c["key"])]: c for c in self.topics}
        self.topic_of: Dict[str, Dict[str, Any]] = {}
        self.provisional: set = set()
        for c in self.topics:
            for m in c["members"]:
                self.topic_of[str(m["id"])] = c
            for pid in c["provisional_ids"]:
                self.topic_of[str(pid)] = c
                self.provisional.add(str(pid))
        self._digests: Optional[Dict[str, Any]] = None

    @property
    def digests(self) -> Dict[str, Any]:
        if self._digests is None:
            self._digests = {str(a.subject): a for a in self.reader.digest_assertions_all()}
        return self._digests

    # -------------------------------------------------------------------------------------------- pieces
    def members(self, c: Dict[str, Any]) -> List[str]:
        ids = [str(m["id"]) for m in c["members"]] + [str(p) for p in c["provisional_ids"]]
        return [i for i in ids if i in self.digests]

    def newest(self, ids: Sequence[str]) -> List[str]:
        return sorted(ids, key=lambda i: str(self.digests[i].observed_at or ""), reverse=True)

    def title_of(self, c: Dict[str, Any]) -> str:
        from .tools import sanitize_tool_surface

        if c["title"] and "title" not in (c["pending"] or ()):
            return sanitize_tool_surface(str(c["title"]), 100)
        return f"(untitled yet - {int(c['size'])} memories)"

    def summary_of(self, c: Dict[str, Any], cap: int) -> str:
        from .tools import sanitize_tool_surface

        if c["summary"] and "summary" not in (c["pending"] or ()):
            return sanitize_tool_surface(str(c["summary"]), cap)
        return "(its summary is not written yet - it is written while you sleep)"

    def member_line(self, gid: str, score: Optional[float] = None) -> str:
        from .memory_reader import memory_tag
        from .tools import sanitize_tool_surface

        a = self.digests[gid]
        tag = memory_tag(gid)
        self.reader.tag_map.setdefault(tag, gid)  # readable at once with read_memory
        attrs = a.attributes if isinstance(a.attributes, dict) else {}
        kind = str(attrs.get("record_kind") or "memory")
        title = sanitize_tool_surface(str(attrs.get("title") or ""), 100) or "(untitled yet)"
        extra = " (placed since your last sleep)" if gid in self.provisional else ""
        close = f" (closeness {score:.2f})" if score is not None else ""
        return f"- #{tag} [{kind} {str(a.observed_at or '')[:10]}] {title}{close}{extra}"

    def header(self, c: Dict[str, Any]) -> List[str]:
        return [f"Topic #{self.tag[str(c['key'])]}: {self.title_of(c)} ({int(c['size'])} memories)",
                self.summary_of(c, 400)]

    def rank(self, ids: Sequence[str], words: str, limit: int) -> Optional[List[Tuple[float, str]]]:
        """(closeness, memory id) best first, by meaning: the store's vector query over exactly these memories'
        digests (memory's VectorMatrix), one embedding of the words. None when no embedder serves this home. The
        query is anchored on the memories' own digest ids (all read from his ladder) and his owner id."""
        from abstractmemory import TripleQuery

        by_owner: Dict[str, List[str]] = {}
        subject_of: Dict[str, str] = {}
        for gid in ids:
            a = self.digests[gid]
            if a.assertion_id:
                by_owner.setdefault(str(a.owner_id or ""), []).append(str(a.assertion_id))
                subject_of[str(a.assertion_id)] = gid
        ranked: List[Tuple[float, str]] = []
        for owner, aids in by_owner.items():
            try:
                rows = self.reader.home.store.query(TripleQuery(
                    predicate="dcterms:abstract", owner_id=owner, assertion_ids=tuple(aids), query_text=words,
                    limit=int(limit)))
            except ValueError:
                return None  # no embedder in this home (query_text needs one)
            for a in rows:
                retrieval = (a.attributes or {}).get("_retrieval") if isinstance(a.attributes, dict) else None
                score = float((retrieval or {}).get("score") or 0.0)
                gid = subject_of.get(str(a.assertion_id))
                if gid is not None:
                    ranked.append((score, gid))
        ranked.sort(key=lambda sg: -sg[0])
        return ranked[: int(limit)]

    # -------------------------------------------------------------------------------------------- answers
    def listing(self) -> str:
        total = sum(int(c["size"]) for c in self.topics)
        lines = [f"Your topics - your memories grouped by what they are about ({len(self.topics)} topics over "
                 f"{total} memories, grouped and titled while you sleep):"]
        for c in self.topics[:TOPICS_SHOWN]:
            newest = str(c["facts"]["span"]["last"] or "?")[:10]  # memory's facts: no digest read for the list
            title = self.title_of(c)
            size = "" if title.startswith("(untitled yet") else f" ({int(c['size'])} memories, newest {newest})"
            lines.append(f"- #{self.tag[str(c['key'])]} {title}{size} - {self.summary_of(c, 140)}")
        if len(self.topics) > TOPICS_SHOWN:
            lines.append(f"(... and {len(self.topics) - TOPICS_SHOWN} smaller topics)")
        lines.append(self.footer())
        return "\n".join(lines)

    def topic(self, c: Dict[str, Any], words: str, *, note: str = "") -> str:
        from .tools import sanitize_tool_surface

        lines = self.header(c) + ([note] if note else [])
        ids = self.members(c)
        if words:
            ranked = self.rank(ids, words, len(ids) or 1)
            if ranked is None:
                lines.append("(no embedder serves this session, so its memories cannot be ranked by meaning - "
                             "newest first instead)")
            else:
                lines.append(f'Its memories closest to "{sanitize_tool_surface(words, 60)}" by meaning:')
                lines.extend(self.member_line(gid, score) for score, gid in ranked[:RANKED_SHOWN])
                lines.append(self.footer())
                return "\n".join(lines)
        lines.append("Its memories, newest first:")
        lines.extend(self.member_line(gid) for gid in self.newest(ids)[:MEMBERS_SHOWN])
        if len(ids) > MEMBERS_SHOWN:
            lines.append(f"(... and {len(ids) - MEMBERS_SHOWN} more - topics #{self.tag[str(c['key'])]} <words> ranks "
                         "them by meaning)")
        lines.append(self.footer())
        return "\n".join(lines)

    def closest(self, words: str) -> str:
        from .tools import sanitize_tool_surface

        pool = [gid for c in self.topics for gid in self.members(c)]
        ranked = self.rank(pool, words, RANK_POOL)
        if ranked is None:
            return ("(no embedder serves this session, so your topics cannot be matched by meaning - here is the "
                    "list; name one by its #tag)\n\n" + self.listing())
        if not ranked:
            return f'Nothing in your topics is close to "{sanitize_tool_surface(words, 60)}".\n\n' + self.listing()
        scores: Dict[str, List[float]] = {}
        for score, gid in ranked:
            scores.setdefault(str(self.topic_of[gid]["key"]), []).append(score)
        # A topic's closeness: the mean of its three closest memories (fewer count as zero - one stray memory does
        # not make a topic close).
        order = sorted(scores, key=lambda key: (-sum(sorted(scores[key], reverse=True)[:3]) / 3.0, key))
        best = next(c for c in self.topics if str(c["key"]) == order[0])
        lines = [f'The topic closest to "{sanitize_tool_surface(words, 60)}":'] + self.header(best)
        lines.append("Its closest memories:")
        lines.extend(self.member_line(gid, score) for score, gid in ranked
                     if self.topic_of[gid] is best)
        if len(lines) > 4 + RANKED_SHOWN:
            lines = lines[:4 + RANKED_SHOWN]
        others = [c for key in order[1:4] for c in self.topics if str(c["key"]) == key]
        if others:
            lines.append("Also close: " + "; ".join(f"#{self.tag[str(c['key'])]} {self.title_of(c)}" for c in others))
        lines.append(self.footer())
        return "\n".join(lines)

    @staticmethod
    def footer() -> str:
        return ("topics #tag shows one topic's memories (or the topic of a memory); topics <words> finds the topic "
                "closest to them; read_memory #tag reads a memory.")
