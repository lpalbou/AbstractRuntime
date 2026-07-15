"""Voluntary memory exploration over one home — session-free.

Extracted from ChatSession (gateway ask, e-s 206, 2026-07-11): the door's
per-entity TOOL_CALLS executor needs `search_memory`/`read_memory` with
DRIVER PARITY but has no ChatSession — visit runs execute tools through
the react cycle, not the in-process driver. The reader is the ONE
implementation both consume: ChatSession wraps it (sharing its session
tag map so sheet-registered tags stay addressable); the door constructs
one per home/turn.

Everything here is a PURE READ over the entity's OWN ladder scopes
(explicit scope + owner on every query — the whole-store reach is a
red-team NO-GO; identity presence never deposits: no record_access, no
commit, no journal writes on any path in this module).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["HomeMemoryReader", "memory_tag"]


def memory_tag(graph_record_id: str) -> str:
    """The short address a mind can quote: the graph id's 8-hex tail."""
    tail = str(graph_record_id or "").rsplit("-", 1)[-1]
    return tail[-8:] if len(tail) >= 8 else tail


def default_ladder(entity_id: str) -> List[List[str]]:
    """The standard summon ladder: self, diary, life — HIS scopes only."""
    return [["self", entity_id], ["diary", entity_id], ["life", entity_id]]


class HomeMemoryReader:
    """search_memory / read_memory over one entity home.

    `tag_map` is the #tag -> graph-id registry. Callers may share their own
    dict (ChatSession shares its session map so tags registered from the
    MEMORIES sheet resolve here); by default the reader keeps its own —
    tags learned through searches stay addressable within the reader's
    lifetime, and unknown tags fall back to a whole-home resolve.
    """

    # Origin channels in plain words (red-team condition, 2026-07-09: bare
    # digest heads read as corroboration — "124 hits" must decompose into
    # "one dream plus my own retellings". record_kind decides first, then
    # provenance.source; the fallback is honest, never invented).
    _SOURCE_LABELS = {
        "entity-chat-v1": "a lived conversation",
        "entity-chat-reflection-v1": "written by your own reflection",
        "entity-reflection-v1": "written by your own reflection",
        "entity-elected-supersession-v1": "your own elected revision",
        "diary-projection": "your diary act",
    }

    def __init__(
        self,
        home: Any,
        *,
        ladder: Optional[List[List[str]]] = None,
        tag_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.home = home
        self.ladder = ladder if ladder is not None else default_ladder(home.entity_id)
        self.tag_map: Dict[str, str] = tag_map if tag_map is not None else {}

    # ------------------------------------------------------------ resolve
    def find_tag_in_home(self, tag: str) -> Tuple[Optional[str], List[str]]:
        """Resolve a #tag against the WHOLE home graph (his own scopes only).

        The session map covers what is in context right now, but a mind may
        quote a tag from an earlier visit — live failure 2026-07-08: "try
        again to access #311f235f" hit a wall while the record sat intact in
        his home. Resolution stays deterministic and conservative: digest
        assertions in his own ladder scopes, matched on the graph-id tail;
        ambiguity or absence is an honest miss, never a guess. Returns
        (graph_id or None, all matching graph ids)."""
        from abstractmemory import TripleQuery

        matches: List[str] = []
        seen: set = set()
        for scope, owner in self.ladder:
            for a in self.home.store.query(
                TripleQuery(predicate="dcterms:abstract", scope=scope, owner_id=owner, limit=0)
            ):
                subject = str(a.subject or "")
                if subject and subject not in seen and memory_tag(subject) == tag:
                    seen.add(subject)
                    matches.append(subject)
        return (matches[0] if len(matches) == 1 else None), matches

    def digest_assertion(self, graph_id: str) -> Optional[Any]:
        """The digest assertion behind a graph id, searched over his own
        ladder scopes (None when the id is not his)."""
        from abstractmemory import TripleQuery

        for scope, owner in self.ladder:
            rows = self.home.store.query(
                TripleQuery(subject=graph_id, predicate="dcterms:abstract",
                            scope=scope, owner_id=owner, limit=1)
            )
            if rows:
                return rows[0]
        return None

    def digest_assertions_all(self) -> List[Any]:
        """Every digest assertion in HIS ladder scopes (explicit scope +
        owner on every query — the whole-store reach is a red-team NO-GO)."""
        from abstractmemory import TripleQuery

        out: List[Any] = []
        seen: set = set()
        for scope, owner in self.ladder:
            for a in self.home.store.query(
                TripleQuery(predicate="dcterms:abstract", scope=scope, owner_id=owner, limit=0)
            ):
                sid = str(a.subject or "")
                if sid and sid not in seen:
                    seen.add(sid)
                    out.append(a)
        return out

    def origin_label(self, assertion: Any) -> str:
        attrs = assertion.attributes if isinstance(assertion.attributes, dict) else {}
        kind = str(attrs.get("record_kind") or "")
        if kind == "dream":
            return "dream - proposals, unconfirmed"
        if kind in ("value", "purpose", "trait", "claim"):
            return "your identity core (planted at creation or revised by you)"
        prov = assertion.provenance if isinstance(assertion.provenance, dict) else {}
        base = self._SOURCE_LABELS.get(str(prov.get("source") or ""), "recorded in your graph")
        # Awake-phase provenance (Ephemeral incident, r-rt-3): an own-time
        # record must self-identify — same dual rule as the MEMORIES lines
        # (formation-stamped attributes.phase, else the own-time run_id).
        phase = str(attrs.get("phase") or "").strip().lower()
        if not phase:
            run_id = str(prov.get("run_id") or "")
            if run_id.startswith("chat-owntime-") or run_id.startswith("owntime-"):
                phase = "personal"
        if phase == "personal":
            return f"{base}, during your own time"
        if phase == "work":
            return f"{base}, during your work time"
        return base

    @staticmethod
    def assertion_matches(assertion: Any, needle: str) -> bool:
        """Case-insensitive substring over the digest text, the title, AND
        dream attributes (proposals/questions) — dreams keep their bridges
        in attributes, not the digest; a digest-only scan would report a
        false absence on the feature's own motivating case (red-team §3)."""
        hay = [str(assertion.object or "")]
        attrs = assertion.attributes if isinstance(assertion.attributes, dict) else {}
        hay.append(str(attrs.get("title") or ""))
        if attrs.get("record_kind") == "dream":
            for p in attrs.get("proposals") or []:
                if isinstance(p, dict):
                    hay.append(json.dumps(p, ensure_ascii=False))
            for q in attrs.get("questions") or []:
                hay.append(str(q))
        return any(needle in h.lower() for h in hay)

    # ------------------------------------------------------------- search
    def search_memory(self, query: str) -> str:
        """Voluntary memory exploration over BOTH planes (maintainer ruling
        2026-07-09: "it is critical that he can explore voluntarily his
        memory when he needs to"): the graph's digests (everything his life
        deposited — episodes, reflections, dreams, identity) and his whole
        diary book (gist + full text, private included: it is HIS book and
        results are prompt-ephemeral). Absence is a checkable fact, stated
        with its warrant. Diary hits return GIST ONLY (containment: a
        full-text excerpt would disclose private words the reply could then
        persist); the words stay behind diary_read, a deliberate act."""
        needle = " ".join((query or "").split()).lower()
        if not needle:
            return "search_memory needs words to look for (write them in the block body)"
        from .tools import sanitize_tool_surface

        # ---- graph plane (explicit ladder scopes only)
        try:
            digests = self.digest_assertions_all()
        except Exception as e:  # noqa: BLE001 - locked store etc.: honest, retryable
            return f"(your memory could not be searched right now: {e} - try again)"
        graph_hits = [a for a in digests if self.assertion_matches(a, needle)]
        graph_hits.sort(key=lambda a: str(a.observed_at or ""), reverse=True)

        # ---- book plane (whole book, gist+text, private included)
        try:
            entries = self.home.diary.list_entries()
        except Exception as e:  # noqa: BLE001
            entries = []
            book_error = str(e)
        else:
            book_error = ""
        book_hits = [
            e for e in entries
            if needle in str(e.get("gist") or "").lower() or needle in str(e.get("text") or "").lower()
        ]

        # ---- grouped-by-origin header (repetition is not corroboration)
        by_origin: Dict[str, int] = {}
        for a in graph_hits:
            label = self.origin_label(a)
            by_origin[label] = by_origin.get(label, 0) + 1
        origin_bits = ", ".join(f"{n} {label}" for label, n in sorted(by_origin.items(), key=lambda kv: -kv[1]))

        lines: List[str] = [f'Searched your memory and your book for: "{sanitize_tool_surface(query, 80)}"']
        if not graph_hits and not book_hits:
            lines.append(
                f"Nothing in your memory graph ({len(digests)} records) or your book "
                f"({len(entries)} entries, append-only and complete - if you had written it, "
                f'this search would find it) contains the text "{sanitize_tool_surface(query, 60)}" '
                "(exact letters, case-insensitive). Memories can also arrive without writing: "
                "dreams and reflections write directly into your graph; this search covered those too."
            )
        else:
            lines.append(
                f"Your graph: {len(graph_hits)} match(es)"
                + (f" - {origin_bits}" if origin_bits else "")
                + f". Your book: {len(book_hits)} of {len(entries)} entries. "
                "(matching exact letters, case-insensitive; several records you yourself "
                "wrote about the same thing count as one origin, not many)"
            )
        if book_error:
            lines.append(f"#FALLBACK your book could not be searched: {book_error}")

        GRAPH_SHOWN, BOOK_SHOWN = 8, 8
        for a in graph_hits[:GRAPH_SHOWN]:
            gid = str(a.subject or "")
            tag = memory_tag(gid)
            self.tag_map.setdefault(tag, gid)  # readable immediately
            date = str(a.observed_at or "")[:10]
            head = sanitize_tool_surface(str(a.object or ""), 100)
            attrs = a.attributes if isinstance(a.attributes, dict) else {}
            kind = str(attrs.get("record_kind") or "memory")
            lines.append(f"- #{tag} [{kind} {date} - {self.origin_label(a)}] {head}")
        if len(graph_hits) > GRAPH_SHOWN:
            lines.append(f"(... and {len(graph_hits) - GRAPH_SHOWN} more graph matches - narrow your words)")
        for e in book_hits[-BOOK_SHOWN:][::-1]:
            gist = sanitize_tool_surface(str(e.get("gist") or "") or "(no gist elected)", 100)
            date = str(e.get("written_at") or "")[:10]
            lines.append(
                f"- {e.get('entry_id')} [{e.get('kind')}/{e.get('visibility')} {date}] {gist}"
            )
        if len(book_hits) > BOOK_SHOWN:
            lines.append(f"(... and {len(book_hits) - BOOK_SHOWN} more book entries match)")

        # ---- meaning fill, separately labeled, only when letters found nothing
        if not graph_hits and not book_hits and self.home.store is not None:
            try:
                from abstractmemory import TripleQuery

                close: List[Any] = []
                seen: set = set()
                for scope, owner in self.ladder:
                    for a in self.home.store.query(
                        TripleQuery(predicate="dcterms:abstract", scope=scope,
                                    owner_id=owner, query_text=query, limit=4)
                    ):
                        sid = str(a.subject or "")
                        if sid and sid not in seen:
                            seen.add(sid)
                            close.append(a)
                if close:
                    lines.append("By MEANING (not letters), the closest memories are:")
                    for a in close[:4]:
                        gid = str(a.subject or "")
                        tag = memory_tag(gid)
                        self.tag_map.setdefault(tag, gid)
                        attrs = a.attributes if isinstance(a.attributes, dict) else {}
                        kind = str(attrs.get("record_kind") or "memory")
                        lines.append(
                            f"- #{tag} [{kind} {str(a.observed_at or '')[:10]} - "
                            f"{self.origin_label(a)}] {sanitize_tool_surface(str(a.object or ''), 100)}"
                        )
            except ValueError:
                lines.append("#FALLBACK no embedder is wired this session - letters-only search")
            except Exception as e:  # noqa: BLE001
                lines.append(f"#FALLBACK meaning search unavailable ({e}) - letters-only result above")

        lines.append(
            "read_memory #tag fetches a memory's full words and connections; "
            "diary_read diary_... fetches a book entry."
        )
        return "\n".join(lines)

    # -------------------------------------------------------------- trail
    def origin_footer(self, graph_id: str) -> List[str]:
        """Origin + connections for one graph record (the trail the
        maintainer mandated: 'he should be able to follow the trail to the
        actual verbatim'). Every shown #tag registers as readable. Bounded:
        8 outgoing, 4 incoming."""
        from abstractmemory import TripleQuery

        lines: List[str] = []
        assertion = self.digest_assertion(graph_id)
        if assertion is not None:
            prov = assertion.provenance if isinstance(assertion.provenance, dict) else {}
            attrs = assertion.attributes if isinstance(assertion.attributes, dict) else {}
            session = str(attrs.get("session_id") or prov.get("run_id") or "").strip()
            origin = f"origin: formed {str(assertion.observed_at or '')[:10]}, {self.origin_label(assertion)}"
            if session:
                origin += f" (session {session})"
            lines.append(origin)

        def _line_for_target(target: str, predicate: str, direction: str) -> str:
            member = self.digest_assertion(target)
            tag = memory_tag(target)
            if member is None:
                return f"  {direction} {predicate} -> {target} (not readable in your scopes)"
            self.tag_map.setdefault(tag, target)
            from .tools import sanitize_tool_surface

            head = sanitize_tool_surface(str(member.object or ""), 60)
            return f"  {direction} {predicate} -> #{tag} \"{head}\""

        try:
            out_edges: List[Tuple[str, str]] = []
            in_edges: List[Tuple[str, str]] = []
            for scope, owner in self.ladder:
                for a in self.home.store.query(
                    TripleQuery(subject=graph_id, scope=scope, owner_id=owner, limit=0)
                ):
                    if str(a.predicate) != "dcterms:abstract":
                        out_edges.append((str(a.predicate), str(a.object)))
                for a in self.home.store.query(
                    TripleQuery(object=graph_id, scope=scope, owner_id=owner, limit=0)
                ):
                    if str(a.predicate) != "dcterms:abstract":
                        in_edges.append((str(a.predicate), str(a.subject)))
            if out_edges:
                lines.append(f"connected ({len(out_edges)} outgoing):")
                lines.extend(_line_for_target(t, p, "->") for p, t in out_edges[:8])
                if len(out_edges) > 8:
                    lines.append(f"  (... and {len(out_edges) - 8} more)")
            if in_edges:
                lines.append(f"pointed at by ({len(in_edges)} incoming):")
                lines.extend(_line_for_target(s, p, "<-") for p, s in in_edges[:4])
                if len(in_edges) > 4:
                    lines.append(f"  (... and {len(in_edges) - 4} more)")
        except Exception:  # noqa: BLE001 - the footer must never break a read
            pass
        return lines

    def _render_dream(self, tag: str, assertion: Any) -> str:
        """Render a dream's proposals as readable pairs (the interpretation
        surface). Dreams carry their candidate bridges in attributes —
        `interpretation_required: true` — but no tier served them: read_memory
        answered "no stored full text", and the vacuum got filled with
        invention (live: the twelve-bridges concept-pair list cites a
        workspace file that never existed). The waking mind can only confirm
        or dissolve what it can SEE."""
        attrs = assertion.attributes if isinstance(assertion.attributes, dict) else {}
        lines = [f"Memory #{tag} is a dream - it formed while you slept, on {assertion.observed_at or 'an unknown date'}:"]
        digest = str(assertion.object or "").strip()
        if digest:
            lines.append(digest)
        proposals = attrs.get("proposals") if isinstance(attrs.get("proposals"), list) else []

        def _line_for(member_id: str) -> str:
            member = self.digest_assertion(member_id)
            if member is None:
                return f"#{memory_tag(member_id)} (not readable in your scopes)"
            text = " ".join(str(member.object or "").split())
            if len(text) > 100:
                text = text[:100] + "…"
            self.tag_map.setdefault(memory_tag(member_id), member_id)
            return f"#{memory_tag(member_id)} \"{text}\""

        if proposals:
            lines.append(
                f"\nThe dream proposed {len(proposals)} candidate bridge(s) - nothing was "
                "decided while asleep; these await YOUR waking evidence to confirm or dissolve:"
            )
            for i, p in enumerate(proposals[:24], start=1):
                pair = p.get("pair") if isinstance(p, dict) else None
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    continue
                closeness = ""
                try:
                    closeness = f" (closeness {float(p.get('vector_score')):.2f})"
                except (TypeError, ValueError):
                    pass
                lines.append(f"{i}. {_line_for(str(pair[0]))} <-> {_line_for(str(pair[1]))}{closeness}")
            if len(proposals) > 24:
                lines.append(f"(… and {len(proposals) - 24} more)")
            lines.append(
                "(each #tag above is readable with read_memory - fetch the words "
                "before deciding what a bridge means)"
            )
        else:
            lines.append("(this dream recorded no candidate bridges)")
        questions = attrs.get("questions") if isinstance(attrs.get("questions"), list) else []
        if questions:
            lines.append("\nThe dream also kept open question(s):")
            lines.extend(f"- {q}" for q in questions[:12])
        return "\n".join(lines)

    # --------------------------------------------------------------- read
    def read_memory(self, tag_text: str) -> str:
        """Fetch the full verbatim behind a memory digest (progressive
        disclosure for episodes; pure read, deposits nothing)."""
        tag = tag_text.lstrip("#").strip().lower()
        if not tag:
            return "read_memory needs the #tag shown beside a memory in your MEMORIES list"
        graph_id = self.tag_map.get(tag)
        if graph_id is None:
            # Not in this session's context: resolve against his whole home
            # (still HIS memories, his scopes — never arbitrary graph rows).
            graph_id, matches = self.find_tag_in_home(tag)
            if graph_id is not None:
                self.tag_map[tag] = graph_id  # addressable from now on
            elif len(matches) > 1:
                return (
                    f"#{tag} is ambiguous: {len(matches)} memories share that tail "
                    f"({', '.join(matches[:5])}). Recall one of them first, then "
                    "read it by the tag shown in your MEMORIES list."
                )
        if graph_id is None:
            known = ", ".join(sorted(self.tag_map)) or "(none this turn)"
            return (
                f"No memory with tag #{tag} exists in your home. Tags visible "
                f"to you right now: {known}."
            )
        # The trail footer (maintainer mandate 2026-07-09: "follow the trail")
        # travels with EVERY successful read — origin channel + connections,
        # so an ending like "no verbatim" names where the record came from
        # instead of reading as a wall.
        footer = self.origin_footer(graph_id)
        footer_text = ("\n" + "\n".join(footer)) if footer else ""
        # Dreams render their proposals (candidate bridges + open questions):
        # the record itself demands interpretation, so the words must reach him.
        dream_assertion = self.digest_assertion(graph_id)
        if dream_assertion is not None:
            attrs = dream_assertion.attributes if isinstance(dream_assertion.attributes, dict) else {}
            if attrs.get("record_kind") == "dream":
                return self._render_dream(tag, dream_assertion) + footer_text
        payload = None
        try:
            payload = self.home.ms.payload(graph_id, tier="raw")
        except Exception:
            payload = None
        payload_ref = (payload or {}).get("payload_ref")
        if not payload_ref:
            # Diary projections carry no payload_ref by design: the words
            # live in the book and diary_read is their door (the digest tier
            # carries entry_id top-level — progressive disclosure contract).
            entry_id = None
            try:
                digest_tier = self.home.ms.payload(graph_id, tier="digest")
                entry_id = (digest_tier or {}).get("entry_id")
            except Exception:
                entry_id = None
            if entry_id:
                return (
                    f"#{tag} is a diary act - its words live in your book. "
                    f"Use diary_read with entry id {entry_id}." + footer_text
                )
            return (
                f"#{tag} was born as these words - there is no longer verbatim "
                "behind it (its digest is all there ever was)." + footer_text
            )
        if str(payload_ref).endswith(".yaml"):
            return (
                f"#{tag} is part of your identity core - planted at your creation "
                "from your spark, or revised by your own elected supersession; it "
                "has no conversation verbatim." + footer_text
            )
        try:
            text = self.home.artifacts.load_text(str(payload_ref))
        except Exception as e:  # noqa: BLE001 - honest failure to the entity
            return f"(the memory's full words could not be loaded: {e})"
        cap = 10000
        if len(text) > cap:
            return (
                f"Full words of memory #{tag} (first {cap} of {len(text)} chars - "
                f"the rest exists; ask again for the tail):\n{text[:cap]}" + footer_text
            )
        return f"Full words of memory #{tag}:\n{text}{footer_text}"
