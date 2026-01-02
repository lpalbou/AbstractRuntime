"""Active context policy + provenance-based recall utilities.

Goal: keep a strict separation between:
- Stored memory (durable): RunStore/LedgerStore/ArtifactStore
- Active context (LLM-visible view): RunState.vars["context"]["messages"]

This module is intentionally small and JSON-safe. It does not implement semantic
search or "graph compression"; it establishes the contracts needed for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..core.models import RunState
from ..core.vars import get_context, get_limits, get_runtime
from ..storage.artifacts import ArtifactMetadata, ArtifactStore
from ..storage.base import RunStore


def _parse_iso(ts: str) -> datetime:
    value = (ts or "").strip()
    if not value:
        raise ValueError("empty timestamp")
    # Accept both "+00:00" and "Z"
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class TimeRange:
    """A closed interval [start, end] in ISO8601 strings.

    If start or end is None, the range is unbounded on that side.
    """

    start: Optional[str] = None
    end: Optional[str] = None

    def contains(self, *, start: Optional[str], end: Optional[str]) -> bool:
        """Return True if [start,end] intersects this range.

        Spans missing timestamps are treated as non-matching when a range is used,
        because we cannot prove they belong to the requested interval.
        """

        if self.start is None and self.end is None:
            return True
        if not start or not end:
            return False

        span_start = _parse_iso(start)
        span_end = _parse_iso(end)
        range_start = _parse_iso(self.start) if self.start else None
        range_end = _parse_iso(self.end) if self.end else None

        # Intersection test for closed ranges:
        # span_end >= range_start AND span_start <= range_end
        if range_start is not None and span_end < range_start:
            return False
        if range_end is not None and span_start > range_end:
            return False
        return True


SpanId = Union[str, int]  # artifact_id or 1-based index into _runtime.memory_spans


class ActiveContextPolicy:
    """Runtime-owned utilities for memory spans and active context."""

    def __init__(
        self,
        *,
        run_store: RunStore,
        artifact_store: ArtifactStore,
    ) -> None:
        self._run_store = run_store
        self._artifact_store = artifact_store

    # ------------------------------------------------------------------
    # Spans: list + filter
    # ------------------------------------------------------------------

    def list_memory_spans(self, run_id: str) -> List[Dict[str, Any]]:
        """Return the run's archived span index (`_runtime.memory_spans`)."""
        run = self._require_run(run_id)
        return self.list_memory_spans_from_run(run)

    @staticmethod
    def list_memory_spans_from_run(run: RunState) -> List[Dict[str, Any]]:
        """Return the archived span index (`_runtime.memory_spans`) from an in-memory RunState."""
        runtime_ns = get_runtime(run.vars)
        spans = runtime_ns.get("memory_spans")
        if not isinstance(spans, list):
            return []
        out: List[Dict[str, Any]] = []
        for s in spans:
            if isinstance(s, dict):
                out.append(dict(s))
        out.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
        return out

    def filter_spans(
        self,
        run_id: str,
        *,
        time_range: Optional[TimeRange] = None,
        tags: Optional[Dict[str, Any]] = None,
        tags_mode: str = "all",
        authors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Filter archived spans by time range, tags, and a basic keyword query.

        Notes:
        - This is a metadata filter, not semantic retrieval.
        - `query` matches against summary messages (if present) and metadata tags.
        """
        run = self._require_run(run_id)
        return self.filter_spans_from_run(
            run,
            artifact_store=self._artifact_store,
            time_range=time_range,
            tags=tags,
            tags_mode=tags_mode,
            authors=authors,
            locations=locations,
            query=query,
            limit=limit,
        )

    @staticmethod
    def filter_spans_from_run(
        run: RunState,
        *,
        artifact_store: ArtifactStore,
        time_range: Optional[TimeRange] = None,
        tags: Optional[Dict[str, Any]] = None,
        tags_mode: str = "all",
        authors: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Like `filter_spans`, but operates on an in-memory RunState."""
        spans = ActiveContextPolicy.list_memory_spans_from_run(run)
        if not spans:
            return []

        summary_by_artifact = ActiveContextPolicy.summary_text_by_artifact_id_from_run(run)
        lowered_query = (query or "").strip().lower() if query else None

        # Notes are small; for keyword filtering we can load their text safely.
        # IMPORTANT: only do this when we actually have a query (avoids unnecessary I/O).
        #
        # Also include summary text from the note's linked conversation span(s) when available,
        # so searching for a topic can surface both the span *and* the derived note.
        note_by_artifact: Dict[str, str] = {}
        if lowered_query:
            for span in spans:
                if not isinstance(span, dict):
                    continue
                if str(span.get("kind") or "") != "memory_note":
                    continue
                artifact_id = str(span.get("artifact_id") or "")
                if not artifact_id:
                    continue
                try:
                    payload = artifact_store.load_json(artifact_id)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue

                parts: list[str] = []
                note = payload.get("note")
                if isinstance(note, str) and note.strip():
                    parts.append(note.strip())

                sources = payload.get("sources")
                if isinstance(sources, dict):
                    span_ids = sources.get("span_ids")
                    if isinstance(span_ids, list):
                        for sid in span_ids:
                            if not isinstance(sid, str) or not sid.strip():
                                continue
                            summary = summary_by_artifact.get(sid.strip())
                            if isinstance(summary, str) and summary.strip():
                                parts.append(summary.strip())

                if parts:
                    note_by_artifact[artifact_id] = "\n".join(parts).strip()

        def _artifact_meta(artifact_id: str) -> Optional[ArtifactMetadata]:
            try:
                return artifact_store.get_metadata(artifact_id)
            except Exception:
                return None

        mode = str(tags_mode or "all").strip().lower() or "all"
        if mode in {"and"}:
            mode = "all"
        if mode in {"or"}:
            mode = "any"
        if mode not in {"all", "any"}:
            mode = "all"

        authors_norm = {str(a).strip().lower() for a in (authors or []) if isinstance(a, str) and a.strip()}
        locations_norm = {str(l).strip().lower() for l in (locations or []) if isinstance(l, str) and l.strip()}

        out: List[Dict[str, Any]] = []
        for span in spans:
            artifact_id = str(span.get("artifact_id") or "")
            if not artifact_id:
                continue

            if time_range is not None:
                if not time_range.contains(
                    start=span.get("from_timestamp"),
                    end=span.get("to_timestamp"),
                ):
                    continue

            meta = _artifact_meta(artifact_id)

            if tags:
                if not ActiveContextPolicy._tags_match(span=span, meta=meta, required=tags, mode=mode):
                    continue

            if authors_norm:
                created_by = span.get("created_by")
                author = str(created_by).strip().lower() if isinstance(created_by, str) else ""
                if not author or author not in authors_norm:
                    continue

            if locations_norm:
                loc = span.get("location")
                loc_str = str(loc).strip().lower() if isinstance(loc, str) else ""
                if not loc_str:
                    # Fallback: allow location to be stored as a tag.
                    span_tags = span.get("tags") if isinstance(span.get("tags"), dict) else {}
                    tag_loc = span_tags.get("location") if isinstance(span_tags, dict) else None
                    loc_str = str(tag_loc).strip().lower() if isinstance(tag_loc, str) else ""
                if not loc_str or loc_str not in locations_norm:
                    continue

            if lowered_query:
                haystack = ActiveContextPolicy._span_haystack(
                    span=span,
                    meta=meta,
                    summary=summary_by_artifact.get(artifact_id),
                    note=note_by_artifact.get(artifact_id),
                )
                if lowered_query not in haystack:
                    continue

            out.append(span)
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # Rehydration: stored span -> active context
    # ------------------------------------------------------------------

    def rehydrate_into_context(
        self,
        run_id: str,
        *,
        span_ids: Sequence[SpanId],
        placement: str = "after_summary",
        dedup_by: str = "message_id",
        max_messages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Rehydrate archived span(s) into `context.messages` and persist the run.

        Args:
            run_id: Run to mutate.
            span_ids: Sequence of artifact_ids or 1-based indices into `_runtime.memory_spans`.
            placement: Where to insert. Supported: "after_summary" (default), "after_system", "end".
            dedup_by: Dedup key (default: metadata.message_id).
            max_messages: Optional cap on inserted messages across all spans (None = unlimited).
        """
        run = self._require_run(run_id)
        out = self.rehydrate_into_context_from_run(
            run,
            span_ids=span_ids,
            placement=placement,
            dedup_by=dedup_by,
            max_messages=max_messages,
        )

        self._run_store.save(run)
        return out

    def rehydrate_into_context_from_run(
        self,
        run: RunState,
        *,
        span_ids: Sequence[SpanId],
        placement: str = "after_summary",
        dedup_by: str = "message_id",
        max_messages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Like `rehydrate_into_context`, but operates on an in-memory RunState.

        This mutates `run.vars["context"]["messages"]` (and `run.output["messages"]` when present),
        but does NOT persist the run.
        """
        spans = self.list_memory_spans_from_run(run)
        resolved_artifacts: List[str] = self.resolve_span_ids_from_spans(span_ids, spans)
        if not resolved_artifacts:
            return {"inserted": 0, "skipped": 0, "artifacts": []}

        ctx = get_context(run.vars)
        active = ctx.get("messages")
        if not isinstance(active, list):
            active = []

        inserted_total = 0
        skipped_total = 0
        per_artifact: List[Dict[str, Any]] = []

        # Build a dedup set for active context.
        existing_keys = self._collect_message_keys(active, dedup_by=dedup_by)

        # Normalize cap.
        try:
            max_messages_int = int(max_messages) if max_messages is not None else None
        except Exception:
            max_messages_int = None
        if max_messages_int is not None and max_messages_int < 0:
            max_messages_int = None

        def _preview_inserted(messages: Sequence[Dict[str, Any]]) -> str:
            """Build a small, human-friendly preview for UI/observability."""
            if not messages:
                return ""
            lines: list[str] = []
            for m in messages[:3]:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "").strip()
                content = str(m.get("content") or "").strip()
                if not content:
                    continue
                # If the synthetic memory note marker is present, keep it as-is (no "role:" prefix).
                if content.startswith("[MEMORY NOTE]"):
                    lines.append(content)
                else:
                    lines.append(f"{role}: {content}" if role else content)
            text = "\n".join([l for l in lines if l]).strip()
            if len(text) > 360:
                return text[:357] + "…"
            return text

        for artifact_id in resolved_artifacts:
            if max_messages_int is not None and inserted_total >= max_messages_int:
                # Deterministic: stop inserting once the global cap is reached.
                per_artifact.append(
                    {"artifact_id": artifact_id, "inserted": 0, "skipped": 0, "error": "max_messages"}
                )
                continue

            archived = self._artifact_store.load_json(artifact_id)
            archived_messages = archived.get("messages") if isinstance(archived, dict) else None

            # Support rehydrating memory_note spans into context as a single synthetic message.
            # This is the most practical "make recalled memory LLM-visible" behavior for
            # visual workflows (users expect "Recall into context" to work with notes).
            if not isinstance(archived_messages, list):
                note_text = None
                if isinstance(archived, dict):
                    raw_note = archived.get("note")
                    if isinstance(raw_note, str) and raw_note.strip():
                        note_text = raw_note.strip()
                if note_text is not None:
                    archived_messages = [
                        {
                            "role": "system",
                            "content": f"[MEMORY NOTE]\n{note_text}",
                            "timestamp": str(archived.get("created_at") or "") if isinstance(archived, dict) else "",
                            "metadata": {
                                "kind": "memory_note",
                                "rehydrated": True,
                                "source_artifact_id": artifact_id,
                                "message_id": f"memory_note:{artifact_id}",
                            },
                        }
                    ]
                else:
                    per_artifact.append(
                        {"artifact_id": artifact_id, "inserted": 0, "skipped": 0, "error": "missing_messages"}
                    )
                    continue

            to_insert: List[Dict[str, Any]] = []
            skipped = 0
            for m in archived_messages:
                if not isinstance(m, dict):
                    continue
                m_copy = dict(m)
                meta_copy = m_copy.get("metadata")
                if not isinstance(meta_copy, dict):
                    meta_copy = {}
                    m_copy["metadata"] = meta_copy
                # Mark as rehydrated view (do not mutate the archived artifact payload).
                meta_copy.setdefault("rehydrated", True)
                meta_copy.setdefault("source_artifact_id", artifact_id)

                key = self._message_key(m_copy, dedup_by=dedup_by)
                if key and key in existing_keys:
                    skipped += 1
                    continue
                if key:
                    existing_keys.add(key)
                to_insert.append(m_copy)

            if max_messages_int is not None:
                remaining = max(0, max_messages_int - inserted_total)
                if remaining <= 0:
                    per_artifact.append(
                        {"artifact_id": artifact_id, "inserted": 0, "skipped": 0, "error": "max_messages"}
                    )
                    continue
                if len(to_insert) > remaining:
                    to_insert = to_insert[:remaining]

            idx = self._insertion_index(active, artifact_id=artifact_id, placement=placement)
            active[idx:idx] = to_insert

            inserted_total += len(to_insert)
            skipped_total += skipped
            entry: Dict[str, Any] = {"artifact_id": artifact_id, "inserted": len(to_insert), "skipped": skipped}
            preview = _preview_inserted(to_insert)
            if preview:
                entry["preview"] = preview
            per_artifact.append(entry)

        ctx["messages"] = active
        if isinstance(getattr(run, "output", None), dict):
            run.output["messages"] = active

        return {"inserted": inserted_total, "skipped": skipped_total, "artifacts": per_artifact}

    # ------------------------------------------------------------------
    # Deriving what the LLM sees
    # ------------------------------------------------------------------

    def select_active_messages_for_llm(
        self,
        run_id: str,
        *,
        max_history_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the active-context view that should be sent to an LLM.

        This does NOT mutate the run; it returns a derived view.

        Rules (minimal, stable):
        - Always preserve system messages
        - Apply `max_history_messages` to non-system messages only
        - If max_history_messages is None, read it from `_limits.max_history_messages`
        """
        run = self._require_run(run_id)
        return self.select_active_messages_for_llm_from_run(
            run,
            max_history_messages=max_history_messages,
        )

    @staticmethod
    def select_active_messages_for_llm_from_run(
        run: RunState,
        *,
        max_history_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Like `select_active_messages_for_llm`, but operates on an in-memory RunState."""
        ctx = get_context(run.vars)
        messages = ctx.get("messages")
        if not isinstance(messages, list):
            return []

        if max_history_messages is None:
            limits = get_limits(run.vars)
            try:
                max_history_messages = int(limits.get("max_history_messages", -1))
            except Exception:
                max_history_messages = -1

        return ActiveContextPolicy.select_messages_view(messages, max_history_messages=max_history_messages)

    @staticmethod
    def select_messages_view(
        messages: Sequence[Any],
        *,
        max_history_messages: int,
    ) -> List[Dict[str, Any]]:
        """Select an LLM-visible view from a message list under a simple history limit."""
        system_msgs: List[Dict[str, Any]] = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
        convo_msgs: List[Dict[str, Any]] = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]

        if max_history_messages == -1:
            return system_msgs + convo_msgs
        if max_history_messages < 0:
            return system_msgs + convo_msgs
        if max_history_messages == 0:
            return system_msgs
        return system_msgs + convo_msgs[-max_history_messages:]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_run(self, run_id: str) -> RunState:
        run = self._run_store.load(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return run

    @staticmethod
    def resolve_span_ids_from_spans(span_ids: Sequence[SpanId], spans: Sequence[Dict[str, Any]]) -> List[str]:
        resolved: List[str] = []
        for sid in span_ids:
            if isinstance(sid, int):
                idx = sid - 1
                if 0 <= idx < len(spans):
                    artifact_id = spans[idx].get("artifact_id")
                    if isinstance(artifact_id, str) and artifact_id:
                        resolved.append(artifact_id)
                continue
            if isinstance(sid, str):
                s = sid.strip()
                if not s:
                    continue
                # If it's a digit string, treat as 1-based index.
                if s.isdigit():
                    idx = int(s) - 1
                    if 0 <= idx < len(spans):
                        artifact_id = spans[idx].get("artifact_id")
                        if isinstance(artifact_id, str) and artifact_id:
                            resolved.append(artifact_id)
                    continue
                resolved.append(s)
        # Preserve order but dedup
        seen = set()
        out: List[str] = []
        for a in resolved:
            if a in seen:
                continue
            seen.add(a)
            out.append(a)
        return out

    def _insertion_index(self, active: List[Any], *, artifact_id: str, placement: str) -> int:
        if placement == "end":
            return len(active)

        if placement == "after_system":
            i = 0
            while i < len(active):
                m = active[i]
                if not isinstance(m, dict) or m.get("role") != "system":
                    break
                i += 1
            return i

        # default: after_summary
        for i, m in enumerate(active):
            if not isinstance(m, dict):
                continue
            if m.get("role") != "system":
                continue
            meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
            if meta.get("kind") == "memory_summary" and meta.get("source_artifact_id") == artifact_id:
                return i + 1

        # Fallback: after system messages.
        return self._insertion_index(active, artifact_id=artifact_id, placement="after_system")

    def _collect_message_keys(self, messages: Iterable[Any], *, dedup_by: str) -> set[str]:
        keys: set[str] = set()
        for m in messages:
            if not isinstance(m, dict):
                continue
            key = self._message_key(m, dedup_by=dedup_by)
            if key:
                keys.add(key)
        return keys

    def _message_key(self, message: Dict[str, Any], *, dedup_by: str) -> Optional[str]:
        if dedup_by == "message_id":
            meta = message.get("metadata")
            if isinstance(meta, dict):
                mid = meta.get("message_id")
                if isinstance(mid, str) and mid:
                    return mid
            return None
        # Unknown dedup key: disable dedup
        return None

    @staticmethod
    def summary_text_by_artifact_id_from_run(run: RunState) -> Dict[str, str]:
        ctx = get_context(run.vars)
        messages = ctx.get("messages")
        if not isinstance(messages, list):
            return {}
        out: Dict[str, str] = {}
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "system":
                continue
            meta = m.get("metadata")
            if not isinstance(meta, dict):
                continue
            if meta.get("kind") != "memory_summary":
                continue
            artifact_id = meta.get("source_artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                out[artifact_id] = str(m.get("content") or "")
        return out

    @staticmethod
    def _span_haystack(
        *,
        span: Dict[str, Any],
        meta: Optional[ArtifactMetadata],
        summary: Optional[str],
        note: Optional[str] = None,
    ) -> str:
        parts: List[str] = []
        if summary:
            parts.append(summary)
        if note:
            parts.append(note)
        for k in ("kind", "compression_mode", "focus", "from_timestamp", "to_timestamp", "created_by", "location"):
            v = span.get(k)
            if isinstance(v, str) and v:
                parts.append(v)
        # Span tags are persisted in run vars (topic/person/project, etc).
        span_tags = span.get("tags")
        if isinstance(span_tags, dict):
            for k, v in span_tags.items():
                if isinstance(v, str) and v:
                    parts.append(str(k))
                    parts.append(v)

        if meta is not None:
            parts.append(meta.content_type or "")
            for k, v in (meta.tags or {}).items():
                parts.append(k)
                parts.append(v)
        return " ".join(parts).lower()

    @staticmethod
    def _tags_match(
        *,
        span: Dict[str, Any],
        meta: Optional[ArtifactMetadata],
        required: Dict[str, Any],
        mode: str = "all",
    ) -> bool:
        def _norm(s: str) -> str:
            return str(s or "").strip().lower()

        tags: Dict[str, str] = {}
        if meta is not None and meta.tags:
            for k, v in meta.tags.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                kk = _norm(k)
                vv = _norm(v)
                if kk and vv and kk not in tags:
                    tags[kk] = vv

        span_tags = span.get("tags")
        if isinstance(span_tags, dict):
            for k, v in span_tags.items():
                if not isinstance(k, str):
                    continue
                kk = _norm(k)
                if not kk or kk in tags:
                    continue
                if isinstance(v, str):
                    vv = _norm(v)
                    if vv:
                        tags[kk] = vv

        # Derived tags from span ref (cheap and keeps filtering usable even
        # if artifact metadata is missing).
        for k in ("kind", "compression_mode", "focus"):
            v = span.get(k)
            if isinstance(v, str) and v:
                kk = _norm(k)
                if kk and kk not in tags:
                    tags[kk] = _norm(v)

        required_norm: Dict[str, List[str]] = {}
        for k, v in (required or {}).items():
            if not isinstance(k, str):
                continue
            kk = _norm(k)
            if not kk or kk == "kind":
                continue
            values: List[str] = []
            if isinstance(v, str):
                vv = _norm(v)
                if vv:
                    values.append(vv)
            elif isinstance(v, (list, tuple)):
                for it in v:
                    if isinstance(it, str) and it.strip():
                        values.append(_norm(it))
            if values:
                # preserve order but dedup
                seen: set[str] = set()
                deduped = []
                for x in values:
                    if x in seen:
                        continue
                    seen.add(x)
                    deduped.append(x)
                required_norm[kk] = deduped

        if not required_norm:
            return True

        def _key_matches(key: str) -> bool:
            cand = tags.get(key)
            if cand is None:
                return False
            allowed = required_norm.get(key) or []
            return cand in allowed

        op = str(mode or "all").strip().lower() or "all"
        if op not in {"all", "any"}:
            op = "all"
        if op == "any":
            return any(_key_matches(k) for k in required_norm.keys())
        return all(_key_matches(k) for k in required_norm.keys())
