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
    """A closed-open interval [start, end] in ISO8601 strings.

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

        # Intersection test for closed-open ranges:
        # span_end > range_start AND span_start < range_end
        if range_start is not None and not (span_end > range_start):
            return False
        if range_end is not None and not (span_start < range_end):
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
        tags: Optional[Dict[str, str]] = None,
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
            query=query,
            limit=limit,
        )

    @staticmethod
    def filter_spans_from_run(
        run: RunState,
        *,
        artifact_store: ArtifactStore,
        time_range: Optional[TimeRange] = None,
        tags: Optional[Dict[str, str]] = None,
        query: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Like `filter_spans`, but operates on an in-memory RunState."""
        spans = ActiveContextPolicy.list_memory_spans_from_run(run)
        if not spans:
            return []

        summary_by_artifact = ActiveContextPolicy.summary_text_by_artifact_id_from_run(run)

        def _artifact_meta(artifact_id: str) -> Optional[ArtifactMetadata]:
            try:
                return artifact_store.get_metadata(artifact_id)
            except Exception:
                return None

        lowered_query = (query or "").strip().lower() if query else None

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
                if not ActiveContextPolicy._tags_match(span=span, meta=meta, required=tags):
                    continue

            if lowered_query:
                haystack = ActiveContextPolicy._span_haystack(
                    span=span, meta=meta, summary=summary_by_artifact.get(artifact_id)
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
    ) -> Dict[str, Any]:
        """Rehydrate archived span(s) into `context.messages` and persist the run.

        Args:
            run_id: Run to mutate.
            span_ids: Sequence of artifact_ids or 1-based indices into `_runtime.memory_spans`.
            placement: Where to insert. Supported: "after_summary" (default), "after_system", "end".
            dedup_by: Dedup key (default: metadata.message_id).
        """
        run = self._require_run(run_id)
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

        for artifact_id in resolved_artifacts:
            archived = self._artifact_store.load_json(artifact_id)
            archived_messages = archived.get("messages") if isinstance(archived, dict) else None
            if not isinstance(archived_messages, list):
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

            idx = self._insertion_index(active, artifact_id=artifact_id, placement=placement)
            active[idx:idx] = to_insert

            inserted_total += len(to_insert)
            skipped_total += skipped
            per_artifact.append({"artifact_id": artifact_id, "inserted": len(to_insert), "skipped": skipped})

        ctx["messages"] = active
        if isinstance(getattr(run, "output", None), dict):
            run.output["messages"] = active
        self._run_store.save(run)

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
    ) -> str:
        parts: List[str] = []
        if summary:
            parts.append(summary)
        for k in ("kind", "compression_mode", "focus", "from_timestamp", "to_timestamp"):
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
        required: Dict[str, str],
    ) -> bool:
        tags: Dict[str, str] = {}
        if meta is not None and meta.tags:
            tags.update(meta.tags)

        span_tags = span.get("tags")
        if isinstance(span_tags, dict):
            for k, v in span_tags.items():
                if isinstance(k, str) and isinstance(v, str) and v and k not in tags:
                    tags[k] = v

        # Derived tags from span ref (cheap and keeps filtering usable even
        # if artifact metadata is missing).
        for k in ("kind", "compression_mode", "focus"):
            v = span.get(k)
            if isinstance(v, str) and v and k not in tags:
                tags[k] = v

        return all(tags.get(k) == v for k, v in required.items())
