"""Memory compaction helpers (text-focused; graph-ready contracts).

This module keeps compaction mechanics small and deterministic:
- normalize message metadata (message_id, timestamp)
- split messages into system / older conversation / recent conversation
- build span metadata for ArtifactStore persistence

LLM prompting is handled by the runtime effect handler; this module stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import uuid


def _ensure_message_id(meta: Dict[str, Any]) -> str:
    mid = meta.get("message_id")
    if isinstance(mid, str) and mid:
        return mid
    mid = f"msg_{uuid.uuid4().hex}"
    meta["message_id"] = mid
    return mid


def normalize_messages(
    messages: Sequence[Any],
    *,
    now_iso: Callable[[], str],
) -> List[Dict[str, Any]]:
    """Return a JSON-safe copy of messages with stable ids and timestamps."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        m_copy = dict(m)
        m_copy["role"] = str(m_copy.get("role") or "user")
        m_copy["content"] = "" if m_copy.get("content") is None else str(m_copy.get("content"))

        meta = m_copy.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            m_copy["metadata"] = meta
        _ensure_message_id(meta)

        ts = m_copy.get("timestamp")
        if not isinstance(ts, str) or not ts.strip():
            m_copy["timestamp"] = str(now_iso())

        out.append(m_copy)
    return out


@dataclass(frozen=True)
class CompactionSplit:
    system_messages: List[Dict[str, Any]]
    older_messages: List[Dict[str, Any]]
    recent_messages: List[Dict[str, Any]]

    @property
    def non_system_count(self) -> int:
        return len(self.older_messages) + len(self.recent_messages)


def split_for_compaction(
    messages: Sequence[Dict[str, Any]],
    *,
    preserve_recent: int,
) -> CompactionSplit:
    preserve = int(preserve_recent)
    if preserve < 0:
        preserve = 0

    system_messages = [m for m in messages if m.get("role") == "system"]
    conversation = [m for m in messages if m.get("role") != "system"]

    if preserve == 0:
        return CompactionSplit(system_messages=system_messages, older_messages=conversation, recent_messages=[])
    if len(conversation) <= preserve:
        return CompactionSplit(system_messages=system_messages, older_messages=[], recent_messages=conversation)

    older = conversation[:-preserve]
    recent = conversation[-preserve:]
    return CompactionSplit(system_messages=system_messages, older_messages=older, recent_messages=recent)


def span_metadata_from_messages(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute provenance-friendly span metadata from a non-empty message list."""
    if not messages:
        raise ValueError("span_metadata_from_messages requires non-empty messages")

    first = messages[0]
    last = messages[-1]
    first_meta = first.get("metadata") if isinstance(first.get("metadata"), dict) else {}
    last_meta = last.get("metadata") if isinstance(last.get("metadata"), dict) else {}

    return {
        "from_timestamp": first.get("timestamp"),
        "to_timestamp": last.get("timestamp"),
        "from_message_id": first_meta.get("message_id"),
        "to_message_id": last_meta.get("message_id"),
        "message_count": len(messages),
    }

