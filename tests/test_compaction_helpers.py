from __future__ import annotations

from abstractruntime.memory.compaction import normalize_messages, split_for_compaction, span_metadata_from_messages


def test_normalize_messages_adds_message_id_and_timestamp() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "metadata": {"message_id": "m2"}},
    ]
    out = normalize_messages(messages, now_iso=lambda: "2025-01-01T00:00:00+00:00")
    assert len(out) == 2
    assert out[0].get("timestamp") == "2025-01-01T00:00:00+00:00"
    assert isinstance((out[0].get("metadata") or {}).get("message_id"), str)
    assert (out[1].get("metadata") or {}).get("message_id") == "m2"


def test_split_for_compaction_preserves_system_and_recent_non_system() -> None:
    messages = [
        {"role": "system", "content": "sys", "timestamp": "t", "metadata": {"message_id": "s"}},
        {"role": "user", "content": "m1", "timestamp": "t", "metadata": {"message_id": "m1"}},
        {"role": "assistant", "content": "m2", "timestamp": "t", "metadata": {"message_id": "m2"}},
        {"role": "user", "content": "m3", "timestamp": "t", "metadata": {"message_id": "m3"}},
    ]
    split = split_for_compaction(messages, preserve_recent=1)
    assert [m.get("content") for m in split.system_messages] == ["sys"]
    assert [m.get("content") for m in split.older_messages] == ["m1", "m2"]
    assert [m.get("content") for m in split.recent_messages] == ["m3"]


def test_span_metadata_from_messages_uses_first_last() -> None:
    older = [
        {"role": "user", "content": "m1", "timestamp": "t1", "metadata": {"message_id": "m1"}},
        {"role": "assistant", "content": "m2", "timestamp": "t2", "metadata": {"message_id": "m2"}},
    ]
    meta = span_metadata_from_messages(older)
    assert meta["from_timestamp"] == "t1"
    assert meta["to_timestamp"] == "t2"
    assert meta["from_message_id"] == "m1"
    assert meta["to_message_id"] == "m2"
    assert meta["message_count"] == 2

