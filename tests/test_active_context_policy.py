from __future__ import annotations

from typing import Any

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.memory import ActiveContextPolicy, TimeRange
from abstractruntime.storage.artifacts import FileArtifactStore, InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore


def _run(
    *,
    run_id: str,
    messages: list[dict[str, Any]],
    memory_spans: list[dict[str, Any]],
) -> RunState:
    return RunState(
        run_id=run_id,
        workflow_id="wf_test",
        status=RunStatus.COMPLETED,
        current_node="done",
        vars={
            "context": {"task": "test", "messages": messages},
            "scratchpad": {},
            "_runtime": {"memory_spans": memory_spans},
            "_temp": {},
            "_limits": {"max_history_messages": -1},
        },
        waiting=None,
        output={"messages": messages},
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )


def test_filter_spans_by_time_tags_and_query() -> None:
    run_id = "run_filter"
    run_store = InMemoryRunStore()
    artifact_store = InMemoryArtifactStore()
    policy = ActiveContextPolicy(run_store=run_store, artifact_store=artifact_store)

    span1_from = "2025-01-01T00:00:00+00:00"
    span1_to = "2025-01-01T00:10:00+00:00"
    span2_from = "2025-01-01T01:00:00+00:00"
    span2_to = "2025-01-01T01:10:00+00:00"

    payload1 = {
        "messages": [
            {"role": "user", "content": "alpha details", "timestamp": span1_from, "metadata": {"message_id": "m1"}},
            {"role": "assistant", "content": "alpha response", "timestamp": span1_to, "metadata": {"message_id": "m2"}},
        ],
        "span": {"from_timestamp": span1_from, "to_timestamp": span1_to, "message_count": 2},
        "created_at": span1_to,
    }
    payload2 = {
        "messages": [
            {"role": "user", "content": "beta details", "timestamp": span2_from, "metadata": {"message_id": "m3"}},
            {"role": "assistant", "content": "beta response", "timestamp": span2_to, "metadata": {"message_id": "m4"}},
        ],
        "span": {"from_timestamp": span2_from, "to_timestamp": span2_to, "message_count": 2},
        "created_at": span2_to,
    }

    meta1 = artifact_store.store_json(payload1, run_id=run_id, tags={"kind": "conversation_span"})
    meta2 = artifact_store.store_json(payload2, run_id=run_id, tags={"kind": "conversation_span"})

    messages = [
        {
            "role": "system",
            "content": "Alpha summary: decided to proceed with approach A.",
            "timestamp": "2025-01-01T00:10:01+00:00",
            "metadata": {"kind": "memory_summary", "source_artifact_id": meta1.artifact_id, "message_id": "s1"},
        },
        {
            "role": "system",
            "content": "Beta summary: chose approach B.",
            "timestamp": "2025-01-01T01:10:01+00:00",
            "metadata": {"kind": "memory_summary", "source_artifact_id": meta2.artifact_id, "message_id": "s2"},
        },
    ]

    spans = [
        {
            "kind": "conversation_span",
            "artifact_id": meta1.artifact_id,
            "created_at": "2025-01-01T00:10:02+00:00",
            "from_timestamp": span1_from,
            "to_timestamp": span1_to,
            "message_count": 2,
            "compression_mode": "standard",
            "focus": "alpha",
            "tags": {"topic": "alpha"},
        },
        {
            "kind": "conversation_span",
            "artifact_id": meta2.artifact_id,
            "created_at": "2025-01-01T01:10:02+00:00",
            "from_timestamp": span2_from,
            "to_timestamp": span2_to,
            "message_count": 2,
            "compression_mode": "standard",
            "focus": "beta",
            "tags": {"topic": "beta"},
        },
    ]

    run_store.save(_run(run_id=run_id, messages=messages, memory_spans=spans))

    # Tag filter (artifact tags)
    by_tag = policy.filter_spans(run_id, tags={"topic": "alpha"})
    assert [s.get("artifact_id") for s in by_tag] == [meta1.artifact_id]

    # Time-range filter (intersects span1 only)
    by_time = policy.filter_spans(
        run_id,
        time_range=TimeRange(
            start="2025-01-01T00:05:00+00:00",
            end="2025-01-01T00:20:00+00:00",
        ),
    )
    assert [s.get("artifact_id") for s in by_time] == [meta1.artifact_id]

    # Query filter (matches summary text)
    by_query = policy.filter_spans(run_id, query="approach a")
    assert [s.get("artifact_id") for s in by_query] == [meta1.artifact_id]


def test_rehydrate_into_context_persists_and_dedups(tmp_path) -> None:
    run_id = "run_rehydrate"
    run_store = JsonFileRunStore(tmp_path / "runs")
    artifact_store = FileArtifactStore(tmp_path / "store")
    policy = ActiveContextPolicy(run_store=run_store, artifact_store=artifact_store)

    ts1 = "2025-01-01T00:00:00+00:00"
    ts2 = "2025-01-01T00:01:00+00:00"
    archived_messages = [
        {"role": "user", "content": "old question", "timestamp": ts1, "metadata": {"message_id": "old_1"}},
        {"role": "assistant", "content": "old answer", "timestamp": ts2, "metadata": {"message_id": "old_2"}},
    ]
    payload = {
        "messages": archived_messages,
        "span": {"from_timestamp": ts1, "to_timestamp": ts2, "message_count": 2},
        "created_at": "2025-01-01T00:01:01+00:00",
    }
    meta = artifact_store.store_json(payload, run_id=run_id, tags={"kind": "conversation_span", "topic": "rehydrate"})

    summary = {
        "role": "system",
        "content": "Summary: prior discussion archived.",
        "timestamp": "2025-01-01T00:02:00+00:00",
        "metadata": {"kind": "memory_summary", "source_artifact_id": meta.artifact_id, "message_id": "sum_1"},
    }
    recent = {
        "role": "user",
        "content": "recent question",
        "timestamp": "2025-01-01T00:03:00+00:00",
        "metadata": {"message_id": "recent_1"},
    }
    run = _run(
        run_id=run_id,
        messages=[summary, recent],
        memory_spans=[
            {
                "kind": "conversation_span",
                "artifact_id": meta.artifact_id,
                "created_at": "2025-01-01T00:01:02+00:00",
                "from_timestamp": ts1,
                "to_timestamp": ts2,
                "message_count": 2,
            }
        ],
    )
    run_store.save(run)

    res1 = policy.rehydrate_into_context(run_id, span_ids=[meta.artifact_id], placement="after_summary")
    assert res1["inserted"] == 2
    assert res1["skipped"] == 0

    loaded1 = run_store.load(run_id)
    assert loaded1 is not None
    msgs1 = loaded1.vars.get("context", {}).get("messages", [])
    assert isinstance(msgs1, list)
    assert [m.get("metadata", {}).get("message_id") for m in msgs1] == ["sum_1", "old_1", "old_2", "recent_1"]

    # Second hydration should dedup by message_id
    res2 = policy.rehydrate_into_context(run_id, span_ids=[meta.artifact_id], placement="after_summary")
    assert res2["inserted"] == 0
    assert res2["skipped"] == 2

    loaded2 = run_store.load(run_id)
    assert loaded2 is not None
    msgs2 = loaded2.vars.get("context", {}).get("messages", [])
    assert isinstance(msgs2, list)
    assert [m.get("metadata", {}).get("message_id") for m in msgs2] == ["sum_1", "old_1", "old_2", "recent_1"]


def test_select_active_messages_for_llm_from_run_applies_history_limit() -> None:
    run_id = "run_select"
    system = {"role": "system", "content": "sys", "timestamp": "2025-01-01T00:00:00+00:00", "metadata": {"message_id": "sys_1"}}
    convo = [
        {"role": "user", "content": f"u{i}", "timestamp": "2025-01-01T00:00:00+00:00", "metadata": {"message_id": f"u{i}"}}
        for i in range(5)
    ]
    run = _run(
        run_id=run_id,
        messages=[system, *convo],
        memory_spans=[],
    )
    run.vars["_limits"]["max_history_messages"] = 2

    selected = ActiveContextPolicy.select_active_messages_for_llm_from_run(run)
    assert [m.get("metadata", {}).get("message_id") for m in selected] == ["sys_1", "u3", "u4"]
