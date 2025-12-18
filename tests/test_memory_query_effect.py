from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_memory_query_effect_deep_scan_and_connected_neighbors() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    # Store two archived spans.
    payload1 = {
        "messages": [
            {"role": "user", "content": "We talked to Alice about the API.", "timestamp": "2025-01-01T00:00:00+00:00", "metadata": {"message_id": "m1"}},
            {"role": "assistant", "content": "Noted. Alice owns the API contract.", "timestamp": "2025-01-01T00:01:00+00:00", "metadata": {"message_id": "m2"}},
        ],
        "span": {"from_timestamp": "2025-01-01T00:00:00+00:00", "to_timestamp": "2025-01-01T00:01:00+00:00", "message_count": 2},
        "created_at": "2025-01-01T00:01:01+00:00",
    }
    payload2 = {
        "messages": [
            {"role": "user", "content": "Follow-up: we chose approach B.", "timestamp": "2025-01-01T00:02:00+00:00", "metadata": {"message_id": "m3"}},
            {"role": "assistant", "content": "Okay.", "timestamp": "2025-01-01T00:03:00+00:00", "metadata": {"message_id": "m4"}},
        ],
        "span": {"from_timestamp": "2025-01-01T00:02:00+00:00", "to_timestamp": "2025-01-01T00:03:00+00:00", "message_count": 2},
        "created_at": "2025-01-01T00:03:01+00:00",
    }

    meta1 = artifact_store.store_json(payload1, run_id="run", tags={"kind": "conversation_span"})
    meta2 = artifact_store.store_json(payload2, run_id="run", tags={"kind": "conversation_span"})

    # Seed a run with memory span index + summaries (summaries do NOT contain "Alice" to force deep scan).
    vars: dict[str, Any] = {
        "context": {
            "task": "test",
            "messages": [
                {
                    "role": "system",
                    "content": f"[SUMMARY span_id={meta1.artifact_id}]: discussed ownership and next steps.",
                    "timestamp": "2025-01-01T00:01:02+00:00",
                    "metadata": {"kind": "memory_summary", "source_artifact_id": meta1.artifact_id, "message_id": "s1"},
                },
                {
                    "role": "system",
                    "content": f"[SUMMARY span_id={meta2.artifact_id}]: selected an approach.",
                    "timestamp": "2025-01-01T00:03:02+00:00",
                    "metadata": {"kind": "memory_summary", "source_artifact_id": meta2.artifact_id, "message_id": "s2"},
                },
            ],
        },
        "scratchpad": {},
        "_runtime": {
            "memory_spans": [
                {
                    "kind": "conversation_span",
                    "artifact_id": meta1.artifact_id,
                    "created_at": "2025-01-01T00:01:01+00:00",
                    "from_timestamp": "2025-01-01T00:00:00+00:00",
                    "to_timestamp": "2025-01-01T00:01:00+00:00",
                    "message_count": 2,
                    "tags": {"topic": "api"},
                },
                {
                    "kind": "conversation_span",
                    "artifact_id": meta2.artifact_id,
                    "created_at": "2025-01-01T00:03:01+00:00",
                    "from_timestamp": "2025-01-01T00:02:00+00:00",
                    "to_timestamp": "2025-01-01T00:03:00+00:00",
                    "message_count": 2,
                    "tags": {"topic": "api"},
                },
            ]
        },
        "_temp": {},
        "_limits": {},
    }

    def recall_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="recall",
            effect=Effect(
                type=EffectType.MEMORY_QUERY,
                payload={"query": "Alice", "connected": True, "neighbor_hops": 1, "limit_spans": 5, "max_messages": 20},
                result_key="_temp.recall",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"recall": run.vars.get("_temp", {}).get("recall")})

    wf = WorkflowSpec(workflow_id="wf_memory_query", entry_node="recall", nodes={"recall": recall_node, "done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    recall = (state.output or {}).get("recall")
    assert isinstance(recall, dict)
    results = recall.get("results")
    assert isinstance(results, list) and results
    output_text = str(results[0].get("output") or "")

    # Deep scan should find the first span even though the summary does not mention Alice.
    assert f"span_id={meta1.artifact_id}" in output_text
    assert "Alice" in output_text

    # Connected expansion should include the adjacent second span.
    assert f"span_id={meta2.artifact_id}" in output_text
