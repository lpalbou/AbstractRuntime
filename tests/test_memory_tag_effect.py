from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.memory.active_context import ActiveContextPolicy
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_memory_tag_effect_merges_tags_and_enables_filtering() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    vars: dict[str, Any] = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {},
        "_runtime": {
            "memory_spans": [
                {
                    "kind": "conversation_span",
                    "artifact_id": "a1",
                    "created_at": "2025-01-01T00:00:10+00:00",
                    "from_timestamp": "2025-01-01T00:00:00+00:00",
                    "to_timestamp": "2025-01-01T00:00:00+00:00",
                    "message_count": 1,
                    "tags": {"topic": "api"},
                },
                {
                    "kind": "conversation_span",
                    "artifact_id": "a2",
                    "created_at": "2025-01-01T00:01:10+00:00",
                    "from_timestamp": "2025-01-01T00:01:00+00:00",
                    "to_timestamp": "2025-01-01T00:01:00+00:00",
                    "message_count": 1,
                },
            ]
        },
        "_temp": {},
        "_limits": {},
    }

    def tag_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="tag",
            effect=Effect(
                type=EffectType.MEMORY_TAG,
                payload={"span_id": "a1", "tags": {"person": "Alice"}},
                result_key="_temp.tag",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"spans": run.vars.get("_runtime", {}).get("memory_spans")})

    wf = WorkflowSpec(workflow_id="wf_memory_tag", entry_node="tag", nodes={"tag": tag_node, "done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    spans = (state.output or {}).get("spans")
    assert isinstance(spans, list)
    assert spans[0].get("artifact_id") == "a1"
    assert spans[0].get("tags") == {"topic": "api", "person": "Alice"}

    # Tag-based filtering should work without relying on artifact metadata.
    filtered = ActiveContextPolicy.filter_spans_from_run(
        state,
        artifact_store=artifact_store,
        tags={"person": "Alice"},
        limit=10,
    )
    assert filtered and filtered[0].get("artifact_id") == "a1"


def test_memory_tag_effect_overwrites_tags_by_index() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

    vars: dict[str, Any] = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {},
        "_runtime": {
            "memory_spans": [
                {"kind": "conversation_span", "artifact_id": "a1", "created_at": "2025-01-01T00:00:10+00:00"},
                {"kind": "conversation_span", "artifact_id": "a2", "created_at": "2025-01-01T00:01:10+00:00", "tags": {"topic": "old"}},
            ]
        },
        "_temp": {},
        "_limits": {},
    }

    def tag_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="tag",
            effect=Effect(
                type=EffectType.MEMORY_TAG,
                payload={"span_id": 2, "tags": {"topic": "runtime"}, "merge": False},
                result_key="_temp.tag",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"spans": run.vars.get("_runtime", {}).get("memory_spans")})

    wf = WorkflowSpec(workflow_id="wf_memory_tag_overwrite", entry_node="tag", nodes={"tag": tag_node, "done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.status == RunStatus.COMPLETED
    spans = (state.output or {}).get("spans")
    assert isinstance(spans, list)
    assert spans[1].get("artifact_id") == "a2"
    assert spans[1].get("tags") == {"topic": "runtime"}

