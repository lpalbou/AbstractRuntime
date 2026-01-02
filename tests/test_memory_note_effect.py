from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.memory import ActiveContextPolicy, TimeRange
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_memory_note_effect_stores_artifact_indexes_span_and_is_queryable() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    vars: dict[str, Any] = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {},
        "_runtime": {"memory_spans": []},
        "_temp": {},
        "_limits": {},
    }

    def note_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="note",
            effect=Effect(
                type=EffectType.MEMORY_NOTE,
                payload={
                    "note": "Remember: Alice owns the API contract.",
                    "tags": {"topic": "api", "person": "alice"},
                    "sources": {"run_id": run.run_id, "message_ids": ["m1"], "span_ids": ["span_x"]},
                },
                result_key="_temp.note",
            ),
            next_node="query",
        )

    def query_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="query",
            effect=Effect(
                type=EffectType.MEMORY_QUERY,
                payload={"query": "Alice", "tags": {"topic": "api"}, "limit_spans": 10, "max_messages": 50},
                result_key="_temp.recall",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={
                "note": run.vars.get("_temp", {}).get("note"),
                "recall": run.vars.get("_temp", {}).get("recall"),
                "spans": run.vars.get("_runtime", {}).get("memory_spans"),
            },
        )

    wf = WorkflowSpec(workflow_id="wf_memory_note", entry_node="note", nodes={"note": note_node, "query": query_node, "done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.status == RunStatus.COMPLETED

    note_out = (state.output or {}).get("note")
    assert isinstance(note_out, dict)
    note_results = note_out.get("results")
    assert isinstance(note_results, list) and note_results
    note_meta = note_results[0].get("meta") if isinstance(note_results[0], dict) else None
    assert isinstance(note_meta, dict)
    assert isinstance(note_meta.get("note_preview"), str)
    assert "Alice owns the API contract" in str(note_meta.get("note_preview"))

    spans = (state.output or {}).get("spans")
    assert isinstance(spans, list) and spans
    note_span = spans[0]
    assert note_span.get("kind") == "memory_note"
    assert note_span.get("tags") == {"topic": "api", "person": "alice"}
    assert isinstance(note_span.get("artifact_id"), str) and note_span.get("artifact_id")

    artifact_id = note_span["artifact_id"]
    payload = artifact_store.load_json(artifact_id)
    assert isinstance(payload, dict)
    assert payload.get("note") == "Remember: Alice owns the API contract."
    assert isinstance(payload.get("sources"), dict)

    recall = (state.output or {}).get("recall")
    assert isinstance(recall, dict)
    results = recall.get("results")
    assert isinstance(results, list) and results
    output_text = str(results[0].get("output") or "")
    assert f"span_id={artifact_id}" in output_text
    assert "kind=memory_note" in output_text
    assert "Alice owns the API contract" in output_text


def test_memory_note_time_range_filter_matches_point_notes() -> None:
    run_id = "run_time_range"
    run_store = InMemoryRunStore()
    artifact_store = InMemoryArtifactStore()
    policy = ActiveContextPolicy(run_store=run_store, artifact_store=artifact_store)

    created_at = "2025-01-01T00:00:00+00:00"
    meta = artifact_store.store_json(
        {"note": "n", "sources": {"run_id": run_id, "span_ids": [], "message_ids": []}, "created_at": created_at},
        run_id=run_id,
        tags={"kind": "memory_note", "topic": "t"},
    )

    run_store.save(
        RunState(
            run_id=run_id,
            workflow_id="wf",
            status=RunStatus.COMPLETED,
            current_node="done",
            vars={
                "context": {"task": "t", "messages": []},
                "scratchpad": {},
                "_runtime": {
                    "memory_spans": [
                        {
                            "kind": "memory_note",
                            "artifact_id": meta.artifact_id,
                            "created_at": created_at,
                            "from_timestamp": created_at,
                            "to_timestamp": created_at,
                            "message_count": 0,
                            "tags": {"topic": "t"},
                        }
                    ]
                },
                "_temp": {},
                "_limits": {},
            },
            waiting=None,
            output={"messages": []},
            error=None,
            created_at=created_at,
            updated_at=created_at,
            actor_id=None,
            session_id=None,
            parent_run_id=None,
        )
    )

    matches = policy.filter_spans(
        run_id,
        time_range=TimeRange(start="2025-01-01T00:00:00+00:00", end="2025-01-01T00:00:00+00:00"),
        tags={"topic": "t"},
    )
    assert [m.get("artifact_id") for m in matches] == [meta.artifact_id]


def test_memory_note_effect_can_target_other_run() -> None:
    """A child run can store a note into a different target run via payload.target_run_id."""
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    target_run_id = "run_target"
    created_at = "2025-01-01T00:00:00+00:00"
    run_store.save(
        RunState(
            run_id=target_run_id,
            workflow_id="wf_target",
            status=RunStatus.COMPLETED,
            current_node="done",
            vars={
                "context": {"task": "t", "messages": []},
                "scratchpad": {},
                "_runtime": {"memory_spans": []},
                "_temp": {},
                "_limits": {},
            },
            waiting=None,
            output={"messages": []},
            error=None,
            created_at=created_at,
            updated_at=created_at,
            actor_id=None,
            session_id=None,
            parent_run_id=None,
        )
    )

    vars: dict[str, Any] = {
        "context": {"task": "child", "messages": []},
        "scratchpad": {},
        "_runtime": {"memory_spans": []},
        "_temp": {},
        "_limits": {},
    }

    def note_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="note",
            effect=Effect(
                type=EffectType.MEMORY_NOTE,
                payload={
                    "target_run_id": target_run_id,
                    "note": "Targeted note",
                    "tags": {"topic": "t"},
                },
                result_key="_temp.note",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"note": run.vars.get("_temp", {}).get("note")})

    wf = WorkflowSpec(workflow_id="wf_child_note", entry_node="note", nodes={"note": note_node, "done": done_node})
    child_run_id = runtime.start(workflow=wf, vars=vars, parent_run_id=target_run_id)
    state = runtime.tick(workflow=wf, run_id=child_run_id)
    assert state.status == RunStatus.COMPLETED

    updated_target = run_store.load(target_run_id)
    assert updated_target is not None
    spans = updated_target.vars.get("_runtime", {}).get("memory_spans")
    assert isinstance(spans, list) and spans
    note_span = spans[0]
    assert note_span.get("kind") == "memory_note"
    assert note_span.get("tags") == {"topic": "t"}
    artifact_id = note_span.get("artifact_id")
    assert isinstance(artifact_id, str) and artifact_id

    payload = artifact_store.load_json(artifact_id)
    assert isinstance(payload, dict)
    assert payload.get("note") == "Targeted note"
