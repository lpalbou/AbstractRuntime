from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_memory_query_return_both_includes_meta_matches_and_span_ids() -> None:
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
                payload={"note": "Remember: Alice owns the API.", "tags": {"topic": "api", "person": "alice"}},
                result_key="_temp.note",
            ),
            next_node="query",
        )

    def query_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="query",
            effect=Effect(
                type=EffectType.MEMORY_QUERY,
                payload={"query": "Alice", "return": "both", "limit_spans": 10},
                result_key="_temp.recall",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"note": run.vars["_temp"]["note"], "recall": run.vars["_temp"]["recall"]})

    wf = WorkflowSpec(workflow_id="wf_scope_meta", entry_node="note", nodes={"note": note_node, "query": query_node, "done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.status == RunStatus.COMPLETED

    recall = (state.output or {}).get("recall")
    assert isinstance(recall, dict)
    results = recall.get("results")
    assert isinstance(results, list) and results
    first = results[0]
    assert isinstance(first, dict)
    assert isinstance(first.get("output"), str) and "Alice owns the API" in str(first.get("output"))
    meta = first.get("meta")
    assert isinstance(meta, dict)
    matches = meta.get("matches")
    assert isinstance(matches, list) and matches
    assert isinstance(meta.get("span_ids"), list) and meta["span_ids"]


def test_memory_note_scope_session_routes_to_root_run_and_keeps_source_run_id() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    # Create a root run (session authority) and a child run that will write into scope=session.
    root_run_id = "run_root"
    run_store.save(
        RunState(
            run_id=root_run_id,
            workflow_id="wf_root",
            status=RunStatus.COMPLETED,
            current_node="done",
            vars={
                "context": {"task": "root", "messages": []},
                "scratchpad": {},
                "_runtime": {"memory_spans": []},
                "_temp": {},
                "_limits": {},
            },
            waiting=None,
            output={"messages": []},
            error=None,
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
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
                payload={"note": "Session note", "tags": {"topic": "t"}, "scope": "session"},
                result_key="_temp.note",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"note": run.vars.get("_temp", {}).get("note")})

    wf = WorkflowSpec(workflow_id="wf_child_session_note", entry_node="note", nodes={"note": note_node, "done": done_node})
    child_run_id = runtime.start(workflow=wf, vars=vars, parent_run_id=root_run_id)
    state = runtime.tick(workflow=wf, run_id=child_run_id)
    assert state.status == RunStatus.COMPLETED

    # The note should be indexed in the root run, not the child run.
    root = run_store.load(root_run_id)
    assert root is not None
    spans = root.vars.get("_runtime", {}).get("memory_spans")
    assert isinstance(spans, list) and spans
    span = spans[0]
    assert span.get("kind") == "memory_note"
    assert span.get("tags") == {"topic": "t"}
    artifact_id = span.get("artifact_id")
    assert isinstance(artifact_id, str) and artifact_id

    payload = artifact_store.load_json(artifact_id)
    assert isinstance(payload, dict)
    sources = payload.get("sources")
    assert isinstance(sources, dict)
    # Default provenance should point at the child run that emitted the effect.
    assert sources.get("run_id") == child_run_id


def test_memory_note_scope_global_routes_to_global_memory_run() -> None:
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
                payload={"note": "Global note", "tags": {"topic": "global"}, "scope": "global"},
                result_key="_temp.note",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"note": run.vars.get("_temp", {}).get("note")})

    wf = WorkflowSpec(workflow_id="wf_global_note", entry_node="note", nodes={"note": note_node, "done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.COMPLETED

    global_run = run_store.load("global_memory")
    assert global_run is not None
    spans = global_run.vars.get("_runtime", {}).get("memory_spans")
    assert isinstance(spans, list) and spans
    assert spans[0].get("kind") == "memory_note"


def test_memory_note_keep_in_context_inserts_synthetic_message() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    vars: dict[str, Any] = {
        "context": {"task": "t", "messages": [{"role": "system", "content": "sys", "timestamp": "2025-01-01T00:00:00+00:00"}]},
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
                payload={"note": "i am laurent-philippe", "tags": {"who": "user"}, "keep_in_context": True},
                result_key="_temp.note",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"note": run.vars.get("_temp", {}).get("note"), "messages": run.vars.get("context", {}).get("messages")})

    wf = WorkflowSpec(workflow_id="wf_keep_in_context", entry_node="note", nodes={"note": note_node, "done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.COMPLETED

    messages = (state.output or {}).get("messages")
    assert isinstance(messages, list)
    assert any(isinstance(m, dict) and str(m.get("content") or "").startswith("[MEMORY NOTE]") for m in messages)
    assert any(isinstance(m, dict) and "laurent-philippe" in str(m.get("content") or "") for m in messages)

    note_result = (state.output or {}).get("note")
    assert isinstance(note_result, dict)
    results = note_result.get("results")
    assert isinstance(results, list) and results
    meta = results[0].get("meta")
    assert isinstance(meta, dict)
    kept = meta.get("kept_in_context")
    assert isinstance(kept, dict)
    assert kept.get("inserted") == 1


def test_memory_rehydrate_inserts_conversation_span_and_includes_memory_note() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    # Create a conversation span artifact.
    span_payload = {
        "messages": [
            {"role": "user", "content": "u1", "timestamp": "2025-01-01T00:00:00+00:00", "metadata": {"message_id": "m1"}},
            {"role": "assistant", "content": "a1", "timestamp": "2025-01-01T00:01:00+00:00", "metadata": {"message_id": "m2"}},
        ],
        "span": {"from_timestamp": "2025-01-01T00:00:00+00:00", "to_timestamp": "2025-01-01T00:01:00+00:00", "message_count": 2},
        "created_at": "2025-01-01T00:01:01+00:00",
    }
    span_meta = artifact_store.store_json(span_payload, run_id="run", tags={"kind": "conversation_span"})

    # Create a memory_note artifact (rehydratable as a synthetic system message).
    note_payload = {"note": "note", "sources": {"run_id": "run", "span_ids": [], "message_ids": []}, "created_at": "2025-01-01T00:02:00+00:00"}
    note_meta = artifact_store.store_json(note_payload, run_id="run", tags={"kind": "memory_note"})

    vars: dict[str, Any] = {
        "context": {"task": "t", "messages": [{"role": "system", "content": "sys", "timestamp": "2025-01-01T00:00:00+00:00", "metadata": {"message_id": "s0"}}]},
        "scratchpad": {},
        "_runtime": {
            "memory_spans": [
                {
                    "kind": "memory_note",
                    "artifact_id": note_meta.artifact_id,
                    "created_at": "2025-01-01T00:02:00+00:00",
                    "from_timestamp": "2025-01-01T00:02:00+00:00",
                    "to_timestamp": "2025-01-01T00:02:00+00:00",
                    "message_count": 0,
                },
                {
                    "kind": "conversation_span",
                    "artifact_id": span_meta.artifact_id,
                    "created_at": "2025-01-01T00:01:01+00:00",
                    "from_timestamp": "2025-01-01T00:00:00+00:00",
                    "to_timestamp": "2025-01-01T00:01:00+00:00",
                    "message_count": 2,
                },
            ]
        },
        "_temp": {},
        "_limits": {},
    }

    def rehydrate_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="rehydrate",
            effect=Effect(
                type=EffectType.MEMORY_REHYDRATE,
                payload={"span_ids": [note_meta.artifact_id, span_meta.artifact_id], "placement": "after_system"},
                result_key="_temp.rehydrate",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"rehydrate": run.vars.get("_temp", {}).get("rehydrate"), "messages": run.vars.get("context", {}).get("messages")},
        )

    wf = WorkflowSpec(workflow_id="wf_rehydrate", entry_node="rehydrate", nodes={"rehydrate": rehydrate_node, "done": done_node})
    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.COMPLETED

    out = state.output or {}
    rehydrate = out.get("rehydrate")
    assert isinstance(rehydrate, dict)
    # 1 synthetic note message + 2 archived convo messages
    assert rehydrate.get("inserted") == 3

    msgs = out.get("messages")
    assert isinstance(msgs, list)
    ids = [m.get("metadata", {}).get("message_id") for m in msgs if isinstance(m, dict)]
    assert "memory_note:" + note_meta.artifact_id in ids

    artifacts = rehydrate.get("artifacts")
    assert isinstance(artifacts, list)
    kinds = {a.get("kind") for a in artifacts if isinstance(a, dict)}
    assert "memory_note" in kinds
    assert "conversation_span" in kinds

    messages = (state.output or {}).get("messages")
    assert isinstance(messages, list)
    # Inserted after the system message.
    assert [m.get("content") for m in messages if isinstance(m, dict)][0] == "sys"
    assert [m.get("content") for m in messages if isinstance(m, dict)][1] == "[MEMORY NOTE]\nnote"
    assert [m.get("content") for m in messages if isinstance(m, dict)][2:4] == ["u1", "a1"]


