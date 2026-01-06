from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _base_vars() -> dict[str, Any]:
    return {
        "context": {"task": "t", "messages": []},
        "scratchpad": {},
        "_runtime": {"memory_spans": []},
        "_temp": {},
        "_limits": {},
    }


def test_memory_query_tags_mode_any_matches_either_tag_key() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    vars = _base_vars()

    def note1(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="note1",
            effect=Effect(
                type=EffectType.MEMORY_NOTE,
                payload={"note": "n1", "tags": {"topic": "ai", "person": "alice"}, "location": "paris"},
                result_key="_temp.note1",
            ),
            next_node="note2",
        )

    def note2(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="note2",
            effect=Effect(
                type=EffectType.MEMORY_NOTE,
                payload={"note": "n2", "tags": {"topic": "cook", "person": "bob"}, "location": "nyc"},
                result_key="_temp.note2",
            ),
            next_node="query_any",
        )

    def query_any(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="query_any",
            effect=Effect(
                type=EffectType.MEMORY_QUERY,
                payload={"tags": {"topic": "ai", "person": "bob"}, "tags_mode": "any", "return": "meta", "limit_spans": 10},
                result_key="_temp.recall_any",
            ),
            next_node="query_all",
        )

    def query_all(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="query_all",
            effect=Effect(
                type=EffectType.MEMORY_QUERY,
                payload={"tags": {"topic": "ai", "person": "bob"}, "tags_mode": "all", "return": "meta", "limit_spans": 10},
                result_key="_temp.recall_all",
            ),
            next_node="done",
        )

    def done(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"any": run.vars["_temp"]["recall_any"], "all": run.vars["_temp"]["recall_all"]},
        )

    wf = WorkflowSpec(
        workflow_id="wf_query_tags_mode",
        entry_node="note1",
        nodes={"note1": note1, "note2": note2, "query_any": query_any, "query_all": query_all, "done": done},
    )

    run_id = runtime.start(workflow=wf, vars=vars, actor_id="todel")
    state = runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
    assert state.status == RunStatus.COMPLETED

    any_out = (state.output or {}).get("any")
    assert isinstance(any_out, dict)
    any_results = any_out.get("results")
    assert isinstance(any_results, list) and any_results
    any_meta = any_results[0].get("meta") if isinstance(any_results[0], dict) else None
    assert isinstance(any_meta, dict)
    matches_any = any_meta.get("matches")
    assert isinstance(matches_any, list)
    assert len(matches_any) == 2  # topic=ai OR person=bob

    all_out = (state.output or {}).get("all")
    assert isinstance(all_out, dict)
    all_results = all_out.get("results")
    assert isinstance(all_results, list) and all_results
    all_meta = all_results[0].get("meta") if isinstance(all_results[0], dict) else None
    assert isinstance(all_meta, dict)
    matches_all = all_meta.get("matches")
    assert isinstance(matches_all, list)
    assert len(matches_all) == 0  # requires topic=ai AND person=bob (no single note has both)


def test_memory_query_multi_value_tag_key_and_location_filter() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    vars = _base_vars()

    def note1(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="note1",
            effect=Effect(
                type=EffectType.MEMORY_NOTE,
                payload={"note": "Alice note", "tags": {"topic": "ai", "person": "alice"}, "location": "Paris"},
                result_key="_temp.note1",
            ),
            next_node="note2",
        )

    def note2(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="note2",
            effect=Effect(
                type=EffectType.MEMORY_NOTE,
                payload={"note": "Bob note", "tags": {"topic": "ai", "person": "bob"}, "location": "NYC"},
                result_key="_temp.note2",
            ),
            next_node="query",
        )

    def query(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="query",
            effect=Effect(
                type=EffectType.MEMORY_QUERY,
                payload={
                    "tags": {"topic": "ai", "person": ["alice", "bob"]},
                    "tags_mode": "all",
                    "locations": ["paris"],
                    "return": "meta",
                    "limit_spans": 10,
                },
                result_key="_temp.recall",
            ),
            next_node="done",
        )

    def done(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"recall": run.vars["_temp"]["recall"]})

    wf = WorkflowSpec(workflow_id="wf_query_multivalue_location", entry_node="note1", nodes={"note1": note1, "note2": note2, "query": query, "done": done})
    run_id = runtime.start(workflow=wf, vars=vars, actor_id="todel")
    state = runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
    assert state.status == RunStatus.COMPLETED

    recall = (state.output or {}).get("recall")
    assert isinstance(recall, dict)
    results = recall.get("results")
    assert isinstance(results, list) and results
    meta = results[0].get("meta") if isinstance(results[0], dict) else None
    assert isinstance(meta, dict)
    matches = meta.get("matches")
    assert isinstance(matches, list)
    assert len(matches) == 1
    first = matches[0]
    assert isinstance(first, dict)
    assert first.get("location") == "Paris"


def test_memory_query_authors_filter_over_global_scope() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    def wf_note(note: str) -> WorkflowSpec:
        def note_node(run, ctx) -> StepPlan:
            return StepPlan(
                node_id="note",
                effect=Effect(
                    type=EffectType.MEMORY_NOTE,
                    payload={"note": note, "tags": {"topic": "ai"}, "scope": "global"},
                    result_key="_temp.note",
                ),
                next_node="done",
            )

        def done_node(run, ctx) -> StepPlan:
            return StepPlan(node_id="done", complete_output={"note": run.vars.get("_temp", {}).get("note")})

        return WorkflowSpec(workflow_id="wf_note", entry_node="note", nodes={"note": note_node, "done": done_node})

    # Two different actors write into global scope.
    wf1 = wf_note("Alice opinion")
    r1 = runtime.start(workflow=wf1, vars=_base_vars(), actor_id="alice")
    s1 = runtime.tick(workflow=wf1, run_id=r1)
    assert s1.status == RunStatus.COMPLETED

    wf2 = wf_note("Bob opinion")
    r2 = runtime.start(workflow=wf2, vars=_base_vars(), actor_id="bob")
    s2 = runtime.tick(workflow=wf2, run_id=r2)
    assert s2.status == RunStatus.COMPLETED

    def query_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="query",
            effect=Effect(
                type=EffectType.MEMORY_QUERY,
                payload={"scope": "global", "tags": {"topic": "ai"}, "authors": ["alice"], "return": "meta", "limit_spans": 10},
                result_key="_temp.recall",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"recall": run.vars.get("_temp", {}).get("recall")})

    wfq = WorkflowSpec(workflow_id="wf_query", entry_node="query", nodes={"query": query_node, "done": done_node})
    rq = runtime.start(workflow=wfq, vars=_base_vars(), actor_id="todel")
    sq = runtime.tick(workflow=wfq, run_id=rq)
    assert sq.status == RunStatus.COMPLETED

    recall = (sq.output or {}).get("recall")
    assert isinstance(recall, dict)
    results = recall.get("results")
    assert isinstance(results, list) and results
    meta = results[0].get("meta") if isinstance(results[0], dict) else None
    assert isinstance(meta, dict)
    matches = meta.get("matches")
    assert isinstance(matches, list)
    assert len(matches) == 1
    first = matches[0]
    assert isinstance(first, dict)
    assert str(first.get("created_by") or "").lower() == "alice"




