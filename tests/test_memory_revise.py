"""MEMORY_REVISE / the `memory_revise` VisualFlow node: memory capture and reconsideration as a workflow step with
ledger provenance (ADR 0009; AbstractMemory ADR 0003).

A REAL `Runtime` executes a compiled VisualFlow on real temporary stores (SqliteRunStore + SqliteLedgerStore, the
SQLite graph home); no mocks.

Mutants this file kills (each checked by hand when it was written):
- the handler skips `facade.stamp_current_step` (no step ref) -> the workflow case fails (and the revision is refused
  as ungrounded);
- the handler keeps a caller-supplied `step` ref -> the runtime-owned case fails;
- the handler lets `ProvenanceRequired` raise (or invents a ref to pass it) -> the outside-any-run case fails;
- the builder opens a graph over a non-graph store, or answers it silently -> the LanceDB case fails;
- `memory_revise` missing from the compiler / executor -> the workflow case fails at compile or dispatch.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractmemory import MemoryFacade, build_memory_kg_effect_handlers
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore
from abstractruntime.visualflow_compiler import compile_visualflow

pytestmark = pytest.mark.basic

pytest.importorskip("abstractmemory.graph")

from abstractmemory import SQLiteTripleStore  # noqa: E402

NOW = "2026-10-27T00:00:00+00:00"
OPS: List[Dict[str, Any]] = [
    {"op": "assert", "payload": {"subject": "ex:alice", "predicate": "schema:knows", "object": "ex:bob"}},
    {"op": "assert", "payload": {"subject": "ex:alice", "predicate": "schema:name", "object": "Alice"}},
]


def _flow(pin_defaults: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": "vf_memory_revise",
        "name": "vf_memory_revise",
        "entryNode": "start",
        "nodes": [
            {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
            {"id": "rev", "type": "memory_revise",
             "data": {"nodeType": "memory_revise", "pinDefaults": dict(pin_defaults)}},
        ],
        "edges": [{"source": "start", "sourceHandle": "exec-out", "target": "rev", "targetHandle": "exec-in"}],
    }


class _Home:
    def __init__(self, tmp_path: Path) -> None:
        db = SqliteDatabase(tmp_path / "runtime.sqlite3")
        self.run_store, self.ledger = SqliteRunStore(db), SqliteLedgerStore(db)
        self.artifacts = FileArtifactStore(tmp_path / "artifacts")
        self.mind = tmp_path / "mind.sqlite3"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.store = SQLiteTripleStore(self.mind)
        self.handlers = build_memory_kg_effect_handlers(store=self.store, run_store=self.run_store,
                                                        now_iso=lambda: NOW, artifact_store=self.artifacts)
        # A second connection on the same home: what a host reading the mind sees.
        self.facade = MemoryFacade.open_mind(self.mind, ledger_store=self.ledger, run_store=self.run_store,
                                             artifact_store=self.artifacts)

    def run_flow(self, pin_defaults: Dict[str, Any]) -> tuple:
        spec = compile_visualflow(_flow(pin_defaults))
        rt = Runtime(run_store=self.run_store, ledger_store=self.ledger, effect_handlers=self.handlers)
        run_id = rt.start(workflow=spec)
        state = rt.tick(workflow=spec, run_id=run_id)
        return run_id, state


def _effect_result(state: RunState) -> Dict[str, Any]:
    return state.vars["_temp"]["effects"]["rev"]


def test_a_memory_revise_step_writes_a_revision_grounded_in_its_own_ledger_row(tmp_path: Path) -> None:
    home = _Home(tmp_path)
    span = home.artifacts.store_text("Alice said she knows Bob.")
    run_id, state = home.run_flow({"ops": OPS, "scope": "global", "cause": "ingest", "summary": "met Bob",
                                   "artifact_ids": [span.artifact_id],
                                   "provenance": {"actor": "entity:e1", "reason": "she said so"}})
    assert state.status == RunStatus.COMPLETED, state.error
    out = _effect_result(state)
    assert out["ok"] is True and out["applied"] is True and len(out["assertion_ids"]) == 2
    assert out["owner_id"] == "global_memory" and out["scope"] == "global"

    started = [r for r in home.ledger.list(run_id) if r["status"] == "started" and r["effect"]["type"] == "memory_revise"]
    assert len(started) == 1
    step_ref = {"kind": "step", "run_id": run_id, "seq": started[0]["seq"], "step_id": started[0]["step_id"]}
    assert step_ref in out["sources"]
    assert {"kind": "artifact", "artifact_id": span.artifact_id, "run_id": run_id} in out["sources"]

    # The graph indexes the revision by its ledger row, both ways.
    assert sorted(home.facade.derived_from_source({"kind": "step", "run_id": run_id, "seq": step_ref["seq"]})) == sorted(
        out["assertion_ids"])
    for aid in out["assertion_ids"]:
        assert step_ref in home.facade.sources_of(aid)
        assert home.facade.history(aid)[0]["revision_id"] == out["revision_id"]
    (rev,), _ = home.facade.graph.revisions()
    assert rev.provenance.actor == "entity:e1" and rev.provenance.reason == "she said so" and rev.summary == "met Bob"

    # The source reads back verbatim: the MEMORY_REVISE STARTED row carries the revision payload as sent.
    back = home.facade.reconstruct_source(step_ref)
    assert back["status"] == "ok" and back["metadata"]["effect_type"] == "memory_revise"
    assert back["metadata"]["status"] == "started"
    assert back["content"]["effect"]["payload"]["ops"] == OPS
    assert home.facade.reconstruct_source(
        {"kind": "artifact", "artifact_id": span.artifact_id})["content"] == "Alice said she knows Bob."


def test_a_revision_derived_from_memory_is_grounded_through_it(tmp_path: Path) -> None:
    home = _Home(tmp_path)
    _, first = home.run_flow({"ops": OPS[:1], "scope": "global"})
    parent = _effect_result(first)["assertion_ids"][0]
    close = [{"op": "close", "payload": {"assertion_id": parent, "valid_until": "2026-10-28T00:00:00+00:00",
                                         "kind": "retract", "reason": "she was joking"}}]
    run_id, state = home.run_flow({"ops": close, "scope": "global", "cause": "reinterpret",
                                   "provenance": {"derived_from": [parent]}})
    out = _effect_result(state)
    assert state.status == RunStatus.COMPLETED, state.error
    assert out["closed"] == [parent] and out["applied"] is True
    assert [h["event"] for h in home.facade.history(parent)] == ["asserted", "closed"]
    assert any(s["kind"] == "step" and s["run_id"] == run_id for s in out["sources"])


def test_outside_any_run_an_ungrounded_revision_is_a_failed_outcome(tmp_path: Path) -> None:
    home = _Home(tmp_path)
    loose = RunState.new(workflow_id="wf", entry_node="n1")  # never stored, never executed by a runtime
    outcome = home.handlers[EffectType.MEMORY_REVISE](
        loose, Effect(type=EffectType.MEMORY_REVISE, payload={"ops": OPS, "scope": "global"}), None)
    assert outcome.status == "failed"
    assert "ProvenanceRequired" in outcome.error
    assert home.facade.graph.revisions()[0] == []  # nothing written, no ref fabricated


def test_run_and_step_refs_are_runtime_owned(tmp_path: Path) -> None:
    home = _Home(tmp_path)
    forged = {"ops": OPS, "scope": "global",
              "provenance": {"sources": [{"kind": "step", "run_id": "someone-else", "seq": 1}]}}
    _, state = home.run_flow(forged)
    assert state.status == RunStatus.FAILED
    assert "runtime-owned" in str(state.error)
    assert home.facade.graph.revisions()[0] == []


@pytest.mark.parametrize("kind", ["STEP", " step ", "Run"])
def test_run_and_step_refs_are_refused_whatever_their_spelling(tmp_path: Path, kind: str) -> None:
    """Review 3a R1: the kind is normalized by SourceRef before the runtime-owned check. Mutant: compare the raw
    kind string -> "STEP" slips through and the forged ledger address of another run is written."""
    home = _Home(tmp_path)
    ref = {"kind": kind, "run_id": "someone-else"}
    if kind.strip().lower() == "step":
        ref["seq"] = 1
    run = RunState.new(workflow_id="wf", entry_node="n1")
    home.run_store.save(run)
    outcome = home.handlers[EffectType.MEMORY_REVISE](run, Effect(type=EffectType.MEMORY_REVISE, payload={
        "ops": OPS, "scope": "global", "provenance": {"sources": [ref]}}), None)
    assert outcome.status == "failed" and "runtime-owned" in outcome.error
    assert home.facade.graph.revisions()[0] == []


def test_a_caller_ref_cannot_name_another_run(tmp_path: Path) -> None:
    """Review 3a R2. Mutant: drop the run_id check -> the flow's memories answer `derived_from_source` for the
    victim run."""
    home = _Home(tmp_path)
    span = home.artifacts.store_text("a real span")
    _, state = home.run_flow({"ops": OPS, "scope": "global", "provenance": {"sources": [
        {"kind": "artifact", "artifact_id": span.artifact_id, "run_id": "victim-run"}]}})
    assert state.status == RunStatus.FAILED and "must be this run" in str(state.error)
    assert home.facade.derived_from_source({"kind": "run", "run_id": "victim-run"}) == []
    run_id, ok = home.run_flow({"ops": OPS[:1], "scope": "global", "provenance": {"sources": [
        {"kind": "artifact", "artifact_id": span.artifact_id}]}})
    assert ok.status == RunStatus.COMPLETED, ok.error  # absent run_id: fine


def test_named_artifacts_must_exist_and_be_checkable(tmp_path: Path) -> None:
    """Review 3a R3. Mutants: skip the existence check -> a revision grounded only in a made-up artifact is applied;
    skip the no-store refusal -> the store-less handler accepts an unchecked name."""
    home = _Home(tmp_path)
    loose = RunState.new(workflow_id="wf", entry_node="n1")
    made_up = home.handlers[EffectType.MEMORY_REVISE](loose, Effect(type=EffectType.MEMORY_REVISE, payload={
        "ops": OPS, "scope": "global", "artifact_ids": ["does-not-exist"]}), None)
    assert made_up.status == "failed" and "do not exist" in made_up.error
    as_ref = home.handlers[EffectType.MEMORY_REVISE](loose, Effect(type=EffectType.MEMORY_REVISE, payload={
        "ops": OPS, "scope": "global", "provenance": {"sources": [{"kind": "artifact", "artifact_id": "nope"}]}}), None)
    assert as_ref.status == "failed" and "nope" in as_ref.error
    blind = build_memory_kg_effect_handlers(store=home.store, run_store=home.run_store, now_iso=lambda: NOW)
    span = home.artifacts.store_text("real")
    unchecked = blind[EffectType.MEMORY_REVISE](loose, Effect(type=EffectType.MEMORY_REVISE, payload={
        "ops": OPS, "scope": "global", "artifact_ids": [span.artifact_id]}), None)
    assert unchecked.status == "failed" and "artifact_store" in unchecked.error
    assert home.facade.graph.revisions()[0] == []
    real = home.handlers[EffectType.MEMORY_REVISE](loose, Effect(type=EffectType.MEMORY_REVISE, payload={
        "ops": OPS, "scope": "global", "artifact_ids": [span.artifact_id]}), None)
    assert real.status == "completed" and real.result["sources"] == [{"kind": "artifact", "artifact_id": span.artifact_id}]


def test_the_default_revision_id_covers_the_callers_sources(tmp_path: Path) -> None:
    """Review 3a R4. Mutant: hash only (run, node, scope, owner, cause, ops) -> the second source is a silent no-op."""
    home = _Home(tmp_path)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    a1 = home.artifacts.store_text("first source").artifact_id
    a2 = home.artifacts.store_text("second, different source").artifact_id
    h = home.handlers[EffectType.MEMORY_REVISE]

    def revise(**extra: Any) -> Any:
        return h(run, Effect(type=EffectType.MEMORY_REVISE, payload={"ops": OPS, "scope": "global", **extra}), None)

    first, second = revise(artifact_ids=[a1]), revise(artifact_ids=[a2])
    assert first.result["applied"] and second.result["applied"]
    assert first.result["revision_id"] != second.result["revision_id"]
    retry = revise(artifact_ids=[a1])  # the same revision again (a retried attempt): a no-op
    assert retry.result["revision_id"] == first.result["revision_id"] and retry.result["applied"] is False
    later = revise(artifact_ids=[a1], occurred_at="2026-10-28T00:00:00+00:00")
    assert later.result["applied"] is True


def test_unknown_payload_keys_are_refused(tmp_path: Path) -> None:
    home = _Home(tmp_path)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    outcome = home.handlers[EffectType.MEMORY_REVISE](
        run, Effect(type=EffectType.MEMORY_REVISE, payload={"ops": OPS, "scope": "global", "op": "assert"}), None)
    assert outcome.status == "failed" and "unknown payload keys ['op']" in outcome.error


def test_a_store_that_is_not_a_graph_home_fails_naming_backlog_0057(tmp_path: Path) -> None:
    from abstractmemory import LanceDBTripleStore

    pytest.importorskip("lancedb")
    store = LanceDBTripleStore(tmp_path / "kg")
    run_store = SqliteRunStore(SqliteDatabase(tmp_path / "runtime.sqlite3"))
    handlers = build_memory_kg_effect_handlers(store=store, run_store=run_store, now_iso=lambda: NOW)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    outcome = handlers[EffectType.MEMORY_REVISE](
        run, Effect(type=EffectType.MEMORY_REVISE, payload={"ops": OPS, "scope": "global"}), None)
    assert outcome.status == "failed" and "0057" in outcome.error and "LanceDBTripleStore" in outcome.error


def test_the_node_builds_the_payload_from_its_pins_only() -> None:
    spec = compile_visualflow(_flow({"ops": OPS, "scope": "session", "summary": "  ", "owner_id": None}))
    run = RunState(run_id="r", workflow_id=str(spec.workflow_id), status=RunStatus.RUNNING, current_node="rev",
                   vars={"_temp": {}})
    plan = spec.nodes["rev"](run, None)
    assert plan.effect.type == EffectType.MEMORY_REVISE
    assert {k: v for k, v in plan.effect.payload.items() if k not in {"type", "resume_to_node"}} == {
        "ops": OPS, "scope": "session"}
