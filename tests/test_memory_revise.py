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
    from abstractruntime.storage.serialize import record_digest

    step_ref = {"kind": "step", "run_id": run_id, "seq": started[0]["seq"], "step_id": started[0]["step_id"],
                "digest": record_digest(started[0])}
    assert step_ref in out["sources"]
    assert {"kind": "artifact", "artifact_id": span.artifact_id, "run_id": run_id, "digest": span.blob_id} in out["sources"]

    # The graph indexes the revision by its ledger row, both ways.
    assert sorted(home.facade.derived_from_source({"kind": "step", "run_id": run_id, "seq": step_ref["seq"]})) == sorted(
        out["assertion_ids"])
    for aid in out["assertion_ids"]:
        assert step_ref in [{k: v for k, v in s.items() if k in step_ref} for s in home.facade.sources_of(aid)]
        assert home.facade.history(aid)[0]["revision_id"] == out["revision_id"]
    (rev,), _ = home.facade.graph.revisions()
    assert rev.provenance.actor == "entity:e1" and rev.provenance.reason == "she said so" and rev.summary == "met Bob"

    # The source reads back verbatim: the MEMORY_REVISE STARTED row carries the revision payload as sent.
    back = home.facade.reconstruct_source(step_ref)
    assert back["status"] == "ok" and back["metadata"]["effect_type"] == "memory_revise"
    assert back["metadata"]["digest_verified"] is True
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
    assert real.status == "completed"
    assert real.result["sources"] == [{"kind": "artifact", "artifact_id": span.artifact_id, "digest": span.blob_id}]


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


def test_record_time_cannot_be_set_by_a_workflow(tmp_path: Path) -> None:
    """Architect A3: a backdated `recorded_at` would rewrite when this mind learned the memory. Mutant: accept the key
    -> the revision is applied with the caller's 2020 record time."""
    home = _Home(tmp_path)
    _, state = home.run_flow({"ops": OPS, "scope": "global", "recorded_at": "2020-01-01T00:00:00+00:00",
                              "occurred_at": "2019-06-01T00:00:00+00:00"})
    assert state.status == RunStatus.FAILED and "recorded_at" in str(state.error)
    assert home.facade.graph.revisions()[0] == []


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


def test_a_run_writes_only_its_own_owner_partitions(tmp_path: Path) -> None:
    """Architect B1: a run in session `attacker-session` must not write into `session_memory_victim-session`, through
    MEMORY_REVISE or MEMORY_KG_ASSERT (payload owner_id or a per-assertion owner_id). Its own run / session / global
    owners stay writable. Mutant: drop the owner check -> the victim's partition holds the attacker's memory."""
    from abstractmemory import TripleQuery

    from abstractruntime.integrations.abstractmemory.effect_handlers import derived_owner_ids, resolve_scope_owner_id

    home = _Home(tmp_path)
    run = RunState.new(workflow_id="wf", entry_node="n1", session_id="attacker-session")
    home.run_store.save(run)
    victim = resolve_scope_owner_id(RunState.new(workflow_id="wf", entry_node="n1", session_id="victim-session"),
                                    scope="session", run_store=home.run_store)
    mine = resolve_scope_owner_id(run, scope="session", run_store=home.run_store)
    assert victim != mine and victim not in derived_owner_ids(run, run_store=home.run_store)
    revise = home.handlers[EffectType.MEMORY_REVISE]
    assert_ = home.handlers[EffectType.MEMORY_KG_ASSERT]
    span = home.artifacts.store_text("evidence")

    bad = revise(run, Effect(type=EffectType.MEMORY_REVISE, payload={
        "ops": OPS, "scope": "session", "owner_id": victim, "artifact_ids": [span.artifact_id]}), None)
    assert bad.status == "failed" and "not a partition this run may write" in bad.error
    claim = {"subject": "ex:x", "predicate": "schema:description", "object": "y"}
    bad_kg = assert_(run, Effect(type=EffectType.MEMORY_KG_ASSERT, payload={
        "assertions": [claim], "scope": "session", "owner_id": victim}), None)
    assert bad_kg.status == "failed" and victim in bad_kg.error
    bad_row = assert_(run, Effect(type=EffectType.MEMORY_KG_ASSERT, payload={
        "assertions": [{**claim, "owner_id": victim}], "scope": "session"}), None)
    assert bad_row.status == "failed"
    assert home.store.query(TripleQuery(owner_id=victim, limit=0)) == []

    ok = revise(run, Effect(type=EffectType.MEMORY_REVISE, payload={
        "ops": OPS[:1], "scope": "session", "owner_id": mine, "artifact_ids": [span.artifact_id]}), None)
    assert ok.status == "completed" and ok.result["owner_id"] == mine
    ok_kg = assert_(run, Effect(type=EffectType.MEMORY_KG_ASSERT, payload={"assertions": [claim], "scope": "global"}),
                    None)
    assert ok_kg.status == "completed"


def _revise_steps(home: "_Home", run_id: str) -> list:
    return [r for r in home.ledger.list(run_id) if r["status"] == "started" and r["effect"]["type"] == "memory_revise"]


def test_the_same_ops_run_twice_are_one_memory_re_observed_by_both_runs(tmp_path: Path) -> None:
    """Architect E1, as resolved with AbstractMemory 8e4279f: a re-run is a NEW revision (the run is in the default
    revision_id) that re-observes the memory the graph holds open: no duplicate claim, and the second run's step is
    recorded on it. Mutant: drop the run from the basis -> the re-run is a no-op and its step is lost."""
    from abstractmemory import TripleQuery

    home = _Home(tmp_path)
    _, s1 = home.run_flow({"ops": OPS, "scope": "global"})
    r2, s2 = home.run_flow({"ops": OPS, "scope": "global"})
    o1, o2 = _effect_result(s1), _effect_result(s2)
    assert o1["revision_id"] != o2["revision_id"] and o2["applied"] is True
    assert sorted(o2["re_observed"]) == sorted(o1["assertion_ids"])
    assert len(home.store.query(TripleQuery(subject="ex:alice", limit=0))) == 2  # one row per claim, not two
    (step2,) = _revise_steps(home, r2)
    ids = home.facade.derived_from_source({"kind": "step", "run_id": r2, "seq": step2["seq"]})
    assert sorted(ids) == sorted(o1["assertion_ids"])


def _retry_after_a_post_apply_crash(tmp_path: Path) -> tuple:
    from abstractruntime.core.policy import DefaultEffectPolicy

    home = _Home(tmp_path)
    inner = home.handlers[EffectType.MEMORY_REVISE]
    calls = {"n": 0}

    def flaky(run: RunState, effect: Effect, nxt: Any = None) -> Any:
        outcome = inner(run, effect, nxt)
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash after the revision was applied")
        return outcome

    home.handlers[EffectType.MEMORY_REVISE] = flaky
    spec = compile_visualflow(_flow({"ops": OPS, "scope": "global"}))
    rt = Runtime(run_store=home.run_store, ledger_store=home.ledger, effect_handlers=home.handlers,
                 effect_policy=DefaultEffectPolicy(default_max_attempts=3, default_backoff_base=0.0,
                                                   default_backoff_max=0.0))
    run_id = rt.start(workflow=spec)
    state = rt.tick(workflow=spec, run_id=run_id)
    assert state.status == RunStatus.COMPLETED, state.error
    return home, run_id, _effect_result(state), _revise_steps(home, run_id)


def test_a_replayed_revision_never_reports_the_retry_step_as_the_memorys_source(tmp_path: Path) -> None:
    """Architect A5: attempt 1 applied the revision and crashed; attempt 2 is a no-op. The outcome must not present
    attempt 2's step ref as the memory's source. Mutant: keep the stamped sources -> attempt 2's step is reported."""
    home, run_id, out, (attempt1, attempt2) = _retry_after_a_post_apply_crash(tmp_path)
    assert out["applied"] is False and out["replayed"] is True
    assert not any(s.get("seq") == attempt2["seq"] for s in out["sources"])
    assert any(s.get("seq") == attempt2["seq"] for s in out["attempt_sources"])
    (rev,), _ = home.facade.graph.revisions()
    assert [s.seq for s in rev.provenance.sources if s.kind == "step"] == [attempt1["seq"]]


def test_a_replayed_revision_reports_the_memorys_ids_and_sources(tmp_path: Path) -> None:
    """Architect A5 (AbstractMemory's RevisionResult carries the logged ids on a replay): the outcome reports the
    memory's ids and the memory's own sources (attempt 1's step). Mutant: keep the stamped sources -> fails."""
    home, run_id, out, (attempt1, _) = _retry_after_a_post_apply_crash(tmp_path)
    assert len(out["assertion_ids"]) == 2
    assert any(s.get("kind") == "step" and s.get("seq") == attempt1["seq"] for s in out["sources"])

