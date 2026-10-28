"""Every memory write the runtime performs names its ledger source (ADR 0011 decisions 3-4; backlog 0856).

REAL `Runtime` runs on real temporary stores (SqliteRunStore + SqliteLedgerStore, FileArtifactStore, the SQLite
AbstractMemory home, a DiaryStore over a SQLite ledger), no mocks. A step ref must resolve through
`LedgerStore.get_record` to the STARTED row of the very effect that wrote the memory, payload verbatim.

Mutants this file kills (each checked by hand when it was written; each test fails with its stamping line removed):
- `seam_handlers._handle_form` without `facade.stamp_current_step(...)` -> the MEMORY_FORM cases fail (no sources);
- `effect_handlers._handle_assert` without the stamp -> the MEMORY_KG_ASSERT case fails;
- `identity_support.project_diary_entry` without the stamp -> the public diary case fails; without its
  `diary_entry` ref -> every diary case fails;
- the stamp reading the step context of ANY run (no run filter) or stamping the synthetic `chat-<sid>` run -> the
  chat case fails (it must carry the artifact ref only).
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.identity.diary import DiaryStore, build_diary_effect_handlers
from abstractruntime.integrations.abstractmemory import (
    MemoryFacade,
    build_memory_kg_effect_handlers,
    build_memory_seam_effect_handlers,
)
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore

pytestmark = pytest.mark.basic

pytest.importorskip("abstractmemory.graph")

from abstractmemory import MemorySystem, SQLiteJournal, SQLiteTripleStore, TripleQuery  # noqa: E402

NOW = "2026-10-27T00:00:00+00:00"
VERBATIM = "turn t-0001: which pooler? pgbouncer, transaction mode."


class _Env:
    def __init__(self, tmp_path: Path) -> None:
        db = SqliteDatabase(tmp_path / "runtime.sqlite3")
        self.run_store, self.ledger = SqliteRunStore(db), SqliteLedgerStore(db)
        self.artifacts = FileArtifactStore(tmp_path / "home")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.store = SQLiteTripleStore(tmp_path / "memory.sqlite3")
            self.journal = SQLiteJournal(tmp_path / "memory.sqlite3")
        self.ms = MemorySystem(store=self.store, journal=self.journal)
        self.facade = MemoryFacade(ledger_store=self.ledger, run_store=self.run_store, artifact_store=self.artifacts)

    def run_one(self, effect_type: EffectType, payload: Dict[str, Any], handlers: Dict[Any, Any]) -> tuple:
        def node(run: RunState, ctx: Any) -> StepPlan:
            if run.vars.get("out") is not None:
                return StepPlan(node_id="n1", complete_output={"ok": True})
            return StepPlan(node_id="n1", effect=Effect(type=effect_type, payload=dict(payload), result_key="out"),
                            next_node="n1")

        wf = WorkflowSpec(workflow_id=f"wf-{effect_type.value}", entry_node="n1", nodes={"n1": node})
        rt = Runtime(run_store=self.run_store, ledger_store=self.ledger, effect_handlers=handlers)
        run_id = rt.start(workflow=wf)
        state = rt.tick(workflow=wf, run_id=run_id)
        assert state.status.value == "completed", state.error
        return run_id, state.vars["out"]

    def provenances(self, *, subject: Optional[str] = None, scope: str, owner_id: str) -> List[Dict[str, Any]]:
        rows = self.store.query(TripleQuery(subject=subject, scope=scope, owner_id=owner_id, limit=0))
        assert rows, "the write left no assertion"
        return [dict(r.provenance or {}) for r in rows]

    def started_step(self, run_id: str, effect_type: str) -> Dict[str, Any]:
        rows = [r for r in self.ledger.list(run_id)
                if r["status"] == "started" and (r.get("effect") or {}).get("type") == effect_type]
        assert len(rows) == 1, rows
        return rows[0]


def _assert_parity(prov: Dict[str, Any]) -> None:
    """Review 3a R8: the runtime's stamped dict (legacy flat keys + `sources`) must read back through AbstractMemory's
    own parser with exactly the refs the runtime stamped (no ref lifted twice from a flat key, none lost).
    Needs AbstractMemory 4c821b2 (`from_legacy` reads the mixed shape)."""
    from abstractmemory.graph import Provenance

    parsed = Provenance.from_legacy(prov)
    assert [s.to_dict() for s in parsed.sources] == prov["sources"], (prov, parsed)


def _step_refs(prov: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in prov.get("sources") or [] if s["kind"] == "step"]


def _assert_step_ref_is_the_writer(env: _Env, prov: Dict[str, Any], run_id: str, effect_type: str) -> Dict[str, Any]:
    started = env.started_step(run_id, effect_type)
    from abstractruntime.storage.serialize import record_digest

    refs = _step_refs(prov)
    # The address, and the digest of the STARTED row as written (architect A4): what the memory vouches it said.
    assert refs == [{"kind": "step", "run_id": run_id, "seq": started["seq"], "step_id": started["step_id"],
                     "digest": record_digest(started)}], prov
    row = env.ledger.get_record(run_id, refs[0]["seq"])
    assert row["status"] == "started" and row["effect"] == started["effect"]  # the writing effect, verbatim
    rebuilt = env.facade.reconstruct_source(refs[0])
    assert rebuilt["status"] == "ok" and rebuilt["metadata"]["effect_type"] == effect_type
    assert rebuilt["metadata"]["digest_verified"] is True
    return row


# -- MEMORY_FORM --------------------------------------------------------------------------------------------------------


def _form_payload(owner: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "records": [{"kind": "memory", "title": "pool decision", "keywords": ["pgbouncer"],
                     "digest": "team chose pgbouncer in transaction mode", "verbatim": VERBATIM,
                     "provenance": {"source": "entity-visit-run-v0"}}],
        "scope": "run", "turn_id": "t-0001",
    }
    if owner:
        payload["owner_id"] = owner
    return payload


def test_memory_form_in_a_real_run_names_its_step_and_verbatim(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    handlers = build_memory_seam_effect_handlers(memory_system=env.ms, run_store=env.run_store,
                                                 now_iso=lambda: NOW, artifact_store=env.artifacts)
    run_id, out = env.run_one(EffectType.MEMORY_FORM, _form_payload(), handlers)
    record_id = out["record_ids"][0]
    provs = env.provenances(subject=record_id, scope="run", owner_id=run_id)
    for prov in provs:
        assert prov["run_id"] == run_id and prov["turn_id"] == "t-0001"  # legacy keys stay
        assert prov["source"] == "entity-visit-run-v0"
        _assert_parity(prov)
        row = _assert_step_ref_is_the_writer(env, prov, run_id, "memory_form")
        assert row["effect"]["payload"]["records"][0]["verbatim"] == VERBATIM
        arts = [s for s in prov["sources"] if s["kind"] == "artifact"]
        assert len(arts) == 1 and arts[0]["run_id"] == run_id
        assert arts[0]["digest"] == hashlib.sha256(VERBATIM.encode("utf-8")).hexdigest()  # the bytes it vouches for
        verbatim = env.facade.reconstruct_source(arts[0])
        assert verbatim["status"] == "ok" and verbatim["content"] == VERBATIM
        assert not [s for s in prov["sources"] if s["kind"] == "run"]  # the step ref is the precise address


def test_memory_form_on_the_synthetic_chat_path_names_the_artifact_only(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    handlers = build_memory_seam_effect_handlers(memory_system=env.ms, run_store=env.run_store,
                                                 now_iso=lambda: NOW, artifact_store=env.artifacts)
    chat = RunState.new(workflow_id="entity-chat", entry_node="n1", run_id="chat-s1")
    # The home-direct driver calls the handler directly: no runtime step, no ledger row, a run no store can load.
    outcome = handlers[EffectType.MEMORY_FORM](chat, Effect(type=EffectType.MEMORY_FORM, payload=_form_payload()), None)
    assert outcome.status == "completed", outcome.error
    record_id = outcome.result["record_ids"][0]
    for prov in env.provenances(subject=record_id, scope="run", owner_id="chat-s1"):
        assert prov["run_id"] == "chat-s1"
        _assert_parity(prov)
        assert [s["kind"] for s in prov["sources"]] == ["artifact"]
        assert "run_id" not in prov["sources"][0]  # the synthetic run is not even attached to the artifact ref
        assert env.facade.reconstruct_source(prov["sources"][0])["content"] == VERBATIM


def test_memory_form_called_directly_for_a_stored_run_gets_no_step_ref(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    handlers = build_memory_seam_effect_handlers(memory_system=env.ms, run_store=env.run_store,
                                                 now_iso=lambda: NOW, artifact_store=env.artifacts)
    run = RunState.new(workflow_id="wf", entry_node="n1")
    env.run_store.save(run)
    outcome = handlers[EffectType.MEMORY_FORM](run, Effect(type=EffectType.MEMORY_FORM, payload=_form_payload()), None)
    assert outcome.status == "completed", outcome.error
    for prov in env.provenances(subject=outcome.result["record_ids"][0], scope="run", owner_id=run.run_id):
        assert not _step_refs(prov)  # no step is executing: no seq is known, none is guessed
        assert {"kind": "run", "run_id": run.run_id} in prov["sources"]


# -- MEMORY_KG_ASSERT -------------------------------------------------------------------------------------------------


def test_memory_kg_assert_in_a_real_run_names_its_step_and_span(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    span = env.artifacts.store_text("the span the fact came from", run_id=None)
    handlers = build_memory_kg_effect_handlers(store=env.store, run_store=env.run_store, now_iso=lambda: NOW,
                                               artifact_store=env.artifacts)
    payload = {"assertions": [{"subject": "ex:alice", "predicate": "schema:description", "object": "a person"}],
               "span_id": span.artifact_id, "scope": "run"}
    run_id, out = env.run_one(EffectType.MEMORY_KG_ASSERT, payload, handlers)
    assert out["count"] == 1
    (prov,) = env.provenances(subject="ex:alice", scope="run", owner_id=run_id)
    assert prov["writer_run_id"] == run_id and prov["span_id"] == span.artifact_id  # legacy keys stay
    _assert_parity(prov)
    _assert_step_ref_is_the_writer(env, prov, run_id, "memory_kg_assert")
    (art,) = [s for s in prov["sources"] if s["kind"] == "artifact"]
    assert art["artifact_id"] == span.artifact_id
    assert art["digest"] == hashlib.sha256(b"the span the fact came from").hexdigest()
    assert env.facade.reconstruct_source(art)["content"] == "the span the fact came from"


# -- DIARY_WRITE projection -------------------------------------------------------------------------------------------


def _diary_env(tmp_path: Path):
    env = _Env(tmp_path)
    diary = DiaryStore(entity_id="ent", ledger_store=SqliteLedgerStore(SqliteDatabase(tmp_path / "home.sqlite3")))
    env.facade.diary_store = diary
    handlers = build_diary_effect_handlers(entity_id="ent", diary_store=diary, memory_system=env.ms,
                                           now_iso=lambda: NOW)
    return env, handlers


@pytest.mark.parametrize("visibility", ["self", "private"])
def test_diary_projection_names_the_entry_and_the_writing_step(tmp_path: Path, visibility: str) -> None:
    env, handlers = _diary_env(tmp_path)
    payload = {"text": "Today I learned the pool is pgbouncer.", "turn_id": "t-0002", "visibility": visibility,
               "gist": "learned the pool"}
    run_id, out = env.run_one(EffectType.DIARY_WRITE, payload, handlers)
    record_id = out["projected_record_id"]
    chain_hash = env.facade.diary_store.get_record(out["entry_id"])["record_hash"]
    entry_ref = {"kind": "diary_entry", "entry_id": out["entry_id"], "run_id": "diary:ent", "digest": chain_hash}
    for prov in env.provenances(subject=record_id, scope="diary", owner_id="ent"):
        assert prov["entry_id"] == out["entry_id"] and prov["source"] == "diary-projection"
        _assert_parity(prov)
        assert entry_ref in prov["sources"]
        entry = env.facade.reconstruct_source(entry_ref)
        assert entry["status"] == "ok" and entry["content"]["text"] == payload["text"]
        assert entry["metadata"]["digest_verified"] is True
        if visibility == "private":
            assert prov["sources"] == [entry_ref]  # a private projection carries no origin, the chain holds it
        else:
            assert prov["run_id"] == run_id
            _assert_step_ref_is_the_writer(env, prov, run_id, "diary_write")


def test_diary_projection_on_the_chat_path_names_the_entry_only(tmp_path: Path) -> None:
    env, handlers = _diary_env(tmp_path)
    chat = RunState.new(workflow_id="entity-chat", entry_node="n1", run_id="chat-s9")
    payload = {"text": "A chat-path diary line.", "turn_id": "t-0003", "gist": "chat line"}
    outcome = handlers[EffectType.DIARY_WRITE](chat, Effect(type=EffectType.DIARY_WRITE, payload=payload), None)
    assert outcome.status == "completed", outcome.error
    for prov in env.provenances(subject=outcome.result["projected_record_id"], scope="diary", owner_id="ent"):
        assert prov["run_id"] == "chat-s9"  # the legacy origin stays
        _assert_parity(prov)
        chain_hash = env.facade.diary_store.get_record(outcome.result["entry_id"])["record_hash"]
        assert prov["sources"] == [{"kind": "diary_entry", "entry_id": outcome.result["entry_id"], "run_id": "diary:ent",
                                    "digest": chain_hash}]


def test_a_duplicate_kg_assertion_reports_the_sources_it_could_not_record(tmp_path: Path) -> None:
    """Review 3a R11: a repeated s/p/o/scope/owner is not re-written, so its new step + span refs cannot land on the
    existing row; the outcome lists them (`skipped_duplicate_rows[].lost_sources`) instead of dropping them silently.
    Mutant: omit the record -> the second outcome has no `skipped_duplicate_rows`."""
    env = _Env(tmp_path)
    s1 = env.artifacts.store_text("first span")
    s2 = env.artifacts.store_text("second span")
    handlers = build_memory_kg_effect_handlers(store=env.store, run_store=env.run_store, now_iso=lambda: NOW)
    a = {"subject": "ex:alice", "predicate": "schema:description", "object": "a person"}
    _, first = env.run_one(EffectType.MEMORY_KG_ASSERT, {"assertions": [a], "scope": "global", "span_id": s1.artifact_id},
                           handlers)
    run_id, second = env.run_one(EffectType.MEMORY_KG_ASSERT,
                                 {"assertions": [a], "scope": "global", "span_id": s2.artifact_id}, handlers)
    assert first["count"] == 1 and second["count"] == 0 and second["skipped_duplicates"] == 1
    (row,) = second["skipped_duplicate_rows"]
    assert row["existing_assertion_id"] == first["assertion_ids"][0]
    kinds = {(s["kind"], s.get("artifact_id"), s.get("run_id")) for s in row["lost_sources"]}
    assert ("artifact", s2.artifact_id, None) in kinds and any(k == "step" and r == run_id for k, _, r in kinds)

