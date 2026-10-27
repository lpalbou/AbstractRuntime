"""`MemoryFacade`: the complete host surface for AbstractMemory (ADR 0009).

Real temporary stores throughout (SqliteRunStore, SqliteLedgerStore, FileArtifactStore, DiaryStore over a SQLite
ledger, JsonFileEventInbox, the SQLite home of `open_mind`); no mocks.

Mutants this file kills (each checked by hand when it was written):
- `reconstruct_source` reads the step row with `get_record_unresolved` (markers left in place) -> the slimmed-step
  case fails on the payload;
- `_reconstruct_artifact` / `_reconstruct_step` answer ok for a missing source -> the source_gone cases fail (a
  handler that RAISES is caught by `reconstruct_source`'s own guard and still answers source_gone, by design);
- `source_refs` stamps a step ref without checking `is_real_run` -> the synthetic chat cases fail;
- `source_refs` stamps a step ref when `seq` is None (guessing) -> the no-seq case fails;
- `stamp` lifts a synthetic flat `run_id` through `from_legacy` -> the chat legacy case fails (a `run:chat-...` ref);
- `__getattr__` returns a copy / wrapper or a name maps to the wrong module -> the identity case fails;
- the facade imports AbstractMemory at module import time -> the boundary subprocess fails.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.email.inbox import JsonFileEventInbox
from abstractruntime.identity.diary import DiaryEntry, DiaryStore
from abstractruntime.integrations.abstractmemory import memory_facade as mf
from abstractruntime.integrations.abstractmemory.memory_facade import MemoryFacade, is_real_run
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.ledger_slim import SLIM_FIELD_THRESHOLD_BYTES, contains_slim_marker
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore

pytestmark = pytest.mark.basic

ROOT = Path(__file__).resolve().parents[1]
BIG = "verbatim " * (SLIM_FIELD_THRESHOLD_BYTES // 4)
BIG_CALLS = [{"name": "write_file", "arguments": {"path": "a.txt", "content": BIG}}]

# The exact list the gateway burn-down relies on (AbstractGateway's direct imports + the graph names).
EXPECTED_GATEWAY_NAMES = (
    "SELF_FRACTION_FLOOR", "ENTITY_CONTEXT_ACCEPTABLE", "ENTITY_CONTEXT_FLOOR", "ENTITY_CONTEXT_RECOMMENDED",
    "entity_recall_budget", "REPLAY_FAMILIES", "REPLAY_STREAM", "REPLAY_STREAM_VERSION", "RESERVED_FAMILIES",
    "resolve_digest_assertion", "DEFAULT_SPARK_TEMPLATE", "lint_spark", "canonical_spark_hash", "MemorySystem",
    "SQLiteJournal", "SQLiteTripleStore", "LanceDBTripleStore", "InMemoryTripleStore", "TripleQuery",
    "TripleAssertion", "ClosureRecord", "cognition_health", "verify_diary_chain", "engram", "sleep_pass",
    "dream_pass", "reembed_store", "read_embedding_pin", "DRIVE_PRESSURE_BOUND", "drive_pressure",
    "promote_candidate", "reject_candidate",
    "SourceRef", "Provenance", "ProvenanceRequired", "InvalidSourceRef", "Revision", "RevisionOp", "RevisionResult",
    "MemoryGraph",
)


# -- fixtures -----------------------------------------------------------------------------------------------------


def _stores(tmp_path: Path):
    db = SqliteDatabase(tmp_path / "runtime.sqlite3")
    return SqliteRunStore(db), SqliteLedgerStore(db)


def _run_slimmed_step(run_store: Any, ledger: Any) -> str:
    """A real Runtime executes one TOOL_CALLS effect far above the slimming threshold: its terminal row is slimmed."""

    def node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("out") is not None:
            return StepPlan(node_id="n1", complete_output={"ok": True})
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": BIG_CALLS, "mode": "x"}, result_key="out"),
            next_node="n1",
        )

    def handler(run: RunState, effect: Effect, default_next_node: Optional[str] = None) -> EffectOutcome:
        return EffectOutcome.completed({"results": [{"name": "write_file", "success": True}]})

    wf = WorkflowSpec(workflow_id="wf-facade", entry_node="n1", nodes={"n1": node})
    rt = Runtime(run_store=run_store, ledger_store=ledger, effect_handlers={EffectType.TOOL_CALLS: handler})
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)
    return run_id


# -- the boundary and the re-exports --------------------------------------------------------------------------------


def test_facade_import_does_not_import_abstractmemory() -> None:
    script = f"""
import importlib.abc, sys
sys.path.insert(0, {str(ROOT / 'src')!r})
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] == 'abstractmemory':
            raise AssertionError('abstractmemory imported at module import time: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
import abstractruntime.integrations.abstractmemory.memory_facade as f
assert 'MemoryFacade' in f.__all__ and 'abstractmemory' not in sys.modules
print('clean')
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT), env=dict(os.environ), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "clean"


def test_gateway_names_are_abstractmemorys_own_objects() -> None:
    import abstractmemory

    assert mf.GATEWAY_NAMES == EXPECTED_GATEWAY_NAMES
    for name in mf.GATEWAY_NAMES:
        ours = getattr(mf, name)
        module = importlib.import_module(mf._REEXPORTS[name])
        assert ours is getattr(module, name), name
        if hasattr(abstractmemory, name):  # names on the package root are the root's objects too
            assert ours is getattr(abstractmemory, name), name
        assert name in mf.__all__
    # The gateway imports these from submodules today; the facade carries the same objects.
    import abstractmemory.records as records
    import abstractmemory.replay as replay
    import abstractmemory.seam as seam

    assert mf.resolve_digest_assertion is records.resolve_digest_assertion
    assert mf.REPLAY_STREAM is replay.REPLAY_STREAM
    assert mf.SELF_FRACTION_FLOOR is seam.SELF_FRACTION_FLOOR
    assert mf.verify_diary_chain is records.verify_diary_chain
    with pytest.raises(AttributeError):
        mf.NotAMemoryName  # noqa: B018 - an unknown name fails loudly


# -- is_real_run and stamp --------------------------------------------------------------------------------------------


def test_is_real_run_refuses_synthetic_and_unloadable_ids(tmp_path: Path) -> None:
    run_store, ledger = _stores(tmp_path)
    run_id = _run_slimmed_step(run_store, ledger)
    for synthetic in ("chat-s1", "diary:ent", "global_memory", "session_memory_s1", "", None, 7):
        assert is_real_run(synthetic) is False, synthetic
        assert is_real_run(synthetic, run_store=run_store) is False, synthetic
    assert is_real_run(run_id, run_store=run_store) is True
    assert is_real_run("never-stored", run_store=run_store) is False
    assert is_real_run("never-stored") is True  # without a run store only the syntax is checked


def test_stamp_adds_a_step_ref_for_a_real_run_and_keeps_legacy_keys(tmp_path: Path) -> None:
    run_store, ledger = _stores(tmp_path)
    run_id = _run_slimmed_step(run_store, ledger)
    facade = MemoryFacade(run_store=run_store, ledger_store=ledger)
    out = facade.stamp(
        {"run_id": run_id, "turn_id": "t-0001", "payload_ref": "art1", "source": "x"},
        run_id=run_id, seq=1, step_id="s-1", artifact_ids=["art2"],
    )
    assert out["run_id"] == run_id and out["turn_id"] == "t-0001" and out["source"] == "x"  # legacy keys stay
    kinds = [(r["kind"], r.get("seq"), r.get("artifact_id")) for r in out["sources"]]
    # The step ref supersedes the run ref lifted from the flat run_id (same run, more precise address).
    assert ("step", 1, None) in kinds and ("run", None, None) not in kinds
    assert ("artifact", None, "art1") in kinds and ("artifact", None, "art2") in kinds
    step = next(r for r in out["sources"] if r["kind"] == "step")
    assert step == {"kind": "step", "run_id": run_id, "seq": 1, "step_id": "s-1"}


def test_stamp_never_fabricates_for_a_synthetic_run_or_a_missing_seq(tmp_path: Path) -> None:
    run_store, ledger = _stores(tmp_path)
    run_id = _run_slimmed_step(run_store, ledger)
    facade = MemoryFacade(run_store=run_store)
    chat = facade.stamp({"run_id": "chat-s1", "turn_id": "t-1", "payload_ref": "v1"}, run_id="chat-s1", seq=3)
    assert chat["sources"] == [{"kind": "artifact", "artifact_id": "v1"}]  # the artifact is the ground, nothing else
    assert chat["run_id"] == "chat-s1"
    no_seq = facade.stamp({}, run_id=run_id, seq=None)
    assert no_seq["sources"] == [{"kind": "run", "run_id": run_id}]  # a real run, but no row address: no step ref
    unknown = facade.stamp({}, run_id="never-stored", seq=4)
    assert unknown == {}  # nothing real to name: nothing new is added
    bad_seq = facade.stamp({}, run_id=run_id, seq=0)
    assert bad_seq["sources"] == [{"kind": "run", "run_id": run_id}]


def test_stamp_graph_shape_round_trips_and_validates() -> None:
    from abstractmemory.graph import InvalidSourceRef, ProvenanceRequired

    facade = MemoryFacade(runtime_id="rt1")
    out = facade.stamp({"derived_from": ["m1"], "actor": "entity:e", "reason": "why"}, run_id="r-1", seq=5)
    assert out["actor"] == "entity:e" and out["derived_from"] == ["m1"] and out["reason"] == "why"
    assert out["sources"] == [{"kind": "step", "run_id": "r-1", "seq": 5, "runtime_id": "rt1"}]
    assert set(out) == {"sources", "derived_from", "actor", "reader", "reason", "extra"}
    with pytest.raises(InvalidSourceRef):
        facade.stamp({"sources": [{"kind": "step", "run_id": "r"}]}, run_id=None)
    with pytest.raises(ProvenanceRequired):
        facade.stamp({"sources": [], "derived_from": [""]}, run_id=None)


# -- reconstruct_source -------------------------------------------------------------------------------------------------


def test_reconstruct_step_returns_the_slimmed_rows_payload_verbatim(tmp_path: Path) -> None:
    run_store, ledger = _stores(tmp_path)
    run_id = _run_slimmed_step(run_store, ledger)
    rows = ledger.list(run_id)
    started = next(r for r in rows if r["status"] == "started")
    terminal = next(r for r in rows if r["status"] == "completed" and r["step_id"] == started["step_id"])
    assert contains_slim_marker(terminal["effect"]), "precondition: the terminal row is slimmed"
    facade = MemoryFacade(ledger_store=ledger, run_store=run_store)

    got = facade.reconstruct_source({"kind": "step", "run_id": run_id, "seq": terminal["seq"],
                                     "step_id": started["step_id"]})
    assert got["status"] == "ok" and got["kind"] == "step"
    assert got["content"]["effect"]["payload"] == started["effect"]["payload"]
    assert got["content"]["effect"]["payload"]["tool_calls"][0]["arguments"] == BIG_CALLS[0]["arguments"]
    assert got["metadata"]["status"] == "completed" and got["metadata"]["effect_type"] == "tool_calls"

    start = facade.reconstruct_source({"kind": "step", "run_id": run_id, "seq": started["seq"]})
    assert start["status"] == "ok" and start["content"]["effect"] == started["effect"]

    wrong_step = facade.reconstruct_source({"kind": "step", "run_id": run_id, "seq": started["seq"], "step_id": "x"})
    assert wrong_step["status"] == "source_gone"
    beyond = facade.reconstruct_source({"kind": "step", "run_id": run_id, "seq": 999})
    assert beyond["status"] == "source_gone" and "999" in beyond["reason"]

    run = facade.reconstruct_source({"kind": "run", "run_id": run_id})
    assert run["status"] == "ok" and run["content"]["run"]["run_id"] == run_id
    assert run["metadata"]["ledger_records"] == len(rows)

    ledger.delete(run_id)
    run_store.delete(run_id)
    assert facade.reconstruct_source({"kind": "step", "run_id": run_id, "seq": started["seq"]})["status"] == "source_gone"
    assert facade.reconstruct_source({"kind": "run", "run_id": run_id})["status"] == "source_gone"


def test_reconstruct_artifact_diary_event_message(tmp_path: Path) -> None:
    artifacts = FileArtifactStore(tmp_path / "home")
    meta = artifacts.store(b"the verbatim", content_type="text/plain", run_id="r-1")
    blob = artifacts.store(b"\xff\x00\xfe", content_type="application/octet-stream")
    diary = DiaryStore(entity_id="ent", ledger_store=SqliteLedgerStore(SqliteDatabase(tmp_path / "home.sqlite3")))
    diary.append_entry(DiaryEntry(entry_id="diary_1", author="ent", text="I wrote this.", written_at="2026-10-27T00:00:00Z"))
    inbox = JsonFileEventInbox(tmp_path / "events")
    inbox.append(stream="user", event_id="ev-1", payload={"hello": "world"})
    facade = MemoryFacade(artifact_store=artifacts, diary_store=diary, event_inbox=inbox)

    art = facade.reconstruct_source({"kind": "artifact", "artifact_id": meta.artifact_id})
    assert art["status"] == "ok" and art["content"] == "the verbatim" and art["metadata"]["run_id"] == "r-1"
    binary = facade.reconstruct_source({"kind": "artifact", "artifact_id": blob.artifact_id})
    assert binary["metadata"]["encoding"] == "base64"
    artifacts.delete(meta.artifact_id)
    gone = facade.reconstruct_source({"kind": "artifact", "artifact_id": meta.artifact_id})
    assert gone["status"] == "source_gone" and meta.artifact_id in gone["reason"]

    entry = facade.reconstruct_source({"kind": "diary_entry", "entry_id": "diary_1", "run_id": "diary:ent"})
    assert entry["status"] == "ok" and entry["content"]["text"] == "I wrote this."
    assert entry["metadata"]["record_hash"]
    assert facade.reconstruct_source({"kind": "diary_entry", "entry_id": "nope"})["status"] == "source_gone"
    other = facade.reconstruct_source({"kind": "diary_entry", "entry_id": "diary_1", "run_id": "diary:other"})
    assert other["status"] == "unsupported"

    ev = facade.reconstruct_source({"kind": "event", "event_id": "ev-1", "stream": "user"})
    assert ev["status"] == "ok" and ev["content"] == {"hello": "world"} and ev["metadata"]["stream"] == "user"
    assert facade.reconstruct_source({"kind": "event", "event_id": "ev-2"})["status"] == "source_gone"

    msg = facade.reconstruct_source({"kind": "message", "message_id": "msg_1"})
    assert msg["status"] == "unsupported"


def test_reconstruct_never_raises_for_missing_stores_or_invalid_refs() -> None:
    facade = MemoryFacade(runtime_id="rt1")
    for ref in (
        {"kind": "step", "run_id": "r", "seq": 1},
        {"kind": "run", "run_id": "r"},
        {"kind": "artifact", "artifact_id": "a"},
        {"kind": "diary_entry", "entry_id": "e"},
        {"kind": "event", "event_id": "e"},
        {"kind": "step", "run_id": "r"},
        {"kind": "step", "run_id": "r", "seq": True},
        {"kind": "nope"},
        "step:r#1",
        {"kind": "artifact", "artifact_id": "a", "runtime_id": "rt2"},
    ):
        out = facade.reconstruct_source(ref)
        assert out["status"] == "unsupported", (ref, out)
        assert out["reason"]


# -- the mind and the graph reads -----------------------------------------------------------------------------------------


def test_open_mind_applies_a_revision_and_reads_its_sources(tmp_path: Path) -> None:
    facade = MemoryFacade.open_mind(tmp_path / "memory.sqlite3")
    from abstractmemory import SQLiteJournal, SQLiteTripleStore
    from abstractmemory.graph import MemoryGraph

    assert isinstance(facade.graph, MemoryGraph)
    assert isinstance(facade.store, SQLiteTripleStore) and isinstance(facade.journal, SQLiteJournal)
    prov = facade.stamp({"sources": [], "actor": "runtime"}, run_id="r-1", seq=2, step_id="s")
    result = facade.apply_revision({
        "revision_id": "rev-1", "scope": "run", "owner_id": "r-1", "cause": "ingest",
        "occurred_at": "2026-10-27T00:00:00+00:00",
        "ops": [{"op": "assert", "payload": {"subject": "ex:a", "predicate": "schema:knows", "object": "ex:b"}}],
        "provenance": prov,
    })
    assert result["applied"] is True and len(result["assertion_ids"]) == 1
    aid = result["assertion_ids"][0]
    assert facade.sources_of(aid) == [{"kind": "step", "run_id": "r-1", "seq": 2, "step_id": "s"}]
    assert facade.derived_from_source({"kind": "step", "run_id": "r-1", "seq": 2}) == [aid]
    assert [h["event"] for h in facade.history(aid)] == ["asserted"]
    with pytest.raises(RuntimeError):
        MemoryFacade().graph  # noqa: B018 - a facade without a mind says so
