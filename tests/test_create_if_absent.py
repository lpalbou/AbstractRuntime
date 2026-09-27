"""RunStore.create_if_absent on every store (automations contract C1).

First creation wins; a later creation with the same identity loads the stored
run unchanged; a different identity raises RunIdentityConflict; nothing is
ever overwritten. Stores without the primitive raise (no emulation) and the
capability preflight sees through the offloading wrapper.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import pytest

from abstractruntime import (
    Runtime,
    RunIdentityConflict,
    StepPlan,
    WorkflowSpec,
)
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.base import RunStore, require_create_if_absent, store_supports_create_if_absent
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.offloading import OffloadingRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

STORES = ["memory", "json", "sqlite", "offload_json", "offload_sqlite"]


def make_store(kind: str, tmp_path):
    if kind == "memory":
        return InMemoryRunStore()
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    if kind == "sqlite":
        return SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    arts = FileArtifactStore(tmp_path / "artifacts")
    inner = JsonFileRunStore(tmp_path / "runs") if kind == "offload_json" else SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    return OffloadingRunStore(inner, artifact_store=arts, max_inline_bytes=64)


def _run(run_id: str = "child-1", *, workflow_id: str = "wf", session_id: Optional[str] = "s1",
         parent_run_id: Optional[str] = "p1", meta: Optional[dict] = None) -> RunState:
    return RunState.new(
        workflow_id=workflow_id,
        entry_node="start",
        vars={"_meta": dict(meta or {"creation_digest": "sha256:abc"})},
        session_id=session_id,
        parent_run_id=parent_run_id,
        run_id=run_id,
    )


@pytest.mark.parametrize("kind", STORES)
def test_first_creation_wins_and_is_never_overwritten(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    assert store_supports_create_if_absent(store)

    first, created = store.create_if_absent(_run())
    assert created is True and first.run_id == "child-1"

    # The run progresses after creation.
    progressed = store.load("child-1")
    progressed.vars["progress"] = 7
    progressed.current_node = "later"
    store.save(progressed)

    again, created2 = store.create_if_absent(_run())
    assert created2 is False
    assert again.vars.get("progress") == 7
    assert again.current_node == "later"
    assert store.load("child-1").vars.get("progress") == 7


@pytest.mark.parametrize("kind", STORES)
@pytest.mark.parametrize(
    "field,override",
    [
        ("workflow_id", {"workflow_id": "other"}),
        ("session_id", {"session_id": "s2"}),
        ("parent_run_id", {"parent_run_id": "p2"}),
        ("vars._meta.creation_digest", {"meta": {"creation_digest": "sha256:def"}}),
        ("vars._meta.occurrence", {"meta": {"creation_digest": "sha256:abc", "occurrence": {"occurrence_index": 2}}}),
    ],
)
def test_identity_mismatch_raises_identity_conflict(kind, field, override, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    store.create_if_absent(_run())
    with pytest.raises(RunIdentityConflict) as info:
        store.create_if_absent(_run(**override))
    assert info.value.field == field
    assert info.value.reason_code == "identity_conflict"
    assert isinstance(info.value, ValueError)
    # The stored run is untouched by the refused creation.
    assert store.load("child-1").workflow_id == "wf"


@pytest.mark.parametrize("kind", ["json", "sqlite", "memory"])
def test_racing_creators_create_exactly_once(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    results: list = []
    barrier = threading.Barrier(8)

    def creator() -> None:
        barrier.wait()
        results.append(store.create_if_absent(_run())[1])

    threads = [threading.Thread(target=creator) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5.0)
    assert sorted(results) == [False] * 7 + [True]


def test_json_publication_leaves_no_temp_and_keeps_the_file(tmp_path) -> None:
    store = JsonFileRunStore(tmp_path / "runs")
    store.create_if_absent(_run())
    path = tmp_path / "runs" / "run_child-1.json"
    inode = path.stat().st_ino
    store.create_if_absent(_run())
    with pytest.raises(RunIdentityConflict):
        store.create_if_absent(_run(workflow_id="other"))
    assert path.stat().st_ino == inode  # never replaced
    assert [p.name for p in (tmp_path / "runs").iterdir() if p.name.endswith(".tmp")] == []


class _SaveOnlyStore(RunStore):
    def __init__(self) -> None:
        self.runs: dict = {}

    def save(self, run: RunState) -> None:
        self.runs[run.run_id] = run

    def load(self, run_id: str):
        return self.runs.get(run_id)


def test_store_without_the_primitive_raises_and_is_refused_at_preflight(tmp_path) -> None:
    bare = _SaveOnlyStore()
    assert store_supports_create_if_absent(bare) is False
    with pytest.raises(NotImplementedError):
        bare.create_if_absent(_run())
    wrapped = OffloadingRunStore(bare, artifact_store=FileArtifactStore(tmp_path / "a"))
    assert store_supports_create_if_absent(wrapped) is False
    with pytest.raises(NotImplementedError):
        require_create_if_absent(wrapped)

    wf = WorkflowSpec(workflow_id="wf", entry_node="n", nodes={"n": lambda run, ctx: StepPlan(node_id="n", complete_output={})})
    runtime = Runtime(run_store=bare, ledger_store=InMemoryLedgerStore())
    with pytest.raises(NotImplementedError):
        runtime.start(workflow=wf, vars={}, run_id="explicit-1")
    assert bare.runs == {}
    # Random-id starts keep working on such stores.
    assert runtime.start(workflow=wf, vars={}) in bare.runs


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_runtime_start_with_explicit_id_loads_without_reseeding(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)
    wf = WorkflowSpec(workflow_id="wf", entry_node="n", nodes={"n": lambda run, ctx: StepPlan(node_id="n", complete_output={})})
    runtime = Runtime(run_store=store, ledger_store=InMemoryLedgerStore())

    rid = runtime.start(workflow=wf, vars={"prompt": "hi"}, session_id="s", run_id="auto-1")
    assert rid == "auto-1"
    run = store.load(rid)
    digest = run.vars["_meta"]["creation_digest"]
    assert digest.startswith("sha256:")
    run.vars["progress"] = "kept"
    store.save(run)

    assert runtime.start(workflow=wf, vars={"prompt": "hi"}, session_id="s", run_id="auto-1") == "auto-1"
    assert store.load(rid).vars.get("progress") == "kept"

    with pytest.raises(RunIdentityConflict) as info:
        runtime.start(workflow=wf, vars={"prompt": "different"}, session_id="s", run_id="auto-1")
    assert info.value.field == "vars._meta.creation_digest"
    with pytest.raises(RunIdentityConflict):
        runtime.start(workflow=wf, vars={"prompt": "hi"}, session_id="other", run_id="auto-1")

    # A caller-supplied digest wins (hosts digest their own request and
    # exclude generated values such as timestamps).
    runtime.start(workflow=wf, vars={"_meta": {"creation_digest": "sha256:req"}, "created_at": "t1"}, run_id="auto-2")
    runtime.start(workflow=wf, vars={"_meta": {"creation_digest": "sha256:req"}, "created_at": "t2"}, run_id="auto-2")


def test_runtime_start_rejects_unsafe_ids(tmp_path) -> None:
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    wf = WorkflowSpec(workflow_id="wf", entry_node="n", nodes={"n": lambda run, ctx: StepPlan(node_id="n", complete_output={})})
    for bad in ("", "../x", "a/b"):
        with pytest.raises(ValueError):
            runtime.start(workflow=wf, run_id=bad)


@pytest.mark.parametrize("kind", ["offload_json", "offload_sqlite"])
def test_offloading_keeps_identity_meta_inline_on_terminal_runs(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)  # max_inline_bytes=64
    big = "x" * 500
    run = _run(meta={"creation_digest": "sha256:abc", "occurrence": {"automation_id": "a", "trigger_envelope": {"payload": big}}})
    run.vars["_temp"] = {"blob": big}
    store.create_if_absent(run)
    run.status = RunStatus.COMPLETED
    store.save(run)
    inner = store.inner
    stored = inner.load("child-1")
    assert stored.vars["_meta"]["occurrence"]["trigger_envelope"]["payload"] == big
    assert stored.vars["_meta"]["creation_digest"] == "sha256:abc"
    # Non-identity private vars still offload.
    assert stored.vars["_temp"]["blob"] != big


def test_json_store_sweeps_stale_temp_files_on_open(tmp_path) -> None:
    runs = tmp_path / "runs"
    JsonFileRunStore(runs)
    stale = runs / "run_child-1.json.deadbeef.tmp"
    fresh = runs / "run_child-2.json.cafebabe.tmp"
    stale.write_text("{partial")
    fresh.write_text("{in flight")
    old = stale.stat().st_mtime - JsonFileRunStore.STALE_TEMP_AGE_S - 60
    os.utime(stale, (old, old))
    JsonFileRunStore(runs)
    assert not stale.exists()
    assert fresh.exists()  # may belong to a live writer
