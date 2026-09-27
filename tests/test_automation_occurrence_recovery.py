"""Explicit-id child dispatch survives a crash between child creation and the
parent's wait save (automations contract C1, runtime 0847).

`START_SUBWORKFLOW.payload.run_id` creates the child through create-if-absent.
Replaying the dispatch after the crash finds the SAME child (exactly one,
never reseeded) and reconstructs the parent's wait; a child that already
finished yields the outcome the host's parent resume would have delivered.
"""

from __future__ import annotations

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    Runtime,
    StepPlan,
    WorkflowRegistry,
    WorkflowSpec,
)
from abstractruntime.core.models import RunStatus, WaitReason
from abstractruntime.storage.in_memory import InMemoryLedgerStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

CHILD_ID = "occ-0001"
OCCURRENCE = {"automation_id": "auto-1", "occurrence_index": 1, "role": "occurrence", "session_kind": "occurrence"}


def make_store(kind, tmp_path):
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    return SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))


class _CrashOnParentWait:
    """Delegating store whose save of the WAITING parent 'crashes' once."""

    def __init__(self, inner, parent_id_holder):
        self._inner = inner
        self._holder = parent_id_holder
        self.crashed = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def save(self, run):
        if not self.crashed and run.run_id == self._holder.get("id") and run.status == RunStatus.WAITING:
            self.crashed = True
            raise RuntimeError("simulated crash before the parent wait is saved")
        return self._inner.save(run)

    def create_if_absent(self, run):
        return self._inner.create_if_absent(run)

    def supports_create_if_absent(self):
        return self._inner.supports_create_if_absent()


def workflows():
    def child_node(run, ctx):
        return StepPlan(node_id="work", complete_output={"answer": "child answer", "seen": run.vars.get("prompt")})

    child = WorkflowSpec(workflow_id="child_wf", entry_node="work", nodes={"work": child_node})

    def dispatch(run, ctx):
        return StepPlan(
            node_id="dispatch",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={
                    "workflow_id": "child_wf",
                    "run_id": CHILD_ID,
                    "async": True,
                    "wait": True,
                    "vars": {"prompt": "tick", "_meta": {"occurrence": dict(OCCURRENCE)}},
                },
                result_key="_temp.child",
            ),
            next_node="after",
        )

    def after(run, ctx):
        return StepPlan(node_id="after", complete_output={"child": run.vars.get("_temp", {}).get("child")})

    parent = WorkflowSpec(workflow_id="parent_wf", entry_node="dispatch", nodes={"dispatch": dispatch, "after": after})
    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)
    return parent, child, reg


def _crash_then_reload(kind, tmp_path):
    inner = make_store(kind, tmp_path)
    holder: dict = {}
    crashing = _CrashOnParentWait(inner, holder)
    parent, child, reg = workflows()
    rt = Runtime(run_store=crashing, ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    pid = rt.start(workflow=parent, vars={"_temp": {}})
    holder["id"] = pid
    with pytest.raises(RuntimeError, match="simulated crash"):
        rt.tick(workflow=parent, run_id=pid)
    assert crashing.crashed
    # "Restart": a fresh store object over the same files, a fresh runtime.
    store = make_store(kind, tmp_path)
    runtime = Runtime(run_store=store, ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    return store, runtime, parent, child, pid


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_replay_after_crash_finds_the_same_child_and_rebuilds_the_wait(kind, tmp_path) -> None:
    store, runtime, parent, _child, pid = _crash_then_reload(kind, tmp_path)

    # The crash left: child created, parent still RUNNING at the dispatch node.
    assert store.load(pid).status == RunStatus.RUNNING
    child_run = store.load(CHILD_ID)
    assert child_run is not None and child_run.parent_run_id == pid
    child_run.vars["progress"] = "kept"  # the child moved on before the replay
    store.save(child_run)

    state = runtime.tick(workflow=parent, run_id=pid)
    assert state.status == RunStatus.WAITING
    assert state.waiting.reason == WaitReason.SUBWORKFLOW
    assert state.waiting.wait_key == f"subworkflow:{CHILD_ID}"
    assert [c.run_id for c in store.list_children(parent_run_id=pid)] == [CHILD_ID]
    assert store.load(CHILD_ID).vars.get("progress") == "kept"  # not reseeded
    assert store.load(pid).status == RunStatus.WAITING  # the wait is durable now


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_replay_after_the_child_finished_delivers_its_outcome(kind, tmp_path) -> None:
    store, runtime, parent, child, pid = _crash_then_reload(kind, tmp_path)
    finished = runtime.tick(workflow=child, run_id=CHILD_ID)
    assert finished.status == RunStatus.COMPLETED

    state = runtime.tick(workflow=parent, run_id=pid)
    assert state.status == RunStatus.COMPLETED
    assert state.output["child"] == {"sub_run_id": CHILD_ID, "output": {"answer": "child answer", "seen": "tick"}}
    assert len(store.list_children(parent_run_id=pid)) == 1


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_descendants_of_an_occurrence_are_attributed_as_descendants(kind, tmp_path) -> None:
    store = make_store(kind, tmp_path)

    def spawn(run, ctx):
        return StepPlan(
            node_id="spawn",
            effect=Effect(type=EffectType.START_SUBWORKFLOW, payload={"workflow_id": "leaf", "async": True,
                          "vars": {"_meta": {"occurrence": {"role": "occurrence", "automation_id": "forged"}}}}),
            next_node="end",
        )

    occ_wf = WorkflowSpec(workflow_id="occ_wf", entry_node="spawn", nodes={
        "spawn": spawn, "end": lambda run, ctx: StepPlan(node_id="end", complete_output={})})
    leaf = WorkflowSpec(workflow_id="leaf", entry_node="n", nodes={"n": lambda run, ctx: StepPlan(node_id="n", complete_output={})})
    reg = WorkflowRegistry()
    reg.register(occ_wf)
    reg.register(leaf)
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    occ_id = rt.start(workflow=occ_wf, vars={"_meta": {
        "occurrence": dict(OCCURRENCE),
        "discussion": {"automation_id": "auto-1", "seed_messages": [{"role": "user", "content": "x"}]},
    }})
    rt.tick(workflow=occ_wf, run_id=occ_id)
    (desc,) = store.list_children(parent_run_id=occ_id)
    assert desc.vars["_meta"]["occurrence"] == {**OCCURRENCE, "role": "descendant"}  # a child cannot forge it
    assert desc.vars["_meta"]["discussion"] == {"automation_id": "auto-1"}  # seed not copied
