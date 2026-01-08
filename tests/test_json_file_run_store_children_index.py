from __future__ import annotations

from abstractruntime import RunState, RunStatus
from abstractruntime.storage.json_files import JsonFileRunStore


def _new_run(
    *,
    workflow_id: str,
    parent_run_id: str | None = None,
    status: RunStatus = RunStatus.RUNNING,
) -> RunState:
    run = RunState.new(workflow_id=workflow_id, entry_node="start", parent_run_id=parent_run_id)
    run.status = status
    return run


def test_list_children_returns_children_and_filters_by_status(tmp_path):
    store = JsonFileRunStore(tmp_path)

    parent = _new_run(workflow_id="parent")
    child1 = _new_run(workflow_id="child1", parent_run_id=parent.run_id, status=RunStatus.RUNNING)
    child2 = _new_run(workflow_id="child2", parent_run_id=parent.run_id, status=RunStatus.WAITING)
    other = _new_run(workflow_id="other")

    store.save(parent)
    store.save(child1)
    store.save(child2)
    store.save(other)

    children = store.list_children(parent_run_id=parent.run_id)
    assert {r.run_id for r in children} == {child1.run_id, child2.run_id}

    waiting_children = store.list_children(parent_run_id=parent.run_id, status=RunStatus.WAITING)
    assert {r.run_id for r in waiting_children} == {child2.run_id}


def test_list_children_builds_index_once_and_updates_on_save(tmp_path):
    store = JsonFileRunStore(tmp_path)

    parent = _new_run(workflow_id="parent")
    child1 = _new_run(workflow_id="child1", parent_run_id=parent.run_id)
    store.save(parent)
    store.save(child1)

    call_count = {"iter_all_runs": 0}
    original_iter_all_runs = store._iter_all_runs

    def wrapped_iter_all_runs():
        call_count["iter_all_runs"] += 1
        return original_iter_all_runs()

    store._iter_all_runs = wrapped_iter_all_runs  # type: ignore[method-assign]

    children1 = store.list_children(parent_run_id=parent.run_id)
    children2 = store.list_children(parent_run_id=parent.run_id)
    assert {r.run_id for r in children1} == {child1.run_id}
    assert {r.run_id for r in children2} == {child1.run_id}
    assert call_count["iter_all_runs"] == 1

    child2 = _new_run(workflow_id="child2", parent_run_id=parent.run_id)
    store.save(child2)

    children3 = store.list_children(parent_run_id=parent.run_id)
    assert {r.run_id for r in children3} == {child1.run_id, child2.run_id}
    assert call_count["iter_all_runs"] == 1

