"""Tests for QueryableRunStore functionality.

Tests both InMemoryRunStore and JsonFileRunStore implementations.
"""

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from abstractruntime import RunState, RunStatus, WaitReason
from abstractruntime.core.models import WaitState
from abstractruntime.storage.in_memory import InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.base import QueryableRunStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run(
    *,
    workflow_id: str = "test_wf",
    status: RunStatus = RunStatus.RUNNING,
    wait_reason: WaitReason | None = None,
    wait_until: str | None = None,
    updated_at: str | None = None,
) -> RunState:
    """Helper to create test RunState objects."""
    run = RunState.new(workflow_id=workflow_id, entry_node="start")
    run.status = status
    if updated_at:
        run.updated_at = updated_at

    if wait_reason is not None:
        run.waiting = WaitState(
            reason=wait_reason,
            wait_key=f"key_{run.run_id[:8]}",
            until=wait_until,
        )

    return run


class TestInMemoryRunStoreQuery:
    """Tests for InMemoryRunStore query methods."""

    def test_implements_queryable_protocol(self):
        """InMemoryRunStore should implement QueryableRunStore protocol."""
        store = InMemoryRunStore()
        assert isinstance(store, QueryableRunStore)

    def test_list_runs_empty(self):
        """list_runs on empty store returns empty list."""
        store = InMemoryRunStore()
        result = store.list_runs()
        assert result == []

    def test_list_runs_all(self):
        """list_runs without filters returns all runs."""
        store = InMemoryRunStore()

        run1 = make_run(status=RunStatus.RUNNING)
        run2 = make_run(status=RunStatus.WAITING, wait_reason=WaitReason.USER)
        run3 = make_run(status=RunStatus.COMPLETED)

        store.save(run1)
        store.save(run2)
        store.save(run3)

        result = store.list_runs()
        assert len(result) == 3

    def test_list_runs_filter_by_status(self):
        """list_runs filters by status correctly."""
        store = InMemoryRunStore()

        run1 = make_run(status=RunStatus.RUNNING)
        run2 = make_run(status=RunStatus.WAITING, wait_reason=WaitReason.USER)
        run3 = make_run(status=RunStatus.COMPLETED)

        store.save(run1)
        store.save(run2)
        store.save(run3)

        waiting = store.list_runs(status=RunStatus.WAITING)
        assert len(waiting) == 1
        assert waiting[0].run_id == run2.run_id

        completed = store.list_runs(status=RunStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].run_id == run3.run_id

    def test_list_runs_filter_by_wait_reason(self):
        """list_runs filters by wait_reason correctly."""
        store = InMemoryRunStore()

        run1 = make_run(status=RunStatus.WAITING, wait_reason=WaitReason.USER)
        run2 = make_run(status=RunStatus.WAITING, wait_reason=WaitReason.UNTIL)
        run3 = make_run(status=RunStatus.WAITING, wait_reason=WaitReason.EVENT)

        store.save(run1)
        store.save(run2)
        store.save(run3)

        user_waits = store.list_runs(wait_reason=WaitReason.USER)
        assert len(user_waits) == 1
        assert user_waits[0].run_id == run1.run_id

        until_waits = store.list_runs(wait_reason=WaitReason.UNTIL)
        assert len(until_waits) == 1
        assert until_waits[0].run_id == run2.run_id

    def test_list_runs_filter_by_workflow_id(self):
        """list_runs filters by workflow_id correctly."""
        store = InMemoryRunStore()

        run1 = make_run(workflow_id="wf_a")
        run2 = make_run(workflow_id="wf_b")
        run3 = make_run(workflow_id="wf_a")

        store.save(run1)
        store.save(run2)
        store.save(run3)

        wf_a_runs = store.list_runs(workflow_id="wf_a")
        assert len(wf_a_runs) == 2
        assert all(r.workflow_id == "wf_a" for r in wf_a_runs)

    def test_list_runs_combined_filters(self):
        """list_runs applies multiple filters correctly."""
        store = InMemoryRunStore()

        run1 = make_run(workflow_id="wf_a", status=RunStatus.WAITING, wait_reason=WaitReason.USER)
        run2 = make_run(workflow_id="wf_a", status=RunStatus.WAITING, wait_reason=WaitReason.UNTIL)
        run3 = make_run(workflow_id="wf_b", status=RunStatus.WAITING, wait_reason=WaitReason.USER)

        store.save(run1)
        store.save(run2)
        store.save(run3)

        result = store.list_runs(workflow_id="wf_a", wait_reason=WaitReason.USER)
        assert len(result) == 1
        assert result[0].run_id == run1.run_id

    def test_list_runs_respects_limit(self):
        """list_runs respects the limit parameter."""
        store = InMemoryRunStore()

        for _ in range(10):
            store.save(make_run())

        result = store.list_runs(limit=3)
        assert len(result) == 3

    def test_list_runs_ordered_by_updated_at_desc(self):
        """list_runs returns results ordered by updated_at descending."""
        store = InMemoryRunStore()

        run1 = make_run(updated_at="2025-01-01T00:00:00Z")
        run2 = make_run(updated_at="2025-01-03T00:00:00Z")
        run3 = make_run(updated_at="2025-01-02T00:00:00Z")

        store.save(run1)
        store.save(run2)
        store.save(run3)

        result = store.list_runs()
        assert result[0].run_id == run2.run_id  # Most recent first
        assert result[1].run_id == run3.run_id
        assert result[2].run_id == run1.run_id

    def test_list_due_wait_until_empty(self):
        """list_due_wait_until on empty store returns empty list."""
        store = InMemoryRunStore()
        result = store.list_due_wait_until(now_iso=utc_now_iso())
        assert result == []

    def test_list_due_wait_until_finds_due_runs(self):
        """list_due_wait_until finds runs whose wait time has passed."""
        store = InMemoryRunStore()

        past_time = "2020-01-01T00:00:00Z"
        future_time = "2099-01-01T00:00:00Z"

        run_due = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until=past_time,
        )
        run_not_due = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until=future_time,
        )
        run_wrong_reason = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.USER,
        )
        run_not_waiting = make_run(status=RunStatus.RUNNING)

        store.save(run_due)
        store.save(run_not_due)
        store.save(run_wrong_reason)
        store.save(run_not_waiting)

        result = store.list_due_wait_until(now_iso=utc_now_iso())
        assert len(result) == 1
        assert result[0].run_id == run_due.run_id

    def test_list_due_wait_until_ordered_by_until_asc(self):
        """list_due_wait_until returns results ordered by until ascending."""
        store = InMemoryRunStore()

        run1 = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until="2020-01-03T00:00:00Z",
        )
        run2 = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until="2020-01-01T00:00:00Z",
        )
        run3 = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until="2020-01-02T00:00:00Z",
        )

        store.save(run1)
        store.save(run2)
        store.save(run3)

        result = store.list_due_wait_until(now_iso=utc_now_iso())
        assert result[0].run_id == run2.run_id  # Earliest first
        assert result[1].run_id == run3.run_id
        assert result[2].run_id == run1.run_id


class TestJsonFileRunStoreQuery:
    """Tests for JsonFileRunStore query methods."""

    def test_implements_queryable_protocol(self, tmp_path: Path):
        """JsonFileRunStore should implement QueryableRunStore protocol."""
        store = JsonFileRunStore(tmp_path)
        assert isinstance(store, QueryableRunStore)

    def test_list_runs_empty(self, tmp_path: Path):
        """list_runs on empty store returns empty list."""
        store = JsonFileRunStore(tmp_path)
        result = store.list_runs()
        assert result == []

    def test_list_runs_filter_by_status(self, tmp_path: Path):
        """list_runs filters by status correctly."""
        store = JsonFileRunStore(tmp_path)

        run1 = make_run(status=RunStatus.RUNNING)
        run2 = make_run(status=RunStatus.WAITING, wait_reason=WaitReason.USER)
        run3 = make_run(status=RunStatus.COMPLETED)

        store.save(run1)
        store.save(run2)
        store.save(run3)

        waiting = store.list_runs(status=RunStatus.WAITING)
        assert len(waiting) == 1
        assert waiting[0].run_id == run2.run_id

    def test_list_runs_filter_by_wait_reason(self, tmp_path: Path):
        """list_runs filters by wait_reason correctly."""
        store = JsonFileRunStore(tmp_path)

        run1 = make_run(status=RunStatus.WAITING, wait_reason=WaitReason.USER)
        run2 = make_run(status=RunStatus.WAITING, wait_reason=WaitReason.UNTIL)

        store.save(run1)
        store.save(run2)

        user_waits = store.list_runs(wait_reason=WaitReason.USER)
        assert len(user_waits) == 1
        assert user_waits[0].run_id == run1.run_id

    def test_list_runs_respects_limit(self, tmp_path: Path):
        """list_runs respects the limit parameter."""
        store = JsonFileRunStore(tmp_path)

        for _ in range(10):
            store.save(make_run())

        result = store.list_runs(limit=3)
        assert len(result) == 3

    def test_list_due_wait_until_finds_due_runs(self, tmp_path: Path):
        """list_due_wait_until finds runs whose wait time has passed."""
        store = JsonFileRunStore(tmp_path)

        past_time = "2020-01-01T00:00:00Z"
        future_time = "2099-01-01T00:00:00Z"

        run_due = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until=past_time,
        )
        run_not_due = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until=future_time,
        )

        store.save(run_due)
        store.save(run_not_due)

        result = store.list_due_wait_until(now_iso=utc_now_iso())
        assert len(result) == 1
        assert result[0].run_id == run_due.run_id

    def test_list_due_wait_until_ordered_by_until_asc(self, tmp_path: Path):
        """list_due_wait_until returns results ordered by until ascending."""
        store = JsonFileRunStore(tmp_path)

        run1 = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until="2020-01-03T00:00:00Z",
        )
        run2 = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until="2020-01-01T00:00:00Z",
        )

        store.save(run1)
        store.save(run2)

        result = store.list_due_wait_until(now_iso=utc_now_iso())
        assert result[0].run_id == run2.run_id  # Earliest first
        assert result[1].run_id == run1.run_id

    def test_persists_and_queries_across_instances(self, tmp_path: Path):
        """Queries work after creating a new store instance (persistence test)."""
        store1 = JsonFileRunStore(tmp_path)

        run = make_run(
            status=RunStatus.WAITING,
            wait_reason=WaitReason.UNTIL,
            wait_until="2020-01-01T00:00:00Z",
        )
        store1.save(run)

        # Create new instance pointing to same directory
        store2 = JsonFileRunStore(tmp_path)

        result = store2.list_due_wait_until(now_iso=utc_now_iso())
        assert len(result) == 1
        assert result[0].run_id == run.run_id
