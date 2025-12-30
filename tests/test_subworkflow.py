"""Tests for subworkflow support (START_SUBWORKFLOW effect).

Tests cover:
- Sync mode: parent waits for child completion
- Sync mode: child waits, parent also waits
- Sync mode: child fails, parent receives error
- Async mode: parent continues immediately
- Nested subworkflows
- Error cases (missing registry, unknown workflow)
"""

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    Runtime,
    RunState,
    RunStatus,
    StepPlan,
    WaitReason,
    WorkflowSpec,
    InMemoryRunStore,
    InMemoryLedgerStore,
    WorkflowRegistry,
    create_scheduled_runtime,
)


# -----------------------------------------------------------------------------
# Helper Workflows
# -----------------------------------------------------------------------------


def make_simple_child_workflow(output_value: str = "child_done") -> WorkflowSpec:
    """A child workflow that completes immediately."""

    def child_node(run: RunState, ctx) -> StepPlan:
        input_val = run.vars.get("input", "default")
        return StepPlan(
            node_id="child",
            complete_output={"result": output_value, "received_input": input_val},
        )

    return WorkflowSpec(
        workflow_id="simple_child",
        entry_node="child",
        nodes={"child": child_node},
    )


def make_waiting_child_workflow() -> WorkflowSpec:
    """A child workflow that waits for user input."""

    def ask_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="ask",
            effect=Effect(
                type=EffectType.ASK_USER,
                payload={"prompt": "Child question?", "wait_key": "child_prompt"},
                result_key="child_answer",
            ),
            next_node="done",
        )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"child_answer": run.vars.get("child_answer")},
        )

    return WorkflowSpec(
        workflow_id="waiting_child",
        entry_node="ask",
        nodes={"ask": ask_node, "done": done_node},
    )


def make_failing_child_workflow() -> WorkflowSpec:
    """A child workflow that fails."""

    def fail_node(run: RunState, ctx) -> StepPlan:
        raise ValueError("Child workflow intentionally failed")

    return WorkflowSpec(
        workflow_id="failing_child",
        entry_node="fail",
        nodes={"fail": fail_node},
    )


def make_parent_workflow(child_workflow_id: str, is_async: bool = False) -> WorkflowSpec:
    """A parent workflow that starts a subworkflow."""

    def start_child(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="start_child",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={
                    "workflow_id": child_workflow_id,
                    "vars": {"input": run.vars.get("parent_input", "from_parent")},
                    "async": is_async,
                },
                result_key="child_result",
            ),
            next_node="after_child",
        )

    def after_child(run: RunState, ctx) -> StepPlan:
        child_result = run.vars.get("child_result", {})
        return StepPlan(
            node_id="after_child",
            complete_output={
                "parent_done": True,
                "child_result": child_result,
            },
        )

    return WorkflowSpec(
        workflow_id="parent_workflow",
        entry_node="start_child",
        nodes={"start_child": start_child, "after_child": after_child},
    )


def make_nested_parent_workflow() -> WorkflowSpec:
    """A parent that starts a child that starts a grandchild."""

    def start_middle(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="start_middle",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={"workflow_id": "middle_child"},
                result_key="middle_result",
            ),
            next_node="done",
        )

    def done(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"final": run.vars.get("middle_result")},
        )

    return WorkflowSpec(
        workflow_id="nested_parent",
        entry_node="start_middle",
        nodes={"start_middle": start_middle, "done": done},
    )


def make_middle_child_workflow() -> WorkflowSpec:
    """A middle workflow that starts a grandchild."""

    def start_grandchild(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="start_grandchild",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={"workflow_id": "simple_child"},
                result_key="grandchild_result",
            ),
            next_node="done",
        )

    def done(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"middle_done": True, "grandchild": run.vars.get("grandchild_result")},
        )

    return WorkflowSpec(
        workflow_id="middle_child",
        entry_node="start_grandchild",
        nodes={"start_grandchild": start_grandchild, "done": done},
    )


# -----------------------------------------------------------------------------
# Tests: Sync Mode
# -----------------------------------------------------------------------------


class TestSubworkflowSyncMode:
    """Tests for sync subworkflow execution."""

    def test_sync_child_completes(self):
        """Sync mode: child completes, parent receives output."""
        child = make_simple_child_workflow()
        parent = make_parent_workflow("simple_child")

        registry = WorkflowRegistry()
        registry.register(child)
        registry.register(parent)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=parent, vars={"parent_input": "hello"})
        state = runtime.tick(workflow=parent, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["parent_done"] is True
        assert state.output["child_result"]["output"]["result"] == "child_done"
        assert state.output["child_result"]["output"]["received_input"] == "hello"

    def test_sync_child_waits_parent_waits(self):
        """Sync mode: child waits, parent also waits."""
        child = make_waiting_child_workflow()
        parent = make_parent_workflow("waiting_child")

        registry = WorkflowRegistry()
        registry.register(child)
        registry.register(parent)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=parent)
        state = runtime.tick(workflow=parent, run_id=run_id)

        assert state.status == RunStatus.WAITING
        assert state.waiting.reason == WaitReason.SUBWORKFLOW
        assert "sub_run_id" in state.waiting.details
        assert state.waiting.details["sub_workflow_id"] == "waiting_child"
        assert state.waiting.details["sub_waiting"]["reason"] == "user"

    def test_sync_child_fails_parent_fails(self):
        """Sync mode: child fails, parent receives error."""
        child = make_failing_child_workflow()
        parent = make_parent_workflow("failing_child")

        registry = WorkflowRegistry()
        registry.register(child)
        registry.register(parent)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=parent)
        state = runtime.tick(workflow=parent, run_id=run_id)

        assert state.status == RunStatus.FAILED
        assert "failing_child" in state.error
        assert "intentionally failed" in state.error


# -----------------------------------------------------------------------------
# Tests: Async Mode
# -----------------------------------------------------------------------------


class TestSubworkflowAsyncMode:
    """Tests for async subworkflow execution."""

    def test_async_parent_continues_immediately(self):
        """Async mode: parent continues without waiting for child."""
        child = make_waiting_child_workflow()  # Would wait in sync mode
        parent = make_parent_workflow("waiting_child", is_async=True)

        registry = WorkflowRegistry()
        registry.register(child)
        registry.register(parent)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=parent)
        state = runtime.tick(workflow=parent, run_id=run_id)

        # Parent should complete even though child hasn't been ticked
        assert state.status == RunStatus.COMPLETED
        assert state.output["parent_done"] is True
        assert state.output["child_result"]["async"] is True
        assert "sub_run_id" in state.output["child_result"]

        # In async mode, child is started but not ticked - it's still RUNNING
        # The caller is responsible for driving the child workflow
        sub_run_id = state.output["child_result"]["sub_run_id"]
        child_state = runtime.get_state(sub_run_id)
        assert child_state.status == RunStatus.RUNNING

        # Now tick the child - it should wait
        child_state = runtime.tick(workflow=child, run_id=sub_run_id)
        assert child_state.status == RunStatus.WAITING


# -----------------------------------------------------------------------------
# Tests: Nested Subworkflows
# -----------------------------------------------------------------------------


class TestNestedSubworkflows:
    """Tests for nested subworkflow execution."""

    def test_nested_three_levels(self):
        """Parent -> Middle -> Grandchild all complete."""
        grandchild = make_simple_child_workflow()
        middle = make_middle_child_workflow()
        parent = make_nested_parent_workflow()

        registry = WorkflowRegistry()
        registry.register(grandchild)
        registry.register(middle)
        registry.register(parent)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=parent)
        state = runtime.tick(workflow=parent, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        # Verify the nested structure
        middle_result = state.output["final"]["output"]
        assert middle_result["middle_done"] is True
        grandchild_result = middle_result["grandchild"]["output"]
        assert grandchild_result["result"] == "child_done"


# -----------------------------------------------------------------------------
# Tests: Error Cases
# -----------------------------------------------------------------------------


class TestSubworkflowErrors:
    """Tests for subworkflow error handling."""

    def test_missing_registry_fails(self):
        """Subworkflow fails if no registry is set."""
        parent = make_parent_workflow("simple_child")

        # No registry set
        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
        )

        run_id = runtime.start(workflow=parent)
        state = runtime.tick(workflow=parent, run_id=run_id)

        assert state.status == RunStatus.FAILED
        assert "workflow_registry" in state.error

    def test_unknown_workflow_fails(self):
        """Subworkflow fails if workflow_id not in registry."""
        parent = make_parent_workflow("nonexistent_workflow")

        registry = WorkflowRegistry()
        registry.register(parent)
        # Note: child workflow NOT registered

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=parent)
        state = runtime.tick(workflow=parent, run_id=run_id)

        assert state.status == RunStatus.FAILED
        assert "nonexistent_workflow" in state.error
        assert "not found" in state.error

    def test_missing_workflow_id_fails(self):
        """Subworkflow fails if payload.workflow_id is missing."""

        def bad_parent_node(run, ctx):
            return StepPlan(
                node_id="bad",
                effect=Effect(
                    type=EffectType.START_SUBWORKFLOW,
                    payload={},  # Missing workflow_id
                ),
                next_node="done",
            )

        def done(run, ctx):
            return StepPlan(node_id="done", complete_output={})

        bad_parent = WorkflowSpec(
            workflow_id="bad_parent",
            entry_node="bad",
            nodes={"bad": bad_parent_node, "done": done},
        )

        registry = WorkflowRegistry()
        registry.register(bad_parent)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=bad_parent)
        state = runtime.tick(workflow=bad_parent, run_id=run_id)

        assert state.status == RunStatus.FAILED
        assert "workflow_id" in state.error


# -----------------------------------------------------------------------------
# Tests: Integration with ScheduledRuntime
# -----------------------------------------------------------------------------


class TestSubworkflowWithScheduledRuntime:
    """Tests for subworkflows with ScheduledRuntime."""

    def test_scheduled_runtime_subworkflow(self):
        """ScheduledRuntime supports subworkflows."""
        child = make_simple_child_workflow()
        parent = make_parent_workflow("simple_child")

        sr = create_scheduled_runtime(
            workflows=[child, parent],
            auto_start=False,
        )

        run_id, state = sr.run(parent)

        assert state.status == RunStatus.COMPLETED
        assert state.output["child_result"]["output"]["result"] == "child_done"

    def test_scheduled_runtime_subworkflow_with_respond(self):
        """ScheduledRuntime can resume waiting subworkflows."""
        child = make_waiting_child_workflow()
        parent = make_parent_workflow("waiting_child")

        sr = create_scheduled_runtime(
            workflows=[child, parent],
            auto_start=False,
        )

        run_id, state = sr.run(parent)

        assert state.status == RunStatus.WAITING
        assert state.waiting.reason == WaitReason.SUBWORKFLOW

        # Get the child run_id and resume it
        sub_run_id = state.waiting.details["sub_run_id"]

        # Resume the child workflow
        child_state = sr.runtime.resume(
            workflow=child,
            run_id=sub_run_id,
            wait_key="child_prompt",
            payload={"text": "child_answer"},
        )
        assert child_state.status == RunStatus.COMPLETED

        # Now resume the parent
        parent_state = sr.runtime.resume(
            workflow=parent,
            run_id=run_id,
            wait_key=state.waiting.wait_key,
            payload=child_state.output,
        )
        assert parent_state.status == RunStatus.COMPLETED
        assert parent_state.output["child_result"]["child_answer"] == {"text": "child_answer"}


# -----------------------------------------------------------------------------
# Tests: Parent-Child Tracking
# -----------------------------------------------------------------------------


class TestSubworkflowParentTracking:
    """Tests for parent-child relationship tracking."""

    def test_child_has_parent_run_id(self):
        """Child workflow has parent_run_id set."""
        child = make_simple_child_workflow()
        parent = make_parent_workflow("simple_child")

        registry = WorkflowRegistry()
        registry.register(child)
        registry.register(parent)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        parent_run_id = runtime.start(workflow=parent)
        state = runtime.tick(workflow=parent, run_id=parent_run_id)

        assert state.status == RunStatus.COMPLETED
        child_run_id = state.output["child_result"]["sub_run_id"]

        # Verify child has parent_run_id
        child_state = runtime.get_state(child_run_id)
        assert child_state.parent_run_id == parent_run_id

    def test_list_children(self):
        """Can list child runs of a parent."""
        child = make_simple_child_workflow()
        parent = make_parent_workflow("simple_child")

        registry = WorkflowRegistry()
        registry.register(child)
        registry.register(parent)

        run_store = InMemoryRunStore()
        runtime = Runtime(
            run_store=run_store,
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        parent_run_id = runtime.start(workflow=parent)
        runtime.tick(workflow=parent, run_id=parent_run_id)

        # List children
        children = run_store.list_children(parent_run_id=parent_run_id)
        assert len(children) == 1
        assert children[0].workflow_id == "simple_child"

    def test_auto_resume_parent_on_child_complete(self):
        """Scheduler can auto-resume parent when child completes."""
        child = make_waiting_child_workflow()
        parent = make_parent_workflow("waiting_child")

        sr = create_scheduled_runtime(
            workflows=[child, parent],
            auto_start=False,
        )

        # Start parent - it will wait for child
        parent_run_id, parent_state = sr.run(parent)
        assert parent_state.status == RunStatus.WAITING
        assert parent_state.waiting.reason == WaitReason.SUBWORKFLOW

        sub_run_id = parent_state.waiting.details["sub_run_id"]

        # Resume child to completion
        child_state = sr.runtime.resume(
            workflow=child,
            run_id=sub_run_id,
            wait_key="child_prompt",
            payload={"answer": "yes"},
        )
        assert child_state.status == RunStatus.COMPLETED

        # Use scheduler to auto-resume parent
        resumed_parent = sr.resume_subworkflow_parent(
            child_run_id=sub_run_id,
            child_output=child_state.output,
        )

        assert resumed_parent is not None
        assert resumed_parent.status == RunStatus.COMPLETED
        assert resumed_parent.output["parent_done"] is True


# -----------------------------------------------------------------------------
# Tests: Cancellation
# -----------------------------------------------------------------------------


class TestSubworkflowCancellation:
    """Tests for run cancellation."""

    def test_cancel_running_run(self):
        """Can cancel a running run."""
        child = make_waiting_child_workflow()

        registry = WorkflowRegistry()
        registry.register(child)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=child)
        # Don't tick - leave it RUNNING

        state = runtime.cancel_run(run_id, reason="Test cancellation")

        assert state.status == RunStatus.CANCELLED
        assert state.error == "Test cancellation"

    def test_tick_is_noop_for_cancelled_run(self):
        """Regression: cancelled runs must never progress when tick() is called."""
        child = make_simple_child_workflow()

        registry = WorkflowRegistry()
        registry.register(child)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=child)
        runtime.cancel_run(run_id, reason="Stop now")

        state = runtime.tick(workflow=child, run_id=run_id)
        assert state.status == RunStatus.CANCELLED
        assert state.error == "Stop now"

    def test_cancel_waiting_run(self):
        """Can cancel a waiting run."""
        child = make_waiting_child_workflow()

        registry = WorkflowRegistry()
        registry.register(child)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=child)
        runtime.tick(workflow=child, run_id=run_id)  # Now WAITING

        state = runtime.cancel_run(run_id)

        assert state.status == RunStatus.CANCELLED
        assert state.waiting is None  # Cleared

    def test_cancel_completed_run_noop(self):
        """Cancelling a completed run is a no-op."""
        child = make_simple_child_workflow()

        registry = WorkflowRegistry()
        registry.register(child)

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflow_registry=registry,
        )

        run_id = runtime.start(workflow=child)
        runtime.tick(workflow=child, run_id=run_id)  # Now COMPLETED

        state = runtime.cancel_run(run_id)

        assert state.status == RunStatus.COMPLETED  # Unchanged

    def test_cancel_with_children(self):
        """Cascading cancellation cancels parent and all children."""
        child = make_waiting_child_workflow()
        parent = make_parent_workflow("waiting_child")

        sr = create_scheduled_runtime(
            workflows=[child, parent],
            auto_start=False,
        )

        # Start parent - it will wait for child
        parent_run_id, parent_state = sr.run(parent)
        assert parent_state.status == RunStatus.WAITING

        sub_run_id = parent_state.waiting.details["sub_run_id"]

        # Cancel parent with children
        cancelled = sr.cancel_with_children(parent_run_id, reason="Shutdown")

        assert len(cancelled) == 2  # Parent and child

        # Verify both are cancelled
        parent_state = sr.get_state(parent_run_id)
        child_state = sr.get_state(sub_run_id)

        assert parent_state.status == RunStatus.CANCELLED
        assert child_state.status == RunStatus.CANCELLED

    def test_cancel_with_nested_children(self):
        """Cascading cancellation works with nested subworkflows."""
        grandchild = make_waiting_child_workflow()
        middle = make_parent_workflow("waiting_child")
        parent = make_parent_workflow("parent_workflow")

        # Rename middle workflow
        middle = WorkflowSpec(
            workflow_id="parent_workflow",
            entry_node=middle.entry_node,
            nodes=middle.nodes,
        )

        # Create a top-level parent that calls middle
        def top_start(run, ctx):
            return StepPlan(
                node_id="top_start",
                effect=Effect(
                    type=EffectType.START_SUBWORKFLOW,
                    payload={"workflow_id": "parent_workflow"},
                    result_key="result",
                ),
                next_node="top_done",
            )

        def top_done(run, ctx):
            return StepPlan(node_id="top_done", complete_output={"done": True})

        top = WorkflowSpec(
            workflow_id="top_workflow",
            entry_node="top_start",
            nodes={"top_start": top_start, "top_done": top_done},
        )

        sr = create_scheduled_runtime(
            workflows=[grandchild, middle, top],
            auto_start=False,
        )

        # Start top - it will create middle, which creates grandchild
        top_run_id, top_state = sr.run(top)
        assert top_state.status == RunStatus.WAITING

        # Cancel from top
        cancelled = sr.cancel_with_children(top_run_id, reason="Full shutdown")

        # Should cancel top, middle, and grandchild
        assert len(cancelled) == 3

        # All should be cancelled
        for run in cancelled:
            assert run.status == RunStatus.CANCELLED
