"""Tests for the Scheduler and related components.

Tests cover:
- WorkflowRegistry
- Scheduler (polling, resume_event, start/stop)
- ScheduledRuntime convenience wrapper
- create_scheduled_runtime factory
"""

import time
import threading
from datetime import datetime, timezone, timedelta

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
    Scheduler,
    SchedulerStats,
    WorkflowRegistry,
    ScheduledRuntime,
    create_scheduled_runtime,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def future_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def past_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


# -----------------------------------------------------------------------------
# Test Workflows
# -----------------------------------------------------------------------------


def make_wait_until_workflow(wait_seconds: float = 0.1) -> WorkflowSpec:
    """Create a workflow that waits until a time, then completes."""

    def start_node(run: RunState, ctx) -> StepPlan:
        until_time = future_iso(wait_seconds)
        return StepPlan(
            node_id="start",
            effect=Effect(
                type=EffectType.WAIT_UNTIL,
                payload={"until": until_time},
                result_key="wait_result",
            ),
            next_node="done",
        )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"status": "completed", "waited": True},
        )

    return WorkflowSpec(
        workflow_id="wait_until_wf",
        entry_node="start",
        nodes={"start": start_node, "done": done_node},
    )


def make_wait_event_workflow() -> WorkflowSpec:
    """Create a workflow that waits for an event, then completes."""

    def start_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="start",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={"wait_key": "my_event"},
                result_key="event_data",
            ),
            next_node="done",
        )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"status": "completed", "event_data": run.vars.get("event_data")},
        )

    return WorkflowSpec(
        workflow_id="wait_event_wf",
        entry_node="start",
        nodes={"start": start_node, "done": done_node},
    )


def make_ask_user_workflow() -> WorkflowSpec:
    """Create a workflow that asks user, then completes."""

    def start_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="start",
            effect=Effect(
                type=EffectType.ASK_USER,
                payload={"prompt": "What is your name?", "wait_key": "user_prompt"},
                result_key="user_response",
            ),
            next_node="done",
        )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"greeting": f"Hello, {run.vars.get('user_response', {}).get('text', 'unknown')}!"},
        )

    return WorkflowSpec(
        workflow_id="ask_user_wf",
        entry_node="start",
        nodes={"start": start_node, "done": done_node},
    )


# -----------------------------------------------------------------------------
# WorkflowRegistry Tests
# -----------------------------------------------------------------------------


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry."""

    def test_register_and_get(self):
        """Can register and retrieve a workflow."""
        registry = WorkflowRegistry()
        wf = make_wait_event_workflow()

        registry.register(wf)

        assert registry.get(wf.workflow_id) is wf
        assert wf.workflow_id in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self):
        """Registering duplicate workflow_id raises ValueError."""
        registry = WorkflowRegistry()
        wf = make_wait_event_workflow()

        registry.register(wf)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(wf)

    def test_unregister(self):
        """Can unregister a workflow."""
        registry = WorkflowRegistry()
        wf = make_wait_event_workflow()

        registry.register(wf)
        registry.unregister(wf.workflow_id)

        assert registry.get(wf.workflow_id) is None
        assert wf.workflow_id not in registry

    def test_unregister_unknown_raises(self):
        """Unregistering unknown workflow raises KeyError."""
        registry = WorkflowRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("unknown")

    def test_get_or_raise(self):
        """get_or_raise returns workflow or raises KeyError."""
        registry = WorkflowRegistry()
        wf = make_wait_event_workflow()

        registry.register(wf)

        assert registry.get_or_raise(wf.workflow_id) is wf

        with pytest.raises(KeyError, match="not registered"):
            registry.get_or_raise("unknown")

    def test_list_ids(self):
        """list_ids returns all registered workflow IDs."""
        registry = WorkflowRegistry()
        wf1 = make_wait_event_workflow()
        wf2 = make_ask_user_workflow()

        registry.register(wf1)
        registry.register(wf2)

        ids = registry.list_ids()
        assert set(ids) == {wf1.workflow_id, wf2.workflow_id}


# -----------------------------------------------------------------------------
# Scheduler Tests
# -----------------------------------------------------------------------------


class TestScheduler:
    """Tests for Scheduler."""

    def test_requires_queryable_run_store(self):
        """Scheduler raises TypeError if run_store is not QueryableRunStore."""

        class NotQueryableStore:
            def save(self, run): pass
            def load(self, run_id): return None

        runtime = Runtime(
            run_store=NotQueryableStore(),  # type: ignore
            ledger_store=InMemoryLedgerStore(),
        )
        registry = WorkflowRegistry()

        with pytest.raises(TypeError, match="QueryableRunStore"):
            Scheduler(runtime=runtime, registry=registry)

    def test_start_and_stop(self):
        """Scheduler can start and stop cleanly."""
        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
        )
        registry = WorkflowRegistry()
        scheduler = Scheduler(runtime=runtime, registry=registry, poll_interval_s=0.1)

        assert not scheduler.is_running

        scheduler.start()
        assert scheduler.is_running

        scheduler.stop()
        assert not scheduler.is_running

    def test_start_twice_raises(self):
        """Starting scheduler twice raises RuntimeError."""
        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
        )
        registry = WorkflowRegistry()
        scheduler = Scheduler(runtime=runtime, registry=registry)

        scheduler.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                scheduler.start()
        finally:
            scheduler.stop()

    def test_poll_once_resumes_due_wait_until(self):
        """poll_once resumes runs whose wait_until time has passed."""
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        # Create workflow that waits for a short future time
        # We'll manually create a waiting run with a past time to simulate
        # a run that was waiting and is now due
        def start_node(run, ctx):
            return StepPlan(
                node_id="start",
                effect=Effect(
                    type=EffectType.WAIT_UNTIL,
                    payload={"until": future_iso(60)},  # 60 seconds in future
                ),
                next_node="done",
            )

        def done_node(run, ctx):
            return StepPlan(node_id="done", complete_output={"done": True})

        wf = WorkflowSpec(
            workflow_id="wait_wf",
            entry_node="start",
            nodes={"start": start_node, "done": done_node},
        )

        registry = WorkflowRegistry()
        registry.register(wf)

        scheduler = Scheduler(runtime=runtime, registry=registry)

        # Start a run and tick to waiting state
        run_id = runtime.start(workflow=wf)
        state = runtime.tick(workflow=wf, run_id=run_id)
        assert state.status == RunStatus.WAITING

        # Manually modify the waiting.until to be in the past
        # This simulates time passing
        state.waiting.until = past_iso(10)
        run_store.save(state)

        # Poll once should resume it
        resumed_count = scheduler.poll_once()
        assert resumed_count == 1

        # Check it's completed
        state = runtime.get_state(run_id)
        assert state.status == RunStatus.COMPLETED

    def test_resume_event(self):
        """resume_event resumes a run waiting for an event."""
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        wf = make_wait_event_workflow()
        registry = WorkflowRegistry()
        registry.register(wf)

        scheduler = Scheduler(runtime=runtime, registry=registry)

        # Start and tick to waiting
        run_id = runtime.start(workflow=wf)
        state = runtime.tick(workflow=wf, run_id=run_id)
        assert state.status == RunStatus.WAITING
        assert state.waiting.wait_key == "my_event"

        # Resume via scheduler
        state = scheduler.resume_event(
            run_id=run_id,
            wait_key="my_event",
            payload={"value": 42},
        )

        assert state.status == RunStatus.COMPLETED
        assert state.output["event_data"] == {"value": 42}

    def test_resume_event_wrong_wait_key_raises(self):
        """resume_event with wrong wait_key raises ValueError."""
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        wf = make_wait_event_workflow()
        registry = WorkflowRegistry()
        registry.register(wf)

        scheduler = Scheduler(runtime=runtime, registry=registry)

        run_id = runtime.start(workflow=wf)
        runtime.tick(workflow=wf, run_id=run_id)

        with pytest.raises(ValueError, match="wait_key mismatch"):
            scheduler.resume_event(
                run_id=run_id,
                wait_key="wrong_key",
                payload={},
            )

    def test_resume_event_not_waiting_raises(self):
        """resume_event on non-waiting run raises ValueError."""
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        # Workflow that completes immediately
        def start_node(run, ctx):
            return StepPlan(node_id="start", complete_output={"done": True})

        wf = WorkflowSpec(
            workflow_id="immediate_wf",
            entry_node="start",
            nodes={"start": start_node},
        )

        registry = WorkflowRegistry()
        registry.register(wf)

        scheduler = Scheduler(runtime=runtime, registry=registry)

        run_id = runtime.start(workflow=wf)
        runtime.tick(workflow=wf, run_id=run_id)

        with pytest.raises(ValueError, match="not waiting"):
            scheduler.resume_event(run_id=run_id, wait_key="x", payload={})

    def test_find_waiting_runs(self):
        """find_waiting_runs returns waiting runs."""
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        wf = make_wait_event_workflow()
        registry = WorkflowRegistry()
        registry.register(wf)

        scheduler = Scheduler(runtime=runtime, registry=registry)

        # Start two runs
        run_id1 = runtime.start(workflow=wf)
        run_id2 = runtime.start(workflow=wf)
        runtime.tick(workflow=wf, run_id=run_id1)
        runtime.tick(workflow=wf, run_id=run_id2)

        waiting = scheduler.find_waiting_runs()
        assert len(waiting) == 2

        waiting_event = scheduler.find_waiting_runs(wait_reason=WaitReason.EVENT)
        assert len(waiting_event) == 2

        waiting_user = scheduler.find_waiting_runs(wait_reason=WaitReason.USER)
        assert len(waiting_user) == 0

    def test_stats_tracking(self):
        """Scheduler tracks statistics."""
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        registry = WorkflowRegistry()
        scheduler = Scheduler(runtime=runtime, registry=registry)

        assert scheduler.stats.poll_cycles == 0
        assert scheduler.stats.runs_resumed == 0

        scheduler.poll_once()

        assert scheduler.stats.poll_cycles == 1
        assert scheduler.stats.last_poll_at is not None

    def test_background_polling_resumes_wait_until(self):
        """Background scheduler automatically resumes wait_until runs."""
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        # Workflow that waits 0.1 seconds
        wf = make_wait_until_workflow(wait_seconds=0.1)
        registry = WorkflowRegistry()
        registry.register(wf)

        scheduler = Scheduler(runtime=runtime, registry=registry, poll_interval_s=0.05)

        # Start run and tick to waiting
        run_id = runtime.start(workflow=wf)
        state = runtime.tick(workflow=wf, run_id=run_id)
        assert state.status == RunStatus.WAITING

        # Start scheduler
        scheduler.start()
        try:
            # Wait for the run to be resumed (should happen within ~0.2s)
            for _ in range(20):  # Max 1 second
                time.sleep(0.05)
                state = runtime.get_state(run_id)
                if state.status == RunStatus.COMPLETED:
                    break

            assert state.status == RunStatus.COMPLETED
            assert scheduler.stats.runs_resumed >= 1
        finally:
            scheduler.stop()

    def test_callbacks_on_resume(self):
        """Callbacks are called when runs are resumed."""
        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()
        runtime = Runtime(run_store=run_store, ledger_store=ledger_store)

        # Track callbacks
        resumed_runs = []
        failed_runs = []

        def on_resumed(state):
            resumed_runs.append(state.run_id)

        def on_failed(run_id, error):
            failed_runs.append((run_id, str(error)))

        # Workflow that waits for future time
        def start_node(run, ctx):
            return StepPlan(
                node_id="start",
                effect=Effect(
                    type=EffectType.WAIT_UNTIL,
                    payload={"until": future_iso(60)},  # Future time
                ),
                next_node="done",
            )

        def done_node(run, ctx):
            return StepPlan(node_id="done", complete_output={"done": True})

        wf = WorkflowSpec(
            workflow_id="callback_wf",
            entry_node="start",
            nodes={"start": start_node, "done": done_node},
        )

        registry = WorkflowRegistry()
        registry.register(wf)

        scheduler = Scheduler(
            runtime=runtime,
            registry=registry,
            on_run_resumed=on_resumed,
            on_run_failed=on_failed,
        )

        run_id = runtime.start(workflow=wf)
        state = runtime.tick(workflow=wf, run_id=run_id)
        assert state.status == RunStatus.WAITING

        # Manually modify the waiting.until to be in the past
        state.waiting.until = past_iso(10)
        run_store.save(state)

        scheduler.poll_once()

        assert run_id in resumed_runs
        assert len(failed_runs) == 0


# -----------------------------------------------------------------------------
# ScheduledRuntime Tests
# -----------------------------------------------------------------------------


class TestScheduledRuntime:
    """Tests for ScheduledRuntime convenience wrapper."""

    def test_create_scheduled_runtime(self):
        """create_scheduled_runtime creates a working ScheduledRuntime."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[wf],
            auto_start=False,
        )

        assert not sr.is_running
        assert wf.workflow_id in sr.registry

    def test_create_scheduled_runtime_auto_start(self):
        """create_scheduled_runtime with auto_start=True starts scheduler."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[wf],
            auto_start=True,
        )

        try:
            assert sr.is_running
        finally:
            sr.stop()

    def test_scheduled_runtime_start_and_tick(self):
        """ScheduledRuntime can start and tick workflows."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[wf],
            auto_start=False,
        )

        run_id = sr.start(workflow=wf)
        state = sr.tick(run_id)  # Simplified: no workflow needed

        assert state.status == RunStatus.WAITING

    def test_scheduled_runtime_resume_event(self):
        """ScheduledRuntime.resume_event works."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[wf],
            auto_start=False,
        )

        run_id = sr.start(workflow=wf)
        sr.tick(run_id)

        state = sr.resume_event(
            run_id=run_id,
            wait_key="my_event",
            payload={"data": "test"},
        )

        assert state.status == RunStatus.COMPLETED
        assert state.output["event_data"] == {"data": "test"}

    def test_scheduled_runtime_auto_registers_workflow(self):
        """ScheduledRuntime.start auto-registers unknown workflows."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[],  # No pre-registered workflows
        )

        # Should auto-register
        run_id = sr.start(workflow=wf)
        assert wf.workflow_id in sr.registry

    def test_scheduled_runtime_find_waiting_runs(self):
        """ScheduledRuntime.find_waiting_runs works."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[wf],
            auto_start=False,
        )

        run_id = sr.start(workflow=wf)
        sr.tick(run_id)

        waiting = sr.find_waiting_runs()
        assert len(waiting) == 1
        assert waiting[0].run_id == run_id

    def test_scheduled_runtime_get_state_and_ledger(self):
        """ScheduledRuntime.get_state and get_ledger work."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            workflows=[wf],
            auto_start=False,
        )

        run_id = sr.start(workflow=wf)
        sr.tick(run_id)

        state = sr.get_state(run_id)
        assert state.run_id == run_id

        ledger = sr.get_ledger(run_id)
        assert len(ledger) >= 1

    def test_scheduled_runtime_run_method(self):
        """ScheduledRuntime.run() does start + tick in one call."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(auto_start=False)

        run_id, state = sr.run(wf)

        assert run_id is not None
        assert state.status == RunStatus.WAITING
        assert wf.workflow_id in sr.registry

    def test_scheduled_runtime_respond_method(self):
        """ScheduledRuntime.respond() resumes with auto wait_key lookup."""
        wf = make_wait_event_workflow()

        sr = create_scheduled_runtime(auto_start=False)

        run_id, state = sr.run(wf)
        assert state.status == RunStatus.WAITING

        # respond() doesn't need wait_key - it finds it automatically
        state = sr.respond(run_id, {"value": 42})

        assert state.status == RunStatus.COMPLETED
        assert state.output["event_data"] == {"value": 42}

    def test_scheduled_runtime_zero_config(self):
        """ScheduledRuntime works with zero config (all defaults)."""
        wf = make_wait_event_workflow()

        # Zero config - just works
        sr = create_scheduled_runtime()

        try:
            assert sr.is_running  # auto_start=True by default

            run_id, state = sr.run(wf)
            assert state.status == RunStatus.WAITING

            state = sr.respond(run_id, {"answer": "yes"})
            assert state.status == RunStatus.COMPLETED
        finally:
            sr.stop()
