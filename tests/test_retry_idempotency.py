"""Tests for effect retries and idempotency.

Tests cover:
- Retry logic with configurable max attempts
- Exponential backoff between retries
- Idempotency: skip re-execution on restart
- Policy configuration per effect type
- Ledger records attempt numbers
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from abstractruntime import (
    Runtime,
    RunState,
    RunStatus,
    StepPlan,
    Effect,
    EffectType,
    WorkflowSpec,
    InMemoryRunStore,
    InMemoryLedgerStore,
    JsonlLedgerStore,
    DefaultEffectPolicy,
    RetryPolicy,
    NoRetryPolicy,
    compute_idempotency_key,
    create_scheduled_runtime,
)
from abstractruntime.core.runtime import EffectOutcome


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


class FailingEffectHandler:
    """Effect handler that fails a configurable number of times."""

    def __init__(self, fail_count: int = 0):
        self.fail_count = fail_count
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, run: RunState, effect: Effect) -> EffectOutcome:
        self.call_count += 1
        self.calls.append({"run_id": run.run_id, "payload": effect.payload})

        if self.call_count <= self.fail_count:
            return EffectOutcome.failed(f"Simulated failure {self.call_count}")

        return EffectOutcome.completed({"success": True, "attempt": self.call_count})


def make_effect_workflow(effect_type: EffectType = EffectType.LLM_CALL) -> WorkflowSpec:
    """Create a simple workflow that executes one effect."""

    def start_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="start",
            effect=Effect(
                type=effect_type,
                payload={"prompt": "test"},
                result_key="result",
            ),
            next_node="done",
        )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="done",
            complete_output={"result": run.vars.get("result")},
        )

    return WorkflowSpec(
        workflow_id="effect_workflow",
        entry_node="start",
        nodes={"start": start_node, "done": done_node},
    )


# -----------------------------------------------------------------------------
# Tests: Retry Logic
# -----------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for effect retry behavior."""

    def test_no_retry_by_default(self):
        """Default policy does not retry."""
        handler = FailingEffectHandler(fail_count=1)
        workflow = make_effect_workflow()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: handler},
        )

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.FAILED
        assert handler.call_count == 1  # No retry

    def test_handler_exception_does_not_double_invoke(self):
        """Effect handler exceptions should not trigger a second invocation via compatibility fallbacks."""
        call_count = 0

        def boom(run: RunState, effect: Effect, default_next_node) -> EffectOutcome:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        workflow = make_effect_workflow()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: boom},
        )

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.FAILED
        assert call_count == 1

    def test_retry_with_policy(self):
        """RetryPolicy enables retries."""
        handler = FailingEffectHandler(fail_count=2)  # Fail twice, succeed on third
        workflow = make_effect_workflow()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: handler},
            effect_policy=RetryPolicy(llm_max_attempts=3),
        )

        run_id = runtime.start(workflow=workflow)

        # Patch sleep to avoid actual delays
        with patch("time.sleep"):
            state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert handler.call_count == 3  # Two failures + one success

    def test_retry_exhausted(self):
        """Fails after max attempts exhausted."""
        handler = FailingEffectHandler(fail_count=10)  # Always fail
        workflow = make_effect_workflow()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: handler},
            effect_policy=RetryPolicy(llm_max_attempts=3),
        )

        run_id = runtime.start(workflow=workflow)

        with patch("time.sleep"):
            state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.FAILED
        assert "after 3 attempts" in state.error
        assert handler.call_count == 3

    def test_different_retry_per_effect_type(self):
        """Different effect types can have different retry counts."""
        policy = RetryPolicy(llm_max_attempts=3, tool_max_attempts=2)

        assert policy.max_attempts(Effect(type=EffectType.LLM_CALL, payload={})) == 3
        assert policy.max_attempts(Effect(type=EffectType.TOOL_CALLS, payload={})) == 2
        assert policy.max_attempts(Effect(type=EffectType.WAIT_EVENT, payload={})) == 1


class TestBackoff:
    """Tests for backoff timing."""

    def test_exponential_backoff(self):
        """Backoff increases exponentially."""
        policy = DefaultEffectPolicy(
            default_max_attempts=5,
            default_backoff_base=1.0,
            default_backoff_max=60.0,
        )
        effect = Effect(type=EffectType.LLM_CALL, payload={})

        assert policy.backoff_seconds(effect=effect, attempt=1) == 1.0
        assert policy.backoff_seconds(effect=effect, attempt=2) == 2.0
        assert policy.backoff_seconds(effect=effect, attempt=3) == 4.0
        assert policy.backoff_seconds(effect=effect, attempt=4) == 8.0

    def test_backoff_capped_at_max(self):
        """Backoff is capped at max value."""
        policy = DefaultEffectPolicy(
            default_max_attempts=10,
            default_backoff_base=1.0,
            default_backoff_max=10.0,
        )
        effect = Effect(type=EffectType.LLM_CALL, payload={})

        # 2^6 = 64, but capped at 10
        assert policy.backoff_seconds(effect=effect, attempt=7) == 10.0

    def test_backoff_called_between_retries(self):
        """Sleep is called between retry attempts."""
        handler = FailingEffectHandler(fail_count=2)
        workflow = make_effect_workflow()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: handler},
            effect_policy=RetryPolicy(llm_max_attempts=3, backoff_base=1.0),
        )

        run_id = runtime.start(workflow=workflow)

        with patch("time.sleep") as mock_sleep:
            runtime.tick(workflow=workflow, run_id=run_id)

        # Should sleep twice (after attempt 1 and 2, not after 3)
        assert mock_sleep.call_count == 2


# -----------------------------------------------------------------------------
# Tests: Idempotency
# -----------------------------------------------------------------------------


class TestIdempotency:
    """Tests for idempotency (skip re-execution on restart)."""

    def test_idempotency_key_computation(self):
        """Idempotency key is deterministic."""
        effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "test"})

        key1 = compute_idempotency_key(
            run_id="run-1", node_id="node-1", effect=effect
        )
        key2 = compute_idempotency_key(
            run_id="run-1", node_id="node-1", effect=effect
        )

        assert key1 == key2
        assert len(key1) == 32

    def test_idempotency_key_differs_by_run(self):
        """Different runs get different keys."""
        effect = Effect(type=EffectType.LLM_CALL, payload={"prompt": "test"})

        key1 = compute_idempotency_key(
            run_id="run-1", node_id="node-1", effect=effect
        )
        key2 = compute_idempotency_key(
            run_id="run-2", node_id="node-1", effect=effect
        )

        assert key1 != key2

    def test_idempotency_key_differs_by_payload(self):
        """Different payloads get different keys."""
        effect1 = Effect(type=EffectType.LLM_CALL, payload={"prompt": "test1"})
        effect2 = Effect(type=EffectType.LLM_CALL, payload={"prompt": "test2"})

        key1 = compute_idempotency_key(
            run_id="run-1", node_id="node-1", effect=effect1
        )
        key2 = compute_idempotency_key(
            run_id="run-1", node_id="node-1", effect=effect2
        )

        assert key1 != key2

    def test_skip_reexecution_on_restart(self):
        """Prior completed result is reused on restart."""
        handler = FailingEffectHandler(fail_count=0)
        workflow = make_effect_workflow()

        run_store = InMemoryRunStore()
        ledger_store = InMemoryLedgerStore()

        runtime = Runtime(
            run_store=run_store,
            ledger_store=ledger_store,
            effect_handlers={EffectType.LLM_CALL: handler},
        )

        # First execution
        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)
        assert state.status == RunStatus.COMPLETED
        assert handler.call_count == 1

        # Simulate restart: create new runtime with same stores
        # but reset the run to before the effect was executed
        run = run_store.load(run_id)
        run.status = RunStatus.RUNNING
        run.current_node = "start"
        run.output = None
        run_store.save(run)

        # Create new runtime (simulating process restart)
        runtime2 = Runtime(
            run_store=run_store,
            ledger_store=ledger_store,
            effect_handlers={EffectType.LLM_CALL: handler},
        )

        # Re-execute - should skip the effect
        state2 = runtime2.tick(workflow=workflow, run_id=run_id)
        assert state2.status == RunStatus.COMPLETED

        # Handler was NOT called again
        assert handler.call_count == 1

    def test_ledger_records_idempotency_key(self):
        """Ledger records include idempotency key."""
        handler = FailingEffectHandler(fail_count=0)
        workflow = make_effect_workflow()

        ledger_store = InMemoryLedgerStore()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=ledger_store,
            effect_handlers={EffectType.LLM_CALL: handler},
        )

        run_id = runtime.start(workflow=workflow)
        runtime.tick(workflow=workflow, run_id=run_id)

        records = ledger_store.list(run_id)
        effect_records = [r for r in records if r.get("effect") is not None]

        assert len(effect_records) >= 1
        assert effect_records[0].get("idempotency_key") is not None
        assert len(effect_records[0]["idempotency_key"]) == 32


class TestLedgerAttemptTracking:
    """Tests for attempt tracking in ledger."""

    def test_ledger_records_attempt_number(self):
        """Ledger records include attempt number."""
        handler = FailingEffectHandler(fail_count=2)
        workflow = make_effect_workflow()

        ledger_store = InMemoryLedgerStore()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=ledger_store,
            effect_handlers={EffectType.LLM_CALL: handler},
            effect_policy=RetryPolicy(llm_max_attempts=3),
        )

        run_id = runtime.start(workflow=workflow)

        with patch("time.sleep"):
            runtime.tick(workflow=workflow, run_id=run_id)

        records = ledger_store.list(run_id)
        effect_records = [r for r in records if r.get("effect") is not None]

        # Should have 6 records: 3 starts + 3 completions
        # Each pair has attempt 1, 2, 3
        attempts = [r.get("attempt") for r in effect_records if r.get("status") == "started"]
        assert attempts == [1, 2, 3]


# -----------------------------------------------------------------------------
# Tests: Policy Configuration
# -----------------------------------------------------------------------------


class TestPolicyConfiguration:
    """Tests for policy configuration."""

    def test_no_retry_policy(self):
        """NoRetryPolicy never retries."""
        policy = NoRetryPolicy()
        effect = Effect(type=EffectType.LLM_CALL, payload={})

        assert policy.max_attempts(effect) == 1

    def test_custom_policy(self):
        """Can create custom policy with specific settings."""
        policy = DefaultEffectPolicy(
            default_max_attempts=5,
            effect_max_attempts={"llm_call": 10},
        )

        llm_effect = Effect(type=EffectType.LLM_CALL, payload={})
        tool_effect = Effect(type=EffectType.TOOL_CALLS, payload={})

        assert policy.max_attempts(llm_effect) == 10
        assert policy.max_attempts(tool_effect) == 5

    def test_scheduled_runtime_with_policy(self):
        """ScheduledRuntime accepts effect_policy."""
        policy = RetryPolicy(llm_max_attempts=5)

        sr = create_scheduled_runtime(
            effect_policy=policy,
            auto_start=False,
        )

        assert sr.runtime.effect_policy is policy


# -----------------------------------------------------------------------------
# Tests: Edge Cases
# -----------------------------------------------------------------------------


class TestRetryEdgeCases:
    """Tests for edge cases in retry logic."""

    def test_waiting_effect_not_retried(self):
        """Waiting effects are not retried."""
        workflow = WorkflowSpec(
            workflow_id="wait_workflow",
            entry_node="start",
            nodes={
                "start": lambda run, ctx: StepPlan(
                    node_id="start",
                    effect=Effect(
                        type=EffectType.WAIT_EVENT,
                        payload={"wait_key": "user_input"},
                        result_key="input",
                    ),
                    next_node="done",
                ),
                "done": lambda run, ctx: StepPlan(
                    node_id="done",
                    complete_output={"done": True},
                ),
            },
        )

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_policy=RetryPolicy(llm_max_attempts=3),
        )

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        # Should be waiting, not failed
        assert state.status == RunStatus.WAITING

    def test_success_on_first_attempt(self):
        """Successful first attempt doesn't trigger retry logic."""
        handler = FailingEffectHandler(fail_count=0)
        workflow = make_effect_workflow()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: handler},
            effect_policy=RetryPolicy(llm_max_attempts=3),
        )

        run_id = runtime.start(workflow=workflow)

        with patch("time.sleep") as mock_sleep:
            state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert handler.call_count == 1
        assert mock_sleep.call_count == 0  # No backoff needed

    def test_exception_in_handler_triggers_retry(self):
        """Exceptions in effect handlers are caught and trigger retry."""
        call_count = 0

        def raising_handler(run: RunState, effect: Effect) -> EffectOutcome:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Simulated exception")
            return EffectOutcome.completed({"success": True})

        workflow = make_effect_workflow()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: raising_handler},
            effect_policy=RetryPolicy(llm_max_attempts=3),
        )

        run_id = runtime.start(workflow=workflow)

        with patch("time.sleep"):
            state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert call_count == 3


# -----------------------------------------------------------------------------
# Tests: File-based Ledger Persistence
# -----------------------------------------------------------------------------


class TestFileLedgerRetryFields:
    """Tests for retry fields in file-based ledger."""

    def test_file_ledger_persists_retry_fields(self):
        """File-based ledger persists attempt and idempotency_key."""
        import tempfile
        from abstractruntime import JsonlLedgerStore

        handler = FailingEffectHandler(fail_count=1)
        workflow = make_effect_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_store = JsonlLedgerStore(tmpdir)

            runtime = Runtime(
                run_store=InMemoryRunStore(),
                ledger_store=ledger_store,
                effect_handlers={EffectType.LLM_CALL: handler},
                effect_policy=RetryPolicy(llm_max_attempts=3),
            )

            run_id = runtime.start(workflow=workflow)

            with patch("time.sleep"):
                runtime.tick(workflow=workflow, run_id=run_id)

            # Read ledger from new instance
            ledger_store2 = JsonlLedgerStore(tmpdir)
            records = ledger_store2.list(run_id)

            # Find effect records
            effect_records = [r for r in records if r.get("effect") is not None]

            # Should have attempt 1 (failed) and attempt 2 (success)
            attempts = sorted(set(r.get("attempt") for r in effect_records))
            assert attempts == [1, 2]

            # All should have idempotency_key
            for r in effect_records:
                assert r.get("idempotency_key") is not None
                assert len(r["idempotency_key"]) == 32
