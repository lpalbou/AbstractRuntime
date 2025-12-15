"""Integration tests with AbstractCore and Ollama.

These tests verify that AbstractRuntime correctly integrates with
AbstractCore for real LLM calls. Uses local Ollama with gemma3:1b-it-q4_K_M.

Tests are minimal due to resource constraints.
"""

import pytest
from typing import Dict, Any

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
    RetryPolicy,
    create_scheduled_runtime,
)
from abstractruntime.integrations.abstractcore import create_local_runtime


# Skip if Ollama not available
def ollama_available() -> bool:
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not available"
)


# -----------------------------------------------------------------------------
# Test: Basic LLM Call
# -----------------------------------------------------------------------------


class TestAbstractCoreIntegration:
    """Integration tests with AbstractCore."""

    def test_simple_llm_call(self):
        """Execute a simple LLM call through AbstractRuntime."""

        def ask_llm(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="ask",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": "Reply with exactly one word: hello",
                        "params": {"max_tokens": 10},
                    },
                    result_key="llm_response",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            response = run.vars.get("llm_response", {})
            return StepPlan(
                node_id="done",
                complete_output={
                    "content": response.get("content"),
                    "trace_id": response.get("trace_id") or (response.get("metadata") or {}).get("trace_id"),
                    "has_response": response.get("content") is not None,
                },
            )

        workflow = WorkflowSpec(
            workflow_id="simple_llm",
            entry_node="ask",
            nodes={"ask": ask_llm, "done": done},
        )

        runtime = create_local_runtime(
            provider="ollama",
            model="gemma3:1b-it-q4_K_M",
        )

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["has_response"] is True
        assert state.output["content"] is not None
        assert state.output["trace_id"]

    def test_multi_turn_conversation(self):
        """Execute a multi-turn conversation workflow."""

        def turn1(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="turn1",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": "What is 2+2? Reply with just the number.",
                        "params": {"max_tokens": 10},
                    },
                    result_key="answer1",
                ),
                next_node="turn2",
            )

        def turn2(run: RunState, ctx) -> StepPlan:
            first_answer = run.vars.get("answer1", {}).get("content", "")
            return StepPlan(
                node_id="turn2",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": f"You said '{first_answer}'. Is that correct? Reply yes or no.",
                        "params": {"max_tokens": 10},
                    },
                    result_key="answer2",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={
                    "answer1": run.vars.get("answer1", {}).get("content"),
                    "answer2": run.vars.get("answer2", {}).get("content"),
                    "trace1": run.vars.get("answer1", {}).get("trace_id") or (run.vars.get("answer1", {}).get("metadata") or {}).get("trace_id"),
                    "trace2": run.vars.get("answer2", {}).get("trace_id") or (run.vars.get("answer2", {}).get("metadata") or {}).get("trace_id"),
                },
            )

        workflow = WorkflowSpec(
            workflow_id="multi_turn",
            entry_node="turn1",
            nodes={"turn1": turn1, "turn2": turn2, "done": done},
        )

        runtime = create_local_runtime(
            provider="ollama",
            model="gemma3:1b-it-q4_K_M",
        )

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["answer1"] is not None
        assert state.output["answer2"] is not None
        assert state.output["trace1"]
        assert state.output["trace2"]

    def test_llm_with_retry_policy(self):
        """LLM calls work with retry policy."""

        def ask(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="ask",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": "Say 'ok'",
                        "params": {"max_tokens": 5},
                    },
                    result_key="response",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={"response": run.vars.get("response")},
            )

        workflow = WorkflowSpec(
            workflow_id="retry_test",
            entry_node="ask",
            nodes={"ask": ask, "done": done},
        )

        runtime = create_local_runtime(
            provider="ollama",
            model="gemma3:1b-it-q4_K_M",
            effect_policy=RetryPolicy(llm_max_attempts=2),
        )

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED


# -----------------------------------------------------------------------------
# Test: Subworkflow with LLM
# -----------------------------------------------------------------------------


class TestSubworkflowWithLLM:
    """Test subworkflows with real LLM calls."""

    def test_parent_child_llm_workflow(self):
        """Parent workflow spawns child that makes LLM call."""
        from abstractruntime.scheduler import WorkflowRegistry

        # Child workflow: makes LLM call
        def child_ask(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="ask",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": "Reply with 'child done'",
                        "params": {"max_tokens": 10},
                    },
                    result_key="response",
                ),
                next_node="done",
            )

        def child_done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={"child_response": run.vars.get("response", {}).get("content")},
            )

        child_workflow = WorkflowSpec(
            workflow_id="child_llm",
            entry_node="ask",
            nodes={"ask": child_ask, "done": child_done},
        )

        # Parent workflow: spawns child
        def parent_start(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="start",
                effect=Effect(
                    type=EffectType.START_SUBWORKFLOW,
                    payload={"workflow_id": "child_llm"},
                    result_key="child_result",
                ),
                next_node="done",
            )

        def parent_done(run: RunState, ctx) -> StepPlan:
            child_result = run.vars.get("child_result", {})
            return StepPlan(
                node_id="done",
                complete_output={
                    "parent_done": True,
                    "child_output": child_result.get("output"),
                },
            )

        parent_workflow = WorkflowSpec(
            workflow_id="parent_llm",
            entry_node="start",
            nodes={"start": parent_start, "done": parent_done},
        )

        # Create runtime with registry
        registry = WorkflowRegistry()
        registry.register(child_workflow)
        registry.register(parent_workflow)

        runtime = create_local_runtime(
            provider="ollama",
            model="gemma3:1b-it-q4_K_M",
        )
        runtime.set_workflow_registry(registry)

        run_id = runtime.start(workflow=parent_workflow)
        state = runtime.tick(workflow=parent_workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["parent_done"] is True
        assert state.output["child_output"] is not None
        assert state.output["child_output"]["child_response"] is not None
