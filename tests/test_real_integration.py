"""Real integration tests with AbstractCore and Ollama.

These tests prove that AbstractRuntime + AbstractCore work together
for real agent workflows. Uses local Ollama with gemma3:1b-it-q4_K_M.

Each test demonstrates a specific capability with empirical evidence.
"""

import pytest
import json
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
    InMemoryArtifactStore,
    RetryPolicy,
    WaitReason,
    artifact_ref,
    resolve_artifact,
)
from abstractruntime.integrations.abstractcore import (
    create_local_runtime,
    build_effect_handlers,
    LocalAbstractCoreLLMClient,
    PassthroughToolExecutor,
)
from abstractruntime.scheduler import WorkflowRegistry


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

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

MODEL = "gemma3:1b-it-q4_K_M"
PROVIDER = "ollama"


def create_test_runtime(**kwargs) -> Runtime:
    """Create a runtime for testing with AbstractCore."""
    return create_local_runtime(provider=PROVIDER, model=MODEL, **kwargs)


# -----------------------------------------------------------------------------
# Test 1: Basic LLM Calls with Verification
# -----------------------------------------------------------------------------

class TestBasicLLMCalls:
    """Verify LLM calls work and produce meaningful responses."""

    def test_simple_math_question(self):
        """LLM can answer a simple math question."""
        
        def ask(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="ask",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": "What is 2+2? Reply with just the number."},
                    result_key="answer",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={"answer": run.vars.get("answer", {}).get("content")},
            )

        workflow = WorkflowSpec(
            workflow_id="math_test",
            entry_node="ask",
            nodes={"ask": ask, "done": done},
        )

        runtime = create_test_runtime()
        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["answer"] is not None
        # The answer should contain "4"
        assert "4" in state.output["answer"]
        print(f"Math answer: {state.output['answer']}")

    def test_classification_task(self):
        """LLM can classify text."""
        
        def classify(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="classify",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": "Classify this sentiment as POSITIVE or NEGATIVE. Reply with one word only.\nText: I love this product, it's amazing!",
                    },
                    result_key="classification",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={"result": run.vars.get("classification", {}).get("content")},
            )

        workflow = WorkflowSpec(
            workflow_id="classify_test",
            entry_node="classify",
            nodes={"classify": classify, "done": done},
        )

        runtime = create_test_runtime()
        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        result = state.output["result"].upper()
        assert "POSITIVE" in result or "POS" in result
        print(f"Classification: {state.output['result']}")

    def test_json_extraction(self):
        """LLM can extract structured data."""
        
        def extract(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="extract",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": 'Extract the name and age from this text. Reply in JSON format {"name": "...", "age": ...}\nText: John Smith is 35 years old.',
                    },
                    result_key="extracted",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={"result": run.vars.get("extracted", {}).get("content")},
            )

        workflow = WorkflowSpec(
            workflow_id="extract_test",
            entry_node="extract",
            nodes={"extract": extract, "done": done},
        )

        runtime = create_test_runtime()
        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        result = state.output["result"]
        assert "John" in result or "john" in result
        assert "35" in result
        print(f"Extraction: {result}")


# -----------------------------------------------------------------------------
# Test 2: Multi-Step Reasoning Workflow
# -----------------------------------------------------------------------------

class TestMultiStepReasoning:
    """Test workflows that chain multiple LLM calls."""

    def test_think_then_answer(self):
        """Two-step: think about the problem, then answer."""
        
        def think(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="think",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": "Think step by step: If I have 3 apples and buy 2 more, then give away 1, how many do I have? Show your reasoning.",
                    },
                    result_key="reasoning",
                ),
                next_node="answer",
            )

        def answer(run: RunState, ctx) -> StepPlan:
            reasoning = run.vars.get("reasoning", {}).get("content", "")
            return StepPlan(
                node_id="answer",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": f"Based on this reasoning:\n{reasoning}\n\nWhat is the final answer? Reply with just the number.",
                    },
                    result_key="final_answer",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={
                    "reasoning": run.vars.get("reasoning", {}).get("content"),
                    "answer": run.vars.get("final_answer", {}).get("content"),
                },
            )

        workflow = WorkflowSpec(
            workflow_id="think_answer",
            entry_node="think",
            nodes={"think": think, "answer": answer, "done": done},
        )

        runtime = create_test_runtime()
        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["reasoning"] is not None
        assert state.output["answer"] is not None
        # Answer should be 4 (3 + 2 - 1)
        assert "4" in state.output["answer"]
        print(f"Reasoning: {state.output['reasoning'][:100]}...")
        print(f"Answer: {state.output['answer']}")

    def test_summarize_then_translate(self):
        """Two-step: summarize text, then translate summary."""
        
        def summarize(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="summarize",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": "Summarize in one sentence: The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
                    },
                    result_key="summary",
                ),
                next_node="translate",
            )

        def translate(run: RunState, ctx) -> StepPlan:
            summary = run.vars.get("summary", {}).get("content", "")
            return StepPlan(
                node_id="translate",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": f"Translate to French: {summary}",
                    },
                    result_key="french",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={
                    "summary": run.vars.get("summary", {}).get("content"),
                    "french": run.vars.get("french", {}).get("content"),
                },
            )

        workflow = WorkflowSpec(
            workflow_id="summarize_translate",
            entry_node="summarize",
            nodes={"summarize": summarize, "translate": translate, "done": done},
        )

        runtime = create_test_runtime()
        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["summary"] is not None
        assert state.output["french"] is not None
        print(f"Summary: {state.output['summary']}")
        print(f"French: {state.output['french']}")


# -----------------------------------------------------------------------------
# Test 3: Subworkflows with LLM
# -----------------------------------------------------------------------------

class TestSubworkflowsWithLLM:
    """Test parent-child workflow composition with LLM calls."""

    def test_parent_delegates_to_specialist_child(self):
        """Parent workflow delegates a task to a specialist child."""
        
        # Child: Math specialist
        def child_solve(run: RunState, ctx) -> StepPlan:
            problem = run.vars.get("problem", "2+2")
            return StepPlan(
                node_id="solve",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": f"Solve this math problem. Reply with just the answer: {problem}"},
                    result_key="solution",
                ),
                next_node="done",
            )

        def child_done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={"solution": run.vars.get("solution", {}).get("content")},
            )

        child_workflow = WorkflowSpec(
            workflow_id="math_specialist",
            entry_node="solve",
            nodes={"solve": child_solve, "done": child_done},
        )

        # Parent: Coordinator
        def parent_start(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="start",
                effect=Effect(
                    type=EffectType.START_SUBWORKFLOW,
                    payload={
                        "workflow_id": "math_specialist",
                        "vars": {"problem": "15 * 3"},
                    },
                    result_key="child_result",
                ),
                next_node="report",
            )

        def parent_report(run: RunState, ctx) -> StepPlan:
            child_output = run.vars.get("child_result", {}).get("output", {})
            solution = child_output.get("solution", "unknown")
            return StepPlan(
                node_id="report",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": f"The math specialist says the answer is {solution}. Is this correct for 15*3? Reply yes or no."},
                    result_key="verification",
                ),
                next_node="done",
            )

        def parent_done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={
                    "child_solution": run.vars.get("child_result", {}).get("output", {}).get("solution"),
                    "verification": run.vars.get("verification", {}).get("content"),
                },
            )

        parent_workflow = WorkflowSpec(
            workflow_id="coordinator",
            entry_node="start",
            nodes={"start": parent_start, "report": parent_report, "done": parent_done},
        )

        # Setup registry
        registry = WorkflowRegistry()
        registry.register(child_workflow)
        registry.register(parent_workflow)

        runtime = create_test_runtime()
        runtime.set_workflow_registry(registry)

        run_id = runtime.start(workflow=parent_workflow)
        state = runtime.tick(workflow=parent_workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["child_solution"] is not None
        assert "45" in state.output["child_solution"]
        print(f"Child solution: {state.output['child_solution']}")
        print(f"Verification: {state.output['verification']}")


# -----------------------------------------------------------------------------
# Test 4: Artifacts with LLM Output
# -----------------------------------------------------------------------------

class TestArtifactsWithLLM:
    """Test storing LLM outputs as artifacts."""

    def test_store_large_llm_output_as_artifact(self):
        """Store a large LLM response as an artifact."""
        
        artifact_store = InMemoryArtifactStore()

        def generate(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="generate",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": "Write a short story about a robot learning to paint. Make it at least 3 paragraphs."},
                    result_key="story",
                ),
                next_node="store",
            )

        def store(run: RunState, ctx) -> StepPlan:
            story = run.vars.get("story", {}).get("content", "")
            # Store as artifact
            metadata = artifact_store.store_text(story, run_id=run.run_id, content_type="text/plain")
            run.vars["story_artifact_id"] = metadata.artifact_id
            return StepPlan(node_id="store", next_node="done")

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={
                    "artifact_id": run.vars.get("story_artifact_id"),
                    "story_preview": run.vars.get("story", {}).get("content", "")[:100],
                },
            )

        workflow = WorkflowSpec(
            workflow_id="artifact_test",
            entry_node="generate",
            nodes={"generate": generate, "store": store, "done": done},
        )

        runtime = create_test_runtime()
        runtime.set_artifact_store(artifact_store)

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["artifact_id"] is not None

        # Verify artifact was stored
        artifact = artifact_store.load(state.output["artifact_id"])
        assert artifact is not None
        story_text = artifact.as_text()
        assert len(story_text) > 50  # Should have substantial content
        print(f"Artifact ID: {state.output['artifact_id']}")
        print(f"Story length: {len(story_text)} chars")
        print(f"Preview: {story_text[:100]}...")


# -----------------------------------------------------------------------------
# Test 5: Wait and Resume with LLM
# -----------------------------------------------------------------------------

class TestWaitResumeWithLLM:
    """Test wait/resume patterns with LLM calls."""

    def test_llm_then_wait_then_llm(self):
        """LLM call, wait for user input, then another LLM call."""
        
        def ask_question(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="ask",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": "Generate a simple yes/no question about weather."},
                    result_key="question",
                ),
                next_node="wait",
            )

        def wait_answer(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="wait",
                effect=Effect(
                    type=EffectType.WAIT_EVENT,
                    payload={"wait_key": "user_answer"},
                    result_key="user_input",
                ),
                next_node="respond",
            )

        def respond(run: RunState, ctx) -> StepPlan:
            question = run.vars.get("question", {}).get("content", "")
            answer = run.vars.get("user_input", {}).get("answer", "")
            return StepPlan(
                node_id="respond",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": f"The user was asked: {question}\nThey answered: {answer}\nProvide a brief response to their answer."},
                    result_key="response",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={
                    "question": run.vars.get("question", {}).get("content"),
                    "user_answer": run.vars.get("user_input", {}).get("answer"),
                    "response": run.vars.get("response", {}).get("content"),
                },
            )

        workflow = WorkflowSpec(
            workflow_id="wait_resume_test",
            entry_node="ask",
            nodes={"ask": ask_question, "wait": wait_answer, "respond": respond, "done": done},
        )

        runtime = create_test_runtime()

        # Start and run until wait
        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.WAITING
        assert state.waiting.reason == WaitReason.EVENT
        assert state.waiting.wait_key == "user_answer"
        print(f"Question generated: {state.vars.get('question', {}).get('content')}")

        # Resume with user input
        state = runtime.resume(
            workflow=workflow,
            run_id=run_id,
            wait_key="user_answer",
            payload={"answer": "Yes, I like sunny weather"},
        )

        assert state.status == RunStatus.COMPLETED
        assert state.output["question"] is not None
        assert state.output["response"] is not None
        print(f"Response: {state.output['response']}")


# -----------------------------------------------------------------------------
# Test 6: Ledger and Idempotency
# -----------------------------------------------------------------------------

class TestLedgerAndIdempotency:
    """Test that ledger records LLM calls and idempotency works."""

    def test_ledger_records_llm_calls(self):
        """Verify ledger contains LLM call records."""
        
        ledger_store = InMemoryLedgerStore()
        run_store = InMemoryRunStore()

        def ask(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="ask",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": "Say hello"},
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
            workflow_id="ledger_test",
            entry_node="ask",
            nodes={"ask": ask, "done": done},
        )

        runtime = create_local_runtime(
            provider=PROVIDER,
            model=MODEL,
            run_store=run_store,
            ledger_store=ledger_store,
        )

        run_id = runtime.start(workflow=workflow)
        runtime.tick(workflow=workflow, run_id=run_id)

        # Check ledger
        records = ledger_store.list(run_id)
        assert len(records) >= 2  # At least start and complete

        # Find LLM call record
        llm_records = [r for r in records if r.get("effect") and r["effect"].get("type") == "llm_call"]
        assert len(llm_records) >= 1

        # Verify idempotency key was recorded
        assert llm_records[0].get("idempotency_key") is not None
        print(f"Ledger has {len(records)} records")
        print(f"LLM call idempotency key: {llm_records[0]['idempotency_key']}")


# -----------------------------------------------------------------------------
# Test 7: Error Handling
# -----------------------------------------------------------------------------

class TestErrorHandling:
    """Test error handling with LLM calls."""

    def test_workflow_continues_after_llm_success(self):
        """Workflow properly continues after successful LLM call."""
        
        steps_executed = []

        def step1(run: RunState, ctx) -> StepPlan:
            steps_executed.append("step1")
            return StepPlan(
                node_id="step1",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": "Say 'step1 done'"},
                    result_key="r1",
                ),
                next_node="step2",
            )

        def step2(run: RunState, ctx) -> StepPlan:
            steps_executed.append("step2")
            return StepPlan(
                node_id="step2",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={"prompt": "Say 'step2 done'"},
                    result_key="r2",
                ),
                next_node="step3",
            )

        def step3(run: RunState, ctx) -> StepPlan:
            steps_executed.append("step3")
            return StepPlan(
                node_id="step3",
                complete_output={
                    "steps": steps_executed.copy(),
                    "r1": run.vars.get("r1", {}).get("content"),
                    "r2": run.vars.get("r2", {}).get("content"),
                },
            )

        workflow = WorkflowSpec(
            workflow_id="multi_step",
            entry_node="step1",
            nodes={"step1": step1, "step2": step2, "step3": step3},
        )

        runtime = create_test_runtime()
        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["steps"] == ["step1", "step2", "step3"]
        assert state.output["r1"] is not None
        assert state.output["r2"] is not None
        print(f"Steps executed: {state.output['steps']}")


# -----------------------------------------------------------------------------
# Test 8: Full Agent-like Workflow
# -----------------------------------------------------------------------------

class TestAgentWorkflow:
    """Test a complete agent-like workflow."""

    def test_simple_react_pattern(self):
        """Test a simplified ReAct pattern: Reason -> Act -> Observe."""
        
        def reason(run: RunState, ctx) -> StepPlan:
            task = run.vars.get("task", "What is the capital of France?")
            return StepPlan(
                node_id="reason",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": f"Task: {task}\n\nThink about what you need to do to answer this. What is your reasoning?",
                    },
                    result_key="reasoning",
                ),
                next_node="act",
            )

        def act(run: RunState, ctx) -> StepPlan:
            reasoning = run.vars.get("reasoning", {}).get("content", "")
            return StepPlan(
                node_id="act",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": f"Based on your reasoning:\n{reasoning}\n\nNow provide the answer to the original task.",
                    },
                    result_key="action",
                ),
                next_node="observe",
            )

        def observe(run: RunState, ctx) -> StepPlan:
            action = run.vars.get("action", {}).get("content", "")
            return StepPlan(
                node_id="observe",
                effect=Effect(
                    type=EffectType.LLM_CALL,
                    payload={
                        "prompt": f"You answered: {action}\n\nIs this answer complete and correct? Reply with 'COMPLETE' if done, or explain what's missing.",
                    },
                    result_key="observation",
                ),
                next_node="done",
            )

        def done(run: RunState, ctx) -> StepPlan:
            return StepPlan(
                node_id="done",
                complete_output={
                    "task": run.vars.get("task"),
                    "reasoning": run.vars.get("reasoning", {}).get("content"),
                    "action": run.vars.get("action", {}).get("content"),
                    "observation": run.vars.get("observation", {}).get("content"),
                },
            )

        workflow = WorkflowSpec(
            workflow_id="react_agent",
            entry_node="reason",
            nodes={"reason": reason, "act": act, "observe": observe, "done": done},
        )

        runtime = create_test_runtime()
        run_id = runtime.start(workflow=workflow, vars={"task": "What is the capital of France?"})
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["reasoning"] is not None
        assert state.output["action"] is not None
        assert state.output["observation"] is not None
        # The answer should mention Paris
        assert "Paris" in state.output["action"] or "paris" in state.output["action"].lower()
        
        print("=== ReAct Agent Execution ===")
        print(f"Task: {state.output['task']}")
        print(f"Reasoning: {state.output['reasoning'][:150]}...")
        print(f"Action: {state.output['action']}")
        print(f"Observation: {state.output['observation']}")
