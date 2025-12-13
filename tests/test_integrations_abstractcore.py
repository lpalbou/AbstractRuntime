import pytest

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class StubLLMClient:
    def __init__(self, result):
        self._result = result

    def generate(self, **kwargs):
        return dict(self._result)


class StubToolExecutor:
    def __init__(self, result):
        self._result = result

    def execute(self, *, tool_calls):
        return dict(self._result)


def test_llm_call_handler_sets_result_key():
    from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers

    llm = StubLLMClient({"content": "ok"})
    tools = StubToolExecutor({"mode": "executed", "results": []})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_effect_handlers(llm=llm, tools=tools),
    )

    def n1(run, ctx):
        return StepPlan(
            node_id="n1",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": "hello"},
                result_key="llm",
            ),
            next_node="n2",
        )

    def n2(run, ctx):
        return StepPlan(node_id="n2", complete_output={"llm": run.vars.get("llm")})

    wf = WorkflowSpec(workflow_id="wf", entry_node="n1", nodes={"n1": n1, "n2": n2})
    run_id = rt.start(workflow=wf)
    state = rt.tick(workflow=wf, run_id=run_id)

    assert state.status.value == "completed"
    assert state.vars["llm"]["content"] == "ok"


def test_tool_calls_passthrough_waits_and_resumes_to_next_node():
    from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers

    llm = StubLLMClient({"content": "ok"})
    tools = StubToolExecutor({"mode": "passthrough", "tool_calls": [{"name": "x", "arguments": {}}]})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_effect_handlers(llm=llm, tools=tools),
    )

    def n1(run, ctx):
        return StepPlan(
            node_id="n1",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "x", "arguments": {}}]},
                result_key="tools",
            ),
            next_node="n2",
        )

    def n2(run, ctx):
        # Must run only after resume; verifies resume_to_node wiring.
        return StepPlan(node_id="n2", complete_output={"tools": run.vars.get("tools")})

    wf = WorkflowSpec(workflow_id="wf", entry_node="n1", nodes={"n1": n1, "n2": n2})

    run_id = rt.start(workflow=wf)
    state = rt.tick(workflow=wf, run_id=run_id)

    assert state.status.value == "waiting"
    assert state.waiting is not None
    assert state.waiting.reason.value == "event"
    assert state.waiting.resume_to_node == "n2"
    assert state.waiting.details["tool_calls"][0]["name"] == "x"

    resumed = rt.resume(
        workflow=wf,
        run_id=run_id,
        wait_key=state.waiting.wait_key,
        payload={"results": [{"name": "x", "output": 1}]},
    )

    assert resumed.status.value == "completed"
    assert resumed.vars["tools"]["results"][0]["output"] == 1


def test_abstractcore_tool_executor_executes_registered_tool():
    abstractcore = pytest.importorskip("abstractcore")

    from abstractcore.tools.registry import clear_registry, register_tool
    from abstractruntime.integrations.abstractcore.tool_executor import AbstractCoreToolExecutor

    clear_registry()

    @register_tool
    def add(x: int, y: int) -> int:
        return x + y

    executor = AbstractCoreToolExecutor()
    result = executor.execute(tool_calls=[{"name": "add", "arguments": {"x": 2, "y": 3}, "call_id": "c1"}])

    assert result["mode"] == "executed"
    assert result["results"][0]["call_id"] == "c1"
    assert result["results"][0]["success"] is True
    assert result["results"][0]["output"] == 5

