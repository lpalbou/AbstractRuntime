import tempfile
from pathlib import Path

from abstractruntime import Effect, EffectType, Runtime, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore


class StubLLMClient:
    def generate(self, **kwargs):
        raise AssertionError("LLM should not be called in this test")


def test_tool_calls_persist_and_continue_across_restart():
    def add(x: int, y: int) -> int:
        return x + y

    executor = MappingToolExecutor.from_tools([add])

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)

        wf = WorkflowSpec(
            workflow_id="wf_tools_file",
            entry_node="ACT",
            nodes={
                "ACT": lambda run, ctx: StepPlan(
                    node_id="ACT",
                    effect=Effect(
                        type=EffectType.TOOL_CALLS,
                        payload={
                            "tool_calls": [
                                {"name": "add", "arguments": {"x": 2, "y": 3}, "call_id": "c1"}
                            ]
                        },
                        result_key="tool_results",
                    ),
                    next_node="DONE",
                ),
                "DONE": lambda run, ctx: StepPlan(
                    node_id="DONE",
                    complete_output={"tool_results": run.vars.get("tool_results")},
                ),
            },
        )

        rt1 = Runtime(
            run_store=JsonFileRunStore(base),
            ledger_store=JsonlLedgerStore(base),
            effect_handlers=build_effect_handlers(llm=StubLLMClient(), tools=executor),
        )

        run_id = rt1.start(workflow=wf, vars={"task": "add 2 + 3"})
        state1 = rt1.tick(workflow=wf, run_id=run_id, max_steps=1)
        assert state1.status == RunStatus.RUNNING
        assert state1.current_node == "DONE"

        # New runtime instance simulates a process restart.
        rt2 = Runtime(
            run_store=JsonFileRunStore(base),
            ledger_store=JsonlLedgerStore(base),
            effect_handlers=build_effect_handlers(llm=StubLLMClient(), tools=executor),
        )

        state2 = rt2.get_state(run_id)
        assert state2.status == RunStatus.RUNNING
        assert state2.vars["tool_results"]["mode"] == "executed"
        assert state2.vars["tool_results"]["results"][0]["name"] == "add"
        assert state2.vars["tool_results"]["results"][0]["output"] == 5

        final = rt2.tick(workflow=wf, run_id=run_id)
        assert final.status == RunStatus.COMPLETED
        assert final.output["tool_results"]["results"][0]["output"] == 5

