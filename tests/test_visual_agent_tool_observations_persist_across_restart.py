import tempfile
from pathlib import Path

from abstractruntime import Runtime, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.visualflow_compiler.compiler import compile_visualflow
from abstractruntime.visualflow_compiler.visual.agent_ids import visual_react_workflow_id


class StubLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs):
        # First call: request a tool call. Second call: provide final answer.
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "I'll list files.",
                "tool_calls": [
                    {"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c1"}
                ],
            }
        return {"content": "Done.", "tool_calls": []}


def test_visual_agent_tool_observations_persist_across_restart():
    """Level B: file-backed run store + restart simulation.

    This proves that tool-evidence persisted into parent active context survives restart,
    which is required for RALPH-style outer loops to have durable progression evidence.
    """
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic
    from abstractcore.tools.core import ToolDefinition

    def list_files(directory_path: str = ".") -> str:
        """List files in a directory (stub)."""
        _ = directory_path
        return "file_a.py\nfile_b.py"

    tool_def = ToolDefinition.from_function(list_files)
    logic = ReActLogic(tools=[tool_def])

    flow_id = "test-flow"
    node_id = "node-agent"
    react_id = visual_react_workflow_id(flow_id=flow_id, node_id=node_id)
    react_spec = create_react_workflow(logic=logic, workflow_id=react_id)

    parent_spec = compile_visualflow(
        {
            "id": flow_id,
            "name": "test",
            "nodes": [
                {
                    "id": "start",
                    "type": "on_flow_start",
                    "data": {
                        "outputs": [
                            {"id": "exec-out", "type": "execution"},
                            {"id": "prompt", "type": "string"},
                            {"id": "provider", "type": "provider"},
                            {"id": "model", "type": "model"},
                            {"id": "tools", "type": "tools"},
                        ]
                    },
                },
                {
                    "id": node_id,
                    "type": "agent",
                    "data": {
                        "inputs": [
                            {"id": "exec-in", "type": "execution"},
                            {"id": "include_context", "type": "boolean"},
                            {"id": "provider", "type": "provider"},
                            {"id": "model", "type": "model"},
                            {"id": "prompt", "type": "string"},
                            {"id": "tools", "type": "tools"},
                        ],
                        "pinDefaults": {"include_context": True},
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": node_id, "targetHandle": "exec-in"},
                {"source": "start", "sourceHandle": "prompt", "target": node_id, "targetHandle": "prompt"},
                {"source": "start", "sourceHandle": "provider", "target": node_id, "targetHandle": "provider"},
                {"source": "start", "sourceHandle": "model", "target": node_id, "targetHandle": "model"},
                {"source": "start", "sourceHandle": "tools", "target": node_id, "targetHandle": "tools"},
            ],
            "entryNode": "start",
        }
    )

    registry = WorkflowRegistry()
    registry.register(parent_spec)
    registry.register(react_spec)

    executor = MappingToolExecutor.from_tools([list_files])

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)

        rt1 = Runtime(
            run_store=JsonFileRunStore(base),
            ledger_store=JsonlLedgerStore(base),
            effect_handlers=build_effect_handlers(llm=StubLLM(), tools=executor),
            workflow_registry=registry,
        )

        run_id = rt1.start(
            workflow=parent_spec,
            vars={
                "prompt": "List files in the repo.",
                "provider": "lmstudio",
                "model": "dummy",
                "tools": ["list_files"],
                "context": {},
            },
        )

        state1 = rt1.tick(workflow=parent_spec, run_id=run_id, max_steps=10)
        assert state1.status == RunStatus.WAITING
        assert state1.waiting is not None
        sub_run_id = state1.waiting.details.get("sub_run_id")
        assert isinstance(sub_run_id, str) and sub_run_id

        # Drive the sub-run to completion.
        child_state = rt1.tick(workflow=react_spec, run_id=sub_run_id)
        assert child_state.status == RunStatus.COMPLETED

        payload = {
            "sub_run_id": sub_run_id,
            "output": child_state.output,
            "node_traces": rt1.get_node_traces(sub_run_id),
        }

        final = rt1.resume(
            workflow=parent_spec,
            run_id=run_id,
            wait_key=f"subworkflow:{sub_run_id}",
            payload=payload,
            max_steps=50,
        )
        assert final.status == RunStatus.COMPLETED

        ctx = final.vars.get("context")
        assert isinstance(ctx, dict)
        msgs = ctx.get("messages")
        assert isinstance(msgs, list)
        assert any("[list_files]:" in str(m.get("content") or "") for m in msgs if isinstance(m, dict))

        # Restart simulation: load from disk and ensure evidence persisted.
        rt2 = Runtime(
            run_store=JsonFileRunStore(base),
            ledger_store=JsonlLedgerStore(base),
            effect_handlers=build_effect_handlers(llm=StubLLM(), tools=executor),
            workflow_registry=registry,
        )
        loaded = rt2.get_state(run_id)
        ctx2 = loaded.vars.get("context")
        assert isinstance(ctx2, dict)
        msgs2 = ctx2.get("messages")
        assert isinstance(msgs2, list)
        assert any("[list_files]:" in str(m.get("content") or "") for m in msgs2 if isinstance(m, dict))
