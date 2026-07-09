"""Regression: START_SUBWORKFLOW supports async+wait mode.

Goal:
- Allow hosts to start a child run without blocking the parent tick (async=True),
  but still keep the parent in a durable waiting state (wait=True) until the host
  resumes it with the child's final output.
"""

from __future__ import annotations

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryArtifactStore,
    InMemoryLedgerStore,
    InMemoryRunStore,
    OffloadingLedgerStore,
    OffloadingRunStore,
    RunState,
    RunStatus,
    Runtime,
    StepPlan,
    WaitReason,
    WorkflowRegistry,
    WorkflowSpec,
    get_artifact_id,
    is_artifact_ref,
)
from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
from abstractruntime.visualflow_compiler.flow import Flow


def test_start_subworkflow_async_wait_puts_parent_in_subworkflow_wait() -> None:
    def child_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="child", complete_output={"answer": "ok"})

    child = WorkflowSpec(workflow_id="child_wf", entry_node="child", nodes={"child": child_node})

    def parent_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="parent",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={"workflow_id": "child_wf", "async": True, "wait": True},
                result_key="sub_result",
            ),
            next_node="after",
        )

    def after_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="after", complete_output={"sub": run.vars.get("sub_result")})

    parent = WorkflowSpec(workflow_id="parent_wf", entry_node="parent", nodes={"parent": parent_node, "after": after_node})

    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    run_id = rt.start(workflow=parent, vars={})

    st = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    assert st.status == RunStatus.WAITING
    assert st.waiting is not None
    assert st.waiting.reason == WaitReason.SUBWORKFLOW
    assert isinstance(st.waiting.wait_key, str) and st.waiting.wait_key.startswith("subworkflow:")


def test_sync_effect_results_resolve_artifact_backed_subworkflow_output() -> None:
    store = InMemoryArtifactStore()
    meta = store.store_json({"answer": "artifact answer", "status": "ok"}, run_id="child-run")

    flow = Flow("vf-subflow")
    flow.add_node(
        "sub",
        handler=lambda value: value,
        effect_type="start_subworkflow",
        effect_config={"output_pins": ["answer", "status"]},
    )
    flow.set_entry("sub")
    flow._node_outputs = {}  # type: ignore[attr-defined]

    run = RunState.new(
        workflow_id=flow.flow_id,
        entry_node="sub",
        vars={
            "_temp": {
                "effects": {
                    "sub": {
                        "sub_run_id": "child-run",
                        "output": {"$artifact": meta.artifact_id},
                    }
                }
            }
        },
    )
    setattr(run, "_runtime_artifact_store", store)

    _sync_effect_results_to_node_outputs(run, flow)

    out = flow._node_outputs["sub"]  # type: ignore[attr-defined]
    assert out["sub_run_id"] == "child-run"
    assert out["output"] == {"answer": "artifact answer", "status": "ok"}
    assert out["answer"] == "artifact answer"
    assert out["status"] == "ok"


def test_async_wait_wrap_as_tool_result_resolves_artifact_backed_child_output() -> None:
    def child_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="child",
            complete_output={"answer": "artifact answer", "report": "r" * 4000},
        )

    child = WorkflowSpec(workflow_id="child_wf", entry_node="child", nodes={"child": child_node})

    def parent_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="parent",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={
                    "workflow_id": "child_wf",
                    "async": True,
                    "wait": True,
                    "wrap_as_tool_result": True,
                    "tool_name": "delegate_child",
                    "call_id": "call-1",
                },
                result_key="sub_result",
            ),
            next_node="after",
        )

    def after_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="after", complete_output={"sub_result": run.vars.get("sub_result")})

    parent = WorkflowSpec(
        workflow_id="parent_wf",
        entry_node="parent",
        nodes={"parent": parent_node, "after": after_node},
    )

    artifact_store = InMemoryArtifactStore()
    run_store = OffloadingRunStore(InMemoryRunStore(), artifact_store=artifact_store, max_inline_bytes=256)
    ledger_store = OffloadingLedgerStore(InMemoryLedgerStore(), artifact_store=artifact_store, max_inline_bytes=256)
    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=reg, artifact_store=artifact_store)

    parent_run_id = rt.start(workflow=parent)
    parent_wait = rt.tick(workflow=parent, run_id=parent_run_id, max_steps=1)
    assert parent_wait.status == RunStatus.WAITING
    assert parent_wait.waiting is not None
    sub_run_id = parent_wait.waiting.details["sub_run_id"]

    child_state = rt.tick(workflow=child, run_id=sub_run_id)
    assert child_state.status == RunStatus.COMPLETED

    persisted_child = rt.get_state(sub_run_id)
    assert isinstance(persisted_child.output, dict)
    assert is_artifact_ref(persisted_child.output.get("report"))
    artifact = artifact_store.load(get_artifact_id(persisted_child.output["report"]))
    assert artifact is not None

    resumed_parent = rt.resume(
        workflow=parent,
        run_id=parent_run_id,
        wait_key=parent_wait.waiting.wait_key,
        payload={"sub_run_id": sub_run_id, "output": persisted_child.output},
    )
    assert resumed_parent.status == RunStatus.COMPLETED
    result = resumed_parent.output["sub_result"]["results"][0]
    assert result["name"] == "delegate_child"
    assert result["success"] is True
    assert result["output"]["answer"] == "artifact answer"
