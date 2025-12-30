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
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunState,
    RunStatus,
    Runtime,
    StepPlan,
    WaitReason,
    WorkflowRegistry,
    WorkflowSpec,
)


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


