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


def test_start_subworkflow_inherits_workspace_policy_from_parent_when_missing(tmp_path) -> None:
    def child_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="child", complete_output={"ok": True})

    child = WorkflowSpec(workflow_id="child_wf", entry_node="child", nodes={"child": child_node})

    def parent_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="parent",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={"workflow_id": "child_wf", "vars": {}, "async": True, "wait": True},
                result_key="sub_result",
            ),
            next_node="after",
        )

    parent = WorkflowSpec(workflow_id="parent_wf", entry_node="parent", nodes={"parent": parent_node})

    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    ws_root = str(tmp_path / "ws")
    run_id = rt.start(
        workflow=parent,
        vars={
            "workspace_root": ws_root,
            "workspace_access_mode": "workspace_only",
            "workspace_allowed_paths": "",
            "workspace_ignored_paths": "node_modules\nsecret",
        },
    )

    st = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    assert st.status == RunStatus.WAITING
    assert st.waiting is not None
    assert st.waiting.reason == WaitReason.SUBWORKFLOW

    wait_key = str(st.waiting.wait_key or "")
    assert wait_key.startswith("subworkflow:")
    sub_run_id = wait_key.split(":", 1)[1]
    assert sub_run_id

    child_run = rt.run_store.load(sub_run_id)
    assert child_run is not None
    assert child_run.vars.get("workspace_root") == ws_root
    assert child_run.vars.get("workspace_access_mode") == "workspace_only"
    assert child_run.vars.get("workspace_ignored_paths") == "node_modules\nsecret"


def test_start_subworkflow_does_not_override_explicit_child_workspace(tmp_path) -> None:
    def child_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="child", complete_output={"ok": True})

    child = WorkflowSpec(workflow_id="child_wf", entry_node="child", nodes={"child": child_node})

    explicit = str(tmp_path / "explicit_ws")

    def parent_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="parent",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={"workflow_id": "child_wf", "vars": {"workspace_root": explicit}, "async": True, "wait": True},
                result_key="sub_result",
            ),
            next_node="after",
        )

    parent = WorkflowSpec(workflow_id="parent_wf", entry_node="parent", nodes={"parent": parent_node})

    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    run_id = rt.start(workflow=parent, vars={"workspace_root": str(tmp_path / "parent_ws")})

    st = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    assert st.status == RunStatus.WAITING
    assert st.waiting is not None
    assert st.waiting.reason == WaitReason.SUBWORKFLOW

    wait_key = str(st.waiting.wait_key or "")
    sub_run_id = wait_key.split(":", 1)[1]
    child_run = rt.run_store.load(sub_run_id)
    assert child_run is not None
    assert child_run.vars.get("workspace_root") == explicit

