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


def test_start_subworkflow_inherits_skills_block(tmp_path) -> None:
    """0087 adversary P1-2 (2026-07-15): the skills teaching held exactly ONE
    composition level — wrapping an Agent node in a subflow (the ordinary
    modularization gesture) silently stripped `_runtime.skills_block` from
    the child, so the grandchild agent lost its skills with no warning.
    Same setdefault semantics as the workspace keys."""

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
    run_id = rt.start(
        workflow=parent,
        vars={"_runtime": {"skills_block": "## Skills\n- how to greet"}},
    )

    st = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    assert st.status == RunStatus.WAITING
    sub_run_id = str(st.waiting.wait_key or "").split(":", 1)[1]
    child_run = rt.run_store.load(sub_run_id)
    assert child_run is not None
    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("skills_block") == "## Skills\n- how to greet", (
        "the block must cross the subflow hop (setdefault, like workspace keys)"
    )


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



def test_start_subworkflow_inherits_tool_policy(tmp_path) -> None:
    """coder-tui c4384 live find (2026-07-22, the skills_block P1-2 class
    exactly): thin clients send `_runtime.tool_policy` on the ROOT run, but
    bundle-hosted agents execute tool_calls in CHILD runs whose fresh vars
    never carried it - the run-policy consumer found nothing and the
    approval wait still round-tripped despite the client's accepted-tier
    policy. Same setdefault semantics; an explicit child policy wins."""

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

    policy = {"auto_approve_tools": ["write_file", "edit_file"], "require_approval_tools": []}
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    run_id = rt.start(workflow=parent, vars={"_runtime": {"tool_policy": policy}})

    st = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    assert st.status == RunStatus.WAITING
    sub_run_id = str(st.waiting.wait_key or "").split(":", 1)[1]
    child_run = rt.run_store.load(sub_run_id)
    assert child_run is not None
    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("tool_policy") == policy, (
        "the per-run tool policy must cross the subflow hop or server-side "
        "auto-approve never fires on bundle-hosted agents"
    )
    # A COPY crosses, never the parent's aliased dict (a child mutation
    # must not rewrite the parent's policy).
    assert child_rt["tool_policy"] is not policy


def test_start_subworkflow_inherits_operator_email(tmp_path) -> None:
    """gateway c4702 / dm#246 (the FOURTH rider of the skills_block P1-2
    class): operator_email must cross the START_SUBWORKFLOW hop or the
    send_email recipient refiner in a bundle agent's CHILD run sees no
    self-value and the self-send auto-path is dead. setdefault; a string
    copied by assignment; explicit child value wins."""

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
    run_id = rt.start(workflow=parent, vars={"_runtime": {"operator_email": "op@self.com"}})

    st = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    assert st.status == RunStatus.WAITING
    sub_run_id = str(st.waiting.wait_key or "").split(":", 1)[1]
    child_run = rt.run_store.load(sub_run_id)
    assert child_run is not None
    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("operator_email") == "op@self.com", (
        "the self-value must cross the hop or the refiner is dead in child runs"
    )
