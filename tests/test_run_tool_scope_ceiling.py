"""Tool approval can remove a prompt, never widen an explicit enabled set."""

from __future__ import annotations

from copy import deepcopy

import pytest

from abstractruntime import (
    Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, RunState,
    RunStatus, Runtime, StepPlan, WorkflowRegistry, WorkflowSpec,
)
from abstractruntime.core.tool_scope import ToolScopeError, resolve_tool_scope
from abstractruntime.integrations.abstractcore.effect_handlers import (
    make_tool_calls_handler, make_tool_invoke_handler,
)
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor, ToolApprovalPolicy,
)
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore


_ABSENT = object()


def _run(store, scope=_ABSENT, *, parent=None):
    run = RunState.new(
        workflow_id="test", entry_node="tools", parent_run_id=parent,
        vars={} if scope is _ABSENT else {"_runtime": {"allowed_tools": scope}},
    )
    store.save(run)
    return run


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, *, tool_calls):
        self.calls.extend(deepcopy(tool_calls))
        return {"mode": "executed", "results": [
            {"name": tc["name"], "call_id": tc.get("call_id"), "success": True, "output": "ok"}
            for tc in tool_calls
        ]}


def _calls(*names):
    return [{"name": name, "arguments": {}, "call_id": f"c{idx}"} for idx, name in enumerate(names)]


def test_absent_is_unrestricted_and_explicit_empty_denies_everything():
    store = InMemoryRunStore()
    assert resolve_tool_scope(_run(store), run_store=store) is None
    assert resolve_tool_scope(_run(store, []), run_store=store) == frozenset()
    assert resolve_tool_scope(_run(store), local_scope={"allowed_tools": []}) == frozenset()


def test_scope_intersects_every_ancestor_and_effect_without_mutating_them():
    store = InMemoryRunStore()
    root = _run(store, ["alpha", "beta"])
    wrapper = _run(store, parent=root.run_id)
    child = _run(store, ["beta", "gamma"], parent=wrapper.run_id)
    local = {"allowed_tools": [" beta ", "gamma", None, 3]}
    original = deepcopy((root.vars, wrapper.vars, child.vars, local))
    assert resolve_tool_scope(child, run_store=store, local_scope=local) == {"beta"}
    assert (root.vars, wrapper.vars, child.vars, local) == original


@pytest.mark.parametrize("malformed", [None, "alpha", {}, True, 7])
def test_malformed_explicit_allowlist_is_not_unrestricted(malformed):
    store = InMemoryRunStore()
    with pytest.raises(ToolScopeError, match="must be a list"):
        resolve_tool_scope(_run(store, malformed), run_store=store)


@pytest.mark.parametrize("failure", ["missing", "cycle", "unavailable", "io", "mismatched"])
def test_unverifiable_ancestry_fails_closed(failure):
    store = InMemoryRunStore()
    root = _run(store, ["alpha"])
    child = _run(store, parent=root.run_id)
    if failure == "missing":
        child.parent_run_id = "unknown-parent"
    elif failure == "cycle":
        root.parent_run_id = child.run_id
        store.save(root)
    elif failure == "unavailable":
        store = None
    elif failure == "io":
        class BrokenStore:
            def load(self, run_id):
                raise OSError("unavailable")
        store = BrokenStore()
    elif failure == "mismatched":
        class WrongStore:
            def load(self, run_id):
                return child
        store = WrongStore()
    with pytest.raises(ToolScopeError, match="Cannot resolve tool scope"):
        resolve_tool_scope(child, run_store=store)


def test_deep_ancestry_is_bounded_and_does_not_become_unrestricted():
    store = InMemoryRunStore()
    parent = _run(store)
    for _ in range(256):
        parent = _run(store, parent=parent.run_id)
    with pytest.raises(ToolScopeError, match="exceeds"):
        resolve_tool_scope(parent, run_store=store)


def test_initial_calls_intersect_effect_and_run_and_preserve_result_positions():
    store = InMemoryRunStore()
    root = _run(store, ["alpha", "beta"])
    child = _run(store, ["alpha", "beta", "gamma"], parent=root.run_id)
    tools = RecordingExecutor()
    handler = make_tool_calls_handler(tools=tools, run_store=store)
    out = handler(child, Effect(type=EffectType.TOOL_CALLS, payload={
        "tool_calls": _calls("gamma", "alpha", "beta"), "allowed_tools": ["beta", "gamma"],
    }), None)
    assert out.status == "completed"
    assert [call["name"] for call in tools.calls] == ["beta"]
    results = out.result["results"]
    assert [(result["name"], result["success"]) for result in results] == [
        ("gamma", False), ("alpha", False), ("beta", True),
    ]
    assert [result["call_id"] for result in results] == ["c0", "c1", "c2"]


def test_explicit_empty_blocks_authored_tool_invoke_even_without_approval():
    store = InMemoryRunStore()
    tools = RecordingExecutor()
    handler = make_tool_invoke_handler(tools=tools, run_store=store)
    out = handler(_run(store, []), Effect(type=EffectType.TOOL_INVOKE, payload={
        "name": "alpha", "arguments": {},
    }), None)
    assert out.status == "completed"
    assert out.result["results"][0]["success"] is False
    assert "not allowed" in out.result["results"][0]["error"]
    assert tools.calls == []


def test_scope_failure_reaches_no_executor_or_preexecution_helpers():
    store = InMemoryRunStore()
    tools = RecordingExecutor()
    handler = make_tool_calls_handler(tools=tools, run_store=store)
    out = handler(_run(store, parent="missing"), Effect(type=EffectType.TOOL_CALLS, payload={
        "tool_calls": _calls("open_attachment", "alpha"),
    }), None)
    assert out.status == "failed"
    assert "parent run" in out.error
    assert not out.retryable
    assert tools.calls == []


def _approval_workflow():
    def tools_node(run, ctx):
        return StepPlan(node_id="tools", next_node="done", effect=Effect(
            type=EffectType.TOOL_CALLS, result_key="tool_results",
            payload={"tool_calls": _calls("outside", "alpha", "beta"), "allowed_tools": ["alpha", "beta"]},
        ))

    def done_node(run, ctx):
        return StepPlan(node_id="done", complete_output={"tool_results": run.vars["tool_results"]})

    return WorkflowSpec(workflow_id="scope-approval", entry_node="tools", nodes={"tools": tools_node, "done": done_node})


@pytest.mark.parametrize("narrowed", [["beta"], []])
def test_approval_resume_rechecks_ancestor_and_merges_original_blocked_indices(narrowed):
    store = InMemoryRunStore()
    parent = _run(store, ["alpha", "beta"])
    delegate = RecordingExecutor()
    tools = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy(
        auto_approve_tools=set(), require_approval_tools={"alpha", "beta"},
    ))
    # Deliberately omit the handler store: the runtime's transient store must
    # protect old host construction paths as well as new explicit wiring.
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore(), effect_handlers={
        EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools),
    })
    rt.set_tool_executor_for_resume(tools)
    workflow = _approval_workflow()
    run_id = rt.start(workflow=workflow, parent_run_id=parent.run_id)
    state = rt.tick(workflow=workflow, run_id=run_id)
    assert state.status == RunStatus.WAITING
    assert [tc["name"] for tc in state.waiting.details["tool_calls"]] == ["alpha", "beta"]
    assert delegate.calls == []
    parent.vars["_runtime"]["allowed_tools"] = narrowed
    store.save(parent)
    result = rt.resume(workflow=workflow, run_id=run_id, wait_key=state.waiting.wait_key, payload={"approved": True})
    assert result.status == RunStatus.COMPLETED
    assert [tc["name"] for tc in delegate.calls] == narrowed
    results = result.output["tool_results"]["results"]
    assert [r["name"] for r in results] == ["outside", "alpha", "beta"]
    assert [r["call_id"] for r in results] == ["c0", "c1", "c2"]
    assert [r["success"] for r in results] == [False, False, "beta" in narrowed]
    assert "not allowed" in results[1]["error"]


def test_approval_resume_missing_ancestor_cannot_execute():
    store = InMemoryRunStore()
    delegate = RecordingExecutor()
    tools = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy(
        auto_approve_tools=set(), require_approval_tools={"alpha", "beta"},
    ))
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore(), effect_handlers={
        EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools),
    })
    rt.set_tool_executor_for_resume(tools)
    workflow = _approval_workflow()
    run_id = rt.start(workflow=workflow)
    state = rt.tick(workflow=workflow, run_id=run_id)
    assert state.status == RunStatus.WAITING
    state.parent_run_id = "missing"
    store.save(state)
    result = rt.resume(workflow=workflow, run_id=run_id, wait_key=state.waiting.wait_key, payload={"approved": True})
    assert result.status == RunStatus.COMPLETED
    assert delegate.calls == []
    results = result.output["tool_results"]["results"]
    assert all(not item["success"] for item in results)
    assert "parent run" in results[1]["error"]


def test_persisted_approval_after_runtime_restart_reloads_the_ancestor_ceiling(tmp_path):
    store = JsonFileRunStore(tmp_path / "runs")
    parent = _run(store, ["alpha", "beta"])
    delegate = RecordingExecutor()
    tools = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy(
        auto_approve_tools=set(), require_approval_tools={"alpha", "beta"},
    ))
    workflow = _approval_workflow()
    runtime = Runtime(run_store=store, ledger_store=JsonlLedgerStore(tmp_path / "ledger"), effect_handlers={
        EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools),
    })
    run_id = runtime.start(workflow=workflow, parent_run_id=parent.run_id)
    state = runtime.tick(workflow=workflow, run_id=run_id)
    assert state.status == RunStatus.WAITING
    wait_key = state.waiting.wait_key
    parent.vars["_runtime"]["allowed_tools"] = []
    store.save(parent)

    # A fresh store and Runtime must not rely on the former runtime's object
    # aliases, transient store stamp, or remembered approval decision.
    restarted = Runtime(
        run_store=JsonFileRunStore(tmp_path / "runs"), ledger_store=JsonlLedgerStore(tmp_path / "ledger"),
    )
    restarted.set_tool_executor_for_resume(tools)
    result = restarted.resume(workflow=workflow, run_id=run_id, wait_key=wait_key, payload={"approved": True})
    assert result.status == RunStatus.COMPLETED
    assert delegate.calls == []
    assert all(not item["success"] for item in result.output["tool_results"]["results"])
    assert "not allowed" in result.output["tool_results"]["results"][1]["error"]


@pytest.mark.parametrize("parent_scope,child_scope,expected", [
    (_ABSENT, _ABSENT, _ABSENT),
    (_ABSENT, ["alpha"], ["alpha"]),
    (["alpha"], _ABSENT, ["alpha"]),
    (["alpha"], ["alpha", "read_skill", "beta"], ["alpha"]),
    ([], ["alpha"], []),
    (["alpha"], [], []),
])
def test_child_tool_scope_is_inherited_without_widening_or_fabricating_defaults(parent_scope, child_scope, expected):
    child = WorkflowSpec(workflow_id="child", entry_node="done", nodes={
        "done": lambda run, ctx: StepPlan(node_id="done", complete_output={"ok": True}),
    })
    child_vars = {} if child_scope is _ABSENT else {"_runtime": {"allowed_tools": child_scope}}
    original = deepcopy(child_vars)
    parent = WorkflowSpec(workflow_id="parent", entry_node="spawn", nodes={
        "spawn": lambda run, ctx: StepPlan(node_id="spawn", effect=Effect(
            type=EffectType.START_SUBWORKFLOW, result_key="child",
            payload={"workflow_id": "child", "vars": child_vars, "async": True, "wait": True},
        )),
    })
    registry = WorkflowRegistry()
    registry.register(child)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=registry)
    parent_vars = {} if parent_scope is _ABSENT else {"_runtime": {"allowed_tools": parent_scope}}
    run_id = rt.start(workflow=parent, vars=parent_vars)
    state = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    assert state.status == RunStatus.WAITING
    child_id = state.waiting.wait_key.split(":", 1)[1]
    child_runtime = rt.run_store.load(child_id).vars.get("_runtime", {})
    if expected is _ABSENT:
        assert "allowed_tools" not in child_runtime
    else:
        assert child_runtime["allowed_tools"] == expected
        child_runtime["allowed_tools"].append("unrelated")
    assert child_vars == original
    if parent_scope is not _ABSENT:
        assert rt.run_store.load(run_id).vars["_runtime"]["allowed_tools"] == parent_scope


def test_subworkflow_missing_ancestor_fails_without_creating_child():
    child = WorkflowSpec(workflow_id="child", entry_node="done", nodes={
        "done": lambda run, ctx: StepPlan(node_id="done", complete_output={"ok": True}),
    })
    registry = WorkflowRegistry()
    registry.register(child)
    store = InMemoryRunStore()
    parent = _run(store, parent="missing")
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore(), workflow_registry=registry)
    outcome = rt._handle_start_subworkflow(parent, Effect(
        type=EffectType.START_SUBWORKFLOW, payload={"workflow_id": "child", "async": True},
    ), None)
    assert outcome.status == "failed"
    assert "parent run" in outcome.error
    assert store.list_children(parent_run_id=parent.run_id) == []
