"""Read-only workspaces (automations contract C4, runtime half).

A discussion forked from an automation runs on the occurrence's workspace
mounted read-only (`workspace_read_only`). Pinned here:

* ONE classification table (`tool_effects.TOOL_EFFECT_CLASSES`) covers every
  tool the runtime can expose — RED when a new tool arrives unclassified;
* under read-only, write/exec tools and unclassified tools are refused, reads
  run; a missing root is refused and never created;
* the flag reaches child runs before the protection merge's early return,
  cannot be cleared by a child, and overrides VisualFlow node inputs; VisualFlow
  file/export writers refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowRegistry, WorkflowSpec
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_effects import (
    EFFECT_CLASSES,
    TOOL_EFFECT_CLASSES,
    read_only_refusal,
)
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
    WorkspaceScope,
    describe_workspace_scope,
    rewrite_tool_arguments,
)
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.utils.workspace_paths import merge_builtin_workspace_protection


# --------------------------------------------------------------------------
# 1. Classification coverage
# --------------------------------------------------------------------------

def _every_default_tool_name(monkeypatch) -> List[str]:
    import abstractruntime.integrations.abstractcore.default_tools as dt

    for gate in ("comms_tools_enabled", "email_tools_enabled", "whatsapp_tools_enabled",
                 "telegram_tools_enabled", "agora_tools_enabled", "shell_tools_enabled"):
        monkeypatch.setattr(dt, gate, lambda: True)
    return [str(spec["name"]) for spec in dt.list_default_tool_specs()]


def test_every_exposable_runtime_tool_is_classified(monkeypatch) -> None:
    names = _every_default_tool_name(monkeypatch)
    assert {"write_file", "execute_command", "shell_exec", "local_helper_start", "open_attachment"} <= set(names)
    unclassified = sorted(n for n in names if n not in TOOL_EFFECT_CLASSES)
    assert not unclassified, (
        "exposed tool(s) with no effect class: " + ", ".join(unclassified)
        + " — add each to tool_effects.TOOL_EFFECT_CLASSES (unclassified tools are refused in read-only workspaces)"
    )
    assert set(TOOL_EFFECT_CLASSES.values()) <= set(EFFECT_CLASSES)


def test_abstractagent_tools_the_gateway_exposes_are_classified() -> None:
    agent_tools = pytest.importorskip("abstractagent.tools")  # not a runtime dependency
    from abstractagent.logic import builtins

    names = {getattr(t, "_tool_definition", None).name if getattr(t, "_tool_definition", None) else t.__name__
             for t in agent_tools.ALL_TOOLS}
    names |= {getattr(builtins, n).name for n in dir(builtins) if n.endswith("_TOOL")}
    assert "execute_python" in names
    assert sorted(n for n in names if n not in TOOL_EFFECT_CLASSES) == []


def test_execute_python_is_classified_exec_without_abstractagent() -> None:
    # The gateway's CodeAct loop exposes it; the classification must not depend
    # on abstractagent being importable here.
    assert TOOL_EFFECT_CLASSES["execute_python"] == "exec"


# --------------------------------------------------------------------------
# 2. Tool refusals
# --------------------------------------------------------------------------

@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "occurrence-ws"
    root.mkdir()
    (root / "report.md").write_text("price: 101.2")
    return root


def _ro_scope(ws: Path, **extra: Any) -> WorkspaceScope:
    scope = WorkspaceScope.from_input_data({"workspace_root": str(ws), "workspace_read_only": True, **extra})
    assert scope is not None and scope.read_only
    return scope


@pytest.mark.parametrize(
    "tool,args",
    [
        ("write_file", {"file_path": "x.md", "content": "y"}),
        ("edit_file", {"file_path": "report.md", "pattern": "1", "replacement": "2"}),
        ("execute_command", {"command": "touch x"}),
        ("shell_exec", {"command": "touch x"}),
        ("local_helper_start", {"command": "python -m http.server"}),
        ("execute_python", {"code": "open('x','w').write('y')"}),
        ("some_new_unclassified_tool", {"path": "x"}),
    ],
)
def test_write_exec_and_unclassified_tools_are_refused(ws, tool, args) -> None:
    with pytest.raises(ValueError, match="read-only"):
        rewrite_tool_arguments(tool_name=tool, args=args, scope=_ro_scope(ws))


def test_reads_still_run_and_a_writable_scope_is_unchanged(ws) -> None:
    out = rewrite_tool_arguments(tool_name="read_file", args={"file_path": "report.md"}, scope=_ro_scope(ws))
    assert out["file_path"] == str((ws / "report.md").resolve())
    assert rewrite_tool_arguments(tool_name="list_files", args={}, scope=_ro_scope(ws))["directory_path"]
    assert read_only_refusal("web_search") is None
    writable = WorkspaceScope.from_input_data({"workspace_root": str(ws)})
    assert not writable.read_only
    rewrite_tool_arguments(tool_name="write_file", args={"file_path": "x.md"}, scope=writable)


def test_trusted_runtime_policy_key_also_makes_it_read_only(ws) -> None:
    scope = WorkspaceScope.from_input_data({"workspace_root": str(ws), "_runtime": {"workspace_read_only": True}})
    assert scope.read_only
    assert "READ-ONLY" in describe_workspace_scope(scope)


def test_a_missing_read_only_root_is_refused_and_never_created(tmp_path) -> None:
    missing = tmp_path / "gone"
    with pytest.raises(ValueError, match="does not exist"):
        WorkspaceScope.from_input_data({"workspace_root": str(missing), "workspace_read_only": True})
    assert not missing.exists()
    with pytest.raises(ValueError, match="no workspace_root"):
        WorkspaceScope.from_input_data({"workspace_read_only": True})
    # Writable scopes keep creating their root.
    WorkspaceScope.from_input_data({"workspace_root": str(missing)})
    assert missing.is_dir()


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def execute(self, *, tool_calls):
        self.calls.extend(tool_calls)
        return {"mode": "executed", "results": [
            {"call_id": c["call_id"], "name": c["name"], "success": True, "output": "ok", "error": None} for c in tool_calls]}


def test_the_tool_calls_effect_refuses_writes_and_runs_reads(ws) -> None:
    executor = _RecordingExecutor()
    handler = make_tool_calls_handler(tools=executor, run_store=InMemoryRunStore())
    run = RunState.new(workflow_id="wf", entry_node="n", vars={"workspace_root": str(ws), "workspace_read_only": True})
    outcome = handler(run, Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [
        {"call_id": "w", "name": "write_file", "arguments": {"file_path": "evil.md", "content": "x"}},
        {"call_id": "x", "name": "execute_python", "arguments": {"code": "print(1)"}},
        {"call_id": "r", "name": "read_file", "arguments": {"file_path": "report.md"}},
    ]}), None)
    results = {r["call_id"]: r for r in outcome.result["results"]}
    assert results["w"]["success"] is False and "read-only" in results["w"]["error"]
    assert results["x"]["success"] is False and "read-only" in results["x"]["error"]
    assert results["r"]["success"] is True
    assert [c["name"] for c in executor.calls] == ["read_file"]
    assert not (ws / "evil.md").exists()


def test_the_tool_calls_effect_fails_closed_without_a_root() -> None:
    handler = make_tool_calls_handler(tools=_RecordingExecutor(), run_store=InMemoryRunStore())
    run = RunState.new(workflow_id="wf", entry_node="n", vars={"workspace_read_only": True})
    outcome = handler(run, Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [
        {"call_id": "w", "name": "write_file", "arguments": {"file_path": "/tmp/x", "content": "x"}}]}), None)
    assert outcome.status == "failed"


# --------------------------------------------------------------------------
# 3. Propagation
# --------------------------------------------------------------------------

def test_merge_propagates_read_only_even_without_host_protection() -> None:
    assert merge_builtin_workspace_protection({"workspace_read_only": True}, {"workspace_read_only": False}) == {
        "workspace_read_only": True}
    assert merge_builtin_workspace_protection({"_runtime": {"workspace_read_only": True}}, {}) == {
        "workspace_read_only": True}
    merged = merge_builtin_workspace_protection(
        {"workspace_read_only": True, "workspace_builtin_deny_prefixes": ["/data"]}, {})
    assert merged["workspace_read_only"] is True and merged["workspace_builtin_deny_prefixes"] == ["/data"]
    assert merge_builtin_workspace_protection({}, {"workspace_read_only": True}) == {}


def test_a_child_run_cannot_clear_the_flag(ws) -> None:
    child = WorkflowSpec(workflow_id="child", entry_node="n", nodes={
        "n": lambda r, c: StepPlan(node_id="n", complete_output={})})
    parent = WorkflowSpec(workflow_id="parent", entry_node="spawn", nodes={
        "spawn": lambda r, c: StepPlan(node_id="spawn", effect=Effect(
            type=EffectType.START_SUBWORKFLOW,
            payload={"workflow_id": "child", "vars": {"workspace_read_only": False}, "async": True, "wait": True},
            result_key="child"))})
    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    rid = rt.start(workflow=parent, vars={"workspace_root": str(ws), "workspace_read_only": True})
    state = rt.tick(workflow=parent, run_id=rid, max_steps=1)
    child_vars = rt.get_state(state.waiting.wait_key.split(":", 1)[1]).vars
    assert child_vars["workspace_read_only"] is True
    assert WorkspaceScope.from_input_data(child_vars).read_only


def _visual_writer_flow(node_type: str, file_path: str, extra_inputs: Dict[str, Any]):
    from abstractruntime.visualflow_compiler.compiler import compile_flow
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    inputs = [{"id": "exec-in", "label": "", "type": "execution"},
              {"id": "file_path", "label": "file_path", "type": "string"},
              {"id": "content", "label": "content", "type": "string"}]
    inputs += [{"id": k, "label": k, "type": "boolean"} for k in extra_inputs]
    vf = load_visualflow_json({
        "id": f"t-ro-{node_type}", "name": f"t-ro-{node_type}",
        "nodes": [
            {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
            # The writer's payload is this node's output, which carries no
            # workspace keys: only the compiler's ambient policy can bring the flag.
            {"id": "prev", "type": "code", "data": {"nodeType": "code", "codeBody": "return {'a': 1}"}},
            {"id": "io", "type": node_type, "data": {"nodeType": node_type, "inputs": inputs,
                                                     "pinDefaults": {"file_path": file_path, "content": "x", **extra_inputs}}},
            {"id": "end", "type": "on_flow_end", "data": {"nodeType": "on_flow_end", "inputs": [
                {"id": "exec-in", "label": "", "type": "execution"}]}},
        ],
        "edges": [
            {"source": "start", "sourceHandle": "exec-out", "target": "prev", "targetHandle": "exec-in"},
            {"source": "prev", "sourceHandle": "exec-out", "target": "io", "targetHandle": "exec-in"},
            {"source": "io", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
        ],
    })
    return compile_flow(visual_to_flow(vf))


@pytest.mark.parametrize("pins", [{}, {"workspace_read_only": False}], ids=["ambient", "pin-says-false"])
@pytest.mark.parametrize("node_type,name", [("write_file", "evil.md"), ("write_pdf", "evil.pdf")])
def test_visualflow_writers_refuse_and_node_inputs_cannot_clear_the_flag(ws, node_type, name, pins) -> None:
    workflow = _visual_writer_flow(node_type, name, pins)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=workflow, vars={"workspace_root": str(ws), "workspace_read_only": True})
    state = rt.tick(workflow=workflow, run_id=rid, max_steps=20)
    assert not (ws / name).exists()
    assert "read-only" in str(state.error or state.output)


def test_visualflow_reads_still_work_under_read_only(ws) -> None:
    workflow = _visual_writer_flow("read_file", "report.md", {})
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=workflow, vars={"workspace_root": str(ws), "workspace_read_only": True})
    state = rt.tick(workflow=workflow, run_id=rid, max_steps=20)
    assert state.status == RunStatus.COMPLETED, state.error
    assert "read-only" not in str(state.output)


def test_visualflow_writer_still_writes_when_not_read_only(ws) -> None:
    workflow = _visual_writer_flow("write_file", "ok.md", {})
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=workflow, vars={"workspace_root": str(ws)})
    rt.tick(workflow=workflow, run_id=rid, max_steps=20)
    assert (ws / "ok.md").read_text() == "x"


# --------------------------------------------------------------------------
# 4. Read-only MOUNTS (operator ruling 2026-09-27): the discussion's own
#    workspace is writable; the automation's workspace is mounted read-only.
# --------------------------------------------------------------------------

import os  # noqa: E402

from abstractruntime.utils.workspace_paths import path_is_read_only, read_only_paths  # noqa: E402


@pytest.fixture()
def mounted(tmp_path: Path):
    own = tmp_path / "discussion-ws"
    own.mkdir()
    mount = tmp_path / "automation-ws"
    mount.mkdir()
    (mount / "report.md").write_text("price: 101.2")
    link = tmp_path / "mount-link"
    link.symlink_to(mount)
    vars_ = {"workspace_root": str(own), "workspace_access_mode": "all_except_ignored",
             "_runtime": {"workspace_read_only_paths": [str(link)]}}  # realpath'd like `pwd -P`
    return own, mount, vars_


def test_mount_paths_are_realpathed_and_detected(mounted) -> None:
    own, mount, vars_ = mounted
    assert read_only_paths(vars_) == (os.path.realpath(str(mount)),)
    assert path_is_read_only(vars_, mount / "report.md")
    assert not path_is_read_only(vars_, own / "notes.md")


def test_writes_into_a_mount_are_refused_reads_and_exec_are_allowed(mounted) -> None:
    own, mount, vars_ = mounted
    scope = WorkspaceScope.from_input_data(vars_)
    assert not scope.read_only and scope.read_only_paths
    for tool, args in (("write_file", {"file_path": str(mount / "x.md"), "content": "y"}),
                       ("edit_file", {"file_path": str(mount / "report.md"), "pattern": "1", "replacement": "2"})):
        with pytest.raises(ValueError, match="read-only mount") as info:
            rewrite_tool_arguments(tool_name=tool, args=args, scope=scope)
        assert str(info.value).startswith(f"Tool '{tool}' is refused")
    # Own workspace: writable.
    out = rewrite_tool_arguments(tool_name="write_file", args={"file_path": "notes.md", "content": "y"}, scope=scope)
    assert out["file_path"] == str((own / "notes.md").resolve())
    # Reads inside the mount and exec tools are allowed.
    rewrite_tool_arguments(tool_name="read_file", args={"file_path": str(mount / "report.md")}, scope=scope)
    rewrite_tool_arguments(tool_name="execute_command", args={"command": "ls"}, scope=scope)
    rewrite_tool_arguments(tool_name="execute_python", args={"code": "print(1)"}, scope=scope)
    assert "Read-only mounts" in describe_workspace_scope(scope)


def test_a_child_cannot_clear_or_shrink_the_mounts(mounted, tmp_path) -> None:
    own, mount, vars_ = mounted
    extra = tmp_path / "extra"
    extra.mkdir()
    merged = merge_builtin_workspace_protection(vars_, {"workspace_read_only_paths": [str(extra)],
                                                        "_runtime": {"workspace_read_only_paths": []}})
    assert merged["workspace_read_only_paths"] == sorted([os.path.realpath(str(mount)), os.path.realpath(str(extra))])

    child = WorkflowSpec(workflow_id="child", entry_node="n", nodes={
        "n": lambda r, c: StepPlan(node_id="n", complete_output={})})
    parent = WorkflowSpec(workflow_id="parent", entry_node="spawn", nodes={
        "spawn": lambda r, c: StepPlan(node_id="spawn", effect=Effect(
            type=EffectType.START_SUBWORKFLOW,
            payload={"workflow_id": "child", "async": True, "wait": True,
                     "vars": {"workspace_read_only_paths": [], "_runtime": {"workspace_read_only_paths": []}}},
            result_key="child"))})
    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    rid = rt.start(workflow=parent, vars=dict(vars_))
    state = rt.tick(workflow=parent, run_id=rid, max_steps=1)
    child_vars = rt.get_state(state.waiting.wait_key.split(":", 1)[1]).vars
    assert path_is_read_only(child_vars, mount / "report.md")
    with pytest.raises(ValueError, match="read-only mount"):
        rewrite_tool_arguments(tool_name="write_file", args={"file_path": str(mount / "x"), "content": "y"},
                               scope=WorkspaceScope.from_input_data(child_vars))


@pytest.mark.parametrize("pins", [{}, {"workspace_read_only_paths": []}], ids=["ambient", "pin-says-none"])
def test_visualflow_writers_refuse_into_a_mount_and_write_in_the_own_root(mounted, pins) -> None:
    own, mount, vars_ = mounted
    target = str(mount / "evil.md")
    workflow = _visual_writer_flow("write_file", target, pins)
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=workflow, vars=dict(vars_))
    state = rt.tick(workflow=workflow, run_id=rid, max_steps=20)
    assert not (mount / "evil.md").exists()
    assert "read-only mount" in str(state.error or state.output)

    ok = _visual_writer_flow("write_file", "mine.md", {})
    rid2 = rt.start(workflow=ok, vars=dict(vars_))
    rt.tick(workflow=ok, run_id=rid2, max_steps=20)
    assert (own / "mine.md").read_text() == "x"
