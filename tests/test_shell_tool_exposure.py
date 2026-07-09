"""Persistent shell tool exposure (backlog 0220).

Pins the safety posture: run-id namespace stamping (trust boundary), approval gating,
terminal-seam teardown (completed AND cancelled), opt-in absence from default toolsets,
honest schema wording, and the new-session (non-durability) notice.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pytest

from abstractruntime import Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunState
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.factory import register_shell_session_teardown
from abstractruntime.integrations.abstractcore.tool_executor import (
    _DEFAULT_REQUIRE_APPROVAL,
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)

from abstractcore.tools.shell_session import get_shell_session_registry, namespaced_session_id
from abstractcore.tools.shell_tools import SHELL_TOOLS, shell_close, shell_exec, shell_write_stdin

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shells only")


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    get_shell_session_registry().close_all()


def _executor() -> MappingToolExecutor:
    return MappingToolExecutor.from_tools(list(SHELL_TOOLS))


def _handler():
    return make_tool_calls_handler(tools=_executor())


def _shell_effect(command: str, *, session_id: str = "main", extra_args: Dict[str, Any] | None = None, call_id: str = "c1") -> Effect:
    args: Dict[str, Any] = {"command": command, "session_id": session_id}
    if extra_args:
        args.update(extra_args)
    return Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"name": "shell_exec", "arguments": args, "call_id": call_id}]},
    )


def _run_effect(handler, run: RunState, effect: Effect) -> Dict[str, Any]:
    outcome = handler(run, effect, None)
    assert str(outcome.status) == "completed", outcome
    return outcome.result["results"][0]


# ---------------------------------------------------------------------------
# State persistence + namespace stamping
# ---------------------------------------------------------------------------

def test_cwd_and_env_persist_across_calls_within_run(tmp_path):
    handler = _handler()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars={})

    r1 = _run_effect(handler, run, _shell_effect(f"cd '{tmp_path}' && export MARKER=xyz42 && echo ready"))
    assert r1["success"] is True
    assert "new shell session" in str(r1["output"])  # first call announces the fresh session

    r2 = _run_effect(handler, run, _shell_effect("pwd && echo $MARKER", call_id="c2"))
    out2 = str(r2["output"])
    assert str(tmp_path) in out2 and "xyz42" in out2
    assert "new shell session" not in out2  # same session, no re-open notice

    # The session is registered under THIS run's namespace.
    assert get_shell_session_registry().get(namespaced_session_id(run.run_id, "main")) is not None


def test_model_supplied_namespace_is_overwritten(tmp_path):
    """A tool call claiming another run's namespace must be re-stamped with the real run id."""
    handler = _handler()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars={})

    r = _run_effect(handler, run, _shell_effect("echo hi", extra_args={"_registry_namespace": "victim-run"}))
    assert r["success"] is True

    registry = get_shell_session_registry()
    assert registry.get(namespaced_session_id("victim-run", "main")) is None
    assert registry.get(namespaced_session_id(run.run_id, "main")) is not None


def test_two_runs_get_isolated_sessions():
    handler = _handler()
    run_a = RunState.new(workflow_id="wf", entry_node="n", session_id="sa", vars={})
    run_b = RunState.new(workflow_id="wf", entry_node="n", session_id="sb", vars={})

    _run_effect(handler, run_a, _shell_effect("export WHO=alpha && echo ok"))
    _run_effect(handler, run_b, _shell_effect("export WHO=beta && echo ok"))
    ra = _run_effect(handler, run_a, _shell_effect("echo $WHO", call_id="c2"))
    rb = _run_effect(handler, run_b, _shell_effect("echo $WHO", call_id="c2"))

    assert "alpha" in str(ra["output"]) and "beta" in str(rb["output"])


# ---------------------------------------------------------------------------
# Approval gating
# ---------------------------------------------------------------------------

def test_shell_tools_require_approval_by_default():
    assert {"shell_exec", "shell_write_stdin", "shell_close"} <= _DEFAULT_REQUIRE_APPROVAL
    tools = ApprovalToolExecutor(delegate=_executor(), policy=ToolApprovalPolicy())
    out = tools.execute(tool_calls=[{"name": "shell_exec", "arguments": {"command": "echo hi"}, "call_id": "c1"}])
    assert out.get("mode") == "approval_required"


def test_approval_resume_executes_shell_with_stamped_namespace():
    """Full wait/resume path: the approved-resume executes with the namespace stamped at plan time."""
    tools = ApprovalToolExecutor(delegate=_executor(), policy=ToolApprovalPolicy())
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )
    runtime.set_tool_executor_for_resume(tools)
    register_shell_session_teardown(runtime)

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="tools",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "shell_exec", "arguments": {"command": "echo approved-run"}, "call_id": "c1"}]},
                result_key="tool_results",
            ),
            next_node="wait_forever",
        )

    def wait_forever(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="wait_forever",
            effect=Effect(type=EffectType.WAIT_EVENT, payload={"wait_key": "park"}),
            next_node="wait_forever",
        )

    workflow = WorkflowSpec(workflow_id="shell_approval_test", entry_node="tools", nodes={"tools": tools_node, "wait_forever": wait_forever})

    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id)
    assert state.status == RunStatus.WAITING
    assert state.waiting is not None and state.waiting.details.get("mode") == "approval_required"

    # The stored wait carries the stamped namespace (not model-controllable).
    stored_calls = state.waiting.details.get("tool_calls")
    assert stored_calls and stored_calls[0]["arguments"]["_registry_namespace"] == run_id

    resumed = runtime.resume(workflow=workflow, run_id=run_id, wait_key=state.waiting.wait_key, payload={"approved": True}, max_steps=5)
    tool_results = resumed.vars.get("tool_results")
    results = tool_results.get("results") if isinstance(tool_results, dict) else None
    assert results and results[0]["success"] is True and "approved-run" in str(results[0]["output"])
    # Session exists under the run's namespace while the run is alive (parked, not terminal).
    assert get_shell_session_registry().get(namespaced_session_id(run_id, "main")) is not None

    # Cancel (terminal transition) tears the session down.
    runtime.cancel_run(run_id, reason="test cleanup")
    assert get_shell_session_registry().get(namespaced_session_id(run_id, "main")) is None


# ---------------------------------------------------------------------------
# Terminal teardown (completed + cancelled)
# ---------------------------------------------------------------------------

def _runtime_with_shell() -> Runtime:
    tools = _executor()
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )
    runtime.set_tool_executor_for_resume(tools)
    register_shell_session_teardown(runtime)
    return runtime


def test_completed_run_closes_its_sessions():
    runtime = _runtime_with_shell()

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="tools",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "shell_exec", "arguments": {"command": "sleep 300 & echo started"}, "call_id": "c1"}]},
                result_key="tool_results",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="done", complete_output={"ok": True})

    workflow = WorkflowSpec(workflow_id="shell_teardown_test", entry_node="tools", nodes={"tools": tools_node, "done": done_node})
    run_id = runtime.start(workflow=workflow)

    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=5)
    assert state.status == RunStatus.COMPLETED
    # Terminal seam closed the namespace (background `sleep` reaped by killpg).
    assert get_shell_session_registry().get(namespaced_session_id(run_id, "main")) is None


# ---------------------------------------------------------------------------
# Opt-in exposure + schema honesty + non-durability notice
# ---------------------------------------------------------------------------

def test_shell_tools_absent_from_default_toolsets(monkeypatch):
    monkeypatch.delenv("ABSTRACT_ENABLE_SHELL_TOOLS", raising=False)
    from abstractruntime.integrations.abstractcore.default_tools import get_default_toolsets

    toolsets = get_default_toolsets()
    assert "shell" not in toolsets
    names = {t.__name__ for spec in toolsets.values() for t in spec["tools"]}
    assert "shell_exec" not in names


def test_shell_tools_env_opt_in(monkeypatch):
    monkeypatch.setenv("ABSTRACT_ENABLE_SHELL_TOOLS", "1")
    from abstractruntime.integrations.abstractcore.default_tools import build_default_tool_map, get_default_toolsets

    toolsets = get_default_toolsets()
    assert "shell" in toolsets
    tool_map = build_default_tool_map()
    assert {"shell_exec", "shell_write_stdin", "shell_close"} <= set(tool_map)


def test_schema_wording_is_honest_and_namespace_hidden():
    d = shell_exec._tool_definition
    text = f"{d.description} {d.when_to_use or ''}".lower()
    assert "not a sandbox" in text
    assert "not durable" in text
    assert "persist" in text
    assert "_registry_namespace" not in d.parameters  # trust-boundary arg hidden from the model
    assert "_registry_namespace" not in shell_write_stdin._tool_definition.parameters
    assert "_registry_namespace" not in shell_close._tool_definition.parameters


def test_write_stdin_without_session_gives_actionable_error():
    out = shell_write_stdin(input="hello", session_id="ghost", _registry_namespace="nowhere")
    assert "no active shell session" in out and "shell_exec" in out


def test_write_stdin_feeds_interactive_process():
    ns = "test-ns-stdin"
    try:
        first = shell_exec(command="cat", timeout=2, _registry_namespace=ns)
        assert "timed out" in first  # cat waits on the tty: honest timeout, session survives
        echoed = shell_write_stdin(input="marco-polo", read_timeout=3, _registry_namespace=ns)
        assert "marco-polo" in echoed
    finally:
        shell_close(_registry_namespace=ns)


def test_shell_close_reports_state():
    ns = "test-ns-close"
    shell_exec(command="echo hi", _registry_namespace=ns)
    assert "closed" in shell_close(_registry_namespace=ns)
    assert "nothing to close" in shell_close(_registry_namespace=ns)
