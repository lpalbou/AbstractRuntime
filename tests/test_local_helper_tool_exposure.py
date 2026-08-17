"""Run-scoped local-helper tool exposure.

Pins the first bounded runtime-owned cut for long-lived local helpers:
- start/status/stop lifecycle
- run-id namespace stamping (trust boundary)
- terminal-hook teardown
- default-tool / approval-policy exposure
"""

from __future__ import annotations

import shlex
import socket
import sys
from typing import Any, Dict

import pytest

from abstractruntime import Effect, EffectType, InMemoryLedgerStore, InMemoryRunStore, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunState
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.default_tools import build_default_tool_map
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.factory import register_shell_session_teardown
from abstractruntime.integrations.abstractcore.local_helper_tools import (
    LOCAL_HELPER_TOOLS,
    get_local_helper_registry,
    local_helper_start,
    local_helper_status,
    local_helper_stop,
    namespaced_helper_id,
)
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor, ToolApprovalPolicy

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX process-group teardown only")


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    get_local_helper_registry().close_all()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _executor() -> MappingToolExecutor:
    return MappingToolExecutor.from_tools(list(LOCAL_HELPER_TOOLS))


def _handler():
    return make_tool_calls_handler(tools=_executor())


def _helper_effect(command: str, *, helper_id: str = "main", extra_args: Dict[str, Any] | None = None, call_id: str = "c1") -> Effect:
    args: Dict[str, Any] = {"command": command, "helper_id": helper_id}
    if extra_args:
        args.update(extra_args)
    return Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"name": "local_helper_start", "arguments": args, "call_id": call_id}]},
    )


def _run_effect(handler, run: RunState, effect: Effect) -> Dict[str, Any]:
    outcome = handler(run, effect, None)
    assert str(outcome.status) == "completed", outcome
    result = outcome.result["results"][0]
    output = result.get("output")
    return output if isinstance(output, dict) else result


def test_local_helper_lifecycle(tmp_path):
    port = _free_port()
    handler = _handler()
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s1", vars={})

    started = _run_effect(
        handler,
        run,
        _helper_effect(
            f"{shlex.quote(sys.executable)} -m http.server {port}",
            extra_args={"working_directory": str(tmp_path), "ready_port": port, "ready_timeout": 10},
        ),
    )
    assert started["success"] is True
    assert started["alive"] is True and started["ready"] is True
    assert started["helper_id"] == "main"
    assert started["port"] == port
    assert get_local_helper_registry().get(namespaced_helper_id(run.run_id, "main")) is not None

    status = local_helper_status(_registry_namespace=run.run_id)
    assert status["success"] is True and status["alive"] is True and status["ready"] is True

    stopped = local_helper_stop(_registry_namespace=run.run_id)
    assert stopped["success"] is True
    assert stopped["alive"] is False
    assert get_local_helper_registry().get(namespaced_helper_id(run.run_id, "main")) is None


def test_model_supplied_namespace_is_overwritten_and_terminal_hook_reaps_helper(tmp_path):
    port = _free_port()
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=_executor())},
    )
    register_shell_session_teardown(runtime)

    def start_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="start",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={
                    "tool_calls": [
                        {
                            "name": "local_helper_start",
                            "arguments": {
                                "command": f"{shlex.quote(sys.executable)} -m http.server {port}",
                                "working_directory": str(tmp_path),
                                "ready_port": port,
                                "ready_timeout": 10,
                                "_registry_namespace": "victim-run",
                            },
                            "call_id": "c1",
                        }
                    ]
                },
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

    workflow = WorkflowSpec(
        workflow_id="local_helper_runtime_test",
        entry_node="start",
        nodes={"start": start_node, "wait_forever": wait_forever},
    )

    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=5)
    assert state.status == RunStatus.WAITING
    assert get_local_helper_registry().get(namespaced_helper_id("victim-run", "main")) is None
    assert get_local_helper_registry().get(namespaced_helper_id(run_id, "main")) is not None

    runtime.cancel_run(run_id, reason="test cleanup")
    assert get_local_helper_registry().get(namespaced_helper_id(run_id, "main")) is None


def test_default_tool_map_and_approval_defaults_include_local_helper_tools():
    tool_map = build_default_tool_map()
    assert {"local_helper_start", "local_helper_status", "local_helper_stop"} <= set(tool_map)

    policy = ToolApprovalPolicy()
    assert "local_helper_status" in policy.auto_approve_tools
    assert {"local_helper_start", "local_helper_stop"} <= policy.require_approval_tools
