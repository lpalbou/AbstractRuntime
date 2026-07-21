"""Per-run tool policy consumer (the restored 2026-02-21 feature).

`_runtime.tool_policy` (the thin-client wire shape) overrides the static
approval policy for THIS run, both directions; malformed or absent policy
= static behavior; plain executors (no approval concept) are untouched.
"""
from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime.integrations.abstractcore.effect_handlers import (
    _execute_with_run_policy,
)


class _Run:
    run_id = "r1"

    def __init__(self, tool_policy=None):
        rt: Dict[str, Any] = {}
        if tool_policy is not None:
            rt["tool_policy"] = tool_policy
        self.vars: Dict[str, Any] = {"_runtime": rt}


class _GatedExec:
    """Static policy: everything requires approval (the strict default)."""

    def __init__(self):
        self.approved_calls: List[Any] = []
        self.executed_calls: List[Any] = []

    def execute(self, *, tool_calls):
        self.executed_calls.append(tool_calls)
        return {"mode": "approval_required", "wait_reason": "user",
                "tool_calls": tool_calls, "details": {"kind": "tool_approval"}}

    def execute_approved(self, *, tool_calls):
        self.approved_calls.append(tool_calls)
        return {"mode": "executed", "results": [{"name": c["name"], "success": True} for c in tool_calls]}


class _PlainExec:
    def execute(self, *, tool_calls):
        return {"mode": "executed", "results": []}


def test_run_auto_list_skips_the_static_gate() -> None:
    tools = _GatedExec()
    run = _Run({"auto_approve_tools": ["web_search"], "require_approval_tools": []})
    out = _execute_with_run_policy(tools, [{"name": "web_search", "arguments": {}}], run)
    assert out["mode"] == "executed", "run-auto bypasses the static everything-asks gate"
    assert tools.approved_calls and not tools.executed_calls


def test_run_require_forces_the_wait_with_source_label() -> None:
    tools = _GatedExec()
    run = _Run({"auto_approve_tools": ["web_search"], "require_approval_tools": ["execute_command"]})
    out = _execute_with_run_policy(tools, [{"name": "execute_command", "arguments": {}}], run)
    assert out["mode"] == "approval_required"
    assert out["details"]["policy_source"] == "run", "the wait names WHO decided"


def test_absent_or_malformed_policy_keeps_static_behavior() -> None:
    tools = _GatedExec()
    out = _execute_with_run_policy(tools, [{"name": "web_search"}], _Run(None))
    assert out["mode"] == "approval_required", "no policy = the static gate"
    tools2 = _GatedExec()
    out2 = _execute_with_run_policy(tools2, [{"name": "web_search"}], _Run("not-a-dict"))
    assert out2["mode"] == "approval_required"


def test_plain_executor_untouched() -> None:
    run = _Run({"auto_approve_tools": ["x"], "require_approval_tools": []})
    out = _execute_with_run_policy(_PlainExec(), [{"name": "x"}], run)
    assert out["mode"] == "executed", "no approval concept to override"


def test_unlisted_tool_asks_under_a_run_policy() -> None:
    """A run policy is a COMPLETE preference: names outside the auto list
    require approval (the ToolApprovalPolicy contract)."""
    tools = _GatedExec()
    run = _Run({"auto_approve_tools": ["web_search"], "require_approval_tools": []})
    out = _execute_with_run_policy(tools, [{"name": "write_file", "arguments": {}}], run)
    assert out["mode"] == "approval_required"
