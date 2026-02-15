from __future__ import annotations

from typing import Any, Dict, List


def test_approval_tool_executor_executes_safe_batch() -> None:
    from abstractruntime.integrations.abstractcore.tool_executor import ApprovalToolExecutor, ToolApprovalPolicy

    class DummyExec:
        def __init__(self) -> None:
            self.calls: List[List[Dict[str, Any]]] = []

        def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
            self.calls.append(list(tool_calls))
            results = []
            for tc in tool_calls:
                results.append(
                    {
                        "call_id": str(tc.get("call_id") or ""),
                        "runtime_call_id": tc.get("runtime_call_id"),
                        "name": str(tc.get("name") or ""),
                        "success": True,
                        "output": {"ran": str(tc.get("name") or "")},
                        "error": None,
                    }
                )
            return {"mode": "executed", "results": results}

    delegate = DummyExec()
    ex = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy())

    out = ex.execute(tool_calls=[{"name": "read_file", "arguments": {"file_path": "x"}}])
    assert out.get("mode") == "executed"
    assert len(delegate.calls) == 1


def test_approval_tool_executor_pauses_dangerous_batch() -> None:
    from abstractruntime.integrations.abstractcore.tool_executor import ApprovalToolExecutor, ToolApprovalPolicy

    class DummyExec:
        def __init__(self) -> None:
            self.calls: List[List[Dict[str, Any]]] = []

        def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
            self.calls.append(list(tool_calls))
            return {"mode": "executed", "results": []}

    delegate = DummyExec()
    ex = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy())

    out = ex.execute(tool_calls=[{"name": "write_file", "arguments": {"file_path": "x", "content": "y"}}])
    assert out.get("mode") == "approval_required"
    assert str(out.get("wait_reason") or "").strip().lower() == "user"
    assert isinstance(out.get("wait_key"), str) and str(out.get("wait_key") or "").strip()
    assert delegate.calls == []


def test_approval_tool_executor_pauses_unknown_tool() -> None:
    from abstractruntime.integrations.abstractcore.tool_executor import ApprovalToolExecutor, ToolApprovalPolicy

    class DummyExec:
        def __init__(self) -> None:
            self.calls: List[List[Dict[str, Any]]] = []

        def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
            self.calls.append(list(tool_calls))
            return {"mode": "executed", "results": []}

    delegate = DummyExec()
    ex = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy())

    out = ex.execute(tool_calls=[{"name": "unknown_tool", "arguments": {}}])
    assert out.get("mode") == "approval_required"
    assert delegate.calls == []


def test_tool_approval_policy_marks_telegram_send_as_safe() -> None:
    from abstractruntime.integrations.abstractcore.tool_executor import ToolApprovalPolicy

    policy = ToolApprovalPolicy()
    assert policy.requires_approval([{"name": "send_telegram_message", "arguments": {"chat_id": 1, "text": "hi"}}]) is False


def test_approval_tool_executor_returns_delegating_mode_for_empty_batch() -> None:
    from abstractruntime.integrations.abstractcore.tool_executor import ApprovalToolExecutor, ToolApprovalPolicy

    class DummyExec:
        def execute(self, *, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {"mode": "executed", "results": []}

    ex = ApprovalToolExecutor(delegate=DummyExec(), policy=ToolApprovalPolicy())
    out = ex.execute(tool_calls=[])
    assert out.get("mode") == "approval_required"
