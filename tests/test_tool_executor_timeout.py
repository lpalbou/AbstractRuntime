import time
import pytest

from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor


def test_mapping_tool_executor_times_out_long_running_tool() -> None:
    def slow() -> str:
        time.sleep(0.05)
        return "ok"

    executor = MappingToolExecutor.from_tools([slow], timeout_s=0.01)
    result = executor.execute(tool_calls=[{"name": "slow", "arguments": {}, "call_id": "c1"}])

    assert result["mode"] == "executed"
    assert result["results"][0]["call_id"] == "c1"
    assert result["results"][0]["success"] is False
    assert result["results"][0]["output"] is None
    assert "timed out" in str(result["results"][0]["error"]).lower()


def test_abstractcore_tool_executor_times_out_long_running_tool() -> None:
    abstractcore = pytest.importorskip("abstractcore")
    del abstractcore

    from abstractcore.tools.registry import clear_registry, register_tool
    from abstractruntime.integrations.abstractcore.tool_executor import AbstractCoreToolExecutor

    clear_registry()

    @register_tool
    def slow() -> str:
        time.sleep(0.05)
        return "ok"

    executor = AbstractCoreToolExecutor(timeout_s=0.01)
    result = executor.execute(tool_calls=[{"name": "slow", "arguments": {}, "call_id": "c1"}])

    assert result["mode"] == "executed"
    assert result["results"][0]["call_id"] == "c1"
    assert result["results"][0]["success"] is False
    assert result["results"][0]["output"] is None
    assert "timed out" in str(result["results"][0]["error"]).lower()


