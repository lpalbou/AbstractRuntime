from __future__ import annotations

from abstractruntime.visualflow_compiler.visual.builtins import get_builtin_handler


def test_has_tools_builtin() -> None:
    has_tools = get_builtin_handler("has_tools")
    assert has_tools is not None

    assert has_tools({"array": []}) is False
    assert has_tools({"array": [{"type": "tool_call"}]}) is True
    assert has_tools({"array": None}) is False
    assert has_tools({"array": ("x",)}) is True

