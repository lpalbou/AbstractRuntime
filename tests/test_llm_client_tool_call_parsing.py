from __future__ import annotations


from abstractruntime.integrations.abstractcore.llm_client import _maybe_parse_tool_calls_from_text


def test_parse_xml_wrapped_tool_call() -> None:
    content = """
<tool_call>
{"name": "execute_command", "arguments": {"command": "pwd"}}
</tool_call>
""".strip()

    tool_calls, cleaned = _maybe_parse_tool_calls_from_text(content=content)

    assert tool_calls == [
        {"name": "execute_command", "arguments": {"command": "pwd"}, "call_id": None}
    ]
    assert cleaned == ""


def test_parse_qwen3_tool_call() -> None:
    content = """
<|tool_call|>
{"name": "read_file", "arguments": {"path": "README.md"}}
</|tool_call|>
""".strip()

    tool_calls, cleaned = _maybe_parse_tool_calls_from_text(content=content)

    assert tool_calls == [
        {"name": "read_file", "arguments": {"path": "README.md"}, "call_id": None}
    ]
    assert cleaned == ""


def test_no_markers_returns_none() -> None:
    tool_calls, cleaned = _maybe_parse_tool_calls_from_text(content="hello")
    assert tool_calls is None
    assert cleaned is None

