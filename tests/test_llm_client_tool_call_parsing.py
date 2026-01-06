from __future__ import annotations


from abstractruntime.integrations.abstractcore.llm_client import _maybe_parse_tool_calls_from_text, _normalize_tool_calls
from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient


def test_local_llm_client_surfaces_reasoning_from_provider_metadata() -> None:
    from abstractcore.core.types import GenerateResponse

    class _DummyReasoningLLM:
        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            return GenerateResponse(content="", metadata={"reasoning": "I will do X."}, gen_time=123.4)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._llm = _DummyReasoningLLM()
    from abstractcore.tools.handler import UniversalToolHandler

    client._tool_handler = UniversalToolHandler(client._model)

    result = client.generate(prompt="do something")
    assert result.get("reasoning") == "I will do X."
    assert result.get("gen_time") == 123.4
    # Observability: local client should always attach a provider request payload.
    assert isinstance(result.get("metadata"), dict)
    assert isinstance(result["metadata"].get("_provider_request"), dict)


def test_local_llm_client_streaming_aggregates_and_surfaces_ttft() -> None:
    from abstractcore.core.types import GenerateResponse

    class _DummyStreamingLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs.get("stream") is True:
                def _gen():
                    yield GenerateResponse(content="Hello", metadata={"_timing": {"ttft_ms": 12.3}})
                    yield GenerateResponse(content=" world", finish_reason="stop")

                return _gen()
            return GenerateResponse(content="Hello world", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._llm = _DummyStreamingLLM()
    from abstractcore.tools.handler import UniversalToolHandler

    client._tool_handler = UniversalToolHandler(client._model)

    result = client.generate(prompt="do something", params={"stream": True})

    assert result.get("content") == "Hello world"
    assert result.get("ttft_ms") == 12.3
    assert isinstance(result.get("gen_time"), (int, float))
    assert isinstance(result.get("metadata"), dict)
    assert result["metadata"].get("_timing", {}).get("ttft_ms") == 12.3
    assert isinstance(result["metadata"].get("_provider_request"), dict)
    assert client._llm.calls and client._llm.calls[0].get("stream") is True


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


def test_parse_bracket_prefix_tool_call() -> None:
    content = "tool: [list_files]: {'directory_path': 'rtype', 'recursive': True}\n"
    tool_calls, cleaned = _maybe_parse_tool_calls_from_text(content=content)
    assert tool_calls == [{"name": "list_files", "arguments": {"directory_path": "rtype", "recursive": True}, "call_id": None}]
    assert cleaned == ""

def test_parse_bracket_prefix_tool_call_multiline() -> None:
    content = (
        "tool: [list_files]: {\n"
        '  "directory_path": "rtype",\n'
        '  "pattern": "*",\n'
        '  "recursive": true\n'
        "}\n"
    )
    tool_calls, cleaned = _maybe_parse_tool_calls_from_text(content=content)
    assert tool_calls == [
        {"name": "list_files", "arguments": {"directory_path": "rtype", "pattern": "*", "recursive": True}, "call_id": None}
    ]
    assert cleaned == ""


def test_bracket_prefix_tool_call_filtered_by_allowlist() -> None:
    content = "tool: [list_files]: {'directory_path': 'rtype', 'recursive': True}\n"
    tool_calls, cleaned = _maybe_parse_tool_calls_from_text(content=content, allowed_tool_names={"read_file"})
    assert tool_calls is None
    assert cleaned is None


def test_normalize_openai_native_tool_calls_arguments_string() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "name": "web_search",
            "arguments": "{\"query\":\"hello\"}",
        }
    ]
    normalized = _normalize_tool_calls(tool_calls)
    assert normalized == [{"name": "web_search", "arguments": {"query": "hello"}, "call_id": "call_1"}]


def test_normalize_openai_function_shape_tool_calls() -> None:
    tool_calls = [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "fetch_url", "arguments": "{\"url\":\"https://example.com\"}"},
        }
    ]
    normalized = _normalize_tool_calls(tool_calls)
    assert normalized == [{"name": "fetch_url", "arguments": {"url": "https://example.com"}, "call_id": "call_2"}]


class _DummyLLM:
    def __init__(self, *, capabilities: list[str]):
        self._capabilities = list(capabilities)
        self.calls: list[dict] = []

    def get_capabilities(self) -> list[str]:
        return list(self._capabilities)

    def generate(self, **kwargs):
        # Record the raw call so tests can assert whether `tools` was passed.
        self.calls.append(dict(kwargs))
        return {"content": "ok", "tool_calls": None}


def _client_with_dummy_llm(*, model: str, capabilities: list[str]) -> LocalAbstractCoreLLMClient:
    # Build an instance without calling __init__ (avoids network/provider construction).
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = model
    client._llm = _DummyLLM(capabilities=capabilities)
    from abstractcore.tools.handler import UniversalToolHandler

    client._tool_handler = UniversalToolHandler(model)
    return client


def test_local_llm_client_uses_prompted_tools_for_qwen_even_if_provider_advertises_tools() -> None:
    client = _client_with_dummy_llm(model="qwen/qwen3-next-80b", capabilities=["tools"])
    tools = [{"name": "read_file", "description": "Read a file", "parameters": {"file_path": {"type": "string"}}}]

    client.generate(prompt="read rtype-plan.md", tools=tools)

    assert len(client._llm.calls) == 1
    call = client._llm.calls[0]
    # Tool-call prompting/parsing is an AbstractCore responsibility; the runtime passes tool schemas through.
    assert "tools" in call


def test_local_llm_client_uses_native_tools_when_model_is_native_and_provider_supports_tools() -> None:
    client = _client_with_dummy_llm(model="google/gemma-3n-e2b", capabilities=["tools"])
    tools = [{"name": "read_file", "description": "Read a file", "parameters": {"file_path": {"type": "string"}}}]

    client.generate(prompt="read rtype-plan.md", tools=tools)

    assert len(client._llm.calls) == 1
    call = client._llm.calls[0]
    # Native tool calling path: pass `tools` through to provider.
    assert "tools" in call


class _DummyNativeToolCallLLM:
    def __init__(self):
        self.calls: list[dict] = []

    def get_capabilities(self) -> list[str]:
        return ["tools"]

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "content": (
                "I will read the file.\n\n"
                "<|tool_call|>\n"
                '{"name":"read_file","arguments":{"path":"README.md"}}\n'
                "</|tool_call|>\n"
            ),
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        }


def test_local_llm_client_cleans_echoed_tool_markup_when_native_tool_calls_present() -> None:
    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "google/gemma-3n-e2b"
    client._llm = _DummyNativeToolCallLLM()
    from abstractcore.tools.handler import UniversalToolHandler

    client._tool_handler = UniversalToolHandler(client._model)

    tools = [{"name": "read_file", "description": "Read a file", "parameters": {"path": {"type": "string"}}}]
    result = client.generate(prompt="read README.md", tools=tools)

    assert result.get("tool_calls") == [{"name": "read_file", "arguments": {"path": "README.md"}, "call_id": "call_1"}]
