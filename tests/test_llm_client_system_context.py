from __future__ import annotations

import re

import pytest

from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient, RemoteAbstractCoreLLMClient


pytestmark = pytest.mark.basic


_HEADER_RE = re.compile(r"^Date:\s*\d{4}-\d{2}-\d{2};\s*Country:\s*.+$", re.IGNORECASE)


def test_local_llm_client_injects_date_and_country_into_system_prompt(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.handler import UniversalToolHandler

    monkeypatch.setenv("ABSTRACT_COUNTRY", "FR")

    class _DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerateResponse(content="ok", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    result = client.generate(prompt="hello")

    assert client._llm.calls, "expected a provider call"
    sys = str(client._llm.calls[0].get("system_prompt") or "")
    first = sys.splitlines()[0].strip()
    assert _HEADER_RE.match(first)
    assert "Country: FR" in first
    assert isinstance(result.get("metadata"), dict)
    payload = result["metadata"]["_provider_request"]["payload"]
    assert payload["messages"][0]["role"] == "system"
    assert "Country: FR" in payload["messages"][0]["content"]


def test_local_llm_client_does_not_double_inject_when_header_is_present(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.handler import UniversalToolHandler

    monkeypatch.setenv("ABSTRACT_COUNTRY", "FR")

    class _DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerateResponse(content="ok", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    sys_in = "Date: 1999-12-31; Country: FR\\n\\nBase system prompt."
    client.generate(prompt="hello", system_prompt=sys_in)

    sys = str(client._llm.calls[0].get("system_prompt") or "")
    assert sys.startswith("Date: 1999-12-31; Country: FR")
    assert sys.count("Date:") == 1


def test_remote_llm_client_injects_system_context_even_without_system_prompt(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACT_COUNTRY", "FR")

    class StubSender:
        def __init__(self):
            self.calls = []

        def post(self, url, *, headers, json, timeout):
            self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return {
                "model": json["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    sender = StubSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://localhost:8080",
        model="openai-compatible/default",
        request_sender=sender,
        headers={"X-Test": "1"},
    )

    client.generate(prompt="hello", params={"max_tokens": 5})

    body = sender.calls[0]["json"]
    assert body["messages"][0]["role"] == "system"
    first = str(body["messages"][0]["content"] or "").splitlines()[0].strip()
    assert _HEADER_RE.match(first)
    assert "Country: FR" in first
