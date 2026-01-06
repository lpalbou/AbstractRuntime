from abstractruntime.integrations.abstractcore.llm_client import RemoteAbstractCoreLLMClient


class StubSender:
    def __init__(self):
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return {
            "model": json["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def test_remote_llm_client_builds_chat_completions_request_and_forwards_base_url():
    sender = StubSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://localhost:8080",
        model="openai-compatible/default",
        request_sender=sender,
        timeout_s=12.0,
        headers={"X-Test": "1"},
    )

    result = client.generate(
        prompt="hello",
        messages=None,
        system_prompt="sys",
        tools=None,
        params={"temperature": 0, "max_tokens": 5, "base_url": "http://localhost:1234/v1"},
    )

    assert result["content"] == "ok"
    assert isinstance(result.get("metadata"), dict)
    assert isinstance(result["metadata"].get("_provider_request"), dict)
    assert result["metadata"]["_provider_request"]["url"] == "http://localhost:8080/v1/chat/completions"
    assert result["metadata"]["_provider_request"]["payload"]["model"] == "openai-compatible/default"

    call = sender.calls[0]
    assert call["url"] == "http://localhost:8080/v1/chat/completions"
    assert call["headers"]["X-Test"] == "1"

    body = call["json"]
    assert body["model"] == "openai-compatible/default"
    assert body["base_url"] == "http://localhost:1234/v1"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 5
    assert body["timeout_s"] == 12.0
    assert body["messages"][0]["role"] == "system"


def test_remote_llm_client_default_timeout_is_long_running() -> None:
    sender = StubSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://localhost:8080",
        model="openai-compatible/default",
        request_sender=sender,
        headers={"X-Test": "1"},
    )

    client.generate(prompt="hello", params={"max_tokens": 5})

    call = sender.calls[0]
    assert call["timeout"] == 7200.0

