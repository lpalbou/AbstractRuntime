from abstractruntime.integrations.abstractcore.llm_client import RemoteAbstractCoreLLMClient


class StubSender:
    def __init__(self):
        self.calls = []

    def get(self, url, *, headers, timeout):
        self.calls.append({"method": "GET", "url": url, "headers": headers, "timeout": timeout})
        return {
            "supported": True,
            "operation": "capabilities",
            "capabilities": {"supported": True, "mode": "keyed"},
        }

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        if "/acore/prompt_cache/" in url:
            return {
                "supported": True,
                "operation": url.rsplit("/", 1)[-1],
                "ok": True,
                "capabilities": {"supported": True, "mode": "keyed"},
            }
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
        params={
            "temperature": 0,
            "max_tokens": 5,
            "base_url": "http://localhost:1234/v1",
            "prompt_cache_key": "sess:abc",
            "api_key": "provider-secret",
        },
    )

    assert result["content"] == "ok"
    assert isinstance(result.get("metadata"), dict)
    assert isinstance(result["metadata"].get("_provider_request"), dict)
    assert result["metadata"]["_provider_request"]["url"] == "http://localhost:8080/v1/chat/completions"
    assert result["metadata"]["_provider_request"]["payload"]["model"] == "openai-compatible/default"

    call = sender.calls[0]
    assert call["url"] == "http://localhost:8080/v1/chat/completions"
    assert call["headers"]["X-Test"] == "1"
    assert call["headers"]["X-AbstractCore-Provider-API-Key"] == "provider-secret"

    body = call["json"]
    assert body["model"] == "openai-compatible/default"
    assert body["base_url"] == "http://localhost:1234/v1"
    assert body["prompt_cache_key"] == "sess:abc"
    assert "api_key" not in body
    assert body["temperature"] == 0
    assert body["max_tokens"] == 5
    assert body["timeout_s"] == 12.0
    assert body["messages"][0]["role"] == "system"
    assert "api_key" not in result["metadata"]["_provider_request"]["payload"]


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


def test_remote_prompt_cache_control_plane_forwards_proxy_context() -> None:
    sender = StubSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://localhost:8080",
        model="openai-compatible/default",
        request_sender=sender,
        timeout_s=12.0,
        headers={"X-Test": "1"},
    )

    caps = client.get_prompt_cache_capabilities(
        base_url="http://localhost:8001/v1",
        api_key="secret",
    )
    assert caps["supported"] is True

    set_resp = client.prompt_cache_set(
        key="sess:abc",
        base_url="http://localhost:8001/v1",
        api_key="secret",
    )
    assert set_resp["supported"] is True
    assert set_resp["ok"] is True

    get_call = sender.calls[0]
    assert get_call["method"] == "GET"
    assert get_call["url"] == "http://localhost:8080/acore/prompt_cache/capabilities?base_url=http%3A%2F%2Flocalhost%3A8001%2Fv1"
    assert get_call["headers"]["X-AbstractCore-Provider-API-Key"] == "secret"
    post_call = sender.calls[1]
    assert post_call["method"] == "POST"
    assert post_call["url"] == "http://localhost:8080/acore/prompt_cache/set"
    assert post_call["headers"]["X-AbstractCore-Provider-API-Key"] == "secret"
    assert post_call["json"]["key"] == "sess:abc"
    assert post_call["json"]["base_url"] == "http://localhost:8001/v1"
    assert "api_key" not in post_call["json"]
