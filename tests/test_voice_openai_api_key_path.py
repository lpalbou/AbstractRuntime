"""The host's `voice_openai_api_key` reaches AbstractVoice, and nothing else.

Two host paths (gateway wave2/console 3f51fab):
1. discovery: `AbstractCoreDiscoveryFacade.get_voice_catalog / list_tts_models /
   list_stt_models(voice_openai_api_key=...)` -> the local clients -> the
   `discovery_queries` owner config the voice plugin reads.
2. execution: `create_local_runtime(llm_kwargs={"voice_openai_api_key": ...})` ->
   `LocalAbstractCoreLLMClient` -> real `create_llm(**llm_kwargs)` -> provider
   `.config` (AbstractCoreInterface stores constructor kwargs there), which is
   the capability-plugin config (`AbstractVoice ... _config_text`).

The rule these tests pin: constructor kwargs land in the provider's `.config`;
providers read only the keys they name, so a `voice_*` key changes no text
request byte (proved on real provider classes with the network refused) and is
never sent to a remote AbstractCore server.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from abstractruntime.integrations.abstractcore.discovery_facade import AbstractCoreDiscoveryFacade
from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
)

SENTINEL = "sk-VOICE-SENTINEL-7f3a"


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> list:
    """Refuse the network, recording every request that tried to leave."""
    sent: list = []

    def _send(self, request, *args, **kwargs):
        sent.append(request)
        raise httpx.ConnectError("network refused by test", request=request)

    async def _asend(self, request, *args, **kwargs):
        return _send(self, request)

    monkeypatch.setattr(httpx.Client, "send", _send)
    monkeypatch.setattr(httpx.AsyncClient, "send", _asend)
    return sent


def _bytes_of(requests: list) -> list:
    return [(str(r.url), bytes(r.content)) for r in requests]


def _leaks(requests: list) -> bool:
    return any(SENTINEL in (str(r.url) + str(dict(r.headers)) + bytes(r.content).decode("utf-8", "replace")) for r in requests)


# ------------------------------------------------------------------ 1. discovery


def _capture_owner_config(monkeypatch: pytest.MonkeyPatch) -> list:
    seen: list = []

    class _FakeVoice:
        def voice_catalog(self, **_kw):
            return {}

        def list_tts_models(self, **_kw):
            return []

        def list_stt_models(self, **_kw):
            return []

    class _FakeRegistry:
        def __init__(self, owner, preferred_backends=None):
            seen.append(dict(owner.config))
            self.voice = _FakeVoice()

    import abstractcore.capabilities as caps

    monkeypatch.setattr(caps, "CapabilityRegistry", _FakeRegistry)
    return seen


@pytest.mark.parametrize("client_cls", [LocalAbstractCoreLLMClient, MultiLocalAbstractCoreLLMClient])
@pytest.mark.parametrize("method", ["get_voice_catalog", "list_tts_models", "list_stt_models"])
def test_discovery_facade_forwards_the_key_to_the_voice_plugin_config(wire, monkeypatch, client_cls, method) -> None:
    client = client_cls(provider="ollama", model="qwen3:4b", llm_kwargs={"base_url": "http://127.0.0.1:9"})
    facade = AbstractCoreDiscoveryFacade(SimpleNamespace(_abstractcore_llm_client=client))
    seen = _capture_owner_config(monkeypatch)
    getattr(facade, method)(provider="openai", voice_openai_api_key=SENTINEL)
    assert seen and seen[-1]["voice_openai_api_key"] == SENTINEL


@pytest.mark.parametrize("method", ["get_voice_catalog", "list_tts_models", "list_stt_models"])
def test_remote_discovery_never_sends_the_host_voice_key(method) -> None:
    sent: list = []

    class _Sender:
        def get(self, url, *, headers=None, timeout=None, **kw):
            sent.append((url, dict(headers or {})))
            return {"items": [], "providers": []}

    client = RemoteAbstractCoreLLMClient(server_base_url="http://127.0.0.1:9", model="m", request_sender=_Sender())
    facade = AbstractCoreDiscoveryFacade(SimpleNamespace(_abstractcore_llm_client=client))
    getattr(facade, method)(provider="openai", voice_openai_api_key=SENTINEL)
    assert sent, "the remote client sent no discovery request"
    assert SENTINEL not in repr(sent)


# ------------------------------------------------------------------ 2. execution (llm_kwargs)


_TEXT_PROVIDERS = [
    ("openai", "gpt-4o-mini", {"api_key": "sk-test"}),
    ("lmstudio", "qwen/qwen3-4b", {"base_url": "http://127.0.0.1:9/v1"}),
    ("ollama", "qwen3:4b", {"base_url": "http://127.0.0.1:9"}),
]


@pytest.mark.parametrize("provider, model, kwargs", _TEXT_PROVIDERS)
def test_llm_kwargs_voice_key_reaches_the_plugin_and_changes_no_text_request(wire, provider, model, kwargs) -> None:
    plain = LocalAbstractCoreLLMClient(provider=provider, model=model, llm_kwargs=dict(kwargs))
    keyed = LocalAbstractCoreLLMClient(
        provider=provider, model=model, llm_kwargs={**kwargs, "voice_openai_api_key": SENTINEL}
    )
    # The voice plugin reads its settings from the provider's config.
    assert keyed._llm.config["voice_openai_api_key"] == SENTINEL
    voice = keyed._llm.capabilities.get_voice()
    assert voice._config_text("voice_openai_api_key") == SENTINEL
    assert "voice_openai_api_key" not in plain._llm.config

    # Text calls: byte-identical requests with and without the key; the key never leaves.
    runs = []
    for client in (plain, keyed):
        wire.clear()
        with pytest.raises(Exception):
            client._llm.generate("hello", max_output_tokens=4)
        runs.append(list(wire))
    assert runs[0], f"{provider}: no request was attempted"
    assert _bytes_of(runs[0]) == _bytes_of(runs[1])
    assert not _leaks(runs[1])


def test_mlx_provider_accepts_the_voice_key_and_hands_it_only_to_the_plugin(wire, monkeypatch) -> None:
    """MLX is in-process (no HTTP): the constructor must accept the key (no
    weights here, so the load is skipped) and expose it to the plugin config."""
    from abstractcore.providers.mlx_provider import MLXProvider

    monkeypatch.setattr(MLXProvider, "_load_model", lambda self: None)
    client = LocalAbstractCoreLLMClient(
        provider="mlx", model="mlx-community/Qwen3-0.6B-4bit", llm_kwargs={"voice_openai_api_key": SENTINEL}
    )
    assert client._llm.config["voice_openai_api_key"] == SENTINEL
    assert client._llm.capabilities.get_voice()._config_text("voice_openai_api_key") == SENTINEL
    assert not wire
