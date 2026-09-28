"""The host-supplied `voice_openai_api_key` reaches AbstractVoice's plugin config.

AbstractVoice (wave2/voice f43591d) reads the OpenAI credential for its `openai`
engines from the plugin setting `voice_openai_api_key` only (no env var), so the
voice discovery entry points must thread it into the capability owner config.
"""

from __future__ import annotations

import pytest

from abstractruntime.integrations.abstractcore import discovery_queries as dq


class _FakeVoice:
    def voice_catalog(self, **_kw):
        return {"providers": []}

    def list_tts_models(self, **_kw):
        return []

    def list_stt_models(self, **_kw):
        return []


def _capture(monkeypatch: pytest.MonkeyPatch) -> list:
    seen: list = []

    class _FakeRegistry:
        def __init__(self, owner, preferred_backends=None):
            seen.append(dict(owner.config))
            self.voice = _FakeVoice()

    import abstractcore.capabilities as caps

    monkeypatch.setattr(caps, "CapabilityRegistry", _FakeRegistry)
    return seen


def test_owner_config_carries_the_key_only_when_given() -> None:
    assert dq._runtime_capability_owner_config(voice_openai_api_key="  sk-host  ")["voice_openai_api_key"] == "sk-host"
    assert "voice_openai_api_key" not in dq._runtime_capability_owner_config()
    assert "voice_openai_api_key" not in dq._runtime_capability_owner_config(voice_openai_api_key="  ")


@pytest.mark.parametrize(
    "entry", [dq.local_get_voice_catalog, dq.local_list_tts_models, dq.local_list_stt_models]
)
def test_voice_discovery_threads_the_key_into_the_plugin_config(monkeypatch: pytest.MonkeyPatch, entry) -> None:
    seen = _capture(monkeypatch)
    entry(voice_openai_api_key="sk-host", provider="openai")
    assert seen and seen[-1]["voice_openai_api_key"] == "sk-host"
    # Distinct from the remote-endpoint key (`voice_remote_api_key`).
    assert "voice_remote_api_key" not in seen[-1]

    seen.clear()
    entry(provider="openai")
    assert seen and "voice_openai_api_key" not in seen[-1]
