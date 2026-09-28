"""An unavailable TTS listing says WHY (voice tag gate follow-up).

`local_list_tts_models` answered `available: false, error: null` for a provider
that cannot run (OpenAI without a key, an unknown provider id): the reason
AbstractVoice (>= 0.13) puts in the catalog's `unavailable_reason` was dropped.
It is now the listing's `error`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from abstractruntime.integrations.abstractcore import discovery_queries as dq

# AbstractVoice is a declared dependency (abstractcore[voice] in the base install):
# the first two tests run the real catalog, engine-free, with no key configured.


@pytest.fixture()
def no_voice_keys(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for name in ("OPENAI_API_KEY", "ABSTRACTVOICE_OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


def test_openai_without_a_key_names_the_missing_key(no_voice_keys) -> None:
    out = dq.local_list_tts_models(provider="openai")
    assert out["available"] is False and out["models"] == []
    assert out["error"] and "OpenAI API key" in out["error"]


def test_unknown_provider_says_it_is_unknown(no_voice_keys) -> None:
    out = dq.local_list_tts_models(provider="no-such-engine")
    assert out["available"] is False
    assert out["error"] and "not a known text-to-speech provider" in out["error"]


def test_the_error_is_the_catalogs_unavailable_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = SimpleNamespace(
        voice_catalog=lambda provider=None: {"tts_providers": [], "unavailable_reason": "engine X is not installed"},
        list_tts_models=lambda provider=None: [],
    )
    monkeypatch.setattr(dq, "_runtime_capability_registry", lambda **_: SimpleNamespace(voice=voice))
    assert dq.local_list_tts_models(provider="x")["error"] == "engine X is not installed"
    # An available listing never carries it.
    voice.voice_catalog = lambda provider=None: {"tts_providers": ["x"], "tts_models_by_provider": {"x": ["m"]},
                                                 "unavailable_reason": None}
    voice.list_tts_models = lambda provider=None: ["m"]
    ok = dq.local_list_tts_models(provider="x")
    assert ok["available"] is True and ok["error"] is None
