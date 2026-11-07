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

# AbstractVoice is a declared dependency (through AbstractCore's light install):
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


# 0968: an engine that RAISES while the listing is built (it cannot run, so building it
# fails) used to answer with the bare exception text. The reason AbstractVoice gives
# in plain words wins; the exception is kept, worded for a person, only as a fallback.


def _raising_voice(full_catalog_error: Exception, light_catalog: object) -> SimpleNamespace:
    def voice_catalog(provider=None, providers_only=False):
        if providers_only:
            if isinstance(light_catalog, Exception):
                raise light_catalog
            return light_catalog
        raise full_catalog_error

    return SimpleNamespace(voice_catalog=voice_catalog, list_tts_models=lambda provider=None: [])


def test_a_raising_engine_answers_with_the_engine_free_catalogs_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _raising_voice(
        ModuleNotFoundError("No module named 'kokoro'"),
        {"tts_providers": [], "unavailable_reason": "Kokoro is not installed; install abstractvoice[kokoro]"},
    )
    monkeypatch.setattr(dq, "_runtime_capability_registry", lambda **_: SimpleNamespace(voice=voice))
    out = dq.local_list_tts_models(provider="kokoro")
    assert out["available"] is False and out["models"] == []
    assert out["error"] == "Kokoro is not installed; install abstractvoice[kokoro]"


def test_a_listing_that_raises_after_the_catalog_keeps_the_catalogs_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def list_tts_models(provider=None):
        raise RuntimeError("engine build failed")

    voice = SimpleNamespace(
        voice_catalog=lambda provider=None, providers_only=False: {"unavailable_reason": "no OpenAI API key is configured"},
        list_tts_models=list_tts_models,
    )
    monkeypatch.setattr(dq, "_runtime_capability_registry", lambda **_: SimpleNamespace(voice=voice))
    assert dq.local_list_tts_models(provider="openai")["error"] == "no OpenAI API key is configured"


def test_with_no_reason_the_exception_is_worded_for_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    for light in ({"tts_providers": ["piper"], "unavailable_reason": None}, RuntimeError("light failed too")):
        voice = _raising_voice(RuntimeError("weights are corrupt"), light)
        monkeypatch.setattr(dq, "_runtime_capability_registry", lambda **_: SimpleNamespace(voice=voice))
        out = dq.local_list_tts_models(provider="piper")
        assert out["available"] is False
        assert out["error"] == "the text-to-speech engine could not list its models: weights are corrupt"


def test_an_empty_exception_still_names_what_went_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_registry(**_):
        raise ImportError()

    monkeypatch.setattr(dq, "_runtime_capability_registry", broken_registry)
    assert dq.local_list_tts_models()["error"] == "the text-to-speech engine could not list its models: ImportError"
