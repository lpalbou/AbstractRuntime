"""Catalog discovery must not depend on the DEFAULT text client being buildable.

Live defect (2026-09-22): the operator's configured text default pointed at
MLX weights that had never finished downloading. The pooled client soft-fails
that warm-up by design (`_default_client = None`, the host still boots), but
every catalog call -- provider model lists, capability lookups, voice / music
/ vision catalogs -- then raised `NoDefaultProviderConfigured` ("no
provider/model is configured ... this call named none"), a message that was
also false: a default WAS configured, its weights were missing. Every model
picker in every UI was empty for every provider, including providers that
had nothing to do with the broken default.

The underlying `LocalAbstractCoreLLMClient` catalog methods are stateless
delegations to `discovery_queries.local_*`; they never touch loaded weights.
The pool must delegate the same way.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractruntime.integrations.abstractcore import discovery_queries
from abstractruntime.integrations.abstractcore.llm_client import (
    MultiLocalAbstractCoreLLMClient,
    NoDefaultProviderConfigured,
)


def _pool_with_unbuildable_default(monkeypatch: pytest.MonkeyPatch) -> MultiLocalAbstractCoreLLMClient:
    """The exact shape the live gateway boots in: a default is configured, its
    client cannot be constructed (weights missing), the pool has NO clients."""

    def _cannot_build(self: Any, provider: str, model: str, *, llm_kwargs_override: Any = None) -> Any:
        raise RuntimeError(f"Model '{model}' not found for {provider} provider (weights not downloaded)")

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_create_client", _cannot_build)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/never-downloaded")
    assert client._default_client is None, "precondition: the warm-up soft-failed"
    assert client._clients == {}, "precondition: nothing is loaded"
    return client


def test_provider_model_listing_does_not_need_the_default_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _pool_with_unbuildable_default(monkeypatch)
    seen: Dict[str, Any] = {}

    def _fake_list(provider_name: str, **kwargs: Any) -> Dict[str, Any]:
        seen["provider_name"] = provider_name
        seen.update(kwargs)
        return {"provider": provider_name, "models": ["a", "b"], "source": "fake"}

    monkeypatch.setattr(discovery_queries, "local_list_provider_models", _fake_list)

    payload = client.list_provider_models(
        "lmstudio",
        base_url="http://127.0.0.1:1234/v1",
        provider_api_key="k",
        input_type="text",
        output_type="text",
        capability_route=["output.text"],
        timeout_s=3.5,
    )

    assert payload["models"] == ["a", "b"]
    # Every per-call knob reaches the query helper (same mapping as the Local client).
    assert seen == {
        "provider_name": "lmstudio",
        "base_url": "http://127.0.0.1:1234/v1",
        "provider_api_key": "k",
        "input_type": "text",
        "output_type": "text",
        "capability_route": ["output.text"],
        "timeout_s": 3.5,
    }


def test_capability_lookup_by_name_does_not_need_the_default_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _pool_with_unbuildable_default(monkeypatch)
    monkeypatch.setattr(
        discovery_queries,
        "local_get_model_capabilities",
        lambda name: {"model": name, "capabilities": {"max_tokens": 4096, "name": name}},
    )

    assert client.lookup_model_capabilities("mlx-community/Qwen3.5-4B-4bit") == {
        "model": "mlx-community/Qwen3.5-4B-4bit",
        "capabilities": {"max_tokens": 4096, "name": "mlx-community/Qwen3.5-4B-4bit"},
    }
    assert client.get_model_capabilities("mlx-community/Qwen3.5-4B-4bit") == {
        "max_tokens": 4096,
        "name": "mlx-community/Qwen3.5-4B-4bit",
    }


def test_capability_lookup_without_a_name_uses_the_configured_default_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default's weights are missing, but its NAME is known: a metadata
    # lookup for it is still answerable without loading anything.
    client = _pool_with_unbuildable_default(monkeypatch)
    monkeypatch.setattr(
        discovery_queries,
        "local_get_model_capabilities",
        lambda name: {"model": name, "capabilities": {"max_tokens": 1}},
    )
    assert client.lookup_model_capabilities()["model"] == "vendor/never-downloaded"


def test_capability_lookup_refuses_only_when_no_model_is_named_and_none_is_configured() -> None:
    # Fresh install: nothing configured, nothing named. This is the ONE case
    # the "no provider/model is configured" refusal is truthful for.
    client = MultiLocalAbstractCoreLLMClient(provider="", model="")
    assert client._default_client is None
    with pytest.raises(NoDefaultProviderConfigured):
        client.lookup_model_capabilities()
    with pytest.raises(NoDefaultProviderConfigured):
        client.get_model_capabilities()


@pytest.mark.parametrize(
    ("method", "helper", "kwargs"),
    [
        ("list_embedding_models", "local_list_embedding_models", {"provider": "huggingface"}),
        ("get_voice_catalog", "local_get_voice_catalog", {"provider": "openai", "model": "tts-1"}),
        ("list_tts_models", "local_list_tts_models", {"provider": "openai"}),
        ("list_stt_models", "local_list_stt_models", {"provider": "openai"}),
        ("list_music_providers", "local_list_music_providers", {"task": "text_to_music"}),
        ("list_music_models", "local_list_music_models", {"provider": "acemusic"}),
        ("list_vision_provider_models", "local_list_vision_provider_models", {"task": "text_to_image"}),
        ("list_vision_adapters", "local_list_vision_adapters", {"model": "flux-dev"}),
        ("list_cached_vision_models", "local_list_cached_vision_models", {"task": "text_to_image"}),
    ],
)
def test_every_catalog_query_survives_an_unbuildable_default(
    monkeypatch: pytest.MonkeyPatch, method: str, helper: str, kwargs: Dict[str, Any]
) -> None:
    client = _pool_with_unbuildable_default(monkeypatch)
    sentinel = {"source": f"fake:{helper}", "kwargs": None}

    def _fake(**call_kwargs: Any) -> Dict[str, Any]:
        sentinel["kwargs"] = dict(call_kwargs)
        return sentinel

    monkeypatch.setattr(discovery_queries, helper, _fake)

    result = getattr(client, method)(**kwargs)

    assert result is sentinel
    for key, value in kwargs.items():
        assert sentinel["kwargs"][key] == value, f"{method}: {key} did not reach {helper}"
