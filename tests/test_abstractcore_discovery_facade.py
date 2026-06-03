from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from abstractruntime.integrations import abstractcore
from abstractruntime.integrations.abstractcore import (
    AbstractCoreDiscoveryFacade,
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
    discovery_facade,
    get_abstractcore_discovery_facade,
)


class _HttpResponse:
    def __init__(self, body: Dict[str, Any], *, headers: Dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = dict(headers or {})


class _RecordingDiscoveryClient:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any, Dict[str, Any]]] = []
        self.responses: Dict[str, Dict[str, Any]] = {
            "list_providers": {"items": [{"name": "mlx"}]},
            "list_provider_models": {"provider": "mlx", "models": ["qwen"]},
            "list_embedding_models": {"providers": ["huggingface"], "models": ["all-minilm-l6-v2"]},
            "lookup_model_capabilities": {"model": "qwen", "capabilities": {"max_tokens": 4096}},
            "get_voice_catalog": {"providers": ["openai"], "profiles": []},
            "list_tts_models": {"providers": ["openai"], "models": ["tts-1"]},
            "list_stt_models": {"providers": ["openai"], "models": ["whisper-1"]},
            "list_music_providers": {"providers": ["acemusic"], "provider_details": [{"provider": "acemusic"}]},
            "list_music_models": {"providers": ["acemusic"], "models": [{"provider": "acemusic", "id": "ace-step"}]},
            "list_vision_provider_models": {"providers": ["mflux"], "models": []},
            "list_cached_vision_models": {"models": [{"id": "flux-dev", "provider": "mflux"}]},
        }

    def _record(self, name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append((name, args[0] if args else None, dict(kwargs)))
        return self.responses[name]

    def list_providers(self, *, include_models: bool = False, **kwargs: Any) -> Dict[str, Any]:
        return self._record("list_providers", include_models, **kwargs)

    def list_provider_models(self, provider_name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._record("list_provider_models", provider_name, **kwargs)

    def list_embedding_models(
        self,
        *,
        base_url: str | None = None,
        provider_api_key: str | None = None,
        provider: str | None = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "list_embedding_models",
            None,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            providers_only=providers_only,
            **kwargs,
        )

    def lookup_model_capabilities(self, model_name: str | None = None) -> Dict[str, Any]:
        return self._record("lookup_model_capabilities", model_name)

    def get_voice_catalog(
        self,
        *,
        base_url: str | None = None,
        provider_api_key: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "get_voice_catalog",
            None,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            model=model,
            providers_only=providers_only,
            **kwargs,
        )

    def list_tts_models(
        self,
        *,
        base_url: str | None = None,
        provider_api_key: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "list_tts_models",
            None,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )

    def list_stt_models(
        self,
        *,
        base_url: str | None = None,
        provider_api_key: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "list_stt_models",
            None,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )

    def list_music_providers(
        self,
        *,
        task: str | None = None,
        base_url: str | None = None,
        provider_api_key: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "list_music_providers",
            None,
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            **kwargs,
        )

    def list_music_models(
        self,
        *,
        task: str | None = None,
        base_url: str | None = None,
        provider_api_key: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "list_music_models",
            None,
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )

    def list_vision_provider_models(
        self,
        *,
        task: str | None = None,
        base_url: str | None = None,
        provider_api_key: str | None = None,
        provider: str | None = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "list_vision_provider_models",
            None,
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            providers_only=providers_only,
            **kwargs,
        )

    def list_cached_vision_models(
        self,
        *,
        task: str | None = None,
        base_url: str | None = None,
        provider_api_key: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "list_cached_vision_models",
            None,
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )


class _LegacyDiscoveryClient(_RecordingDiscoveryClient):
    lookup_model_capabilities = None  # type: ignore[assignment]

    def get_model_capabilities(self, model_name: str | None = None) -> Dict[str, Any]:
        return {"max_tokens": 2048, "model_id": model_name}


class _RecordingRequestSender:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> _HttpResponse:
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers), "timeout": timeout})
        if url == "http://core.test/providers?include_models=true":
            return _HttpResponse({"providers": [{"name": "openai"}, {"name": "ollama"}]})
        if url == "http://core.test/v1/models?provider=ollama&base_url=http%3A%2F%2Fprovider.test%2Fv1":
            return _HttpResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": "ollama/qwen"},
                        {"id": "ollama/llama"},
                    ],
                }
            )
        if url == "http://core.test/v1/models?provider=ollama&input_type=image&output_type=text":
            return _HttpResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": "ollama/llava"},
                    ],
                }
            )
        if url == "http://core.test/v1/models?provider=lmstudio&output_type=embeddings&base_url=http%3A%2F%2Fprovider.test%2Fv1":
            return _HttpResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": "lmstudio/bge-small-en-v1.5", "owned_by": "lmstudio"},
                        {"id": "lmstudio/text-embedding-nomic-embed-text-v1.5", "owned_by": "lmstudio"},
                    ],
                }
            )
        if url == "http://core.test/v1/embeddings/providers?provider=lmstudio":
            return _HttpResponse(
                {
                    "kind": "embedding_providers",
                    "scope": "embedding.text",
                    "provider": "lmstudio",
                    "providers": ["lmstudio"],
                    "available_providers": ["lmstudio"],
                    "embedding_providers": ["lmstudio"],
                    "provider_details": [
                        {
                            "id": "lmstudio",
                            "provider": "lmstudio",
                            "label": "LMStudio",
                            "transport": "openai_compatible_http",
                            "base_url_configurable": True,
                            "base_url_env_vars": ["LMSTUDIO_BASE_URL"],
                            "default_base_url": "http://localhost:1234/v1",
                        },
                    ],
                    "models": [],
                    "embedding_models": [],
                    "models_by_provider": {},
                    "embedding_models_by_provider": {},
                    "provider_models": [],
                }
            )
        if url == "http://core.test/v1/audio/voices?base_url=http%3A%2F%2Fprovider.test%2Fv1&provider=openai&providers_only=true":
            return _HttpResponse(
                {
                    "tts_providers": ["openai"],
                    "providers": ["openai"],
                    "profiles": [{"id": "alloy", "provider": "openai", "model": "tts-1"}],
                    "tts_models_by_provider": {"openai": ["tts-1"]},
                }
            )
        if url == "http://core.test/v1/audio/speech/models?base_url=http%3A%2F%2Fprovider.test%2Fv1&provider=openai":
            return _HttpResponse(
                {
                    "models": ["tts-1"],
                    "models_by_provider": {"openai": ["tts-1"]},
                    "tts_models_by_provider": {"openai": ["tts-1"]},
                    "providers": ["openai"],
                }
            )
        if url == "http://core.test/v1/audio/transcriptions/models?provider=openai":
            return _HttpResponse(
                {
                    "models": ["whisper-1"],
                    "models_by_provider": {"openai": ["whisper-1"]},
                    "stt_models_by_provider": {"openai": ["whisper-1"]},
                    "providers": ["openai"],
                }
            )
        if url == "http://core.test/v1/audio/music/providers?task=text_to_music&base_url=http%3A%2F%2Fprovider.test%2Fv1":
            return _HttpResponse(
                {
                    "providers": [
                        {"provider": "acemusic", "tasks": ["text_to_music"]},
                        {"provider": "suno", "tasks": ["text_to_music"]},
                    ]
                }
            )
        if url == "http://core.test/v1/audio/music/models?task=text_to_music&provider=acemusic&base_url=http%3A%2F%2Fprovider.test%2Fv1":
            return _HttpResponse(
                {
                    "models": [
                        {"provider": "acemusic", "id": "ace-step", "tasks": ["text_to_music"]},
                        {"provider": "acemusic", "model": "ace-pro", "tasks": ["text_to_music"]},
                    ]
                }
            )
        if url == "http://core.test/v1/vision/provider_models?task=text_to_image&provider=mflux&providers_only=true":
            return _HttpResponse(
                {
                    "models": [{"provider": "mflux", "model": "flux-dev"}],
                    "providers": ["mflux"],
                    "available_providers": ["mflux"],
                    "models_by_provider": {"mflux": ["flux-dev"]},
                }
            )
        if url == "http://core.test/v1/vision/models?task=text_to_image&provider=mflux":
            return _HttpResponse(
                {
                    "models": [
                        {"id": "flux-dev", "provider": "mflux", "tasks": ["text_to_image"]},
                        {"id": "other", "provider": "openai", "tasks": ["text_to_image"]},
                    ]
                }
            )
        raise AssertionError(f"Unexpected GET url: {url}")

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> _HttpResponse:
        raise AssertionError(f"Unexpected POST url: {url}")


class _ErrorResponse:
    def __init__(self, *, status_code: int, body: Dict[str, Any]) -> None:
        self.status_code = status_code
        self.body = body
        self.text = str(body)


class _DiscoveryError(Exception):
    def __init__(self, response: _ErrorResponse) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


class _FailingDiscoverySender:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> _HttpResponse:
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers), "timeout": timeout})
        raise _DiscoveryError(_ErrorResponse(status_code=403, body={"error": {"message": "forbidden"}}))

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> _HttpResponse:
        raise AssertionError(f"Unexpected POST url: {url}")


class _TransportFailingDiscoverySender:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> _HttpResponse:
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers), "timeout": timeout})
        raise RuntimeError("connection refused")

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> _HttpResponse:
        raise AssertionError(f"Unexpected POST url: {url}")


def test_public_discovery_facade_exports_are_available() -> None:
    assert abstractcore.AbstractCoreDiscoveryFacade is AbstractCoreDiscoveryFacade
    assert abstractcore.get_abstractcore_discovery_facade is get_abstractcore_discovery_facade
    assert "AbstractCoreDiscoveryFacade" in abstractcore.__all__
    assert "get_abstractcore_discovery_facade" in abstractcore.__all__
    assert "AbstractCoreDiscoveryFacade" in discovery_facade.__all__
    assert "get_abstractcore_discovery_facade" in discovery_facade.__all__


def test_discovery_facade_can_be_built_directly_or_from_runtime_helper() -> None:
    client = _RecordingDiscoveryClient()
    runtime = SimpleNamespace(_abstractcore_llm_client=client)

    direct = AbstractCoreDiscoveryFacade(runtime)
    from_runtime = AbstractCoreDiscoveryFacade.from_runtime(runtime)
    runtime_helper = get_abstractcore_discovery_facade(runtime)

    assert direct._client is client
    assert from_runtime._client is client
    assert runtime_helper._client is client


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AbstractCoreDiscoveryFacade(SimpleNamespace(_abstractcore_llm_client=object())),
        lambda: AbstractCoreDiscoveryFacade.from_runtime(SimpleNamespace(_abstractcore_llm_client=object())),
        lambda: get_abstractcore_discovery_facade(SimpleNamespace(_abstractcore_llm_client=object())),
    ],
)
def test_discovery_facade_rejects_runtime_clients_that_do_not_match_the_contract(factory) -> None:
    with pytest.raises(TypeError, match="Missing methods"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AbstractCoreDiscoveryFacade.from_runtime(object()),
        lambda: get_abstractcore_discovery_facade(object()),
    ],
)
def test_discovery_facade_runtime_helpers_raise_when_runtime_has_no_client(factory) -> None:
    with pytest.raises(RuntimeError, match="Runtime is not wired to AbstractCore discovery helpers"):
        factory()


def test_discovery_facade_delegates_snapshot_queries() -> None:
    client = _RecordingDiscoveryClient()
    facade = AbstractCoreDiscoveryFacade(SimpleNamespace(_abstractcore_llm_client=client))

    providers = facade.list_providers(include_models=True)
    provider_models = facade.list_provider_models("mlx", base_url="http://provider.test/v1")
    embeddings = facade.list_embedding_models(
        base_url="http://provider.test/v1",
        provider_api_key="secret",
        provider="huggingface",
        providers_only=True,
    )
    capabilities = facade.get_model_capabilities("mlx/qwen")
    voices = facade.get_voice_catalog(
        base_url="http://provider.test/v1",
        provider_api_key="secret",
        provider="openai",
        model="tts-1",
        providers_only=True,
    )
    tts = facade.list_tts_models(base_url="http://provider.test/v1", provider="openai")
    stt = facade.list_stt_models(provider="openai")
    music_providers = facade.list_music_providers(
        task="text_to_music",
        base_url="http://provider.test/v1",
        provider_api_key="secret",
    )
    music_models = facade.list_music_models(
        task="text_to_music",
        base_url="http://provider.test/v1",
        provider_api_key="secret",
        provider="acemusic",
    )
    vision = facade.list_vision_provider_models(task="text_to_image", provider="mflux", providers_only=True)
    cached = facade.list_cached_vision_models(task="text_to_image", provider="mflux")

    assert providers == {"items": [{"name": "mlx"}]}
    assert provider_models == {"provider": "mlx", "models": ["qwen"]}
    assert embeddings == {"providers": ["huggingface"], "models": ["all-minilm-l6-v2"]}
    assert capabilities == {"model": "qwen", "capabilities": {"max_tokens": 4096}}
    assert voices == {"providers": ["openai"], "profiles": []}
    assert tts == {"providers": ["openai"], "models": ["tts-1"]}
    assert stt == {"providers": ["openai"], "models": ["whisper-1"]}
    assert music_providers == {"providers": ["acemusic"], "provider_details": [{"provider": "acemusic"}]}
    assert music_models == {"providers": ["acemusic"], "models": [{"provider": "acemusic", "id": "ace-step"}]}
    assert vision == {"providers": ["mflux"], "models": []}
    assert cached == {"models": [{"id": "flux-dev", "provider": "mflux"}]}

    assert client.calls == [
        ("list_providers", True, {}),
        ("list_provider_models", "mlx", {"base_url": "http://provider.test/v1"}),
        (
            "list_embedding_models",
            None,
            {
                "base_url": "http://provider.test/v1",
                "provider_api_key": "secret",
                "provider": "huggingface",
                "providers_only": True,
            },
        ),
        ("lookup_model_capabilities", "mlx/qwen", {}),
        (
            "get_voice_catalog",
            None,
            {
                "base_url": "http://provider.test/v1",
                "provider_api_key": "secret",
                "provider": "openai",
                "model": "tts-1",
                "providers_only": True,
            },
        ),
        (
            "list_tts_models",
            None,
            {
                "base_url": "http://provider.test/v1",
                "provider_api_key": None,
                "provider": "openai",
            },
        ),
        (
            "list_stt_models",
            None,
            {
                "base_url": None,
                "provider_api_key": None,
                "provider": "openai",
            },
        ),
        (
            "list_music_providers",
            None,
            {
                "task": "text_to_music",
                "base_url": "http://provider.test/v1",
                "provider_api_key": "secret",
            },
        ),
        (
            "list_music_models",
            None,
            {
                "task": "text_to_music",
                "base_url": "http://provider.test/v1",
                "provider_api_key": "secret",
                "provider": "acemusic",
            },
        ),
        (
            "list_vision_provider_models",
            None,
            {
                "task": "text_to_image",
                "base_url": None,
                "provider_api_key": None,
                "provider": "mflux",
                "providers_only": True,
            },
        ),
        (
            "list_cached_vision_models",
            None,
            {
                "task": "text_to_image",
                "base_url": None,
                "provider_api_key": None,
                "provider": "mflux",
            },
        ),
    ]


def test_discovery_facade_accepts_legacy_clients_with_raw_get_model_capabilities() -> None:
    client = _LegacyDiscoveryClient()
    runtime = SimpleNamespace(_abstractcore_llm_client=client)

    payload = get_abstractcore_discovery_facade(runtime).get_model_capabilities("legacy/model")

    assert payload == {
        "model": "legacy/model",
        "capabilities": {"max_tokens": 2048, "model_id": "legacy/model"},
    }


def test_multilocal_discovery_methods_use_runtime_helpers(monkeypatch) -> None:
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_providers",
        lambda *, include_models, default_provider, default_model: {
            "items": [{"name": default_provider}],
            "default_model": default_model,
            "include_models": include_models,
        },
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_provider_models",
        lambda provider_name, *, base_url=None, provider_api_key=None, input_type=None, output_type=None, timeout_s=None: {
            "provider": provider_name,
            "models": [base_url or "default"],
            "provider_api_key": provider_api_key,
            "input_type": input_type,
            "output_type": output_type,
            "timeout_s": timeout_s,
        },
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_embedding_models",
        lambda *, base_url=None, provider_api_key=None, provider=None, providers_only=False, timeout_s=None: {
            "provider": provider,
            "providers_only": providers_only,
            "base_url": base_url,
            "provider_api_key": provider_api_key,
            "timeout_s": timeout_s,
            "models": ["bge-small-en-v1.5"],
        },
    )
    monkeypatch.setattr(
        "abstractcore.architectures.detection.get_model_capabilities",
        lambda model_name: {"model_id": model_name, "max_tokens": 123},
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_get_voice_catalog",
        lambda *, base_url=None, provider_api_key=None, provider=None, model=None, providers_only=False: {
            "provider": provider,
            "model": model,
            "providers_only": providers_only,
            "base_url": base_url,
            "provider_api_key": provider_api_key,
        },
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_tts_models",
        lambda *, base_url=None, provider_api_key=None, provider=None: {
            "provider": provider,
            "base_url": base_url,
            "provider_api_key": provider_api_key,
        },
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_stt_models",
        lambda *, base_url=None, provider_api_key=None, provider=None: {
            "provider": provider,
            "base_url": base_url,
            "provider_api_key": provider_api_key,
        },
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_music_providers",
        lambda *, task=None, base_url=None, provider_api_key=None: {
            "task": task,
            "base_url": base_url,
            "provider_api_key": provider_api_key,
            "providers": ["acemusic"],
        },
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_music_models",
        lambda *, task=None, base_url=None, provider_api_key=None, provider=None: {
            "task": task,
            "provider": provider,
            "base_url": base_url,
            "provider_api_key": provider_api_key,
            "models": [{"provider": provider or "acemusic", "id": "ace-step"}],
        },
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_vision_provider_models",
        lambda *, task=None, base_url=None, provider_api_key=None, provider=None, providers_only=False: {
            "task": task,
            "provider": provider,
            "providers_only": providers_only,
            "base_url": base_url,
            "provider_api_key": provider_api_key,
        },
    )
    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries.local_list_cached_vision_models",
        lambda *, task=None, provider=None: {"task": task, "provider": provider},
    )

    default_client = object.__new__(LocalAbstractCoreLLMClient)
    default_client._provider = "mlx"
    default_client._model = "qwen"

    client = object.__new__(MultiLocalAbstractCoreLLMClient)
    client._default_provider = "mlx"
    client._default_model = "qwen"
    client._default_client = default_client

    assert client.list_providers(include_models=True) == {
        "items": [{"name": "mlx"}],
        "default_model": "qwen",
        "include_models": True,
    }
    assert client.list_provider_models("ollama", base_url="http://provider.test/v1", api_key="secret") == {
        "provider": "ollama",
        "models": ["http://provider.test/v1"],
        "provider_api_key": "secret",
        "input_type": None,
        "output_type": None,
        "timeout_s": None,
    }
    assert client.list_embedding_models(base_url="http://provider.test/v1", api_key="secret", provider="lmstudio") == {
        "provider": "lmstudio",
        "providers_only": False,
        "base_url": "http://provider.test/v1",
        "provider_api_key": "secret",
        "timeout_s": None,
        "models": ["bge-small-en-v1.5"],
    }
    assert client.get_model_capabilities("ollama/qwen") == {
        "model_id": "ollama/qwen",
        "max_tokens": 123,
    }
    assert client.get_voice_catalog(base_url="http://provider.test/v1", api_key="secret", provider="openai") == {
        "provider": "openai",
        "model": None,
        "providers_only": False,
        "base_url": "http://provider.test/v1",
        "provider_api_key": "secret",
    }
    assert client.list_tts_models(base_url="http://provider.test/v1", api_key="secret", provider="openai") == {
        "provider": "openai",
        "base_url": "http://provider.test/v1",
        "provider_api_key": "secret",
    }
    assert client.list_stt_models(provider="openai") == {
        "provider": "openai",
        "base_url": None,
        "provider_api_key": None,
    }
    assert client.list_music_providers(task="text_to_music", base_url="http://provider.test/v1", api_key="secret") == {
        "task": "text_to_music",
        "base_url": "http://provider.test/v1",
        "provider_api_key": "secret",
        "providers": ["acemusic"],
    }
    assert client.list_music_models(
        task="text_to_music",
        base_url="http://provider.test/v1",
        api_key="secret",
        provider="acemusic",
    ) == {
        "task": "text_to_music",
        "provider": "acemusic",
        "base_url": "http://provider.test/v1",
        "provider_api_key": "secret",
        "models": [{"provider": "acemusic", "id": "ace-step"}],
    }
    assert client.list_vision_provider_models(task="text_to_image", provider="mflux", providers_only=True) == {
        "task": "text_to_image",
        "provider": "mflux",
        "providers_only": True,
        "base_url": None,
        "provider_api_key": None,
    }
    assert client.list_cached_vision_models(task="text_to_image", provider="mflux") == {
        "task": "text_to_image",
        "provider": "mflux",
    }


def test_local_discovery_methods_shape_snapshot_responses(monkeypatch) -> None:
    class _FakeVoice:
        backend_id = "voice-backend"

        def voice_catalog(
            self,
            provider: str | None = None,
            model: str | None = None,
            providers_only: bool = False,
        ) -> Dict[str, Any]:
            _ = provider, model, providers_only
            return {
                "profiles": [{"id": "alloy", "provider": "openai", "model": "tts-1"}],
                "tts_providers": ["openai"],
                "stt_providers": ["openai"],
                "tts_models_by_provider": {"openai": ["tts-1"]},
                "stt_models_by_provider": {"openai": ["whisper-1"]},
            }

        def list_tts_models(self, provider: str | None = None) -> List[str]:
            _ = provider
            return ["tts-1"]

        def list_stt_models(self) -> List[str]:
            return ["whisper-1"]

    class _FakeVision:
        backend_id = "vision-backend"

        def available_providers(self, *, task: str | None = None) -> Dict[str, Any]:
            _ = task
            return {"providers": ["mflux"], "available_providers": ["mflux"]}

        def list_provider_models(self, *, task: str | None = None) -> List[Dict[str, Any]]:
            _ = task
            return [{"provider": "mflux", "model": "flux-dev"}]

    class _FakeMusic:
        backend_id = "music-backend"

        def available_providers(self, *, task: str | None = None) -> List[Dict[str, Any]]:
            _ = task
            return [{"provider": "acemusic", "tasks": ["text_to_music"]}]

        def list_models(self, *, task: str | None = None, provider: str | None = None) -> List[Dict[str, Any]]:
            _ = task
            models = [
                {"provider": "acemusic", "id": "ace-step"},
                {"provider": "acemusic", "model": "ace-pro"},
            ]
            if provider:
                return [item for item in models if item.get("provider") == provider]
            return models

    class _FakeRegistry:
        def __init__(self) -> None:
            self.voice = _FakeVoice()
            self.music = _FakeMusic()
            self.vision = _FakeVision()

    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.discovery_queries._runtime_capability_registry",
        lambda **_kwargs: _FakeRegistry(),
    )
    monkeypatch.setattr(
        "abstractcore.providers.registry.get_all_providers_with_models",
        lambda include_models=False: [{"name": "mlx", "include_models": include_models}],
    )
    monkeypatch.setattr(
        "abstractcore.providers.registry.get_available_models_for_provider",
        lambda provider_name, **_kwargs: ["qwen", "llama"] if provider_name == "ollama" else [],
    )
    monkeypatch.setattr(
        "abstractcore.embeddings.list_available_models",
        lambda: ["all-MiniLM-L6-v2", "bge-small-en-v1.5"],
    )
    monkeypatch.setattr(
        "abstractcore.architectures.detection.get_model_capabilities",
        lambda model_name: {"model_id": model_name, "max_tokens": 2048},
    )

    def fake_local_vision_catalog() -> Dict[str, Any]:
        return {
            "models": [
                {"id": "flux-dev", "provider": "mflux", "tasks": ["text_to_image"]},
                {"id": "other", "provider": "openai", "tasks": ["text_to_image"]},
            ]
        }

    monkeypatch.setattr(
        "abstractcore.capabilities.vision_catalog.get_local_vision_cache_catalog",
        fake_local_vision_catalog,
    )

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "qwen"

    providers = client.list_providers(include_models=True)
    provider_models = client.list_provider_models("ollama", base_url="http://provider.test/v1", api_key="secret")
    capabilities = client.get_model_capabilities("ollama/qwen")
    voices = client.get_voice_catalog(provider="openai", providers_only=True)
    tts = client.list_tts_models(provider="openai")
    stt = client.list_stt_models(provider="openai")
    music_providers = client.list_music_providers(task="text_to_music")
    music_models = client.list_music_models(task="text_to_music", provider="acemusic")
    vision = client.list_vision_provider_models(task="text_to_image", provider="mflux", providers_only=True)
    cached = client.list_cached_vision_models(task="text_to_image", provider="mflux")
    embeddings = client.list_embedding_models(provider="huggingface")

    assert providers["items"] == [{"name": "mlx", "include_models": True}]
    assert providers["default_provider"] == "mlx"
    assert providers["default_model"] == "qwen"
    assert provider_models["provider"] == "ollama"
    assert provider_models["models"] == ["llama", "qwen"]
    assert embeddings["provider"] == "huggingface"
    assert embeddings["models"]
    assert embeddings["models_by_provider"]["huggingface"] == embeddings["models"]
    assert capabilities == {"model_id": "ollama/qwen", "max_tokens": 2048}
    assert voices["providers"] == ["openai"]
    assert voices["profiles"] == []
    assert tts["models"] == ["tts-1"]
    assert stt["models"] == ["whisper-1"]
    assert music_providers["providers"] == ["acemusic"]
    assert music_providers["provider_details"] == [{"provider": "acemusic", "tasks": ["text_to_music"]}]
    assert music_models["providers"] == ["acemusic"]
    assert music_models["models_by_provider"] == {"acemusic": ["ace-step", "ace-pro"]}
    assert vision["providers"] == ["mflux"]
    assert vision["models"] == []
    assert cached["models"] == [{"id": "flux-dev", "provider": "mflux", "tasks": ["text_to_image"]}]


def test_local_stt_provider_models_use_provider_scoped_fast_path(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import discovery_queries

    calls: list[str] = []

    class _FakeVoice:
        def voice_catalog(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            _ = args, kwargs
            calls.append("voice_catalog")
            raise AssertionError("provider-scoped STT model lookup must not build the full voice catalog")

        def list_stt_models(self, provider: str | None = None) -> List[str]:
            calls.append(f"list_stt_models:{provider}")
            return ["large-v3"] if provider == "faster-whisper" else []

    class _FakeRegistry:
        voice = _FakeVoice()

    monkeypatch.setattr(discovery_queries, "_runtime_capability_registry", lambda **_kwargs: _FakeRegistry())

    payload = discovery_queries.local_list_stt_models(provider="faster-whisper")

    assert calls == ["list_stt_models:faster-whisper"]
    assert payload["models"] == ["large-v3"]
    assert payload["stt_models_by_provider"] == {"faster-whisper": ["large-v3"]}
    assert payload["provider_models"] == [
        {"id": "faster-whisper/large-v3", "provider": "faster-whisper", "model": "large-v3"}
    ]


def test_local_embedding_provider_catalog_comes_from_core(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import discovery_queries

    monkeypatch.setattr(
        "abstractcore.embeddings.models.list_provider_details",
        lambda provider=None: [
            {
                "id": "customembed",
                "provider": "customembed",
                "label": "Custom Embeddings",
                "transport": "openai_compatible_http",
            }
        ]
        if provider in (None, "customembed")
        else [],
    )

    payload = discovery_queries.local_list_embedding_models(provider="customembed", providers_only=True)

    assert not hasattr(discovery_queries, "_EMBEDDING_PROVIDER_DETAILS")
    assert payload["providers"] == ["customembed"]
    assert payload["embedding_providers"] == ["customembed"]
    assert payload["provider_details"] == [
        {
            "id": "customembed",
            "provider": "customembed",
            "label": "Custom Embeddings",
            "transport": "openai_compatible_http",
        }
    ]


def test_local_cached_vision_models_do_not_import_core_server_helper(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import discovery_queries

    monkeypatch.setitem(sys.modules, "abstractcore.server.vision_endpoints", None)
    monkeypatch.setattr(
        "abstractcore.capabilities.vision_catalog.get_local_vision_cache_catalog",
        lambda: {"models": [{"id": "flux-dev", "provider": "mflux", "tasks": ["text_to_image"]}]},
    )

    payload = discovery_queries.local_list_cached_vision_models(task="text_to_image", provider="mflux")

    assert payload["available"] is True
    assert payload["source"] == "abstractvision.local_cache"
    assert payload["models"] == [{"id": "flux-dev", "provider": "mflux", "tasks": ["text_to_image"]}]


def test_local_cached_vision_models_preserve_helper_error_details(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import discovery_queries

    monkeypatch.setattr(
        "abstractcore.capabilities.vision_catalog.get_local_vision_cache_catalog",
        lambda: {
            "models": [],
            "registry_available": False,
            "error": "Failed to initialize AbstractVision registry: registry init failed",
        },
    )

    payload = discovery_queries.local_list_cached_vision_models(task="text_to_image")

    assert payload["available"] is False
    assert payload["error"] == "Failed to initialize AbstractVision registry: registry init failed"
    assert payload["source"] == "abstractvision.local_cache"


def test_local_provider_models_forward_capability_filters(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import discovery_queries

    calls: list[Dict[str, Any]] = []

    def fake_models(provider: str, **kwargs: Any) -> list[str]:
        calls.append({"provider": provider, **kwargs})
        return ["vlm-model"]

    monkeypatch.setattr("abstractcore.providers.registry.get_available_models_for_provider", fake_models)

    payload = discovery_queries.local_list_provider_models(
        "lmstudio",
        input_type="image",
        output_type="text",
    )

    assert payload["models"] == ["vlm-model"]
    assert calls[0]["provider"] == "lmstudio"
    assert [item.value for item in calls[0]["input_capabilities"]] == ["image"]
    assert [item.value for item in calls[0]["output_capabilities"]] == ["text"]


def test_local_provider_models_reject_invalid_capability_filters(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import discovery_queries

    called = False

    def fake_models(provider: str, **kwargs: Any) -> list[str]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr("abstractcore.providers.registry.get_available_models_for_provider", fake_models)

    payload = discovery_queries.local_list_provider_models("lmstudio", input_type="unsupported")

    assert payload["available"] is False
    assert payload["models"] == []
    assert payload["error"] == "Unsupported input_type filter: unsupported"
    assert called is False


def test_discovery_facade_preserves_model_capability_lookup_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        "abstractcore.architectures.detection.get_model_capabilities",
        lambda _model_name: (_ for _ in ()).throw(RuntimeError("capability lookup failed")),
    )

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "qwen"
    runtime = SimpleNamespace(_abstractcore_llm_client=client)

    payload = get_abstractcore_discovery_facade(runtime).get_model_capabilities("bad/model")

    assert payload["model"] == "bad/model"
    assert payload["capabilities"] == {}
    assert payload["available"] is False
    assert payload["error"] == "capability lookup failed"
    assert payload["source"] == "abstractcore.local"


def test_remote_discovery_methods_proxy_core_catalogs_and_normalize_shapes(monkeypatch) -> None:
    monkeypatch.setattr(
        "abstractcore.architectures.detection.get_model_capabilities",
        lambda model_name: {"model_id": model_name, "max_tokens": 777},
    )
    sender = _RecordingRequestSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        timeout_s=12.0,
        headers={"X-Test": "1"},
    )
    runtime = SimpleNamespace(_abstractcore_llm_client=client)
    facade = get_abstractcore_discovery_facade(runtime)

    providers = facade.list_providers(include_models=True)
    provider_models = facade.list_provider_models(
        "ollama",
        base_url="http://provider.test/v1",
        api_key="secret",
    )
    filtered_provider_models = facade.list_provider_models(
        "ollama",
        input_type="image",
        output_type="text",
    )
    embeddings = facade.list_embedding_models(
        provider="lmstudio",
        base_url="http://provider.test/v1",
        api_key="secret",
    )
    embedding_providers = facade.list_embedding_models(
        provider="lmstudio",
        providers_only=True,
        api_key="secret",
    )
    capabilities = facade.get_model_capabilities("openai/gpt-4o-mini")
    voices = facade.get_voice_catalog(
        base_url="http://provider.test/v1",
        provider="openai",
        providers_only=True,
        api_key="secret",
    )
    tts = facade.list_tts_models(
        base_url="http://provider.test/v1",
        provider="openai",
        api_key="secret",
    )
    stt = facade.list_stt_models(provider="openai")
    music_providers = facade.list_music_providers(
        task="text_to_music",
        base_url="http://provider.test/v1",
        api_key="secret",
    )
    music_models = facade.list_music_models(
        task="text_to_music",
        base_url="http://provider.test/v1",
        provider="acemusic",
        api_key="secret",
    )
    vision = facade.list_vision_provider_models(task="text_to_image", provider="mflux", providers_only=True)
    cached = facade.list_cached_vision_models(task="text_to_image", provider="mflux")

    assert providers["items"] == [{"name": "ollama"}, {"name": "openai"}]
    assert providers["default_provider"] == "openai"
    assert providers["default_model"] == "gpt-4o-mini"
    assert provider_models["provider"] == "ollama"
    assert provider_models["models"] == ["llama", "qwen"]
    assert provider_models["source"] == "abstractcore.remote"
    assert provider_models["available"] is True
    assert filtered_provider_models["models"] == ["llava"]
    assert embeddings["provider"] == "lmstudio"
    assert embeddings["models"] == ["bge-small-en-v1.5", "text-embedding-nomic-embed-text-v1.5"]
    assert embeddings["models_by_provider"] == {"lmstudio": ["bge-small-en-v1.5", "text-embedding-nomic-embed-text-v1.5"]}
    assert embeddings["source"] == "abstractcore.remote"
    assert embedding_providers["providers"] == ["lmstudio"]
    assert embedding_providers["provider_details"] == [
        {
            "id": "lmstudio",
            "provider": "lmstudio",
            "label": "LMStudio",
            "transport": "openai_compatible_http",
            "base_url_configurable": True,
            "base_url_env_vars": ["LMSTUDIO_BASE_URL"],
            "default_base_url": "http://localhost:1234/v1",
        },
    ]
    assert embedding_providers["source"] == "abstractcore.remote"
    assert capabilities["model"] == "openai/gpt-4o-mini"
    assert capabilities["capabilities"] == {"model_id": "openai/gpt-4o-mini", "max_tokens": 777}
    assert capabilities["available"] is True
    assert capabilities["error"] is None
    assert capabilities["source"] == "abstractcore.local"
    assert voices["providers"] == ["openai"]
    assert voices["profiles"] == []
    assert tts["models"] == ["tts-1"]
    assert stt["models"] == ["whisper-1"]
    assert music_providers["providers"] == ["acemusic", "suno"]
    assert music_providers["available_providers"] == ["acemusic", "suno"]
    assert music_models["providers"] == ["acemusic"]
    assert music_models["models_by_provider"] == {"acemusic": ["ace-step", "ace-pro"]}
    assert vision["providers"] == ["mflux"]
    assert vision["models"] == []
    assert cached["models"] == [{"id": "flux-dev", "provider": "mflux", "tasks": ["text_to_image"]}]

    assert sender.calls == [
        {
            "method": "GET",
            "url": "http://core.test/providers?include_models=true",
            "headers": {"X-Test": "1"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/models?provider=ollama&base_url=http%3A%2F%2Fprovider.test%2Fv1",
            "headers": {"X-Test": "1", "X-AbstractCore-Provider-API-Key": "secret"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/models?provider=ollama&input_type=image&output_type=text",
            "headers": {"X-Test": "1"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/models?provider=lmstudio&output_type=embeddings&base_url=http%3A%2F%2Fprovider.test%2Fv1",
            "headers": {"X-Test": "1", "X-AbstractCore-Provider-API-Key": "secret"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/embeddings/providers?provider=lmstudio",
            "headers": {"X-Test": "1", "X-AbstractCore-Provider-API-Key": "secret"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/audio/voices?base_url=http%3A%2F%2Fprovider.test%2Fv1&provider=openai&providers_only=true",
            "headers": {"X-Test": "1", "X-AbstractCore-Provider-API-Key": "secret"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/audio/speech/models?base_url=http%3A%2F%2Fprovider.test%2Fv1&provider=openai",
            "headers": {"X-Test": "1", "X-AbstractCore-Provider-API-Key": "secret"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/audio/transcriptions/models?provider=openai",
            "headers": {"X-Test": "1"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/audio/music/providers?task=text_to_music&base_url=http%3A%2F%2Fprovider.test%2Fv1",
            "headers": {"X-Test": "1", "X-AbstractCore-Provider-API-Key": "secret"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/audio/music/models?task=text_to_music&provider=acemusic&base_url=http%3A%2F%2Fprovider.test%2Fv1",
            "headers": {"X-Test": "1", "X-AbstractCore-Provider-API-Key": "secret"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/vision/provider_models?task=text_to_image&provider=mflux&providers_only=true",
            "headers": {"X-Test": "1"},
            "timeout": 12.0,
        },
        {
            "method": "GET",
            "url": "http://core.test/v1/vision/models?task=text_to_image&provider=mflux",
            "headers": {"X-Test": "1"},
            "timeout": 12.0,
        },
    ]


def test_remote_discovery_preserves_status_and_timeout_on_failures() -> None:
    sender = _FailingDiscoverySender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        timeout_s=12.0,
        headers={"X-Test": "1"},
    )

    payload = client.list_providers(include_models=True, timeout_s=3.5)

    assert payload["available"] is False
    assert payload["status_code"] == 403
    assert payload["upstream_error"] == {"error": {"message": "forbidden"}}
    assert sender.calls == [
        {
            "method": "GET",
            "url": "http://core.test/providers?include_models=true",
            "headers": {"X-Test": "1"},
            "timeout": 3.5,
        }
    ]


def test_remote_discovery_marks_transport_failures_as_route_unavailable() -> None:
    sender = _TransportFailingDiscoverySender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        timeout_s=12.0,
        headers={"X-Test": "1"},
    )

    payload = client.list_tts_models(provider="openai")

    assert payload["available"] is False
    assert payload["route_available"] is False
    assert payload["error"] == "connection refused"
