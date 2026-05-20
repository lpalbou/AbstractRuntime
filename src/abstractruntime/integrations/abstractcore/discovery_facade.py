"""Public host-facing AbstractCore discovery facade for Runtime integrations.

Hosts should import this module instead of rebuilding Core catalog queries or
reaching into runtime/client internals directly.

Scope:
- provider discovery
- provider model discovery
- model capability lookup
- audio/voice/TTS/STT catalogs
- music provider/model catalogs
- vision provider catalogs
- cached vision model snapshots

Non-goals:
- durable Runtime effect execution (use `AbstractCoreRunFacade` / `get_abstractcore_run_facade(...)`)
- prompt-cache or residency control operations (use `AbstractCoreHostFacade`)
- provider-private prompt-cache save/load behavior
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

_RUNTIME_ABSTRACTCORE_CLIENT_ATTR = "_abstractcore_llm_client"
_DISCOVERY_METHODS = (
    "list_providers",
    "list_provider_models",
    "get_voice_catalog",
    "list_tts_models",
    "list_stt_models",
    "list_music_providers",
    "list_music_models",
    "list_vision_provider_models",
    "list_cached_vision_models",
)


class AbstractCoreDiscoveryClient(Protocol):
    """Duck-typed discovery client contract implemented by AbstractCore LLM clients."""

    def list_providers(
        self,
        *,
        include_models: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_provider_models(
        self,
        provider_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def lookup_model_capabilities(
        self,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    def get_voice_catalog(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_tts_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_stt_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_music_providers(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_music_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_vision_provider_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_cached_vision_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


def _coerce_discovery_client(client: Any, *, source: str) -> AbstractCoreDiscoveryClient:
    missing = [name for name in _DISCOVERY_METHODS if not callable(getattr(client, name, None))]
    if not callable(getattr(client, "lookup_model_capabilities", None)) and not callable(
        getattr(client, "get_model_capabilities", None)
    ):
        missing.append("lookup_model_capabilities|get_model_capabilities")
    if missing:
        methods = ", ".join(missing)
        raise TypeError(
            f"{source} does not implement the AbstractCore discovery contract. Missing methods: {methods}."
        )
    return client


def _client_from_runtime(runtime: Any) -> AbstractCoreDiscoveryClient:
    client = getattr(runtime, _RUNTIME_ABSTRACTCORE_CLIENT_ATTR, None)
    if client is None:
        raise RuntimeError(
            "Runtime is not wired to AbstractCore discovery helpers. "
            "Use an AbstractCore runtime factory, then obtain the facade via "
            "`abstractruntime.integrations.abstractcore.get_abstractcore_discovery_facade(runtime)`."
        )
    return _coerce_discovery_client(client, source="Runtime AbstractCore client")


class AbstractCoreDiscoveryFacade:
    """Public host-facing snapshot/query surface for AbstractCore-backed runtimes.

    This facade is intentionally query-oriented. Its methods do not create
    durable Runtime history on their own.
    """

    __slots__ = ("_client",)

    def __init__(self, runtime: Any):
        self._client = _client_from_runtime(runtime)

    @classmethod
    def from_runtime(cls, runtime: Any) -> "AbstractCoreDiscoveryFacade":
        """Bind the public discovery facade to an AbstractCore-wired runtime."""

        return cls(runtime)

    def list_providers(
        self,
        *,
        include_models: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_providers(include_models=include_models, **kwargs)

    def list_provider_models(
        self,
        provider_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_provider_models(provider_name, **kwargs)

    def get_model_capabilities(
        self,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        lookup = getattr(self._client, "lookup_model_capabilities", None)
        if callable(lookup):
            result = lookup(model_name=model_name)
        else:
            result = self._client.get_model_capabilities(model_name=model_name)
        if isinstance(result, dict) and "capabilities" in result:
            return result
        resolved_model = model_name
        if resolved_model is None:
            for attr in ("_model", "_default_model"):
                value = getattr(self._client, attr, None)
                if isinstance(value, str) and value.strip():
                    resolved_model = value.strip()
                    break
        return {
            "model": resolved_model,
            "capabilities": dict(result) if isinstance(result, dict) else {},
        }

    def get_voice_catalog(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.get_voice_catalog(
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
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_tts_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )

    def list_stt_models(
        self,
        *,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_stt_models(
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )

    def list_music_providers(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_music_providers(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            **kwargs,
        )

    def list_music_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_music_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )

    def list_vision_provider_models(
        self,
        *,
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        providers_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_vision_provider_models(
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
        task: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_cached_vision_models(
            task=task,
            base_url=base_url,
            provider_api_key=provider_api_key,
            provider=provider,
            **kwargs,
        )


def get_abstractcore_discovery_facade(runtime: Any) -> AbstractCoreDiscoveryFacade:
    """Return the public AbstractCore discovery facade bound to a runtime."""

    return AbstractCoreDiscoveryFacade.from_runtime(runtime)


__all__ = [
    "AbstractCoreDiscoveryFacade",
    "get_abstractcore_discovery_facade",
]
