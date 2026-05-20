"""Public host-facing AbstractCore control facade for Runtime integrations.

Hosts should import this module instead of reaching through
`runtime._abstractcore_llm_client` directly.

Scope:
- prompt-cache control operations
- model-residency control operations

Non-goals:
- durable Runtime effect execution (use `AbstractCoreRunFacade` / `get_abstractcore_run_facade(...)`)
- generated media / TTS / STT helpers
- provider-private prompt-cache save/load behavior
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

_RUNTIME_ABSTRACTCORE_CLIENT_ATTR = "_abstractcore_llm_client"
_HOST_CONTROL_METHODS = (
    "get_prompt_cache_capabilities",
    "get_prompt_cache_stats",
    "prompt_cache_set",
    "prompt_cache_update",
    "prompt_cache_fork",
    "prompt_cache_clear",
    "prompt_cache_prepare_modules",
    "list_model_residency",
    "load_model_residency",
    "unload_model_residency",
)


class AbstractCoreHostControlClient(Protocol):
    """Duck-typed control client contract implemented by AbstractCore LLM clients."""

    def get_prompt_cache_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        ...

    def get_prompt_cache_stats(self, **kwargs: Any) -> Dict[str, Any]:
        ...

    def prompt_cache_set(
        self,
        *,
        key: str,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_fork(
        self,
        *,
        from_key: str,
        to_key: str,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_clear(
        self,
        *,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def load_model_residency(
        self,
        *,
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


def _coerce_control_client(client: Any, *, source: str) -> AbstractCoreHostControlClient:
    missing = [name for name in _HOST_CONTROL_METHODS if not callable(getattr(client, name, None))]
    if missing:
        methods = ", ".join(missing)
        raise TypeError(
            f"{source} does not implement the AbstractCore host control contract. Missing methods: {methods}."
        )
    return client


def _client_from_runtime(runtime: Any) -> AbstractCoreHostControlClient:
    client = getattr(runtime, _RUNTIME_ABSTRACTCORE_CLIENT_ATTR, None)
    if client is None:
        raise RuntimeError(
            "Runtime is not wired to AbstractCore host controls. "
            "Use an AbstractCore runtime factory, then obtain the facade via "
            "`abstractruntime.integrations.abstractcore.get_abstractcore_host_facade(runtime)`."
        )
    return _coerce_control_client(client, source="Runtime AbstractCore client")


class AbstractCoreHostFacade:
    """Public host-facing control surface for AbstractCore-backed runtimes.

    This facade is intentionally narrow and operator-scoped. Its methods are
    synchronous helpers and do not create durable Runtime history on their own.
    """

    __slots__ = ("_client",)

    def __init__(self, runtime: Any):
        self._client = _client_from_runtime(runtime)

    @classmethod
    def from_runtime(cls, runtime: Any) -> "AbstractCoreHostFacade":
        """Bind the public host facade to an AbstractCore-wired runtime."""

        return cls(runtime)

    def get_prompt_cache_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        return self._client.get_prompt_cache_capabilities(**kwargs)

    def get_prompt_cache_stats(self, **kwargs: Any) -> Dict[str, Any]:
        return self._client.get_prompt_cache_stats(**kwargs)

    def prompt_cache_set(
        self,
        *,
        key: str,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_set(
            key=key,
            make_default=make_default,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_update(
            key=key,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_fork(
        self,
        *,
        from_key: str,
        to_key: str,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_fork(
            from_key=from_key,
            to_key=to_key,
            make_default=make_default,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_clear(
        self,
        *,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_clear(key=key, **kwargs)

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_prepare_modules(
            namespace=namespace,
            modules=modules,
            make_default=make_default,
            ttl_s=ttl_s,
            version=version,
            **kwargs,
        )

    def list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_model_residency(
            task=task,
            provider=provider,
            model=model,
            **kwargs,
        )

    def load_model_residency(
        self,
        *,
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.load_model_residency(
            task=task,
            provider=provider,
            model=model,
            options=options,
            pin=pin,
            **kwargs,
        )

    def unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.unload_model_residency(
            task=task,
            runtime_id=runtime_id,
            provider=provider,
            model=model,
            options=options,
            **kwargs,
        )


def get_abstractcore_host_facade(runtime: Any) -> AbstractCoreHostFacade:
    """Return the public AbstractCore host facade bound to a runtime."""

    return AbstractCoreHostFacade.from_runtime(runtime)


__all__ = [
    "AbstractCoreHostFacade",
    "get_abstractcore_host_facade",
]
