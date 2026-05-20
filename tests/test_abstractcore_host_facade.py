from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from abstractruntime.integrations import abstractcore
from abstractruntime.integrations.abstractcore import factory as abstractcore_factory
from abstractruntime.integrations.abstractcore import host_facade
from abstractruntime.integrations.abstractcore.host_facade import AbstractCoreHostFacade, get_abstractcore_host_facade


class _HttpResponse:
    def __init__(self, body: Dict[str, Any], *, headers: Dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = dict(headers or {})


class _RecordingRequestSender:
    def __init__(
        self,
        *,
        get_responses: List[Dict[str, Any]] | None = None,
        post_responses: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._get_responses = list(get_responses or [])
        self._post_responses = list(post_responses or [])

    def get(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        timeout: float,
    ) -> _HttpResponse:
        self.calls.append(
            {
                "method": "get",
                "url": url,
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        return _HttpResponse(self._get_responses.pop(0))

    def post(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: float,
    ) -> _HttpResponse:
        self.calls.append(
            {
                "method": "post",
                "url": url,
                "headers": dict(headers),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        return _HttpResponse(self._post_responses.pop(0))


class _RecordingHostClient:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.responses: Dict[str, Dict[str, Any]] = {
            "get_prompt_cache_capabilities": {"supported": True, "operation": "capabilities"},
            "get_prompt_cache_stats": {"supported": True, "operation": "stats"},
            "prompt_cache_set": {"ok": True, "operation": "set"},
            "prompt_cache_update": {"ok": True, "operation": "update"},
            "prompt_cache_fork": {"ok": True, "operation": "fork"},
            "prompt_cache_clear": {"ok": True, "operation": "clear"},
            "prompt_cache_prepare_modules": {"ok": True, "operation": "prepare_modules"},
            "list_model_residency": {"ok": True, "operation": "list_loaded", "models": []},
            "load_model_residency": {"ok": True, "operation": "load"},
            "unload_model_residency": {"ok": True, "operation": "unload"},
        }

    def _record(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append((name, dict(kwargs)))
        return self.responses[name]

    def get_prompt_cache_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("get_prompt_cache_capabilities", **kwargs)

    def get_prompt_cache_stats(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("get_prompt_cache_stats", **kwargs)

    def prompt_cache_set(self, *, key: str, make_default: bool = True, ttl_s: float | None = None, **kwargs: Any) -> Dict[str, Any]:
        return self._record("prompt_cache_set", key=key, make_default=make_default, ttl_s=ttl_s, **kwargs)

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: str | None = None,
        messages: List[Dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        tools: List[Dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
        ttl_s: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "prompt_cache_update",
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
        ttl_s: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "prompt_cache_fork",
            from_key=from_key,
            to_key=to_key,
            make_default=make_default,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_clear(self, *, key: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        return self._record("prompt_cache_clear", key=key, **kwargs)

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: float | None = None,
        version: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "prompt_cache_prepare_modules",
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
        task: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record("list_model_residency", task=task, provider=provider, model=model, **kwargs)

    def load_model_residency(
        self,
        *,
        task: str,
        provider: str | None = None,
        model: str | None = None,
        options: Dict[str, Any] | None = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "load_model_residency",
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
        task: str | None = None,
        runtime_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        options: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._record(
            "unload_model_residency",
            task=task,
            runtime_id=runtime_id,
            provider=provider,
            model=model,
            options=options,
            **kwargs,
        )


class _FactoryLocalHostClient(_RecordingHostClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self._llm = object()

    def get_model_capabilities(self) -> Dict[str, Any]:
        return {}


class _FactoryToolExecutor:
    def __init__(self, *, timeout_s: float | None = None) -> None:
        self.timeout_s = timeout_s

    def set_timeout_s(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s


def test_public_host_facade_exports_are_available() -> None:
    assert abstractcore.AbstractCoreHostFacade is AbstractCoreHostFacade
    assert abstractcore.get_abstractcore_host_facade is get_abstractcore_host_facade
    assert "AbstractCoreHostFacade" in abstractcore.__all__
    assert "get_abstractcore_host_facade" in abstractcore.__all__
    assert "AbstractCoreHostFacade" in host_facade.__all__
    assert "get_abstractcore_host_facade" in host_facade.__all__


def test_host_facade_can_be_built_directly_or_from_runtime_helper() -> None:
    client = _RecordingHostClient()
    runtime = SimpleNamespace(_abstractcore_llm_client=client)

    direct = AbstractCoreHostFacade(runtime)
    from_runtime = AbstractCoreHostFacade.from_runtime(runtime)
    runtime_helper = get_abstractcore_host_facade(runtime)

    assert direct._client is client
    assert from_runtime._client is client
    assert runtime_helper._client is client


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AbstractCoreHostFacade(SimpleNamespace(_abstractcore_llm_client=object())),
        lambda: AbstractCoreHostFacade.from_runtime(SimpleNamespace(_abstractcore_llm_client=object())),
        lambda: get_abstractcore_host_facade(SimpleNamespace(_abstractcore_llm_client=object())),
    ],
)
def test_host_facade_rejects_runtime_clients_that_do_not_match_the_contract(factory) -> None:
    with pytest.raises(TypeError, match="Missing methods"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AbstractCoreHostFacade.from_runtime(object()),
        lambda: get_abstractcore_host_facade(object()),
    ],
)
def test_host_facade_runtime_helpers_raise_when_runtime_has_no_client(factory) -> None:
    with pytest.raises(RuntimeError, match="Runtime is not wired to AbstractCore host controls"):
        factory()


def test_host_facade_delegates_prompt_cache_operations() -> None:
    client = _RecordingHostClient()
    facade = AbstractCoreHostFacade(SimpleNamespace(_abstractcore_llm_client=client))

    capabilities = facade.get_prompt_cache_capabilities(provider="mlx", model="qwen")
    stats = facade.get_prompt_cache_stats(base_url="http://provider.test/v1")
    set_result = facade.prompt_cache_set(key="sess:1", make_default=False, ttl_s=30, provider="mlx")
    update_result = facade.prompt_cache_update(
        key="sess:1",
        prompt="hello",
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="SYSTEM",
        tools=[{"type": "function", "function": {"name": "calc"}}],
        add_generation_prompt=True,
        ttl_s=45,
        provider="mlx",
        model="qwen",
    )
    fork_result = facade.prompt_cache_fork(
        from_key="sess:1",
        to_key="sess:2",
        make_default=True,
        ttl_s=60,
        base_url="http://provider.test/v1",
    )
    clear_result = facade.prompt_cache_clear(key="sess:2", provider_api_key="sekret")
    prepare_result = facade.prompt_cache_prepare_modules(
        namespace="tenant:model",
        modules=[{"module_id": "system", "system_prompt": "SYSTEM"}],
        make_default=True,
        ttl_s=120,
        version=2,
        provider="mlx",
        model="qwen",
    )

    assert capabilities == {"supported": True, "operation": "capabilities"}
    assert stats == {"supported": True, "operation": "stats"}
    assert set_result == {"ok": True, "operation": "set"}
    assert update_result == {"ok": True, "operation": "update"}
    assert fork_result == {"ok": True, "operation": "fork"}
    assert clear_result == {"ok": True, "operation": "clear"}
    assert prepare_result == {"ok": True, "operation": "prepare_modules"}
    assert client.calls == [
        ("get_prompt_cache_capabilities", {"provider": "mlx", "model": "qwen"}),
        ("get_prompt_cache_stats", {"base_url": "http://provider.test/v1"}),
        ("prompt_cache_set", {"key": "sess:1", "make_default": False, "ttl_s": 30, "provider": "mlx"}),
        (
            "prompt_cache_update",
            {
                "key": "sess:1",
                "prompt": "hello",
                "messages": [{"role": "user", "content": "hi"}],
                "system_prompt": "SYSTEM",
                "tools": [{"type": "function", "function": {"name": "calc"}}],
                "add_generation_prompt": True,
                "ttl_s": 45,
                "provider": "mlx",
                "model": "qwen",
            },
        ),
        (
            "prompt_cache_fork",
            {
                "from_key": "sess:1",
                "to_key": "sess:2",
                "make_default": True,
                "ttl_s": 60,
                "base_url": "http://provider.test/v1",
            },
        ),
        ("prompt_cache_clear", {"key": "sess:2", "provider_api_key": "sekret"}),
        (
            "prompt_cache_prepare_modules",
            {
                "namespace": "tenant:model",
                "modules": [{"module_id": "system", "system_prompt": "SYSTEM"}],
                "make_default": True,
                "ttl_s": 120,
                "version": 2,
                "provider": "mlx",
                "model": "qwen",
            },
        ),
    ]


def test_host_facade_delegates_model_residency_operations() -> None:
    client = _RecordingHostClient()
    facade = AbstractCoreHostFacade(SimpleNamespace(_abstractcore_llm_client=client))

    listed = facade.list_model_residency(task="text_generation", provider="mlx", model="qwen")
    loaded = facade.load_model_residency(
        task="image_generation",
        provider="mflux",
        model="flux",
        options={"steps": 4},
        pin=False,
        base_url="http://provider.test/v1",
        timeout_s=3600,
    )
    unloaded = facade.unload_model_residency(
        task="image_generation",
        runtime_id="rid-1",
        provider="mflux",
        model="flux",
        options={"graceful": True},
        provider_api_key="sekret",
    )

    assert listed == {"ok": True, "operation": "list_loaded", "models": []}
    assert loaded == {"ok": True, "operation": "load"}
    assert unloaded == {"ok": True, "operation": "unload"}
    assert client.calls == [
        ("list_model_residency", {"task": "text_generation", "provider": "mlx", "model": "qwen"}),
        (
            "load_model_residency",
            {
                "task": "image_generation",
                "provider": "mflux",
                "model": "flux",
                "options": {"steps": 4},
                "pin": False,
                "base_url": "http://provider.test/v1",
                "timeout_s": 3600,
            },
        ),
        (
            "unload_model_residency",
            {
                "task": "image_generation",
                "runtime_id": "rid-1",
                "provider": "mflux",
                "model": "flux",
                "options": {"graceful": True},
                "provider_api_key": "sekret",
            },
        ),
    ]


def test_host_facade_works_with_factory_created_remote_runtime() -> None:
    runtime = abstractcore.create_remote_runtime(
        server_base_url="http://core.test/v1",
        model="qwen3:4b",
        timeout_s=45,
    )
    facade = get_abstractcore_host_facade(runtime)
    sender = _RecordingRequestSender(
        get_responses=[
            {
                "supported": True,
                "operation": "capabilities",
                "capabilities": {"supported": True, "mode": "keyed"},
            }
        ],
        post_responses=[
            {
                "ok": True,
                "operation": "load",
                "runtime": {"runtime_id": "rid-1", "resident": True},
            }
        ],
    )

    setattr(getattr(runtime, "_abstractcore_llm_client"), "_sender", sender)

    capabilities = facade.get_prompt_cache_capabilities(
        base_url="http://provider.test/v1",
        provider_api_key="sekret",
    )
    loaded = facade.load_model_residency(
        task="text_generation",
        provider="mlx",
        model="qwen3:4b",
        options={"gpu": 0},
        pin=False,
        provider_api_key="sekret",
        timeout_s=120,
    )

    assert capabilities["capabilities"]["mode"] == "keyed"
    assert loaded["runtime"]["runtime_id"] == "rid-1"
    assert sender.calls == [
        {
            "method": "get",
            "url": "http://core.test/acore/prompt_cache/capabilities?base_url=http%3A%2F%2Fprovider.test%2Fv1",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "timeout": 45.0,
        },
        {
            "method": "post",
            "url": "http://core.test/acore/models/load",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "task": "text_generation",
                "provider": "mlx",
                "model": "qwen3:4b",
                "options": {"gpu": 0},
                "pin": False,
                "timeout_s": 120,
            },
            "timeout": 45.0,
        },
    ]


def test_host_facade_wires_local_factory_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_client = _FactoryLocalHostClient()

    monkeypatch.setattr(
        abstractcore_factory,
        "MultiLocalAbstractCoreLLMClient",
        lambda **kwargs: stub_client,
    )
    monkeypatch.setattr(
        abstractcore_factory,
        "AbstractCoreToolExecutor",
        lambda timeout_s: _FactoryToolExecutor(timeout_s=timeout_s),
    )
    monkeypatch.setattr(abstractcore_factory, "AbstractCoreChatSummarizer", lambda **kwargs: object())
    monkeypatch.setattr(abstractcore_factory, "build_effect_handlers", lambda **kwargs: {})

    runtime = abstractcore.create_local_runtime(
        provider="mlx",
        model="mlx-community/Qwen3-4B-4bit",
        llm_kwargs={"temperature": 0.1},
    )

    facade = get_abstractcore_host_facade(runtime)

    assert facade.get_prompt_cache_capabilities() == {"supported": True, "operation": "capabilities"}
    assert getattr(runtime, "_abstractcore_llm_client") is stub_client


def test_host_facade_wires_hybrid_factory_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_client = _FactoryLocalHostClient()

    monkeypatch.setattr(
        abstractcore_factory,
        "RemoteAbstractCoreLLMClient",
        lambda **kwargs: stub_client,
    )
    monkeypatch.setattr(
        abstractcore_factory,
        "AbstractCoreToolExecutor",
        lambda timeout_s: _FactoryToolExecutor(timeout_s=timeout_s),
    )
    monkeypatch.setattr(abstractcore_factory, "build_effect_handlers", lambda **kwargs: {})

    runtime = abstractcore.create_hybrid_runtime(
        server_base_url="http://core.test/v1",
        model="qwen3:4b",
        timeout_s=45,
        tool_timeout_s=12,
    )

    facade = get_abstractcore_host_facade(runtime)

    assert facade.list_model_residency() == {"ok": True, "operation": "list_loaded", "models": []}
    assert getattr(runtime, "_abstractcore_llm_client") is stub_client
