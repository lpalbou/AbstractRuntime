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
            "upsert_text_bloc": {"ok": True, "operation": "upsert_text", "record": {"bloc_id": 7}},
            "get_bloc_record": {"ok": True, "operation": "record", "record": {"bloc_id": 7}},
            "list_blocs": {"ok": True, "operation": "list", "records": [{"bloc_id": 7}]},
            "get_bloc_kv_manifest": {"ok": True, "operation": "kv_manifest", "manifest": {"binding_id": "bind-1"}},
            "ensure_bloc_kv_artifact": {"ok": True, "operation": "kv_ensure", "artifact": {"binding_id": "bind-1"}},
            "load_bloc_kv_artifact": {
                "ok": True,
                "operation": "kv_load",
                "artifact": {"key": "bloc:bound", "prompt_cache_binding": {"binding_id": "bind-1", "key": "bloc:bound"}},
            },
            "list_bloc_kv_artifacts": {
                "ok": True,
                "operation": "kv_list",
                "artifacts": [{"artifact_path": "/tmp/orbit.kv", "provider": "mlx", "model": "qwen3:4b"}],
            },
            "delete_bloc_kv_artifact": {
                "ok": True,
                "operation": "kv_delete",
                "result": {"deleted": True, "artifact_path": "/tmp/orbit.kv"},
            },
            "prune_bloc_kv_artifacts": {
                "ok": True,
                "operation": "kv_prune",
                "results": [{"deleted": True, "artifact_path": "/tmp/orbit.kv"}],
            },
            "delete_bloc": {"ok": True, "operation": "delete", "result": {"deleted": True, "record": {"bloc_id": 7}}},
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

    def upsert_text_bloc(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("upsert_text_bloc", **kwargs)

    def get_bloc_record(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("get_bloc_record", **kwargs)

    def list_blocs(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("list_blocs", **kwargs)

    def get_bloc_kv_manifest(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("get_bloc_kv_manifest", **kwargs)

    def ensure_bloc_kv_artifact(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("ensure_bloc_kv_artifact", **kwargs)

    def load_bloc_kv_artifact(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("load_bloc_kv_artifact", **kwargs)

    def list_bloc_kv_artifacts(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("list_bloc_kv_artifacts", **kwargs)

    def delete_bloc_kv_artifact(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("delete_bloc_kv_artifact", **kwargs)

    def prune_bloc_kv_artifacts(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("prune_bloc_kv_artifacts", **kwargs)

    def delete_bloc(self, **kwargs: Any) -> Dict[str, Any]:
        return self._record("delete_bloc", **kwargs)

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


def test_host_facade_email_helpers_delegate_to_local_comms_tools_even_for_remote_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = abstractcore.create_remote_runtime(
        server_base_url="http://core.test/v1",
        model="qwen3:4b",
        timeout_s=45,
    )
    facade = get_abstractcore_host_facade(runtime)
    sender = _RecordingRequestSender()
    setattr(getattr(runtime, "_abstractcore_llm_client"), "_sender", sender)

    from abstractcore.tools import comms_tools as core_comms_tools

    calls: List[Tuple[str, Dict[str, Any]]] = []

    def _record(name: str, payload: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((name, dict(payload)))
        return result

    monkeypatch.setattr(
        core_comms_tools,
        "list_email_accounts",
        lambda: _record(
            "list_email_accounts",
            {},
            {"success": True, "accounts": [{"account": "ops"}]},
        ),
    )
    monkeypatch.setattr(
        core_comms_tools,
        "list_emails",
        lambda **kwargs: _record(
            "list_emails",
            kwargs,
            {"success": True, "messages": [{"uid": "42"}]},
        ),
    )
    monkeypatch.setattr(
        core_comms_tools,
        "read_email",
        lambda **kwargs: _record(
            "read_email",
            kwargs,
            {"success": True, "uid": "42", "subject": "Ops"},
        ),
    )
    monkeypatch.setattr(
        core_comms_tools,
        "send_email",
        lambda **kwargs: _record(
            "send_email",
            kwargs,
            {"success": True, "message_id": "<m-1>"},
        ),
    )

    accounts = facade.list_email_accounts()
    messages = facade.list_emails(account="ops", since="7d", status="unread", limit=5, timeout_s=12)
    email = facade.read_email(uid="42", account="ops", mailbox="INBOX", timeout_s=10, max_body_chars=8000)
    sent = facade.send_email(
        ["ops@example.com"],
        "Status",
        account="ops",
        body_text="All green",
        cc="cc@example.com",
        timeout_s=20,
        headers={"X-Test": "1"},
    )

    assert accounts == {"success": True, "accounts": [{"account": "ops"}]}
    assert messages == {"success": True, "messages": [{"uid": "42"}]}
    assert email == {"success": True, "uid": "42", "subject": "Ops"}
    assert sent == {"success": True, "message_id": "<m-1>"}
    assert calls == [
        ("list_email_accounts", {}),
        (
            "list_emails",
            {"account": "ops", "mailbox": None, "since": "7d", "status": "unread", "limit": 5, "timeout_s": 12},
        ),
        (
            "read_email",
            {"uid": "42", "account": "ops", "mailbox": "INBOX", "timeout_s": 10, "max_body_chars": 8000},
        ),
        (
            "send_email",
            {
                "to": ["ops@example.com"],
                "subject": "Status",
                "account": "ops",
                "body_text": "All green",
                "body_html": None,
                "cc": "cc@example.com",
                "bcc": None,
                "timeout_s": 20,
                "headers": {"X-Test": "1"},
            },
        ),
    ]
    assert sender.calls == []


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


def test_host_facade_delegates_bloc_and_kv_operations() -> None:
    client = _RecordingHostClient()
    facade = AbstractCoreHostFacade(SimpleNamespace(_abstractcore_llm_client=client))

    upserted = facade.upsert_text_bloc(
        path="notes/orbit.txt",
        content="Launch window is Tuesday.",
        media_type="text",
        format="text/plain",
    )
    record = facade.get_bloc_record(bloc_id=7)
    listed = facade.list_blocs(bloc_id=7)
    manifest = facade.get_bloc_kv_manifest(bloc_id=7)
    ensured = facade.ensure_bloc_kv_artifact(bloc_id=7, force_rebuild=True, debug=True)
    loaded = facade.load_bloc_kv_artifact(bloc_id=7, key="bloc:bound", make_default=True, debug=True)
    artifacts = facade.list_bloc_kv_artifacts(bloc_id=7, provider="mlx", model="qwen3:4b")
    deleted_artifact = facade.delete_bloc_kv_artifact(
        bloc_id=7,
        artifact_path="/tmp/orbit.kv",
        provider="mlx",
        model="qwen3:4b",
        clear_loaded=True,
        dry_run=True,
        debug=True,
    )
    pruned = facade.prune_bloc_kv_artifacts(
        bloc_id=7,
        provider="mlx",
        model="qwen3:4b",
        force=True,
        debug=True,
    )
    deleted_bloc = facade.delete_bloc(bloc_id=7, clear_loaded=True, dry_run=True)

    assert upserted["record"]["bloc_id"] == 7
    assert record["record"]["bloc_id"] == 7
    assert listed["records"][0]["bloc_id"] == 7
    assert manifest["manifest"]["binding_id"] == "bind-1"
    assert ensured["artifact"]["binding_id"] == "bind-1"
    assert loaded["artifact"]["prompt_cache_binding"]["key"] == "bloc:bound"
    assert artifacts["artifacts"][0]["provider"] == "mlx"
    assert deleted_artifact["result"]["deleted"] is True
    assert pruned["results"][0]["deleted"] is True
    assert deleted_bloc["result"]["record"]["bloc_id"] == 7
    assert client.calls == [
        (
            "upsert_text_bloc",
            {
                "path": "notes/orbit.txt",
                "content": "Launch window is Tuesday.",
                "sha256": None,
                "content_sha256": None,
                "media_type": "text",
                "size_bytes": None,
                "mtime_ns": None,
                "format": "text/plain",
                "estimated_tokens": None,
                "relpath_base": None,
                "summary": None,
                "keywords": None,
            },
        ),
        ("get_bloc_record", {"sha256": None, "bloc_id": 7}),
        ("list_blocs", {"sha256": None, "bloc_id": 7}),
        ("get_bloc_kv_manifest", {"sha256": None, "bloc_id": 7, "artifact_path": None}),
        (
            "ensure_bloc_kv_artifact",
            {"sha256": None, "bloc_id": 7, "artifact_path": None, "force_rebuild": True, "debug": True},
        ),
        (
            "load_bloc_kv_artifact",
            {
                "sha256": None,
                "bloc_id": 7,
                "artifact_path": None,
                "stable_cache_key": None,
                "key": "bloc:bound",
                "make_default": True,
                "force_rebuild": False,
                "debug": True,
            },
        ),
        (
            "list_bloc_kv_artifacts",
            {"sha256": None, "bloc_id": 7, "provider": "mlx", "model": "qwen3:4b"},
        ),
        (
            "delete_bloc_kv_artifact",
            {
                "sha256": None,
                "bloc_id": 7,
                "artifact_path": "/tmp/orbit.kv",
                "provider": "mlx",
                "model": "qwen3:4b",
                "clear_loaded": True,
                "force": False,
                "dry_run": True,
                "debug": True,
            },
        ),
        (
            "prune_bloc_kv_artifacts",
            {
                "sha256": None,
                "bloc_id": 7,
                "provider": "mlx",
                "model": "qwen3:4b",
                "clear_loaded": False,
                "force": True,
                "dry_run": False,
                "debug": True,
            },
        ),
        (
            "delete_bloc",
            {
                "sha256": None,
                "bloc_id": 7,
                "delete_kv": True,
                "clear_loaded": True,
                "force": False,
                "dry_run": True,
            },
        ),
    ]


def test_host_facade_forwards_bloc_routes_through_factory_created_remote_runtime() -> None:
    runtime = abstractcore.create_remote_runtime(
        server_base_url="http://core.test/v1",
        model="mlx/qwen3:4b",
        timeout_s=45,
    )
    facade = get_abstractcore_host_facade(runtime)
    sender = _RecordingRequestSender(
        get_responses=[
            {"ok": True, "operation": "record", "record": {"bloc_id": 7}},
            {"ok": True, "operation": "kv_manifest", "manifest": {"binding_id": "bind-1"}},
        ],
        post_responses=[
            {"ok": True, "operation": "upsert_text", "record": {"bloc_id": 7}},
            {"ok": True, "operation": "kv_ensure", "artifact": {"binding_id": "bind-1"}},
            {"ok": True, "operation": "kv_load", "artifact": {"key": "work:orbit", "binding_id": "bind-1"}},
        ],
    )
    setattr(getattr(runtime, "_abstractcore_llm_client"), "_sender", sender)

    upserted = facade.upsert_text_bloc(
        path="notes/orbit.txt",
        content="Orbit notes",
        media_type="text",
        format="text/plain",
        provider_api_key="sekret",
    )
    record = facade.get_bloc_record(bloc_id=7, provider_api_key="sekret")
    manifest = facade.get_bloc_kv_manifest(bloc_id=7, artifact_path="/tmp/orbit.kv", provider_api_key="sekret")
    ensured = facade.ensure_bloc_kv_artifact(bloc_id=7, force_rebuild=True, provider_api_key="sekret")
    loaded = facade.load_bloc_kv_artifact(
        bloc_id=7,
        stable_cache_key="stable:orbit",
        key="work:orbit",
        make_default=True,
        provider_api_key="sekret",
    )

    assert upserted["record"]["bloc_id"] == 7
    assert record["record"]["bloc_id"] == 7
    assert manifest["manifest"]["binding_id"] == "bind-1"
    assert ensured["artifact"]["binding_id"] == "bind-1"
    assert loaded["artifact"]["key"] == "work:orbit"
    assert sender.calls == [
        {
            "method": "post",
            "url": "http://core.test/acore/blocs/upsert_text",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "path": "notes/orbit.txt",
                "content": "Orbit notes",
                "media_type": "text",
                "format": "text/plain",
                "provider": "mlx",
                "model": "qwen3:4b",
            },
            "timeout": 45.0,
        },
        {
            "method": "get",
            "url": "http://core.test/acore/blocs/record?provider=mlx&model=qwen3%3A4b&bloc_id=7",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "timeout": 45.0,
        },
        {
            "method": "get",
            "url": "http://core.test/acore/blocs/kv/manifest?provider=mlx&model=qwen3%3A4b&bloc_id=7&artifact_path=%2Ftmp%2Forbit.kv",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "timeout": 45.0,
        },
        {
            "method": "post",
            "url": "http://core.test/acore/blocs/kv/ensure",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "bloc_id": 7,
                "force_rebuild": True,
                "debug": False,
                "provider": "mlx",
                "model": "qwen3:4b",
            },
            "timeout": 45.0,
        },
        {
            "method": "post",
            "url": "http://core.test/acore/blocs/kv/load",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "bloc_id": 7,
                "stable_cache_key": "stable:orbit",
                "key": "work:orbit",
                "make_default": True,
                "force_rebuild": False,
                "debug": False,
                "provider": "mlx",
                "model": "qwen3:4b",
            },
            "timeout": 45.0,
        },
    ]


def test_host_facade_forwards_bloc_lifecycle_routes_through_factory_created_remote_runtime() -> None:
    runtime = abstractcore.create_remote_runtime(
        server_base_url="http://core.test/v1",
        model="mlx/qwen3:4b",
        timeout_s=45,
    )
    facade = get_abstractcore_host_facade(runtime)
    sender = _RecordingRequestSender(
        get_responses=[
            {"ok": True, "operation": "list", "records": [{"bloc_id": 7}]},
            {
                "ok": True,
                "operation": "kv_list",
                "artifacts": [{"artifact_path": "/tmp/orbit.kv", "provider": "mlx", "model": "qwen3:4b"}],
            },
        ],
        post_responses=[
            {"ok": True, "operation": "kv_delete", "result": {"deleted": True, "artifact_path": "/tmp/orbit.kv"}},
            {"ok": True, "operation": "kv_prune", "results": [{"deleted": True, "artifact_path": "/tmp/orbit.kv"}]},
            {"ok": True, "operation": "delete", "result": {"deleted": False, "record": {"bloc_id": 7}, "dry_run": True}},
        ],
    )
    setattr(getattr(runtime, "_abstractcore_llm_client"), "_sender", sender)

    listed = facade.list_blocs(bloc_id=7, provider_api_key="sekret")
    artifacts = facade.list_bloc_kv_artifacts(bloc_id=7, provider_api_key="sekret")
    deleted_artifact = facade.delete_bloc_kv_artifact(
        bloc_id=7,
        artifact_path="/tmp/orbit.kv",
        clear_loaded=True,
        dry_run=True,
        provider_api_key="sekret",
    )
    pruned = facade.prune_bloc_kv_artifacts(
        bloc_id=7,
        provider="mlx",
        model="qwen3:4b",
        force=True,
        provider_api_key="sekret",
    )
    deleted_bloc = facade.delete_bloc(bloc_id=7, clear_loaded=True, dry_run=True, provider_api_key="sekret")

    assert listed["records"][0]["bloc_id"] == 7
    assert artifacts["artifacts"][0]["artifact_path"] == "/tmp/orbit.kv"
    assert deleted_artifact["result"]["deleted"] is True
    assert pruned["results"][0]["deleted"] is True
    assert deleted_bloc["result"]["dry_run"] is True
    assert sender.calls == [
        {
            "method": "get",
            "url": "http://core.test/acore/blocs?bloc_id=7",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "timeout": 45.0,
        },
        {
            "method": "get",
            "url": "http://core.test/acore/blocs/kv/list?bloc_id=7",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "timeout": 45.0,
        },
        {
            "method": "post",
            "url": "http://core.test/acore/blocs/kv/delete",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "bloc_id": 7,
                "artifact_path": "/tmp/orbit.kv",
                "clear_loaded": True,
                "force": False,
                "dry_run": True,
                "debug": False,
            },
            "timeout": 45.0,
        },
        {
            "method": "post",
            "url": "http://core.test/acore/blocs/kv/prune",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "bloc_id": 7,
                "provider": "mlx",
                "model": "qwen3:4b",
                "clear_loaded": False,
                "force": True,
                "dry_run": False,
                "debug": False,
            },
            "timeout": 45.0,
        },
        {
            "method": "post",
            "url": "http://core.test/acore/blocs/delete",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "bloc_id": 7,
                "delete_kv": True,
                "clear_loaded": True,
                "force": False,
                "dry_run": True,
            },
            "timeout": 45.0,
        },
    ]


def test_host_facade_bloc_kv_proxy_base_url_omits_local_runtime_selector() -> None:
    runtime = abstractcore.create_remote_runtime(
        server_base_url="http://core.test/v1",
        model="mlx/qwen3:4b",
        timeout_s=45,
    )
    facade = get_abstractcore_host_facade(runtime)
    sender = _RecordingRequestSender(
        get_responses=[
            {"ok": True, "operation": "kv_manifest", "manifest": {"binding_id": "bind-1"}},
        ],
        post_responses=[
            {"ok": True, "operation": "kv_ensure", "artifact": {"binding_id": "bind-1"}},
            {"ok": True, "operation": "kv_load", "artifact": {"key": "work:orbit", "binding_id": "bind-1"}},
        ],
    )
    setattr(getattr(runtime, "_abstractcore_llm_client"), "_sender", sender)

    manifest = facade.get_bloc_kv_manifest(
        bloc_id=7,
        base_url="http://provider.test/v1",
        provider_api_key="sekret",
    )
    ensured = facade.ensure_bloc_kv_artifact(
        bloc_id=7,
        base_url="http://provider.test/v1",
        provider_api_key="sekret",
    )
    loaded = facade.load_bloc_kv_artifact(
        bloc_id=7,
        base_url="http://provider.test/v1",
        key="work:orbit",
        provider_api_key="sekret",
    )

    assert manifest["manifest"]["binding_id"] == "bind-1"
    assert ensured["artifact"]["binding_id"] == "bind-1"
    assert loaded["artifact"]["key"] == "work:orbit"
    assert sender.calls == [
        {
            "method": "get",
            "url": "http://core.test/acore/blocs/kv/manifest?base_url=http%3A%2F%2Fprovider.test%2Fv1&bloc_id=7",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "timeout": 45.0,
        },
        {
            "method": "post",
            "url": "http://core.test/acore/blocs/kv/ensure",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "bloc_id": 7,
                "force_rebuild": False,
                "debug": False,
                "base_url": "http://provider.test/v1",
            },
            "timeout": 45.0,
        },
        {
            "method": "post",
            "url": "http://core.test/acore/blocs/kv/load",
            "headers": {"X-AbstractCore-Provider-API-Key": "sekret"},
            "json": {
                "bloc_id": 7,
                "key": "work:orbit",
                "make_default": False,
                "force_rebuild": False,
                "debug": False,
                "base_url": "http://provider.test/v1",
            },
            "timeout": 45.0,
        },
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
    assert facade.get_bloc_record(bloc_id=7) == {"ok": True, "operation": "record", "record": {"bloc_id": 7}}
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
