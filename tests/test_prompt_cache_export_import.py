from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import abstractcore
from abstractcore.core import factory as ac_factory

from abstractruntime.integrations import abstractcore as runtime_abstractcore
from abstractruntime.integrations.abstractcore import factory as runtime_abstractcore_factory
from abstractruntime.integrations.abstractcore.host_facade import get_abstractcore_host_facade
from abstractruntime.integrations.abstractcore.llm_client import (
    MultiLocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
)


class _FakePromptCacheProvider:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.saved: List[Dict[str, Any]] = []
        self.loaded: List[Dict[str, Any]] = []
        self.cleared: List[Any] = []

    def get_prompt_cache_capabilities(self) -> Dict[str, Any]:
        return {
            "supported": True,
            "mode": "local_control_plane",
            "supports_set": True,
            "supports_clear": True,
            "supports_update": True,
            "supports_fork": True,
            "supports_prepare_modules": True,
            "supports_stats": True,
            "supports_save": True,
            "supports_load": True,
        }

    def prompt_cache_artifact_extension(self) -> str:
        if self.provider == "huggingface":
            return ".npz"
        return ".safetensors"

    def prompt_cache_artifact_format(self) -> str:
        if self.provider == "huggingface":
            return "abstractcore-gguf-prompt-cache/v1"
        return "abstractcore-prompt-cache/v1"

    def prompt_cache_save(self, key: str, filename: str, **kwargs: Any) -> Dict[str, Any]:
        meta = dict(kwargs.get("meta") or {})
        meta.setdefault("saved_at", "2026-05-21T10:00:00Z")
        meta.setdefault("token_count", 42)
        meta.setdefault("provider", self.provider)
        meta.setdefault("model", self.model)
        Path(filename).write_text(f"{self.provider}:{self.model}:{key}", encoding="utf-8")
        payload = {
            "supported": True,
            "operation": "save",
            "provider": self.provider,
            "model": self.model,
            "key": key,
            "filename": filename,
            "meta": meta,
        }
        self.saved.append(payload)
        return payload

    def prompt_cache_load(
        self,
        filename: str,
        *,
        key: str | None = None,
        make_default: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        _ = kwargs
        payload = {
            "supported": True,
            "operation": "load",
            "provider": self.provider,
            "model": self.model,
            "key": key or "loaded:auto",
            "make_default": bool(make_default),
            "filename": filename,
            "content": Path(filename).read_text(encoding="utf-8"),
        }
        self.loaded.append(payload)
        return payload

    def prompt_cache_clear(self, key: str | None = None) -> bool:
        self.cleared.append(key)
        return True


class _UnsupportedPromptCacheProvider:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    def get_prompt_cache_capabilities(self) -> Dict[str, Any]:
        return {
            "supported": True,
            "mode": "keyed",
            "supports_set": True,
            "supports_clear": True,
            "supports_update": False,
            "supports_fork": False,
            "supports_prepare_modules": False,
            "supports_stats": True,
            "supports_save": False,
            "supports_load": False,
        }


def test_multilocal_prompt_cache_export_list_and_import_roundtrip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    created: Dict[Tuple[str, str], _FakePromptCacheProvider] = {}

    def _create_llm(provider: str, *, model: str, **kwargs: Any) -> Any:
        _ = kwargs
        if provider == "openai":
            return _UnsupportedPromptCacheProvider(provider=provider, model=model)
        fake = _FakePromptCacheProvider(provider=provider, model=model)
        created[(provider, model)] = fake
        return fake

    monkeypatch.setattr(abstractcore, "create_llm", _create_llm, raising=True)
    monkeypatch.setattr(ac_factory, "create_llm", _create_llm, raising=True)

    client = MultiLocalAbstractCoreLLMClient(
        provider="mlx",
        model="mlx-community/Qwen3-4B-4bit",
        prompt_cache_export_root_dir=tmp_path,
    )

    exported_mlx = client.prompt_cache_export(name="orbit cache", key="sess:orbit", q8=True)
    exported_hf = client.prompt_cache_export(
        name="orbit cache",
        key="sess:orbit",
        provider="huggingface",
        model="bartowski/Qwen3-4B-GGUF",
    )
    exported_hf_collision = client.prompt_cache_export(
        name="orbit cache",
        key="sess:orbit",
        provider="huggingface",
        model="bartowski-Qwen3-4B-GGUF",
    )

    listed_mlx = client.list_prompt_cache_exports()
    listed_hf = client.list_prompt_cache_exports(provider="huggingface", model="bartowski/Qwen3-4B-GGUF")
    imported_mlx = client.prompt_cache_import(
        name="orbit-cache",
        clear_existing=True,
    )
    unsupported_export = client.prompt_cache_export(
        name="unsupported",
        key="sess:bad",
        provider="openai",
        model="gpt-4o-mini",
    )
    unsupported_import = client.prompt_cache_import(
        name="unsupported",
        provider="openai",
        model="gpt-4o-mini",
    )

    assert exported_mlx["supported"] is True
    assert exported_mlx["name"] == "orbit-cache"
    assert exported_mlx["artifact_filename"].endswith(".safetensors")
    assert Path(exported_mlx["artifact_path"]).exists()
    assert Path(exported_mlx["meta_path"]).exists()
    assert exported_hf["artifact_filename"].endswith(".npz")
    assert exported_hf["artifact_path"] != exported_mlx["artifact_path"]
    assert exported_hf_collision["artifact_path"] != exported_hf["artifact_path"]

    assert listed_mlx["supported"] is True
    assert len(listed_mlx["items"]) == 1
    assert listed_mlx["items"][0]["name"] == "orbit-cache"
    assert listed_mlx["items"][0]["token_count"] == 42
    assert listed_mlx["items"][0]["artifact_extension"] == ".safetensors"

    assert listed_hf["supported"] is True
    assert len(listed_hf["items"]) == 1
    assert listed_hf["items"][0]["artifact_extension"] == ".npz"
    assert listed_hf["items"][0]["artifact_path"] == exported_hf["artifact_path"]

    assert imported_mlx["supported"] is True
    assert imported_mlx["name"] == "orbit-cache"
    assert imported_mlx["provider_response"]["key"] == "loaded:auto"
    assert imported_mlx["key"] == "loaded:auto"
    assert unsupported_export["supported"] is False
    assert unsupported_export["code"] == "prompt_cache_unsupported"
    assert unsupported_import["supported"] is False
    assert unsupported_import["code"] == "prompt_cache_unsupported"

    mlx_provider = created[("mlx", "mlx-community/Qwen3-4B-4bit")]
    hf_provider = created[("huggingface", "bartowski/Qwen3-4B-GGUF")]

    assert mlx_provider.cleared == [None]
    assert len(mlx_provider.loaded) == 2
    assert mlx_provider.loaded[-1]["filename"] == exported_mlx["artifact_path"]
    assert "mlx:mlx-community/Qwen3-4B-4bit:sess:orbit" in mlx_provider.loaded[-1]["content"]
    assert hf_provider.saved[0]["filename"] == exported_hf["artifact_path"]

    mlx_meta = json.loads(Path(exported_mlx["meta_path"]).read_text(encoding="utf-8"))
    hf_meta = json.loads(Path(exported_hf["meta_path"]).read_text(encoding="utf-8"))
    assert mlx_meta["schema"] == "abstractruntime-prompt-cache-export/v1"
    assert mlx_meta["provider"] == "mlx"
    assert mlx_meta["artifact_extension"] == ".safetensors"
    assert hf_meta["provider"] == "huggingface"
    assert hf_meta["artifact_extension"] == ".npz"


def test_multilocal_prompt_cache_export_listing_does_not_require_loading_target_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _create_llm(provider: str, *, model: str, **kwargs: Any) -> Any:
        _ = kwargs
        if provider == "huggingface" and model == "lazy/model":
            raise AssertionError("listing should not instantiate the target provider/model client")
        return _FakePromptCacheProvider(provider=provider, model=model)

    monkeypatch.setattr(abstractcore, "create_llm", _create_llm, raising=True)
    monkeypatch.setattr(ac_factory, "create_llm", _create_llm, raising=True)

    export_dir = tmp_path / quote("huggingface", safe="") / quote("lazy/model", safe="")
    export_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = export_dir / "lazy-cache.npz"
    artifact_path.write_text("lazy", encoding="utf-8")
    meta_path = export_dir / "lazy-cache.npz.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "schema": "abstractruntime-prompt-cache-export/v1",
                "name": "lazy-cache",
                "provider": "huggingface",
                "model": "lazy/model",
                "saved_at": "2026-05-21T11:00:00Z",
                "artifact_filename": artifact_path.name,
                "artifact_extension": ".npz",
                "artifact_format": "abstractcore-gguf-prompt-cache/v1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    client = MultiLocalAbstractCoreLLMClient(
        provider="mlx",
        model="mlx-community/Qwen3-4B-4bit",
        prompt_cache_export_root_dir=tmp_path,
    )

    listed = client.list_prompt_cache_exports(provider="huggingface", model="lazy/model")

    assert listed["supported"] is True
    assert len(listed["items"]) == 1
    assert listed["items"][0]["name"] == "lazy-cache"
    assert listed["items"][0]["artifact_path"] == str(artifact_path)
    assert listed["capabilities"]["supported"] is False


def test_host_facade_can_export_list_and_import_through_a_real_local_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    created: Dict[Tuple[str, str], _FakePromptCacheProvider] = {}

    def _create_llm(provider: str, *, model: str, **kwargs: Any) -> _FakePromptCacheProvider:
        _ = kwargs
        fake = _FakePromptCacheProvider(provider=provider, model=model)
        created[(provider, model)] = fake
        return fake

    monkeypatch.setattr(abstractcore, "create_llm", _create_llm, raising=True)
    monkeypatch.setattr(ac_factory, "create_llm", _create_llm, raising=True)
    monkeypatch.setattr(runtime_abstractcore_factory, "build_effect_handlers", lambda **kwargs: {})
    monkeypatch.setattr(runtime_abstractcore_factory, "AbstractCoreChatSummarizer", lambda **kwargs: object())
    monkeypatch.setattr(
        runtime_abstractcore_factory,
        "AbstractCoreToolExecutor",
        lambda timeout_s: SimpleNamespace(set_timeout_s=lambda value: None, timeout_s=timeout_s),
    )

    runtime = runtime_abstractcore.create_local_file_runtime(
        base_dir=tmp_path / "runtime",
        provider="mlx",
        model="mlx-community/Qwen3-4B-4bit",
    )
    facade = get_abstractcore_host_facade(runtime)

    exported = facade.prompt_cache_export(name="host-facade", key="sess:host")
    listed = facade.list_prompt_cache_exports()
    imported = facade.prompt_cache_import(name="host-facade", key="loaded:host")

    assert exported["supported"] is True
    assert listed["items"][0]["name"] == "host-facade"
    assert imported["supported"] is True
    assert imported["key"] == "loaded:host"


class _ExplodingSender:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    def get(self, url: str, *, headers: Dict[str, str], timeout: float) -> Any:
        self.calls.append(("GET", url, headers, timeout))
        raise AssertionError("remote export/import helpers must not call HTTP")

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Any:
        self.calls.append(("POST", url, headers, json, timeout))
        raise AssertionError("remote export/import helpers must not call HTTP")


def test_remote_prompt_cache_export_import_surface_is_local_only() -> None:
    sender = _ExplodingSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://localhost:8080",
        model="openai-compatible/default",
        request_sender=sender,
    )

    listed = client.list_prompt_cache_exports(provider="mlx", model="qwen")
    exported = client.prompt_cache_export(name="orbit-cache", key="sess:orbit")
    imported = client.prompt_cache_import(name="orbit-cache", key="loaded:orbit", clear_existing=True)

    assert listed["supported"] is False
    assert listed["code"] == "prompt_cache_local_only"
    assert exported["supported"] is False
    assert exported["code"] == "prompt_cache_local_only"
    assert imported["supported"] is False
    assert imported["code"] == "prompt_cache_local_only"
    assert sender.calls == []
