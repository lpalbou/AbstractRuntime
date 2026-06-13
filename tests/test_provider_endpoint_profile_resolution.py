from __future__ import annotations

from typing import Any, Dict, List, Optional

from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler


class _ProfileResolvingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def resolve_provider_endpoint_profile(self, provider_id: str) -> dict[str, Any] | None:
        if provider_id != "endpoint:office-vllm":
            return None
        return {
            "id": "office-vllm",
            "virtual_provider": "endpoint:office-vllm",
            "display_name": "Office vLLM",
            "provider": "openai-compatible",
            "base_url": "https://llm.example.test/v1",
            "api_key": "secret-key",
            "scope": "gateway",
        }

    def generate(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "media": media,
                "params": dict(params or {}),
            }
        )
        return {"content": "ok", "metadata": {}}


def test_llm_call_resolves_gateway_endpoint_profile_and_redacts_key_from_observability() -> None:
    llm = _ProfileResolvingLLM()
    handler = make_llm_call_handler(llm=llm)
    run = RunState.new(workflow_id="wf", entry_node="node-1", vars={})
    run.current_node = "node-1"
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "provider": "endpoint:office-vllm",
            "model": "qwen/qwen3",
            "params": {"temperature": 0.2, "max_output_tokens": 2048, "max_input_tokens": 8192},
        },
        result_key="llm",
    )

    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    assert llm.calls
    params = llm.calls[-1]["params"]
    assert params["_provider"] == "openai-compatible"
    assert params["_model"] == "qwen/qwen3"
    assert params["base_url"] == "https://llm.example.test/v1"
    assert params["provider_api_key"] == "secret-key"
    assert params["_provider_endpoint_profile"] == {
        "id": "office-vllm",
        "virtual_provider": "endpoint:office-vllm",
        "display_name": "Office vLLM",
        "provider_family": "openai-compatible",
        "scope": "gateway",
        "base_url_configured": True,
        "api_key_set": True,
    }

    result = outcome.result
    metadata = result.get("metadata") if isinstance(result, dict) else None
    assert isinstance(metadata, dict)
    runtime_obs = metadata.get("_runtime_observability")
    assert isinstance(runtime_obs, dict)
    observed_params = runtime_obs["llm_generate_kwargs"]["params"]
    assert observed_params["provider_api_key"] == "[redacted]"
    assert observed_params["max_output_tokens"] == 2048
    assert observed_params["max_input_tokens"] == 8192
    assert "secret-key" not in str(runtime_obs)


def test_multilocal_client_uses_endpoint_profile_url_and_key_for_local_provider_construction(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import llm_client as llm_client_mod

    created: list[dict[str, Any]] = []

    class FakeLocalClient:
        def __init__(self, *, provider, model, llm_kwargs=None, artifact_store=None, bloc_root_dir=None, prompt_cache_export_root_dir=None):
            created.append({"provider": provider, "model": model, "llm_kwargs": dict(llm_kwargs or {})})
            self._llm = object()

        def generate(self, *, prompt, messages=None, system_prompt=None, tools=None, media=None, params=None):
            return {"content": "ok", "metadata": {"params": dict(params or {})}}

        def get_model_capabilities(self, model_name=None):
            return {"max_tokens": 8192}

    monkeypatch.setattr(llm_client_mod, "LocalAbstractCoreLLMClient", FakeLocalClient)

    client = llm_client_mod.MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="local-model")
    result = client.generate(
        prompt="hello",
        params={
            "_provider": "openai-compatible",
            "_model": "remote-model",
            "base_url": "https://llm.example.test/v1",
            "provider_api_key": "secret-key",
            "temperature": 0.2,
        },
    )

    assert result["content"] == "ok"
    assert created[-1] == {
        "provider": "openai-compatible",
        "model": "remote-model",
        "llm_kwargs": {"base_url": "https://llm.example.test/v1", "api_key": "secret-key"},
    }
    assert result["metadata"]["params"] == {"temperature": 0.2}


def test_multilocal_client_propagates_endpoint_profile_resolver_to_cached_clients(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import llm_client as llm_client_mod

    created: list[Any] = []

    class FakeUnderlyingLLM:
        pass

    class FakeLocalClient:
        def __init__(self, *, provider, model, llm_kwargs=None, artifact_store=None, bloc_root_dir=None, prompt_cache_export_root_dir=None):
            _ = provider, model, llm_kwargs, artifact_store, bloc_root_dir, prompt_cache_export_root_dir
            self._llm = FakeUnderlyingLLM()
            created.append(self)

        def generate(self, *, prompt, messages=None, system_prompt=None, tools=None, media=None, params=None):
            _ = prompt, messages, system_prompt, tools, media, params
            return {"content": "ok"}

        def get_model_capabilities(self, model_name=None):
            _ = model_name
            return {"max_tokens": 8192}

    monkeypatch.setattr(llm_client_mod, "LocalAbstractCoreLLMClient", FakeLocalClient)

    resolver = lambda provider_id: {"provider": "openai-compatible", "id": provider_id}  # noqa: E731
    client = llm_client_mod.MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="local-model")

    client.set_provider_endpoint_profile_resolver(resolver)

    assert created
    default_client = created[0]
    assert getattr(default_client, "resolve_provider_endpoint_profile") is resolver
    assert getattr(default_client._llm, "resolve_provider_endpoint_profile") is resolver

    routed_client = client._get_client("openai-compatible", "remote-model")
    assert getattr(routed_client, "resolve_provider_endpoint_profile") is resolver
    assert getattr(routed_client._llm, "resolve_provider_endpoint_profile") is resolver


def test_multilocal_client_attaches_scoped_core_capability_defaults_to_clients(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import llm_client as llm_client_mod

    created: list[Any] = []

    class FakeUnderlyingLLM:
        pass

    class FakeLocalClient:
        def __init__(self, *, provider, model, llm_kwargs=None, artifact_store=None):
            _ = provider, model, llm_kwargs, artifact_store
            self._llm = FakeUnderlyingLLM()
            created.append(self)

        def generate(self, *, prompt, messages=None, system_prompt=None, tools=None, media=None, params=None):
            _ = prompt, messages, system_prompt, tools, media, params
            return {"content": "ok"}

        def get_model_capabilities(self, model_name=None):
            _ = model_name
            return {"max_tokens": 8192}

    monkeypatch.setattr(llm_client_mod, "LocalAbstractCoreLLMClient", FakeLocalClient)

    defaults = {
        "routes": [
            {
                "key": "input.voice",
                "provider": "faster-whisper",
                "model": "large-v3",
                "source": "abstractcore.runtime",
            }
        ]
    }
    client = llm_client_mod.MultiLocalAbstractCoreLLMClient(
        provider="lmstudio",
        model="local-model",
        core_config_file="/tmp/runtime/config/abstractcore.json",
        capability_defaults=defaults,
    )

    default_client = created[0]
    assert getattr(default_client, "_abstractcore_config_file") == "/tmp/runtime/config/abstractcore.json"
    assert getattr(default_client._llm, "_abstractcore_config_file") == "/tmp/runtime/config/abstractcore.json"
    assert getattr(default_client, "_abstractcore_capability_defaults")["input.voice"]["model"] == "large-v3"
    assert getattr(default_client._llm, "_abstractcore_capability_defaults")["input.voice"]["provider"] == "faster-whisper"

    routed_client = client._get_client("openai-compatible", "remote-model")
    assert getattr(routed_client._llm, "_abstractcore_capability_defaults")["input.voice"]["model"] == "large-v3"


def test_runtime_applies_task_specific_media_capability_defaults() -> None:
    from abstractruntime.integrations.abstractcore import llm_client as llm_client_mod

    defaults = {
        "output.image.text_to_image": {
            "provider": "mlx-gen",
            "model": "AbstractFramework/z-image-turbo-8bit",
        },
        "output.image.image_to_image": {
            "provider": "mlx-gen",
            "model": "AbstractFramework/qwen-image-edit-2511-4bit",
        },
        "output.image.image_upscale": {
            "provider": "mlx-gen",
            "model": "AbstractFramework/seedvr2-3b-8bit",
            "options": {"resolution": "2x", "softness": 0.25},
        },
        "output.video.text_to_video": {
            "provider": "mlx-gen",
            "model": "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
        },
        "output.video.image_to_video": {
            "provider": "mlx-gen",
            "model": "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
        },
    }
    normalized = llm_client_mod._normalize_core_capability_defaults(defaults)

    assert llm_client_mod._with_capability_default_route(
        {"modality": "image", "task": "image_generation"},
        normalized,
    )["model"] == "AbstractFramework/z-image-turbo-8bit"
    assert llm_client_mod._with_capability_default_route(
        {"modality": "image", "task": "image_edit"},
        normalized,
    )["model"] == "AbstractFramework/qwen-image-edit-2511-4bit"
    upscale = llm_client_mod._with_capability_default_route(
        {"modality": "image", "task": "image_upscale"},
        normalized,
    )
    assert upscale["model"] == "AbstractFramework/seedvr2-3b-8bit"
    assert upscale["resolution"] == "2x"
    assert upscale["softness"] == 0.25
    assert llm_client_mod._with_capability_default_route(
        {"modality": "video", "task": "text_to_video"},
        normalized,
    )["model"] == "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
    assert llm_client_mod._with_capability_default_route(
        {"modality": "video", "task": "image_to_video"},
        normalized,
    )["model"] == "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"

    explicit = llm_client_mod._with_capability_default_route(
        {"modality": "image", "task": "image_upscale", "provider": "custom", "model": "explicit"},
        normalized,
    )
    assert explicit["provider"] == "custom"
    assert explicit["model"] == "explicit"
    assert "resolution" not in explicit
