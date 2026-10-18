"""An output spec that names an `endpoint:<id>` provider keeps it.

The LLM_CALL handler resolves the call's OWN text provider (`_provider`) into
Core routing params. It used to fall back to the first output spec's provider
and rewrite that spec to the bare provider family, moving the endpoint's URL
and key into text-client kwargs the media path never reads: an image
generation routed to a gateway endpoint profile (the web console's sandbox,
the image route set to an OpenAI-compatible relay serving `gpt-image-2`)
reached AbstractVision as a keyless `openai-compatible` request with no base
URL and failed with "Missing vision_base_url / OPENAI_BASE_URL" -- in the
one-shot media subprocess, since the spec no longer said `endpoint:`.
Regression for the operator report of 2026-10-18.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.storage.artifacts import InMemoryArtifactStore

_PROFILE = {
    "id": "airelays",
    "virtual_provider": "endpoint:airelays",
    "display_name": "airelays",
    "provider": "openai-compatible",
    "base_url": "http://127.0.0.1:18921/v1",
    "api_key": "relay-secret-key",
    "scope": "gateway",
}


class _ProfileResolvingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.resolved: list[str] = []

    def resolve_provider_endpoint_profile(self, provider_id: str) -> dict[str, Any] | None:
        self.resolved.append(provider_id)
        return dict(_PROFILE) if provider_id == "endpoint:airelays" else None

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
        self.calls.append({"prompt": prompt, "media": media, "params": dict(params or {})})
        return {"content": "", "metadata": {}}


def _image_effect(*, provider: Optional[str] = None) -> Effect:
    payload: Dict[str, Any] = {
        "prompt": "a red circle",
        "output": {
            "modality": "image",
            "task": "image_generation",
            "format": "png",
            "provider": "endpoint:airelays",
            "model": "gpt-image-2",
        },
    }
    if provider:
        payload["provider"] = provider
    return Effect(type=EffectType.LLM_CALL, payload=payload, result_key="llm")


def _run() -> RunState:
    run = RunState.new(workflow_id="wf", entry_node="node-1", vars={})
    run.current_node = "node-1"
    return run


def test_media_spec_endpoint_provider_reaches_the_client_untouched() -> None:
    llm = _ProfileResolvingLLM()
    handler = make_llm_call_handler(llm=llm, artifact_store=InMemoryArtifactStore())

    outcome = handler(_run(), _image_effect(), None)

    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert params["output"]["provider"] == "endpoint:airelays"
    assert params["output"]["model"] == "gpt-image-2"
    # The endpoint is the SPEC's, not the call's text route: no text routing
    # params are minted from it, and nothing is resolved on its behalf.
    assert "_provider" not in params
    assert "base_url" not in params
    assert "provider_api_key" not in params
    assert "_provider_endpoint_profile" not in params
    assert llm.resolved == []


def test_text_endpoint_provider_is_resolved_while_the_media_spec_keeps_its_own() -> None:
    llm = _ProfileResolvingLLM()
    handler = make_llm_call_handler(llm=llm, artifact_store=InMemoryArtifactStore())

    outcome = handler(_run(), _image_effect(provider="endpoint:airelays"), None)

    assert outcome.status == "completed"
    params = llm.calls[-1]["params"]
    assert params["_provider"] == "openai-compatible"
    assert params["base_url"] == "http://127.0.0.1:18921/v1"
    assert params["provider_api_key"] == "relay-secret-key"
    assert params["_provider_endpoint_profile"]["virtual_provider"] == "endpoint:airelays"
    assert params["output"]["provider"] == "endpoint:airelays"
    assert llm.resolved == ["endpoint:airelays"]


def test_media_host_client_carries_the_endpoint_profile_resolver(monkeypatch) -> None:
    """The pooled client's media host serves every media-only call; the media
    plugins resolve `endpoint:<id>` through the resolver on THAT client."""
    from abstractruntime.integrations.abstractcore import llm_client as llm_client_mod

    created: list[Any] = []

    class FakeUnderlyingLLM:
        pass

    class FakeLocalClient:
        def __init__(self, *, provider, model, llm_kwargs=None, **kwargs):
            _ = provider, model, llm_kwargs, kwargs
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

    # Resolver set BEFORE the media host is built (the gateway's boot order).
    client = llm_client_mod.MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="local-model")
    client.set_provider_endpoint_profile_resolver(resolver)
    media_host = client._media_host_client()
    assert getattr(media_host, "resolve_provider_endpoint_profile") is resolver
    assert getattr(media_host._llm, "resolve_provider_endpoint_profile") is resolver

    # Resolver (re)set AFTER the media host exists (a profile refresh).
    later = lambda provider_id: {"provider": "openai", "id": provider_id}  # noqa: E731
    client.set_provider_endpoint_profile_resolver(later)
    assert client._media_host_client() is media_host
    assert getattr(media_host, "resolve_provider_endpoint_profile") is later
    assert getattr(media_host._llm, "resolve_provider_endpoint_profile") is later
