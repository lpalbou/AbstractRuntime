"""A media spec on an `endpoint:<id>` profile reaches that endpoint, end to end.

The LLM_CALL handler, the pooled client, AbstractCore's capability host and
the vision plugin's owner-resolver contract are real; only the vision backend
is a fake that records which endpoint it was given. Both production paths:
the media-only call without a text provider (the web console sandbox) and
the call that names its text provider (the Assistant orchestrator).
"""

from __future__ import annotations

import base64
import socket
import subprocess
from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient
from abstractruntime.storage.artifacts import InMemoryArtifactStore

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="
)


def _resolver(key: str):
    def resolve(provider_id: str) -> Optional[Dict[str, Any]]:
        if provider_id != "endpoint:relay":
            return None
        return {
            "id": "relay",
            "virtual_provider": "endpoint:relay",
            "provider": "openai-compatible",
            "base_url": "http://127.0.0.1:18921/v1",
            "api_key": key,
        }

    return resolve


@pytest.fixture
def pool(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an endpoint-profile media spec never opens a socket or spawns the media subprocess")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setenv("ABSTRACTRUNTIME_LOCAL_IMAGE_SUBPROCESS", "1")
    from abstractcore.capabilities.registry import CapabilityRegistry
    from abstractcore.config import manager

    config_file = str(tmp_path / "core.json")
    monkeypatch.setattr(manager, "_config_manager", manager.ConfigurationManager(config_file=config_file, apply_env=False))
    calls: List[tuple[str, str, str]] = []

    class RecordingVision:
        backend_id = "test-recording-vision"

        def __init__(self, owner: Any) -> None:
            self.owner = owner

        def t2i(self, prompt: str, **kwargs: Any) -> bytes:
            # The vision plugin's contract: `endpoint:<id>` resolves through the
            # resolver on the OWNING Core provider (the media host's), never
            # through the Core config file (a gateway profile has no row there).
            resolved = self.owner.resolve_provider_endpoint_profile(kwargs["provider"])
            calls.append((kwargs["model"], resolved["base_url"], resolved["api_key"]))
            return PNG

    monkeypatch.setattr(
        CapabilityRegistry,
        "_ensure_plugins_loaded",
        lambda registry: registry.register_vision_backend(
            backend_id=RecordingVision.backend_id, factory=RecordingVision, priority=1000
        ),
    )
    store = InMemoryArtifactStore()
    client = MultiLocalAbstractCoreLLMClient(
        provider="lmstudio", model="local-model", artifact_store=store, core_config_file=config_file
    )
    return client, store, calls


def _image_effect(*, provider: Optional[str] = None) -> Effect:
    payload: Dict[str, Any] = {
        "prompt": "a red circle",
        "output": {
            "modality": "image",
            "task": "image_generation",
            "format": "png",
            "provider": "endpoint:relay",
            "model": "gpt-image-2",
        },
    }
    if provider:
        payload["provider"] = provider
        payload["model"] = "local-model"
    return Effect(type=EffectType.LLM_CALL, payload=payload, result_key="llm")


def _run() -> RunState:
    run = RunState.new(workflow_id="wf", entry_node="node-1", vars={})
    run.current_node = "node-1"
    return run


@pytest.mark.parametrize("call_provider", [None, "lmstudio"], ids=["sandbox-media-only", "orchestrator-text-provider"])
def test_llm_call_image_on_an_endpoint_profile_reaches_it(pool, call_provider) -> None:
    client, store, calls = pool
    client.set_provider_endpoint_profile_resolver(_resolver("relay-key"))
    handler = make_llm_call_handler(llm=client, artifact_store=store)

    outcome = handler(_run(), _image_effect(provider=call_provider), None)

    assert outcome.status == "completed", outcome.error
    assert calls == [("gpt-image-2", "http://127.0.0.1:18921/v1", "relay-key")]
    image = outcome.result["outputs"]["image"][0]
    assert store.load(image["artifact_ref"]["$artifact"]).content == PNG
    assert "relay-key" not in str(outcome.result)


def test_resolver_set_after_the_media_host_exists_and_kept_across_rebuilds(pool) -> None:
    client, _store, calls = pool
    media_host = client._media_host_client()
    client.set_provider_endpoint_profile_resolver(_resolver("first-key"))
    handler = make_llm_call_handler(llm=client, artifact_store=_store)

    assert handler(_run(), _image_effect(), None).status == "completed"
    assert calls[-1] == ("gpt-image-2", "http://127.0.0.1:18921/v1", "first-key")
    assert client._media_host_client() is media_host

    # A profile refresh rotates the resolver on the host that already exists.
    client.set_provider_endpoint_profile_resolver(_resolver("rotated-key"))
    assert handler(_run(), _image_effect(), None).status == "completed"
    assert calls[-1][2] == "rotated-key"

    # Changing the capability routes rebuilds the media host; it is wired like any pooled client.
    assert client.set_capability_defaults({"output.image": {"provider": "endpoint:relay", "model": "gpt-image-2"}})
    assert client._media_host_client() is not media_host
    assert handler(_run(), _image_effect(), None).status == "completed"
    assert calls[-1][2] == "rotated-key"


def test_pool_wide_settings_reach_every_owned_client(pool) -> None:
    client, _store, _calls = pool
    media_host = client._media_host_client()
    assert media_host in client._owned_clients()

    on_token = lambda delta, meta: None  # noqa: E731
    client.set_on_token(on_token)
    assert media_host._on_token is on_token

    resolver = _resolver("key")
    client.set_provider_endpoint_profile_resolver(resolver)
    assert media_host.resolve_provider_endpoint_profile is resolver
    assert media_host._llm.resolve_provider_endpoint_profile is resolver
