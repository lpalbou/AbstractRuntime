"""Voice, music, image and transcription requests never build the text client (0.8.1).

The 0.7.0 end-to-end proof on a 128 GB Mac with the light install profile: the text default was
`mlx/Qwen3.8-Flash-Next-4bit` (MLX not installed) and EVERY text-to-speech request failed with
"MLX dependencies not installed": the pooled client built the default TEXT client first, then
ran the voice plugin on it. Now a media-only request runs on AbstractCore's capability host (no
text model); text requests still fail, attributed to the default route that cannot be built.

Also pinned: while the run facade executes a child run in-process it is `inline_run_active`,
so a host runner never ticks it (the gateway failed such children as "workflow not registered
(after 40 attempts)").
"""

from __future__ import annotations

import pytest

from abstractruntime import Runtime, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore import get_abstractcore_run_facade
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient
from abstractruntime.integrations.abstractcore.run_facade import inline_run_active
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


@pytest.fixture
def light_mac(monkeypatch):
    """`create_llm("mlx", ...)` fails like on the light profile; the voice plugin is faked."""
    import abstractcore
    from abstractcore.core import factory
    from abstractcore.core.multimodal_generation import GeneratedItem
    from abstractcore.providers.base import BaseProvider

    built = []
    seen = {}

    def create_llm(provider, model=None, **kwargs):
        built.append((provider, model))
        raise ImportError('MLX dependencies not installed. Install with: pip install "abstractcore[apple]"')

    monkeypatch.setattr(abstractcore, "create_llm", create_llm)
    monkeypatch.setattr(factory, "create_llm", create_llm)

    def fake_voice(self, *, result, spec, prompt, media, artifact_store):
        from abstractruntime.integrations.abstractcore import run_facade

        seen["provider"] = self.provider
        seen["inline_runs"] = set(run_facade._INLINE_RUNS)
        result.add_output("voice", GeneratedItem(modality="voice", task="tts", data=b"RIFF-fake-wav", content_type="audio/wav", format="wav"))

    monkeypatch.setattr(BaseProvider, "_run_voice_output", fake_voice)
    return built, seen


def _runtime(client, store):
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), artifact_store=store,
                 effect_handlers=build_effect_handlers(llm=client, artifact_store=store))
    parent = WorkflowSpec("wf_parent", "done", {"done": lambda run, ctx: StepPlan(node_id="done", complete_output={"ok": True})})
    rid = rt.start(workflow=parent, vars={})
    rt.tick(workflow=parent, run_id=rid)
    return rt, rid


def test_text_to_speech_never_builds_the_unbuildable_text_default(light_mac):
    built, seen = light_mac
    store = InMemoryArtifactStore()
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="mlx-community/Qwen3.8-Flash-Next-4bit", artifact_store=store)
    rt, rid = _runtime(client, store)
    built.clear()  # the start-up warm-up may try (and soft-fail); a TTS request must not
    child = get_abstractcore_run_facade(rt).generate_voice(rid, text="Hello there.")
    assert child.status == RunStatus.COMPLETED, child.error
    voice = child.output["result"]["outputs"]["voice"][0]
    assert voice["content_type"] == "audio/wav" and voice.get("artifact_id")
    assert built == []  # no text provider was built for the voice request
    assert seen["provider"] == "abstractcore-capabilities"
    # The child was registered as in-process while it ran, and is not anymore.
    assert child.run_id in seen["inline_runs"] and not inline_run_active(child.run_id)


def test_a_text_request_still_reports_the_default_route_it_cannot_build(light_mac):
    built, _seen = light_mac
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="mlx-community/Qwen3.8-Flash-Next-4bit")
    with pytest.raises(Exception) as exc:
        client.generate(prompt="Hi")
    assert "MLX dependencies not installed" in str(exc.value)
    assert ("mlx", "mlx-community/Qwen3.8-Flash-Next-4bit") in built


def test_the_capability_host_refuses_text_with_a_typed_error():
    from abstractcore.exceptions import InvalidRequestError
    from abstractcore.providers.capability_host import CapabilityHostProvider, TextGenerationUnavailable

    host = CapabilityHostProvider()
    with pytest.raises(TextGenerationUnavailable) as exc:
        host.generate("write a poem")
    assert isinstance(exc.value, InvalidRequestError) and "needs a text model" in str(exc.value)


def test_a_capability_defaults_change_rebuilds_the_media_host(monkeypatch):
    import abstractcore
    from types import SimpleNamespace
    from abstractcore.core import factory

    monkeypatch.setattr(abstractcore, "create_llm", lambda provider, model=None, **kw: SimpleNamespace(provider=provider, model=model))
    monkeypatch.setattr(factory, "create_llm", lambda provider, model=None, **kw: SimpleNamespace(provider=provider, model=model))
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="m", capability_defaults={})
    first = client._media_host_client()
    assert client._media_host_client() is first
    client.set_default_provider_model(provider="mlx", model="m", capability_defaults={
        "output.voice": {"provider": "supertonic", "model": "supertonic-3"}})
    assert client._media_host_client() is not first
