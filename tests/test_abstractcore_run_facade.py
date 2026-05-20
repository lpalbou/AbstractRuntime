from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractruntime import Runtime, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.integrations import abstractcore
from abstractruntime.integrations.abstractcore import AbstractCoreRunFacade, get_abstractcore_run_facade
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class _NoopTools:
    def execute(self, *, tool_calls):
        return {"mode": "executed", "results": []}


def _completed_parent_workflow() -> WorkflowSpec:
    def done(run, ctx):
        _ = (run, ctx)
        return StepPlan(node_id="done", complete_output={"ok": True})

    return WorkflowSpec("wf_parent", "done", {"done": done})


def test_public_run_facade_exports_are_available() -> None:
    assert abstractcore.AbstractCoreRunFacade is AbstractCoreRunFacade
    assert abstractcore.get_abstractcore_run_facade is get_abstractcore_run_facade
    assert "AbstractCoreRunFacade" in abstractcore.__all__
    assert "get_abstractcore_run_facade" in abstractcore.__all__


def test_run_facade_rejects_objects_without_runtime_contract() -> None:
    with pytest.raises(TypeError, match="Missing methods"):
        AbstractCoreRunFacade(object())


def test_run_facade_generate_image_creates_durable_child_run_with_truthful_media_metadata(monkeypatch) -> None:
    store = InMemoryArtifactStore()

    class _InProcessImageLLM:
        def _run_multimodal_spec(self, **_kwargs):
            raise AssertionError("image generation must stay on the subprocess-safe path")

    def fake_subprocess(**kwargs):
        assert kwargs["provider"] == "mlx"
        assert kwargs["model"] == "qwen-chat"
        return {
            "outputs": {
                "image": [
                    {
                        "modality": "image",
                        "task": "image_generation",
                        "data": b"png-child",
                        "content_type": "image/png",
                        "format": "png",
                        "provider": "mflux",
                        "model": "flux-dev",
                    }
                ]
            },
            "metadata": {
                "media_only": True,
                "subprocess": True,
                "runtime_provider": "mlx",
                "runtime_model": "qwen-chat",
                "execution_mode": "local_one_shot_subprocess",
            },
        }

    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.llm_client._run_local_image_subprocess",
        fake_subprocess,
    )

    llm = object.__new__(LocalAbstractCoreLLMClient)
    llm._provider = "mlx"
    llm._model = "qwen-chat"
    llm._llm_kwargs = {}
    llm._artifact_store = store
    llm._generate_lock = None
    llm._llm = _InProcessImageLLM()
    llm._maybe_prepare_prompt_cache = lambda **_kwargs: None

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=llm, tools=_NoopTools(), artifact_store=store),
    )
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(
        workflow=parent,
        session_id="sess-media",
        vars={"_runtime": {"prompt_cache": {"enabled": True}}},
    )
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.generate_image(
        parent_run_id,
        prompt="A red mug.",
        output={"provider": "mflux", "model": "flux-dev", "format": "png"},
    )

    assert child.status == RunStatus.COMPLETED
    assert child.parent_run_id == parent_run_id
    assert child.session_id == "sess-media"
    assert child.vars["_runtime"]["prompt_cache"]["enabled"] is True

    result = child.output["result"]
    assert result["runtime_provider"] == "mlx"
    assert result["runtime_model"] == "qwen-chat"
    assert result["media_provider"] == "mflux"
    assert result["media_model"] == "flux-dev"
    assert result["model"] == "flux-dev"

    item = result["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-child"

    ledger = runtime.get_ledger(child.run_id)
    effect_types = [
        effect.get("type")
        for entry in ledger
        if isinstance(entry, dict)
        for effect in [entry.get("effect")]
        if isinstance(effect, dict)
    ]
    assert "llm_call" in effect_types


def test_run_facade_transcribe_audio_inherits_runtime_media_defaults() -> None:
    store = InMemoryArtifactStore()
    meta = store.store(b"fake-wav", content_type="audio/wav", tags={"filename": "speech.wav"})
    seen: Dict[str, Any] = {}

    class _CapturingLLM:
        def generate(self, **kwargs):
            seen.update(kwargs)
            return {"content": "bonjour", "metadata": {}}

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=_CapturingLLM(), tools=_NoopTools(), artifact_store=store),
    )
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(
        workflow=parent,
        session_id="sess-audio",
        vars={"_runtime": {"stt_language": "fr"}},
    )
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.transcribe_audio(
        parent_run_id,
        media={"$artifact": meta.artifact_id, "filename": "speech.wav"},
    )

    assert child.status == RunStatus.COMPLETED
    assert child.parent_run_id == parent_run_id
    assert child.output["result"]["content"] == "bonjour"
    assert seen["params"]["stt_language"] == "fr"
    assert seen["params"]["output"]["task"] == "transcription"
    assert seen["media"][0]["artifact_id"] == meta.artifact_id
