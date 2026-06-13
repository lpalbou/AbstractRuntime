from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractruntime import Runtime, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.integrations import abstractcore
from abstractruntime.integrations.abstractcore import AbstractCoreRunFacade, get_abstractcore_run_facade
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    PassthroughToolExecutor,
    ToolApprovalPolicy,
)
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
                        "metadata": {
                            "token": "secret-token",
                            "nested": {"access_token": "secret-access", "safe": "kept"},
                        },
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
    descriptor = artifact.metadata.descriptor
    assert descriptor.semantic_kind == "image"
    assert descriptor.render_kind == "image"
    assert descriptor.modality == "image"
    assert descriptor.task == "image_generation"
    assert descriptor.classification_source == "producer"
    assert descriptor.session_id == "sess-media"
    assert descriptor.workflow_id == "wf_abstractcore_run_facade_image_generation"
    assert descriptor.producer["provider"] == "mflux"
    assert descriptor.producer["model"] == "flux-dev"
    assert descriptor.producer["runtime_provider"] == "mlx"
    assert descriptor.generation["prompt"] == "A red mug."
    assert descriptor.generation["requested_format"] == "png"
    assert descriptor.provenance["run_id"] == child.run_id
    assert descriptor.security["redaction"] == "bounded_secret_key_redaction_v1"
    assert descriptor.security["sensitivity"] == "user_content"
    assert artifact.metadata.metadata["schema"] == "abstractruntime.generated_media_metadata.v1"
    assert artifact.metadata.metadata["security"]["recorded_user_content_fields"] == ["prompt"]
    assert artifact.metadata.metadata["capability_metadata"]["token"] == "[redacted]"
    assert artifact.metadata.metadata["capability_metadata"]["nested"]["access_token"] == "[redacted]"
    assert artifact.metadata.metadata["capability_metadata"]["nested"]["safe"] == "kept"

    ledger = runtime.get_ledger(child.run_id)
    effect_types = [
        effect.get("type")
        for entry in ledger
        if isinstance(entry, dict)
        for effect in [entry.get("effect")]
        if isinstance(effect, dict)
    ]
    assert "llm_call" in effect_types


def test_run_facade_edit_image_creates_durable_child_run_with_media() -> None:
    store = InMemoryArtifactStore()
    source = store.store(b"png-source", content_type="image/png", tags={"filename": "source.png"})
    seen: Dict[str, Any] = {}

    class _ImageEditProvider:
        def generate(self, **kwargs):
            from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

            seen.update(kwargs)
            return MultimodalGenerateResponse(
                outputs={
                    "image": [
                        GeneratedItem(
                            modality="image",
                            task="image_edit",
                            data=b"png-edited",
                            content_type="image/png",
                            format="png",
                            provider="openai-compatible",
                            model="gpt-image-1",
                        )
                    ]
                }
            )

    llm = object.__new__(LocalAbstractCoreLLMClient)
    llm._provider = "mlx"
    llm._model = "qwen-chat"
    llm._artifact_store = store
    llm._generate_lock = None
    llm._llm = _ImageEditProvider()
    llm._maybe_prepare_prompt_cache = lambda **_kwargs: None

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=llm, tools=_NoopTools(), artifact_store=store),
    )
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(workflow=parent, session_id="sess-edit")
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.edit_image(
        parent_run_id,
        prompt="Make the jacket red.",
        media={"$artifact": source.artifact_id, "filename": "source.png"},
        output={"provider": "openai-compatible", "model": "gpt-image-1", "format": "png"},
    )

    assert child.status == RunStatus.COMPLETED
    assert seen["prompt"] == "Make the jacket red."
    assert seen["output"]["task"] == "image_edit"
    assert seen["media"][0]["artifact_id"] == source.artifact_id
    result = child.output["result"]
    item = result["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-edited"
    assert artifact.metadata.run_id == child.run_id
    descriptor = artifact.metadata.descriptor
    assert descriptor.semantic_kind == "image"
    assert descriptor.task == "image_edit"
    assert descriptor.generation["prompt"] == "Make the jacket red."
    assert descriptor.source_refs
    source_ref = descriptor.source_refs[0]
    assert source_ref["kind"] == "artifact"
    assert source_ref["artifact_id"] == source.artifact_id
    assert source_ref["content_type"] == "image/png"
    assert source_ref["modality"] == "image"
    assert source_ref["filename"] == "source.png"


def test_run_facade_upscale_image_creates_durable_child_run_with_media() -> None:
    store = InMemoryArtifactStore()
    source = store.store(b"png-source", content_type="image/png", tags={"filename": "source.png"})
    seen: Dict[str, Any] = {}

    class _ImageUpscaleProvider:
        def generate(self, **kwargs):
            from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

            seen.update(kwargs)
            return MultimodalGenerateResponse(
                outputs={
                    "image": [
                        GeneratedItem(
                            modality="image",
                            task="image_upscale",
                            data=b"png-upscaled",
                            content_type="image/png",
                            format="png",
                            provider="mlx-gen",
                            model="AbstractFramework/seedvr2-3b-8bit",
                        )
                    ]
                }
            )

    llm = object.__new__(LocalAbstractCoreLLMClient)
    llm._provider = "mlx"
    llm._model = "qwen-chat"
    llm._artifact_store = store
    llm._generate_lock = None
    llm._llm = _ImageUpscaleProvider()
    llm._maybe_prepare_prompt_cache = lambda **_kwargs: None

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=llm, tools=_NoopTools(), artifact_store=store),
    )
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(workflow=parent, session_id="sess-upscale")
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.upscale_image(
        parent_run_id,
        media={"$artifact": source.artifact_id, "filename": "source.png"},
        output={
            "provider": "mlx-gen",
            "model": "AbstractFramework/seedvr2-3b-8bit",
            "format": "png",
            "scale": "2x",
            "softness": 0.25,
            "quantize": 8,
        },
    )

    assert child.status == RunStatus.COMPLETED
    assert seen["output"]["task"] == "image_upscale"
    assert seen["output"]["scale"] == "2x"
    assert seen["output"]["softness"] == 0.25
    assert seen["output"]["quantize"] == 8
    assert seen["media"][0]["artifact_id"] == source.artifact_id
    item = child.output["result"]["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-upscaled"
    assert artifact.metadata.run_id == child.run_id
    assert artifact.metadata.descriptor.task == "image_upscale"


def test_run_facade_generate_video_creates_durable_child_run_with_progress() -> None:
    store = InMemoryArtifactStore()
    ledger = InMemoryLedgerStore()
    seen: Dict[str, Any] = {}

    class _VideoProvider:
        def generate(self, **kwargs):
            from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

            seen.update(kwargs)
            callback = kwargs.get("on_progress")
            assert callable(callback)
            callback({"phase": "frames", "frame": 2, "total_frames": 4, "progress": 0.5})
            return MultimodalGenerateResponse(
                outputs={
                    "video": [
                        GeneratedItem(
                            modality="video",
                            task="text_to_video",
                            data=b"mp4-child",
                            content_type="video/mp4",
                            format="mp4",
                            provider="mlx-gen",
                            model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                        )
                    ]
                }
            )

    llm = object.__new__(LocalAbstractCoreLLMClient)
    llm._provider = "mlx"
    llm._model = "qwen-chat"
    llm._artifact_store = store
    llm._generate_lock = None
    llm._llm = _VideoProvider()
    llm._maybe_prepare_prompt_cache = lambda **_kwargs: None

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=llm, tools=_NoopTools(), artifact_store=store),
    )
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(workflow=parent, session_id="sess-video")
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.generate_video(
        parent_run_id,
        prompt="A logo reveal.",
        output={
            "provider": "mlx-gen",
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "format": "mp4",
            "num_frames": 4,
        },
    )

    assert child.status == RunStatus.COMPLETED
    assert seen["output"]["modality"] == "video"
    assert seen["output"]["task"] == "text_to_video"
    assert seen["output"]["provider"] == "mlx-gen"
    result = child.output["result"]
    item = result["outputs"]["video"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"mp4-child"
    progress = [
        r for r in ledger.list(child.run_id)
        if ((r.get("effect") or {}).get("payload") or {}).get("name") == "abstract.progress"
    ]
    assert progress


def test_run_facade_image_to_video_creates_durable_child_run_with_media() -> None:
    store = InMemoryArtifactStore()
    source = store.store(b"png-source", content_type="image/png", tags={"filename": "source.png"})
    seen: Dict[str, Any] = {}

    class _ImageToVideoProvider:
        def generate(self, **kwargs):
            from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

            seen.update(kwargs)
            return MultimodalGenerateResponse(
                outputs={
                    "video": [
                        GeneratedItem(
                            modality="video",
                            task="image_to_video",
                            data=b"mp4-i2v",
                            content_type="video/mp4",
                            format="mp4",
                            provider="mlx-gen",
                            model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                        )
                    ]
                }
            )

    llm = object.__new__(LocalAbstractCoreLLMClient)
    llm._provider = "mlx"
    llm._model = "qwen-chat"
    llm._artifact_store = store
    llm._generate_lock = None
    llm._llm = _ImageToVideoProvider()
    llm._maybe_prepare_prompt_cache = lambda **_kwargs: None

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=llm, tools=_NoopTools(), artifact_store=store),
    )
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(workflow=parent, session_id="sess-i2v")
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.image_to_video(
        parent_run_id,
        prompt="Add a slow camera orbit.",
        media={"$artifact": source.artifact_id, "filename": "source.png"},
        output={"provider": "mlx-gen", "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "format": "mp4"},
    )

    assert child.status == RunStatus.COMPLETED
    assert seen["output"]["task"] == "image_to_video"
    assert seen["media"][0]["artifact_id"] == source.artifact_id
    item = child.output["result"]["outputs"]["video"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"mp4-i2v"
    assert artifact.metadata.run_id == child.run_id


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


def test_run_facade_generate_voice_creates_durable_child_run_with_artifact_descriptor() -> None:
    store = InMemoryArtifactStore()
    seen: Dict[str, Any] = {}

    class _VoiceProvider:
        def generate(self, **kwargs):
            from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

            seen.update(kwargs)
            return MultimodalGenerateResponse(
                outputs={
                    "voice": [
                        GeneratedItem(
                            modality="voice",
                            task="tts",
                            data=b"wav-voice",
                            content_type="audio/wav",
                            format="wav",
                            provider="abstractvoice",
                            model="tts-test",
                            metadata={"voice_id": "narrator", "language": "en", "cloned_voice": False},
                        )
                    ]
                }
            )

    llm = object.__new__(LocalAbstractCoreLLMClient)
    llm._provider = "mlx"
    llm._model = "qwen-chat"
    llm._artifact_store = store
    llm._generate_lock = None
    llm._llm = _VoiceProvider()
    llm._maybe_prepare_prompt_cache = lambda **_kwargs: None

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=llm, tools=_NoopTools(), artifact_store=store),
    )
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(workflow=parent, session_id="sess-voice")
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.generate_voice(
        parent_run_id,
        text="Read this clearly.",
        output={"provider": "abstractvoice", "model": "tts-test", "format": "wav", "voice": "narrator", "language": "en"},
    )

    assert child.status == RunStatus.COMPLETED
    assert seen["prompt"] == "Read this clearly."
    assert seen["output"]["modality"] == "voice"
    assert seen["output"]["task"] == "tts"
    item = child.output["result"]["outputs"]["voice"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"wav-voice"
    descriptor = artifact.metadata.descriptor
    assert descriptor.semantic_kind == "voice"
    assert descriptor.render_kind == "audio"
    assert descriptor.modality == "voice"
    assert descriptor.task == "tts"
    assert descriptor.generation["text"] == "Read this clearly."
    assert descriptor.generation["params"]["voice"] == "narrator"
    assert descriptor.generation["params"]["language"] == "en"
    assert descriptor.producer["provider"] == "abstractvoice"
    assert descriptor.producer["model"] == "tts-test"
    assert descriptor.security["recorded_user_content_fields"] == ["text"]
    assert artifact.metadata.metadata["capability_metadata"]["voice_id"] == "narrator"
    assert artifact.metadata.metadata["capability_metadata"]["cloned_voice"] is False


def test_run_facade_generate_music_creates_durable_child_run_with_artifact_backed_output() -> None:
    store = InMemoryArtifactStore()
    seen: Dict[str, Any] = {}

    class _MusicProvider:
        def generate(self, **kwargs):
            from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

            seen.update(kwargs)
            return MultimodalGenerateResponse(
                outputs={
                    "music": [
                        GeneratedItem(
                            modality="music",
                            task="music_generation",
                            data=b"wav-child",
                            content_type="audio/wav",
                            format="wav",
                            provider="acemusic",
                            model="ace-step",
                        )
                    ]
                }
            )

    llm = object.__new__(LocalAbstractCoreLLMClient)
    llm._provider = "mlx"
    llm._model = "qwen-chat"
    llm._artifact_store = store
    llm._generate_lock = None
    llm._llm = _MusicProvider()
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
        session_id="sess-music",
        vars={"_runtime": {"prompt_cache": {"enabled": True}}},
    )
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.generate_music(
        parent_run_id,
        prompt="Warm lo-fi piano with brushed drums.",
        output={"provider": "acemusic", "model": "ace-step", "format": "wav"},
    )

    assert child.status == RunStatus.COMPLETED
    assert child.parent_run_id == parent_run_id
    assert child.session_id == "sess-music"
    assert seen["prompt"] == "Warm lo-fi piano with brushed drums."
    assert seen["output"]["modality"] == "music"
    assert seen["output"]["task"] == "music_generation"
    assert seen["output"]["provider"] == "acemusic"
    assert seen["output"]["model"] == "ace-step"

    result = child.output["result"]
    assert result["media_provider"] == "acemusic"
    assert result["media_model"] == "ace-step"
    assert result["model"] == "ace-step"
    item = result["outputs"]["music"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"wav-child"
    assert artifact.metadata.run_id == child.run_id
    assert artifact.metadata.tags["kind"] == "generated_media"
    descriptor = artifact.metadata.descriptor
    assert descriptor.semantic_kind == "music"
    assert descriptor.render_kind == "audio"
    assert descriptor.modality == "music"
    assert descriptor.task == "music_generation"
    assert descriptor.classification_source == "producer"
    assert descriptor.session_id == "sess-music"
    assert descriptor.workflow_id == "wf_abstractcore_run_facade_music_generation"
    assert descriptor.producer["provider"] == "acemusic"
    assert descriptor.producer["model"] == "ace-step"
    assert descriptor.producer["runtime_provider"] == "mlx"
    assert descriptor.generation["prompt"] == "Warm lo-fi piano with brushed drums."
    assert descriptor.generation["requested_format"] == "wav"
    assert descriptor.provenance["run_id"] == child.run_id
    assert descriptor.security["recorded_user_content_fields"] == ["prompt"]
    assert artifact.metadata.metadata["schema"] == "abstractruntime.generated_media_metadata.v1"
    assert artifact.metadata.metadata["generation"]["prompt"] == "Warm lo-fi piano with brushed drums."


def test_run_facade_generate_music_rejects_legacy_backend_fields() -> None:
    class _StubRuntime:
        def start(self, **kwargs):
            raise AssertionError("legacy music backend fields should be rejected before runtime start")

        def tick(self, **kwargs):
            raise AssertionError("legacy music backend fields should be rejected before runtime tick")

        def get_state(self, run_id: str):
            raise AssertionError(f"unexpected get_state({run_id})")

    facade = AbstractCoreRunFacade(_StubRuntime())

    with pytest.raises(ValueError, match="provider.*backend selector"):
        facade.generate_music("run-1", prompt="Warm lo-fi piano.", output={"backend": "acemusic"})


def test_run_facade_send_telegram_message_creates_durable_child_tool_run() -> None:
    seen: Dict[str, Any] = {}

    def send_telegram_message(
        *,
        chat_id: int,
        text: str,
        parse_mode: str = "",
        disable_web_page_preview: bool = False,
        timeout_s: float = 20.0,
        bot_token_env_var: str = "ABSTRACT_TELEGRAM_BOT_TOKEN",
    ) -> Dict[str, Any]:
        seen.update(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_web_page_preview,
                "timeout_s": timeout_s,
                "bot_token_env_var": bot_token_env_var,
            }
        )
        return {"success": True, "transport": "bot_api", "message_ids": [7]}

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_effect_handlers(
            llm=object(),
            tools=MappingToolExecutor({"send_telegram_message": send_telegram_message}),
        ),
    )
    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(
        workflow=parent,
        session_id="sess-telegram",
        vars={"_runtime": {"prompt_cache": {"enabled": True}}},
    )
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child = facade.send_telegram_message(
        parent_run_id,
        chat_id=123,
        text="Status green",
        parse_mode="Markdown",
        disable_web_page_preview=True,
        timeout_s=11,
        bot_token_env_var="TG_TOKEN",
    )

    assert child.status == RunStatus.COMPLETED
    assert child.parent_run_id == parent_run_id
    assert child.session_id == "sess-telegram"
    assert child.vars["_runtime"]["prompt_cache"]["enabled"] is True
    assert seen == {
        "chat_id": 123,
        "text": "Status green",
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
        "timeout_s": 11.0,
        "bot_token_env_var": "TG_TOKEN",
    }

    result = child.output["result"]
    assert result["mode"] == "executed"
    results = result["results"]
    assert isinstance(results, list) and len(results) == 1
    assert results[0]["name"] == "send_telegram_message"
    assert results[0]["success"] is True
    assert results[0]["output"]["message_ids"] == [7]

    ledger = runtime.get_ledger(child.run_id)
    effect_types = [
        effect.get("type")
        for entry in ledger
        if isinstance(entry, dict)
        for effect in [entry.get("effect")]
        if isinstance(effect, dict)
    ]
    assert "tool_calls" in effect_types


def test_run_facade_send_email_waits_for_approval_and_resumes_durably() -> None:
    seen: Dict[str, Any] = {}

    def send_email(
        *,
        to: Any,
        subject: str,
        account: str | None = None,
        body_text: str | None = None,
        body_html: str | None = None,
        cc: Any = None,
        bcc: Any = None,
        timeout_s: float = 30.0,
        headers: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        seen.update(
            {
                "to": to,
                "subject": subject,
                "account": account,
                "body_text": body_text,
                "body_html": body_html,
                "cc": cc,
                "bcc": bcc,
                "timeout_s": timeout_s,
                "headers": headers,
            }
        )
        return {"success": True, "message_id": "<m-1>", "account": account}

    delegate = MappingToolExecutor({"send_email": send_email})
    tools = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy())

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_effect_handlers(llm=object(), tools=tools),
    )
    runtime.set_tool_executor_for_resume(tools)

    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(workflow=parent, session_id="sess-email")
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child_waiting = facade.send_email(
        parent_run_id,
        to=["ops@example.com"],
        subject="Runtime status",
        account="ops",
        body_text="All green",
        cc="cc@example.com",
        timeout_s=12,
        headers={"X-Test": "1"},
    )

    assert child_waiting.status == RunStatus.WAITING
    assert child_waiting.waiting is not None
    assert child_waiting.waiting.reason.value == "user"
    assert child_waiting.waiting.details["mode"] == "approval_required"
    wait_calls = child_waiting.waiting.details.get("tool_calls")
    assert isinstance(wait_calls, list) and len(wait_calls) == 1
    assert wait_calls[0]["name"] == "send_email"

    child_done = facade.resume_tool_calls(
        child_waiting.run_id,
        payload={"approved": True},
        max_steps=10,
    )

    assert child_done.status == RunStatus.COMPLETED
    assert seen == {
        "to": ["ops@example.com"],
        "subject": "Runtime status",
        "account": "ops",
        "body_text": "All green",
        "body_html": None,
        "cc": "cc@example.com",
        "bcc": None,
        "timeout_s": 12.0,
        "headers": {"X-Test": "1"},
    }
    result = child_done.output["result"]
    assert result["mode"] == "executed"
    results = result["results"]
    assert isinstance(results, list) and len(results) == 1
    assert results[0]["name"] == "send_email"
    assert results[0]["success"] is True
    assert results[0]["output"]["message_id"] == "<m-1>"


def test_run_facade_resume_tool_calls_accepts_passthrough_results() -> None:
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=build_effect_handlers(llm=object(), tools=PassthroughToolExecutor()),
    )

    parent = _completed_parent_workflow()
    parent_run_id = runtime.start(workflow=parent, session_id="sess-tool-passthrough")
    runtime.tick(workflow=parent, run_id=parent_run_id)

    facade = get_abstractcore_run_facade(runtime)
    child_waiting = facade.send_telegram_message(
        parent_run_id,
        chat_id=789,
        text="Needs host send",
    )

    assert child_waiting.status == RunStatus.WAITING
    assert child_waiting.waiting is not None
    assert child_waiting.waiting.details["mode"] == "passthrough"

    child_done = facade.resume_tool_calls(
        child_waiting.run_id,
        payload={
            "mode": "executed",
            "results": [
                {
                    "call_id": "send_telegram_message",
                    "runtime_call_id": "host-call-1",
                    "name": "send_telegram_message",
                    "success": True,
                    "output": {"transport": "host", "message_ids": [99]},
                    "error": None,
                }
            ],
        },
        max_steps=10,
    )

    assert child_done.status == RunStatus.COMPLETED
    result = child_done.output["result"]
    assert result["mode"] == "executed"
    results = result["results"]
    assert isinstance(results, list) and len(results) == 1
    assert results[0]["name"] == "send_telegram_message"
    assert results[0]["success"] is True
    assert results[0]["output"]["message_ids"] == [99]
