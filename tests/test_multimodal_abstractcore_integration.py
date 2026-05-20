from __future__ import annotations

import base64
from pathlib import Path

import pytest

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.llm_client import (
    HttpBinaryResponse,
    LocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
    _normalize_local_response,
    _run_local_image_subprocess,
)
from abstractruntime.integrations.abstractcore.output_specs import (
    is_abstractcore_output_request,
    normalize_output_specs_for_runtime,
    output_request_has_generated_media,
    output_request_has_non_text_result,
)
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class _NoopTools:
    def execute(self, *, tool_calls):
        return {"mode": "executed", "results": []}


def test_normalize_multimodal_response_stores_generated_bytes_as_artifact() -> None:
    from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

    store = InMemoryArtifactStore()
    resp = MultimodalGenerateResponse(
        outputs={
            "image": [
                GeneratedItem(
                    modality="image",
                    task="image_generation",
                    data=b"png-bytes",
                    content_type="image/png",
                    format="png",
                    backend_id="fake-vision",
                    provider="fake",
                    model="fake-image-model",
                )
            ]
        }
    )

    out = _normalize_local_response(
        resp,
        artifact_store=store,
        run_id="run-mm",
        default_tags={"source": "test"},
    )

    assert out["media_provider"] == "fake"
    assert out["media_model"] == "fake-image-model"
    assert out["model"] == "fake-image-model"
    item = out["outputs"]["image"][0]
    assert item["artifact_id"]
    assert "data_base64" not in item
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-bytes"
    assert artifact.metadata.content_type == "image/png"
    assert artifact.metadata.run_id == "run-mm"
    assert artifact.metadata.tags["kind"] == "generated_media"
    assert artifact.metadata.tags["modality"] == "image"
    assert artifact.metadata.tags["task"] == "image_generation"


def test_normalize_multimodal_response_rejects_generated_bytes_without_artifact_store() -> None:
    from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

    resp = MultimodalGenerateResponse(
        outputs={
            "image": [
                GeneratedItem(
                    modality="image",
                    task="image_generation",
                    data=b"png-bytes",
                    content_type="image/png",
                    format="png",
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="ArtifactStore"):
        _normalize_local_response(resp)


def test_local_image_output_keeps_runtime_metadata_out_of_core_kwargs() -> None:
    from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

    store = InMemoryArtifactStore()
    seen = {}

    class _FakeLLM:
        def generate(self, **kwargs):
            seen.update(kwargs)
            return MultimodalGenerateResponse(
                outputs={
                    "image": [
                        GeneratedItem(
                            modality="image",
                            task="image_generation",
                            data=b"png-from-core",
                            content_type="image/png",
                            format="png",
                        )
                    ]
                }
            )

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "openai"
    client._model = "gpt-4o-mini"
    client._artifact_store = store
    client._generate_lock = None
    client._llm = _FakeLLM()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    out = client.generate(
        prompt="A red mug.",
        params={
            "output": {
                "modality": "image",
                "run_id": "run-img",
                "tags": {"node_id": "n-img", "tenant": "demo"},
            },
            "trace_metadata": {"run_id": "run-img", "node_id": "n-img"},
        },
    )

    assert seen["output"] == {"modality": "image"}
    assert "artifact_store" not in seen
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-from-core"
    assert artifact.metadata.run_id == "run-img"
    assert artifact.metadata.tags["node_id"] == "n-img"
    assert artifact.metadata.tags["tenant"] == "demo"


def test_local_image_media_only_runs_in_subprocess_and_stores_generated_bytes(monkeypatch) -> None:
    store = InMemoryArtifactStore()
    calls = []

    class _InProcessImageLLM:
        def _run_multimodal_spec(self, **_kwargs):
            raise AssertionError("image generation must be isolated from the runtime process")

    def fake_subprocess(**kwargs):
        calls.append(kwargs)
        return {
            "outputs": {
                "image": [
                    {
                        "modality": "image",
                        "task": "image_generation",
                        "data": b"png-from-subprocess",
                        "content_type": "image/png",
                        "format": "png",
                        "provider": "mflux",
                        "model": "flux2-klein-4b",
                    }
                ]
            },
            "metadata": {
                "media_only": True,
                "subprocess": True,
                "runtime_provider": "mlx",
                "runtime_model": "qwen3.5-2b",
                "execution_mode": "local_one_shot_subprocess",
            },
        }

    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.llm_client._run_local_image_subprocess",
        fake_subprocess,
    )

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "qwen3.5-2b"
    client._llm_kwargs = {"enable_tracing": True}
    client._artifact_store = store
    client._generate_lock = None
    client._llm = _InProcessImageLLM()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    out = client.generate(
        prompt="A red mug.",
        params={
            "output": {
                "modality": "image",
                "provider": "mflux",
                "model": "flux2-klein-4b",
                "run_id": "run-img",
                "tags": {"node_id": "n-img"},
            },
            "trace_metadata": {"run_id": "run-img", "node_id": "n-img"},
        },
    )

    assert calls
    assert calls[0]["provider"] == "mlx"
    assert calls[0]["model"] == "qwen3.5-2b"
    assert calls[0]["specs"][0]["provider"] == "mflux"
    assert calls[0]["specs"][0]["model"] == "flux2-klein-4b"
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-from-subprocess"
    assert artifact.metadata.run_id == "run-img"
    assert artifact.metadata.tags["node_id"] == "n-img"
    assert artifact.metadata.tags["kind"] == "generated_media"
    assert out["runtime_provider"] == "mlx"
    assert out["runtime_model"] == "qwen3.5-2b"
    assert out["media_provider"] == "mflux"
    assert out["media_model"] == "flux2-klein-4b"
    assert out["model"] == "flux2-klein-4b"


def test_local_image_subprocess_native_abort_becomes_python_error(monkeypatch) -> None:
    class _AbortResult:
        returncode = 134
        stdout = ""
        stderr = "failed assertion `A command encoder is already encoding to this command buffer'"

    def fake_run(*_args, **_kwargs):
        return _AbortResult()

    monkeypatch.setattr("abstractruntime.integrations.abstractcore.llm_client.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Local image generation subprocess exited with code 134"):
        _run_local_image_subprocess(
            provider="mlx",
            model="qwen3.5-2b",
            llm_kwargs={},
            prompt="A red mug.",
            specs=[{"modality": "image", "task": "image_generation"}],
            timeout_s=30,
        )


def test_local_generated_media_requires_artifact_store_before_provider_call() -> None:
    class _UnexpectedLLM:
        def generate(self, **kwargs):
            raise AssertionError("generated media should be rejected before provider call")

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "openai"
    client._model = "gpt-4o-mini"
    client._artifact_store = None
    client._generate_lock = None
    client._llm = _UnexpectedLLM()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    with pytest.raises(ValueError, match="ArtifactStore"):
        client.generate(prompt="A red mug.", params={"output": "image"})


def test_llm_call_allows_media_only_transcription_output_and_augments_output_spec() -> None:
    store = InMemoryArtifactStore()
    meta = store.store(
        b"fake-wav",
        content_type="audio/wav",
        tags={"filename": "speech.wav"},
    )
    seen = {}

    class _CapturingLLM:
        def generate(self, **kwargs):
            seen.update(kwargs)
            return {"content": "transcribed text", "metadata": {}}

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=_CapturingLLM(), tools=_NoopTools(), artifact_store=store),
    )

    def call(run, ctx):
        return StepPlan(
            node_id="call",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "media": {"$artifact": meta.artifact_id, "filename": "speech.wav"},
                    "output": "text",
                },
                result_key="llm",
            ),
            next_node="done",
        )

    def done(run, ctx):
        return StepPlan(node_id="done", complete_output={"llm": run.vars.get("llm")})

    wf = WorkflowSpec("wf-mm-stt", "call", {"call": call, "done": done})
    run_id = runtime.start(workflow=wf, session_id="sess-mm")
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.output["llm"]["content"] == "transcribed text"
    assert seen["prompt"] == ""
    assert isinstance(seen["media"], list)
    media_item = seen["media"][0]
    assert isinstance(media_item, dict)
    assert media_item.get("$artifact") == meta.artifact_id
    assert media_item.get("artifact_id") == meta.artifact_id
    assert media_item.get("content_type") == "audio/wav"
    assert media_item.get("type") == "audio"
    assert str(media_item.get("file_path") or "").endswith(".wav")
    output = seen["params"]["output"]
    assert output["modality"] == "text"
    assert output["run_id"] == run_id
    assert output["tags"]["kind"] == "llm_output"
    assert output["tags"]["session_id"] == "sess-mm"


def test_llm_call_generated_media_requires_artifact_store() -> None:
    class _UnexpectedLLM:
        def generate(self, **kwargs):
            raise AssertionError("generated media should be rejected before provider call")

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=None,
        effect_handlers=build_effect_handlers(llm=_UnexpectedLLM(), tools=_NoopTools(), artifact_store=None),
    )

    def call(run, ctx):
        return StepPlan(
            node_id="call",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": "red cube", "output": "image"},
                result_key="llm",
            ),
            next_node="done",
        )

    wf = WorkflowSpec("wf-mm-no-artifacts", "call", {"call": call})
    run_id = runtime.start(workflow=wf)
    state = runtime.tick(workflow=wf, run_id=run_id)

    assert state.status.value == "failed"
    assert "ArtifactStore" in str(state.error)


def test_runtime_output_selector_adapter_delegates_to_public_abstractcore_contract() -> None:
    from abstractcore.core.output_specs import (
        is_output_request,
        normalize_output_specs,
        output_has_generated_media,
        output_requires_non_chat_dispatch,
    )

    cases = [
        "text",
        "audio",
        {"modality": "audio"},
        {"modality": "voice"},
        {"task": "tts"},
        {"task": "text_generation"},
        {"task": "image_edit"},
        [{"modality": "text"}, {"modality": "image"}],
        "json",
        [],
    ]

    for case in cases:
        assert is_abstractcore_output_request(case) is is_output_request(case)

    text_generation = {"task": "text_generation"}
    assert normalize_output_specs_for_runtime(text_generation) == normalize_output_specs(text_generation)
    assert output_request_has_generated_media("image") is output_has_generated_media("image")
    assert output_request_has_generated_media(text_generation) is False
    assert output_request_has_non_text_result(text_generation) is output_requires_non_chat_dispatch(text_generation)
    assert output_request_has_non_text_result(text_generation) is False
    assert output_request_has_generated_media({"task": "voice_clone"}) is False


def test_local_voice_clone_resource_does_not_require_artifact_store() -> None:
    from abstractcore.core.multimodal_generation import GeneratedResource, MultimodalGenerateResponse

    seen = {}

    class _FakeLLM:
        def generate(self, **kwargs):
            seen.update(kwargs)
            return MultimodalGenerateResponse(
                resources={
                    "voice": [
                        GeneratedResource(
                            modality="voice",
                            task="voice_clone",
                            resource_type="voice",
                            resource_id="voice-demo",
                        )
                    ]
                }
            )

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "openai"
    client._model = "gpt-4o-mini"
    client._artifact_store = None
    client._generate_lock = None
    client._llm = _FakeLLM()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    out = client.generate(
        prompt="Reference text.",
        media=[{"type": "audio", "path": "reference.wav"}],
        params={"output": {"task": "voice_clone", "run_id": "run-voice", "tags": {"node_id": "n-voice"}}},
    )

    assert seen["output"] == {"task": "voice_clone"}
    resource = out["resources"]["voice"][0]
    assert resource["resource_id"] == "voice-demo"


class _RemoteImageSender:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, *, headers, timeout):
        raise AssertionError("not used")

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        return {"created": 1, "data": [{"b64_json": base64.b64encode(b"png-remote").decode("ascii")}]}


def test_remote_image_output_uses_abstractcore_server_endpoint_and_stores_artifact() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="A red cube on a white table.",
        params={
            "output": {"modality": "image", "model": "openai-compatible/gpt-image-1", "format": "png"},
            "trace_metadata": {"run_id": "run-img", "node_id": "n-img"},
        },
    )

    assert sender.calls[0]["url"] == "http://core.test/v1/images/generations"
    assert sender.calls[0]["json"]["model"] == "openai-compatible/gpt-image-1"
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-remote"
    assert artifact.metadata.run_id == "run-img"
    assert artifact.metadata.tags["node_id"] == "n-img"


def test_remote_image_output_does_not_reuse_chat_model_as_generation_model() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(prompt="A red cube.", params={"output": {"modality": "image", "format": "png"}})

    assert "model" not in sender.calls[0]["json"]


def test_remote_image_output_preserves_mflux_provider_and_model_separately() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(
        prompt="A red cube.",
        params={
            "output": {
                "modality": "image",
                "provider": "mflux",
                "model": "flux2-klein-4b",
                "format": "png",
            }
        },
    )

    assert sender.calls[0]["url"] == "http://core.test/v1/images/generations"
    assert sender.calls[0]["json"]["provider"] == "mflux"
    assert sender.calls[0]["json"]["model"] == "flux2-klein-4b"
    assert not str(sender.calls[0]["json"]["model"]).startswith("diffusers/")


def test_remote_image_output_rejects_input_media_instead_of_ignoring_it(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-input")
    store = InMemoryArtifactStore()
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    with pytest.raises(ValueError, match="does not accept input media"):
        client.generate(prompt="make it watercolor", media=[str(image_path)], params={"output": "image"})

    assert sender.calls == []


def test_remote_generated_media_requires_artifact_store_before_provider_call() -> None:
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    with pytest.raises(ValueError, match="ArtifactStore"):
        client.generate(prompt="A red cube.", params={"output": "image"})

    assert sender.calls == []


def test_remote_image_output_preserves_output_spec_tags() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="A red cube on a white table.",
        params={
            "output": {
                "modality": "image",
                "format": "png",
                "run_id": "run-img-tags",
                "tags": {"tenant": "demo", "node_id": "node-from-output"},
            },
        },
    )

    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.metadata.run_id == "run-img-tags"
    assert artifact.metadata.tags["tenant"] == "demo"
    assert artifact.metadata.tags["node_id"] == "node-from-output"


def test_remote_image_output_extracts_prompt_from_content_array_message() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(
        prompt="",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "A blue glass orb."}],
            }
        ],
        params={"output": {"modality": "image", "format": "png"}},
    )

    assert sender.calls[0]["json"]["prompt"] == "A blue glass orb."


class _RemoteTtsSender:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, *, headers, timeout):
        raise AssertionError("not used")

    def post(self, url, *, headers, json, timeout):
        raise AssertionError("post_bytes should be used for binary speech")

    def post_bytes(self, url, *, headers, json, timeout):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        return HttpBinaryResponse(content=b"wav-remote", headers={"content-type": "audio/wav"})


def test_remote_voice_output_uses_speech_endpoint_and_stores_artifact() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteTtsSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini-tts",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="Hello from runtime.",
        params={
            "output": {"modality": "voice", "voice": "coral", "format": "wav"},
            "trace_metadata": {"run_id": "run-tts"},
        },
    )

    assert sender.calls[0]["url"] == "http://core.test/v1/audio/speech"
    assert sender.calls[0]["json"]["input"] == "Hello from runtime."
    assert sender.calls[0]["json"]["voice"] == "coral"
    item = out["outputs"]["voice"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"wav-remote"
    assert artifact.metadata.content_type == "audio/wav"
    assert artifact.metadata.run_id == "run-tts"


def test_remote_voice_output_does_not_reuse_chat_model_as_tts_model() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteTtsSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(prompt="Hello.", params={"output": {"modality": "voice", "voice": "coral", "format": "wav"}})

    assert "model" not in sender.calls[0]["json"]


def test_remote_voice_output_uses_provider_scoped_speech_endpoint() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteTtsSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(
        prompt="Hello.",
        params={
            "output": {
                "modality": "voice",
                "provider": "supertonic",
                "model": "supertonic-3",
                "voice": "M1",
                "format": "wav",
            }
        },
    )

    assert sender.calls[0]["url"] == "http://core.test/supertonic/v1/audio/speech"
    assert sender.calls[0]["json"]["model"] == "supertonic-3"
    assert sender.calls[0]["json"]["voice"] == "M1"
    assert "provider" not in sender.calls[0]["json"]


def test_remote_voice_output_does_not_inject_default_voice_for_local_core() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteTtsSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(prompt="Hello.", params={"output": {"modality": "voice", "format": "wav"}})

    assert "voice" not in sender.calls[0]["json"]


def test_remote_voice_output_rejects_input_audio_instead_of_ignoring_it(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"wav-input")
    store = InMemoryArtifactStore()
    sender = _RemoteTtsSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    with pytest.raises(ValueError, match="does not accept input audio"):
        client.generate(prompt="Hello.", media=[str(audio_path)], params={"output": "voice"})

    assert sender.calls == []


class _RemoteTranscriptionSender:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, *, headers, timeout):
        raise AssertionError("not used")

    def post(self, url, *, headers, json, timeout):
        raise AssertionError("post_multipart should be used for transcription")

    def post_multipart(self, url, *, headers, data, files, timeout):
        self.calls.append(
            {"method": "POST", "url": url, "headers": headers, "data": data, "files": files, "timeout": timeout}
        )
        return {"text": "hello from audio"}


def test_remote_transcription_resolves_artifact_refs_for_direct_client_use() -> None:
    store = InMemoryArtifactStore()
    meta = store.store(
        b"wav-input",
        content_type="audio/wav",
        tags={"filename": "speech.wav"},
    )
    sender = _RemoteTranscriptionSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/whisper-1",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="",
        media=[{"$artifact": meta.artifact_id, "filename": "speech.wav"}],
        params={"output": {"modality": "text", "task": "transcription", "language": "en"}},
    )

    assert out["content"] == "hello from audio"
    assert sender.calls[0]["url"] == "http://core.test/v1/audio/transcriptions"
    assert sender.calls[0]["data"]["language"] == "en"
    file_tuple = sender.calls[0]["files"]["file"]
    assert file_tuple[0].endswith(".wav")
    assert file_tuple[1] == b"wav-input"
    assert file_tuple[2] in {"audio/wav", "audio/x-wav"}


def test_remote_transcription_does_not_reuse_chat_model_as_stt_model() -> None:
    store = InMemoryArtifactStore()
    meta = store.store(b"wav-input", content_type="audio/wav", tags={"filename": "speech.wav"})
    sender = _RemoteTranscriptionSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(
        prompt="",
        media=[{"$artifact": meta.artifact_id, "filename": "speech.wav"}],
        params={"output": {"modality": "text", "task": "transcription"}},
    )

    assert "model" not in sender.calls[0]["data"]


def test_remote_transcription_uses_provider_scoped_endpoint() -> None:
    store = InMemoryArtifactStore()
    meta = store.store(b"wav-input", content_type="audio/wav", tags={"filename": "speech.wav"})
    sender = _RemoteTranscriptionSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    client.generate(
        prompt="",
        media=[{"$artifact": meta.artifact_id, "filename": "speech.wav"}],
        params={
            "output": {
                "modality": "text",
                "task": "transcription",
                "provider": "faster-whisper",
                "model": "base",
            }
        },
    )

    assert sender.calls[0]["url"] == "http://core.test/faster-whisper/v1/audio/transcriptions"
    assert sender.calls[0]["data"]["model"] == "base"
    assert "provider" not in sender.calls[0]["data"]


def test_remote_transcription_rejects_non_audio_media(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-input")
    sender = _RemoteTranscriptionSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    with pytest.raises(ValueError, match="exactly one audio"):
        client.generate(prompt="", media=[str(image_path)], params={"output": {"modality": "text"}})

    assert sender.calls == []


def test_remote_transcription_rejects_audio_url_without_downloading() -> None:
    sender = _RemoteTranscriptionSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    with pytest.raises(ValueError, match="local file path"):
        client.generate(
            prompt="",
            media=["https://example.test/speech.wav"],
            params={"output": {"modality": "text", "task": "transcription"}},
        )

    assert sender.calls == []


def test_remote_transcription_rejects_inline_audio_without_temp_file_guesswork() -> None:
    sender = _RemoteTranscriptionSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    with pytest.raises(ValueError, match="resolve to a local file path"):
        client.generate(
            prompt="",
            media=[{"content": b"wav-input", "content_type": "audio/wav"}],
            params={"output": {"modality": "text", "task": "transcription"}},
        )

    assert sender.calls == []


class _RemoteChatMediaSender:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, *, headers, timeout):
        raise AssertionError("not used")

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        return {
            "model": json["model"],
            "choices": [{"message": {"role": "assistant", "content": "looks red"}, "finish_reason": "stop"}],
        }


def test_remote_chat_media_is_sent_as_openai_content_array(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-input")
    sender = _RemoteChatMediaSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    out = client.generate(prompt="What color is it?", media=[str(image_path)])

    assert out["content"] == "looks red"
    body = sender.calls[0]["json"]
    content = body["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    observed = out["metadata"]["_provider_request"]["payload"]["messages"][-1]["content"][1]["image_url"]["url"]
    assert observed.startswith("data:image/png;base64,<redacted ")
    assert "cG5nLWlucHV0" not in observed


def test_remote_text_alias_is_promoted_to_chat_prompt() -> None:
    sender = _RemoteChatMediaSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    client.generate(prompt="", params={"text": "Hello through text alias."})

    msg = sender.calls[0]["json"]["messages"][-1]
    assert msg["role"] == "user"
    assert msg["content"].endswith("Hello through text alias.")


def test_remote_chat_media_resolves_artifact_refs_for_direct_client_use() -> None:
    store = InMemoryArtifactStore()
    meta = store.store(
        b"png-input",
        content_type="image/png",
        tags={"filename": "sample.png"},
    )
    sender = _RemoteChatMediaSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="What color is it?",
        media=[{"$artifact": meta.artifact_id, "filename": "sample.png"}],
    )

    assert out["content"] == "looks red"
    content = sender.calls[0]["json"]["messages"][-1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_remote_chat_media_sends_inline_content_dict() -> None:
    sender = _RemoteChatMediaSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    out = client.generate(
        prompt="What color is it?",
        media=[{"content": b"png-input", "content_type": "image/png"}],
    )

    assert out["content"] == "looks red"
    content = sender.calls[0]["json"]["messages"][-1]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_remote_chat_media_preserves_http_url_media() -> None:
    sender = _RemoteChatMediaSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    client.generate(prompt="What color is it?", media=["https://example.test/sample.png"])

    content = sender.calls[0]["json"]["messages"][-1]["content"]
    assert content[1] == {"type": "image_url", "image_url": {"url": "https://example.test/sample.png"}}


def test_remote_chat_media_rejects_unresolved_media_dict() -> None:
    sender = _RemoteChatMediaSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    with pytest.raises(ValueError, match="missing file path"):
        client.generate(prompt="Describe it.", media=[{"content_type": "image/png"}])

    assert sender.calls == []


def test_remote_chat_media_preserves_existing_content_array(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-input")
    sender = _RemoteChatMediaSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
    )

    client.generate(
        prompt="",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Keep this text."}],
            }
        ],
        media=[str(image_path)],
    )

    content = sender.calls[0]["json"]["messages"][-1]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"].endswith("Keep this text.")
    assert content[1]["type"] == "image_url"
