from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.llm_client import (
    HttpBinaryResponse,
    LocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
    _is_subprocess_safe_image_specs,
    _normalize_local_response,
    _resolve_media_artifacts,
    _run_local_image_subprocess,
)
from abstractruntime.integrations.abstractcore.output_specs import (
    is_abstractcore_output_request,
    normalize_output_specs_for_runtime,
    output_request_has_generated_media,
    output_request_has_non_text_result,
)
from abstractruntime.storage.artifacts import FileArtifactStore, InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class _NoopTools:
    def execute(self, *, tool_calls):
        return {"mode": "executed", "results": []}


def test_resolve_media_artifacts_uses_typed_temp_path_for_blob_store(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "runtime")
    meta = store.store(
        b"png-input",
        content_type="image/png",
        tags={"filename": "content.png"},
    )
    temp_dir = tmp_path / "media"
    temp_dir.mkdir()

    resolved = _resolve_media_artifacts(
        [{"$artifact": meta.artifact_id, "filename": "content.png"}],
        artifact_store=store,
        temp_dir=str(temp_dir),
    )

    item = resolved[0]
    assert item["content_type"] == "image/png"
    assert item["mime_type"] == "image/png"
    assert item["type"] == "image"
    assert item["file_path"].endswith(".png")
    assert Path(item["file_path"]).read_bytes() == b"png-input"


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


def test_generated_media_artifact_identity_is_step_scoped() -> None:
    from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

    store = InMemoryArtifactStore()

    class _FakeLLM:
        def generate(self, **_kwargs):
            return MultimodalGenerateResponse(
                outputs={
                    "voice": [
                        GeneratedItem(
                            modality="voice",
                            task="tts",
                            data=b"same-wav-bytes",
                            content_type="audio/wav",
                            format="wav",
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

    out1 = client.generate(
        prompt="Say this.",
        params={
            "output": {"modality": "voice", "format": "wav"},
            "trace_metadata": {"run_id": "run-loop", "node_id": "voice", "step_id": "step-1"},
        },
    )
    out2 = client.generate(
        prompt="Say this.",
        params={
            "output": {"modality": "voice", "format": "wav"},
            "trace_metadata": {"run_id": "run-loop", "node_id": "voice", "step_id": "step-2"},
        },
    )

    item1 = out1["outputs"]["voice"][0]
    item2 = out2["outputs"]["voice"][0]
    assert item1["artifact_id"] != item2["artifact_id"]
    artifact1 = store.load(item1["artifact_id"])
    artifact2 = store.load(item2["artifact_id"])
    assert artifact1 is not None
    assert artifact2 is not None
    assert artifact1.content == artifact2.content == b"same-wav-bytes"
    assert artifact1.metadata.blob_id == artifact2.metadata.blob_id
    assert artifact1.metadata.tags["step_id"] == "step-1"
    assert artifact2.metadata.tags["step_id"] == "step-2"


def test_runtime_injects_generated_media_progress_callback_into_llm_call() -> None:
    from abstractcore.core.multimodal_generation import GeneratedItem, MultimodalGenerateResponse

    store = InMemoryArtifactStore()
    ledger = InMemoryLedgerStore()
    seen: dict[str, object] = {}

    class _ProgressLLM:
        def generate(self, **kwargs):
            params = dict(kwargs.get("params") or {})
            seen["params"] = params
            callback = kwargs.get("on_progress") or params.get("on_progress")
            assert callable(callback)

            class _Event:
                phase = "denoise"
                frame = 3
                total_frames = 5
                step = 2
                total_steps = 4
                progress = 0.5
                step_progress = 0.5
                frame_progress = 0.6
                task = "text_to_video"

            callback(_Event())
            return MultimodalGenerateResponse(
                outputs={
                    "video": [
                        GeneratedItem(
                            modality="video",
                            task="text_to_video",
                            data=b"mp4-bytes",
                            content_type="video/mp4",
                            format="mp4",
                        )
                    ]
                }
            )

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "openai"
    client._model = "gpt-4o-mini"
    client._artifact_store = store
    client._generate_lock = None
    client._capability_defaults = {}
    client._llm = _ProgressLLM()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers=build_effect_handlers(llm=client, artifact_store=store),
        artifact_store=store,
    )

    def start(run, ctx):
        del run, ctx
        return StepPlan(
            node_id="video",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": "A logo reveal.",
                    "output": {"modality": "video", "task": "text_to_video", "format": "mp4"},
                },
                result_key="result",
            ),
            next_node="done",
        )

    def done(run, ctx):
        del ctx
        return StepPlan(node_id="done", complete_output={"result": run.vars.get("result")})

    workflow = WorkflowSpec(workflow_id="progress_video", entry_node="video", nodes={"video": start, "done": done})
    run_id = runtime.start(workflow=workflow, vars={})
    state = runtime.tick(workflow=workflow, run_id=run_id)

    assert state.status.value == "completed"
    assert seen["params"] == {}
    records = ledger.list(run_id)
    llm_records = [r for r in records if (r.get("effect") or {}).get("type") == "llm_call"]
    assert llm_records
    assert "on_progress" not in ((llm_records[0].get("effect") or {}).get("payload") or {}).get("params", {})
    progress_records = [r for r in records if ((r.get("effect") or {}).get("payload") or {}).get("name") == "abstract.progress"]
    assert progress_records
    payloads = [
        ((record.get("effect") or {}).get("payload") or {}).get("payload") or {}
        for record in progress_records
    ]
    assert not any(payload.get("progress_mode") == "unreported" or payload.get("reported") is False for payload in payloads)
    progress_payload = {}
    for record in progress_records:
        candidate = ((record.get("effect") or {}).get("payload") or {}).get("payload") or {}
        if candidate.get("phase") == "denoise":
            progress_payload = candidate
            break
    assert progress_payload["phase"] == "denoise"
    assert progress_payload["step"] == 2
    assert progress_payload["total_steps"] == 4
    assert progress_payload["progress"] == 0.5
    assert progress_payload["step_progress"] == 0.5
    assert progress_payload["frame"] == 3
    assert progress_payload["total_frames"] == 5
    assert progress_payload["frame_progress"] == 0.6
    assert progress_payload["task"] == "text_to_video"
    item = (state.vars["result"]["outputs"]["video"])[0]
    assert store.load(item["artifact_id"]) is not None


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
                        "model": "AbstractFramework/flux.2-klein-4b-4bit",
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
                "model": "AbstractFramework/flux.2-klein-4b-4bit",
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
    assert calls[0]["specs"][0]["model"] == "AbstractFramework/flux.2-klein-4b-4bit"
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
    assert out["media_model"] == "AbstractFramework/flux.2-klein-4b-4bit"
    assert out["model"] == "AbstractFramework/flux.2-klein-4b-4bit"


def test_local_image_subprocess_native_abort_becomes_python_error(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import llm_client

    class _FakeStdin:
        def write(self, _value):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def has_line(self) -> bool:
            return False

        def readline(self) -> str:
            return ""

        def read(self) -> str:
            return "failed assertion `A command encoder is already encoding to this command buffer'"

    class _FakeProc:
        stdin = _FakeStdin()
        stdout = _FakeStdout()

        def poll(self):
            return 134

        def wait(self):
            return 134

        def kill(self):
            return None

    class _FakeSelector:
        def register(self, fileobj, _event):
            self.fileobj = fileobj

        def select(self, timeout=0):
            return []

        def close(self):
            return None

    monkeypatch.setattr(llm_client.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProc())
    monkeypatch.setattr(llm_client.selectors, "DefaultSelector", _FakeSelector)

    with pytest.raises(RuntimeError, match="Local image generation subprocess exited with code 134"):
        _run_local_image_subprocess(
            provider="mlx",
            model="qwen3.5-2b",
            llm_kwargs={},
            prompt="A red mug.",
            specs=[{"modality": "image", "task": "image_generation"}],
        )


def test_local_image_subprocess_streams_progress_events(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import llm_client

    final_payload = {
        "ok": True,
        "response": {
            "outputs": {
                "image": [
                    {
                        "modality": "image",
                        "task": "image_generation",
                        "data_b64": base64.b64encode(b"png-output").decode("ascii"),
                        "content_type": "image/png",
                        "format": "png",
                    }
                ]
            }
        },
    }
    lines = [
        json.dumps({"type": "progress", "event": {"phase": "denoise", "step": 2, "total_steps": 4, "progress": 0.5}}) + "\n",
        json.dumps(final_payload) + "\n",
    ]

    class _FakeStdin:
        def write(self, _value):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def __init__(self):
            self.lines = list(lines)

        def has_line(self) -> bool:
            return bool(self.lines)

        def readline(self) -> str:
            return self.lines.pop(0) if self.lines else ""

        def read(self) -> str:
            return ""

    class _FakeProc:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()

        def poll(self):
            return None if self.stdout.has_line() else 0

        def wait(self):
            return 0

        def kill(self):
            return None

    class _SelectorKey:
        def __init__(self, fileobj):
            self.fileobj = fileobj

    class _FakeSelector:
        def register(self, fileobj, _event):
            self.fileobj = fileobj

        def select(self, timeout=0):
            if self.fileobj.has_line():
                return [(_SelectorKey(self.fileobj), None)]
            return []

        def close(self):
            return None

    events = []
    monkeypatch.setattr(llm_client.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProc())
    monkeypatch.setattr(llm_client.selectors, "DefaultSelector", _FakeSelector)

    out = _run_local_image_subprocess(
        provider="mlx",
        model="qwen3.5-2b",
        llm_kwargs={},
        prompt="A red mug.",
        specs=[{"modality": "image", "task": "image_generation"}],
        progress_callback=events.append,
    )

    assert events == [{"phase": "denoise", "step": 2, "total_steps": 4, "progress": 0.5}]
    assert out["outputs"]["image"][0]["data"] == b"png-output"


def test_local_inprocess_media_spec_receives_progress_callback() -> None:
    from abstractcore.core.multimodal_generation import GeneratedItem

    store = InMemoryArtifactStore()
    specs_seen = []
    progress_events = []

    class _InProcessImageLLM:
        def _run_multimodal_spec(self, *, result, spec, prompt, media, artifact_store):
            del prompt, media, artifact_store
            specs_seen.append(spec)
            callback = spec.get("extra", {}).get("on_progress")
            if callable(callback):
                callback({"phase": "denoise", "step": 1, "total_steps": 2, "progress": 0.5})
            result.outputs.setdefault("image", []).append(
                GeneratedItem(
                    modality="image",
                    task="image_edit",
                    data=b"png-edited",
                    content_type="image/png",
                    format="png",
                    provider="mlx-gen",
                    model="AbstractFramework/qwen-image-edit-2511-4bit",
                )
            )

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "qwen3.5-2b"
    client._llm_kwargs = {}
    client._artifact_store = store
    client._generate_lock = None
    client._llm = _InProcessImageLLM()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    out = client.generate(
        prompt="Change the color.",
        media=[{"type": "image", "content": b"png-source", "role": "source"}],
        params={
            "output": {
                "modality": "image",
                "task": "image_edit",
                "provider": "mlx-gen",
                "model": "AbstractFramework/qwen-image-edit-2511-4bit",
            },
            "on_progress": progress_events.append,
        },
    )

    assert callable(specs_seen[0]["extra"]["on_progress"])
    assert progress_events == [{"phase": "denoise", "step": 1, "total_steps": 2, "progress": 0.5}]
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-edited"


def test_local_video_media_only_runs_in_subprocess_and_stores_generated_bytes(monkeypatch, tmp_path) -> None:
    store = InMemoryArtifactStore()
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-source")
    calls = []
    progress_events = []

    class _InProcessVideoLLM:
        def _run_multimodal_spec(self, **_kwargs):
            raise AssertionError("video generation must be isolated from the runtime process")

    def fake_subprocess(**kwargs):
        calls.append(kwargs)
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback({"phase": "denoise", "step": 1, "total_steps": 2, "frame": 1, "total_frames": 2, "progress": 0.5})
        return {
            "outputs": {
                "video": [
                    {
                        "modality": "video",
                        "task": "image_to_video",
                        "data": b"mp4-from-subprocess",
                        "content_type": "video/mp4",
                        "format": "mp4",
                        "provider": "mlx-gen",
                        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                    }
                ]
            },
            "metadata": {
                "media_only": True,
                "subprocess": True,
                "runtime_provider": "mlx",
                "runtime_model": "qwen3.5-2b",
                "execution_mode": "local_video_subprocess",
            },
        }

    monkeypatch.setattr(
        "abstractruntime.integrations.abstractcore.llm_client._run_local_video_subprocess",
        fake_subprocess,
    )

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"
    client._model = "qwen3.5-2b"
    client._llm_kwargs = {"enable_tracing": True}
    client._artifact_store = store
    client._generate_lock = None
    client._llm = _InProcessVideoLLM()
    client._maybe_prepare_prompt_cache = lambda **_kwargs: None

    out = client.generate(
        prompt="make it move",
        media=[{"file_path": str(image_path), "content_type": "image/png", "type": "image"}],
        params={
            "output": {
                "modality": "video",
                "task": "image_to_video",
                "provider": "mlx-gen",
                "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "run_id": "run-video",
                "tags": {"node_id": "n-video"},
            },
            "trace_metadata": {"run_id": "run-video", "node_id": "n-video"},
            "on_progress": progress_events.append,
        },
    )

    assert calls
    assert calls[0]["provider"] == "mlx"
    assert calls[0]["model"] == "qwen3.5-2b"
    assert "timeout_s" not in calls[0]
    assert calls[0]["specs"][0]["provider"] == "mlx-gen"
    assert calls[0]["media"][0]["file_path"] == str(image_path)
    assert progress_events and progress_events[0]["progress"] == 0.5
    item = out["outputs"]["video"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"mp4-from-subprocess"
    assert artifact.metadata.run_id == "run-video"
    assert artifact.metadata.tags["node_id"] == "n-video"
    assert artifact.metadata.tags["kind"] == "generated_media"
    assert out["runtime_provider"] == "mlx"
    assert out["runtime_model"] == "qwen3.5-2b"
    assert out["media_provider"] == "mlx-gen"
    assert out["media_model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


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
        if "/videos/" in url:
            return {"created": 1, "data": [{"b64_json": base64.b64encode(b"mp4-remote").decode("ascii")}]}
        return {"created": 1, "data": [{"b64_json": base64.b64encode(b"png-remote").decode("ascii")}]}

    def post_multipart(self, url, *, headers, data, files, timeout):
        self.calls.append(
            {
                "method": "MULTIPART",
                "url": url,
                "headers": headers,
                "data": dict(data),
                "files": {name: (value[0], value[1], value[2]) for name, value in files.items()},
                "timeout": timeout,
            }
        )
        if "/videos/" in url:
            return {"created": 1, "data": [{"b64_json": base64.b64encode(b"mp4-edited").decode("ascii")}]}
        return {"created": 1, "data": [{"b64_json": base64.b64encode(b"png-edited").decode("ascii")}]}


def _assert_generated_media_calls_have_no_timeout(sender) -> None:
    assert sender.calls
    assert all(call.get("timeout") is None for call in sender.calls)


class _RemoteImageGenerationJobSender(_RemoteImageSender):
    def __init__(self) -> None:
        super().__init__()
        self.poll_count = 0

    def get(self, url, *, headers, timeout):
        self.calls.append({"method": "GET", "url": url, "headers": headers, "timeout": timeout})
        self.poll_count += 1
        if self.poll_count == 1:
            return {
                "id": "job-image",
                "state": "running",
                "progress": {
                    "phase": "denoise",
                    "step": 1,
                    "total_steps": 2,
                    "progress": 0.5,
                    "last_event": {"task": "image_generation", "step_progress": 0.5},
                },
            }
        return {
            "id": "job-image",
            "state": "succeeded",
            "progress": {
                "phase": "done",
                "step": 2,
                "total_steps": 2,
                "progress": 1.0,
                "last_event": {"task": "image_generation", "step_progress": 1.0},
            },
            "result": {"created": 1, "data": [{"b64_json": base64.b64encode(b"png-image-job").decode("ascii")}]},
        }

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        if url.endswith("/v1/vision/jobs/images/generations"):
            return {"job_id": "job-image"}
        return super().post(url, headers=headers, json=json, timeout=timeout)


def test_runtime_records_remote_image_generation_progress_from_core_job() -> None:
    store = InMemoryArtifactStore()
    ledger = InMemoryLedgerStore()
    sender = _RemoteImageGenerationJobSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers=build_effect_handlers(llm=client, artifact_store=store),
        artifact_store=store,
    )

    def start(run, ctx):
        del run, ctx
        return StepPlan(
            node_id="image",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": "A red cube.",
                    "params": {
                        "output": {
                            "modality": "image",
                            "provider": "mlx-gen",
                            "model": "AbstractFramework/flux.2-klein-4b-4bit",
                            "format": "png",
                            "steps": 2,
                        },
                        "poll_interval_s": 0.05,
                    },
                },
                result_key="result",
            ),
            next_node="done",
        )

    def done(run, ctx):
        del ctx
        return StepPlan(node_id="done", complete_output={"result": run.vars.get("result")})

    workflow = WorkflowSpec(workflow_id="remote_image_progress", entry_node="image", nodes={"image": start, "done": done})
    run_id = runtime.start(workflow=workflow, vars={})
    state = runtime.tick(workflow=workflow, run_id=run_id)

    assert state.status.value == "completed"
    assert sender.calls[0]["url"] == "http://core.test/v1/vision/jobs/images/generations"
    _assert_generated_media_calls_have_no_timeout(sender)
    records = ledger.list(run_id)
    progress_records = [
        r
        for r in records
        if ((r.get("effect") or {}).get("payload") or {}).get("name") == "abstract.progress"
    ]
    assert progress_records
    payloads = [((r.get("effect") or {}).get("payload") or {}).get("payload") or {} for r in progress_records]
    assert any(p.get("last_event", {}).get("task") == "image_generation" for p in payloads)
    assert any(p.get("progress") == 1.0 for p in payloads)
    item = state.vars["result"]["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-image-job"


def test_remote_vision_job_poll_has_no_total_timeout(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import llm_client

    sender = _RemoteVideoJobSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        timeout_s=0.01,
    )
    ticks = iter([100.0, 200.0, 300.0])
    monkeypatch.setattr(llm_client.time, "time", lambda: next(ticks, 300.0))
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)

    out = client._poll_remote_vision_job(
        job_id="job-video",
        headers={},
        params={"poll_interval_s": 0.01},
    )

    assert out["data"][0]["b64_json"] == base64.b64encode(b"mp4-job").decode("ascii")
    assert sender.poll_count == 2
    _assert_generated_media_calls_have_no_timeout(sender)


class _RemoteImageEditJobSender(_RemoteImageSender):
    def __init__(self) -> None:
        super().__init__()
        self.poll_count = 0

    def get(self, url, *, headers, timeout):
        self.calls.append({"method": "GET", "url": url, "headers": headers, "timeout": timeout})
        self.poll_count += 1
        if self.poll_count == 1:
            return {
                "id": "job-edit",
                "state": "running",
                "progress": {
                    "phase": "denoise",
                    "step": 1,
                    "total_steps": 2,
                    "progress": 0.5,
                    "last_event": {"task": "image_edit", "step_progress": 0.5},
                },
            }
        return {
            "id": "job-edit",
            "state": "succeeded",
            "progress": {
                "phase": "done",
                "step": 2,
                "total_steps": 2,
                "progress": 1.0,
                "last_event": {"task": "image_edit", "step_progress": 1.0},
            },
            "result": {"created": 1, "data": [{"b64_json": base64.b64encode(b"png-edit-job").decode("ascii")}]},
        }

    def post_multipart(self, url, *, headers, data, files, timeout):
        self.calls.append(
            {
                "method": "MULTIPART",
                "url": url,
                "headers": headers,
                "data": dict(data),
                "files": {name: (value[0], value[1], value[2]) for name, value in files.items()},
                "timeout": timeout,
            }
        )
        if url.endswith("/v1/vision/jobs/images/edits"):
            return {"job_id": "job-edit"}
        return super().post_multipart(url, headers=headers, data=data, files=files, timeout=timeout)


class _RemoteVideoJobSender(_RemoteImageSender):
    def __init__(self) -> None:
        super().__init__()
        self.poll_count = 0

    def get(self, url, *, headers, timeout):
        self.calls.append({"method": "GET", "url": url, "headers": headers, "timeout": timeout})
        self.poll_count += 1
        if self.poll_count == 1:
            return {
                "id": "job-video",
                "state": "running",
                "progress": {"phase": "denoise", "step": 1, "total_steps": 2, "progress": 0.5},
            }
        return {
            "id": "job-video",
            "state": "succeeded",
            "progress": {"phase": "done", "step": 2, "total_steps": 2, "progress": 1.0},
            "result": {"created": 1, "data": [{"b64_json": base64.b64encode(b"mp4-job").decode("ascii")}]},
        }

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        if url.endswith("/v1/vision/jobs/videos/generations"):
            return {"job_id": "job-video"}
        return super().post(url, headers=headers, json=json, timeout=timeout)


class _RemoteImageToVideoJobSender(_RemoteImageSender):
    def __init__(self) -> None:
        super().__init__()
        self.poll_count = 0

    def get(self, url, *, headers, timeout):
        self.calls.append({"method": "GET", "url": url, "headers": headers, "timeout": timeout})
        self.poll_count += 1
        if self.poll_count == 1:
            return {
                "id": "job-i2v",
                "state": "running",
                "progress": {
                    "phase": "denoise",
                    "frame": 20,
                    "total_frames": 81,
                    "step": 5,
                    "total_steps": 20,
                    "progress": 0.25,
                    "step_progress": 0.25,
                    "frame_progress": 20 / 81,
                    "last_event": {
                        "task": "image_to_video",
                        "step": 5,
                        "total_steps": 20,
                        "step_progress": 0.25,
                        "frame": 20,
                        "total_frames": 81,
                        "frame_progress": 20 / 81,
                    },
                },
            }
        return {
            "id": "job-i2v",
            "state": "succeeded",
            "progress": {
                "phase": "done",
                "step": 20,
                "total_steps": 20,
                "progress": 1.0,
                "step_progress": 1.0,
                "last_event": {"task": "image_to_video", "step_progress": 1.0},
            },
            "result": {"created": 1, "data": [{"b64_json": base64.b64encode(b"mp4-i2v-job").decode("ascii")}]},
        }

    def post_multipart(self, url, *, headers, data, files, timeout):
        self.calls.append(
            {
                "method": "MULTIPART",
                "url": url,
                "headers": headers,
                "data": dict(data),
                "files": {name: (value[0], value[1], value[2]) for name, value in files.items()},
                "timeout": timeout,
            }
        )
        if url.endswith("/v1/vision/jobs/videos/edits"):
            return {"job_id": "job-i2v"}
        return super().post_multipart(url, headers=headers, data=data, files=files, timeout=timeout)


class _RemoteImageUpscaleJobSender(_RemoteImageSender):
    def __init__(self) -> None:
        super().__init__()
        self.poll_count = 0

    def get(self, url, *, headers, timeout):
        self.calls.append({"method": "GET", "url": url, "headers": headers, "timeout": timeout})
        self.poll_count += 1
        if self.poll_count == 1:
            return {
                "id": "job-upscale",
                "state": "running",
                "progress": {
                    "phase": "denoise",
                    "step": 1,
                    "total_steps": 2,
                    "progress": 0.5,
                    "last_event": {"task": "image_upscale", "step_progress": 0.5},
                },
            }
        if self.poll_count == 2:
            return {
                "id": "job-upscale",
                "state": "running",
                "progress": {
                    "phase": "complete",
                    "step": 2,
                    "total_steps": 2,
                    "progress": 1.0,
                    "last_event": {"task": "image_upscale", "step_progress": 1.0},
                },
            }
        return {
            "id": "job-upscale",
            "state": "succeeded",
            "progress": {
                "phase": "done",
                "step": 2,
                "total_steps": 2,
                "progress": 1.0,
                "last_event": {"task": "image_upscale", "step_progress": 1.0},
            },
            "result": {"created": 1, "data": [{"b64_json": base64.b64encode(b"png-upscale-job").decode("ascii")}]},
        }

    def post_multipart(self, url, *, headers, data, files, timeout):
        self.calls.append(
            {
                "method": "MULTIPART",
                "url": url,
                "headers": headers,
                "data": dict(data),
                "files": {name: (value[0], value[1], value[2]) for name, value in files.items()},
                "timeout": timeout,
            }
        )
        if url.endswith("/v1/vision/jobs/images/upscale"):
            return {"job_id": "job-upscale"}
        return super().post_multipart(url, headers=headers, data=data, files=files, timeout=timeout)


def test_local_image_subprocess_safety_accepts_edit_and_upscale_file_media(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"png-source")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"png-mask")

    assert _is_subprocess_safe_image_specs(
        [{"modality": "image", "task": "image_edit"}],
        [
            {"file_path": str(source), "type": "image", "role": "source"},
            {"file_path": str(mask), "type": "image", "role": "mask"},
        ],
    ) is True
    assert _is_subprocess_safe_image_specs(
        [{"modality": "image", "task": "image_upscale"}],
        [{"file_path": str(source), "type": "image", "role": "source"}],
    ) is True
    assert _is_subprocess_safe_image_specs(
        [{"modality": "image", "task": "image_edit"}],
        [{"file_path": "https://example.com/source.png", "type": "image", "role": "source"}],
    ) is False


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
    _assert_generated_media_calls_have_no_timeout(sender)
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
                "model": "AbstractFramework/flux.2-klein-4b-4bit",
                "format": "png",
            }
        },
    )

    assert sender.calls[0]["url"] == "http://core.test/v1/images/generations"
    _assert_generated_media_calls_have_no_timeout(sender)
    assert sender.calls[0]["json"]["provider"] == "mflux"
    assert sender.calls[0]["json"]["model"] == "AbstractFramework/flux.2-klein-4b-4bit"
    assert not str(sender.calls[0]["json"]["model"]).startswith("diffusers/")


def test_remote_image_output_forwards_batch_seeds_and_lora_adapters() -> None:
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=InMemoryArtifactStore(),
    )

    client.generate(
        prompt="A castle at dawn.",
        params={
            "output": {
                "modality": "image",
                "task": "text_to_image",
                "provider": "mlx-gen",
                "model": "AbstractFramework/qwen-image-2512-8bit",
                "format": "png",
                "count": 2,
                "seeds": [101, 202],
                "lora_adapters": [
                    {"id": "cinematic-lighting", "scale": 0.75},
                    {"id": "ink-outline", "scale": 0.4},
                ],
            }
        },
    )

    call = sender.calls[0]
    assert call["url"] == "http://core.test/v1/images/generations"
    assert call["json"]["n"] == 2
    assert call["json"]["seeds"] == [101, 202]
    assert call["json"]["lora_adapters"] == [
        {"id": "cinematic-lighting", "scale": 0.75},
        {"id": "ink-outline", "scale": 0.4},
    ]


def test_remote_image_output_with_progress_uses_core_job_endpoint() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteImageGenerationJobSender()
    events = []
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="A red cube.",
        params={
            "output": {
                "modality": "image",
                "provider": "mlx-gen",
                "model": "AbstractFramework/flux.2-klein-4b-4bit",
                "format": "png",
                "steps": 2,
            },
            "on_progress": events.append,
            "poll_interval_s": 0.05,
            "trace_metadata": {"run_id": "run-img-job", "node_id": "n-img-job"},
        },
    )

    assert sender.calls[0]["method"] == "POST"
    assert sender.calls[0]["url"] == "http://core.test/v1/vision/jobs/images/generations"
    _assert_generated_media_calls_have_no_timeout(sender)
    assert sender.calls[0]["json"]["provider"] == "mlx-gen"
    assert sender.calls[1]["method"] == "GET"
    assert sender.calls[1]["url"] == "http://core.test/v1/vision/jobs/job-image?consume=true"
    assert events[0]["last_event"]["task"] == "image_generation"
    assert events[-1]["progress"] == 1.0
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-image-job"


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


def test_remote_image_edit_uses_abstractcore_images_edits_endpoint_and_stores_artifact(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-input")
    mask_path = tmp_path / "mask.png"
    mask_path.write_bytes(b"png-mask")
    store = InMemoryArtifactStore()
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="Make the jacket red.",
        media=[
            {"file_path": str(image_path), "type": "image", "role": "source"},
            {"file_path": str(mask_path), "type": "image", "role": "mask"},
        ],
        params={
            "output": {
                "modality": "image",
                "task": "image_edit",
                "provider": "openai-compatible",
                "model": "gpt-image-1",
                "format": "png",
                "size": "1024x1024",
                "quality": "low",
                "extra": {"background": "auto"},
            },
            "base_url": "http://provider.test/v1",
            "trace_metadata": {"run_id": "run-edit", "node_id": "n-edit"},
        },
    )

    call = sender.calls[0]
    assert call["method"] == "MULTIPART"
    assert call["url"] == "http://core.test/openai-compatible/v1/images/edits"
    assert call["data"]["prompt"] == "Make the jacket red."
    assert call["data"]["model"] == "gpt-image-1"
    assert call["data"]["size"] == "1024x1024"
    assert call["data"]["base_url"] == "http://provider.test/v1"
    assert "provider" not in call["data"]
    assert json.loads(call["data"]["extra_json"]) == {"background": "auto", "quality": "low"}
    assert call["files"]["image"][1] == b"png-input"
    assert call["files"]["mask"][1] == b"png-mask"
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-edited"
    assert artifact.metadata.run_id == "run-edit"
    assert artifact.metadata.tags["node_id"] == "n-edit"


def test_remote_image_edit_forwards_batch_seeds_and_lora_adapters(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-input")
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=InMemoryArtifactStore(),
    )

    client.generate(
        prompt="Convert to watercolor.",
        media=[{"file_path": str(image_path), "type": "image", "role": "source"}],
        params={
            "output": {
                "modality": "image",
                "task": "image_edit",
                "provider": "mlx-gen",
                "model": "AbstractFramework/qwen-image-edit-2511-8bit",
                "count": 3,
                "seeds": [1, 2, 3],
                "lora_adapters": [
                    {"id": "watercolor", "scale": 0.8},
                    {"id": "paper-grain", "scale": 0.35},
                ],
            }
        },
    )

    call = sender.calls[0]
    assert call["url"] == "http://core.test/mlx-gen/v1/images/edits"
    assert call["data"]["n"] == "3"
    assert call["data"]["seeds"] == "1,2,3"
    assert json.loads(call["data"]["lora_adapters_json"]) == [
        {"id": "watercolor", "scale": 0.8},
        {"id": "paper-grain", "scale": 0.35},
    ]


def test_remote_image_edit_with_progress_uses_core_job_endpoint(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-input")
    store = InMemoryArtifactStore()
    sender = _RemoteImageEditJobSender()
    events = []
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="Make the jacket red.",
        media=[{"file_path": str(image_path), "type": "image", "role": "source"}],
        params={
            "output": {
                "modality": "image",
                "task": "image_edit",
                "provider": "mlx-gen",
                "model": "AbstractFramework/qwen-image-edit-2511-4bit",
                "format": "png",
                "steps": 2,
            },
            "on_progress": events.append,
            "poll_interval_s": 0.05,
            "trace_metadata": {"run_id": "run-edit-job", "node_id": "n-edit-job"},
        },
    )

    assert sender.calls[0]["method"] == "MULTIPART"
    assert sender.calls[0]["url"] == "http://core.test/v1/vision/jobs/images/edits"
    _assert_generated_media_calls_have_no_timeout(sender)
    assert sender.calls[0]["data"]["provider"] == "mlx-gen"
    assert sender.calls[0]["data"]["model"] == "AbstractFramework/qwen-image-edit-2511-4bit"
    assert sender.calls[1]["method"] == "GET"
    assert sender.calls[1]["url"] == "http://core.test/v1/vision/jobs/job-edit?consume=true"
    assert events[0]["last_event"]["task"] == "image_edit"
    assert events[-1]["progress"] == 1.0
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-edit-job"


def test_remote_video_output_uses_abstractcore_server_endpoint_and_stores_artifact() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="A logo reveal.",
        params={
            "output": {
                "modality": "video",
                "task": "text_to_video",
                "provider": "mlx-gen",
                "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "format": "mp4",
                "frames": 41,
                "fps": 24,
                "steps": 10,
            },
            "trace_metadata": {"run_id": "run-video", "node_id": "n-video"},
        },
    )

    assert sender.calls[0]["method"] == "POST"
    assert sender.calls[0]["url"] == "http://core.test/v1/videos/generations"
    _assert_generated_media_calls_have_no_timeout(sender)
    assert sender.calls[0]["json"]["provider"] == "mlx-gen"
    assert sender.calls[0]["json"]["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert sender.calls[0]["json"]["num_frames"] == 41
    item = out["outputs"]["video"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"mp4-remote"
    assert artifact.metadata.content_type == "video/mp4"
    assert artifact.metadata.run_id == "run-video"
    assert artifact.metadata.tags["node_id"] == "n-video"


def test_remote_video_output_with_progress_uses_core_job_endpoint() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteVideoJobSender()
    events = []
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="A logo reveal.",
        params={
            "output": {
                "modality": "video",
                "task": "text_to_video",
                "provider": "mlx-gen",
                "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "format": "mp4",
            },
            "on_progress": events.append,
            "poll_interval_s": 0.05,
            "trace_metadata": {"run_id": "run-video-job", "node_id": "n-video-job"},
        },
    )

    assert sender.calls[0]["method"] == "POST"
    assert sender.calls[0]["url"] == "http://core.test/v1/vision/jobs/videos/generations"
    _assert_generated_media_calls_have_no_timeout(sender)
    assert sender.calls[1]["method"] == "GET"
    assert sender.calls[1]["url"] == "http://core.test/v1/vision/jobs/job-video?consume=true"
    assert events[0]["phase"] == "denoise"
    assert events[-1]["progress"] == 1.0
    item = out["outputs"]["video"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"mp4-job"


def test_remote_video_output_forwards_batch_flow_shift_seeds_and_lora_adapters() -> None:
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=InMemoryArtifactStore(),
    )

    client.generate(
        prompt="A drone shot over a frozen lake.",
        params={
            "output": {
                "modality": "video",
                "task": "text_to_video",
                "provider": "mlx-gen",
                "model": "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
                "count": 2,
                "seeds": [77, 88],
                "flow_shift": 3.0,
                "lora_adapters": [
                    {"id": "documentary-motion", "scale": 0.6},
                    {"id": "cool-grade", "scale": 0.25},
                ],
            }
        },
    )

    call = sender.calls[0]
    assert call["url"] == "http://core.test/v1/videos/generations"
    assert call["json"]["n"] == 2
    assert call["json"]["seeds"] == [77, 88]
    assert call["json"]["flow_shift"] == 3.0
    assert call["json"]["lora_adapters"] == [
        {"id": "documentary-motion", "scale": 0.6},
        {"id": "cool-grade", "scale": 0.25},
    ]


def test_remote_image_to_video_uses_abstractcore_videos_edits_endpoint_and_stores_artifact(tmp_path: Path) -> None:
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

    out = client.generate(
        prompt="Add a slow camera orbit.",
        media=[{"file_path": str(image_path), "type": "image", "role": "source"}],
        params={
            "output": {
                "modality": "video",
                "task": "image_to_video",
                "provider": "mlx-gen",
                "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "format": "mp4",
                "num_frames": 41,
                "extra": {"motion": "orbit"},
            },
            "trace_metadata": {"run_id": "run-i2v", "node_id": "n-i2v"},
        },
    )

    call = sender.calls[0]
    assert call["method"] == "MULTIPART"
    assert call["url"] == "http://core.test/mlx-gen/v1/videos/edits"
    assert call["data"]["prompt"] == "Add a slow camera orbit."
    assert call["data"]["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert call["data"]["num_frames"] == 41
    assert json.loads(call["data"]["extra_json"]) == {"motion": "orbit"}
    assert call["files"]["image"][1] == b"png-input"
    item = out["outputs"]["video"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"mp4-edited"
    assert artifact.metadata.content_type == "video/mp4"
    assert artifact.metadata.run_id == "run-i2v"
    assert artifact.metadata.tags["node_id"] == "n-i2v"


def test_remote_image_to_video_with_progress_uses_core_job_endpoint_and_step_progress(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-input")
    store = InMemoryArtifactStore()
    sender = _RemoteImageToVideoJobSender()
    events = []
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="Add a slow camera orbit.",
        media=[{"file_path": str(image_path), "type": "image", "role": "source"}],
        params={
            "output": {
                "modality": "video",
                "task": "image_to_video",
                "provider": "mlx-gen",
                "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "format": "mp4",
                "num_frames": 81,
                "steps": 20,
            },
            "on_progress": events.append,
            "poll_interval_s": 0.05,
            "trace_metadata": {"run_id": "run-i2v-job", "node_id": "n-i2v-job"},
        },
    )

    assert sender.calls[0]["method"] == "MULTIPART"
    assert sender.calls[0]["url"] == "http://core.test/v1/vision/jobs/videos/edits"
    _assert_generated_media_calls_have_no_timeout(sender)
    assert sender.calls[1]["method"] == "GET"
    assert sender.calls[1]["url"] == "http://core.test/v1/vision/jobs/job-i2v?consume=true"
    assert events[0]["phase"] == "denoise"
    assert events[0]["step"] == 5
    assert events[0]["total_steps"] == 20
    assert events[0]["step_progress"] == 0.25
    assert events[0]["frame"] == 20
    assert events[0]["total_frames"] == 81
    assert events[0]["frame_progress"] == 20 / 81
    assert events[0]["last_event"]["task"] == "image_to_video"
    assert events[-1]["progress"] == 1.0
    item = out["outputs"]["video"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"mp4-i2v-job"


def test_remote_image_to_video_forwards_batch_flow_shift_seeds_and_lora_adapters(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-input")
    sender = _RemoteImageSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=InMemoryArtifactStore(),
    )

    client.generate(
        prompt="Orbit around the ship.",
        media=[{"file_path": str(image_path), "type": "image", "role": "source"}],
        params={
            "output": {
                "modality": "video",
                "task": "image_to_video",
                "provider": "mlx-gen",
                "model": "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
                "count": 2,
                "seeds": [301, 302],
                "flow_shift": 5.0,
                "lora_adapters": [
                    {"id": "cinematic-camera", "scale": 0.7},
                    {"id": "contrast-pop", "scale": 0.3},
                ],
            }
        },
    )

    call = sender.calls[0]
    assert call["url"] == "http://core.test/mlx-gen/v1/videos/edits"
    assert call["data"]["n"] == "2"
    assert call["data"]["seeds"] == "301,302"
    assert call["data"]["flow_shift"] == 5.0
    assert json.loads(call["data"]["lora_adapters_json"]) == [
        {"id": "cinematic-camera", "scale": 0.7},
        {"id": "contrast-pop", "scale": 0.3},
    ]


def test_remote_image_upscale_uses_abstractcore_images_upscale_endpoint_and_stores_artifact(tmp_path: Path) -> None:
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

    out = client.generate(
        prompt="",
        media=[{"file_path": str(image_path), "type": "image", "role": "source"}],
        params={
            "output": {
                "modality": "image",
                "task": "image_upscale",
                "provider": "mlx-gen",
                "model": "AbstractFramework/seedvr2-3b-8bit",
                "format": "png",
                "scale": "2x",
                "resolution": 2048,
                "softness": 0.25,
                "seed": 2405,
                "quantize": 8,
                "vae_tiling": True,
                "extra": {"tile_overlap": 32},
            },
            "trace_metadata": {"run_id": "run-upscale", "node_id": "n-upscale"},
        },
    )

    call = sender.calls[0]
    assert call["method"] == "MULTIPART"
    assert call["url"] == "http://core.test/mlx-gen/v1/images/upscale"
    assert call["data"]["model"] == "AbstractFramework/seedvr2-3b-8bit"
    assert call["data"]["response_format"] == "b64_json"
    assert call["data"]["scale"] == "2x"
    assert call["data"]["resolution"] == 2048
    assert call["data"]["softness"] == 0.25
    assert call["data"]["seed"] == 2405
    assert call["data"]["quantize"] == 8
    assert call["data"]["vae_tiling"] is True
    assert json.loads(call["data"]["extra_json"]) == {"tile_overlap": 32}
    assert call["files"]["image"][1] == b"png-input"
    item = out["outputs"]["image"][0]
    assert item["task"] == "image_upscale"
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-edited"
    assert artifact.metadata.content_type == "image/png"
    assert artifact.metadata.run_id == "run-upscale"
    assert artifact.metadata.tags["node_id"] == "n-upscale"


def test_remote_image_upscale_with_progress_uses_core_job_endpoint(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-input")
    store = InMemoryArtifactStore()
    sender = _RemoteImageUpscaleJobSender()
    events = []
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="",
        media=[{"file_path": str(image_path), "type": "image", "role": "source"}],
        params={
            "output": {
                "modality": "image",
                "task": "image_upscale",
                "provider": "mlx-gen",
                "model": "AbstractFramework/seedvr2-3b-8bit",
                "scale": "2x",
            },
            "on_progress": events.append,
            "poll_interval_s": 0.05,
            "trace_metadata": {"run_id": "run-upscale-job", "node_id": "n-upscale-job"},
        },
    )

    assert sender.calls[0]["method"] == "MULTIPART"
    assert sender.calls[0]["url"] == "http://core.test/v1/vision/jobs/images/upscale"
    _assert_generated_media_calls_have_no_timeout(sender)
    assert sender.calls[0]["data"]["provider"] == "mlx-gen"
    assert sender.calls[1]["method"] == "GET"
    assert sender.calls[1]["url"] == "http://core.test/v1/vision/jobs/job-upscale?consume=true"
    assert events[0]["last_event"]["task"] == "image_upscale"
    finalizing = [event for event in events if event.get("progress_mode") == "finalizing"]
    assert finalizing
    assert finalizing[0]["phase"] == "finalizing"
    assert finalizing[0]["progress"] < 1.0
    assert finalizing[0]["job_state"] == "running"
    assert events[-1]["progress"] == 1.0
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-upscale-job"


def test_remote_image_upscale_without_route_delegates_default_to_core_job(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"png-input")
    store = InMemoryArtifactStore()
    sender = _RemoteImageUpscaleJobSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="",
        media=[{"file_path": str(image_path), "type": "image", "role": "source"}],
        params={
            "output": {
                "modality": "image",
                "task": "image_upscale",
                "scale": "2x",
            },
            "on_progress": lambda _event: None,
            "poll_interval_s": 0.05,
            "trace_metadata": {"run_id": "run-upscale-default", "node_id": "n-upscale-default"},
        },
    )

    assert sender.calls[0]["method"] == "MULTIPART"
    assert sender.calls[0]["url"] == "http://core.test/v1/vision/jobs/images/upscale"
    _assert_generated_media_calls_have_no_timeout(sender)
    assert "provider" not in sender.calls[0]["data"]
    assert "model" not in sender.calls[0]["data"]
    item = out["outputs"]["image"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"png-upscale-job"


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


class _RemoteMusicSender:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, *, headers, timeout):
        raise AssertionError("not used")

    def post(self, url, *, headers, json, timeout):
        raise AssertionError("post_bytes should be used for binary music")

    def post_bytes(self, url, *, headers, json, timeout):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json, "timeout": timeout})
        return HttpBinaryResponse(content=b"wav-music", headers={"content-type": "audio/wav"})


def test_remote_music_output_uses_music_endpoint_and_stores_artifact() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteMusicSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    out = client.generate(
        prompt="Warm lo-fi piano with brushed drums.",
        params={
            "output": {
                "modality": "music",
                "provider": "acemusic",
                "model": "ace-step",
                "format": "wav",
            },
            "trace_metadata": {"run_id": "run-music", "node_id": "n-music"},
        },
    )

    assert sender.calls[0]["url"] == "http://core.test/v1/audio/music"
    assert sender.calls[0]["json"]["prompt"] == "Warm lo-fi piano with brushed drums."
    assert sender.calls[0]["json"]["provider"] == "acemusic"
    assert sender.calls[0]["json"]["model"] == "ace-step"
    assert sender.calls[0]["json"]["task"] == "text_to_music"
    item = out["outputs"]["music"][0]
    artifact = store.load(item["artifact_id"])
    assert artifact is not None
    assert artifact.content == b"wav-music"
    assert artifact.metadata.content_type == "audio/wav"
    assert artifact.metadata.run_id == "run-music"
    assert artifact.metadata.tags["node_id"] == "n-music"
    assert out["media_provider"] == "acemusic"
    assert out["media_model"] == "ace-step"


def test_remote_music_output_rejects_legacy_backend_fields() -> None:
    store = InMemoryArtifactStore()
    sender = _RemoteMusicSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://core.test",
        model="openai/gpt-4o-mini",
        request_sender=sender,
        artifact_store=store,
    )

    with pytest.raises(ValueError, match="provider.*backend selector"):
        client.generate(
            prompt="Warm lo-fi piano with brushed drums.",
            params={"output": {"modality": "music", "backend": "acemusic", "format": "wav"}},
        )

    assert sender.calls == []


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
