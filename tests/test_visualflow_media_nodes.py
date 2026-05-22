from __future__ import annotations

from abstractruntime.core.models import EffectType, RunState, RunStatus
from abstractruntime.visualflow_compiler import compile_visualflow


def _plan_for_node(node_type: str, effect_config: dict, input_data: dict | None = None) -> object:
    spec = compile_visualflow(
        {
            "id": f"vf_{node_type}",
            "name": f"vf_{node_type}",
            "entryNode": "node",
            "nodes": [
                {
                    "id": "node",
                    "type": node_type,
                    "data": {"nodeType": node_type, "effectConfig": effect_config},
                }
            ],
            "edges": [],
        }
    )
    run = RunState(
        run_id="run",
        workflow_id=str(spec.workflow_id),
        status=RunStatus.RUNNING,
        current_node="node",
        vars={"_temp": {}},
    )
    if isinstance(input_data, dict):
        run.vars["_last_output"] = dict(input_data)
    return spec.nodes["node"](run, input_data)


def test_generate_image_node_compiles_to_llm_call_output_selector() -> None:
    plan = _plan_for_node(
        "generate_image",
        {
            "prompt": "a horse",
            "runtime_provider": "openai",
            "runtime_model": "gpt-4.1-mini",
            "image_provider": "abstractvision",
            "image_model": "flux-test",
            "format": "png",
            "width": 512,
            "height": 512,
            "steps": 8,
        },
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert payload["prompt"] == "a horse"
    assert "provider" not in payload
    assert "model" not in payload
    assert output["modality"] == "image"
    assert output["task"] == "image_generation"
    assert output["format"] == "png"
    assert output["provider"] == "abstractvision"
    assert output["model"] == "flux-test"
    assert output["width"] == 512
    assert output["height"] == 512
    assert output["steps"] == 8


def test_generate_image_legacy_provider_model_are_not_media_fallbacks() -> None:
    plan = _plan_for_node(
        "generate_image",
        {
            "prompt": "a horse",
            "provider": "abstractvision",
            "model": "flux-test",
        },
    )

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert "provider" not in payload
    assert "model" not in payload
    assert "provider" not in output
    assert "model" not in output


def test_generate_image_does_not_treat_runtime_provider_input_as_image_provider() -> None:
    plan = _plan_for_node(
        "generate_image",
        {"prompt": "a horse", "image_model": "gpt-image-1-mini"},
        {"provider": "lmstudio", "model": "chat-model"},
    )

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert "provider" not in output
    assert output["model"] == "gpt-image-1-mini"


def test_edit_image_node_compiles_to_llm_call_image_edit_selector() -> None:
    plan = _plan_for_node(
        "edit_image",
        {
            "prompt": "Make the jacket red.",
            "image_artifact": {"$artifact": "source-img", "content_type": "image/png"},
            "mask_artifact": "mask-img",
            "image_provider": "openai-compatible",
            "image_model": "gpt-image-1",
            "format": "png",
            "size": "1024x1024",
            "strength": 0.65,
        },
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    media = list(payload.get("media") or [])
    assert payload["prompt"] == "Make the jacket red."
    assert output["modality"] == "image"
    assert output["task"] == "image_edit"
    assert output["provider"] == "openai-compatible"
    assert output["model"] == "gpt-image-1"
    assert output["size"] == "1024x1024"
    assert output["strength"] == 0.65
    assert media[0]["$artifact"] == "source-img"
    assert media[0]["type"] == "image"
    assert media[0]["role"] == "source"
    assert media[1]["$artifact"] == "mask-img"
    assert media[1]["role"] == "mask"


def test_image_to_image_alias_compiles_to_image_edit_selector() -> None:
    plan = _plan_for_node(
        "image_to_image",
        {"prompt": "Add clouds.", "source_image": "img-1"},
    )

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert output["task"] == "image_edit"
    assert payload["media"] == [{"type": "image", "role": "source", "$artifact": "img-1"}]


def test_generate_voice_node_compiles_to_llm_call_tts_selector() -> None:
    plan = _plan_for_node(
        "generate_voice",
        {
            "text": "hello",
            "voice": "clone:laurent",
            "profile": "piper/en_US",
            "tts_model": "en_US-lessac-medium.onnx",
            "format": "wav",
            "speed": 1.1,
        },
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert payload["prompt"] == "hello"
    assert "model" not in payload
    assert output["modality"] == "voice"
    assert output["task"] == "tts"
    assert output["voice"] == "clone:laurent"
    assert output["profile"] == "piper/en_US"
    assert output["model"] == "en_US-lessac-medium.onnx"
    assert output["format"] == "wav"
    assert output["speed"] == 1.1


def test_generate_music_node_compiles_to_llm_call_music_selector() -> None:
    plan = _plan_for_node(
        "generate_music",
        {
            "prompt": "Warm lo-fi piano with brushed drums.",
            "music_provider": "acemusic",
            "music_model": "ace-step",
            "music_backend": "acemusic",
            "format": "wav",
            "duration_s": 12,
            "seed": 7,
            "num_inference_steps": 20,
            "guidance_scale": 1.5,
            "instrumental": True,
            "lyrics": "no lyrics",
            "vocal_language": "en",
            "negative_prompt": "distortion",
            "sample_rate": 44100,
            "bpm": 92,
            "keyscale": "C minor",
            "timesignature": "4/4",
            "composition_plan": {"sections": ["intro", "loop"]},
            "positive_styles": ["lo-fi", "piano"],
            "negative_styles": ["metal"],
            "planning": False,
        },
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert payload["prompt"] == "Warm lo-fi piano with brushed drums."
    assert "provider" not in payload
    assert "model" not in payload
    assert output["modality"] == "music"
    assert output["task"] == "music_generation"
    assert output["format"] == "wav"
    assert output["provider"] == "acemusic"
    assert output["model"] == "ace-step"
    assert output["backend"] == "acemusic"
    assert output["duration_s"] == 12.0
    assert output["seed"] == 7
    assert output["num_inference_steps"] == 20
    assert output["guidance_scale"] == 1.5
    assert output["instrumental"] is True
    assert output["lyrics"] == "no lyrics"
    assert output["vocal_language"] == "en"
    assert output["negative_prompt"] == "distortion"
    assert output["sample_rate"] == 44100
    assert output["bpm"] == 92.0
    assert output["keyscale"] == "C minor"
    assert output["timesignature"] == "4/4"
    assert output["composition_plan"] == {"sections": ["intro", "loop"]}
    assert output["positive_styles"] == ["lo-fi", "piano"]
    assert output["negative_styles"] == ["metal"]
    assert output["planning"] is False


def test_generate_music_legacy_provider_model_are_not_media_fallbacks() -> None:
    plan = _plan_for_node(
        "generate_music",
        {
            "prompt": "a beat",
            "provider": "acemusic",
            "model": "ace-step",
        },
    )

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert "provider" not in payload
    assert "model" not in payload
    assert "provider" not in output
    assert "model" not in output


def test_generate_music_does_not_treat_runtime_provider_input_as_music_provider() -> None:
    plan = _plan_for_node(
        "generate_music",
        {"prompt": "a beat", "music_model": "ace-step"},
        {"provider": "lmstudio", "model": "chat-model"},
    )

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert "provider" not in output
    assert output["model"] == "ace-step"


def test_transcribe_audio_node_compiles_to_llm_call_media_selector() -> None:
    plan = _plan_for_node(
        "transcribe_audio",
        {
            "audio_artifact": {"$artifact": "audio-1", "content_type": "audio/wav"},
            "language": "en",
            "stt_model": "whisper-test",
            "format": "json",
            "temperature": 0,
        },
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert payload["media"] == [{"$artifact": "audio-1", "content_type": "audio/wav"}]
    assert "model" not in payload
    assert output["modality"] == "text"
    assert output["task"] == "transcription"
    assert output["language"] == "en"
    assert output["model"] == "whisper-test"
    assert output["format"] == "json"
    assert output["temperature"] == 0.0


def test_listen_voice_node_compiles_to_voice_wait_event() -> None:
    plan = _plan_for_node(
        "listen_voice",
        {
            "prompt": "Say your answer",
            "wait_key": "voice_answer",
            "language": "fr",
            "stt_provider": "openai",
            "stt_model": "whisper-test",
            "max_duration_s": 12,
        },
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.WAIT_EVENT
    payload = dict(plan.effect.payload or {})
    details = dict(payload.get("details") or {})
    assert payload["wait_key"] == "voice_answer"
    assert payload["prompt"] == "Say your answer"
    assert payload["allow_free_text"] is True
    assert details["input_mode"] == "voice"
    assert details["language"] == "fr"
    assert details["provider"] == "openai"
    assert details["model"] == "whisper-test"
    assert details["max_duration_s"] == 12.0
