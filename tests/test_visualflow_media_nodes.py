from __future__ import annotations

import pytest

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
            "seed": 1234,
            "guidance_scale": 6.5,
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
    assert output["seed"] == 1234
    assert output["guidance_scale"] == 6.5


def test_generate_image_node_compiles_batch_seed_and_lora_fields() -> None:
    plan = _plan_for_node(
        "generate_image",
        {
            "prompt": "a horse",
            "count": 2,
            "seeds": [11, 22],
            "lora_adapters": [
                {"id": "cinematic-lighting", "scale": 0.7},
                {"id": "ink-outline", "scale": 0.35},
            ],
        },
    )

    output = dict((plan.effect.payload or {}).get("output") or {})
    assert output["count"] == 2
    assert output["seeds"] == [11, 22]
    assert output["lora_adapters"] == [
        {"id": "cinematic-lighting", "scale": 0.7},
        {"id": "ink-outline", "scale": 0.35},
    ]


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
            "seed": 1234,
            "guidance_scale": 6.5,
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
    assert output["seed"] == 1234
    assert output["guidance_scale"] == 6.5
    assert media[0]["$artifact"] == "source-img"
    assert media[0]["type"] == "image"
    assert media[0]["role"] == "source"
    assert media[1]["$artifact"] == "mask-img"
    assert media[1]["role"] == "mask"


def test_image_to_image_alias_compiles_to_image_edit_selector() -> None:
    plan = _plan_for_node(
        "image_to_image",
        {"prompt": "Add clouds.", "source_image": "img-1", "seed": 42, "guidance_scale": 5.5},
    )

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert output["task"] == "image_edit"
    assert output["seed"] == 42
    assert output["guidance_scale"] == 5.5
    assert payload["media"] == [{"type": "image", "role": "source", "$artifact": "img-1"}]


def test_edit_image_node_compiles_batch_seed_and_lora_fields() -> None:
    plan = _plan_for_node(
        "edit_image",
        {
            "prompt": "Add clouds.",
            "source_image": "img-1",
            "count": 3,
            "seeds": [31, 32, 33],
            "lora_adapters": '[{"id":"watercolor","scale":0.8},{"id":"paper-grain","scale":0.25}]',
        },
    )

    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert output["count"] == 3
    assert output["seeds"] == [31, 32, 33]
    assert output["lora_adapters"] == [
        {"id": "watercolor", "scale": 0.8},
        {"id": "paper-grain", "scale": 0.25},
    ]


def test_generate_video_node_compiles_to_llm_call_video_selector() -> None:
    plan = _plan_for_node(
        "generate_video",
        {
            "prompt": "A logo reveal.",
            "video_provider": "mlx-gen",
            "video_model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "format": "mp4",
            "width": 1280,
            "height": 704,
            "frames": 121,
            "fps": 24,
            "steps": 50,
            "seed": 4321,
            "guidance_scale": 5.0,
            "guidance_2": 3.0,
        },
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert payload["prompt"] == "A logo reveal."
    assert "provider" not in payload
    assert "model" not in payload
    assert output["modality"] == "video"
    assert output["task"] == "text_to_video"
    assert output["provider"] == "mlx-gen"
    assert output["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert output["format"] == "mp4"
    assert output["width"] == 1280
    assert output["height"] == 704
    assert output["num_frames"] == 121
    assert output["fps"] == 24
    assert output["steps"] == 50
    assert output["seed"] == 4321
    assert output["guidance_scale"] == 5.0
    assert output["guidance_2"] == 3.0


def test_generate_video_node_compiles_batch_flow_shift_seed_and_lora_fields() -> None:
    plan = _plan_for_node(
        "generate_video",
        {
            "prompt": "A logo reveal.",
            "count": 2,
            "seeds": [101, 202],
            "flow_shift": 3.0,
            "loraAdapters": [
                {"id": "documentary-motion", "scale": 0.6},
                {"id": "cool-grade", "scale": 0.25},
            ],
        },
    )

    output = dict((plan.effect.payload or {}).get("output") or {})
    assert output["count"] == 2
    assert output["seeds"] == [101, 202]
    assert output["flow_shift"] == 3.0
    assert output["lora_adapters"] == [
        {"id": "documentary-motion", "scale": 0.6},
        {"id": "cool-grade", "scale": 0.25},
    ]


def test_upscale_image_node_compiles_to_llm_call_image_upscale_selector() -> None:
    plan = _plan_for_node(
        "upscale_image",
        {
            "image_artifact": "img-1",
            "image_provider": "mlx-gen",
            "image_model": "AbstractFramework/seedvr2-3b-8bit",
            "format": "png",
            "resolution": "2x",
            "softness": 0.25,
            "seed": 2405,
            "quantize": 8,
            "vae_tiling": True,
        },
    )

    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    media = list(payload.get("media") or [])
    assert output["modality"] == "image"
    assert output["task"] == "image_upscale"
    assert output["provider"] == "mlx-gen"
    assert output["model"] == "AbstractFramework/seedvr2-3b-8bit"
    assert "scale" not in output
    assert output["resolution"] == "2x"
    assert output["softness"] == 0.25
    assert output["seed"] == 2405
    assert output["quantize"] == 8
    assert output["vae_tiling"] is True
    assert media == [{"type": "image", "role": "source", "$artifact": "img-1"}]


def test_image_to_video_node_compiles_to_llm_call_video_selector() -> None:
    plan = _plan_for_node(
        "image_to_video",
        {
            "prompt": "Add a slow camera orbit.",
            "source_image": "img-1",
            "video_provider": "mlx-gen",
            "video_model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "frames": 41,
            "seed": 4321,
            "guidance_scale": 5.0,
        },
    )

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert output["modality"] == "video"
    assert output["task"] == "image_to_video"
    assert output["provider"] == "mlx-gen"
    assert output["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert output["num_frames"] == 41
    assert output["seed"] == 4321
    assert output["guidance_scale"] == 5.0
    assert payload["media"] == [{"type": "image", "role": "source", "$artifact": "img-1"}]


def test_image_to_video_node_compiles_batch_flow_shift_seed_and_lora_fields() -> None:
    plan = _plan_for_node(
        "image_to_video",
        {
            "prompt": "Orbit around the subject.",
            "source_image": "img-1",
            "count": 2,
            "seeds": [301, 302],
            "flowShift": 5.0,
            "lora_adapters": [{"id": "cinematic-camera", "scale": 0.7}],
        },
    )

    payload = dict(plan.effect.payload or {})
    output = dict(payload.get("output") or {})
    assert output["count"] == 2
    assert output["seeds"] == [301, 302]
    assert output["flow_shift"] == 5.0
    assert output["lora_adapters"] == [{"id": "cinematic-camera", "scale": 0.7}]


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
            "format": "wav",
            "duration_s": 12,
            "seed": 7,
            "num_inference_steps": 20,
            "guidance_scale": 1.5,
            "instrumental": True,
            "structure_prompt": True,
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
    assert output["duration_s"] == 12.0
    assert output["seed"] == 7
    assert output["num_inference_steps"] == 20
    assert output["guidance_scale"] == 1.5
    assert output["instrumental"] is True
    assert output["structure_prompt"] is True
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


def test_generate_music_node_rejects_legacy_music_backend_field() -> None:
    with pytest.raises(ValueError, match="music_provider"):
        _plan_for_node(
            "generate_music",
            {
                "prompt": "Warm lo-fi piano with brushed drums.",
                "music_backend": "acemusic",
            },
        )


def test_generate_music_node_preserves_false_structure_prompt_bool() -> None:
    plan = _plan_for_node(
        "generate_music",
        {"prompt": "a beat", "structure_prompt": True},
        {"structure_prompt": False},
    )

    assert plan.effect is not None
    output = dict((plan.effect.payload or {}).get("output") or {})
    assert output["structure_prompt"] is False


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
