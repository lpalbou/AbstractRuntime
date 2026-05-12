from __future__ import annotations

from abstractruntime.core.models import EffectType, RunState, RunStatus
from abstractruntime.visualflow_compiler import compile_visualflow


def _plan_for_node(node_type: str, effect_config: dict) -> object:
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
    return spec.nodes["node"](run, None)


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
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-4.1-mini"
    assert output["modality"] == "image"
    assert output["task"] == "image_generation"
    assert output["format"] == "png"
    assert output["provider"] == "abstractvision"
    assert output["model"] == "flux-test"
    assert output["width"] == 512
    assert output["height"] == 512
    assert output["steps"] == 8


def test_generate_image_legacy_provider_model_stay_in_output_spec() -> None:
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
    assert output["provider"] == "abstractvision"
    assert output["model"] == "flux-test"


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
    assert details["max_duration_s"] == 12.0
