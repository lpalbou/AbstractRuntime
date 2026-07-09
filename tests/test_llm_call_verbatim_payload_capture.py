from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler


class _StubLLM:
    def generate(self, **kwargs):
        return {"content": "ok", "metadata": {}}


class _ResolvedRouteStubLLM:
    def generate(self, **kwargs):
        return {
            "content": "ok",
            "metadata": {
                "_resolved_generate_route": {
                    "request": {"text": "hello", "has_messages": False, "message_count": 0, "media_count": 0, "media_types": []},
                    "outputs": [{"modality": "image", "task": "text_to_image", "provider": "mlx-gen", "model": "z-image"}],
                    "text_route": {
                        "route_key": "input.text",
                        "provider": "openai",
                        "model": "gpt-5",
                        "field_sources": {"provider": "explicit", "model": "explicit", "route": "explicit"},
                    },
                    "input_routes": [{"route_key": "input.text"}],
                    "output_routes": [{"route_key": "output.image.text_to_image", "provider": "mlx-gen", "model": "z-image"}],
                    "reasoning": "medium",
                    "reasoning_source": "explicit",
                }
            },
        }


def test_llm_call_handler_attaches_runtime_observability_payload() -> None:
    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={"prompt": "hello", "params": {"temperature": 0.2}},
    )

    handler = make_llm_call_handler(llm=_StubLLM())
    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    assert isinstance(outcome.result, dict)
    meta = outcome.result.get("metadata")
    assert isinstance(meta, dict)
    obs = meta.get("_runtime_observability")
    assert isinstance(obs, dict)
    captured = obs.get("llm_generate_kwargs")
    assert isinstance(captured, dict)
    assert captured["prompt"] == "hello"
    assert isinstance(captured.get("params"), dict)
    assert "trace_metadata" in captured["params"]


def test_llm_call_handler_attaches_runtime_resolved_action_metadata() -> None:
    run = RunState.new(workflow_id="wf", entry_node="n1", vars={})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={"prompt": "hello", "params": {"temperature": 0.2}},
    )

    handler = make_llm_call_handler(llm=_ResolvedRouteStubLLM())
    outcome = handler(run, effect, None)

    assert outcome.status == "completed"
    meta = outcome.result.get("metadata")
    assert isinstance(meta, dict)
    action = meta.get("_runtime_resolved_action")
    assert isinstance(action, dict)
    assert action["action_id"] == "generated_image"
    assert action["modality"] == "image"
    assert action["task"] == "text_to_image"
    assert action["normalized_output"] == [
        {"modality": "image", "task": "text_to_image", "provider": "mlx-gen", "model": "z-image"}
    ]
