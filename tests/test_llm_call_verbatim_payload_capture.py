from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler


class _StubLLM:
    def generate(self, **kwargs):
        return {"content": "ok", "metadata": {}}


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

