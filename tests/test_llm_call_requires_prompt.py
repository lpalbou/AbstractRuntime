from __future__ import annotations


def test_llm_call_accepts_prompt() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
            self.calls.append(
                {
                    "prompt": prompt,
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "media": media,
                    "tools": tools,
                    "params": params,
                }
            )
            return {"content": "OK"}

    def _run_with(payload: dict[str, object]) -> tuple[str, str, int]:
        llm = DummyLLM()
        handler = make_llm_call_handler(llm=llm)
        run = RunState.new(workflow_id="wf", entry_node="node-1", vars={})
        run.current_node = "node-1"
        effect = Effect(type=EffectType.LLM_CALL, payload=payload, result_key="_temp.llm")
        outcome = handler(run, effect, None)
        prompt = str(llm.calls[0]["prompt"] or "") if llm.calls else ""
        return outcome.status, prompt, len(llm.calls)

    status, prompt, called = _run_with({"prompt": "Hello", "params": {}})
    assert status == "completed"
    assert prompt == "Hello"
    assert called == 1


def test_llm_call_rejects_request_alias_when_prompt_missing() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class DummyLLM:
        def generate(self, *, prompt, messages, system_prompt, media, tools, params):
            raise AssertionError("should not be called")

    handler = make_llm_call_handler(llm=DummyLLM())
    run = RunState.new(workflow_id="wf", entry_node="node-1", vars={})
    run.current_node = "node-1"

    effect = Effect(type=EffectType.LLM_CALL, payload={"request": "Hello", "params": {}}, result_key="_temp.llm")
    outcome = handler(run, effect, None)
    assert outcome.status == "failed"
    assert isinstance(outcome.error, str)
    assert outcome.error == "llm_call requires payload.prompt or payload.messages"
