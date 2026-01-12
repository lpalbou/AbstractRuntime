def test_llm_call_use_context_appends_turn_into_active_context():
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler

    class DummyLLM:
        def generate(self, *, prompt, messages, system_prompt, tools, params):
            return {"content": "OK"}

    handler = make_llm_call_handler(llm=DummyLLM())

    run = RunState.new(
        workflow_id="wf",
        entry_node="node-1",
        vars={"context": {}},
    )
    run.current_node = "node-1"

    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "Hello",
            "include_context": True,
            "params": {},
        },
        result_key="_temp.llm",
    )

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    ctx = run.vars.get("context")
    assert isinstance(ctx, dict)
    messages = ctx.get("messages")
    assert isinstance(messages, list)
    assert [m.get("role") for m in messages] == ["user", "assistant"]
    assert messages[0].get("content") == "Hello"
    assert messages[1].get("content") == "OK"
