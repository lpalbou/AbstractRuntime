def test_create_local_runtime_defaults_timeouts_from_abstractcore_config(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import factory as rt_factory

    captured: dict[str, object] = {}

    class FakeCfg:
        def get_default_timeout(self) -> float:
            return 123.0

        def get_tool_timeout(self) -> float:
            return 45.0

    def fake_get_config_manager() -> FakeCfg:
        return FakeCfg()

    # AbstractCore config manager is consulted only when explicit overrides are absent.
    import abstractcore.config.manager as cfg_manager

    monkeypatch.setattr(cfg_manager, "get_config_manager", fake_get_config_manager)

    class FakeLLMClient:
        def __init__(self, *, provider: str, model: str, llm_kwargs=None):
            captured["provider"] = provider
            captured["model"] = model
            captured["llm_kwargs"] = dict(llm_kwargs or {})
            self._llm = object()

        def get_model_capabilities(self):
            return {"max_tokens": 8192}

    class FakeTools:
        def __init__(self, *, timeout_s: float):
            captured["tool_timeout_init"] = timeout_s

        def set_timeout_s(self, timeout_s: float) -> None:
            captured["tool_timeout_set"] = timeout_s

    class FakeSummarizer:
        def __init__(self, llm, *, max_tokens: int, max_output_tokens: int):
            captured["summarizer_llm"] = llm
            captured["summarizer_max_tokens"] = max_tokens
            captured["summarizer_max_output_tokens"] = max_output_tokens

    monkeypatch.setattr(rt_factory, "MultiLocalAbstractCoreLLMClient", FakeLLMClient)
    monkeypatch.setattr(rt_factory, "AbstractCoreToolExecutor", FakeTools)
    monkeypatch.setattr(rt_factory, "AbstractCoreChatSummarizer", FakeSummarizer)
    monkeypatch.setattr(rt_factory, "build_effect_handlers", lambda **_kwargs: {})

    rt_factory.create_local_runtime(provider="lmstudio", model="qwen/qwen3-next-80b")

    llm_kwargs = captured.get("llm_kwargs")
    assert isinstance(llm_kwargs, dict)
    assert llm_kwargs.get("timeout") == 123.0

    assert captured.get("tool_timeout_init") == 45.0
    assert captured.get("tool_timeout_set") == 45.0


def test_create_local_runtime_respects_explicit_overrides(monkeypatch) -> None:
    from abstractruntime.integrations.abstractcore import factory as rt_factory

    captured: dict[str, object] = {}

    class FakeLLMClient:
        def __init__(self, *, provider: str, model: str, llm_kwargs=None):
            captured["llm_kwargs"] = dict(llm_kwargs or {})
            self._llm = object()

        def get_model_capabilities(self):
            return {"max_tokens": 8192}

    class FakeTools:
        def __init__(self, *, timeout_s: float):
            captured["tool_timeout_init"] = timeout_s

        def set_timeout_s(self, timeout_s: float) -> None:
            captured["tool_timeout_set"] = timeout_s

    class FakeSummarizer:
        def __init__(self, llm, *, max_tokens: int, max_output_tokens: int):
            self._llm = llm

    monkeypatch.setattr(rt_factory, "MultiLocalAbstractCoreLLMClient", FakeLLMClient)
    monkeypatch.setattr(rt_factory, "AbstractCoreToolExecutor", FakeTools)
    monkeypatch.setattr(rt_factory, "AbstractCoreChatSummarizer", FakeSummarizer)
    monkeypatch.setattr(rt_factory, "build_effect_handlers", lambda **_kwargs: {})

    rt_factory.create_local_runtime(
        provider="lmstudio",
        model="qwen/qwen3-next-80b",
        llm_kwargs={"timeout": None},
        tool_timeout_s=321.0,
    )

    llm_kwargs = captured.get("llm_kwargs")
    assert isinstance(llm_kwargs, dict)
    assert "timeout" in llm_kwargs and llm_kwargs["timeout"] is None

    assert captured.get("tool_timeout_init") == 321.0
    assert captured.get("tool_timeout_set") == 321.0
