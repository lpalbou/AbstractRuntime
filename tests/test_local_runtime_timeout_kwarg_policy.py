from __future__ import annotations


def test_create_local_runtime_injects_timeout_kwarg_by_default(monkeypatch) -> None:
    """Runtime orchestration passes a timeout kwarg unless the caller overrides it."""
    from abstractruntime.integrations.abstractcore import factory

    captured: dict[str, object] = {}

    class DummyLLMClient:
        def __init__(self, *, provider: str, model: str, llm_kwargs: dict):  # type: ignore[no-untyped-def]
            captured["provider"] = provider
            captured["model"] = model
            captured["llm_kwargs"] = dict(llm_kwargs or {})
            self._llm = object()

        def get_model_capabilities(self):  # type: ignore[no-untyped-def]
            return {}

    class DummySummarizer:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

    class DummyTools:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

    monkeypatch.setattr(factory, "MultiLocalAbstractCoreLLMClient", DummyLLMClient)
    monkeypatch.setattr(factory, "AbstractCoreChatSummarizer", DummySummarizer)
    monkeypatch.setattr(factory, "AbstractCoreToolExecutor", DummyTools)
    monkeypatch.setattr(factory, "build_effect_handlers", lambda **kwargs: {})

    factory.create_local_runtime(provider="mlx", model="mlx-community/gpt-oss-20b-MXFP4-Q8")

    llm_kwargs = captured.get("llm_kwargs")
    assert isinstance(llm_kwargs, dict)
    assert "timeout" in llm_kwargs
    assert llm_kwargs["timeout"] is not None
