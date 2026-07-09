"""create_local_runtime defaults to a RetryPolicy (backlog 0217).

LLM_CALL is side-effect-free and gets retries (transient provider failures should not fail a
durable run). TOOL_CALLS are NOT retried by default (a partially-applied side-effecting batch
must not be blindly re-executed; idempotency only makes REPLAY of a completed result safe).

The LLM client / tool executor are stubbed so the test does not touch a live provider.
"""
from __future__ import annotations

from abstractruntime.core.models import Effect, EffectType
from abstractruntime.core.policy import NoRetryPolicy, RetryPolicy


def _effect(kind: EffectType) -> Effect:
    return Effect(type=kind, payload={}, result_key="_temp.x")


def _stub_factory(monkeypatch):
    from abstractruntime.integrations.abstractcore import factory as rt_factory

    class FakeLLMClient:
        def __init__(self, *, provider, model, llm_kwargs=None, artifact_store=None, **kwargs):
            self._llm = object()

        def get_model_capabilities(self):
            return {"max_tokens": 8192}

    class FakeTools:
        def __init__(self, *, timeout_s: float):
            pass

        def set_timeout_s(self, timeout_s: float) -> None:
            pass

    class FakeSummarizer:
        def __init__(self, llm, *, max_tokens: int, max_output_tokens: int):
            pass

    monkeypatch.setattr(rt_factory, "MultiLocalAbstractCoreLLMClient", FakeLLMClient)
    monkeypatch.setattr(rt_factory, "AbstractCoreToolExecutor", FakeTools)
    monkeypatch.setattr(rt_factory, "AbstractCoreChatSummarizer", FakeSummarizer)
    monkeypatch.setattr(rt_factory, "build_effect_handlers", lambda **_kwargs: {})
    return rt_factory


def test_factory_default_policy_is_retry(monkeypatch) -> None:
    rt_factory = _stub_factory(monkeypatch)
    rt = rt_factory.create_local_runtime(provider="lmstudio", model="qwen/qwen3-next-80b")
    policy = rt.effect_policy
    assert isinstance(policy, RetryPolicy)
    assert policy.max_attempts(_effect(EffectType.LLM_CALL)) == 3
    assert policy.max_attempts(_effect(EffectType.TOOL_CALLS)) == 1


def test_factory_explicit_policy_is_respected(monkeypatch) -> None:
    rt_factory = _stub_factory(monkeypatch)
    rt = rt_factory.create_local_runtime(
        provider="lmstudio", model="qwen/qwen3-next-80b", effect_policy=NoRetryPolicy()
    )
    assert isinstance(rt.effect_policy, NoRetryPolicy)
    assert rt.effect_policy.max_attempts(_effect(EffectType.LLM_CALL)) == 1
