"""A DEFAULT is a default: hosts must be able to re-point it while running.

The execution-host default provider/model was resolved once at host
construction and frozen there -- in the pooled LLM client AND in RuntimeConfig.
An operator who changed it in the console saw no effect until the process
restarted (reproduced live 2026-07-31). These pin the two setters that make a
live change possible, and the invariant that a refresh touches ONLY the
default: per-call pins and already-started runs are never rewritten.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    StepPlan,
    WorkflowSpec,
)
from abstractruntime.core.config import RuntimeConfig
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.llm_client import _split_connection_scoped_llm_kwargs


def _runtime(config: RuntimeConfig) -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), config=config)


def test_runtime_default_provider_model_is_repointable() -> None:
    rt = _runtime(RuntimeConfig(provider="lmstudio", model="old-model", model_capabilities={"tool_support": "native"}))

    assert rt.set_default_provider_model(provider="ollama", model="new-model", model_capabilities={"tool_support": "prompted"})
    assert rt.config.provider == "ollama"
    assert rt.config.model == "new-model"
    assert rt.config.model_capabilities == {"tool_support": "prompted"}

    # Idempotent: an unchanged refresh reports no change.
    assert rt.set_default_provider_model(provider="ollama", model="new-model", model_capabilities={"tool_support": "prompted"}) is False


def test_repointed_default_reaches_new_runs_only() -> None:
    """A started run keeps the default it was stamped with; only later runs move."""
    rt = _runtime(RuntimeConfig(provider="lmstudio", model="old-model"))

    def node(run, ctx) -> StepPlan:
        del ctx, run
        return StepPlan(node_id="n", complete_output={})

    workflow = WorkflowSpec(workflow_id="default_refresh_test", entry_node="n", nodes={"n": node})

    first = rt.get_state(rt.start(workflow=workflow))
    assert first.vars["_runtime"]["model"] == "old-model"

    rt.set_default_provider_model(provider="lmstudio", model="new-model")

    second = rt.get_state(rt.start(workflow=workflow))
    assert second.vars["_runtime"]["model"] == "new-model"
    # The earlier run's durable vars are untouched.
    assert rt.get_state(first.run_id).vars["_runtime"]["model"] == "old-model"


def test_caller_supplied_runtime_pin_survives_the_default_seed() -> None:
    """The seed is `setdefault`: an app override is never clobbered."""
    rt = _runtime(RuntimeConfig(provider="lmstudio", model="default-model"))

    def node(run, ctx) -> StepPlan:
        del ctx, run
        return StepPlan(node_id="n", complete_output={})

    workflow = WorkflowSpec(workflow_id="default_pin_test", entry_node="n", nodes={"n": node})
    run_id = rt.start(workflow=workflow, vars={"_runtime": {"provider": "openai", "model": "app-pinned"}})
    state = rt.get_state(run_id)
    assert state.vars["_runtime"]["provider"] == "openai"
    assert state.vars["_runtime"]["model"] == "app-pinned"


class _FakeCoreLLM:
    def __init__(self, provider: str, model: str, **kwargs: Any) -> None:
        self.provider = provider
        self.model = model
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.kwargs = dict(kwargs)


def test_pool_default_refresh_replaces_the_connection_half(monkeypatch: Any) -> None:
    """ROUTING INVARIANT: a new default must NOT inherit the old endpoint.

    base_url/api_key name WHERE a provider is reached and WITH WHAT credential.
    Carrying them across a default change is exactly the misroute the
    2026-07-31 isolation fix closed -- every pin served by the previous
    endpoint's models.
    """
    import abstractruntime.integrations.abstractcore.llm_client as mod

    built: list[Dict[str, Any]] = []

    class _Client:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Optional[Dict[str, Any]] = None, **_ignored: Any) -> None:
            self._provider = provider
            self._model = model
            self._llm_kwargs = dict(llm_kwargs or {})
            self._llm = _FakeCoreLLM(provider, model, **self._llm_kwargs)
            built.append({"provider": provider, "model": model, **self._llm_kwargs})

        def set_on_token(self, _cb: Any) -> None:
            return None

    monkeypatch.setattr(mod, "LocalAbstractCoreLLMClient", _Client)

    pool = mod.MultiLocalAbstractCoreLLMClient(
        provider="openai-compatible",
        model="relay-model",
        llm_kwargs={"base_url": "http://relay/v1", "api_key": "relay-key", "timeout": 30.0},
    )
    assert built[-1]["base_url"] == "http://relay/v1"

    changed = pool.set_default_provider_model(provider="lmstudio", model="local-model", llm_kwargs={})
    assert changed is True

    shared, connection = _split_connection_scoped_llm_kwargs(pool._llm_kwargs)
    assert connection == {}, "the previous endpoint/credential must not survive a default change"
    # Provider-agnostic knobs the caller did not mention are preserved.
    assert shared.get("timeout") == 30.0
    assert built[-1]["provider"] == "lmstudio"
    assert built[-1]["model"] == "local-model"
    assert "base_url" not in built[-1]
    assert "api_key" not in built[-1]

    # Pooled clients built under the OLD default are evicted, not reused.
    assert pool._clients and list(pool._clients) == [("lmstudio", "local-model")]


def test_pool_default_refresh_is_a_noop_when_nothing_moved(monkeypatch: Any) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as mod

    class _Client:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Optional[Dict[str, Any]] = None, **_ignored: Any) -> None:
            self._provider = provider
            self._model = model
            self._llm_kwargs = dict(llm_kwargs or {})
            self._llm = _FakeCoreLLM(provider, model, **self._llm_kwargs)

        def set_on_token(self, _cb: Any) -> None:
            return None

    monkeypatch.setattr(mod, "LocalAbstractCoreLLMClient", _Client)
    pool = mod.MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="m", llm_kwargs={"timeout": 5.0})
    before = pool._default_client

    assert pool.set_default_provider_model(provider="lmstudio", model="m", llm_kwargs={}) is False
    assert pool._default_client is before, "a no-op refresh must not rebuild the default client"


def test_pool_capability_defaults_refresh_keeps_the_endpoint(monkeypatch: Any) -> None:
    """Refreshing only the capability routes must not drop the default endpoint."""
    import abstractruntime.integrations.abstractcore.llm_client as mod

    class _Client:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Optional[Dict[str, Any]] = None, **_ignored: Any) -> None:
            self._provider = provider
            self._model = model
            self._llm_kwargs = dict(llm_kwargs or {})
            self._llm = _FakeCoreLLM(provider, model, **self._llm_kwargs)

        def set_on_token(self, _cb: Any) -> None:
            return None

    monkeypatch.setattr(mod, "LocalAbstractCoreLLMClient", _Client)
    pool = mod.MultiLocalAbstractCoreLLMClient(
        provider="openai-compatible",
        model="relay-model",
        llm_kwargs={"base_url": "http://relay/v1", "api_key": "relay-key"},
    )

    changed = pool.set_capability_defaults(
        {"routes": [{"key": "output.voice", "provider": "supertonic", "model": "supertonic-3"}]}
    )
    assert changed is True
    assert pool._llm_kwargs.get("base_url") == "http://relay/v1"
    assert pool._llm_kwargs.get("api_key") == "relay-key"
    assert pool._capability_defaults["output.voice"]["provider"] == "supertonic"
