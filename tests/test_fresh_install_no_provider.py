"""Fresh-install guard: a runtime with NO provider configured must construct.

Release gap 1 (gateway c5878, 2026-07-27): a brand-new install has no
provider anywhere. The pooled client used to crash at CONSTRUCTION
("Unknown provider: "), so the shipped catalog could never load and
first-run users saw an empty gateway. Now: construction succeeds with a
warning; calls that bring their own provider work; calls with none fail
with a message that says what to configure.
"""

from __future__ import annotations

import pytest

from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient


def test_blank_provider_and_model_construct_without_crash() -> None:
    client = MultiLocalAbstractCoreLLMClient(provider="", model="")
    assert client._default_client is None


def test_generate_without_any_provider_asks_for_configuration() -> None:
    client = MultiLocalAbstractCoreLLMClient(provider="", model="")
    with pytest.raises(ValueError) as exc:
        client.generate(prompt="hello")
    msg = str(exc.value)
    assert "no provider configured" in msg
    assert "gateway defaults" in msg  # the message names where to fix it


def test_capability_lookups_ask_for_configuration_not_none_crash() -> None:
    client = MultiLocalAbstractCoreLLMClient(provider="", model="")
    with pytest.raises(ValueError) as exc:
        client.get_model_capabilities()
    assert "no provider configured" in str(exc.value)


def test_configured_construction_still_builds_eagerly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard only fires on the both-blank fresh-install shape: with a
    configured pair the default client still builds at construction
    (stubbed here so the test never talks to a live provider)."""
    built = []

    def _stub(self, provider, model, *, llm_kwargs_override=None):
        built.append((provider, model))
        class _C:
            _llm = None
        return _C()

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_get_client", _stub)
    client = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="some-model")
    assert client._default_client is not None
    assert built == [("lmstudio", "some-model")]


def test_create_local_runtime_constructs_with_no_provider() -> None:
    """Release gap 1, second site (gateway c5898): the factory's eager
    capability probe died on the configuration error one line after the
    client guard - the runtime never existed, so the catalog still could
    not load. With a blank pair the probe is skipped (loud log) and the
    runtime constructs; capabilities resolve per call once a provider
    arrives."""
    from abstractruntime.integrations.abstractcore.factory import create_local_runtime

    runtime = create_local_runtime(provider="", model="")
    assert runtime is not None
    # And a run that brings its own provider is the working path; a bare
    # call still gets the actionable message (pinned above).
