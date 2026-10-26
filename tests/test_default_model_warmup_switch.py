"""A process that serves no model never loads the default one (split gateway, 2026-10-26).

`MultiLocalAbstractCoreLLMClient` builds its default client at construction (a warm-up: for a
local MLX default that LOADS the weights, 15 GB for a 27B model). The gateway's split API process
(`serve --no-runner`) builds the same runtime as the runner process to serve reads, so it loaded
a second copy at boot. `set_default_model_warmup(False)` turns the warm-up off for the process:
construction and default switches build nothing, the first call builds the client.
"""

from __future__ import annotations

import pytest

from abstractruntime.integrations.abstractcore import llm_client as lc
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient


@pytest.fixture()
def built(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def _stub(self, provider, model, *, llm_kwargs_override=None):
        calls.append((provider, model))

        class _C:
            _llm = None

        return _C()

    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_get_client", _stub)
    prev = lc.default_model_warmup()
    yield calls
    lc.set_default_model_warmup(prev)


def test_warmup_off_builds_nothing_at_construction_or_switch(built) -> None:
    lc.set_default_model_warmup(False)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="mlx-community/Qwen-27B-4bit")
    assert client._default_client is None
    assert built == [], "no model load in a process that serves none"
    client.set_default_provider_model(provider="mlx", model="mlx-community/Other-9B-4bit")
    assert client._default_client is None and built == []


def test_warmup_on_is_unchanged(built) -> None:
    lc.set_default_model_warmup(True)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="mlx-community/Qwen-27B-4bit")
    assert client._default_client is not None
    assert built == [("mlx", "mlx-community/Qwen-27B-4bit")]
