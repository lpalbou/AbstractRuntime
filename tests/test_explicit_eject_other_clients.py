"""An explicit eject (console / tray / CLI -> `unload_model_residency`) is
process-wide: it frees the model from EVERY holder. It must not take a model
another client in the same process has LOCKED (another user's service, an
entity runtime), unless forced; other clients that merely pool it are ejected
and counted. Fakes only. MUTANT: drop `_explicit_eject_guard` -> RED."""
from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import Any, List

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient


class _Provider:
    def __init__(self, provider: str, model: str):
        self.provider, self.model, self.loaded = provider, model, True

    def get_model_residency(self, **kwargs):
        return {"task": "text_generation", "provider": self.provider, "model": self.model,
                "provider_residency_verified": True, "provider_resident": self.loaded, "loaded": self.loaded,
                "state": "loaded" if self.loaded else "not_loaded", "source": "fake"}

    def unload_model(self, name):
        self.loaded = False


@pytest.fixture
def clients(monkeypatch):
    gc.collect()
    ejects: List[tuple] = []
    monkeypatch.setattr(llm_mod, "_process_eject_for",
                        lambda p, m: ejects.append((p, m)) or {"ok": True, "holders_unloaded": [{"id": 1}], "residual": None})
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [])
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [])
    monkeypatch.setattr(MultiLocalAbstractCoreLLMClient, "_create_client",
                        lambda self, p, m, *, llm_kwargs_override=None: SimpleNamespace(
                            _llm=_Provider(p, m), _drop_prompt_cache_client_state=lambda: None))
    a = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/A")
    b = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/B")
    a._get_client("mlx", "vendor/X")
    b._get_client("mlx", "vendor/X")
    return a, b, ejects


def test_a_model_locked_by_another_client_is_refused(clients):
    a, b, ejects = clients
    b._locked_model_residency.add(("mlx", "vendor/X"))
    out = a.unload_model_residency(provider="mlx", model="vendor/X")
    assert out["ok"] is False and out["refused"] == "model_locked_by_other_client" and out["status_code"] == 409
    assert out["locked_by"] and out["locked_by"][0]["kind"] in {"pool", "lock"}
    assert ejects == [] and a._clients[("mlx", "vendor/X")]._llm.loaded is True, "nothing unloaded"


def test_force_ejects_over_another_clients_lock_and_says_so(clients):
    a, b, ejects = clients
    b._locked_model_residency.add(("mlx", "vendor/X"))
    out = a.unload_model_residency(provider="mlx", model="vendor/X", force=True)
    assert out["ok"] is True and ejects == [("mlx", "vendor/X")]
    assert out["forced_over_other_client_locks"]


def test_a_model_other_clients_only_pool_is_ejected_and_reported(clients):
    a, b, ejects = clients
    out = a.unload_model_residency(provider="mlx", model="vendor/X")
    assert out["ok"] is True and ejects == [("mlx", "vendor/X")]
    assert out["ejected_from_other_clients"] == 1


def test_by_runtime_id_and_another_spelling_the_lock_still_holds(clients):
    a, b, ejects = clients
    b._locked_model_residency.add(("mlx", "vendor/X"))
    out = a.unload_model_residency(runtime_id="local:text_generation:mlx:VENDOR/x")
    assert out.get("refused") == "model_locked_by_other_client" and ejects == []


def test_remote_providers_are_not_guarded(clients):
    a, b, _ = clients
    b._get_client("ollama", "g")
    b._locked_model_residency.add(("ollama", "g"))
    out = a.unload_model_residency(provider="ollama", model="g")
    assert out.get("refused") is None
