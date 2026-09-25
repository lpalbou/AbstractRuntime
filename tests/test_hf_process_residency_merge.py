"""The runtime residency listing/eject must reflect PROCESS-level truth for
the HuggingFace provider too (mission MEM2, 2026-09-25).

Same incident shape as MLX, different mechanism: HuggingFace instances do not
share weights, so a boot-time summarizer / override client / old runtime that
built its own provider for the same model is a SECOND FULL COPY. The pool's
eject freed only its own instance and answered "not loaded" while the other
copy stayed resident (measured: 251 MB of SmolLM2 on MPS, ~1 GB of llama.cpp
buffers). `MultiLocalAbstractCoreLLMClient` now folds
`abstractcore.providers.hf_residency` into its listing and drives the
process-wide eject on unload, exactly as it does for MLX.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient


@pytest.fixture(autouse=True)
def _no_real_backends(monkeypatch):
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [])
    monkeypatch.setattr(llm_mod, "_mlx_process_eject", lambda model: {"ok": True, "holders_unloaded": [], "residual": None})
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [])
    monkeypatch.setattr(llm_mod, "_hf_process_eject", lambda model: {"ok": True, "holders_unloaded": [], "residual": None})
    yield


def _held_row(model: str, holders: int, held: int, lane: str = "transformers") -> Dict[str, Any]:
    return {
        "lane": lane,
        "backend": "huggingface",
        "model_path": f"/hub/models--{model.replace('/', '--')}/snapshots/x",
        "models": [model],
        "holders": holders,
        "copies": holders,
        "shared_weights": False,
        "weights_bytes": held - 100,
        "cache_bytes": 100,
        "held_bytes": held,
        "weights_alive": True,
    }


def test_hf_model_held_only_by_unreachable_holders_is_listed_resident(monkeypatch):
    model = "vendor/held-hf"
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [_held_row(model, 2, 5_000_000)])
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")

    result = client.list_model_residency(task="text_generation")
    rows = {(m["provider"], m["model"]): m for m in result["models"]}
    assert ("huggingface", model) in rows, "process-held HF model must appear in the listing"
    row = rows[("huggingface", model)]
    assert row["resident"] is True and row["loaded"] is True
    assert row["provider_state"] == "resident_via_other_holders"
    assert row["source"] == "abstractcore.provider.huggingface.process"
    assert row["held_bytes"] == 5_000_000 and row["process_holders"] == 2
    assert row["est_weights_bytes"] == 5_000_000 - 100
    assert any("full copy" in w for w in row.get("warnings", []))


def test_provider_filter_keeps_lanes_apart(monkeypatch):
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [_held_row("vendor/held-hf", 1, 10)])
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [
        {"lane": "mlx_lm", "model_path": "/x", "models": ["vendor/held-mlx"], "holders": 1, "weights_bytes": 5,
         "cache_bytes": 0, "held_bytes": 5, "weights_alive": True}])
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")
    only_hf = client.list_model_residency(task="text_generation", provider="huggingface")
    assert [(m["provider"], m["model"]) for m in only_hf["models"]] == [("huggingface", "vendor/held-hf")]
    only_mlx = client.list_model_residency(task="text_generation", provider="mlx")
    assert ("huggingface", "vendor/held-hf") not in {(m["provider"], m["model"]) for m in only_mlx["models"]}


def test_cold_hf_pool_row_flips_to_resident_when_another_copy_is_alive(monkeypatch):
    model = "vendor/default-hf"

    class _ColdProvider:
        provider = "huggingface"

        def get_model_residency(self, **kwargs):
            return {"task": "text_generation", "provider": "huggingface", "model": model,
                    "provider_residency_verified": True, "provider_resident": False,
                    "loaded": False, "state": "not_loaded", "source": "abstractcore.provider.huggingface"}

    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")
    from types import SimpleNamespace
    client._clients[("huggingface", model)] = SimpleNamespace(_llm=_ColdProvider())
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [_held_row(model, 1, 3_000_000, lane="gguf")])

    result = client.list_model_residency(task="text_generation")
    row = next(m for m in result["models"] if m["model"] == model)
    assert row["resident"] is True and row["state"] == "provider_loaded"
    assert row["provider_state"] == "resident_via_other_holders"
    assert row["held_bytes"] == 3_000_000 and row["process_lane"] == "gguf"


def test_unload_of_process_held_hf_model_runs_the_hf_process_eject(monkeypatch):
    """MUTANT guard: an unload path that only ejects MLX (`provider_s == "mlx"`)
    leaves the HF copies resident -> this goes RED."""
    model = "vendor/held-hf"
    ejects: List[str] = []
    state = {"held": True}

    def fake_eject(m):
        ejects.append(m)
        state["held"] = False
        return {"ok": True, "holders_unloaded": [{"id": 1, "lane": "transformers"}, {"id": 2, "lane": "transformers"}],
                "residual": None, "cache_cleared": True}

    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows",
                        lambda: [_held_row(model, 2, 5_000_000)] if state["held"] else [])
    monkeypatch.setattr(llm_mod, "_hf_process_eject", fake_eject)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")

    result = client.unload_model_residency(provider="huggingface", model=model)
    assert ejects == [model], "unload must drive the HF process-wide eject"
    assert result["ok"] is True and result["unloaded"] is True
    assert result["process_eject"]["holders_unloaded"]
    assert not any(m["model"] == model for m in client.list_model_residency(task="text_generation")["models"])


def test_unload_reports_residual_when_the_hf_process_eject_cannot_free(monkeypatch):
    model = "vendor/stuck-hf"
    monkeypatch.setattr(llm_mod, "_hf_process_residency_rows", lambda: [_held_row(model, 1, 4_000_000)])
    monkeypatch.setattr(llm_mod, "_hf_process_eject", lambda m: {
        "ok": False, "holders_unloaded": [], "holders_refused": [{"error": "busy"}],
        "residual": {"holders": 1, "held_bytes": 4_000_000}})
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")

    result = client.unload_model_residency(provider="huggingface", model=model)
    assert result["ok"] is False
    assert "still resident in this process" in str(result.get("error", ""))


class _FakeHFProvider:
    """A pooled HuggingFace provider whose own unload succeeds -- the sibling
    copies (summarizer / override client) are what the process eject frees."""

    provider = "huggingface"

    def __init__(self, model: str):
        self.model = model
        self.loaded = True

    def get_model_residency(self, **kwargs):
        return {"task": "text_generation", "provider": "huggingface", "model": self.model,
                "provider_residency_verified": True, "provider_resident": self.loaded,
                "loaded": self.loaded, "state": "loaded" if self.loaded else "not_loaded",
                "source": "abstractcore.provider.huggingface"}

    def unload_model(self, _name):
        self.loaded = False


def test_unload_of_a_pooled_hf_client_still_ejects_every_other_copy(monkeypatch):
    """MUTANT guard for the pooled path: after the pool's own instance unloads,
    the process-wide HF eject must run (an eject that only ran for MLX left
    the summarizer's copy resident -> RED)."""
    model = "vendor/pooled-hf"
    ejects: List[str] = []
    monkeypatch.setattr(llm_mod, "_hf_process_eject", lambda m: (ejects.append(m), {
        "ok": True, "holders_unloaded": [{"id": 2, "lane": "transformers"}], "residual": None, "cache_cleared": True})[1])
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")
    from types import SimpleNamespace
    provider = _FakeHFProvider(model)
    client._clients[("huggingface", model)] = SimpleNamespace(_llm=provider, _drop_prompt_cache_client_state=lambda: None)

    result = client.unload_model_residency(provider="huggingface", model=model)
    assert provider.loaded is False
    assert ejects == [model], "the process-wide HF eject must run after the pool's own unload"
    assert result["ok"] is True and result["unloaded"] is True
    assert result["process_eject"]["holders_unloaded"]
