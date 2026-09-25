"""The runtime residency listing/eject must reflect PROCESS-level MLX truth.

The 2026-09-25 incident: a gateway's own instance answered "not loaded" after
its eject, while the weights were still alive in the process, held by instances
the runtime pool cannot reach (a boot-time chat summarizer, a per-request
override client, an old runtime after a bundle reload, another principal's
service). The listing said "nothing loaded" over ~92 GB of live MLX buffers,
and the eject route froze nothing.

`MultiLocalAbstractCoreLLMClient` now folds `abstractcore.providers.
mlx_residency` into its text listing and drives a process-wide eject on unload.
These tests stub that core surface (no real MLX) and assert:
  * a model held ONLY by unreachable holders is APPENDED as resident;
  * a pool row whose own instance is cold but whose weights are still held is
    flipped to resident with `provider_state == "resident_via_other_holders"`;
  * unloading such a model runs the process eject and reports the freed/residual.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient


@pytest.fixture(autouse=True)
def _no_real_mlx(monkeypatch):
    """Default: no in-process MLX residency and an inert eject; tests opt in."""
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [])
    monkeypatch.setattr(llm_mod, "_mlx_process_eject", lambda model: {"ok": True, "holders_unloaded": [], "residual": None})
    yield


def _held_row(model: str, holders: int, held: int) -> Dict[str, Any]:
    return {
        "lane": "mlx_vlm",
        "model_path": f"/hub/models--{model.replace('/', '--')}/snapshots/x",
        "models": [model],
        "holders": holders,
        "weights_bytes": held - 100,
        "cache_bytes": 100,
        "held_bytes": held,
        "weights_alive": True,
    }


def test_model_held_only_by_unreachable_holders_is_listed_resident(monkeypatch):
    model = "vendor/held-model"
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [_held_row(model, 2, 5_000_000)])
    # A pool built for a DIFFERENT default; the held model is not in _clients.
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")

    result = client.list_model_residency(task="text_generation")
    rows = {(m["provider"], m["model"]): m for m in result["models"]}
    assert ("mlx", model) in rows, "process-held model must appear in the listing"
    row = rows[("mlx", model)]
    assert row["resident"] is True and row["loaded"] is True
    assert row["provider_state"] == "resident_via_other_holders"
    assert row["source"] == "abstractcore.provider.mlx.process"
    assert row["held_bytes"] == 5_000_000 and row["process_holders"] == 2
    assert any("outside this runtime's pool" in w for w in row.get("warnings", []))


def test_cold_pool_row_flips_to_resident_when_weights_still_held(monkeypatch):
    """The default pair's own instance is cold, but its weights are still held
    by another instance in the process: the row must not read 'not loaded'."""
    model = "vendor/default-model"

    class _ColdProvider:
        provider = "mlx"

        def get_model_residency(self, **kwargs):
            return {
                "task": "text_generation", "provider": "mlx", "model": model,
                "provider_residency_verified": True, "provider_resident": False,
                "loaded": False, "state": "not_loaded", "source": "abstractcore.provider.mlx",
            }

    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model=model)
    # Seed the pool with a cold client for the default pair.
    from types import SimpleNamespace
    client._clients[("mlx", model)] = SimpleNamespace(_llm=_ColdProvider())
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [_held_row(model, 1, 3_000_000)])

    result = client.list_model_residency(task="text_generation")
    row = next(m for m in result["models"] if m["model"] == model)
    assert row["resident"] is True and row["state"] == "provider_loaded"
    assert row["provider_state"] == "resident_via_other_holders"
    assert row["held_bytes"] == 3_000_000
    assert any("still hold them" in w for w in row.get("warnings", []))


def test_unload_of_process_held_model_runs_process_eject(monkeypatch):
    model = "vendor/held-model"
    ejects: List[str] = []
    state = {"held": True}  # the eject frees the weights, so the rows go empty

    def fake_eject(m):
        ejects.append(m)
        state["held"] = False
        return {"ok": True, "holders_unloaded": [{"id": 1}, {"id": 2}], "residual": None, "cache_cleared": True}

    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows",
                        lambda: [_held_row(model, 2, 5_000_000)] if state["held"] else [])
    monkeypatch.setattr(llm_mod, "_mlx_process_eject", fake_eject)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")

    result = client.unload_model_residency(provider="mlx", model=model)
    assert ejects == [model], "unload must drive the process-wide eject"
    assert result["ok"] is True and result["unloaded"] is True
    assert "process_eject" in result and result["process_eject"]["holders_unloaded"]


def test_unload_reports_residual_when_process_eject_cannot_free(monkeypatch):
    model = "vendor/stuck-model"

    def stuck_eject(m):
        return {"ok": False, "holders_unloaded": [], "holders_refused": [{"error": "busy"}],
                "residual": {"holders": 1, "held_bytes": 4_000_000}}

    # Row stays held before AND after (eject could not free it).
    monkeypatch.setattr(llm_mod, "_mlx_process_residency_rows", lambda: [_held_row(model, 1, 4_000_000)])
    monkeypatch.setattr(llm_mod, "_mlx_process_eject", stuck_eject)
    client = MultiLocalAbstractCoreLLMClient(provider="mlx", model="vendor/other")

    result = client.unload_model_residency(provider="mlx", model=model)
    assert result["ok"] is False
    assert "still resident in this process" in str(result.get("error", ""))
