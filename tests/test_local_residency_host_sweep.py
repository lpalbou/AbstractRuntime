"""Local residency listings merge core's provider-server sweep (ADR 0007:
the sweep IS provider-owned truth being relayed; pool/client records win the
dedup and absorb missing memory fields, sweep-only rows are appended tagged
`source: "provider_server"` with no invented task label).

Dedup/filter semantics come from the CANONICAL `abstractcore.utils.residency`
helpers (`SWEEP_PROVIDERS`, `normalize_sweep_model`, `sweep_models_match`) —
the same rules core's `/acore/models/loaded` uses.

The suite-wide autouse fixture stubs the live sweep to []; these tests
monkeypatch their own fakes on top.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict, List

import abstractruntime.integrations.abstractcore.llm_client as llm_mod
from abstractruntime.integrations.abstractcore.llm_client import (
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
)

# Captured at test-module import time, BEFORE the autouse conftest stub
# replaces the module attribute, so the failure-containment test exercises
# the GENUINE guard inside the real function (not the stub).
_REAL_SWEEP_HOST_LOADED_MODELS = llm_mod._sweep_host_loaded_models


class _ResidentProvider:
    def __init__(self, *, provider: str, model: str, loaded: bool = True) -> None:
        self.provider = provider
        self.model = model
        self.loaded = loaded

    def get_model_residency(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "task": "text_generation",
            "provider": self.provider,
            "model": str(kwargs.get("model") or self.model),
            "provider_residency_verified": True,
            "provider_resident": bool(self.loaded),
            "loaded": bool(self.loaded),
            "state": "loaded" if self.loaded else "not_loaded",
            "source": "abstractcore.provider.test",
        }


def _local_client(*, provider_name: str, model: str, loaded: bool = True) -> LocalAbstractCoreLLMClient:
    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    client._provider = provider_name  # type: ignore[attr-defined]
    client._model = model  # type: ignore[attr-defined]
    client._llm_kwargs = {}  # type: ignore[attr-defined]
    client._llm = _ResidentProvider(provider=provider_name, model=model, loaded=loaded)  # type: ignore[attr-defined]
    return client


class _DummyLocal:
    def __init__(self, *, provider: str, model: str, llm_kwargs: Dict[str, Any], artifact_store: Any) -> None:
        _ = llm_kwargs, artifact_store
        self._provider = provider
        self._model = model
        self._llm = _ResidentProvider(provider=provider, model=model, loaded=True)

    def get_model_capabilities(self) -> Dict[str, Any]:
        return {}


def test_local_listing_absorbs_sweep_memory_fields_with_latest_alias_dedup(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_mod,
        "_sweep_host_loaded_models",
        lambda: [
            {
                "provider": "ollama",
                "model": "llama3:latest",
                "resident": True,
                "loaded": True,
                "size_bytes": 111,
                "size_vram_bytes": 99,
                "source": "provider_server",
            }
        ],
    )
    client = _local_client(provider_name="ollama", model="llama3")

    result = client.list_model_residency(task="text_generation")

    assert result["ok"] is True
    assert len(result["models"]) == 1  # the alias deduped into the client record
    record = result["models"][0]
    assert record["source"] == "abstractruntime.local"  # client record wins
    assert record["size_bytes"] == 111
    assert record["size_vram_bytes"] == 99


def test_lmstudio_variant_dedups_against_the_server_key(monkeypatch) -> None:
    """Pool client `qwen3-4b` and sweep key `qwen/qwen3-4b` are the SAME
    resident model (LM Studio's own substring alias rule) — two rows would
    double-count its memory."""
    monkeypatch.setattr(
        llm_mod,
        "_sweep_host_loaded_models",
        lambda: [
            {
                "provider": "lmstudio",
                "model": "qwen/qwen3-4b",
                "provider_instance_ids": ["qwen/qwen3-4b"],
                "resident": True,
                "loaded": True,
                "size_bytes": 222,
                "source": "provider_server",
            }
        ],
    )
    client = _local_client(provider_name="lmstudio", model="qwen3-4b")

    result = client.list_model_residency(task="text_generation")

    assert len(result["models"]) == 1
    record = result["models"][0]
    assert record["model"] == "qwen3-4b"
    assert record["size_bytes"] == 222  # absorbed, not double-counted


def test_multilocal_listing_appends_sweep_only_rows_and_normalizes_sizes(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    sweep_rows: List[Dict[str, Any]] = [
        {
            "provider": "lmstudio",
            "model": "qwen/qwen3-4b",
            "resident": True,
            "loaded": True,
            "size_bytes": 222,
            "source": "provider_server",
        },
        {
            # Raw ollama extras only: the merge normalizes them.
            "provider": "ollama",
            "model": "granite4:small",
            "resident": True,
            "loaded": True,
            "size": 333,
            "size_vram": 300,
            "expires_at": "2026-08-27T00:00:00Z",
        },
    ]
    monkeypatch.setattr(llm_mod, "_sweep_host_loaded_models", lambda: [dict(r) for r in sweep_rows])
    client = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="qwen/qwen3-4b")

    result = client.list_model_residency(task="text_generation")

    assert result["ok"] is True
    by_identity = {(m["provider"], m["model"]): m for m in result["models"]}
    assert set(by_identity) == {("lmstudio", "qwen/qwen3-4b"), ("ollama", "granite4:small")}

    pool_record = by_identity[("lmstudio", "qwen/qwen3-4b")]
    assert pool_record["source"] == "abstractruntime.local"  # pool record shape unchanged
    assert pool_record["runtime_cached"] is True
    assert pool_record["size_bytes"] == 222  # absorbed from the sweep

    sweep_only = by_identity[("ollama", "granite4:small")]
    assert sweep_only["source"] == "provider_server"
    assert sweep_only["loaded"] is True
    assert sweep_only["resident"] is True
    assert "task" not in sweep_only  # relayed, never inferred
    assert sweep_only["size_bytes"] == 333
    assert sweep_only["size_vram_bytes"] == 300
    assert sweep_only["size"] == 333  # originals kept
    assert sweep_only["expires_at"] == "2026-08-27T00:00:00Z"


def test_model_filter_matches_sweep_aliases(monkeypatch) -> None:
    """`model="qwen3"` must match the sweep row `qwen3:latest` — the filter
    uses the same normalization as the dedup."""
    monkeypatch.setattr(
        llm_mod,
        "_sweep_host_loaded_models",
        lambda: [
            {"provider": "ollama", "model": "qwen3:latest", "resident": True, "loaded": True},
            {"provider": "ollama", "model": "granite4:small", "resident": True, "loaded": True},
        ],
    )
    client = _local_client(provider_name="lmstudio", model="unrelated")

    result = client.list_model_residency(task="text_generation", model="qwen3")

    assert [m["model"] for m in result["models"]] == ["qwen3:latest"]


def test_multilocal_listing_applies_provider_filter_to_sweep_rows(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "LocalAbstractCoreLLMClient", _DummyLocal)
    monkeypatch.setattr(
        llm_mod,
        "_sweep_host_loaded_models",
        lambda: [
            {"provider": "ollama", "model": "granite4:small", "resident": True, "loaded": True},
        ],
    )
    client = MultiLocalAbstractCoreLLMClient(provider="lmstudio", model="qwen/qwen3-4b")

    result = client.list_model_residency(task="text_generation", provider="lmstudio")

    assert {m["provider"] for m in result["models"]} == {"lmstudio"}


def test_non_sweep_provider_filter_short_circuits_the_live_sweep(monkeypatch) -> None:
    """A `provider="mlx"` listing can never be answered by the sweep: the
    live HTTP probes must not run at all."""

    def _must_not_be_called() -> List[Dict[str, Any]]:
        raise AssertionError("the live sweep must be short-circuited for non-sweep providers")

    monkeypatch.setattr(llm_mod, "_sweep_host_loaded_models", _must_not_be_called)
    client = _local_client(provider_name="mlx", model="qwen")

    result = client.list_model_residency(task="text_generation", provider="mlx")

    assert result["ok"] is True
    assert len(result["models"]) == 1


def test_all_task_listing_does_not_count_untasked_sweep_rows_as_text_generation(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_mod,
        "_sweep_host_loaded_models",
        lambda: [
            {"provider": "ollama", "model": "granite4:small", "resident": True, "loaded": True},
        ],
    )
    client = _local_client(provider_name="mlx", model="qwen")
    # Capability plugins are out of scope here: a core with no capability
    # facades contributes no media rows.
    client._capability_residency_core = SimpleNamespace(voice=None, audio=None, music=None, vision=None)  # type: ignore[attr-defined]
    client._capability_residency_core_lock = threading.Lock()  # type: ignore[attr-defined]

    result = client.list_model_residency()

    identities = {(m.get("provider"), m.get("model")) for m in result["models"]}
    assert identities == {("mlx", "qwen"), ("ollama", "granite4:small")}
    # The untasked sweep row rides in the listing but is not counted as a
    # verified text_generation runtime.
    assert result["diagnostics"]["task_counts"]["text_generation"] == 1


def test_sweep_failure_never_fails_the_listing(monkeypatch) -> None:
    calls: List[int] = []

    def _raise(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        calls.append(1)
        raise RuntimeError("sweep exploded")

    # Install the GENUINE helper (captured at module import, before the
    # autouse hermetic stub) so its real try/except is what contains the
    # raising core sweep.
    monkeypatch.setattr(llm_mod, "_sweep_host_loaded_models", _REAL_SWEEP_HOST_LOADED_MODELS)
    import abstractcore.utils.residency as core_residency

    monkeypatch.setattr(core_residency, "sweep_loaded_models", _raise)

    client = _local_client(provider_name="ollama", model="llama3")
    result = client.list_model_residency(task="text_generation")

    assert calls  # the real sweep path was exercised, not a stub
    assert result["ok"] is True
    assert len(result["models"]) == 1
