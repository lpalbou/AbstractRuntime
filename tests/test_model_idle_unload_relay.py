"""Backlog 1115 (runtime part): the host's idle-unload time reaches AbstractCore's process residency
(ADR 0012) through one `set_` call, and local residency rows say when a model was last used and when it
unloads. Core owns the reaper and its rules; the runtime relays, it never decides."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from abstractcore.providers import process_residency

from abstractruntime.integrations.abstractcore import llm_client as lc
from abstractruntime.integrations.abstractcore.host_facade import AbstractCoreHostFacade

pytestmark = pytest.mark.basic


@pytest.fixture(autouse=True)
def _restore_idle_unload():
    before = process_residency.get_idle_unload()
    yield
    process_residency.set_idle_unload(before)


def test_the_setter_reaches_core_and_returns_the_value_in_effect() -> None:
    out = lc._set_model_idle_unload(600)
    assert out == {"ok": True, "operation": "set_model_idle_unload", "idle_unload_s": 600.0}
    assert process_residency.get_idle_unload() == 600.0
    assert lc._set_model_idle_unload(0)["idle_unload_s"] is None and process_residency.get_idle_unload() is None


def test_a_refused_value_is_reported_and_leaves_the_setting_unchanged() -> None:
    process_residency.set_idle_unload(900)
    out = lc._set_model_idle_unload(-5)
    assert out["ok"] is False and "0 or more seconds" in out["error"] and out["idle_unload_s"] == 900.0
    assert process_residency.get_idle_unload() == 900.0


def test_both_local_clients_and_the_facade_expose_it_under_a_set_name() -> None:
    for cls in (lc.LocalAbstractCoreLLMClient, lc.MultiLocalAbstractCoreLLMClient):
        assert callable(getattr(cls, "set_model_idle_unload", None)), cls.__name__
    client = lc.MultiLocalAbstractCoreLLMClient.__new__(lc.MultiLocalAbstractCoreLLMClient)
    facade = AbstractCoreHostFacade.__new__(AbstractCoreHostFacade)
    facade._client = client  # type: ignore[attr-defined]
    assert facade.set_model_idle_unload(1800)["idle_unload_s"] == 1800.0
    assert process_residency.get_idle_unload() == 1800.0
    facade._client = object()  # type: ignore[attr-defined]
    assert facade.set_model_idle_unload(60)["supported"] is False


def test_a_local_row_carries_last_used_and_the_unload_time(monkeypatch) -> None:
    states: List[Dict[str, Any]] = [
        {"backend": "mlx", "provider": "mlx", "model": "m", "held_bytes": 1, "holders": 1,
         "last_used_at": 1000.0, "in_use": False, "locked_by": [], "idle_unload_at": 4600.0},
        {"backend": "embeddings", "provider": "huggingface", "task": "embedding", "model": "m",
         "last_used_at": 1.0, "idle_unload_at": 2.0},
    ]
    monkeypatch.setattr(process_residency, "resident_model_states", lambda: states)
    record = {"provider": "mlx", "model": "m"}
    lc._stamp_local_record_idle_unload(record)
    assert record["last_used_at"] == 1000.0 and record["idle_unload_at"] == 4600.0
    assert record["in_use"] is False and record["locked_by"] == []
    assert record["idle_unload_s"] == process_residency.get_idle_unload()
    other = {"provider": "huggingface", "model": "m"}  # the embedding row of the same name is not a text row
    lc._stamp_local_record_idle_unload(other)
    assert "last_used_at" not in other and other["idle_unload_s"] == process_residency.get_idle_unload()
    assert "in_use" not in other and "locked_by" not in other


def test_in_use_and_locked_by_pass_through_unchanged(monkeypatch) -> None:
    # A model in use, or kept loaded by another account, has no idle_unload_at: without these two keys a
    # host could not say why. The runtime relays Core's values as-is; it never interprets them.
    states: List[Dict[str, Any]] = [
        {"backend": "mlx", "provider": "mlx", "model": "busy", "last_used_at": 5.0, "in_use": True,
         "locked_by": [], "idle_unload_at": None},
        {"backend": "mlx", "provider": "mlx", "model": "kept", "last_used_at": 6.0, "in_use": False,
         "locked_by": ["alice", "gateway"], "idle_unload_at": None},
    ]
    monkeypatch.setattr(process_residency, "resident_model_states", lambda: states)
    busy, kept = {"provider": "mlx", "model": "busy"}, {"provider": "mlx", "model": "kept"}
    lc._stamp_local_record_idle_unload(busy)
    lc._stamp_local_record_idle_unload(kept)
    assert (busy["in_use"], busy["locked_by"], busy["idle_unload_at"]) == (True, [], None)
    assert (kept["in_use"], kept["locked_by"], kept["idle_unload_at"]) == (False, ["alice", "gateway"], None)


def test_the_per_load_ttl_warning_names_the_idle_unload_instead_of_never() -> None:
    process_residency.set_idle_unload(3600)
    warnings, unsupported = lc._load_option_warnings(
        provider="mlx", model="m", requested={"ttl_s": 120}, provider_load_result=None, provider_called=True)
    assert unsupported == ["ttl_s"]
    assert "the idle unload setting applies (60 min unused, never while it is kept loaded)" in warnings[0]
    assert "stays loaded until it is unloaded" not in warnings[0]
    process_residency.set_idle_unload(None)
    warnings, _ = lc._load_option_warnings(
        provider="mlx", model="m", requested={"ttl_s": 120}, provider_load_result=None, provider_called=True)
    assert "the idle unload is off" in warnings[0]


def test_a_resident_in_process_row_is_stamped_and_a_non_resident_one_is_not(monkeypatch) -> None:
    state = {"backend": "mlx", "provider": "mlx", "model": "m", "last_used_at": 10.0, "idle_unload_at": 3610.0,
             "in_use": False, "locked_by": []}
    monkeypatch.setattr(process_residency, "resident_model_states", lambda: [state])
    monkeypatch.setattr(lc, "_local_provider_residency_claim", lambda **kw: {
        "provider_residency_verified": True, "provider_resident": True,
        "provider_residency_source": "test", "provider_state": "loaded"})
    row = lc._local_residency_record(provider="mlx", model="m")
    assert row["resident"] is True and row["last_used_at"] == 10.0 and row["idle_unload_at"] == 3610.0
    assert row["in_use"] is False and row["locked_by"] == []
    monkeypatch.setattr(lc, "_local_provider_residency_claim", lambda **kw: {
        "provider_residency_verified": True, "provider_resident": False,
        "provider_residency_source": "test", "provider_state": "unloaded"})
    assert "idle_unload_at" not in lc._local_residency_record(provider="mlx", model="m")
