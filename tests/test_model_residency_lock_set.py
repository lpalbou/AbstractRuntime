"""The host's stored "Keep loaded" set is one `set_` call (`set_model_residency_locks`), so a host that
rebuilds its client (the gateway's model host after a restart) replays it and the locks come back,
with the locked models loaded again (a lock requires a resident model)."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import pytest

from abstractruntime.integrations.abstractcore import llm_client as lc
from abstractruntime.integrations.abstractcore.host_facade import AbstractCoreHostFacade


class _Pool:
    """The three calls `_apply_model_residency_locks` makes, with a client-side lock set."""

    def __init__(self, resident: Set[Tuple[str, str]], unloadable: Set[Tuple[str, str]] = frozenset()) -> None:
        self.resident = set(resident)
        self.unloadable = set(unloadable)
        self._locked_model_residency: Set[Tuple[str, str]] = set()
        self.calls: List[Tuple[str, str, str]] = []

    def lock_model_residency(self, provider: str, model: str) -> Dict[str, Any]:
        self.calls.append(("lock", provider, model))
        if (provider, model) not in self.resident:
            return {"ok": False, "error": "model_not_resident"}
        self._locked_model_residency.add((provider, model))
        return {"ok": True}

    def unlock_model_residency(self, provider: str, model: str) -> Dict[str, Any]:
        self.calls.append(("unlock", provider, model))
        self._locked_model_residency.discard((provider, model))
        return {"ok": True}

    def load_model_residency(self, *, task: str, provider: str, model: str) -> Dict[str, Any]:
        self.calls.append(("load", provider, model))
        if (provider, model) in self.unloadable:
            return {"ok": False, "error": "model_not_downloaded"}
        self.resident.add((provider, model))
        return {"ok": True}


def test_resident_models_are_locked_missing_ones_loaded_then_locked_others_unlocked() -> None:
    pool = _Pool(resident={("mlx", "a"), ("mlx", "old")})
    pool._locked_model_residency.add(("mlx", "old"))
    out = lc._apply_model_residency_locks(pool, [{"provider": "MLX", "model": "a"}, ("huggingface", "b")])
    assert out["ok"] is True
    assert pool._locked_model_residency == {("mlx", "a"), ("huggingface", "b")}
    assert ("unlock", "mlx", "old") in pool.calls and ("load", "huggingface", "b") in pool.calls
    actions = {(r["provider"], r["model"]): r["action"] for r in out["results"]}
    assert actions == {("mlx", "old"): "unlocked", ("mlx", "a"): "locked", ("huggingface", "b"): "loaded_and_locked"}


def test_replaying_the_same_set_is_a_no_op_and_one_failure_never_blocks_the_others() -> None:
    pool = _Pool(resident={("mlx", "a")}, unloadable={("huggingface", "gone")})
    lc._apply_model_residency_locks(pool, [("mlx", "a")])
    pool.calls.clear()
    out = lc._apply_model_residency_locks(pool, [("mlx", "a"), ("huggingface", "gone")])
    assert ("lock", "mlx", "a") not in pool.calls  # already locked: kept, not re-locked
    assert out["ok"] is False
    by_pair = {(r["provider"], r["model"]): r for r in out["results"]}
    assert by_pair[("mlx", "a")]["action"] == "kept" and by_pair[("mlx", "a")]["ok"] is True
    assert by_pair[("huggingface", "gone")]["action"] == "load_failed" and by_pair[("huggingface", "gone")]["error"] == "model_not_downloaded"


@pytest.mark.parametrize("bad", [[{"provider": "mlx"}], ["mlx/a"], [{"provider": "", "model": "a"}]])
def test_a_malformed_lock_is_refused_never_dropped(bad) -> None:
    with pytest.raises(ValueError):
        lc._apply_model_residency_locks(_Pool(resident=set()), bad)


def test_both_local_clients_expose_it_under_a_set_name_and_the_facade_relays_it() -> None:
    for cls in (lc.LocalAbstractCoreLLMClient, lc.MultiLocalAbstractCoreLLMClient):
        assert callable(getattr(cls, "set_model_residency_locks", None)), cls.__name__
    pool = _Pool(resident={("mlx", "a")})
    pool.set_model_residency_locks = lambda locks: lc._apply_model_residency_locks(pool, locks)  # type: ignore[attr-defined]
    facade = AbstractCoreHostFacade.__new__(AbstractCoreHostFacade)
    facade._client = pool  # type: ignore[attr-defined]
    assert facade.set_model_residency_locks([("mlx", "a")])["ok"] is True
    assert pool._locked_model_residency == {("mlx", "a")}
    facade._client = object()  # type: ignore[attr-defined]
    assert facade.set_model_residency_locks([])["supported"] is False
