"""A speculation request must reach `create_llm`, not just `generate`.

Operator report (2026-09-21), a BRAND NEW run with MTP depth chosen in the UI:

    Effect failed after 3 attempts: speculation must be requested when the
    provider is created -- it selects the runtime that loads the weights.
    Pass speculation={'mode': 'native_mtp'} to create_llm(...)

That message is AbstractCore's per-call guard, and it was correct: MLX binds
the MTP drafter to the target while the weights load, so a provider built
without speculation cannot acquire it afterwards. What made it unreachable is
here: the runtime carried the request as a generate PARAM only
(`visual/executor.py`, `compiler.py` both set `params["speculation"]`) and
pools provider instances by (provider, model) alone. A run asking for a depth
therefore built a provider WITHOUT the request microseconds before failing on
it -- so "request it at construction" named something no caller could do.

The request now rides `llm_kwargs_override` into construction AND stays in
params, so Core still validates it and reports the outcome.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


class _StubClient:
    def __init__(self, llm_kwargs_override: Optional[Dict[str, Any]] = None) -> None:
        self.llm_kwargs_override = dict(llm_kwargs_override or {})
        self.params_seen: List[Dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Dict[str, Any]:
        self.params_seen.append(dict(kwargs.get("params") or {}))
        return {"content": "ok"}

    def set_on_token(self, callback: Any) -> None:  # pragma: no cover - pool fan-out
        return None


def _pool(monkeypatch):
    from abstractruntime.integrations.abstractcore import llm_client as mod

    class _StubLocal(_StubClient):
        # The pool builds its DEFAULT client directly in __init__, without
        # going through `_create_client`, so this stub has to answer
        # `generate` too — an "off" call is served by that instance.
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(kwargs.get("llm_kwargs"))
            self._llm = object()

    monkeypatch.setattr(mod, "LocalAbstractCoreLLMClient", _StubLocal)
    pool = mod.MultiLocalAbstractCoreLLMClient(provider="mlx", model="target-model")

    created: List[Dict[str, Any]] = []

    def _fake_create(provider: str, model: str, *, llm_kwargs_override=None):
        created.append(dict(llm_kwargs_override or {}))
        return _StubClient(llm_kwargs_override)

    pool._create_client = _fake_create  # type: ignore[method-assign]
    return pool, created


_DEPTH_3 = {"mode": "native_mtp", "num_draft_tokens": 3, "require_acceleration": True}


def test_a_depth_request_is_passed_to_provider_construction(monkeypatch) -> None:
    pool, created = _pool(monkeypatch)

    pool.generate(prompt="hi", params={"speculation": dict(_DEPTH_3)})

    assert created, "no client was constructed"
    assert created[-1].get("speculation") == _DEPTH_3, (
        "the speculation request never reached create_llm — this is the whole defect"
    )


def test_the_request_still_rides_the_call_so_core_reports_the_outcome(monkeypatch) -> None:
    pool, _created = _pool(monkeypatch)

    pool.generate(prompt="hi", params={"speculation": dict(_DEPTH_3)})

    client = next(iter(pool._override_clients.values()))
    assert client.params_seen[-1].get("speculation") == _DEPTH_3


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"speculation": None},
        {"speculation": False},
        {"speculation": {"mode": "off"}},
    ],
    ids=["absent", "none", "false", "mode-off"],
)
def test_off_and_absent_never_split_the_pool(monkeypatch, params) -> None:
    """Constructing a separate provider for "no speculation" would double the
    loaded weights for nothing, so only an ACTIVE request reaches construction."""
    pool, created = _pool(monkeypatch)

    pool.generate(prompt="hi", params=dict(params))

    assert all("speculation" not in entry for entry in created), created


def test_two_depths_do_not_share_one_loaded_provider(monkeypatch) -> None:
    """Depth is chosen at load time: one instance cannot serve both."""
    pool, _created = _pool(monkeypatch)

    depth3 = pool._get_client("mlx", "m", llm_kwargs_override={"speculation": {"num_draft_tokens": 3}})
    depth5 = pool._get_client("mlx", "m", llm_kwargs_override={"speculation": {"num_draft_tokens": 5}})
    depth3_again = pool._get_client("mlx", "m", llm_kwargs_override={"speculation": {"num_draft_tokens": 3}})

    assert depth3 is not depth5, "a depth-5 request was served by the depth-3 load"
    assert depth3 is depth3_again, "an identical request rebuilt the provider"


def test_speculation_does_not_collide_with_a_different_endpoint(monkeypatch) -> None:
    """The key keeps its existing axes: same depth, different base_url, two clients."""
    pool, _created = _pool(monkeypatch)

    a = pool._get_client("mlx", "m", llm_kwargs_override={"speculation": {"num_draft_tokens": 3}, "base_url": "http://a"})
    b = pool._get_client("mlx", "m", llm_kwargs_override={"speculation": {"num_draft_tokens": 3}, "base_url": "http://b"})

    assert a is not b
