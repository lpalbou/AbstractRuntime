"""Pins: a model EJECT stops the runtime effects using that model first, attributed.

`_unload_local_provider_residency` (the local/multilocal `unload_model_residency`
path behind the gateway's POST /models/unload) cancels every in-flight effect
annotated with the ejected provider/model (`cancelled_by="model_eject"`)
BEFORE calling the provider's `unload_model`; effects of other models are
untouched.
"""

from __future__ import annotations

import threading

from abstractruntime.core.effect_cancellation import (
    InflightEffect,
    effect_inflight_scope,
    request_model_effects_cancel,
)
from abstractruntime.integrations.abstractcore.llm_client import _unload_local_provider_residency


def _entry(step: str, provider, model) -> InflightEffect:
    e = InflightEffect(run_id=f"run-{step}", node_id="n", step_id=step, effect_type="llm_call")
    e.provider = provider
    e.model = model
    return e


def test_request_model_effects_cancel_matches_model_and_provider():
    a = _entry("a", "mlx", "mlx-community/Qwen3.5-4B-4bit")
    b = _entry("b", "mlx", "other-model")
    c = _entry("c", None, "mlx-community/Qwen3.5-4B-4bit")  # handler did not record a provider
    d = _entry("d", "lmstudio", "mlx-community/Qwen3.5-4B-4bit")
    with effect_inflight_scope(a), effect_inflight_scope(b), effect_inflight_scope(c), effect_inflight_scope(d):
        out = request_model_effects_cancel("MLX", "mlx-community/Qwen3.5-4B-4bit", reason="eject")
    assert {s["step_id"] for s in out} == {"a", "c"}
    assert a.cancel_event.is_set() and c.cancel_event.is_set()
    assert not b.cancel_event.is_set() and not d.cancel_event.is_set()
    assert a.cancelled_by == "model_eject" and a.cancel_reason == "eject"
    assert request_model_effects_cancel("mlx", "") == []


def test_unload_cancels_effects_before_the_provider_unload():
    e = _entry("e", "mlx", "m1")
    seen = {}

    class _Provider:
        provider = "mlx"

        def unload_model(self, model_name):
            seen["event_set_at_unload"] = e.cancel_event.is_set()
            return None

    with effect_inflight_scope(e):
        result, error = _unload_local_provider_residency(provider_instance=_Provider(), model="m1")
    assert error is None
    assert seen == {"event_set_at_unload": True}
    assert e.cancelled_by == "model_eject"
