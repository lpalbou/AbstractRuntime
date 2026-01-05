from __future__ import annotations

from abstractruntime.core.config import RuntimeConfig


def test_runtime_config_persists_default_max_output_tokens_from_capabilities_when_unspecified() -> None:
    cfg = RuntimeConfig(model_capabilities={"max_tokens": 1000, "max_output_tokens": 1234})
    limits = cfg.to_limits_dict()
    assert limits["max_tokens"] == 1000
    assert limits["max_output_tokens"] == 1234


def test_runtime_config_keeps_explicit_max_output_tokens_override() -> None:
    cfg = RuntimeConfig(max_output_tokens=2048, model_capabilities={"max_output_tokens": 9999})
    limits = cfg.to_limits_dict()
    assert limits["max_output_tokens"] == 2048



