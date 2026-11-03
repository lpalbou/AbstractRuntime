"""Hosts tell model errors apart through the Runtime facade, never by importing AbstractCore
(gateway ADR 0001). The facade hands out AbstractCore's own classes, so `except` catches exactly
what Core raises, and `ModelNotDownloadedError` keeps its `model` / `provider` attributes."""

from __future__ import annotations

import abstractcore.exceptions as core_exceptions

from abstractruntime.integrations.abstractcore import discovery_facade
from abstractruntime.integrations.abstractcore.discovery_facade import ModelNotDownloadedError, ModelNotFoundError


def test_the_facade_exports_cores_own_model_error_classes() -> None:
    assert ModelNotDownloadedError is core_exceptions.ModelNotDownloadedError
    assert ModelNotFoundError is core_exceptions.ModelNotFoundError
    assert issubclass(ModelNotDownloadedError, ModelNotFoundError)
    assert {"ModelNotDownloadedError", "ModelNotFoundError"} <= set(discovery_facade.__all__)


def test_a_core_raise_is_caught_through_the_facade_with_its_attributes() -> None:
    try:
        raise core_exceptions.ModelNotDownloadedError("not here", model="Qwen/Qwen3-Embedding-0.6B", provider="huggingface")
    except ModelNotDownloadedError as exc:
        assert (exc.model, exc.provider) == ("Qwen/Qwen3-Embedding-0.6B", "huggingface")
