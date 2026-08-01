"""THE FRESH-INSTALL REFUSAL IS THE UX.

A brand-new user meets "no provider configured" before they meet any document,
so the message has to name the fix, and the run must not spend three attempts
reaching it. Two failures used to live here:

  1. "no provider configured - set one in the request, the workflow, or the
     gateway defaults" named no route, no command and no route key. A user who
     read it still did not know what to type.
  2. It was a plain ValueError, so the retry policy treated a CONFIGURATION
     refusal as transient and tripled the wait before showing it.

Plus the attribution case: when the provider that fails IS the configured host
default, the bare "Unknown provider: x" never says the operator set it and
never says where. The wrapper adds that, and must NOT change how retryable the
underlying failure was.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractruntime.integrations.abstractcore.effect_handlers import _llm_error_is_retryable
from abstractruntime.integrations.abstractcore.llm_client import (
    DefaultRouteProviderError,
    NoDefaultProviderConfigured,
    default_route_provider_error,
    no_default_provider_configured_error,
)


def test_the_refusal_names_both_entry_points_and_the_route() -> None:
    message = str(no_default_provider_configured_error())

    assert "abstractcore config set-default output.text --provider <provider> --model <model>" in message
    assert "/api/gateway/config/capability-defaults/output/text" in message
    assert "_runtime.provider" in message, "the per-call pin is the third way out"
    assert "abstractcore config defaults" in message, "and how to see what is set"


def test_the_refusal_names_the_store_only_when_that_store_exists(tmp_path: Path) -> None:
    """A host may hand down a scoped config path that was never created; the
    AbstractCore manager then resolves its own store. Printing the phantom path
    as "the store" sends the operator to edit a file nobody reads."""
    missing = tmp_path / "never-created" / "abstractcore.json"
    assert "Store consulted" not in str(no_default_provider_configured_error(core_config_file=str(missing)))

    real = tmp_path / "abstractcore.json"
    real.write_text("{}", encoding="utf-8")
    assert f"Store consulted: {real}" in str(no_default_provider_configured_error(core_config_file=str(real)))


def test_a_configuration_refusal_is_never_retried() -> None:
    assert _llm_error_is_retryable(no_default_provider_configured_error()) is False
    assert isinstance(no_default_provider_configured_error(), NoDefaultProviderConfigured)


def test_the_default_route_wrapper_says_the_operator_configured_this() -> None:
    wrapped = default_route_provider_error(
        ValueError("Unknown provider: notaprovider. Available providers: openai, lmstudio"),
        provider="notaprovider",
        model="nope-1",
    )
    message = str(wrapped)

    assert "Unknown provider: notaprovider" in message, "the original failure survives verbatim"
    assert "is the execution host's configured default" in message
    assert "capability route output.text" in message
    assert "abstractcore config set-default output.text" in message
    assert "/api/gateway/config/capability-defaults/output/text" in message


@pytest.mark.parametrize(
    "cause,expected_retryable",
    [
        (ValueError("Unknown provider: notaprovider. Available providers: openai"), False),
        (RuntimeError("OpenAI-compatible server API error (429): slow down"), True),
        (RuntimeError("connection reset by peer"), True),
        (RuntimeError("OpenAI-compatible server API error (400): bad request"), False),
    ],
)
def test_the_wrapper_inherits_its_causes_retryability(cause, expected_retryable) -> None:
    """Attribution must not change classification. The wrapper adds "and this
    came from your config" to somebody else's failure; a transient failure that
    happened to hit the default route is still transient."""
    wrapped = default_route_provider_error(cause, provider="p", model="m")
    assert isinstance(wrapped, DefaultRouteProviderError)
    assert _llm_error_is_retryable(wrapped) is expected_retryable
    assert _llm_error_is_retryable(cause) is expected_retryable


def test_an_unresolvable_provider_name_is_deterministic() -> None:
    """The registry does not gain a provider between attempts."""
    assert _llm_error_is_retryable(ValueError("Unknown provider: nope. Available providers: openai")) is False
