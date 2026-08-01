"""THE AUTHORITY CONTRACT for the reasoning effort, on the execution side.

AbstractCore stores a reasoning effort on the text-generation capability route.
The Runtime is what turns a stored default into a call, so this file pins the
cascade it applies:

  1. an explicit `thinking` on the call wins, INCLUDING ``False``
  2. otherwise the effort configured on the text route applies
  3. otherwise no reasoning parameter is sent at all

The route keys and their precedence come from AbstractCore
(`config/capability_defaults.py::capability_default_reasoning`); the Runtime
holds no second copy of that mapping, which the last test asserts by name.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from abstractruntime.integrations.abstractcore.llm_client import (
    _with_capability_default_reasoning,
    _with_capability_default_route,
)
from abstractruntime.integrations.abstractcore.output_specs import (
    capability_default_reasoning_for_text,
)


def _defaults(**routes: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {key.replace("__", "."): dict(value) for key, value in routes.items()}


_CONFIGURED = _defaults(
    output__text={"key": "output.text", "provider": "lmstudio", "model": "qwen3", "reasoning": "high"},
    input__text={"key": "input.text", "provider": "lmstudio", "model": "qwen3", "reasoning": "high"},
)


def test_the_configured_effort_applies_when_the_call_names_none() -> None:
    params: Dict[str, Any] = {}
    assert _with_capability_default_reasoning(params, _CONFIGURED) == "high"
    assert params["thinking"] == "high"


@pytest.mark.parametrize("pinned", ["low", "minimal", "medium", True, False])
def test_an_explicit_pin_beats_the_configured_default(pinned: Any) -> None:
    """``False`` is a decision -- reasoning off for this call -- not an absence."""
    params: Dict[str, Any] = {"thinking": pinned}
    assert _with_capability_default_reasoning(params, _CONFIGURED) == pinned
    assert params["thinking"] == pinned


@pytest.mark.parametrize(
    "capability_defaults",
    [
        None,
        {},
        _defaults(output__text={"key": "output.text", "provider": "lmstudio", "model": "qwen3"}),
        _defaults(output__text={"key": "output.text", "source": "not_configured"}),
        _defaults(output__voice={"key": "output.voice", "reasoning": "high"}),
    ],
    ids=["none", "empty", "no-reasoning-field", "not-configured", "wrong-route"],
)
def test_no_configured_effort_sends_no_reasoning_parameter(capability_defaults: Any) -> None:
    params: Dict[str, Any] = {}
    assert _with_capability_default_reasoning(params, capability_defaults) is None
    assert "thinking" not in params


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_thinking_is_an_absence_not_a_pin(blank: Optional[str]) -> None:
    params: Dict[str, Any] = {"thinking": blank}
    assert _with_capability_default_reasoning(params, _CONFIGURED) == "high"
    assert params["thinking"] == "high"


@pytest.mark.parametrize("pinned", [0, "0", "auto", "off"])
def test_any_value_a_caller_actually_typed_is_a_pin(pinned: Any) -> None:
    """Falsy is not the same as absent.

    `0` and `"0"` are not values AbstractCore accepts for `thinking`, and
    `"auto"` and `"off"` are. What they share is that a caller wrote them down:
    treating any of them as an absence would replace a caller's decision -- or a
    caller's mistake -- with the host's configured effort, so the wrong reasoning
    level would be applied and the mistake would never surface. Only `None` and
    whitespace mean "the call named nothing".
    """
    params: Dict[str, Any] = {"thinking": pinned}
    assert _with_capability_default_reasoning(params, _CONFIGURED) == pinned
    assert params["thinking"] == pinned


def test_a_blank_thinking_is_dropped_when_nothing_is_configured() -> None:
    """An empty string must never travel to a provider as a reasoning value."""
    params: Dict[str, Any] = {"thinking": ""}
    assert _with_capability_default_reasoning(params, {}) is None
    assert "thinking" not in params


def test_the_storage_key_answers_when_the_canonical_row_is_absent() -> None:
    """A config carrying only `input.text` still resolves its effort."""
    params: Dict[str, Any] = {}
    defaults = _defaults(input__text={"key": "input.text", "provider": "p", "model": "m", "reasoning": "low"})
    assert _with_capability_default_reasoning(params, defaults) == "low"


def test_the_effort_is_normalized_to_lower_case() -> None:
    defaults = _defaults(output__text={"key": "output.text", "reasoning": "  HIGH  "})
    assert capability_default_reasoning_for_text(defaults) == "high"


def test_the_reasoning_read_never_raises_on_a_malformed_payload() -> None:
    """A broken row degrades to "send nothing"; it must not break a call."""
    assert capability_default_reasoning_for_text({"output.text": "not-a-row"}) is None
    assert capability_default_reasoning_for_text([]) is None
    assert capability_default_reasoning_for_text(None) is None


def test_a_reasoning_only_route_is_honoured_and_routes_nothing() -> None:
    """The two questions a route row answers are not the same question.

    "Has the operator configured this route?" counts the reasoning effort -- the
    grid says `configured` and the effort is applied. "Does this row name
    somewhere to send the call?" does not: a row carrying only an effort names
    no provider, model or base URL, so the routing merge leaves the call alone
    rather than inventing a target from an empty row.
    """
    defaults = _defaults(
        output__text={"key": "output.text", "reasoning": "high"},
        input__text={"key": "input.text", "reasoning": "high"},
    )
    params: Dict[str, Any] = {}
    assert _with_capability_default_reasoning(params, defaults) == "high"

    spec = {"output": {"kind": "text"}}
    assert _with_capability_default_route(spec, defaults) == spec


def test_the_provider_model_merge_is_unaffected_by_the_reasoning_dial() -> None:
    """Media specs route on provider/model/base_url/options, not on reasoning."""
    routed = _with_capability_default_route(
        {"modality": "voice", "task": "tts"},
        _defaults(output__voice={"key": "output.voice", "provider": "abstractvoice", "model": "supertonic"}),
    )
    assert routed["provider"] == "abstractvoice"
    assert routed["model"] == "supertonic"
    assert "reasoning" not in routed
    assert "thinking" not in routed


def test_the_route_keys_come_from_abstractcore() -> None:
    """One definition of where a reasoning default lives, and it is Core's."""
    from abstractcore.config.capability_defaults import capability_default_reasoning

    payload = {"output.text": {"reasoning": "medium"}}
    assert capability_default_reasoning_for_text(payload) == capability_default_reasoning(payload)
