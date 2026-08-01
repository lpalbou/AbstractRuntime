"""Runtime adapters around AbstractCore's public output-selector contract."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

def _version_tuple(value: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for raw in str(value or "").split(".")[:3]:
        digits = []
        for char in raw:
            if not char.isdigit():
                break
            digits.append(char)
        parts.append(int("".join(digits) or "0"))
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


# ONE floor, and it is the one `pyproject.toml` declares. The comparison tuple
# is DERIVED from the text rather than written beside it: the two had drifted a
# patch apart, so the guard admitted a version its own message called too old.
_MIN_ABSTRACTCORE_VERSION_TEXT = "2.13.38"
_MIN_ABSTRACTCORE_VERSION = _version_tuple(_MIN_ABSTRACTCORE_VERSION_TEXT)


try:
    from abstractcore.utils.version import __version__ as _abstractcore_version
    from abstractcore.config.capability_defaults import (
        capability_default_reasoning as _capability_default_reasoning,
        capability_route_keys_for_output,
    )
    from abstractcore.core.output_specs import (
        is_output_request,
        normalize_output_spec,
        normalize_output_specs,
        output_has_generated_media,
        output_requires_non_chat_dispatch,
        strip_runtime_output_metadata,
    )
except ImportError as exc:  # pragma: no cover - import-time dependency guard
    raise ImportError(
        f"abstractruntime.integrations.abstractcore requires abstractcore>={_MIN_ABSTRACTCORE_VERSION_TEXT} "
        "for the public output selector contract."
    ) from exc

if _version_tuple(_abstractcore_version) < _MIN_ABSTRACTCORE_VERSION:  # pragma: no cover - environment guard
    raise ImportError(
        f"abstractruntime.integrations.abstractcore requires abstractcore>={_MIN_ABSTRACTCORE_VERSION_TEXT} "
        f"for the current server-auth, provider-key, generated-media, and capability-catalog contracts; "
        f"found abstractcore {_abstractcore_version}."
    )


def capability_default_route_keys_for_spec(
    spec: Any,
    *,
    has_source_image: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    """The (exact, broad-fallback) capability-default route keys for an output spec.

    ONE STORE, ONE TABLE. The mapping from the generation-task vocabulary to
    capability route keys lives in AbstractCore
    (`config/capability_defaults.py::_OUTPUT_ROUTE_TABLE`) because AbstractCore
    owns the store those keys address. The Runtime used to keep a second copy of
    that mapping, and it had drifted: it minted `output.voice.tts`,
    `input.voice.stt`, `output.music.text_to_music` and `output.sound.text_to_sound`
    -- keys the store can NEVER hold, because none of those task names are in
    `CAPABILITY_ROUTE_TASKS`. Every one of them silently fell through to the broad
    modality key, so the table was correct only by accident, and it had no
    `scene3d` row at all. This adapter is the single import point.
    """

    if not isinstance(spec, dict):
        return None, None
    return capability_route_keys_for_output(
        spec.get("modality"),
        spec.get("task"),
        has_source_image=has_source_image,
    )


def capability_default_reasoning_for_text(capability_defaults: Any) -> Optional[str]:
    """The execution host's configured reasoning effort for text generation.

    ONE STORE, ONE DEFINITION. The route keys that carry a reasoning default and
    their precedence live in AbstractCore
    (`config/capability_defaults.py::capability_default_reasoning`) because
    AbstractCore owns the store. This adapter is the single import point, so the
    Runtime never re-derives the key order.

    Returns ``None`` when no reasoning default is configured, which means "send
    nothing and let the model behave as it does by default".
    """

    if not isinstance(capability_defaults, dict):
        return None
    try:
        return _capability_default_reasoning(capability_defaults)
    except Exception:  # pragma: no cover - a malformed row must not break a call
        return None


def is_abstractcore_output_request(output: Any) -> bool:
    """Return True when `output` is AbstractCore's multimodal output selector."""

    return is_output_request(output)


def normalize_output_spec_for_runtime(output: Any) -> Dict[str, Any]:
    """Normalize an AbstractCore output selector using the same public aliases as core."""

    return normalize_output_spec(output)


def normalize_output_specs_for_runtime(output: Any) -> List[Dict[str, Any]]:
    return normalize_output_specs(output)


def strip_runtime_output_metadata_for_core(output: Any) -> Any:
    """Keep runtime artifact metadata out of AbstractCore capability kwargs."""

    return strip_runtime_output_metadata(output)


def output_runtime_metadata(output: Any) -> tuple[Optional[str], Dict[str, Any]]:
    """Extract runtime storage metadata from an AbstractCore output selector."""

    if not is_abstractcore_output_request(output):
        return None, {}

    specs = normalize_output_specs_for_runtime(output)
    run_id: Optional[str] = None
    tags: Dict[str, Any] = {}
    for spec in specs:
        raw_run_id = spec.get("run_id")
        if run_id is None and isinstance(raw_run_id, str) and raw_run_id.strip():
            run_id = raw_run_id.strip()

        raw_tags = spec.get("tags")
        if isinstance(raw_tags, dict):
            tags.update({str(k): str(v) for k, v in raw_tags.items() if k is not None and v is not None})

    return run_id, tags


def output_request_has_generated_media(output: Any) -> bool:
    """Return True when an output selector can produce generated binary media."""

    return output_has_generated_media(output)


def output_request_has_non_text_result(output: Any) -> bool:
    return output_requires_non_chat_dispatch(output)
