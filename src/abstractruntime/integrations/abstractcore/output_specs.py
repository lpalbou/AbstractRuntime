"""Runtime adapters around AbstractCore's public output-selector contract."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_MIN_ABSTRACTCORE_VERSION = (2, 13, 11)
_MIN_ABSTRACTCORE_VERSION_TEXT = "2.13.11"


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


try:
    from abstractcore.utils.version import __version__ as _abstractcore_version
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
