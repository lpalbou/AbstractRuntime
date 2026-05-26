"""Small run lifecycle metadata helpers.

Runs may carry a private ``_run_lifecycle`` vars namespace so control-plane
clients can distinguish draft tests, published executions, and retention hints
without exposing the full run input payload.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

RUN_LIFECYCLE_VAR_KEY = "_run_lifecycle"

_STRING_FIELDS = (
    "source",
    "purpose",
    "visibility",
    "run_mode",
    "editor_session_id",
    "flow_id",
    "bundle_id",
    "bundle_version",
)
_RETENTION_STRING_FIELDS = ("mode", "expires_at")
_RETENTION_NUMBER_FIELDS = ("ttl_s",)
_MAX_STRING_LEN = 240


def _clean_string(value: Any, *, max_len: int = _MAX_STRING_LEN) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:max_len]


def sanitize_run_lifecycle(value: Any) -> Optional[Dict[str, Any]]:
    """Return a small HTTP-safe lifecycle object, or ``None`` if absent/empty."""

    if not isinstance(value, dict):
        return None

    out: Dict[str, Any] = {}
    for field in _STRING_FIELDS:
        cleaned = _clean_string(value.get(field))
        if cleaned is not None:
            out[field] = cleaned

    retention_raw = value.get("retention")
    if isinstance(retention_raw, dict):
        retention: Dict[str, Any] = {}
        for field in _RETENTION_STRING_FIELDS:
            cleaned = _clean_string(retention_raw.get(field))
            if cleaned is not None:
                retention[field] = cleaned
        for field in _RETENTION_NUMBER_FIELDS:
            raw = retention_raw.get(field)
            if isinstance(raw, bool) or raw is None:
                continue
            try:
                number = int(raw)
            except Exception:
                continue
            if number > 0:
                retention[field] = number
        if retention:
            out["retention"] = retention

    return out or None


def extract_run_lifecycle(vars_obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(vars_obj, dict):
        return None
    return sanitize_run_lifecycle(vars_obj.get(RUN_LIFECYCLE_VAR_KEY))


def normalize_run_lifecycle_vars(vars_obj: Dict[str, Any]) -> None:
    """Normalize ``_run_lifecycle`` in-place and drop invalid lifecycle data."""

    lifecycle = sanitize_run_lifecycle(vars_obj.get(RUN_LIFECYCLE_VAR_KEY))
    if lifecycle is None:
        vars_obj.pop(RUN_LIFECYCLE_VAR_KEY, None)
        return
    vars_obj[RUN_LIFECYCLE_VAR_KEY] = lifecycle


def run_lifecycle_index_fields(vars_obj: Any) -> Dict[str, Any]:
    lifecycle = extract_run_lifecycle(vars_obj)
    return {"run_lifecycle": lifecycle}
