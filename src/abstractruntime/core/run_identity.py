"""Creation identity for runs started with an explicit id (automations contract C1).

A run created with a caller-chosen id goes through `RunStore.create_if_absent`:
the first creation wins and every later creation with the same id either loads
that run (same identity) or is refused with `RunIdentityConflict`. Nothing ever
overwrites an existing run.

Identity = `workflow_id`, `session_id`, `parent_run_id`, and — when either side
carries them — `vars._meta.occurrence` and `vars._meta.creation_digest` (the
sha256 of the canonical JSON of the caller's creation request, generated
defaults excluded; see `creation_digest`).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

from .models import RunState

IDENTITY_META_KEYS = ("occurrence", "creation_digest")


class RunIdentityConflict(ValueError):
    """A run with this id already exists with a different creation identity.

    `reason_code` is the wire spelling hosts put in error envelopes. A
    ValueError subclass, like `StaleResumeError`, so existing `except
    ValueError` handling keeps working.
    """

    reason_code = "identity_conflict"

    def __init__(self, run_id: str, field: str, existing: Any, requested: Any) -> None:
        self.run_id = str(run_id)
        self.field = str(field)
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Run {self.run_id} already exists with a different {self.field} "
            f"(existing={existing!r}, requested={requested!r})"
        )


# Short alias used in the automation mission text; the contract name is
# `RunIdentityConflict` (automations contract §3 seams).
IdentityConflict = RunIdentityConflict


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 kept."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def creation_digest(
    *,
    workflow_id: str,
    session_id: Optional[str],
    parent_run_id: Optional[str],
    vars: Mapping[str, Any],
) -> str:
    """`sha256:<hex>` of the canonical creation request.

    Computed over what the CALLER passed (before `Runtime.start` seeds its
    generated defaults such as `_limits`), so replaying the same request
    yields the same digest.
    """
    request = {
        "workflow_id": str(workflow_id),
        "session_id": str(session_id) if session_id else None,
        "parent_run_id": str(parent_run_id) if parent_run_id else None,
        "vars": dict(vars),
    }
    return "sha256:" + hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def _meta_of(run: RunState) -> Dict[str, Any]:
    vars_obj = run.vars if isinstance(run.vars, dict) else {}
    meta = vars_obj.get("_meta")
    return meta if isinstance(meta, dict) else {}


def _norm(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def verify_run_identity(existing: RunState, requested: RunState) -> None:
    """Raise `RunIdentityConflict` unless `existing` is the run `requested` would create."""
    for field in ("workflow_id", "session_id", "parent_run_id"):
        a = _norm(getattr(existing, field, None))
        b = _norm(getattr(requested, field, None))
        if a != b:
            raise RunIdentityConflict(requested.run_id, field, a, b)
    existing_meta = _meta_of(existing)
    requested_meta = _meta_of(requested)
    for key in IDENTITY_META_KEYS:
        a = existing_meta.get(key)
        b = requested_meta.get(key)
        if a is None and b is None:
            continue
        if canonical_json(a) != canonical_json(b):
            raise RunIdentityConflict(requested.run_id, f"vars._meta.{key}", a, b)


__all__ = [
    "IDENTITY_META_KEYS",
    "IdentityConflict",
    "RunIdentityConflict",
    "canonical_json",
    "creation_digest",
    "verify_run_identity",
]
