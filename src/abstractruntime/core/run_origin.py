"""Who or what started a run: `vars._meta.origin` (framework backlog 1038).

Every run records its origin ONCE, at start, in one shape:

    _meta.origin = {
        "kind":   "person" | "trigger" | "agent" | "system",
        "actor":  "<tenant>:<user>" | None,    # the principal on whose behalf the run started
        "client": "<interface>" | None,        # console, assistant, watch, code, flow, automation, bridge:<name>...
        "ref":    {                            # pointers, never copies (all optional)
            "automation_id", "occurrence_index", "event_id", "source",   # a trigger-started run
            "origin_run_id", "session_id",                               # an agent-started run (SCHEDULE_RUN)
            "bridge",                                                     # a system bridge
        },
    }

Who stamps it:

- the host, for runs a person starts (`/runs/start`: kind `person`, the principal, the client) and for
  bridge runs (kind `system`): `origin_for(...)` builds the dict, the host puts it in the run's
  `_meta.origin` before `Runtime.start`;
- the automation controller, for every occurrence (`build_prepared`): kind `trigger` (a scheduled or
  event admission; `ref.source` = `<source_id>@<version>`), kind `person` for a manual `run_now`
  (the owner asked), kind `agent` when the automation is a wake an agent scheduled
  (`SCHEDULE_RUN`, `ref.origin_run_id`). The actor is the automation's owner principal
  (`_meta.automation_owner` on the controller, `"<tenant>:<user>"` of the creation request).

The trigger envelope is never copied into the origin: `ref.automation_id` + `ref.event_id` point at
the controller's `automation.admitted` ledger record that holds it. `origin_of(vars)` reads it back
(None for runs started before origins existed: readers keep their former classification).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

ORIGIN_KEY = "origin"
ORIGIN_KINDS = ("person", "trigger", "agent", "system")
ORIGIN_REF_KEYS = ("automation_id", "occurrence_index", "event_id", "source", "origin_run_id", "session_id", "bridge")


def _opt_text(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"origin.{field} must be a non-empty string or None")
    return value.strip()


def origin_for(
    kind: str,
    *,
    actor: Optional[str] = None,
    client: Optional[str] = None,
    automation_id: Optional[str] = None,
    occurrence_index: Optional[int] = None,
    event_id: Optional[str] = None,
    source: Optional[str] = None,
    origin_run_id: Optional[str] = None,
    session_id: Optional[str] = None,
    bridge: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a validated `_meta.origin` (see the module doc). Raises ValueError."""
    if kind not in ORIGIN_KINDS:
        raise ValueError(f"origin.kind must be one of {list(ORIGIN_KINDS)}, got {kind!r}")
    ref: Dict[str, Any] = {}
    for name, value in (
        ("automation_id", automation_id),
        ("event_id", event_id),
        ("source", source),
        ("origin_run_id", origin_run_id),
        ("session_id", session_id),
        ("bridge", bridge),
    ):
        text = _opt_text(value, f"ref.{name}")
        if text is not None:
            ref[name] = text
    if occurrence_index is not None:
        if isinstance(occurrence_index, bool) or not isinstance(occurrence_index, int) or occurrence_index < 1:
            raise ValueError("origin.ref.occurrence_index must be an integer >= 1")
        ref["occurrence_index"] = int(occurrence_index)
    if kind == "trigger" and "automation_id" not in ref:
        raise ValueError("a trigger origin names its automation (ref.automation_id)")
    if kind == "agent" and "origin_run_id" not in ref:
        raise ValueError("an agent origin names the run that scheduled it (ref.origin_run_id)")
    return {"kind": kind, "actor": _opt_text(actor, "actor"), "client": _opt_text(client, "client"), "ref": ref}


def origin_of(vars_obj: Any) -> Optional[Dict[str, Any]]:
    """The run's recorded origin, or None (a run started before 1038, or a malformed value)."""
    meta = vars_obj.get("_meta") if isinstance(vars_obj, Mapping) else None
    origin = meta.get(ORIGIN_KEY) if isinstance(meta, Mapping) else None
    if not isinstance(origin, Mapping) or origin.get("kind") not in ORIGIN_KINDS:
        return None
    ref = origin.get("ref") if isinstance(origin.get("ref"), Mapping) else {}
    return {
        "kind": origin["kind"],
        "actor": origin.get("actor"),
        "client": origin.get("client"),
        "ref": {k: ref[k] for k in ORIGIN_REF_KEYS if k in ref},
    }


def stamp_origin(vars_obj: Dict[str, Any], origin: Mapping[str, Any]) -> Dict[str, Any]:
    """Set `vars._meta.origin` (a run's origin is written once: an existing one is kept)."""
    meta = vars_obj.get("_meta") if isinstance(vars_obj.get("_meta"), dict) else {}
    if not isinstance(meta.get(ORIGIN_KEY), Mapping):
        meta = {**meta, ORIGIN_KEY: dict(origin)}
    vars_obj["_meta"] = meta
    return vars_obj


__all__ = ["ORIGIN_KEY", "ORIGIN_KINDS", "ORIGIN_REF_KEYS", "origin_for", "origin_of", "stamp_origin"]
