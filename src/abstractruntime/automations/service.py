"""Automation service API for hosts (the gateway) and standalone use.

- `create_automation(runtime, request)` creates the controller root run with
  the explicit id `uuid5(AUTOMATION_NAMESPACE, "<tenant>:<user>:<request_id>")`
  through create-if-absent: the same request replays to the same automation,
  a different request with the same `request_id` is an `identity_conflict`.
  The request's target must be concrete (the host resolves `@default`).
- `get_automation`, `list_occurrences`: projections of the controller run and
  its ledger.
- `start_discussion(...)`: a NEW root run in its own session, seeded once from
  the automation's conversation through a chosen occurrence, on the
  occurrence's workspace mounted read-only.
- `adopt_legacy_schedule_projection(run)`: read-only summary of a legacy
  `scheduled:*` wrapper root (`legacy: true`); legacy roots are never migrated.
- `drive_automation(runtime, automation_id)`: a minimal host loop for use
  without the gateway (ticks the controller and its current occurrence, and
  resumes the controller when the occurrence ends).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

from ..core.models import RunState, RunStatus, WaitReason
from ..triggers.protocol import format_timestamp
from ..triggers.registry import get_trigger_adapter
from ..utils.workspace_paths import READ_ONLY_KEY
from .attention import resolve_strict
from .bundle import controller_workflow_spec
from .ledger import (
    append_observation,
    automation_records,
    definition_of,
    find_by_idempotency_key,
    record_key,
    record_payload,
    state_of,
)
from .models import (
    AutomationError,
    automation_id_for,
    automation_status,
    build_definition,
    discussion_ids,
    initial_state,
    request_digest,
    wake_wait_key,
)

OCCURRENCE_CURSOR_PREFIX = "occ1:"


def _now() -> str:
    return format_timestamp(datetime.now(timezone.utc))


def _load_automation(run_store: Any, automation_id: str) -> RunState:
    run = run_store.load(str(automation_id))
    if run is None:
        raise AutomationError(f"Automation {automation_id} does not exist.", reason_code="automation_not_found")
    try:
        definition_of(run)
        state_of(run)
    except LookupError as exc:
        raise AutomationError(str(exc), reason_code="automation_not_found") from exc
    return run


# --- creation -----------------------------------------------------------------


def create_automation(runtime: Any, request: Mapping[str, Any], *, now: Optional[str] = None) -> Tuple[str, int]:
    """Validate `request` and create (or re-find) the automation; returns `(automation_id, revision)`.

    Request: `{request_id, title, target: {workflow_id, bundle_ref, flow_id,
    input_data?}, trigger: {source_id, source_version, config}, context?,
    policy?, workspace_root, tenant?, user?}`. Raises AutomationError
    (`invalid_definition`, `unsupported_feature`, `unknown_trigger_source`,
    `identity_conflict`). The host ticks the controller afterwards like any run.
    """
    from ..core.run_identity import RunIdentityConflict

    at = now or _now()
    if not isinstance(request, Mapping):
        raise AutomationError("request must be an object", reason_code="invalid_definition")
    tenant = str(request.get("tenant") or "local")
    user = str(request.get("user") or "local")
    automation_id = automation_id_for(tenant=tenant, user=user, request_id=str(request.get("request_id") or ""))
    definition = build_definition(request, automation_id=automation_id, now=at)
    binding = definition["trigger"]
    adapter = get_trigger_adapter(binding["source_id"], binding["source_version"])
    state = initial_state(definition, trigger_state=adapter.initial_state(binding["config"]))
    controller = controller_workflow_spec()
    try:
        runtime.start(
            workflow=controller,
            vars={
                "_meta": {"automation": definition, "creation_digest": request_digest(request)},
                "_runtime": {"automation": state},
                "workspace_root": definition["workspace_root"],
            },
            session_id=definition["session_id"],
            run_id=automation_id,
        )
    except RunIdentityConflict as exc:
        raise AutomationError(
            f"request_id {request.get('request_id')!r} was already used for a different automation request",
            reason_code="identity_conflict",
            field="request_id",
        ) from exc
    run = runtime.run_store.load(automation_id)
    persisted = definition_of(run)
    append_observation(
        run,
        ledger_store=runtime.ledger_store,
        name="automation.created",
        key=record_key("automation.created", automation_id, 1),
        fields={"definition": persisted},
        at=persisted["created_at"],
        node_id="create",
    )
    return automation_id, int(persisted["revision"])


# --- reads --------------------------------------------------------------------------


def get_automation(run_store: Any, automation_id: str) -> Dict[str, Any]:
    """`{automation_id, definition, active_revision, state, status, next_fire_at}`."""
    run = _load_automation(run_store, automation_id)
    definition, state = definition_of(run), state_of(run)
    next_fire_at = None
    waiting = run.waiting
    if (
        run.status == RunStatus.WAITING
        and waiting is not None
        and waiting.wait_key == wake_wait_key(run.run_id)
        and state.get("pending_occurrence") is None
    ):
        next_fire_at = waiting.until
    return {
        "automation_id": run.run_id,
        "definition": copy.deepcopy(definition),
        "active_revision": int(state.get("active_revision") or definition["revision"]),
        "state": {k: copy.deepcopy(v) for k, v in state.items() if k != "intent"},
        "status": automation_status(run),
        "next_fire_at": next_fire_at,
    }


def _cursor_index(cursor: Optional[str]) -> Optional[int]:
    if cursor is None:
        return None
    if not isinstance(cursor, str) or not cursor.startswith(OCCURRENCE_CURSOR_PREFIX):
        raise AutomationError(f"invalid occurrence cursor {cursor!r}", reason_code="invalid_request", field="cursor")
    try:
        return int(cursor[len(OCCURRENCE_CURSOR_PREFIX):])
    except ValueError as exc:
        raise AutomationError(f"invalid occurrence cursor {cursor!r}", reason_code="invalid_request", field="cursor") from exc


def list_occurrences(
    runtime: Any, automation_id: str, *, cursor: Optional[str] = None, limit: int = 50
) -> Dict[str, Any]:
    """Occurrences, newest first, from the automation's ledger: `Page{items, next_cursor}`.

    Item: `{index, run_id, run_ids, attempts, revision, event_id, fired_at,
    trigger: {source_id, source_version}, user_turn, status, finished_at,
    notify, attention}`; `status` is `admitted | running | backoff | completed
    | failed | cancelled`.
    """
    _load_automation(runtime.run_store, automation_id)
    before = _cursor_index(cursor)
    limit = max(1, min(int(limit), 500))
    rows: Dict[int, Dict[str, Any]] = {}
    for rec in automation_records(
        runtime.ledger_store,
        automation_id,
        "automation.admitted",
        "automation.dispatched",
        "automation.retry_scheduled",
        "automation.completed",
    ):
        p = rec["payload"]
        index = int(p["index"])
        if rec["name"] == "automation.admitted":
            envelope = p.get("trigger_envelope") or {}
            prompt = ((p.get("prepared") or {}).get("input_data") or {}).get("prompt")
            rows[index] = {
                "index": index,
                "run_id": p["run_id"],
                "run_ids": [],
                "attempts": 1,
                "revision": p.get("revision"),
                "event_id": p.get("event_id"),
                "fired_at": envelope.get("fired_at"),
                "trigger": {"source_id": envelope.get("source_id"), "source_version": envelope.get("source_version")},
                "user_turn": prompt if isinstance(prompt, str) else None,
                "status": "admitted",
                "finished_at": None,
                "notify": None,
                "attention": None,
            }
            continue
        row = rows.get(index)
        if row is None:
            continue
        if rec["name"] == "automation.dispatched":
            row.update(run_id=p["run_id"], attempts=int(p["attempt"]), status="running")
            if p["run_id"] not in row["run_ids"]:
                row["run_ids"].append(p["run_id"])
        elif rec["name"] == "automation.retry_scheduled":
            row["status"] = "backoff"
        else:
            row.update(
                status=p["status"],
                attempts=int(p["attempts"]),
                finished_at=p.get("finished_at"),
                notify=p.get("notify"),
                attention=p.get("attention"),
            )
    ordered = sorted((r for i, r in rows.items() if before is None or i < before), key=lambda r: -r["index"])
    page = ordered[:limit]
    next_cursor = f"{OCCURRENCE_CURSOR_PREFIX}{page[-1]['index']}" if len(ordered) > limit else None
    return {"items": page, "next_cursor": next_cursor}


# --- discussion --------------------------------------------------------------------------


def start_discussion(
    runtime: Any,
    *,
    automation_id: str,
    occurrence_index: int,
    request_id: str,
    prompt: str,
) -> Dict[str, str]:
    """Start (or re-find) a discussion forked from occurrence `occurrence_index`.

    A new ROOT run in session `discussion-session:<request_id>` (run id
    `uuid5(automation_id, "discuss:" + request_id)`), running the occurrence's
    workflow with its frozen inputs, `prompt` as the new user turn, the
    automation's conversation through that occurrence as `context.messages`
    (read strictly: no seed, no discussion), and the occurrence's workspace
    mounted read-only. The seed is stored once in `_meta.discussion`; nothing
    is ever written back into the automation's session or state.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise AutomationError("prompt must be a non-empty string", reason_code="invalid_request", field="prompt")
    if not isinstance(request_id, str) or not request_id.strip():
        raise AutomationError("request_id must be a non-empty string", reason_code="invalid_request", field="request_id")
    _load_automation(runtime.run_store, automation_id)
    index = int(occurrence_index)
    admitted = find_by_idempotency_key(runtime.ledger_store, automation_id, record_key("automation.admitted", automation_id, index))
    if admitted is None:
        raise AutomationError(
            f"Automation {automation_id} has no occurrence {index}.", reason_code="occurrence_not_found", field="occurrence_index"
        )
    admitted_p = record_payload(admitted)
    completed = find_by_idempotency_key(runtime.ledger_store, automation_id, record_key("automation.completed", automation_id, index))
    seed_run_id = record_payload(completed)["run_id"] if completed is not None else admitted_p["run_id"]
    prepared = resolve_strict(admitted_p["prepared"], artifact_store=runtime.artifact_store)

    workflow = runtime.workflow_registry.get(prepared["workflow_id"]) if runtime.workflow_registry is not None else None
    if workflow is None:
        raise LookupError(f"workflow {prepared['workflow_id']!r} of occurrence {index} is not registered on this runtime")

    from ..session_history import session_chat_messages

    seed = session_chat_messages(
        run_store=runtime.run_store,
        ledger_store=runtime.ledger_store,
        artifact_store=runtime.artifact_store,
        session_id=prepared["session_id"],
        automation_id=automation_id,
        through_occurrence=index,
        strict=True,
    )
    ids = discussion_ids(automation_id, request_id)
    input_data = copy.deepcopy(prepared["input_data"])
    input_data["prompt"] = prompt
    context = input_data.get("context") if isinstance(input_data.get("context"), dict) else {}
    input_data["context"] = {**context, "messages": list(seed)}
    meta = input_data.get("_meta") if isinstance(input_data.get("_meta"), dict) else {}
    meta["discussion"] = {
        "automation_id": automation_id,
        "occurrence_index": index,
        "revision": int(admitted_p["revision"]),
        "seed_run_id": seed_run_id,
        "request_id": request_id,
        "discussion_root_run_id": ids["run_id"],
        "seed_messages": list(seed),
    }
    meta["creation_digest"] = request_digest(
        {"automation_id": automation_id, "occurrence_index": index, "request_id": request_id, "prompt": prompt}
    )
    input_data["_meta"] = meta
    input_data["workspace_root"] = prepared["workspace_root"]
    # Host key (contract C4) and the trusted runtime-policy key, which a client
    # payload cannot carry: both mean "this workspace is mounted read-only".
    input_data[READ_ONLY_KEY] = True
    runtime_ns = input_data.get("_runtime") if isinstance(input_data.get("_runtime"), dict) else {}
    input_data["_runtime"] = {**runtime_ns, READ_ONLY_KEY: True}
    runtime.start(workflow=workflow, vars=input_data, session_id=ids["session_id"], run_id=ids["run_id"])
    return {"session_id": ids["session_id"], "run_id": ids["run_id"], "session_kind": "discussion"}


# --- legacy -------------------------------------------------------------------------------


def adopt_legacy_schedule_projection(run: RunState) -> Dict[str, Any]:
    """Read-only summary of a legacy `scheduled:*` wrapper root (`legacy: true`).

    Legacy roots keep their existing controls (pause/resume/cancel); nothing is
    migrated or written.
    """
    meta = (run.vars.get("_meta") or {}) if isinstance(run.vars, dict) else {}
    schedule = meta.get("schedule")
    if not isinstance(schedule, dict) or schedule.get("kind") != "scheduled_run":
        raise AutomationError(f"run {run.run_id} is not a legacy scheduled root", reason_code="automation_not_found")
    config = {
        k: v
        for k, v in (
            ("start_at", schedule.get("start_at")),
            ("every", schedule.get("interval")),
            ("count", schedule.get("repeat_count")),
            ("until", schedule.get("repeat_until")),
        )
        if v is not None
    }
    status = {
        RunStatus.COMPLETED: "completed",
        RunStatus.FAILED: "failed",
        RunStatus.CANCELLED: "archived",
    }.get(run.status, "active")
    from ..core.vars import is_paused_vars

    if status == "active" and is_paused_vars(run.vars):
        status = "paused"
    return {
        "automation_id": run.run_id,
        "title": str(schedule.get("target_flow_id") or schedule.get("target_workflow_id") or run.workflow_id),
        "status": status,
        "trigger": {"binding_id": None, "source_id": "schedule", "source_version": 1, "config": config},
        "context_mode": "growing" if schedule.get("share_context") else "independent",
        "target": {
            "workflow_id": schedule.get("target_workflow_id"),
            "bundle_ref": schedule.get("target_bundle_ref"),
            "flow_id": schedule.get("target_flow_id"),
        },
        "legacy": True,
        "revision": None,
        "created_at": schedule.get("created_at") or run.created_at,
        "updated_at": run.updated_at,
        "capabilities": ["pause", "resume", "cancel"],
        "session_kind": "automation",
    }


# --- standalone host loop -----------------------------------------------------------------


def drive_automation(runtime: Any, automation_id: str, *, max_rounds: int = 50) -> RunState:
    """Advance an automation until it parks (for hosts without a run loop, and tests).

    Ticks the controller; when it waits on its occurrence, ticks that child
    (which needs its workflow in `runtime.workflow_registry`) and resumes the
    controller once the child is terminal. Stops when the controller parks on
    its wake wait, ends, or the occurrence waits on something else (a human).
    """
    spec = controller_workflow_spec()
    state = runtime.get_state(automation_id)
    for _ in range(max_rounds):
        if state.status == RunStatus.RUNNING:
            state = runtime.tick(workflow=spec, run_id=automation_id)
            continue
        if state.status != RunStatus.WAITING or state.waiting is None:
            return state
        if state.waiting.reason != WaitReason.SUBWORKFLOW:
            return state  # parked on its wake wait (or a deadline the caller advances)
        child_id = str((state.waiting.details or {}).get("sub_run_id") or "")
        child = runtime.get_state(child_id)
        if child.status == RunStatus.RUNNING:
            workflow = runtime.workflow_registry.get(child.workflow_id)
            if workflow is None:
                raise LookupError(f"occurrence workflow {child.workflow_id!r} is not registered")
            child = runtime.tick(workflow=workflow, run_id=child_id)
        if child.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            output = child.output if child.status == RunStatus.COMPLETED else {"success": False, "error": child.error}
            state = runtime.resume(
                workflow=spec,
                run_id=automation_id,
                wait_key=state.waiting.wait_key,
                payload={"sub_run_id": child_id, "output": output},
            )
            continue
        if child.status == RunStatus.WAITING:
            return state
    return state


__all__ = [
    "adopt_legacy_schedule_projection",
    "create_automation",
    "drive_automation",
    "get_automation",
    "list_occurrences",
    "start_discussion",
]
