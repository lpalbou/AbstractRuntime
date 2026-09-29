"""Automation commands (contract A/D, C8, C16): the ONE applier.

`apply_automation_command` applies `automation.revise | pause | resume | run_now
| stop_current | archive` under the automation's `run_mutation_lock`, through
the decision protocol (`automations.ledger`): reconcile, exact key lookup of
`automation:command_result:<automation_id>:<command_id>`, decide from persisted
state (including `expected_revision`), append the decision, apply it, save.

The `automation.command_result` record IS the decision (its `delta` is the
applied change; a rejection has an empty delta). Replaying a command id returns
the recorded result and re-runs only the idempotent follow-ups (observation
record, controller wake, child cancellation), so a crash after the decision
never loses them. Domain rejections are returned, never raised.

Semantics:
- pause: gates SCHEDULED admission only (never the runtime pause gate); the
  current occurrence, including its retries, finishes; run_now stays allowed.
- resume: re-arms the trigger at the first tick after now; never fires.
- run_now: records `manual_pending` (admitted at the controller's next
  boundary); rejected `automation_busy` while an occurrence or a manual run is
  pending, `invalid_state` when archived or exhausted. No queue.
- revise: commits the next revision now; the controller activates it at its
  next boundary; a changed trigger gets a new binding id and is re-armed so no
  past tick fires; an admitted occurrence keeps its frozen inputs.
- stop_current: cancels the current occurrence tree (or its backoff); it
  completes `cancelled` (quiet).
- archive: no further admission; the current occurrence finishes; history kept.
Repeats of an already-true state (pause while paused, ...) are applied no-ops.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from ..core.models import RunStatus, WaitReason
from ..triggers.protocol import format_timestamp
from ..triggers.registry import get_trigger_adapter
from .ledger import (
    append_observation,
    commit_decision,
    definition_of,
    find_by_idempotency_key,
    reconcile,
    record_key,
    record_payload,
    state_of,
)
from .models import AutomationError, revise_definition, trigger_changed, trigger_state_of, wake_wait_key

AUTOMATION_COMMAND_TYPES = (
    "automation.revise",
    "automation.pause",
    "automation.resume",
    "automation.run_now",
    "automation.stop_current",
    "automation.archive",
)


def _now() -> str:
    from datetime import datetime, timezone

    return format_timestamp(datetime.now(timezone.utc))


def _reject(reason_code: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"status": "rejected", "error": {"reason_code": reason_code, "message": message, **extra}}


def _decide(
    run: Any,
    command_type: str,
    payload: Dict[str, Any],
    *,
    now: str,
    expected_revision: Optional[int],
    has_event_inbox: bool = True,
) -> Dict[str, Any]:
    """`{"delta": ...}` (the state change to apply) or a rejection."""
    definition, state = definition_of(run), state_of(run)
    archived = bool(definition.get("archived_at"))
    if expected_revision is not None and int(expected_revision) != int(definition["revision"]):
        return _reject(
            "revision_conflict",
            f"Automation is at revision {definition['revision']}, not {expected_revision}.",
            field="expected_revision",
        )
    terminal = run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)

    if command_type == "automation.pause":
        if archived or terminal:
            return _reject("invalid_state", "An archived or finished automation cannot be paused.")
        return {"delta": {"state": {"paused": True}}}

    if command_type == "automation.resume":
        if archived or terminal:
            return _reject("invalid_state", "An archived or finished automation cannot be resumed.")
        if not state.get("paused"):
            return {"delta": {}}
        binding = definition["trigger"]
        adapter = get_trigger_adapter(binding["source_id"], binding["source_version"])
        rearmed = adapter.rearm(binding, state=trigger_state_of(state), now=now)
        return {"delta": {"state": {**rearmed, "paused": False}}}

    if command_type == "automation.run_now":
        if archived or terminal or state.get("exhausted"):
            return _reject("invalid_state", "An archived, finished or exhausted automation cannot run.")
        if state.get("pending_occurrence") is not None or state.get("manual_pending") is not None:
            return _reject("automation_busy", "An occurrence is already running or waiting to run.")
        return {"delta": {"state": {"manual_pending": {"command_id": payload["_command_id"]}}}}

    if command_type == "automation.stop_current":
        pending = state.get("pending_occurrence")
        if pending is None:
            return _reject("invalid_state", "No occurrence is running.")
        return {"delta": {"state": {"pending_occurrence": {**pending, "stop_requested": True}}}}

    if command_type == "automation.archive":
        if archived:
            return {"delta": {}}
        return {"delta": {"definition": {**copy.deepcopy(definition), "archived_at": now}}}

    if command_type == "automation.revise":
        if archived or terminal:
            return _reject("invalid_state", "An archived or finished automation cannot be revised.")
        try:
            new_def = revise_definition(definition, payload.get("changes"), automation_id=run.run_id, now=now)
        except AutomationError as exc:
            return _reject(exc.reason_code, str(exc), **({"field": exc.field} if exc.field else {}))
        delta: Dict[str, Any] = {"definition": new_def}
        if trigger_changed(definition, new_def):
            binding = new_def["trigger"]
            adapter = get_trigger_adapter(binding["source_id"], binding["source_version"])
            kind = ((getattr(adapter, "descriptor", None) or {}).get("capabilities") or {}).get("kind")
            if kind == "event" and not has_event_inbox:
                return _reject(
                    "unsupported_feature",
                    f"trigger {binding['source_id']}@{binding['source_version']} needs the runtime's event inbox, "
                    "and this runtime has none (the host must call Runtime.set_event_inbox)",
                    field="changes.trigger.source_id",
                )
            fresh = adapter.initial_state(binding["config"])
            delta["state"] = dict(adapter.rearm(binding, state=fresh, now=now))
        return {"delta": delta}

    return _reject("invalid_request", f"unknown automation command type {command_type!r}", field="type")


def command_digest(type: str, payload: Optional[Dict[str, Any]], expected_revision: Optional[int]) -> str:
    """`sha256:<hex>` of the whole command, recorded with its result so a reused
    `command_id` is recognized only for the SAME command."""
    import hashlib

    from .models import canonical_json

    body = {"type": type, "payload": payload or {}, "expected_revision": expected_revision}
    return "sha256:" + hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _cancel_tree(runtime: Any, run_id: str) -> None:
    run_store = runtime.run_store
    stack = [run_id]
    while stack:
        rid = stack.pop()
        run = run_store.load(rid)
        if run is None:
            continue
        stack.extend(child.run_id for child in run_store.list_children(parent_run_id=rid))
        if run.status in (RunStatus.RUNNING, RunStatus.WAITING):
            runtime.cancel_run(rid, reason="Stopped by automation.stop_current", cancelled_by="command")


def _follow_ups(runtime: Any, automation_id: str, command_type: str) -> None:
    """Idempotent effects of an applied command (re-run on replay)."""
    from ..core.runtime import StaleResumeError
    from .bundle import controller_workflow_spec

    run = runtime.run_store.load(automation_id)
    state = state_of(run)
    pending = state.get("pending_occurrence")
    if command_type == "automation.stop_current" and pending and pending.get("stop_requested") and pending.get("phase") == "dispatched":
        child = runtime.run_store.load(pending["run_id"])
        if child is not None:
            _cancel_tree(runtime, child.run_id)
            child = runtime.run_store.load(pending["run_id"])
        waiting = run.waiting
        if (
            child is not None
            and run.status == RunStatus.WAITING
            and waiting is not None
            and waiting.reason == WaitReason.SUBWORKFLOW
            and waiting.wait_key == f"subworkflow:{child.run_id}"
        ):
            try:
                runtime.resume(
                    workflow=controller_workflow_spec(),
                    run_id=automation_id,
                    wait_key=waiting.wait_key,
                    payload={"sub_run_id": child.run_id, "output": {"success": False, "cancelled": True}},
                    max_steps=0,
                )
            except StaleResumeError:
                pass  # the host resumed it first
        return
    # Every other command changes what the idle controller should do: wake it.
    if run.status == RunStatus.WAITING and run.waiting is not None and run.waiting.wait_key == wake_wait_key(automation_id):
        try:
            runtime.resume(
                workflow=controller_workflow_spec(),
                run_id=automation_id,
                wait_key=run.waiting.wait_key,
                payload={"wake": command_type},
                max_steps=0,
            )
        except StaleResumeError:
            pass  # woken by someone else; the controller re-reads state anyway


def apply_automation_command(
    runtime: Any,
    *,
    automation_id: str,
    command_id: str,
    type: str,
    payload: Optional[Dict[str, Any]] = None,
    actor: Optional[str] = None,
    expected_revision: Optional[int] = None,
    now: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply one automation command; returns `{status, error?, duplicate}`.

    The controller is woken with `max_steps=0` (its wait is committed, not
    ticked): the host drives it like any other resumed run.
    """
    from ..core.runtime import run_mutation_lock

    command_id = str(command_id or "").strip()
    if not command_id:
        return {**_reject("invalid_request", "command_id is required", field="command_id"), "duplicate": False}
    if type not in AUTOMATION_COMMAND_TYPES:
        return {**_reject("invalid_request", f"unknown automation command type {type!r}", field="type"), "duplicate": False}
    at = now or _now()
    key = record_key("automation.command_result", automation_id, command_id)

    with run_mutation_lock(automation_id):
        run = runtime.run_store.load(automation_id)
        try:
            definition_of(run) if run is not None else None
        except LookupError:
            run = None
        if run is None:
            return {**_reject("automation_not_found", f"Automation {automation_id} does not exist."), "duplicate": False}
        reconcile(run, run_store=runtime.run_store, ledger_store=runtime.ledger_store)

        digest = command_digest(type, payload, expected_revision)
        existing = find_by_idempotency_key(runtime.ledger_store, automation_id, key)
        if existing is not None:
            recorded = record_payload(existing)
            if recorded.get("command_digest") != digest:
                # The same id for a DIFFERENT command: never "applied", never recorded.
                return {
                    **_reject(
                        "identity_conflict",
                        f"command_id {command_id!r} was already used for a different command ({recorded.get('type')}).",
                        field="command_id",
                    ),
                    "duplicate": False,
                }
            result = {"status": recorded["status"], "duplicate": True}
            if recorded.get("error"):
                result["error"] = recorded["error"]
        else:
            decision = _decide(
                run,
                type,
                {**(payload or {}), "_command_id": command_id},
                now=at,
                expected_revision=expected_revision,
                has_event_inbox=getattr(runtime, "event_inbox", None) is not None,
            )
            fields: Dict[str, Any] = {"command_id": command_id, "type": type, "actor": actor, "command_digest": digest}
            if decision.get("status") == "rejected":
                fields.update(status="rejected", error=decision["error"])
                delta: Dict[str, Any] = {}
            else:
                fields["status"] = "applied"
                delta = decision["delta"]
            commit_decision(
                run,
                run_store=runtime.run_store,
                ledger_store=runtime.ledger_store,
                name="automation.command_result",
                key=key,
                fields=fields,
                delta=delta,
                at=at,
                node_id="command",
                command_id=command_id,
            )
            result = {"status": fields["status"], "duplicate": False}
            if decision.get("status") == "rejected":
                result["error"] = decision["error"]
        if result["status"] != "applied":
            return result

        # Follow-ups: observation record (once per key), then wake/cancel.
        obs = _observation_for(run, type, command_id=command_id, now=at)
        if obs is not None:
            name, discriminator, obs_fields = obs
            append_observation(
                run,
                ledger_store=runtime.ledger_store,
                name=name,
                key=record_key(name, automation_id, discriminator),
                fields=obs_fields,
                at=at,
                node_id="command",
                command_id=command_id,
            )
        _follow_ups(runtime, automation_id, type)
        return result


def _observation_for(run: Any, command_type: str, *, command_id: str, now: str) -> Optional[tuple]:
    """The observation record of an APPLIED command, rebuilt from the recorded decision."""
    definition, state = definition_of(run), state_of(run)
    if command_type == "automation.pause":
        return ("automation.paused", command_id, {})
    if command_type == "automation.resume":
        binding = definition["trigger"]
        adapter = get_trigger_adapter(binding["source_id"], binding["source_version"])
        if state.get("paused"):
            return None
        wait = adapter.prepare(binding, state=trigger_state_of(state), now=now)
        return ("automation.resumed", command_id, {"next_fire_at": wait.get("until")})
    if command_type == "automation.archive":
        pending = state.get("pending_occurrence")
        return ("automation.archived", command_id, {"active_occurrence_run_id": pending["run_id"]} if pending else {})
    if command_type == "automation.revise":
        return (
            "automation.revised",
            int(definition["revision"]),
            {"previous_revision": int(definition["revision"]) - 1, "definition": definition},
        )
    return None


def record_automation_command_result(
    runtime: Any,
    *,
    automation_id: str,
    command_id: str,
    type: str,
    error: Dict[str, Any],
    actor: Optional[str] = None,
    now: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    expected_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Record a host-side failure of a command as a rejected `command_result` (same protocol).

    Pass the command's `payload`/`expected_revision` so a later replay of the
    same command is recognized as a duplicate (a different one is refused).
    """
    from ..core.runtime import run_mutation_lock

    at = now or _now()
    key = record_key("automation.command_result", automation_id, command_id)
    with run_mutation_lock(automation_id):
        run = runtime.run_store.load(automation_id)
        if run is None:
            raise LookupError(f"automation {automation_id} does not exist")
        reconcile(run, run_store=runtime.run_store, ledger_store=runtime.ledger_store)
        payload = commit_decision(
            run,
            run_store=runtime.run_store,
            ledger_store=runtime.ledger_store,
            name="automation.command_result",
            key=key,
            fields={"command_id": command_id, "type": type, "actor": actor, "status": "rejected", "error": dict(error),
                    "command_digest": command_digest(type, payload, expected_revision)},
            delta={},
            at=at,
            node_id="command",
            command_id=command_id,
        )
    return {"status": payload["status"], "error": payload.get("error")}


__all__ = ["AUTOMATION_COMMAND_TYPES", "apply_automation_command", "record_automation_command_result"]
