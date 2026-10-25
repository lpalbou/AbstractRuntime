"""Wakes: a run schedules work for later (`EffectType.SCHEDULE_RUN`, framework backlog 1045).

An agent (the `schedule_run` builtin) or a flow node asks for a run LATER — at a time, after a delay,
or on a trigger event — with a task (work to do) or a note ("ping me in 5 minutes about X"). A wake
IS a one-shot automation:

- `trigger`: `schedule@2 {kind: once, start_at}` for `at` / `after_s`, or the source named in `on`
  (`{source_id, source_version?, connection_id?, config}`, one occurrence then the automation ends);
- `session_id`: the caller's session by default. `mode: "same_session"` (default) runs the occurrence
  as a new turn OF THAT SESSION (growing context: the model's replay sees the conversation, and the
  conversation shows the run with its `[Trigger …]` header); `"independent"` runs it in a fresh
  session;
- tools: AT MOST the caller's. The caller's ceiling is resolved over its run tree
  (`core.tool_scope.resolve_tool_scope`) and intersected with the requested `tools`; `schedule_run`
  itself is never passed on (a wake cannot schedule wakes). The result is written as the occurrence's
  explicit `_runtime.allowed_tools` (an occurrence is a root run: no parent walk would find the
  ceiling). A note wake gets no tools. A caller without any recorded ceiling must name its tools.
- approval: `policy.tool_approval = "ask"` (the person's interactive rules, never an unattended
  grant); the caller's own `_runtime.tool_policy` is copied when it has one; a caller that reads
  untrusted input (`_runtime.untrusted_input`) makes the wake untrusted too (`untrusted_origin`).
- `origin_run_id`: the calling run; the occurrence's `_meta.origin` is `{kind: "agent", ref:
  {origin_run_id, session_id, automation_id, ...}}`.

The runtime builds the PLAN (`plan_schedule_run`); the HOST creates the automation, because only it
knows the target workflow (the session's agent), the owner and the workspace: the host registers
`Runtime.set_schedule_run_handler(handler)`, `handler(run, plan) -> {"automation_id": ...}`, and
usually implements it with `create_scheduled_run(runtime, plan, target=..., tenant=..., user=...,
workspace_root=...)`. The host also appends the user-stream event (`wake_scheduled_event`):

    {"type": "wake.scheduled", "data": {"automation_id", "session_id", "at" | "on", "origin_run_id"}}

and, when the automation is deleted (archived) before it fires, `wake_cancelled_event`:

    {"type": "wake.cancelled", "data": {"automation_id", "session_id", "origin_run_id"}}
"""

from __future__ import annotations

import copy
import hashlib
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional

from ..core.tool_scope import ToolScopeError, resolve_tool_scope
from ..triggers.protocol import format_timestamp, parse_timestamp

SCHEDULE_RUN_TOOL = "schedule_run"
SCHEDULE_RUN_MODES = ("same_session", "independent")
# Never passed on to a wake (a wake cannot schedule wakes).
WAKE_EXCLUDED_TOOLS = frozenset({SCHEDULE_RUN_TOOL})
MAX_AFTER_S = 366 * 86400
MAX_TEXT = 20_000
TITLE_MAX = 120
_PAYLOAD_KEYS = ("at", "after_s", "on", "task", "note", "tools", "session_id", "mode", "title", "tool_name", "call_id")


class ScheduleRunError(ValueError):
    """An invalid SCHEDULE_RUN request (`reason_code`, `field`)."""

    def __init__(self, message: str, *, reason_code: str = "invalid_schedule_run", field: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.field = field


def _text(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ScheduleRunError(f"{field} must be a non-empty string", field=field)
    if len(value) > MAX_TEXT:
        raise ScheduleRunError(f"{field} must be at most {MAX_TEXT} characters", field=field)
    return value.strip()


def validate_schedule_run(payload: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
    """Normalize a SCHEDULE_RUN payload: `{when: {at} | {on}, task | note, tools, session_id, mode, title}`."""
    if not isinstance(payload, Mapping):
        raise ScheduleRunError("schedule_run payload must be an object")
    unknown = sorted(k for k in payload if k not in _PAYLOAD_KEYS)
    if unknown:
        raise ScheduleRunError(f"unknown schedule_run field(s): {unknown}", field=unknown[0])
    given = [k for k in ("at", "after_s", "on") if payload.get(k) is not None]
    if len(given) != 1:
        raise ScheduleRunError("give exactly one of at, after_s or on", field="at")
    when: Dict[str, Any]
    now_dt = parse_timestamp(now, field="now")
    if given[0] == "at":
        try:
            at_dt = parse_timestamp(payload["at"], field="at")
        except ValueError as exc:
            raise ScheduleRunError(str(exc), field="at") from exc
        if at_dt <= now_dt:
            raise ScheduleRunError("at must be in the future", field="at")
        when = {"at": format_timestamp(at_dt)}
    elif given[0] == "after_s":
        after = payload["after_s"]
        if isinstance(after, bool) or not isinstance(after, (int, float)) or not 1 <= after <= MAX_AFTER_S:
            raise ScheduleRunError(f"after_s must be a number of seconds 1..{MAX_AFTER_S}", field="after_s")
        when = {"at": format_timestamp(now_dt + timedelta(seconds=float(after))), "after_s": after}
    else:
        on = payload["on"]
        if not isinstance(on, Mapping):
            raise ScheduleRunError("on must be an object {source_id, source_version?, connection_id?, config?}", field="on")
        extra = sorted(k for k in on if k not in ("source_id", "source_version", "connection_id", "config"))
        if extra:
            raise ScheduleRunError(f"unknown on field(s): {extra}", field=f"on.{extra[0]}")
        source_id = _text(on.get("source_id"), "on.source_id")
        version = on.get("source_version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ScheduleRunError("on.source_version must be an integer >= 1", field="on.source_version")
        config = on.get("config") or {}
        if not isinstance(config, Mapping):
            raise ScheduleRunError("on.config must be an object", field="on.config")
        config = copy.deepcopy(dict(config))
        if on.get("connection_id") is not None:
            config["connection_id"] = _text(on.get("connection_id"), "on.connection_id")
        when = {"on": {"source_id": source_id, "source_version": version, "config": config}}
    task, note = _text(payload.get("task"), "task"), _text(payload.get("note"), "note")
    if (task is None) == (note is None):
        raise ScheduleRunError("give exactly one of task (work to do) or note (a reminder)", field="task")
    tools = payload.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or any(not isinstance(t, str) or not t.strip() for t in tools):
            raise ScheduleRunError("tools must be a list of tool names", field="tools")
        tools = sorted({t.strip() for t in tools})
    mode = payload.get("mode", "same_session")
    if mode not in SCHEDULE_RUN_MODES:
        raise ScheduleRunError(f"mode must be one of {list(SCHEDULE_RUN_MODES)}", field="mode")
    title = _text(payload.get("title"), "title")
    return {
        "when": when,
        "task": task,
        "note": note,
        "tools": tools,
        "session_id": _text(payload.get("session_id"), "session_id"),
        "mode": mode,
        "title": title[:TITLE_MAX] if title else None,
    }


def _caller_runtime(run: Any) -> Dict[str, Any]:
    vars_obj = run.vars if isinstance(getattr(run, "vars", None), dict) else {}
    return vars_obj.get("_runtime") if isinstance(vars_obj.get("_runtime"), dict) else {}


def plan_schedule_run(run: Any, payload: Mapping[str, Any], *, now: str, run_store: Any = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    """The wake to create for `run` (see the module doc). Raises ScheduleRunError."""
    req = validate_schedule_run(payload, now=now)
    try:
        ceiling = resolve_tool_scope(run, run_store=run_store)
    except ToolScopeError as exc:
        raise ScheduleRunError(f"the caller's tool ceiling cannot be established: {exc}", reason_code="tool_scope_unavailable") from exc
    if ceiling is not None and SCHEDULE_RUN_TOOL not in ceiling:
        raise ScheduleRunError("this run may not schedule runs (schedule_run is outside its tools)", reason_code="not_permitted")
    dropped: List[str] = []
    if req["note"] is not None:
        allowed: List[str] = []  # a reminder does not act
    else:
        requested = req["tools"]
        if requested is None:
            if ceiling is None:
                raise ScheduleRunError(
                    "name the tools the scheduled run may use (this run has no recorded tool ceiling to inherit)",
                    reason_code="tools_required",
                    field="tools",
                )
            requested = sorted(ceiling)
        allowed = [t for t in requested if (ceiling is None or t in ceiling) and t not in WAKE_EXCLUDED_TOOLS]
        dropped = [t for t in requested if t not in allowed]
    caller_session = getattr(run, "session_id", None)
    session_id = req["session_id"] or caller_session
    if req["session_id"] is not None and req["session_id"] != caller_session:
        raise ScheduleRunError("a run schedules into its own session only", reason_code="not_permitted", field="session_id")
    if req["mode"] == "same_session" and not session_id:
        raise ScheduleRunError("the caller has no session; use mode independent", field="mode")
    if "at" in req["when"]:
        trigger = {"source_id": "schedule", "source_version": 2, "config": {"kind": "once", "start_at": req["when"]["at"]}}
    else:
        on = req["when"]["on"]
        trigger = {"source_id": on["source_id"], "source_version": on["source_version"], "config": copy.deepcopy(on["config"])}
    caller_rt = _caller_runtime(run)
    untrusted = bool(caller_rt.get("untrusted_input")) or bool((caller_rt.get("tool_policy") or {}).get("untrusted_input"))
    runtime_ns: Dict[str, Any] = {"allowed_tools": list(allowed)}
    if isinstance(caller_rt.get("tool_policy"), dict) and not untrusted:
        runtime_ns["tool_policy"] = copy.deepcopy(caller_rt["tool_policy"])
    if untrusted:
        runtime_ns["untrusted_input"] = True
    if req["note"] is not None:
        prompt = f"Reminder you asked for earlier: {req['note']}"
        title = req["title"] or f"Reminder: {req['note']}"[:TITLE_MAX]
    else:
        prompt = str(req["task"])
        title = req["title"] or f"Scheduled: {req['task']}"[:TITLE_MAX]
    rid = request_id or "wake:" + hashlib.sha256(f"{run.run_id}\n{now}\n{sorted(payload.items())!r}".encode("utf-8")).hexdigest()[:32]
    plan: Dict[str, Any] = {
        "request_id": rid,
        "kind": "note" if req["note"] is not None else "task",
        "title": title,
        "trigger": trigger,
        "context": {"mode": "growing" if req["mode"] == "same_session" else "independent"},
        "policy": {"tool_approval": "ask"},
        "input_data": {"prompt": prompt, "_runtime": runtime_ns},
        "session_id": session_id if req["mode"] == "same_session" else None,
        "caller_session_id": caller_session,
        "origin_run_id": str(run.run_id),
        "mode": req["mode"],
        "one_shot": True,
        "untrusted_origin": untrusted,
        "allowed_tools": list(allowed),
        "dropped_tools": dropped,
        "when": copy.deepcopy(req["when"]),
    }
    return plan


def create_scheduled_run(
    runtime: Any,
    plan: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    tenant: str,
    user: str,
    workspace_root: str,
    notify: Optional[Mapping[str, Any]] = None,
    actor_id: Optional[str] = "gateway",
    now: Optional[str] = None,
) -> Dict[str, Any]:
    """Create the wake's one-shot automation (what a host's `schedule_run_handler` usually does).

    `target`: `{workflow_id, bundle_ref, flow_id, input_data?}` (the session's agent); the plan's
    `input_data` (prompt, `_runtime.allowed_tools`, ...) is laid over the target's. Returns
    `{automation_id, revision}`; idempotent on the plan's `request_id`."""
    from .service import create_automation

    target_inputs = dict(target.get("input_data") or {})
    plan_inputs = copy.deepcopy(dict(plan["input_data"]))
    runtime_ns = {**dict(target_inputs.get("_runtime") or {}), **plan_inputs.pop("_runtime", {})}
    input_data = {**target_inputs, **plan_inputs, "_runtime": runtime_ns}
    request: Dict[str, Any] = {
        "request_id": plan["request_id"],
        "tenant": tenant,
        "user": user,
        "title": plan["title"],
        "target": {**dict(target), "input_data": input_data},
        "trigger": copy.deepcopy(plan["trigger"]),
        "context": copy.deepcopy(plan["context"]),
        "policy": copy.deepcopy(plan["policy"]),
        "workspace_root": workspace_root,
        "origin_run_id": plan["origin_run_id"],
        "one_shot": bool(plan.get("one_shot", True)),
    }
    if plan.get("session_id"):
        request["session_id"] = plan["session_id"]
    if plan.get("untrusted_origin"):
        request["untrusted_origin"] = True
    if notify is not None:
        request["notify"] = dict(notify)
    from .ledger import definition_of
    from .models import AutomationError, automation_id_for

    try:
        automation_id, revision = create_automation(runtime, request, now=now, actor_id=actor_id)
    except AutomationError as exc:
        # A replayed SCHEDULE_RUN (same request_id) whose `after_s` resolved to a later time: the
        # wake already exists for this very call. Re-find it instead of failing or duplicating.
        if exc.reason_code != "identity_conflict":
            raise
        existing_id = automation_id_for(tenant=tenant, user=user, request_id=str(plan["request_id"]))
        existing = runtime.run_store.load(existing_id)
        if existing is None or definition_of(existing).get("origin_run_id") != plan["origin_run_id"]:
            raise
        return {"automation_id": existing_id, "revision": int(definition_of(existing)["revision"]), "replayed": True}
    return {"automation_id": automation_id, "revision": revision}


def wake_scheduled_event(plan: Mapping[str, Any], automation_id: str) -> Dict[str, Any]:
    """The user-stream event the host appends once the wake exists."""
    data: Dict[str, Any] = {
        "automation_id": str(automation_id),
        "session_id": plan.get("session_id") or plan.get("caller_session_id"),
        "origin_run_id": plan.get("origin_run_id"),
    }
    when = plan.get("when") or {}
    if "at" in when:
        data["at"] = when["at"]
    else:
        data["on"] = copy.deepcopy(when.get("on"))
    return {"type": "wake.scheduled", "data": data}


def wake_cancelled_event(*, automation_id: str, session_id: Optional[str], origin_run_id: Optional[str]) -> Dict[str, Any]:
    """The user-stream event the host appends when a wake is deleted before it fired."""
    return {"type": "wake.cancelled", "data": {"automation_id": str(automation_id), "session_id": session_id, "origin_run_id": origin_run_id}}


__all__ = [
    "SCHEDULE_RUN_MODES",
    "SCHEDULE_RUN_TOOL",
    "ScheduleRunError",
    "WAKE_EXCLUDED_TOOLS",
    "create_scheduled_run",
    "plan_schedule_run",
    "validate_schedule_run",
    "wake_cancelled_event",
    "wake_scheduled_event",
]
