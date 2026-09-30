"""Automation definition, controller state and identifiers (automations contract A/B).

An automation IS a durable runtime root run (`automation_id == run_id`) that
executes the shipped controller flow. Its definition lives in
`vars._meta.automation` (the latest committed revision), its cursors in
`vars._runtime.automation`, and every transition is an `automation.*` record in
its ledger (see `automations.ledger`).

Everything here is pure: validation, defaults, deterministic ids, backoff math.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from datetime import timedelta
from typing import Any, Dict, Mapping, Optional, TypedDict

from ..triggers.protocol import TriggerConfigError, parse_duration, parse_timestamp, format_timestamp

SCHEMA_VERSION = 1  # the `automation.*` ledger record schema
# The definition schema (`_meta.automation.schema_version`). v2 (framework backlog 0992 B5):
# `policy.email_allowed_recipients` and the top-level `notify`. A v1 definition reads with
# their defaults and becomes v2 at its next revision.
DEFINITION_SCHEMA_VERSION = 2

CONTROLLER_BUNDLE_ID = "abstractframework.automation-controller"
CONTROLLER_BUNDLE_VERSION = "1.0.0"
CONTROLLER_FLOW_ID = "controller"
CONTROLLER_BUNDLE_REF = f"{CONTROLLER_BUNDLE_ID}@{CONTROLLER_BUNDLE_VERSION}"
# The workflow id persisted on every controller root run (contract D,
# amendment 5): the same `<bundle_id>@<version>:<flow_id>` construction hosts
# use when they namespace bundle flows, so a restart resolves this pinned version.
CONTROLLER_WORKFLOW_ID = f"{CONTROLLER_BUNDLE_REF}:{CONTROLLER_FLOW_ID}"

# Namespace for automation ids: uuid5(AUTOMATION_NAMESPACE, f"{tenant}:{user}:{request_id}").
AUTOMATION_NAMESPACE = uuid.UUID("4a0d5a4e-6f1c-5b8e-9a53-2f7c1e0b9d61")

TITLE_MAX = 120
NOTIFY_TITLE_MAX = 120
NOTIFY_BODY_MAX = 2000
NOTIFY_DEFAULT_BODY_MAX = 280

DEFAULT_RETRY = {"max_attempts": 3, "backoff": {"initial": "30s", "factor": 2, "max": "10m"}}
TOOL_APPROVAL_MODES = ("auto", "ask")
# Recipients an occurrence may email WITHOUT an approval wait (framework backlog 0992 B3/B5):
# "self" (the registered address of the runtime's user, `_runtime.operator_email`) and exact
# addresses the user named. Separate from the account's recipient policy (allowlist / denylist),
# which AbstractCore enforces on every send whatever this says.
DEFAULT_EMAIL_ALLOWED_RECIPIENTS = ["self"]
MAX_EMAIL_ALLOWED_RECIPIENTS = 50
# Tools the user NAMED individually that an occurrence whose trigger delivers untrusted content
# (`email.received@1`) may run without asking although they reach a model-chosen destination
# (`fetch_url`, `browser_probe`). Default: none. "Allow all tools" (`tool_approval: "auto"`)
# never grants them to such an occurrence (operator decision 2026-09-30, framework backlog 0992).
DEFAULT_UNTRUSTED_INPUT_TOOLS: list = []
MAX_UNTRUSTED_INPUT_TOOLS = 20
DEFAULT_POLICY = {
    "serial": True, "misfire": "coalesce", "failure": "continue", "retry": DEFAULT_RETRY, "tool_approval": "auto",
    "email_allowed_recipients": DEFAULT_EMAIL_ALLOWED_RECIPIENTS,
    "untrusted_input_tools": DEFAULT_UNTRUSTED_INPUT_TOOLS,
}
NOTIFY_CHANNELS = ("console", "email")
DEFAULT_NOTIFY = {"channels": ["console"]}

CONTEXT_MODES = ("independent", "growing")
# Growing-mode history: the one session history window
# (`session_history.HISTORY_REPLAY_MAX_TOKENS`, most recent 50k tokens of whole turns).


class AutomationDefinition(TypedDict):
    """`vars._meta.automation` (built by `build_definition`, validated strictly)."""

    schema_version: int
    revision: int
    title: str
    controller: Dict[str, str]  # {bundle_ref, flow_id}
    target: Dict[str, Any]  # {workflow_id, bundle_ref, flow_id, input_data}
    trigger: Dict[str, Any]  # TriggerBinding
    context: Dict[str, Any]  # {mode, growing}
    policy: Dict[str, Any]  # {serial, misfire, failure, retry, tool_approval, email_allowed_recipients, untrusted_input_tools}
    notify: Dict[str, Any]  # {channels: ["console"] | ["console", "email"]} (schema v2)
    session_id: str
    workspace_root: str
    created_at: str
    archived_at: Optional[str]


class AutomationState(TypedDict):
    """`vars._runtime.automation` (see `initial_state`)."""

    state_version: int
    active_revision: int
    pending_occurrence: Optional[Dict[str, Any]]
    anchor: Optional[str]
    tick: int
    next_index: int
    scheduled_count: int
    paused: bool
    attention_seq: int
    exhausted: bool
    manual_pending: Optional[Dict[str, str]]
    last_outcome: Optional[Dict[str, Any]]
    intent: Optional[Dict[str, Any]]


class AutomationError(ValueError):
    """A domain failure with a wire reason code (`invalid_definition`,
    `unsupported_feature`, `unknown_trigger_source`, `automation_not_found`,
    `identity_conflict`, ...) and the offending field when there is one."""

    def __init__(self, message: str, *, reason_code: str, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.field = field

    def to_error(self) -> Dict[str, Any]:
        err: Dict[str, Any] = {"reason_code": self.reason_code, "message": str(self)}
        if self.field:
            err["field"] = self.field
        return err


def _invalid(message: str, field: str, reason_code: str = "invalid_definition") -> AutomationError:
    return AutomationError(message, reason_code=reason_code, field=field)


# --- identifiers ------------------------------------------------------------


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_digest(request: Mapping[str, Any]) -> str:
    """`sha256:<hex>` of the caller's creation request (canonical JSON, as sent)."""
    return "sha256:" + hashlib.sha256(canonical_json(dict(request)).encode("utf-8")).hexdigest()


def automation_id_for(*, tenant: str, user: str, request_id: str) -> str:
    return str(uuid.uuid5(AUTOMATION_NAMESPACE, f"{tenant}:{user}:{request_id}"))


def _child_uuid(automation_id: str, name: str) -> str:
    return str(uuid.uuid5(uuid.UUID(str(automation_id)), name))


def occurrence_run_id(
    automation_id: str,
    *,
    revision: int,
    index: int,
    attempt: int = 1,
    command_id: Optional[str] = None,
) -> str:
    """Deterministic occurrence child id (contract B).

    Scheduled: `uuid5(automation_id, f"{revision}:{index}")`; manual:
    `uuid5(automation_id, "manual:" + command_id)`; retry attempt n >= 2 appends
    `f":a{n}"`. Creation goes through create-if-absent, so a replayed dispatch
    loads the same child instead of starting a second one.
    """
    base = f"manual:{command_id}" if command_id else f"{int(revision)}:{int(index)}"
    if int(attempt) >= 2:
        base = f"{base}:a{int(attempt)}"
    return _child_uuid(automation_id, base)


def binding_id_for(automation_id: str, revision: int) -> str:
    """Server-minted trigger binding id for the revision that (re)bound the trigger."""
    return _child_uuid(automation_id, f"binding:{int(revision)}")


def automation_session_id(automation_id: str) -> str:
    return f"automation:{automation_id}"


def wake_wait_key(automation_id: str) -> str:
    """The controller's single command/timer wait key (contract D)."""
    return f"automation:{automation_id}:wake"


def discussion_ids(automation_id: str, request_id: str) -> Dict[str, str]:
    """Discussion root run id and session id, both scoped to the automation.

    The session id embeds the (automation-scoped) run id, so two automations
    using the same client `request_id` never share a discussion session.
    """
    run_id = _child_uuid(automation_id, f"discuss:{request_id}")
    return {"run_id": run_id, "session_id": f"discussion-session:{run_id}"}


# --- validation ---------------------------------------------------------------


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid(f"{field} must be an object", field)
    return value


def _reject_unknown(obj: Mapping[str, Any], allowed: tuple, field: str) -> None:
    unknown = sorted(k for k in obj if k not in allowed)
    if unknown:
        raise _invalid(f"unknown field(s) in {field}: {unknown}", f"{field}.{unknown[0]}" if field else unknown[0])


def _nonempty_str(value: Any, field: str, *, max_len: Optional[int] = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{field} must be a non-empty string", field)
    text = value.strip()
    if max_len is not None and len(text) > max_len:
        raise _invalid(f"{field} must be at most {max_len} characters", field)
    return text


def _json_value(value: Any, field: str) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise _invalid(f"{field} must be JSON", field) from exc


def validate_title(value: Any) -> str:
    return _nonempty_str(value, "title", max_len=TITLE_MAX)


def validate_target(value: Any) -> Dict[str, Any]:
    """A concrete target (the gateway resolves `@default` before calling)."""
    target = _require_mapping(value, "target")
    _reject_unknown(target, ("workflow_id", "bundle_ref", "flow_id", "input_data"), "target")
    input_data = target.get("input_data", {})
    if not isinstance(input_data, Mapping):
        raise _invalid("target.input_data must be an object", "target.input_data")
    flow_id = _nonempty_str(target.get("flow_id"), "target.flow_id")
    if flow_id == "@default":
        raise _invalid("target.flow_id '@default' must be resolved by the host before creation", "target.flow_id")
    return {
        "workflow_id": _nonempty_str(target.get("workflow_id"), "target.workflow_id"),
        "bundle_ref": _nonempty_str(target.get("bundle_ref"), "target.bundle_ref"),
        "flow_id": flow_id,
        "input_data": _json_value(dict(input_data), "target.input_data"),
    }


def validate_context(value: Any) -> Dict[str, Any]:
    ctx = _require_mapping(value if value is not None else {}, "context")
    _reject_unknown(ctx, ("mode", "growing"), "context")
    mode = ctx.get("mode", "independent")
    if mode not in CONTEXT_MODES:
        raise _invalid(f"context.mode must be one of {list(CONTEXT_MODES)}", "context.mode")
    growing = _require_mapping(ctx.get("growing", {}), "context.growing")
    _reject_unknown(growing, ("summary",), "context.growing")
    if "summary" in growing:
        raise _invalid(
            "automatic growing-mode summaries are not supported in v1",
            "context.growing.summary",
            reason_code="unsupported_feature",
        )
    return {"mode": mode, "growing": {}}


def _duration(value: Any, field: str) -> str:
    try:
        parse_duration(value, field=field)
    except TriggerConfigError as exc:
        raise _invalid(str(exc), field) from exc
    return value


def validate_email_allowed_recipients(value: Any) -> list:
    """`["self" | "name@example.test", ...]`, lower-cased and de-duplicated (order kept).

    Exact addresses only: no display names, groups, domains or patterns (a pre-authorisation
    must name the person). Empty list = nobody, not even self, runs unattended.
    """
    field = "policy.email_allowed_recipients"
    if not isinstance(value, list):
        raise _invalid(f"{field} must be a list of 'self' or email addresses", field)
    if len(value) > MAX_EMAIL_ALLOWED_RECIPIENTS:
        raise _invalid(f"{field} holds at most {MAX_EMAIL_ALLOWED_RECIPIENTS} entries", field)
    out: list = []
    for i, entry in enumerate(value):
        if not isinstance(entry, str) or not entry.strip():
            raise _invalid(f"{field}[{i}] must be 'self' or an email address", f"{field}[{i}]")
        item = entry.strip().lower()
        if item != "self":
            local, at, domain = item.partition("@")
            if (
                not at or not local or not domain or "@" in domain or "." not in domain
                or any(c.isspace() or c in "<>,;:\"()[]*" for c in item)
            ):
                raise _invalid(f"{field}[{i}] must be 'self' or a plain address like name@example.test", f"{field}[{i}]")
        if item not in out:
            out.append(item)
    return out


def validate_untrusted_input_tools(value: Any) -> list:
    """`["fetch_url", ...]`: tool names, each named individually (no "all", no wildcard).

    Order kept, de-duplicated. The grant uses them only for an occurrence whose trigger
    delivers untrusted content, only within the target's tool ceiling, and never for a
    message-sending tool (those always go through the recipient refiner).
    """
    field = "policy.untrusted_input_tools"
    if not isinstance(value, list):
        raise _invalid(f"{field} must be a list of tool names", field)
    if len(value) > MAX_UNTRUSTED_INPUT_TOOLS:
        raise _invalid(f"{field} holds at most {MAX_UNTRUSTED_INPUT_TOOLS} entries", field)
    out: list = []
    for i, entry in enumerate(value):
        name = entry.strip() if isinstance(entry, str) else ""
        if (
            not name or len(name) > 100 or name.lower() in ("all", "any")
            or not all(c.isalnum() or c in "_.-" for c in name)
        ):
            raise _invalid(f"{field}[{i}] must be one tool name, such as fetch_url (not all or a pattern)", f"{field}[{i}]")
        if name not in out:
            out.append(name)
    return out


def validate_notify(value: Any) -> Dict[str, Any]:
    """`{channels: [...]}`: where the automation's attention items are delivered (schema v2).

    `console` (the attention list every client reads) and/or `email` (the host's notification
    dispatcher mails the owner's registered address).
    """
    notify = _require_mapping(value if value is not None else {}, "notify")
    _reject_unknown(notify, ("channels",), "notify")
    channels = notify.get("channels", DEFAULT_NOTIFY["channels"])
    if not isinstance(channels, list) or not channels:
        raise _invalid(f"notify.channels must be a non-empty list of {list(NOTIFY_CHANNELS)}", "notify.channels")
    out = []
    for i, ch in enumerate(channels):
        if ch not in NOTIFY_CHANNELS:
            raise _invalid(f"notify.channels[{i}] must be one of {list(NOTIFY_CHANNELS)}", f"notify.channels[{i}]")
        if ch not in out:
            out.append(ch)
    return {"channels": [c for c in NOTIFY_CHANNELS if c in out]}


def validate_policy(value: Any) -> Dict[str, Any]:
    policy = _require_mapping(value if value is not None else {}, "policy")
    _reject_unknown(
        policy,
        ("serial", "misfire", "failure", "retry", "tool_approval", "email_allowed_recipients", "untrusted_input_tools"),
        "policy",
    )
    tool_approval = policy.get("tool_approval", "auto")
    if tool_approval not in TOOL_APPROVAL_MODES:
        raise _invalid(f"policy.tool_approval must be one of {list(TOOL_APPROVAL_MODES)}", "policy.tool_approval")
    for key, fixed in (("serial", True), ("misfire", "coalesce"), ("failure", "continue")):
        if key in policy and policy[key] != fixed:
            raise _invalid(f"policy.{key} must be {fixed!r} in v1", f"policy.{key}", reason_code="unsupported_feature")
    retry = _require_mapping(policy.get("retry", {}), "policy.retry")
    _reject_unknown(retry, ("max_attempts", "backoff"), "policy.retry")
    max_attempts = retry.get("max_attempts", DEFAULT_RETRY["max_attempts"])
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 10:
        raise _invalid("policy.retry.max_attempts must be an integer 1..10", "policy.retry.max_attempts")
    backoff = _require_mapping(retry.get("backoff", {}), "policy.retry.backoff")
    _reject_unknown(backoff, ("initial", "factor", "max"), "policy.retry.backoff")
    factor = backoff.get("factor", DEFAULT_RETRY["backoff"]["factor"])
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 1 <= factor <= 10:
        raise _invalid("policy.retry.backoff.factor must be a number 1..10", "policy.retry.backoff.factor")
    return {
        "serial": True,
        "misfire": "coalesce",
        "failure": "continue",
        "retry": {
            "max_attempts": max_attempts,
            "backoff": {
                "initial": _duration(backoff.get("initial", DEFAULT_RETRY["backoff"]["initial"]), "policy.retry.backoff.initial"),
                "factor": factor,
                "max": _duration(backoff.get("max", DEFAULT_RETRY["backoff"]["max"]), "policy.retry.backoff.max"),
            },
        },
        "tool_approval": tool_approval,
        "email_allowed_recipients": validate_email_allowed_recipients(
            policy.get("email_allowed_recipients", list(DEFAULT_EMAIL_ALLOWED_RECIPIENTS))
        ),
        "untrusted_input_tools": validate_untrusted_input_tools(
            policy.get("untrusted_input_tools", list(DEFAULT_UNTRUSTED_INPUT_TOOLS))
        ),
    }


def validate_workspace_root(value: Any) -> str:
    root = _nonempty_str(value, "workspace_root")
    if not os.path.isabs(root):
        raise _invalid("workspace_root must be an absolute path", "workspace_root")
    return root


def validate_trigger_request(value: Any, *, now: str, binding_id: str) -> Dict[str, Any]:
    """`{source_id, source_version, config}` -> TriggerBinding (config normalized)."""
    from ..triggers.registry import UnknownTriggerSource, get_trigger_adapter

    trig = _require_mapping(value, "trigger")
    _reject_unknown(trig, ("source_id", "source_version", "config"), "trigger")
    source_id = _nonempty_str(trig.get("source_id"), "trigger.source_id")
    version = trig.get("source_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise _invalid("trigger.source_version must be an integer", "trigger.source_version")
    try:
        adapter = get_trigger_adapter(source_id, version)
    except UnknownTriggerSource as exc:
        raise AutomationError(str(exc), reason_code="unknown_trigger_source", field="trigger.source_id") from exc
    try:
        config = adapter.validate(_require_mapping(trig.get("config", {}), "trigger.config"), now=now)
    except TriggerConfigError as exc:
        raise AutomationError(str(exc), reason_code=exc.reason_code, field=f"trigger.{exc.field}") from exc
    return {"binding_id": binding_id, "source_id": source_id, "source_version": version, "config": config}


CREATE_REQUEST_FIELDS = (
    "request_id", "tenant", "user", "title", "target", "trigger", "context", "policy", "notify", "workspace_root",
)


def build_definition(request: Mapping[str, Any], *, automation_id: str, now: str) -> Dict[str, Any]:
    """Validate a creation request and build revision 1 of `_meta.automation`."""
    req = _require_mapping(request, "request")
    _reject_unknown(req, CREATE_REQUEST_FIELDS, "")
    _nonempty_str(req.get("request_id"), "request_id")
    return {
        "schema_version": DEFINITION_SCHEMA_VERSION,
        "revision": 1,
        "title": validate_title(req.get("title")),
        "controller": {"bundle_ref": CONTROLLER_BUNDLE_REF, "flow_id": CONTROLLER_FLOW_ID},
        "target": validate_target(req.get("target")),
        "trigger": validate_trigger_request(req.get("trigger"), now=now, binding_id=binding_id_for(automation_id, 1)),
        "context": validate_context(req.get("context")),
        "policy": validate_policy(req.get("policy")),
        "notify": validate_notify(req.get("notify")),
        "session_id": automation_session_id(automation_id),
        "workspace_root": validate_workspace_root(req.get("workspace_root")),
        "created_at": now,
        "archived_at": None,
    }


REVISABLE_FIELDS = ("title", "target", "trigger", "context", "policy", "notify")


def revise_definition(
    definition: Mapping[str, Any], changes: Any, *, automation_id: str, now: str
) -> Dict[str, Any]:
    """The next revision of a definition with `changes` applied (each field replaced whole).

    A trigger whose source or config changes gets a new server-minted
    `binding_id`, so schedule event ids of the new revision never repeat old ones.
    """
    ch = _require_mapping(changes, "changes")
    _reject_unknown(ch, REVISABLE_FIELDS, "changes")
    if not ch:
        raise _invalid("changes must name at least one field", "changes")
    new = copy.deepcopy(dict(definition))
    revision = int(definition["revision"]) + 1
    new["revision"] = revision
    new["schema_version"] = DEFINITION_SCHEMA_VERSION
    new["notify"] = validate_notify(definition.get("notify"))  # a v1 definition gets the default
    if "title" in ch:
        new["title"] = validate_title(ch["title"])
    if "target" in ch:
        new["target"] = validate_target(ch["target"])
    if "context" in ch:
        new["context"] = validate_context(ch["context"])
    if "policy" in ch:
        # Merge: a field the client did not send keeps its current value (a
        # retry-only change must never reset tool_approval to its default).
        new["policy"] = validate_policy({**dict(definition["policy"]), **dict(_require_mapping(ch["policy"], "changes.policy"))})
    if "notify" in ch:
        new["notify"] = validate_notify(ch["notify"])
    if "trigger" in ch:
        old = definition["trigger"]
        candidate = validate_trigger_request(ch["trigger"], now=now, binding_id=old["binding_id"])
        same = (
            candidate["source_id"] == old["source_id"]
            and candidate["source_version"] == old["source_version"]
            and canonical_json(candidate["config"]) == canonical_json(old["config"])
        )
        if not same:
            candidate["binding_id"] = binding_id_for(automation_id, revision)
        new["trigger"] = candidate
    return new


def trigger_changed(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    return old["trigger"]["binding_id"] != new["trigger"]["binding_id"]


# --- controller state ---------------------------------------------------------


def initial_state(definition: Mapping[str, Any], *, trigger_state: Mapping[str, Any]) -> Dict[str, Any]:
    """`_runtime.automation` at creation (contract A defaults)."""
    state = {
        "state_version": 0,
        "active_revision": int(definition["revision"]),
        "pending_occurrence": None,
        "anchor": trigger_state.get("anchor"),
        "tick": int(trigger_state.get("tick") or 0),
        "next_index": 1,
        "scheduled_count": int(trigger_state.get("scheduled_count") or 0),
        "paused": False,
        "attention_seq": 0,
        "exhausted": bool(trigger_state.get("exhausted")),
        "manual_pending": None,
        "last_outcome": None,
        # Decision in flight (see automations.ledger): the key of the decision
        # about to be appended, so reconciliation is an exact key lookup.
        "intent": None,
    }
    if trigger_state.get("source_state") is not None:
        state["source_state"] = copy.deepcopy(dict(trigger_state["source_state"]))
    return state


def trigger_state_of(state: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "anchor": state.get("anchor"),
        "tick": int(state.get("tick") or 0),
        "scheduled_count": int(state.get("scheduled_count") or 0),
        "exhausted": bool(state.get("exhausted")),
    }
    if isinstance(state.get("source_state"), Mapping):
        out["source_state"] = copy.deepcopy(dict(state["source_state"]))
    return out


def definition_notify(definition: Mapping[str, Any]) -> Dict[str, Any]:
    """The definition's `notify` (a v1 definition: the default)."""
    raw = definition.get("notify")
    return validate_notify(raw) if raw is not None else copy.deepcopy(DEFAULT_NOTIFY)


def definition_email_allowed_recipients(definition: Mapping[str, Any]) -> list:
    """The definition's pre-authorised recipients (a v1 definition: `["self"]`)."""
    raw = (definition.get("policy") or {}).get("email_allowed_recipients")
    return validate_email_allowed_recipients(raw) if raw is not None else list(DEFAULT_EMAIL_ALLOWED_RECIPIENTS)


def definition_untrusted_input_tools(definition: Mapping[str, Any]) -> list:
    """The tools the user named for untrusted-input occurrences (older definitions: none)."""
    raw = (definition.get("policy") or {}).get("untrusted_input_tools")
    return validate_untrusted_input_tools(raw) if raw is not None else list(DEFAULT_UNTRUSTED_INPUT_TOOLS)


def backoff_delay(policy: Mapping[str, Any], attempt: int) -> timedelta:
    """Delay before attempt `attempt + 1`: min(initial * factor^(attempt-1), max)."""
    backoff = policy["retry"]["backoff"]
    initial = parse_duration(backoff["initial"], field="policy.retry.backoff.initial").total_seconds()
    cap = parse_duration(backoff["max"], field="policy.retry.backoff.max").total_seconds()
    raw = initial * math.pow(float(backoff["factor"]), max(0, int(attempt) - 1))
    return timedelta(seconds=min(raw, cap))


def add_delay(now: str, delay: timedelta) -> str:
    return format_timestamp(parse_timestamp(now, field="now") + delay)


AUTOMATION_STATUSES = ("active", "paused", "completed", "failed", "archived")


def automation_status(run: Any) -> str:
    """`archived | failed | completed | paused | active` (contract A) — the ONE rule.

    Archived (`archived_at` set) wins over everything, including a controller
    that ended when archived. Any other cancelled controller reads `failed`: the
    automation can never run again and nobody archived it. Used by
    `get_automation` and `automation_queries.list_automations`.
    """
    vars_obj = run.vars if isinstance(getattr(run, "vars", None), dict) else {}
    definition = (vars_obj.get("_meta") or {}).get("automation") or {}
    state = (vars_obj.get("_runtime") or {}).get("automation") or {}
    status = getattr(getattr(run, "status", None), "value", getattr(run, "status", None))
    if definition.get("archived_at"):
        return "archived"
    if status in ("failed", "cancelled"):
        return "failed"  # the controller itself ended abnormally (a cancelled controller never runs again)
    if status == "completed":
        return "completed"
    if state.get("paused"):
        return "paused"
    return "active"


__all__ = [
    "AUTOMATION_NAMESPACE",
    "AutomationDefinition",
    "AutomationError",
    "AutomationState",
    "AUTOMATION_STATUSES",
    "CONTROLLER_BUNDLE_ID",
    "CONTROLLER_BUNDLE_REF",
    "CONTROLLER_BUNDLE_VERSION",
    "CONTROLLER_FLOW_ID",
    "CONTROLLER_WORKFLOW_ID",
    "DEFAULT_EMAIL_ALLOWED_RECIPIENTS",
    "DEFAULT_NOTIFY",
    "DEFAULT_POLICY",
    "DEFINITION_SCHEMA_VERSION",
    "NOTIFY_CHANNELS",
    "SCHEMA_VERSION",
    "add_delay",
    "automation_id_for",
    "automation_session_id",
    "automation_status",
    "backoff_delay",
    "binding_id_for",
    "build_definition",
    "canonical_json",
    "definition_email_allowed_recipients",
    "definition_untrusted_input_tools",
    "definition_notify",
    "discussion_ids",
    "initial_state",
    "occurrence_run_id",
    "request_digest",
    "revise_definition",
    "trigger_changed",
    "trigger_state_of",
    "validate_context",
    "validate_email_allowed_recipients",
    "validate_untrusted_input_tools",
    "validate_notify",
    "validate_policy",
    "validate_target",
    "wake_wait_key",
]
