"""The send-email action: an automation target that emails without a model (framework backlog 0992).

"Forward invoices", "tell me when X writes": an automation whose steps need no model targets
this workflow. It renders fixed templates from the trigger's emails and sends through the
ordinary `send_email` tool call, so everything that guards an agent's send guards it too:

- the approval gate and the `send_email_recipient@v2` refiner: unattended only when every
  recipient is "self" (the registered address) or pre-authorised in the automation definition;
  anything else waits for a person (`tool_approval`);
- AbstractCore's `guarded_send`: the account's recipient policy (allowlist / denylist over To,
  Cc and Bcc) and the send limits;
- the ledger records the call (recipients, subject, body), never a credential.

Input (`input_data.email_action`):

    {"to": ["self", "boss@example.test"], "cc": [], "subject": "New mail from {from}",
     "body": "{subject}\\n\\n{text}", "mode": "each" | "digest"}

`mode: each` (default) sends one message per email; `digest` one message for the batch.
Placeholders (a fixed list; anything else is refused when the action is validated):
each -> {from} {from_address} {to} {subject} {date} {text} {uid} {automation_title};
digest -> {count} {list} {automation_title}. `{{` and `}}` are literal braces. Rendered subjects
are one line. Field values are inserted as text (inbound content stays data).
"""

from __future__ import annotations

import string
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..core.models import Effect, EffectType, StepPlan
from ..core.spec import WorkflowSpec

EMAIL_ACTION_BUNDLE_ID = "abstractframework.email-actions"
EMAIL_ACTION_BUNDLE_VERSION = "1.0.0"
EMAIL_ACTION_BUNDLE_REF = f"{EMAIL_ACTION_BUNDLE_ID}@{EMAIL_ACTION_BUNDLE_VERSION}"
EMAIL_ACTION_FLOW_ID = "send_email"
EMAIL_ACTION_WORKFLOW_ID = f"{EMAIL_ACTION_BUNDLE_REF}:{EMAIL_ACTION_FLOW_ID}"
EMAIL_ACTION_KEY = "email_action"
RESULT_KEY = "email_action_result"

EACH_FIELDS = ("from", "from_address", "to", "subject", "date", "text", "uid", "automation_title")
DIGEST_FIELDS = ("count", "list", "automation_title")
MODES = ("each", "digest")
MAX_RECIPIENTS = 50
MAX_TEMPLATE_CHARS = 20_000


class EmailActionError(ValueError):
    """An invalid action: `code`, `cause`, `fix` (the automation's failure names both)."""

    def __init__(self, cause: str, fix: str, *, code: str = "email_action_invalid") -> None:
        super().__init__(f"{cause} Fix: {fix}")
        self.code = code
        self.cause = cause
        self.fix = fix

    def to_output(self) -> Dict[str, Any]:
        return {"success": False, "error": str(self), "error_code": self.code, "cause": self.cause, "fix": self.fix}


def _template_fields(template: str) -> List[str]:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise EmailActionError(
            f"The template {template[:60]!r} is not valid ({exc}).",
            "Write placeholders as {name}; write a literal brace as {{ or }}.",
        ) from None
    names: List[str] = []
    for _literal, field_name, spec, conversion in parsed:
        if field_name is None:
            continue
        if spec or conversion or not field_name or "." in field_name or "[" in field_name:
            raise EmailActionError(
                f"The placeholder {{{field_name}{('!' + conversion) if conversion else ''}{(':' + spec) if spec else ''}}} is not allowed.",
                "Use a plain placeholder such as {subject}, without formatting.",
            )
        names.append(field_name)
    return names


def render_email_template(template: str, fields: Mapping[str, Any], *, allowed: Sequence[str]) -> str:
    """Render `{name}` placeholders from `fields` (only names in `allowed`)."""
    for name in _template_fields(template):
        if name not in allowed:
            raise EmailActionError(
                f"The placeholder {{{name}}} is not available here.",
                f"Use one of: {', '.join('{' + n + '}' for n in allowed)}.",
            )
    return template.format_map({k: ("" if fields.get(k) is None else str(fields.get(k))) for k in allowed})


def _recipients(value: Any, field: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise EmailActionError(f"email_action.{field} must be a list of addresses.", f'Give email_action.{field} as ["self"] or ["name@example.test"].')
    out: List[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise EmailActionError(f"email_action.{field} holds an empty entry.", f"Remove it from email_action.{field}.")
        item = entry.strip()
        low = item.lower()
        if low != "self":
            local, at, domain = low.partition("@")
            if not at or not local or not domain or "@" in domain or any(c.isspace() or c in "<>,;:\"()[]" for c in low):
                raise EmailActionError(
                    f"email_action.{field} entry {item!r} is not a plain address.",
                    "Give recipients as name@example.test (or \"self\" for your registered address).",
                )
        if low not in out:
            out.append(low)
    if len(out) > MAX_RECIPIENTS:
        raise EmailActionError(f"email_action.{field} holds more than {MAX_RECIPIENTS} recipients.", "Send to fewer recipients.")
    return out


def validate_email_action(config: Any) -> Dict[str, Any]:
    """The normalized action (raises `EmailActionError`). Hosts call it when a client creates one."""
    if not isinstance(config, Mapping):
        raise EmailActionError("email_action must be an object.", 'Give {"to": ["self"], "subject": "...", "body": "..."}.')
    unknown = sorted(k for k in config if k not in ("to", "cc", "subject", "body", "mode"))
    if unknown:
        raise EmailActionError(f"email_action has unknown field(s): {unknown}.", "Use only to, cc, subject, body and mode.")
    mode = config.get("mode", "each")
    if mode not in MODES:
        raise EmailActionError(f"email_action.mode {mode!r} is not one of: each, digest.", "Use mode each or digest.")
    to = _recipients(config.get("to"), "to")
    cc = _recipients(config.get("cc"), "cc")
    if not to and not cc:
        raise EmailActionError("email_action has no recipient.", 'Give email_action.to, for example ["self"].')
    allowed = EACH_FIELDS if mode == "each" else DIGEST_FIELDS
    out: Dict[str, Any] = {"to": to, "cc": cc, "mode": mode}
    for key in ("subject", "body"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise EmailActionError(f"email_action.{key} is missing.", f"Give email_action.{key} (placeholders allowed).")
        if len(value) > MAX_TEMPLATE_CHARS:
            raise EmailActionError(f"email_action.{key} is longer than {MAX_TEMPLATE_CHARS} characters.", "Shorten the template.")
        for name in _template_fields(value):
            if name not in allowed:
                raise EmailActionError(
                    f"The placeholder {{{name}}} is not available in mode {mode}.",
                    f"Use one of: {', '.join('{' + n + '}' for n in allowed)}.",
                )
        out[key] = value
    return out


def _resolve_self(addresses: List[str], operator_email: Optional[str]) -> List[str]:
    out: List[str] = []
    for a in addresses:
        if a == "self":
            if not operator_email:
                raise EmailActionError(
                    "The action sends to \"self\", but this user has no registered email address.",
                    "Add your email address to your account (Settings), or name the recipient explicitly.",
                    code="email_action_no_self_address",
                )
            a = operator_email.strip().lower()
        if a not in out:
            out.append(a)
    return out


def _one_line(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def plan_email_action(
    config: Any,
    *,
    emails: Sequence[Mapping[str, Any]],
    operator_email: Optional[str],
    automation_title: str = "",
) -> List[Dict[str, Any]]:
    """The `send_email` argument dicts the action will issue (pure)."""
    action = validate_email_action(config)
    to = _resolve_self(action["to"], operator_email)
    cc = _resolve_self(action["cc"], operator_email)
    calls: List[Dict[str, Any]] = []

    def args(subject: str, body: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"to": list(to), "subject": _one_line(subject) or "(no subject)", "body_text": body}
        if cc:
            out["cc"] = list(cc)
        return out

    if action["mode"] == "each":
        for e in emails:
            fields = {
                "from": e.get("from"),
                "from_address": e.get("from_address"),
                "to": e.get("to"),
                "subject": e.get("subject"),
                "date": e.get("date"),
                "text": e.get("body_text") or "",
                "uid": e.get("uid"),
                "automation_title": automation_title,
            }
            calls.append(args(render_email_template(action["subject"], fields, allowed=EACH_FIELDS),
                              render_email_template(action["body"], fields, allowed=EACH_FIELDS)))
    elif emails:
        listing = "\n".join(f"- {e.get('from') or ''}: {e.get('subject') or '(no subject)'} ({e.get('date') or ''})" for e in emails)
        fields = {"count": len(emails), "list": listing, "automation_title": automation_title}
        calls.append(args(render_email_template(action["subject"], fields, allowed=DIGEST_FIELDS),
                          render_email_template(action["body"], fields, allowed=DIGEST_FIELDS)))
    return calls


# --- the workflow ------------------------------------------------------------------------


def _vars_runtime(run: Any) -> Dict[str, Any]:
    rt = (run.vars or {}).get("_runtime") if isinstance(getattr(run, "vars", None), dict) else None
    return rt if isinstance(rt, dict) else {}


def _plan_node(run: Any, ctx: Any) -> StepPlan:
    vars_obj = run.vars if isinstance(run.vars, dict) else {}
    trigger = vars_obj.get("trigger") if isinstance(vars_obj.get("trigger"), dict) else {}
    emails = [e for e in (trigger.get("emails") or []) if isinstance(e, dict)]
    meta = vars_obj.get("_meta") if isinstance(vars_obj.get("_meta"), dict) else {}
    try:
        calls = plan_email_action(
            vars_obj.get(EMAIL_ACTION_KEY),
            emails=emails,
            operator_email=_vars_runtime(run).get("operator_email"),
            automation_title=str(meta.get("automation_title") or ""),
        )
    except EmailActionError as err:
        return StepPlan(node_id="plan", complete_output=err.to_output())
    if not calls:
        return StepPlan(
            node_id="plan",
            complete_output={"success": True, "response": "No email to send: the occurrence carried no new message.", "sent": []},
        )
    tool_calls = [{"name": "send_email", "arguments": a, "call_id": f"email_action_{i}"} for i, a in enumerate(calls, start=1)]
    return StepPlan(
        node_id="plan",
        effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": tool_calls}, result_key=RESULT_KEY),
        next_node="done",
    )


def _done_node(run: Any, ctx: Any) -> StepPlan:
    result = (run.vars or {}).get(RESULT_KEY) if isinstance(run.vars, dict) else None
    results = (result or {}).get("results") if isinstance(result, dict) else None
    sent: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    refused: List[str] = []
    for r in results or []:
        out = r.get("output") if isinstance(r, dict) else None
        if isinstance(r, dict) and r.get("success") and isinstance(out, dict) and out.get("success"):
            sent.append({"message_id": out.get("message_id"), "to": out.get("to"), "subject": out.get("subject")})
            continue
        if isinstance(out, dict):
            failed.append({k: out.get(k) for k in ("error_code", "cause", "fix", "error")})
        else:
            # The send never ran: a person refused the approval, or the run's tool ceiling
            # blocked it. A decision, not a failure: retrying would only ask again.
            refused.append(str((r or {}).get("error") or "The send was not approved."))
    if refused and not failed:
        return StepPlan(
            node_id="done",
            complete_output={
                "success": True,
                "response": f"Sent {len(sent)} email(s); {len(refused)} not sent: {refused[0]}",
                "sent": sent,
                "not_sent": refused,
            },
        )
    if not failed:
        return StepPlan(node_id="done", complete_output={"success": True, "response": f"Sent {len(sent)} email(s).", "sent": sent})
    first = failed[0]
    reason = first.get("cause") or first.get("error") or "The send failed."
    fix = first.get("fix") or "Open the run to see each send's result."
    if sent:
        # Partial: the messages that went out are never sent again by a retry.
        return StepPlan(
            node_id="done",
            complete_output={
                "success": True,
                "response": f"Sent {len(sent)} email(s); {len(failed)} could not be sent: {reason} Fix: {fix}",
                "sent": sent,
                "failed": failed,
                "notify": {"title": "Some emails were not sent", "body": f"{len(failed)} could not be sent: {reason} Fix: {fix}"},
            },
        )
    return StepPlan(
        node_id="done",
        complete_output={
            "success": False,
            "error": f"No email was sent: {reason} Fix: {fix}",
            "error_code": first.get("error_code") or "email_action_send_failed",
            "cause": reason,
            "fix": fix,
            "failed": failed,
        },
    )


def email_action_workflow_spec() -> WorkflowSpec:
    return WorkflowSpec(
        workflow_id=EMAIL_ACTION_WORKFLOW_ID,
        entry_node="plan",
        nodes={"plan": _plan_node, "done": _done_node},
    )


def email_use_for_workflow(workflow: Any) -> str:
    """"action" when `workflow` is the runtime's own send-email action, else "agent_tool".

    Decided on the IDENTITY of the action's node function (`_plan_node`), never on a workflow
    id or a payload a bundle could copy: only the spec built by `email_action_workflow_spec`
    carries it. The host's resolver receives it as `use=` (see `abstractruntime.email.binding`).
    """
    from .binding import EMAIL_USE_ACTION, EMAIL_USE_AGENT_TOOL

    nodes = getattr(workflow, "nodes", None)
    if isinstance(nodes, dict) and nodes.get("plan") is _plan_node:
        return EMAIL_USE_ACTION
    return EMAIL_USE_AGENT_TOOL


def register_email_action_workflow(registry: Any) -> WorkflowSpec:
    """Register the action in a workflow registry (anything with `register(spec)`)."""
    spec = email_action_workflow_spec()
    registry.register(spec)
    return spec


def email_action_target(action: Mapping[str, Any]) -> Dict[str, Any]:
    """A ready automation `target` for this action (validated)."""
    return {
        "workflow_id": EMAIL_ACTION_WORKFLOW_ID,
        "bundle_ref": EMAIL_ACTION_BUNDLE_REF,
        "flow_id": EMAIL_ACTION_FLOW_ID,
        "input_data": {EMAIL_ACTION_KEY: validate_email_action(action)},
    }


__all__ = [
    "DIGEST_FIELDS",
    "EACH_FIELDS",
    "EMAIL_ACTION_BUNDLE_ID",
    "EMAIL_ACTION_BUNDLE_REF",
    "EMAIL_ACTION_BUNDLE_VERSION",
    "EMAIL_ACTION_FLOW_ID",
    "EMAIL_ACTION_KEY",
    "EMAIL_ACTION_WORKFLOW_ID",
    "EmailActionError",
    "email_action_target",
    "email_action_workflow_spec",
    "email_use_for_workflow",
    "plan_email_action",
    "register_email_action_workflow",
    "render_email_template",
    "validate_email_action",
]
