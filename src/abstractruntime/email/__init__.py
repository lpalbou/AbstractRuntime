"""Email for runtimes (framework backlog 0992 WP2): one user, one runtime, one mailbox.

- `binding`: the run-scoped account binding and the host resolver seam (credentials are
  resolved by the host at the moment of use; they never enter run vars, the ledger or logs).
- `inbox`: the durable external-event inbox (0929's minimal core).
- `feeder`: mailbox -> inbox, the poll step a mail watcher runs (read-only, per-message cursor).
- `automations`: which automations consume mail, and how to wake them.
- `frame`: how inbound mail reaches an occurrence (structured, marked untrusted).
- `actions`: the send-email action (templates) for automations without a model.

The `email.received@1` trigger source lives in `abstractruntime.triggers.email_received`; the
`send_email_recipient@v2` approval refiner in `integrations.abstractcore.effect_handlers`.
"""

from .actions import (
    DIGEST_FIELDS,
    EACH_FIELDS,
    EMAIL_ACTION_BUNDLE_REF,
    EMAIL_ACTION_FLOW_ID,
    EMAIL_ACTION_KEY,
    EMAIL_ACTION_WORKFLOW_ID,
    EmailActionError,
    email_action_target,
    email_action_workflow_spec,
    plan_email_action,
    register_email_action_workflow,
    render_email_template,
    validate_email_action,
)
from .automations import email_trigger_consumers, wake_email_automations
from .binding import (
    EMAIL_ACCOUNT_KEY,
    EMAIL_ALLOWED_RECIPIENTS_KEY,
    EmailBinding,
    bind_email_account,
    binding_of,
    email_run_scope,
    install_core_resolver,
    resolve_email_context,
    strip_client_email_keys,
    uninstall_core_resolver,
)
from .feeder import EmailInboxFeeder, PollReport, email_event_id, email_event_payload, email_stream
from .frame import UNTRUSTED_NOTICE, email_frame, email_trigger_input
from .inbox import AppendResult, EventInbox, InMemoryEventInbox, JsonFileEventInbox

__all__ = [
    "AppendResult",
    "DIGEST_FIELDS",
    "EACH_FIELDS",
    "EMAIL_ACCOUNT_KEY",
    "EMAIL_ACTION_BUNDLE_REF",
    "EMAIL_ACTION_FLOW_ID",
    "EMAIL_ACTION_KEY",
    "EMAIL_ACTION_WORKFLOW_ID",
    "EMAIL_ALLOWED_RECIPIENTS_KEY",
    "EmailActionError",
    "EmailBinding",
    "EmailInboxFeeder",
    "EventInbox",
    "InMemoryEventInbox",
    "JsonFileEventInbox",
    "PollReport",
    "UNTRUSTED_NOTICE",
    "bind_email_account",
    "binding_of",
    "email_action_target",
    "email_action_workflow_spec",
    "email_event_id",
    "email_event_payload",
    "email_frame",
    "email_run_scope",
    "email_stream",
    "email_trigger_consumers",
    "email_trigger_input",
    "install_core_resolver",
    "plan_email_action",
    "register_email_action_workflow",
    "render_email_template",
    "resolve_email_context",
    "strip_client_email_keys",
    "uninstall_core_resolver",
    "validate_email_action",
    "wake_email_automations",
]
