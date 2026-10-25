"""Host helpers for email-triggered automations (framework backlog 0992 B4).

- `email_trigger_consumers(runtime)`: the active automations bound to `email.received@1` (a
  watcher runs only while there is at least one).
- `wake_email_automations(runtime)`: after the watcher appended mail, wake the idle
  controllers so they read the inbox now (a controller also re-reads it at every wake, so a
  missed wake only delays delivery until its next one; nothing is lost).
- `stream_consumers(runtime, stream)` / `wake_stream_consumers(runtime, stream)`: the same for
  ANY inbox stream (framework backlog 1042): `trigger:<connection_id>` (a webhook or feed
  connection), `email:<tenant>:<user>:<account_id>:<folder>` (one mailbox folder). Which
  automations read a stream is decided by their binding alone (`automation_reads_stream`).
- `prune_email_inbox(runtime, retention=...)`: apply the inbox retention (default 90 days AND
  10,000 events) without removing mail an active email automation has not read yet.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from ..triggers.email_received import SOURCE_ID, SOURCE_VERSION

EMAIL_STREAM_PREFIX = "email:"
TRIGGER_STREAM_PREFIX = "trigger:"
# A binding's `account` that names the runtime's own (first) mailbox: `self` (the 0992 form).
SELF_ACCOUNT = "self"
DEFAULT_ACCOUNT_ID = "default"


def email_stream_account_id(stream: str) -> Optional[str]:
    """The account id of an email stream `email:<tenant>:<user>:<account_id>:<folder>`, else None."""
    if not isinstance(stream, str) or not stream.startswith(EMAIL_STREAM_PREFIX):
        return None
    parts = stream.split(":", 4)
    return parts[3] if len(parts) >= 5 and parts[3] else None


def binding_account_id(trigger: Mapping[str, Any]) -> str:
    """The mailbox an `email.received@1` binding reads: its `config.account` (`self` and a
    missing value are the runtime's first mailbox, `default`)."""
    config = (trigger or {}).get("config") or {}
    account = config.get("account") if isinstance(config, Mapping) else None
    account = str(account or "").strip()
    return DEFAULT_ACCOUNT_ID if account in ("", SELF_ACCOUNT) else account


def automation_reads_stream(trigger: Mapping[str, Any], stream: str, *, self_account_ref: Optional[str] = None) -> bool:
    """Whether an automation bound by `trigger` reads inbox stream `stream`.

    - a connection-bound binding (`config.connection_id`) reads `trigger:<connection_id>` only;
    - an `email.received@1` binding reads the folders of its mailbox: `config.account` names an
      account id (`email:<tenant>:<user>:<account_id>:<folder>`); `self` / unset is the
      runtime's bound mailbox (`self_account_ref`, the runtime's `email_binding.account_ref`),
      `default` when the runtime has none;
    - nothing else reads an inbox stream.
    """
    if not isinstance(trigger, Mapping) or not isinstance(stream, str):
        return False
    config = trigger.get("config") or {}
    cid = config.get("connection_id") if isinstance(config, Mapping) else None
    if cid:
        return stream == f"{TRIGGER_STREAM_PREFIX}{cid}"
    if trigger.get("source_id") == SOURCE_ID and int(trigger.get("source_version") or 0) == SOURCE_VERSION:
        account = str((config.get("account") if isinstance(config, Mapping) else None) or "").strip()
        if account in ("", SELF_ACCOUNT) and self_account_ref:
            return stream.startswith(f"{EMAIL_STREAM_PREFIX}{self_account_ref}:")
        return email_stream_account_id(stream) == binding_account_id(trigger)
    return False


def _active_consumers(runtime: Any, wanted: Callable[[Mapping[str, Any]], bool]) -> List[str]:
    from ..automations.ledger import definition_of
    from ..automations.models import automation_status

    run_store = runtime.run_store
    out: List[str] = []
    for row in run_store.list_run_index(limit=1_000_000, role="controller"):
        run = run_store.load(str(row.get("run_id")))
        if run is None:
            continue
        try:
            trigger = definition_of(run)["trigger"]
        except (LookupError, KeyError, TypeError):
            continue
        if not isinstance(trigger, Mapping) or not wanted(trigger):
            continue
        if automation_status(run) == "active":
            out.append(run.run_id)
    return sorted(out)


def stream_consumers(runtime: Any, stream: str) -> List[str]:
    """Ids of active (not paused, archived or finished) automations that read inbox `stream`."""
    binding = getattr(runtime, "email_binding", None)
    self_ref = str(getattr(binding, "account_ref", "") or "") or None
    return _active_consumers(runtime, lambda trigger: automation_reads_stream(trigger, stream, self_account_ref=self_ref))


def _wake(runtime: Any, automation_ids: List[str], *, reason: str) -> List[str]:
    from ..automations.bundle import controller_workflow_spec
    from ..automations.models import wake_wait_key
    from ..core.models import RunStatus
    from ..core.runtime import StaleResumeError

    woken: List[str] = []
    for automation_id in automation_ids:
        run = runtime.run_store.load(automation_id)
        if run is None or run.status != RunStatus.WAITING or run.waiting is None:
            continue
        if run.waiting.wait_key != wake_wait_key(automation_id):
            continue  # an occurrence is running; the controller reads the inbox when it ends
        try:
            runtime.resume(
                workflow=controller_workflow_spec(),
                run_id=automation_id,
                wait_key=run.waiting.wait_key,
                payload={"wake": reason},
                max_steps=0,
            )
        except StaleResumeError:
            continue  # woken by someone else; it re-reads the inbox anyway
        woken.append(automation_id)
    return woken


def wake_stream_consumers(runtime: Any, stream: str) -> List[str]:
    """After a feeder appended to inbox `stream`, resume the wake wait of every idle automation
    that reads it (`stream_consumers`); returns the woken ids. Committed with `max_steps=0` (the
    host ticks the controller like any resumed run). A missed wake only delays delivery: a
    controller re-reads its stream at every wake."""
    return _wake(runtime, stream_consumers(runtime, stream), reason=str(stream))


def email_trigger_consumers(runtime: Any, *, account_id: Optional[str] = None) -> List[str]:
    """Ids of active (not paused, archived or finished) automations bound to `email.received@1`
    (to mailbox `account_id` when given; `self`/unset bindings read `default`)."""

    def wanted(trigger: Mapping[str, Any]) -> bool:
        if trigger.get("source_id") != SOURCE_ID or int(trigger.get("source_version") or 0) != SOURCE_VERSION:
            return False
        return account_id is None or binding_account_id(trigger) == account_id

    return _active_consumers(runtime, wanted)


def wake_email_automations(runtime: Any, *, account_id: Optional[str] = None) -> List[str]:
    """Resume the wake wait of every idle email-triggered automation (of mailbox `account_id`
    when given); returns the woken ids.

    The resume is committed with `max_steps=0` (the host ticks the controller like any
    resumed run), exactly as an automation command wakes it.
    """
    return _wake(runtime, email_trigger_consumers(runtime, account_id=account_id), reason="email.received")


def prune_email_inbox(runtime: Any, *, retention: Any = None, now: Any = None) -> Dict[str, Any]:
    """Apply the inbox retention without losing unread mail; returns the inbox's report.

    `retention`: an `EventInboxRetention`, a `{keep_days, keep_events}` dict, or None (the
    default: 90 days AND 10,000 events). Events an ACTIVE email automation has not read yet
    (past its `source_state.cursor_seq`) are never removed, nor events a connection-bound
    automation (webhook, feed) has not read yet; mail received while an automation
    is paused is not processed anyway, so paused automations do not hold events back. Raises
    `RuntimeError` when the runtime has no event inbox (`set_event_inbox`).
    """
    from ..automations.ledger import state_of

    inbox = getattr(runtime, "event_inbox", None)
    if inbox is None:
        raise RuntimeError("This runtime has no event inbox; call runtime.set_event_inbox(...) first.")
    head = int(inbox.head_seq())
    cursors: List[int] = []
    # Every automation that reads the inbox holds its unread events back: email-triggered ones
    # and connection-bound ones (`trigger:<connection_id>`, framework backlog 1042).
    readers = _active_consumers(
        runtime,
        lambda t: (t.get("source_id") == SOURCE_ID and int(t.get("source_version") or 0) == SOURCE_VERSION)
        or bool((t.get("config") or {}).get("connection_id") if isinstance(t.get("config"), Mapping) else False),
    )
    for automation_id in readers:
        run = runtime.run_store.load(automation_id)
        if run is None:
            continue
        try:
            src = state_of(run).get("source_state") or {}
        except LookupError:
            continue
        cursors.append(int(src.get("cursor_seq") or 0) if isinstance(src, dict) else 0)
    protect_after = min(cursors) if cursors else head
    report = dict(inbox.prune(retention=retention, protect_after_seq=protect_after, now=now))
    report["protect_after_seq"] = protect_after
    return report


__all__ = [
    "automation_reads_stream",
    "binding_account_id",
    "email_stream_account_id",
    "email_trigger_consumers",
    "prune_email_inbox",
    "stream_consumers",
    "wake_email_automations",
    "wake_stream_consumers",
]
