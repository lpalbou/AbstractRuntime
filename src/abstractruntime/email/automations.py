"""Host helpers for email-triggered automations (framework backlog 0992 B4).

- `email_trigger_consumers(runtime)`: the active automations bound to `email.received@1` (a
  watcher runs only while there is at least one).
- `wake_email_automations(runtime)`: after the watcher appended mail, wake the idle
  controllers so they read the inbox now (a controller also re-reads it at every wake, so a
  missed wake only delays delivery until its next one; nothing is lost).
"""

from __future__ import annotations

from typing import Any, List

from ..triggers.email_received import SOURCE_ID, SOURCE_VERSION


def email_trigger_consumers(runtime: Any) -> List[str]:
    """Ids of active (not paused, archived or finished) automations bound to `email.received@1`."""
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
        if trigger.get("source_id") != SOURCE_ID or int(trigger.get("source_version") or 0) != SOURCE_VERSION:
            continue
        if automation_status(run) == "active":
            out.append(run.run_id)
    return sorted(out)


def wake_email_automations(runtime: Any) -> List[str]:
    """Resume the wake wait of every idle email-triggered automation; returns the woken ids.

    The resume is committed with `max_steps=0` (the host ticks the controller like any
    resumed run), exactly as an automation command wakes it.
    """
    from ..automations.bundle import controller_workflow_spec
    from ..automations.models import wake_wait_key
    from ..core.models import RunStatus
    from ..core.runtime import StaleResumeError

    woken: List[str] = []
    for automation_id in email_trigger_consumers(runtime):
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
                payload={"wake": "email.received"},
                max_steps=0,
            )
        except StaleResumeError:
            continue  # woken by someone else; it re-reads the inbox anyway
        woken.append(automation_id)
    return woken


__all__ = ["email_trigger_consumers", "wake_email_automations"]
