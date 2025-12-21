"""abstractruntime.integrations.abstractcore.observability

Bridge AbstractRuntime execution events onto AbstractCore's GlobalEventBus.

Why this module exists:
- ADR-0001: Runtime kernel must not import AbstractCore.
- ADR-0004: Prefer a unified event bus for real-time observability.

This adapter consumes runtime ledger append events (StepRecord dicts) and emits
workflow-step events to `abstractcore.events.GlobalEventBus`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from abstractcore.events import EventType, GlobalEventBus

from ...core.models import StepStatus


def _normalize_step_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    # StepStatus inherits from str, but be defensive.
    raw = getattr(value, "value", None)
    if isinstance(raw, str):
        return raw
    return str(value)


def _event_type_for_status(status: Optional[str]) -> Optional[EventType]:
    if status == StepStatus.STARTED.value:
        return EventType.WORKFLOW_STEP_STARTED
    if status == StepStatus.COMPLETED.value:
        return EventType.WORKFLOW_STEP_COMPLETED
    if status == StepStatus.WAITING.value:
        return EventType.WORKFLOW_STEP_WAITING
    if status == StepStatus.FAILED.value:
        return EventType.WORKFLOW_STEP_FAILED
    return None


def emit_step_record(record: Dict[str, Any], *, source: str = "abstractruntime") -> None:
    """Emit a GlobalEventBus event for a StepRecord dict."""
    status = _normalize_step_status(record.get("status"))
    event_type = _event_type_for_status(status)
    if event_type is None:
        return

    # Keep the top-level fields filterable without deep inspection.
    data = {
        "run_id": record.get("run_id"),
        "node_id": record.get("node_id"),
        "step_id": record.get("step_id"),
        "status": status,
        "record": record,
    }
    GlobalEventBus.emit(event_type, data, source=source)


def attach_global_event_bus_bridge(
    *,
    runtime: Any,
    run_id: Optional[str] = None,
    source: str = "abstractruntime",
) -> Callable[[], None]:
    """Subscribe to runtime ledger appends and emit to GlobalEventBus.

    Requires the configured runtime ledger store to support subscriptions
    (wrap it with `ObservableLedgerStore`).
    """

    def _on_record(rec: Dict[str, Any]) -> None:
        emit_step_record(rec, source=source)

    return runtime.subscribe_ledger(_on_record, run_id=run_id)

