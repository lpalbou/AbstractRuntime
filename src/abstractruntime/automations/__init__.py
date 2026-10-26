"""Automations (runtime 0847): an automation IS a durable controller root run.

- `models`: definition/state contracts, ids, validation.
- `ledger`: `automation.*` records and the decision protocol.
- `controller` / `adapters`: the controller flow's node logic and its
  VisualFlow adapters (`automation.<node_id>`).
- `bundle`: the packaged controller bundle
  `abstractframework.automation-controller@1.0.0`.
- `commands`: `apply_automation_command`, the only command applier.
- `attention`: notify convention, attention paging, human-wait detection.
- `service`: create / read / discuss / legacy projection / standalone driver.
- `consumers`: the active-consumers index (which active automations read which stream),
  one per runtime, rebuilt only when an automation changes.
"""

from .attention import (
    ANSWER_PAYLOADS,
    WAIT_KINDS,
    is_interactive_wait,
    list_attention,
    normalize_occurrence_output,
    notify_payload,
    pending_waits,
    pending_waits_by_automation,
    typed_wait,
    wait_kind,
)
from .bundle import (
    ControllerBundleError,
    controller_bundle_path,
    controller_workflow_spec,
    register_controller_bundle,
)
from .consumers import ActiveConsumersIndex, active_consumers_index, automation_generation, invalidate_active_consumers
from .commands import AUTOMATION_COMMAND_TYPES, apply_automation_command, record_automation_command_result
from .ledger import find_by_idempotency_key
from .models import (
    AUTOMATION_NAMESPACE,
    CONTROLLER_BUNDLE_REF,
    CONTROLLER_WORKFLOW_ID,
    AutomationDefinition,
    AutomationError,
    AutomationState,
    automation_status,
    occurrence_run_id,
)
from .service import (
    adopt_legacy_schedule_projection,
    create_automation,
    drive_automation,
    get_automation,
    latest_occurrence,
    list_occurrences,
    start_discussion,
)

__all__ = [
    "ANSWER_PAYLOADS",
    "ActiveConsumersIndex",
    "AUTOMATION_COMMAND_TYPES",
    "AUTOMATION_NAMESPACE",
    "AutomationDefinition",
    "AutomationError",
    "AutomationState",
    "CONTROLLER_BUNDLE_REF",
    "CONTROLLER_WORKFLOW_ID",
    "ControllerBundleError",
    "active_consumers_index",
    "adopt_legacy_schedule_projection",
    "automation_generation",
    "apply_automation_command",
    "automation_status",
    "controller_bundle_path",
    "controller_workflow_spec",
    "create_automation",
    "drive_automation",
    "find_by_idempotency_key",
    "get_automation",
    "invalidate_active_consumers",
    "is_interactive_wait",
    "latest_occurrence",
    "list_attention",
    "list_occurrences",
    "normalize_occurrence_output",
    "notify_payload",
    "occurrence_run_id",
    "pending_waits",
    "pending_waits_by_automation",
    "record_automation_command_result",
    "register_controller_bundle",
    "start_discussion",
    "typed_wait",
    "wait_kind",
    "WAIT_KINDS",
]
