"""`manual@1`: an automation that runs only when asked (`automation.run_now`).

The controller parks on its command wait with no deadline (`idle`); scheduled
admission never happens. Manual admissions of ANY automation (whatever its
trigger) are normalized through this adapter, so every "run now" occurrence
carries a `manual` envelope.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .protocol import (
    TriggerAdmission,
    TriggerBinding,
    TriggerConfigError,
    TriggerEnvelope,
    TriggerSource,
    TriggerState,
    TriggerWait,
)


def manual_event_id(command_id: str) -> str:
    return f"manual:{command_id}"


class ManualTriggerAdapter:
    descriptor: TriggerSource = {
        "id": "manual",
        "version": 1,
        "label": "Manual",
        "config_schema": {"type": "object", "additionalProperties": False, "properties": {}},
        "event_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command_id"],
            "properties": {"command_id": {"type": "string"}},
        },
        "capabilities": {"kind": "manual"},
    }

    def validate(self, config: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TriggerConfigError("manual config must be an object", field="config")
        if config:
            first = sorted(config)[0]
            raise TriggerConfigError(f"manual@1 takes no configuration (got {sorted(config)})", field=f"config.{first}")
        return {}

    def initial_state(self, config: Mapping[str, Any]) -> TriggerState:
        return {"anchor": None, "tick": 0, "scheduled_count": 0, "exhausted": False}

    def prepare(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerWait:
        return {"kind": "idle"}

    def admit(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> Optional[TriggerAdmission]:
        return None

    def rearm(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerState:
        return {
            "anchor": state.get("anchor"),
            "tick": int(state.get("tick") or 0),
            "scheduled_count": int(state.get("scheduled_count") or 0),
            "exhausted": False,
        }

    def normalize(
        self,
        binding: TriggerBinding,
        *,
        event_id: str,
        fired_at: str,
        payload: Mapping[str, Any],
    ) -> TriggerEnvelope:
        return {
            "event_id": str(event_id),
            "source_id": "manual",
            "source_version": 1,
            "fired_at": str(fired_at),
            "payload": dict(payload),
            "binding_id": str(binding["binding_id"]),
        }


__all__ = ["ManualTriggerAdapter", "manual_event_id"]
