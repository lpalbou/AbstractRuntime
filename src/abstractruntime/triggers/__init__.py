"""Trigger sources for automations (contract C): protocol, registry, built-ins."""

from .manual import ManualTriggerAdapter, manual_event_id
from .protocol import (
    TRIGGER_STATE_KEYS,
    TriggerAdapter,
    TriggerAdmission,
    TriggerBinding,
    TriggerConfigError,
    TriggerEnvelope,
    TriggerSource,
    TriggerState,
    TriggerWait,
    format_timestamp,
    parse_duration,
    parse_timestamp,
)
from .registry import (
    BUILTIN_TRIGGER_SOURCES,
    ENTRY_POINT_GROUP,
    TriggerRegistryError,
    UnknownTriggerSource,
    get_trigger_adapter,
    reset_trigger_registry,
    trigger_sources,
)
from .schedule import ScheduleTriggerAdapter

__all__ = [
    "BUILTIN_TRIGGER_SOURCES",
    "ENTRY_POINT_GROUP",
    "ManualTriggerAdapter",
    "ScheduleTriggerAdapter",
    "TRIGGER_STATE_KEYS",
    "TriggerAdapter",
    "TriggerAdmission",
    "TriggerBinding",
    "TriggerConfigError",
    "TriggerEnvelope",
    "TriggerRegistryError",
    "TriggerSource",
    "TriggerState",
    "TriggerWait",
    "UnknownTriggerSource",
    "format_timestamp",
    "get_trigger_adapter",
    "manual_event_id",
    "parse_duration",
    "parse_timestamp",
    "reset_trigger_registry",
    "trigger_sources",
]
