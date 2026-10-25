"""Trigger sources for automations (contract C): protocol, registry, built-ins
(`schedule@1`, `manual@1`, `email.received@1`), and per-user connections (`connections`)."""

from .connection_events import ConnectionEventAdapter
from .connections import (
    TriggerConnection,
    TriggerConnectionError,
    TriggerConnectionStore,
    append_connection_event,
    connection_principal,
    connection_stream,
    test_fire,
    validate_binding_connection,
    wake_connection_automations,
)
from .email_received import EmailReceivedTriggerAdapter
from .feed import FeedItemTriggerAdapter, FeedPoller, parse_feed, poll_feed_connections
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
from .schedule import ScheduleTriggerAdapter, ScheduleV2TriggerAdapter, time_zone_names, validate_time_zone

__all__ = [
    "BUILTIN_TRIGGER_SOURCES",
    "ConnectionEventAdapter",
    "TriggerConnection",
    "TriggerConnectionError",
    "TriggerConnectionStore",
    "append_connection_event",
    "connection_principal",
    "connection_stream",
    "test_fire",
    "validate_binding_connection",
    "wake_connection_automations",
    "ENTRY_POINT_GROUP",
    "EmailReceivedTriggerAdapter",
    "FeedItemTriggerAdapter",
    "FeedPoller",
    "parse_feed",
    "poll_feed_connections",
    "ManualTriggerAdapter",
    "ScheduleTriggerAdapter",
    "ScheduleV2TriggerAdapter",
    "time_zone_names",
    "validate_time_zone",
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
