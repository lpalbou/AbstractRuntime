"""Trigger-source protocol (automations contract C).

A trigger source is a small, deterministic adapter: it validates its binding
configuration, owns a slice of the automation controller's state, and answers
three questions from persisted state and the current time only:

- `prepare`: what should the controller wait for next?
- `admit`: is an occurrence due now (and which one)?
- `rearm`: after a resume or a revision, where does the schedule restart
  (every tick at or before now is skipped, nothing fires)?

Adapters never perform I/O and never read the wall clock themselves: `now`
is always passed in, so the controller, the command applier and the tests see
the same answers for the same inputs.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional, Protocol, TypedDict, runtime_checkable


class TriggerSource(TypedDict):
    """Descriptor of a trigger source, as listed to hosts and apps."""

    id: str
    version: int
    label: str
    config_schema: Dict[str, Any]
    event_schema: Dict[str, Any]
    capabilities: Dict[str, Any]  # {"kind": "time" | "manual" | "event"}


class TriggerBinding(TypedDict):
    """One automation's use of a trigger source (stored in `_meta.automation.trigger`)."""

    binding_id: str
    source_id: str
    source_version: int
    config: Dict[str, Any]


class TriggerEnvelope(TypedDict):
    """The normalized event an occurrence was admitted for."""

    event_id: str
    source_id: str
    source_version: int
    fired_at: str
    payload: Dict[str, Any]
    binding_id: str


class TriggerState(TypedDict):
    """The trigger-owned slice of `_runtime.automation`."""

    anchor: Optional[str]
    tick: int
    scheduled_count: int
    exhausted: bool


class _TriggerWaitBase(TypedDict):
    kind: str


class TriggerWait(_TriggerWaitBase, total=False):
    """What the controller waits for: `until` (a deadline), `idle` (commands
    only), `exhausted` (no further admission can happen) or `event` (v2)."""

    until: str
    scope: str
    name: str


class _TriggerAdmissionBase(TypedDict):
    event_id: str
    fired_at: str
    payload: Dict[str, Any]
    state: TriggerState


class TriggerAdmission(_TriggerAdmissionBase, total=False):
    coalesced: Dict[str, int]  # {first_tick, last_tick, missed_count}


TRIGGER_STATE_KEYS = ("anchor", "tick", "scheduled_count", "exhausted")


class TriggerConfigError(ValueError):
    """Invalid trigger configuration.

    `reason_code` is `invalid_definition` or `unsupported_feature`; `field` is
    the dotted path of the offending value inside the trigger config.
    """

    def __init__(self, message: str, *, field: str, reason_code: str = "invalid_definition") -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.field = field


@runtime_checkable
class TriggerAdapter(Protocol):
    descriptor: TriggerSource

    def validate(self, config: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
        """Normalized config with defaults filled; raises TriggerConfigError."""
        ...

    def initial_state(self, config: Mapping[str, Any]) -> TriggerState: ...

    def prepare(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerWait: ...

    def admit(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> Optional[TriggerAdmission]:
        """The admission due at `now`, or None (nothing due: a stale wake)."""
        ...

    def rearm(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerState:
        """Resume/revise: skip every tick at or before `now`; never admits."""
        ...

    def normalize(
        self,
        binding: TriggerBinding,
        *,
        event_id: str,
        fired_at: str,
        payload: Mapping[str, Any],
    ) -> TriggerEnvelope: ...


# --- time helpers (UTC only; fixed-length units, no calendar/DST) ---------

_DURATION_RE = re.compile(r"^([1-9][0-9]*)([smhd])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: Any, *, field: str) -> timedelta:
    """`^[1-9][0-9]*[smhd]$` -> timedelta (s=1, m=60, h=3600, d=86400 seconds)."""
    if not isinstance(value, str):
        raise TriggerConfigError(f"{field} must be a duration string like '5m'", field=field)
    m = _DURATION_RE.match(value)
    if m is None:
        raise TriggerConfigError(
            f"{field} must match ^[1-9][0-9]*[smhd]$ (whole seconds, minutes, hours or days), got {value!r}",
            field=field,
        )
    return timedelta(seconds=int(m.group(1)) * _UNIT_SECONDS[m.group(2)])


def parse_timestamp(value: Any, *, field: str) -> datetime:
    """RFC3339 timestamp WITH an offset -> aware UTC datetime.

    Naive timestamps are refused: a schedule anchored in an unknown zone
    would silently shift by the host's offset.
    """
    if not isinstance(value, str) or not value.strip():
        raise TriggerConfigError(f"{field} must be an RFC3339 timestamp", field=field)
    raw = value.strip()
    candidate = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise TriggerConfigError(f"{field} is not an RFC3339 timestamp: {value!r}", field=field) from exc
    if dt.tzinfo is None:
        raise TriggerConfigError(f"{field} must carry a UTC offset (e.g. 'Z'): {value!r}", field=field)
    return dt.astimezone(timezone.utc)


def format_timestamp(dt: datetime) -> str:
    """Aware UTC isoformat (`+00:00`), the runtime's wait-deadline spelling."""
    return dt.astimezone(timezone.utc).isoformat()


def utc_now_iso() -> str:
    return format_timestamp(datetime.now(timezone.utc))


__all__ = [
    "TRIGGER_STATE_KEYS",
    "TriggerAdapter",
    "TriggerAdmission",
    "TriggerBinding",
    "TriggerConfigError",
    "TriggerEnvelope",
    "TriggerSource",
    "TriggerState",
    "TriggerWait",
    "format_timestamp",
    "parse_duration",
    "parse_timestamp",
    "utc_now_iso",
]
