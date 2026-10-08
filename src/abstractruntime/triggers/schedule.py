"""`schedule@1` (fixed-interval UTC) and `schedule@2` (+ calendar rules in a time zone).

Both versions share ONE tick model (an integer cursor over an ordered grid of
instants) and therefore one admission / coalescing / pause policy; they differ
only in how the grid is built. `schedule@1` is frozen: its configs normalize
byte-identically and its grid never changes (tests/test_schedule_v1_corpus.py).

## schedule@1

Config: `{start_at?, every?, until?, count?, anchor?}`.

- Recurring (`every` set): ticks are `T_k = anchor + k*every` (k = 0, 1, ...),
  eligible while `T_k >= start_at`, `T_k < until` (exclusive) and fewer than
  `count` scheduled admissions happened. Whole units only (`[smhd]`), fixed
  lengths, UTC: there is no calendar or daylight-saving arithmetic, so "every
  24 hours" is exactly that.
- One-shot (no `every`): the only tick is `T_0 = start_at`; after its
  admission the binding is exhausted.
- Missed ticks coalesce: when several ticks are due at once (downtime, or a
  long occurrence), ONE admission happens for the latest due tick; its event
  payload `{tick, scheduled_at, coalesced?: {first_tick, last_tick,
  missed_count}}` reports the skipped range.
- The anchor math lives only here; the `on_schedule` VisualFlow node keeps its
  own (drift-based) behaviour and is not used by automations.

## schedule@2 (R16.1)
Config `{kind, at?, days?, day?, time_zone?, start_at?, every?, until?, count?}`:

- `kind: "every"` / `"once"` = schedule@1 exactly (fixed UTC interval / one
  instant). A `once` may give its instant as a wall time `at: "YYYY-MM-DDTHH:MM"`
  in `time_zone` instead of `start_at`.
- `kind: "daily"` `{at: "HH:MM"}`, `"weekly"` `{days: ["mon".."sun"], at}`,
  `"monthly"` `{day: 1..31 | "last", at}` (a day beyond the month's length is
  that month's last day). Evaluated on WALL TIME in `time_zone` (IANA, required)
  and converted to UTC with zoneinfo: a wall time that does not exist that day
  (spring forward) runs at the shifted instant (02:30 -> 03:30), a repeated one
  (fall back) runs once, at its first occurrence. Tick 0 is the first rule time
  at or after `start_at` (default: now).
- Missed ticks coalesce exactly as in schedule@1 (one admission for the latest
  due tick); a resume after a pause skips the ticks that passed.
"""

from __future__ import annotations

import calendar
import functools
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from .protocol import (
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

_CONFIG_KEYS = ("start_at", "every", "until", "count", "anchor")
_CONFIG_KEYS_V2 = ("kind", "at", "days", "day", "time_zone") + _CONFIG_KEYS
MAX_COUNT = 1_000_000
SCHEDULE_KINDS = ("every", "once", "daily", "weekly", "monthly")
CALENDAR_KINDS = ("daily", "weekly", "monthly")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@functools.lru_cache(maxsize=1)
def _zone_names() -> frozenset:
    return frozenset(available_timezones())


def validate_time_zone(value: Any, *, field: str = "time_zone") -> str:
    """An IANA zone name this host's zoneinfo knows (e.g. "Europe/Paris", "UTC"), else TriggerConfigError."""
    if not isinstance(value, str) or not value.strip():
        raise TriggerConfigError(f"{field} must be an IANA time zone name such as 'Europe/Paris'", field=field)
    name = value.strip()
    if name not in _zone_names():
        raise TriggerConfigError(f"{field} {name!r} is not a time zone this gateway knows (IANA names such as 'Europe/Paris')", field=field)
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise TriggerConfigError(f"{field} {name!r} cannot be loaded: {exc}", field=field) from exc
    return name


def time_zone_names() -> List[str]:
    """Every IANA zone name this host knows, sorted (what a time-zone picker offers)."""
    return sorted(_zone_names())


def wall_to_utc(day: date, at: time, zone: ZoneInfo) -> datetime:
    """Wall time `day at` in `zone` -> aware UTC (fold=0: a skipped wall time maps forward by the
    gap, a repeated one to its FIRST instant)."""
    return datetime.combine(day, at, tzinfo=zone).astimezone(timezone.utc)


def _parse_hhmm(value: Any, *, field: str) -> time:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":" or not (value[:2] + value[3:]).isdigit():
        raise TriggerConfigError(f"{field} must be a 24-hour time 'HH:MM', got {value!r}", field=field)
    hh, mm = int(value[:2]), int(value[3:])
    if hh > 23 or mm > 59:
        raise TriggerConfigError(f"{field} must be a 24-hour time 'HH:MM' (00:00..23:59), got {value!r}", field=field)
    return time(hh, mm)


def _parse_local_datetime(value: Any, *, field: str) -> datetime:
    """`YYYY-MM-DDTHH:MM` (no offset: a wall time in the schedule's time zone) -> naive datetime."""
    if not isinstance(value, str) or len(value) != 16 or value[10] != "T":
        raise TriggerConfigError(f"{field} must be a wall time 'YYYY-MM-DDTHH:MM', got {value!r}", field=field)
    try:
        day = date.fromisoformat(value[:10])
    except ValueError as exc:
        raise TriggerConfigError(f"{field} must be a wall time 'YYYY-MM-DDTHH:MM', got {value!r}", field=field) from exc
    return datetime.combine(day, _parse_hhmm(value[11:], field=field))


class _CalendarGrid:
    """Calendar ticks: candidate n (an integer, possibly negative, counted from an epoch near
    `start_at`) is a local wall time; tick k = candidate n0 + k, n0 = the first candidate at or
    after `start_at`. Every instant is computed in O(1) from its index (no iteration over time)."""

    one_shot = False

    def __init__(self, config: Mapping[str, Any], start: datetime) -> None:
        self.anchor = start
        self.kind = str(config["kind"])
        self.zone = ZoneInfo(str(config["time_zone"]))
        self.at = _parse_hhmm(config["at"], field="config.at")
        local_start = start.astimezone(self.zone).date()
        if self.kind == "weekly":
            self.days = [WEEKDAYS.index(d) for d in config["days"]]
            self.epoch = local_start - timedelta(days=local_start.weekday())
        elif self.kind == "monthly":
            self.day = config["day"]
            self.epoch_month = local_start.year * 12 + (local_start.month - 1)
        else:
            self.epoch = local_start
        n = self._estimate(start)
        while self._candidate(n) < start:
            n += 1
        while self._candidate(n - 1) >= start:
            n -= 1
        self.n0 = n

    def _candidate_day(self, n: int) -> date:
        if self.kind == "daily":
            return self.epoch + timedelta(days=n)
        if self.kind == "weekly":
            weeks, pos = divmod(n, len(self.days))
            return self.epoch + timedelta(days=7 * weeks + self.days[pos])
        year, month0 = divmod(self.epoch_month + n, 12)
        last = calendar.monthrange(year, month0 + 1)[1]
        dom = last if self.day == "last" else min(int(self.day), last)
        return date(year, month0 + 1, dom)

    def _candidate(self, n: int) -> datetime:
        return wall_to_utc(self._candidate_day(n), self.at, self.zone)

    def _estimate(self, t: datetime) -> int:
        """A candidate index on (or one off) the local date of `t`."""
        local = t.astimezone(self.zone).date()
        if self.kind == "daily":
            return (local - self.epoch).days
        if self.kind == "weekly":
            weeks = (local - self.epoch).days // 7
            return weeks * len(self.days) + sum(1 for d in self.days if d <= local.weekday()) - 1
        return local.year * 12 + (local.month - 1) - self.epoch_month

    def time(self, k: int) -> datetime:
        return self._candidate(self.n0 + k)

    def floor(self, t: datetime) -> int:
        """Largest k >= 0 with T_k <= t, or -1 when t is before T_0."""
        n = self._estimate(t)
        while self._candidate(n + 1) <= t:
            n += 1
        while n >= self.n0 and self._candidate(n) > t:
            n -= 1
        return max(n - self.n0, -1)


class _FixedGrid:
    """schedule@1's grid: T_k = anchor + k*every (one-shot: T_0 = anchor only)."""

    def __init__(self, anchor: datetime, every: Optional[timedelta]) -> None:
        self.anchor = anchor
        self.every = every
        self.one_shot = every is None

    def time(self, k: int) -> datetime:
        return self.anchor if self.every is None else self.anchor + self.every * k

    def floor(self, t: datetime) -> int:
        if self.every is None:
            return 0 if t >= self.anchor else -1
        return -1 if t < self.anchor else (t - self.anchor) // self.every


class ScheduleTriggerAdapter:
    descriptor: TriggerSource = {
        "id": "schedule",
        "version": 1,
        "label": "Schedule",
        "config_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "start_at": {"type": "string", "format": "date-time"},
                "every": {"type": "string", "format": "duration", "pattern": "^[1-9][0-9]*[smhd]$"},
                "until": {"type": "string", "format": "date-time"},
                "count": {"type": "integer", "minimum": 1, "maximum": 1000000},
                "anchor": {"type": "string", "format": "date-time"},
            },
        },
        "event_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["tick", "scheduled_at"],
            "properties": {
                "tick": {"type": "integer", "minimum": 0},
                "scheduled_at": {"type": "string", "format": "date-time"},
                "coalesced": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["first_tick", "last_tick", "missed_count"],
                    "properties": {
                        "first_tick": {"type": "integer", "minimum": 0},
                        "last_tick": {"type": "integer", "minimum": 0},
                        "missed_count": {"type": "integer", "minimum": 1},
                    },
                },
            },
        },
        "capabilities": {"kind": "time"},
    }

    # --- validation -------------------------------------------------------

    def validate(self, config: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TriggerConfigError("schedule config must be an object", field="config")
        unknown = sorted(k for k in config if k not in _CONFIG_KEYS)
        if unknown:
            raise TriggerConfigError(f"unknown schedule field(s): {unknown}", field=f"config.{unknown[0]}")

        now_dt = parse_timestamp(now, field="now")
        start = parse_timestamp(config["start_at"], field="config.start_at") if config.get("start_at") is not None else now_dt
        anchor = parse_timestamp(config["anchor"], field="config.anchor") if config.get("anchor") is not None else start
        if anchor != start:
            raise TriggerConfigError(
                "schedule@1 requires anchor == start_at (a separate anchor is not supported in v1)",
                field="config.anchor",
                reason_code="unsupported_feature",
            )
        out: Dict[str, Any] = {"start_at": format_timestamp(start), "anchor": format_timestamp(anchor)}

        if config.get("every") is not None:
            parse_duration(config["every"], field="config.every")
            out["every"] = config["every"]

        if config.get("count") is not None:
            count = config["count"]
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
                raise TriggerConfigError(f"count must be an integer 1..{MAX_COUNT}", field="config.count")
            if "every" not in out and count != 1:
                raise TriggerConfigError("count > 1 requires every", field="config.count")
            out["count"] = count

        if config.get("until") is not None:
            until = parse_timestamp(config["until"], field="config.until")
            if until <= start:
                raise TriggerConfigError("until must be after start_at", field="config.until")
            out["until"] = format_timestamp(until)
        return out

    def initial_state(self, config: Mapping[str, Any]) -> TriggerState:
        return {"anchor": config["anchor"], "tick": 0, "scheduled_count": 0, "exhausted": False}

    # --- schedule math ----------------------------------------------------

    @staticmethod
    def _grid(config: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[datetime, Optional[timedelta]]:
        anchor = parse_timestamp(state.get("anchor") or config["anchor"], field="state.anchor")
        every = parse_duration(config["every"], field="config.every") if config.get("every") else None
        return anchor, every

    @staticmethod
    def _last_eligible_tick(config: Mapping[str, Any], anchor: datetime, every: timedelta) -> Optional[int]:
        """Largest k with T_k < until (None when unbounded)."""
        if not config.get("until"):
            return None
        until = parse_timestamp(config["until"], field="config.until")
        span = until - anchor
        # ceil(span / every) - 1, in whole microseconds (exact integer math).
        k = -((-(span // timedelta(microseconds=1))) // (every // timedelta(microseconds=1))) - 1
        return k

    @staticmethod
    def _count_left(config: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
        count = config.get("count")
        return count is None or int(state.get("scheduled_count") or 0) < int(count)

    def _grid_of(self, config: Mapping[str, Any], state: Mapping[str, Any]) -> "_FixedGrid | _CalendarGrid":
        anchor, every = self._grid(config, state)
        return _FixedGrid(anchor, every)

    def _last_eligible(self, config: Mapping[str, Any], grid: Any) -> Optional[int]:
        """Largest k with T_k < until (None when unbounded)."""
        if isinstance(grid, _FixedGrid):
            return None if grid.every is None else self._last_eligible_tick(config, grid.anchor, grid.every)
        if not config.get("until"):
            return None
        until = parse_timestamp(config["until"], field="config.until")
        k = grid.floor(until)
        if k >= 0 and grid.time(k) >= until:
            k -= 1
        return k

    def _next_tick(self, config: Mapping[str, Any], state: Mapping[str, Any]) -> Optional[int]:
        """The next eligible tick index at or after `state.tick`, or None (exhausted)."""
        if state.get("exhausted") or not self._count_left(config, state):
            return None
        tick = int(state.get("tick") or 0)
        grid = self._grid_of(config, state)
        if grid.one_shot:
            return 0 if tick == 0 else None
        last = self._last_eligible(config, grid)
        if last is not None and tick > last:
            return None
        return tick

    @staticmethod
    def _tick_time(anchor: datetime, every: Optional[timedelta], k: int) -> datetime:
        return anchor if every is None else anchor + every * k

    def _event_prefix(self) -> str:
        return f"schedule@{self.descriptor['version']}"

    def prepare(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerWait:
        config = binding["config"]
        k = self._next_tick(config, state)
        if k is None:
            return {"kind": "exhausted"}
        grid = self._grid_of(config, state)
        return {"kind": "until", "until": format_timestamp(grid.time(k))}

    def admit(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> Optional[TriggerAdmission]:
        config = binding["config"]
        first = self._next_tick(config, state)
        if first is None:
            return None
        now_dt = parse_timestamp(now, field="now")
        grid = self._grid_of(config, state)
        if grid.one_shot:
            if grid.anchor > now_dt:
                return None
            fired = 0
        else:
            fired = grid.floor(now_dt)  # largest k with T_k <= now (-1: before the first tick)
            if fired < 0:
                return None
            last = self._last_eligible(config, grid)
            if last is not None:
                fired = min(fired, last)
            if fired < first:
                return None
        fired_at = format_timestamp(grid.time(fired))
        scheduled_count = int(state.get("scheduled_count") or 0) + 1
        new_state: TriggerState = {
            "anchor": format_timestamp(grid.anchor),
            "tick": fired + 1,
            "scheduled_count": scheduled_count,
            "exhausted": False,
        }
        new_state["exhausted"] = self._next_tick(config, new_state) is None
        admission: TriggerAdmission = {
            "event_id": f"{self._event_prefix()}:{binding['binding_id']}:{fired}",
            "fired_at": fired_at,
            "payload": {"tick": fired, "scheduled_at": fired_at},
            "state": new_state,
        }
        if fired > first:
            coalesced = {"first_tick": first, "last_tick": fired, "missed_count": fired - first}
            admission["coalesced"] = coalesced
            admission["payload"]["coalesced"] = dict(coalesced)
        return admission

    def rearm(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerState:
        config = binding["config"]
        grid = self._grid_of(config, state)
        now_dt = parse_timestamp(now, field="now")
        tick = int(state.get("tick") or 0)
        if grid.one_shot:
            if tick == 0 and grid.anchor <= now_dt:
                tick = 1  # the single tick passed while paused: never fired late
        else:
            passed = grid.floor(now_dt)
            if passed >= 0:
                tick = max(tick, passed + 1)  # smallest k with T_k > now
        new_state: TriggerState = {
            "anchor": format_timestamp(grid.anchor),
            "tick": tick,
            "scheduled_count": int(state.get("scheduled_count") or 0),
            "exhausted": False,
        }
        new_state["exhausted"] = self._next_tick(config, new_state) is None
        return new_state

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
            "source_id": "schedule",
            "source_version": int(self.descriptor["version"]),
            "fired_at": str(fired_at),
            "payload": dict(payload),
            "binding_id": str(binding["binding_id"]),
        }


_V2_PROPERTIES: Dict[str, Any] = {
    "kind": {"type": "string", "enum": list(SCHEDULE_KINDS)},
    "at": {"type": "string", "description": "daily/weekly/monthly: 'HH:MM'; once: 'YYYY-MM-DDTHH:MM' (wall time in time_zone)"},
    "days": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "enum": list(WEEKDAYS)}},
    "day": {"oneOf": [{"type": "integer", "minimum": 1, "maximum": 31}, {"type": "string", "enum": ["last"]}]},
    "time_zone": {"type": "string", "description": "IANA time zone name, e.g. Europe/Paris"},
    **ScheduleTriggerAdapter.descriptor["config_schema"]["properties"],
}


class ScheduleV2TriggerAdapter(ScheduleTriggerAdapter):
    """`schedule@2`: schedule@1's `every`/`once` plus daily/weekly/monthly rules in a time zone."""

    descriptor: TriggerSource = {
        **ScheduleTriggerAdapter.descriptor,
        "version": 2,
        "label": "Schedule",
        "config_schema": {"type": "object", "additionalProperties": False, "properties": _V2_PROPERTIES},
    }

    def validate(self, config: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TriggerConfigError("schedule config must be an object", field="config")
        unknown = sorted(k for k in config if k not in _CONFIG_KEYS_V2)
        if unknown:
            raise TriggerConfigError(f"unknown schedule field(s): {unknown}", field=f"config.{unknown[0]}")
        kind = config.get("kind")
        if kind is None:
            kind = "every" if config.get("every") is not None else "once"
        if kind not in SCHEDULE_KINDS:
            raise TriggerConfigError(f"kind must be one of {'|'.join(SCHEDULE_KINDS)}, got {kind!r}", field="config.kind")
        zone = validate_time_zone(config["time_zone"], field="config.time_zone") if config.get("time_zone") is not None else None
        out: Dict[str, Any] = {"kind": kind}

        if kind in ("every", "once"):
            for key in ("days", "day"):
                if config.get(key) is not None:
                    raise TriggerConfigError(f"{key} is only used by weekly/monthly schedules", field=f"config.{key}")
            v1 = {k: config[k] for k in _CONFIG_KEYS if config.get(k) is not None}
            if kind == "every" and v1.get("every") is None:
                raise TriggerConfigError("an 'every' schedule needs every (e.g. '8h')", field="config.every")
            if kind == "once":
                if v1.get("every") is not None:
                    raise TriggerConfigError("a 'once' schedule has no every", field="config.every")
                if config.get("at") is not None:
                    if zone is None:
                        raise TriggerConfigError("a 'once' schedule with a wall time 'at' needs time_zone", field="config.time_zone")
                    wall = _parse_local_datetime(config["at"], field="config.at")
                    start = wall_to_utc(wall.date(), wall.time(), ZoneInfo(zone))
                    v1["start_at"] = format_timestamp(start)
                    v1.pop("anchor", None)
                    out["at"] = config["at"]
            elif config.get("at") is not None:
                raise TriggerConfigError("an 'every' schedule has no at (it is a fixed UTC interval from start_at)", field="config.at")
            if zone is not None:
                out["time_zone"] = zone
            out.update(super().validate(v1, now=now))
            return out

        # Calendar rules.
        if config.get("every") is not None:
            raise TriggerConfigError(f"a {kind} schedule has no every", field="config.every")
        if zone is None:
            raise TriggerConfigError(f"a {kind} schedule needs time_zone (IANA, e.g. 'Europe/Paris')", field="config.time_zone")
        if config.get("at") is None:
            raise TriggerConfigError(f"a {kind} schedule needs at ('HH:MM')", field="config.at")
        _parse_hhmm(config["at"], field="config.at")
        out["at"] = config["at"]
        if kind == "weekly":
            days = config.get("days")
            if not isinstance(days, (list, tuple)) or not days:
                raise TriggerConfigError("a weekly schedule needs days, e.g. ['mon', 'thu']", field="config.days")
            bad = [d for d in days if d not in WEEKDAYS]
            if bad:
                raise TriggerConfigError(f"days must be among {', '.join(WEEKDAYS)}, got {bad[0]!r}", field="config.days")
            out["days"] = [d for d in WEEKDAYS if d in days]
        elif config.get("days") is not None:
            raise TriggerConfigError("days is only used by weekly schedules", field="config.days")
        if kind == "monthly":
            day = config.get("day")
            if day != "last" and (isinstance(day, bool) or not isinstance(day, int) or not 1 <= day <= 31):
                raise TriggerConfigError("a monthly schedule needs day: 1..31 or 'last'", field="config.day")
            out["day"] = day
        elif config.get("day") is not None:
            raise TriggerConfigError("day is only used by monthly schedules", field="config.day")
        out["time_zone"] = zone
        now_dt = parse_timestamp(now, field="now")
        start = parse_timestamp(config["start_at"], field="config.start_at") if config.get("start_at") is not None else now_dt
        anchor = parse_timestamp(config["anchor"], field="config.anchor") if config.get("anchor") is not None else start
        if anchor != start:
            raise TriggerConfigError("schedule@2 requires anchor == start_at", field="config.anchor", reason_code="unsupported_feature")
        out["start_at"] = format_timestamp(start)
        out["anchor"] = format_timestamp(anchor)
        if config.get("count") is not None:
            count = config["count"]
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
                raise TriggerConfigError(f"count must be an integer 1..{MAX_COUNT}", field="config.count")
            out["count"] = count
        if config.get("until") is not None:
            until = parse_timestamp(config["until"], field="config.until")
            if until <= start:
                raise TriggerConfigError("until must be after start_at", field="config.until")
            out["until"] = format_timestamp(until)
        return out

    def _grid_of(self, config: Mapping[str, Any], state: Mapping[str, Any]) -> "_FixedGrid | _CalendarGrid":
        if config.get("kind") in CALENDAR_KINDS:
            anchor = parse_timestamp(state.get("anchor") or config["anchor"], field="state.anchor")
            return _CalendarGrid(config, anchor)
        return super()._grid_of(config, state)


__all__ = [
    "CALENDAR_KINDS",
    "SCHEDULE_KINDS",
    "WEEKDAYS",
    "ScheduleTriggerAdapter",
    "ScheduleV2TriggerAdapter",
    "time_zone_names",
    "validate_time_zone",
    "wall_to_utc",
]
