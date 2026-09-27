"""`schedule@1`: fixed-interval UTC schedules (automations contract C).

Config: `{start_at?, every?, until?, count?, anchor?}`.

- Recurring (`every` set): ticks are `T_k = anchor + k*every` (k = 0, 1, ...),
  eligible while `T_k >= start_at`, `T_k < until` (exclusive) and fewer than
  `count` scheduled admissions happened. Whole units only (`[smhd]`), fixed
  lengths, UTC: there is no calendar or daylight-saving arithmetic, so "every
  24 hours" is exactly that.
- One-shot (no `every`): the only tick is `T_0 = start_at`; after its
  admission the binding is exhausted.
- Missed ticks coalesce: when several ticks are due at once (downtime, or a
  long occurrence), ONE admission happens for the latest due tick and the
  admission reports the skipped range.
- The anchor math lives only here; the `on_schedule` VisualFlow node keeps its
  own (drift-based) behaviour and is not used by automations.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Mapping, Optional

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
                "count": {"type": "integer", "minimum": 1},
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
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise TriggerConfigError("count must be an integer >= 1", field="config.count")
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

    def _next_tick(self, config: Mapping[str, Any], state: Mapping[str, Any]) -> Optional[int]:
        """The next eligible tick index at or after `state.tick`, or None (exhausted)."""
        if state.get("exhausted") or not self._count_left(config, state):
            return None
        tick = int(state.get("tick") or 0)
        anchor, every = self._grid(config, state)
        if every is None:
            return 0 if tick == 0 else None
        last = self._last_eligible_tick(config, anchor, every)
        if last is not None and tick > last:
            return None
        return tick

    @staticmethod
    def _tick_time(anchor: datetime, every: Optional[timedelta], k: int) -> datetime:
        return anchor if every is None else anchor + every * k

    def prepare(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerWait:
        config = binding["config"]
        k = self._next_tick(config, state)
        if k is None:
            return {"kind": "exhausted"}
        anchor, every = self._grid(config, state)
        return {"kind": "until", "until": format_timestamp(self._tick_time(anchor, every, k))}

    def admit(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> Optional[TriggerAdmission]:
        config = binding["config"]
        first = self._next_tick(config, state)
        if first is None:
            return None
        now_dt = parse_timestamp(now, field="now")
        anchor, every = self._grid(config, state)
        if every is None:
            if anchor > now_dt:
                return None
            fired = 0
        else:
            if now_dt < anchor:
                return None
            fired = (now_dt - anchor) // every  # largest k with T_k <= now
            last = self._last_eligible_tick(config, anchor, every)
            if last is not None:
                fired = min(fired, last)
            if fired < first:
                return None
        fired_at = format_timestamp(self._tick_time(anchor, every, fired))
        scheduled_count = int(state.get("scheduled_count") or 0) + 1
        new_state: TriggerState = {
            "anchor": format_timestamp(anchor),
            "tick": fired + 1,
            "scheduled_count": scheduled_count,
            "exhausted": False,
        }
        new_state["exhausted"] = self._next_tick(config, new_state) is None
        admission: TriggerAdmission = {
            "event_id": f"schedule@1:{binding['binding_id']}:{fired}",
            "fired_at": fired_at,
            "payload": {"tick": fired, "scheduled_at": fired_at},
            "state": new_state,
        }
        if fired > first:
            admission["coalesced"] = {"first_tick": first, "last_tick": fired, "missed_count": fired - first}
        return admission

    def rearm(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerState:
        config = binding["config"]
        anchor, every = self._grid(config, state)
        now_dt = parse_timestamp(now, field="now")
        tick = int(state.get("tick") or 0)
        if every is None:
            if tick == 0 and anchor <= now_dt:
                tick = 1  # the single tick passed while paused: never fired late
        elif now_dt >= anchor:
            tick = max(tick, (now_dt - anchor) // every + 1)  # smallest k with T_k > now
        new_state: TriggerState = {
            "anchor": format_timestamp(anchor),
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
            "source_version": 1,
            "fired_at": str(fired_at),
            "payload": dict(payload),
            "binding_id": str(binding["binding_id"]),
        }


__all__ = ["ScheduleTriggerAdapter"]
