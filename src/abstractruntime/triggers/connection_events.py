"""The shared adapter of connection-bound event sources (`feed.item@1`, `webhook.received@1`).

A connection-bound source reads the events its connection's feeder appended to the stream
`trigger:<connection_id>` (framework backlog 1040). Like every adapter it does no I/O: the
controller reads the stream after the automation's cursor and passes the records in (`events=`).

Binding config (validated strictly):

    {"connection_id": "<id>", "max_batch": 20, "start_at": <RFC3339, default now>}

- every new event of the stream (appended after `start_at`, and after a resume) is admitted
  once, oldest first; one admission carries every pending event up to `max_batch` (a feed poll
  that found five items becomes ONE occurrence: coalescing); the rest waits for the next one;
- the admission payload is `{count, event_ids, events: [<CloudEvents payloads>], first_seq,
  last_seq, content_trust}`; `events[i].data` is what the source delivered (a feed item, a
  webhook body) and is untrusted text unless the event says `abstracttrust: "trusted"` (a
  console test-fire).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .cloudevents import with_cloudevent_attributes
from .connections import check_connection_id, connection_source_uri
from .protocol import (
    TriggerAdmission,
    TriggerBinding,
    TriggerConfigError,
    TriggerEnvelope,
    TriggerState,
    TriggerWait,
    format_timestamp,
    parse_timestamp,
)

DEFAULT_MAX_BATCH = 20
MAX_BATCH = 200
_BINDING_KEYS = ("connection_id", "max_batch", "start_at")


def _parse_iso(value: Any):
    try:
        return parse_timestamp(value, field="time")
    except TriggerConfigError:
        return None


class ConnectionEventAdapter:
    """Base class: subclasses set `descriptor` (with `needs_connection: True`) and implement
    `validate_connection(config, *, now)` (and optionally `connection_secrets(config)`)."""

    descriptor: Dict[str, Any] = {}

    @property
    def source_id(self) -> str:
        return str(self.descriptor["id"])

    @property
    def source_version(self) -> int:
        return int(self.descriptor["version"])

    @staticmethod
    def binding_config_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["connection_id"],
            "properties": {
                "connection_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "max_batch": {"type": "integer", "minimum": 1, "maximum": MAX_BATCH, "default": DEFAULT_MAX_BATCH},
                "start_at": {"type": "string", "format": "date-time"},
            },
        }

    @staticmethod
    def batch_event_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["count", "event_ids", "events"],
            "properties": {
                "count": {"type": "integer", "minimum": 1},
                "event_ids": {"type": "array", "items": {"type": "string"}},
                "events": {"type": "array", "items": {"type": "object"}},
                "first_seq": {"type": "integer"},
                "last_seq": {"type": "integer"},
                "content_trust": {"type": "string", "enum": ["untrusted", "trusted"]},
            },
        }

    def occurrence_inputs(self, *, payload: Mapping[str, Any], inputs: Optional[Mapping[str, Any]] = None,
                          index: int, mailbox: Optional[str] = None) -> Dict[str, Any]:
        """Named occurrence inputs; each source defines its own (base: none)."""
        return {}

    # --- binding validation -----------------------------------------------------------------

    def validate(self, config: Mapping[str, Any], *, now: str) -> Dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TriggerConfigError(f"{self.source_id} config must be an object", field="config")
        unknown = sorted(k for k in config if k not in _BINDING_KEYS)
        if unknown:
            raise TriggerConfigError(f"unknown {self.source_id} field(s): {unknown}", field=f"config.{unknown[0]}")
        cid = check_connection_id(config.get("connection_id"), field="config.connection_id")
        max_batch = config.get("max_batch", DEFAULT_MAX_BATCH)
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or not 1 <= max_batch <= MAX_BATCH:
            raise TriggerConfigError(f"max_batch must be an integer 1..{MAX_BATCH}", field="config.max_batch")
        start = (
            parse_timestamp(config["start_at"], field="config.start_at")
            if config.get("start_at") is not None
            else parse_timestamp(now, field="now")
        )
        return {"connection_id": cid, "max_batch": max_batch, "start_at": format_timestamp(start)}

    def initial_state(self, config: Mapping[str, Any]) -> TriggerState:
        state: Dict[str, Any] = {"anchor": config.get("start_at"), "tick": 0, "scheduled_count": 0, "exhausted": False}
        state["source_state"] = {"cursor_seq": 0, "since": config.get("start_at")}
        return state  # type: ignore[return-value]

    # --- selection (pure) ---------------------------------------------------------------------

    @staticmethod
    def _src(state: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
        src = state.get("source_state") if isinstance(state.get("source_state"), Mapping) else {}
        return {"cursor_seq": int(src.get("cursor_seq") or 0), "since": src.get("since") or config.get("start_at")}

    def candidates(self, binding: TriggerBinding, *, state: Mapping[str, Any], events: Optional[Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
        config = binding["config"]
        src = self._src(state, config)
        since = _parse_iso(src["since"]) if src["since"] else None
        stream = f"trigger:{config['connection_id']}"
        out: List[Mapping[str, Any]] = []
        for rec in sorted(events or (), key=lambda r: int(r.get("seq") or 0)):
            if int(rec.get("seq") or 0) <= src["cursor_seq"] or rec.get("stream") != stream:
                continue
            appended = _parse_iso(rec.get("appended_at"))
            if since is not None and (appended is None or appended <= since):
                continue
            if isinstance(rec.get("payload"), Mapping):
                out.append(rec)
        return out

    def skip_ahead(self, binding: TriggerBinding, *, state: Mapping[str, Any], events: Optional[Sequence[Mapping[str, Any]]]) -> int:
        src = self._src(state, binding["config"])
        cands = self.candidates(binding, state=state, events=events)
        if cands:
            return max(src["cursor_seq"], int(cands[0]["seq"]) - 1)
        return max([src["cursor_seq"], *(int(r.get("seq") or 0) for r in events or ())])

    # --- protocol -----------------------------------------------------------------------------

    def prepare(self, binding: TriggerBinding, *, state: TriggerState, now: str, events: Optional[Sequence[Mapping[str, Any]]] = None) -> TriggerWait:
        if not self.candidates(binding, state=state, events=events):
            return {"kind": "idle"}  # woken by the feeder (wake_connection_automations)
        return {"kind": "until", "until": format_timestamp(parse_timestamp(now, field="now"))}

    def admit(self, binding: TriggerBinding, *, state: TriggerState, now: str, events: Optional[Sequence[Mapping[str, Any]]] = None) -> Optional[TriggerAdmission]:
        config = binding["config"]
        cands = self.candidates(binding, state=state, events=events)
        if not cands:
            return None
        src = self._src(state, config)
        batch = list(cands[: int(config.get("max_batch") or DEFAULT_MAX_BATCH)])
        if len(cands) > len(batch):
            cursor_seq = int(batch[-1]["seq"])
        else:
            cursor_seq = max([src["cursor_seq"], *(int(r.get("seq") or 0) for r in events or ())])
        event_ids = [str(r["event_id"]) for r in batch]
        digest = hashlib.sha256("\n".join(event_ids).encode("utf-8")).hexdigest()[:32]
        payloads = [dict(r["payload"]) for r in batch]
        trusted = all(p.get("abstracttrust") == "trusted" for p in payloads)
        fired_at = format_timestamp(parse_timestamp(now, field="now"))
        return {  # type: ignore[return-value]
            "event_id": f"{self.source_id}@{self.source_version}:{binding['binding_id']}:{digest}",
            "fired_at": fired_at,
            "payload": {
                "count": len(batch),
                "event_ids": event_ids,
                "first_seq": int(batch[0]["seq"]),
                "last_seq": int(batch[-1]["seq"]),
                "content_trust": "trusted" if trusted else "untrusted",
                "connection_id": config["connection_id"],
                "events": payloads,
            },
            "state": {
                "anchor": state.get("anchor"),
                "tick": int(state.get("tick") or 0) + 1,
                "scheduled_count": int(state.get("scheduled_count") or 0) + 1,
                "exhausted": False,
                "source_state": {"cursor_seq": cursor_seq, "since": src["since"], "last_admitted_at": fired_at},
            },
        }

    def rearm(self, binding: TriggerBinding, *, state: TriggerState, now: str) -> TriggerState:
        src = self._src(state, binding["config"])
        src["since"] = format_timestamp(parse_timestamp(now, field="now"))  # events up to now are never admitted
        return {  # type: ignore[return-value]
            "anchor": state.get("anchor"),
            "tick": int(state.get("tick") or 0),
            "scheduled_count": int(state.get("scheduled_count") or 0),
            "exhausted": False,
            "source_state": src,
        }

    def normalize(self, binding: TriggerBinding, *, event_id: str, fired_at: str, payload: Mapping[str, Any]) -> TriggerEnvelope:
        envelope: Dict[str, Any] = {
            "event_id": str(event_id),
            "source_id": self.source_id,
            "source_version": self.source_version,
            "fired_at": str(fired_at),
            "payload": dict(payload),
            "binding_id": str(binding["binding_id"]),
        }
        # Framework backlog 1039: one event -> its own subject (feed item id, webhook id); a batch
        # -> its inbox range. The connection is the CloudEvents `source`.
        events = payload.get("events") if isinstance(payload.get("events"), list) else []
        single = events[0] if len(events) == 1 and isinstance(events[0], Mapping) else None
        subject = str(single.get("subject")) if single and single.get("subject") else f"batch:{payload.get('first_seq')}-{payload.get('last_seq')}"
        cid = str(payload.get("connection_id") or (binding.get("config") or {}).get("connection_id") or "")
        return with_cloudevent_attributes(  # type: ignore[return-value]
            envelope,
            subject=subject,
            dedupekey=str(event_id),
            trust="trusted" if payload.get("content_trust") == "trusted" else "untrusted",
            source=connection_source_uri(cid) if cid else None,
            connectionid=cid or None,
            sequence=payload.get("last_seq") if isinstance(payload.get("last_seq"), int) else None,
        )


__all__ = ["ConnectionEventAdapter", "DEFAULT_MAX_BATCH", "MAX_BATCH"]
