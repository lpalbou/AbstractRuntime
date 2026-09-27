"""Automation ledger records and the decision protocol (contract A, amendment 7).

Every controller transition and every command application is a DECISION:

1. reconcile: finish a decision that was appended but whose state change was
   not saved (a crash between the two);
2. look the decision's idempotency key up exactly — present means it was
   already decided: return it, append nothing;
3. decide from the persisted state;
4. record the INTENT (`_runtime.automation.intent = {key, state_version}`) and
   save the run;
5. append the decision record (it carries `state_version = n + 1` and the full
   state `delta`);
6. apply the delta, set `state_version = n + 1`, clear the intent, save.

A crash after (4) leaves an intent whose key is either absent from the ledger
(the decision never happened: the intent is dropped) or present (the delta is
applied). So reconciliation is one exact key lookup, never a ledger scan, and
at most one decision can ever be outstanding. Keys are checked, never assumed
unique: the SQLite ledger does not enforce idempotency-key uniqueness.

Records are ordinary StepRecords: `effect.type = "emit_event"`,
`effect.payload = {name, payload}`, `idempotency_key =
"automation:<name>:<automation_id>:<discriminator>"`.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional

from ..core.models import RunState, StepRecord, StepStatus
from .models import SCHEMA_VERSION

EMIT_EVENT = "emit_event"
RECORD_PREFIX = "automation."


def record_key(name: str, automation_id: str, discriminator: Any) -> str:
    """`automation:<name>:<automation_id>:<discriminator>` (name without the `automation.` prefix)."""
    short = name[len(RECORD_PREFIX):] if name.startswith(RECORD_PREFIX) else name
    return f"automation:{short}:{automation_id}:{discriminator}"


# --- exact key lookup -----------------------------------------------------------


def _innermost(store: Any) -> Any:
    seen = 0
    cur = store
    while seen < 8:
        inner = getattr(cur, "_inner", None)
        if inner is None:
            return cur
        cur = inner
        seen += 1
    return cur


def _has_exact_point_lookup(store: Any) -> bool:
    from ..storage.sqlite import SqliteLedgerStore

    return isinstance(_innermost(store), SqliteLedgerStore)


def find_by_idempotency_key(ledger_store: Any, run_id: str, key: str) -> Optional[Dict[str, Any]]:
    """The ledger record of `run_id` whose idempotency key is exactly `key`, or None.

    Exact, unlike `LedgerStore.find_completed_result` (a tail-window probe
    for effect replay): SQLite ledgers answer "is it there?" with their indexed
    point query and are read in full only on a hit (rare: crash recovery or a
    replayed command); other stores are scanned in full. The oldest matching
    record wins, as for effect replay.
    """
    if _has_exact_point_lookup(ledger_store):
        if ledger_store.find_completed_result(run_id, key) is None:
            return None
    for record in ledger_store.list(run_id):
        if isinstance(record, dict) and record.get("idempotency_key") == key and record.get("status") == StepStatus.COMPLETED.value:
            return record
    return None


def record_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    effect = record.get("effect") or {}
    payload = (effect.get("payload") or {}).get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"ledger record {record.get('idempotency_key')!r} is not an automation record")
    return payload


def record_name(record: Mapping[str, Any]) -> Optional[str]:
    effect = record.get("effect") if isinstance(record, Mapping) else None
    if not isinstance(effect, Mapping) or effect.get("type") != EMIT_EVENT:
        return None
    name = (effect.get("payload") or {}).get("name")
    return name if isinstance(name, str) and name.startswith(RECORD_PREFIX) else None


def automation_records(ledger_store: Any, automation_id: str, *names: str) -> List[Dict[str, Any]]:
    """`[{name, payload, key}]` of the automation's records (all, or only `names`), in append order."""
    wanted = set(names)
    out: List[Dict[str, Any]] = []
    for record in ledger_store.list(automation_id):
        name = record_name(record)
        if name is None or (wanted and name not in wanted):
            continue
        out.append({"name": name, "payload": record_payload(record), "key": record.get("idempotency_key")})
    return out


# --- run-state accessors ----------------------------------------------------------


def definition_of(run: RunState) -> Dict[str, Any]:
    meta = run.vars.get("_meta") if isinstance(run.vars, dict) else None
    definition = meta.get("automation") if isinstance(meta, dict) else None
    if not isinstance(definition, dict):
        raise LookupError(f"run {run.run_id} is not an automation (no vars._meta.automation)")
    return definition


def state_of(run: RunState) -> Dict[str, Any]:
    rt = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
    state = rt.get("automation") if isinstance(rt, dict) else None
    if not isinstance(state, dict):
        raise LookupError(f"run {run.run_id} is not an automation (no vars._runtime.automation)")
    return state


def _apply_delta(run: RunState, delta: Mapping[str, Any]) -> None:
    state = state_of(run)
    for key, value in (delta.get("state") or {}).items():
        state[key] = copy.deepcopy(value)
    if "definition" in delta:
        run.vars.setdefault("_meta", {})["automation"] = copy.deepcopy(delta["definition"])


def _step_record(run: RunState, *, name: str, key: str, payload: Dict[str, Any], node_id: str, at: str) -> StepRecord:
    return StepRecord(
        run_id=run.run_id,
        step_id=key,
        node_id=node_id,
        status=StepStatus.COMPLETED,
        effect={"type": EMIT_EVENT, "payload": {"name": name, "payload": payload}, "result_key": None},
        result={"name": name, "state_version": payload.get("state_version")},
        started_at=at,
        ended_at=at,
        actor_id=run.actor_id,
        session_id=run.session_id,
        idempotency_key=key,
    )


# --- the protocol -------------------------------------------------------------------


def reconcile(run: RunState, *, run_store: Any, ledger_store: Any) -> bool:
    """Finish or drop an outstanding decision intent; True when the run was saved."""
    state = state_of(run)
    intent = state.get("intent")
    if not intent:
        return False
    record = find_by_idempotency_key(ledger_store, run.run_id, intent["key"])
    if record is not None:
        payload = record_payload(record)
        if int(payload.get("state_version") or 0) > int(state.get("state_version") or 0):
            _apply_delta(run, payload.get("delta") or {})
            state["state_version"] = int(payload["state_version"])
    state["intent"] = None
    run_store.save(run)
    return True


def commit_decision(
    run: RunState,
    *,
    run_store: Any,
    ledger_store: Any,
    name: str,
    key: str,
    fields: Mapping[str, Any],
    delta: Mapping[str, Any],
    at: str,
    node_id: str = "automation",
    command_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Steps 2 and 4-6 of the protocol (the caller reconciled and decided).

    Returns the decision payload — the existing one when `key` was already decided.
    """
    existing = find_by_idempotency_key(ledger_store, run.run_id, key)
    if existing is not None:
        return record_payload(existing)
    state = state_of(run)
    version = int(state.get("state_version") or 0) + 1
    revision = int((delta.get("definition") or definition_of(run))["revision"])
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "automation_id": run.run_id,
        "revision": revision,
        "at": at,
        "state_version": version,
        "delta": copy.deepcopy(dict(delta)),
        **copy.deepcopy(dict(fields)),
    }
    if command_id is not None:
        payload["command_id"] = command_id
    state["intent"] = {"key": key, "state_version": version}
    run_store.save(run)
    ledger_store.append(_step_record(run, name=name, key=key, payload=payload, node_id=node_id, at=at))
    _apply_delta(run, delta)
    state["state_version"] = version
    state["intent"] = None
    run_store.save(run)
    return payload


def append_observation(
    run: RunState,
    *,
    ledger_store: Any,
    name: str,
    key: str,
    fields: Mapping[str, Any],
    at: str,
    node_id: str = "automation",
    command_id: Optional[str] = None,
) -> Dict[str, Any]:
    """An observation record (no state change), appended once per key."""
    existing = find_by_idempotency_key(ledger_store, run.run_id, key)
    if existing is not None:
        return record_payload(existing)
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "automation_id": run.run_id,
        "revision": int(definition_of(run)["revision"]),
        "at": at,
        "state_version": int(state_of(run).get("state_version") or 0),
        "delta": {},
        **copy.deepcopy(dict(fields)),
    }
    if command_id is not None:
        payload["command_id"] = command_id
    ledger_store.append(_step_record(run, name=name, key=key, payload=payload, node_id=node_id, at=at))
    return payload


__all__ = [
    "append_observation",
    "automation_records",
    "commit_decision",
    "definition_of",
    "find_by_idempotency_key",
    "reconcile",
    "record_key",
    "record_name",
    "record_payload",
    "state_of",
]
