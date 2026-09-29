"""The automation controller's node logic (contract D).

The controller flow is

    start -> read_definition -> wait -> admit -> prepare_context -> dispatch
          -> record_outcome -> next -> read_definition ...
    read_definition / wait -> end   (exhausted, or archived with nothing running)

Each function below is one node. It runs inside `Runtime.tick`, which holds the
run's mutation lock for the whole tick (`run_mutation_lock`), so commands and
controller steps never interleave. Every state change goes through the decision
protocol in `automations.ledger`. Nodes only ever issue existing effects:

- `wait` parks on ONE `WAIT_EVENT` (`automation:<id>:wake`) whose optional
  deadline is the next tick or the retry time. Commands wake it; the payload
  of a wake is ignored — the controller re-reads persisted state, so a stale
  wake can never turn into an extra admission;
- `dispatch` starts the occurrence with `START_SUBWORKFLOW {async, wait,
  run_id}`: the child runs outside the controller's tick (the host drives it)
  and the controller waits for its completion like any subworkflow parent.
  The child id is deterministic and created through create-if-absent, so a
  replayed dispatch loads the same child.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..core.models import Effect, EffectType, RunState, RunStatus
from ..triggers.manual import manual_event_id
from ..triggers.protocol import format_timestamp, parse_timestamp
from ..triggers.registry import get_trigger_adapter
from .attention import normalize_occurrence_output, notify_payload, resolve_strict
from .ledger import (
    append_observation,
    commit_decision,
    definition_of,
    reconcile,
    record_key,
    state_of,
)
from .models import (
    add_delay,
    backoff_delay,
    definition_email_allowed_recipients,
    definition_notify,
    occurrence_run_id,
    trigger_state_of,
    wake_wait_key,
)

# Where the subworkflow wait delivers `{sub_run_id, output}` (transient; the
# outcome is always read from the reloaded child, never from this payload).
DISPATCH_RESULT_KEY = "_temp.automation_dispatch"

TERMINAL = (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


class ControllerSeamError(RuntimeError):
    """A runtime seam the controller requires is missing (fails the tick loudly)."""


def _now_iso() -> str:
    return format_timestamp(datetime.now(timezone.utc))


@dataclass
class Turn:
    """One controller node invocation: the run and the stores the tick exposes."""

    run: RunState
    run_store: Any
    ledger_store: Any
    artifact_store: Any
    now: str
    # Email (framework backlog 0992 WP2): the runtime's durable event inbox and
    # the account its occurrences are bound to (both host-set on the Runtime).
    event_inbox: Any = None
    email_binding: Any = None

    @classmethod
    def from_run(cls, run: RunState, *, now: Optional[str] = None) -> "Turn":
        run_store = getattr(run, "_runtime_run_store", None)
        ledger_store = getattr(run, "_runtime_ledger_store", None)
        if run_store is None or ledger_store is None:
            raise ControllerSeamError(
                "automation controller nodes need run._runtime_run_store and run._runtime_ledger_store "
                "(set by Runtime.tick); this runtime does not provide them"
            )
        return cls(
            run=run,
            run_store=run_store,
            ledger_store=ledger_store,
            artifact_store=getattr(run, "_runtime_artifact_store", None),
            now=now or _now_iso(),
            event_inbox=getattr(run, "_runtime_event_inbox", None),
            email_binding=getattr(run, "_runtime_email_binding", None),
        )

    @property
    def definition(self) -> Dict[str, Any]:
        return definition_of(self.run)

    @property
    def state(self) -> Dict[str, Any]:
        return state_of(self.run)

    @property
    def automation_id(self) -> str:
        return self.run.run_id

    def reconcile(self) -> None:
        reconcile(self.run, run_store=self.run_store, ledger_store=self.ledger_store)

    def decide(self, name: str, discriminator: Any, *, fields: Dict[str, Any], state: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        return commit_decision(
            self.run,
            run_store=self.run_store,
            ledger_store=self.ledger_store,
            name=name,
            key=record_key(name, self.automation_id, discriminator),
            fields=fields,
            delta={"state": state},
            at=self.now,
            node_id=node_id,
        )


def _due(now: str, at: Optional[str]) -> bool:
    return at is not None and parse_timestamp(at, field="time") <= parse_timestamp(now, field="now")


def adapter_kind(adapter: Any) -> str:
    caps = (getattr(adapter, "descriptor", None) or {}).get("capabilities") or {}
    return str(caps.get("kind") or "")


def adapter_delivers_untrusted(adapter: Any) -> bool:
    """A source whose events carry text written by other people (`email.received@1`)."""
    caps = (getattr(adapter, "descriptor", None) or {}).get("capabilities") or {}
    return caps.get("content_trust") == "untrusted"


def source_events(turn: Turn, adapter: Any, state: Dict[str, Any]) -> Optional[list]:
    """The inbox records after the automation's cursor, for an event source (None otherwise).

    Records of another account's stream are passed with an empty payload: never admitted,
    but the cursor may move past them. An event source on a runtime without an inbox is a
    host wiring error and fails the tick loudly (creation already refuses it).
    """
    if adapter_kind(adapter) != "event":
        return None
    inbox = turn.event_inbox
    if inbox is None:
        raise ControllerSeamError(
            "this automation's trigger reads the runtime's event inbox, but the runtime has none "
            "(the host must call Runtime.set_event_inbox before ticking email-triggered automations)"
        )
    src = state.get("source_state") if isinstance(state.get("source_state"), dict) else {}
    records = inbox.read(after_seq=int(src.get("cursor_seq") or 0))
    binding = turn.email_binding
    if binding is None:
        return records
    prefix = f"email:{binding.account_ref}:"
    return [
        r if str(r.get("stream") or "").startswith(prefix) else {**{k: r.get(k) for k in ("seq", "event_id", "stream", "appended_at")}, "payload": {}}
        for r in records
    ]


def _skip_ahead(adapter: Any, binding: Dict[str, Any], state: Dict[str, Any], events: list) -> None:
    """Move the source cursor past records the automation will never admit.

    Derived from the binding's config and the append-only inbox, so it is replay-safe and
    needs no decision record (like `active_revision` in `read_definition`)."""
    skip = getattr(adapter, "skip_ahead", None)
    if not callable(skip):
        return
    src = dict(state.get("source_state") or {})
    target = int(skip(binding, state=trigger_state_of(state), events=events))
    if target > int(src.get("cursor_seq") or 0):
        src["cursor_seq"] = target
        state["source_state"] = src


# --- read_definition ------------------------------------------------------------------


def read_definition(turn: Turn) -> str:
    """Activate the latest revision; `end` when nothing can ever run again."""
    turn.reconcile()
    definition, state = turn.definition, turn.state
    state["active_revision"] = int(definition["revision"])  # derived, replay-safe
    if state.get("pending_occurrence") is None and (definition.get("archived_at") or state.get("exhausted")):
        return "end"
    return "continue"


# --- wait -------------------------------------------------------------------------------


def wait_decision(turn: Turn) -> Dict[str, Any]:
    """`{"go": True}` (something to do now), `{"end": True}`, or `{"until": iso|None}` (park)."""
    turn.reconcile()
    definition, state = turn.definition, turn.state
    pending = state.get("pending_occurrence")
    if pending is not None:
        if pending.get("phase") == "backoff" and not pending.get("stop_requested") and not _due(turn.now, pending.get("retry_at")):
            return {"until": pending["retry_at"]}
        return {"go": True}
    if state.get("manual_pending"):
        return {"go": True}
    if definition.get("archived_at"):
        return {"end": True}
    if state.get("paused"):
        return {"until": None}
    binding = definition["trigger"]
    adapter = get_trigger_adapter(binding["source_id"], binding["source_version"])
    events = source_events(turn, adapter, state)
    if events is not None:
        _skip_ahead(adapter, binding, state, events)
        wait = adapter.prepare(binding, state=trigger_state_of(state), now=turn.now, events=events)
    else:
        wait = adapter.prepare(binding, state=trigger_state_of(state), now=turn.now)
    if wait["kind"] == "exhausted":
        return {"end": True}
    if wait["kind"] == "idle":
        return {"until": None}
    if wait["kind"] == "until":
        return {"go": True} if _due(turn.now, wait["until"]) else {"until": wait["until"]}
    raise ControllerSeamError(f"trigger wait kind {wait['kind']!r} is not supported by the v1 controller")


def wake_effect(turn: Turn, until: Optional[str]) -> Effect:
    payload: Dict[str, Any] = {"wait_key": wake_wait_key(turn.automation_id)}
    if until is not None:
        payload["until"] = until
    return Effect(type=EffectType.WAIT_EVENT, payload=payload, result_key=None)


# --- admit ------------------------------------------------------------------------------


def _render_prompt(input_data: Dict[str, Any], *, envelope: Dict[str, Any], index: int) -> Dict[str, Any]:
    out = copy.deepcopy(input_data)
    prompt = out.get("prompt")
    if isinstance(prompt, str):
        header = (
            f"[Trigger {envelope['source_id']}@{envelope['source_version']} · occurrence {index}"
            f" · fired {envelope['fired_at']}]"
        )
        out["prompt"] = f"{header}\n{prompt}"
    return out


AUTOMATION_GRANT_SOURCE = "automation-policy"


def _grant_excludes(name: str) -> bool:
    """True for a tool the automation grant never pre-approves: one that sends a message people
    receive to recipients the MODEL chooses (the served row's `comms_send` fact: `send_email`,
    `send_whatsapp_message`, `send_telegram_*`). Framework backlog 0992 WP0: an unattended
    occurrence reading untrusted input (an inbound email, a fetched page) must not be able to mail
    data to an address that input names. Decided on the row's typed facts, never on the name; an
    unreadable inventory withholds every `comms` tool (fail closed)."""
    try:
        from ..integrations.abstractcore.effect_handlers import _risk_row_for_tool

        row = _risk_row_for_tool(name)
    except Exception:  # noqa: BLE001 - no facts: withhold the whole comms class
        from ..integrations.abstractcore.tool_effects import COMMS, TOOL_EFFECT_CLASSES

        return TOOL_EFFECT_CLASSES.get(name) == COMMS
    return bool(row is not None and row.get("comms_send"))


def _withholds_destination(name: str) -> bool:
    """True for a tool whose served row carries `model_controlled_destination` (`fetch_url`,
    `browser_probe`, and every message-sending tool): the model chooses where data goes.
    An unreadable inventory withholds the tool (fail closed)."""
    try:
        from ..integrations.abstractcore.effect_handlers import _risk_row_for_tool

        row = _risk_row_for_tool(name)
    except Exception:  # noqa: BLE001
        return True
    return bool(row is None or row.get("model_controlled_destination"))


def grant_tool_approval(input_data: Dict[str, Any], *, untrusted_input: bool = False) -> None:
    """`policy.tool_approval == "auto"`: pre-approve the target's tools for this occurrence.

    Creating the automation is the consent (an unattended run cannot ask a
    person every tick). The grant is the runtime's existing per-run policy
    `_runtime.tool_policy.auto_approve_tools`, which child runs inherit. The
    tools named are the target's explicit `_runtime.allowed_tools` when it has
    one, else every tool the runtime can expose (`TOOL_EFFECT_CLASSES`); a name
    outside the run's tool ceiling grants nothing. A tool_policy the target
    already carries is the target author's word and is left untouched.
    `ask_user` questions are not tool approvals and still wait for a person.

    Never granted (listed in `withheld_tools`): tools that send messages to
    model-chosen recipients (`_grant_excludes`). Their calls go through the
    normal approval point: the per-call refiner (`send_email_recipient@v2`)
    auto-approves a `send_email` whose every recipient is the registered user's
    own address (`_runtime.operator_email`, set by the host) or one the user
    pre-authorised in the definition (`_runtime.email_allowed_recipients`); any
    other recipient parks the occurrence on a `tool_approval` wait, exactly as
    under `"ask"`.

    `untrusted_input=True` (the trigger delivers text written by other people:
    `email.received@1`): tools whose row carries `model_controlled_destination`
    (`fetch_url`, `browser_probe`) are withheld too, so an inbound email cannot
    steer an unattended occurrence into sending data to a URL it names
    (framework backlog 0992, the WP0 follow-up). Schedule and manual
    automations keep them in the grant (decision recorded in 0992).
    """
    from ..integrations.abstractcore.tool_effects import TOOL_EFFECT_CLASSES

    runtime_ns = input_data.get("_runtime") if isinstance(input_data.get("_runtime"), dict) else {}
    if "tool_policy" in runtime_ns:
        return
    allowed = runtime_ns.get("allowed_tools")
    names = allowed if isinstance(allowed, list) else list(TOOL_EFFECT_CLASSES)
    candidates = sorted({n.strip() for n in names if isinstance(n, str) and n.strip()})
    withheld = [n for n in candidates if _grant_excludes(n) or (untrusted_input and _withholds_destination(n))]
    tools = [n for n in candidates if n not in withheld]
    input_data["_runtime"] = {
        **runtime_ns,
        "tool_policy": {"auto_approve_tools": tools, "withheld_tools": withheld, "source": AUTOMATION_GRANT_SOURCE},
    }


# The flow input that makes an agent read `context.messages` (basic-agent's
# `use_context` pin defaults to False; `include_context` is its alias, and
# wins when present, so both are set).
_CONTEXT_INPUT_KEYS = ("use_context", "include_context")


def apply_context_mode(input_data: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    """Make the automation's context mode the ONE history control (operator 2026-09-28).

    History is injected by the automation (growing occurrences, discussion
    seeds), so whether the target reads it is the automation's decision, not
    a target input frozen at creation: an old definition carrying
    `use_context: false` silently replayed NO history into a growing
    automation or a discussion. Growing and discussion → on; independent →
    off (it has no history to read). Returns the receipt recorded in the run
    as `_runtime.automation_context`: `{mode, use_context,
    target_use_context}` (the value the target input carried, or None).
    """
    if mode not in ("growing", "independent", "discussion"):
        raise ValueError(f"unknown automation context mode {mode!r}")
    use_context = mode != "independent"
    target_value = next((input_data[k] for k in _CONTEXT_INPUT_KEYS if k in input_data), None)
    input_data["use_context"] = use_context
    if "include_context" in input_data:
        input_data["include_context"] = use_context
    receipt = {"mode": mode, "use_context": use_context, "target_use_context": target_value}
    runtime_ns = input_data.get("_runtime") if isinstance(input_data.get("_runtime"), dict) else {}
    input_data["_runtime"] = {**runtime_ns, "automation_context": receipt}
    return receipt


def build_prepared(
    turn: Turn,
    *,
    index: int,
    envelope: Dict[str, Any],
    first_run_id: str,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Freeze the occurrence's inputs (contract D `prepare_context`).

    Independent: a fresh session (the attempt-1 run id), no history injected.
    Growing: the automation's session, with its prior turns as
    `context.messages` through the strict history path (a missing seed or an
    unreadable history fails the admission instead of running without context),
    windowed to the most recent `HISTORY_REPLAY_MAX_TOKENS` tokens of whole
    turns; the window's report rides `_runtime.session_history`.
    Both: the context mode decides whether the target reads history
    (`apply_context_mode`), whatever the target's frozen `use_context` says.

    Email (framework backlog 0992 WP2): an `email.received@1` admission's
    messages become `input_data.trigger` (marked untrusted) and, for a target
    with a string `prompt`, a fixed untrusted frame appended to it
    (`abstractruntime.email.frame`). Every occurrence gets the definition's
    pre-authorised recipients as `_runtime.email_allowed_recipients` and, when
    the runtime has one, its account binding as `_runtime.email_account` (both
    SET over whatever the target inputs carried).
    """
    from ..email.binding import EMAIL_ACCOUNT_KEY, EMAIL_ALLOWED_RECIPIENTS_KEY

    definition = turn.definition
    trigger_binding = definition["trigger"]
    try:
        untrusted = adapter_delivers_untrusted(get_trigger_adapter(trigger_binding["source_id"], trigger_binding["source_version"]))
    except Exception:  # noqa: BLE001 - an unknown source: treat its input as untrusted
        untrusted = True
    input_data = _render_prompt(definition["target"].get("input_data") or {}, envelope=envelope, index=index)
    emails = (inputs or {}).get("emails")
    if isinstance(emails, list):
        from ..email.frame import email_frame, email_trigger_input

        input_data["trigger"] = email_trigger_input(emails)
        if isinstance(input_data.get("prompt"), str):
            input_data["prompt"] = f"{input_data['prompt']}\n\n{email_frame(emails)}"
    if definition["policy"].get("tool_approval", "auto") == "auto":
        grant_tool_approval(input_data, untrusted_input=untrusted)
    apply_context_mode(input_data, mode=definition["context"]["mode"])
    if definition["context"]["mode"] == "growing":
        from ..session_history import session_chat_messages

        messages = session_chat_messages(
            run_store=turn.run_store,
            ledger_store=turn.ledger_store,
            artifact_store=turn.artifact_store,
            session_id=definition["session_id"],
            automation_id=turn.automation_id,
            strict=True,
        )
        context = input_data.get("context") if isinstance(input_data.get("context"), dict) else {}
        input_data["context"] = {**context, "messages": list(messages)}
        # The history window's receipt (ADR-0026: explicit and observable),
        # frozen with the inputs, so the occurrence run and its
        # `automation.admitted` record both say what was replayed and dropped.
        runtime_ns = input_data.get("_runtime") if isinstance(input_data.get("_runtime"), dict) else {}
        input_data["_runtime"] = {**runtime_ns, "session_history": {**messages.report, "strict": True, "session_kind": "automation"}}
        session_id = definition["session_id"]
    else:
        session_id = first_run_id
    runtime_ns = dict(input_data.get("_runtime")) if isinstance(input_data.get("_runtime"), dict) else {}
    runtime_ns[EMAIL_ALLOWED_RECIPIENTS_KEY] = definition_email_allowed_recipients(definition)
    if turn.email_binding is not None:
        runtime_ns[EMAIL_ACCOUNT_KEY] = turn.email_binding.to_dict()
    input_data["_runtime"] = runtime_ns
    # The automation's title for templates (the send-email action's {automation_title}).
    meta_ns = dict(input_data.get("_meta")) if isinstance(input_data.get("_meta"), dict) else {}
    meta_ns["automation_title"] = definition["title"]
    input_data["_meta"] = meta_ns
    return {
        "workflow_id": definition["target"]["workflow_id"],
        "session_id": session_id,
        "workspace_root": definition["workspace_root"],
        "input_data": input_data,
    }


def admit(turn: Turn) -> str:
    """Admit a due occurrence (`prepare`), resume in-flight work (`dispatch`), or re-arm (`rearm`).

    Admission happens only for (a) a due scheduled tick computed from persisted
    state while not paused, or (b) a persisted `manual_pending`.
    """
    turn.reconcile()
    definition, state = turn.definition, turn.state
    pending = state.get("pending_occurrence")
    if pending is not None:
        if pending.get("stop_requested") and pending.get("phase") != "dispatched":
            complete_occurrence(turn, status="cancelled", outcome=None, node_id="admit")
            return "rearm"
        if pending.get("phase") == "backoff" and not _due(turn.now, pending.get("retry_at")):
            return "rearm"
        return "dispatch"

    binding = definition["trigger"]
    index = int(state.get("next_index") or 1)
    revision = int(definition["revision"])
    manual = state.get("manual_pending")
    coalesced = None
    trigger_update: Dict[str, Any] = {}
    inputs: Optional[Dict[str, Any]] = None
    if manual:
        command_id = str(manual["command_id"])
        manual_adapter = get_trigger_adapter("manual", 1)
        envelope = manual_adapter.normalize(
            binding, event_id=manual_event_id(command_id), fired_at=turn.now, payload={"command_id": command_id}
        )
        first_run_id = occurrence_run_id(turn.automation_id, revision=revision, index=index, command_id=command_id)
    else:
        if state.get("paused") or definition.get("archived_at"):
            return "rearm"
        adapter = get_trigger_adapter(binding["source_id"], binding["source_version"])
        events = source_events(turn, adapter, state)
        if events is not None:
            admission = adapter.admit(binding, state=trigger_state_of(state), now=turn.now, events=events)
        else:
            admission = adapter.admit(binding, state=trigger_state_of(state), now=turn.now)
        if admission is None:
            return "rearm"
        inputs = admission.get("inputs")
        command_id = None
        envelope = adapter.normalize(
            binding, event_id=admission["event_id"], fired_at=admission["fired_at"], payload=admission["payload"]
        )
        trigger_update = dict(admission["state"])
        coalesced = admission.get("coalesced")
        first_run_id = occurrence_run_id(turn.automation_id, revision=revision, index=index)

    prepared = build_prepared(turn, index=index, envelope=envelope, first_run_id=first_run_id, inputs=inputs)
    new_pending = {
        "run_id": first_run_id,
        "index": index,
        "attempt": 1,
        "event_id": envelope["event_id"],
        "revision": revision,
        "envelope": envelope,
        "phase": "admitted",
        "command_id": command_id,
        "prepared": prepared,
        "retry": copy.deepcopy(definition["policy"]["retry"]),
        "session_kind": "automation" if definition["context"]["mode"] == "growing" else "occurrence",
        "stop_requested": False,
    }
    if coalesced:
        append_observation(
            turn.run,
            ledger_store=turn.ledger_store,
            name="automation.coalesced",
            key=record_key("automation.coalesced", turn.automation_id, envelope["event_id"]),
            fields={**coalesced, "event_id": envelope["event_id"]},
            at=turn.now,
            node_id="admit",
        )
    turn.decide(
        "automation.admitted",
        index,
        fields={
            "run_id": first_run_id,
            "index": index,
            "attempt": 1,
            "event_id": envelope["event_id"],
            "trigger_envelope": envelope,
            "prepared": prepared,
        },
        state={
            **trigger_update,
            "pending_occurrence": new_pending,
            "next_index": index + 1,
            "manual_pending": None,
        },
        node_id="admit",
    )
    return "prepare"


# --- prepare_context ----------------------------------------------------------------------


def prepare_context(turn: Turn) -> None:
    """Check the frozen inputs of the pending occurrence resolve (strictly)."""
    turn.reconcile()
    pending = turn.state.get("pending_occurrence")
    if pending is None:
        raise ControllerSeamError("prepare_context reached without a pending occurrence")
    resolve_strict(pending["prepared"], artifact_store=turn.artifact_store)


# --- dispatch --------------------------------------------------------------------------------


def _occurrence_vars(pending: Dict[str, Any], prepared: Dict[str, Any], *, automation_id: str) -> Dict[str, Any]:
    """The child's vars, built ONLY from what was frozen at admission.

    A replayed dispatch must send byte-identical creation vars (create-if-absent
    compares a digest of them), so nothing here may come from the current
    definition, the clock or a re-rendering.
    """
    child_vars = copy.deepcopy(prepared["input_data"])
    meta = child_vars.get("_meta") if isinstance(child_vars.get("_meta"), dict) else {}
    meta["occurrence"] = {
        "automation_id": automation_id,
        "occurrence_index": int(pending["index"]),
        "attempt": int(pending["attempt"]),
        "event_id": pending["event_id"],
        "revision": int(pending["revision"]),
        "role": "occurrence",
        "session_kind": pending["session_kind"],
        "fired_at": pending["envelope"]["fired_at"],
        "trigger_envelope": pending["envelope"],
    }
    child_vars["_meta"] = meta
    child_vars["workspace_root"] = prepared["workspace_root"]
    return child_vars


def dispatch(turn: Turn) -> Effect:
    """Record the dispatch of the current attempt and start (or re-attach to) its child."""
    turn.reconcile()
    state = turn.state
    pending = state.get("pending_occurrence")
    if pending is None:
        raise ControllerSeamError("dispatch reached without a pending occurrence")
    if pending["phase"] != "dispatched":
        child_id = occurrence_run_id(
            turn.automation_id,
            revision=pending["revision"],
            index=pending["index"],
            attempt=pending["attempt"],
            command_id=pending.get("command_id"),
        )
        turn.decide(
            "automation.dispatched",
            child_id,
            fields={"run_id": child_id, "index": pending["index"], "attempt": pending["attempt"]},
            state={"pending_occurrence": {**pending, "phase": "dispatched", "run_id": child_id, "retry_at": None}},
            node_id="dispatch",
        )
        pending = turn.state["pending_occurrence"]
    prepared = resolve_strict(pending["prepared"], artifact_store=turn.artifact_store)
    return Effect(
        type=EffectType.START_SUBWORKFLOW,
        payload={
            "workflow_id": prepared["workflow_id"],
            "vars": _occurrence_vars(pending, prepared, automation_id=turn.automation_id),
            "session_id": prepared["session_id"],
            "run_id": pending["run_id"],
            "async": True,
            "wait": True,
        },
        result_key=DISPATCH_RESULT_KEY,
    )


# --- record_outcome ------------------------------------------------------------------------------


def complete_occurrence(turn: Turn, *, status: str, outcome: Optional[Dict[str, Any]], node_id: str) -> Dict[str, Any]:
    """The single `automation.completed` decision (and attention allocation) of an occurrence."""
    definition, state = turn.definition, turn.state
    pending = state["pending_occurrence"]
    attention = None
    notify = None
    seq = int(state.get("attention_seq") or 0)
    if status == "completed" and outcome is not None:
        notify = notify_payload(outcome.get("notify"), title=definition["title"], answer=outcome.get("answer") or "")
        if notify is not None:
            seq += 1
            attention = {"kind": "notify", "seq": seq, **notify}
    elif status == "failed":
        seq += 1
        body = str((outcome or {}).get("error") or "The occurrence failed.")[:2000]
        attention = {"kind": "failure", "seq": seq, "title": f"{definition['title']} failed", "body": body}
    if attention is not None:
        # Where the item is delivered (schema v2 `notify.channels`): "email" asks the host's
        # notification dispatcher to mail the owner (framework backlog 0992 B5/C5).
        attention["channels"] = list(definition_notify(definition)["channels"])
    last_outcome = {
        "run_id": pending["run_id"],
        "index": int(pending["index"]),
        "status": status,
        "attempts": int(pending["attempt"]),
        "finished_at": turn.now,
    }
    return turn.decide(
        "automation.completed",
        int(pending["index"]),
        fields={
            "run_id": pending["run_id"],
            "index": int(pending["index"]),
            "status": status,
            "attempts": int(pending["attempt"]),
            "finished_at": turn.now,
            "notify": notify,
            "attention": attention,
        },
        state={"pending_occurrence": None, "last_outcome": last_outcome, "attention_seq": seq},
        node_id=node_id,
    )


def record_outcome(turn: Turn) -> str:
    """Read the child's terminal state; retry (`next` after scheduling) or complete (`next`).

    Returns `dispatch` when the child is not terminal yet (the controller was
    resumed early): dispatch re-attaches to the same child and waits again.
    """
    turn.reconcile()
    state = turn.state
    pending = state.get("pending_occurrence")
    if pending is None or pending.get("phase") != "dispatched":
        return "next"  # already recorded (a replayed step)
    delivered = turn.run.vars.get("_temp", {}).get("automation_dispatch") if isinstance(turn.run.vars.get("_temp"), dict) else None
    if isinstance(delivered, dict) and delivered.get("sub_run_id") not in (None, pending["run_id"]):
        raise ControllerSeamError(
            f"START_SUBWORKFLOW started {delivered.get('sub_run_id')} instead of the deterministic "
            f"occurrence id {pending['run_id']}: this runtime ignores START_SUBWORKFLOW.payload.run_id"
        )
    child = turn.run_store.load(pending["run_id"])
    if child is None:
        raise ControllerSeamError(f"occurrence run {pending['run_id']} does not exist after dispatch")
    if child.status not in TERMINAL:
        return "dispatch"
    outcome = normalize_occurrence_output(child, artifact_store=turn.artifact_store)
    if outcome["success"]:
        complete_occurrence(turn, status="completed", outcome=outcome, node_id="record_outcome")
        return "next"
    if child.status == RunStatus.CANCELLED or pending.get("stop_requested"):
        complete_occurrence(turn, status="cancelled", outcome=outcome, node_id="record_outcome")
        return "next"
    retry = pending.get("retry") or {"max_attempts": 1}
    attempt = int(pending["attempt"])
    if attempt < int(retry["max_attempts"]):
        retry_at = add_delay(turn.now, backoff_delay({"retry": retry}, attempt))
        turn.decide(
            "automation.retry_scheduled",
            pending["run_id"],
            fields={
                "run_id": pending["run_id"],
                "index": int(pending["index"]),
                "attempt": attempt,
                "error": outcome.get("error"),
                "retry_at": retry_at,
            },
            state={"pending_occurrence": {**pending, "attempt": attempt + 1, "phase": "backoff", "retry_at": retry_at}},
            node_id="record_outcome",
        )
        return "next"
    complete_occurrence(turn, status="failed", outcome=outcome, node_id="record_outcome")
    return "next"


def next_step(turn: Turn) -> None:
    """Drop the transient dispatch result before the next cycle."""
    temp = turn.run.vars.get("_temp")
    if isinstance(temp, dict):
        temp.pop("automation_dispatch", None)


# --- projection for readers ------------------------------------------------------------

_PHASE_STATUS = {"admitted": "admitted", "dispatched": "running", "backoff": "backoff"}


def current_occurrence(run: RunState) -> Optional[Dict[str, Any]]:
    """`{index, run_id, attempt, status: admitted|running|backoff}` of the occurrence in flight, or None."""
    pending = (((run.vars or {}).get("_runtime") or {}).get("automation") or {}).get("pending_occurrence")
    if not isinstance(pending, dict):
        return None
    return {
        "index": int(pending["index"]),
        "run_id": pending.get("run_id"),
        "attempt": int(pending.get("attempt") or 1),
        "status": _PHASE_STATUS.get(str(pending.get("phase")), str(pending.get("phase"))),
    }


def next_fire_at(run: RunState, *, now: Optional[str] = None) -> Optional[str]:
    """When the next occurrence will be admitted, as the controller will decide it.

    - Parked on its wake wait: the wait's deadline (the next tick, or the retry
      time during backoff).
    - Otherwise (an occurrence running, or the controller between steps): the
      trigger adapter's own answer on the persisted cursor — the next grid
      tick if it is still ahead, else the tick a coalesced admission will
      fire as soon as the running occurrence ends (`admit` at `now`). No
      schedule arithmetic lives outside the adapter.
    - None for a manual trigger, an exhausted schedule, or an automation that
      is not active (paused, archived, completed, failed).
    """
    from .models import automation_status

    if automation_status(run) != "active":
        return None
    waiting = run.waiting
    if run.status == RunStatus.WAITING and waiting is not None and waiting.wait_key == wake_wait_key(run.run_id):
        return waiting.until or None
    definition, state = definition_of(run), state_of(run)
    pending = state.get("pending_occurrence")
    if isinstance(pending, dict) and pending.get("phase") == "backoff":
        return pending.get("retry_at")
    binding = definition["trigger"]
    adapter = get_trigger_adapter(binding["source_id"], binding["source_version"])
    if adapter_kind(adapter) == "event":
        return None  # runs when mail arrives (after its batch interval), not at a clock time
    at = now or _now_iso()
    trigger_state = trigger_state_of(state)
    wait = adapter.prepare(binding, state=trigger_state, now=at)
    if wait["kind"] != "until":
        return None
    if not _due(at, wait["until"]):
        return wait["until"]
    admission = adapter.admit(binding, state=trigger_state, now=at)
    return admission["fired_at"] if admission is not None else wait["until"]


__all__ = [
    "ControllerSeamError",
    "adapter_delivers_untrusted",
    "adapter_kind",
    "source_events",
    "apply_context_mode",
    "current_occurrence",
    "next_fire_at",
    "DISPATCH_RESULT_KEY",
    "Turn",
    "admit",
    "build_prepared",
    "complete_occurrence",
    "dispatch",
    "next_step",
    "prepare_context",
    "read_definition",
    "record_outcome",
    "wait_decision",
    "wake_effect",
]
