"""Shared fixtures for the automation tests: deterministic targets, stores, a fake clock.

Targets are plain WorkflowSpecs (no provider, no tools):

- `echo`: answers `echo:<prompt> | history=<n>` where n = len(context.messages);
  copies `notify` from its input into its output.
- `flaky`: fails (structurally: `{success: False, error}`) while
  `_meta.occurrence.attempt < fail_until`, then echoes.
- `ask`: parks on an ASK_USER wait (a human wait).
- `ask_event`: parks on a WAIT_EVENT carrying a prompt (an interactive EVENT wait).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.automations import controller as controller_mod
from abstractruntime.automations import create_automation, drive_automation, register_controller_bundle
from abstractruntime.automations.bundle import controller_workflow_spec
from abstractruntime.core.models import RunStatus
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore

T0 = "2026-01-01T00:00:00+00:00"


def _echo_node(run, ctx):
    prompt = run.vars.get("prompt")
    messages = ((run.vars.get("context") or {}).get("messages")) or []
    out: Dict[str, Any] = {"response": f"echo:{prompt} | history={len(messages)}", "success": True}
    if "notify" in run.vars:
        out["notify"] = run.vars["notify"]
    return StepPlan(node_id="answer", complete_output=out)


def _flaky_node(run, ctx):
    attempt = int(((run.vars.get("_meta") or {}).get("occurrence") or {}).get("attempt") or 1)
    if attempt < int(run.vars.get("fail_until") or 0):
        return StepPlan(node_id="answer", complete_output={"success": False, "error": f"boom on attempt {attempt}"})
    return _echo_node(run, ctx)


def _ask_node(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "Proceed?", "choices": ["yes", "no"]}, result_key="answer"),
        next_node="done",
    )


def _ask_event_node(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(type=EffectType.WAIT_EVENT, payload={"wait_key": f"approve:{run.run_id}", "prompt": "Approve?"}, result_key="answer"),
        next_node="done",
    )


def _ask_named_event_node(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(type=EffectType.WAIT_EVENT, payload={"scope": "session", "name": "approval.requested", "prompt": "Go?"},
                      result_key="answer"),
        next_node="done",
    )


def _done_node(run, ctx):
    return StepPlan(node_id="done", complete_output={"response": "done", "success": True})


TARGETS = {
    "echo": WorkflowSpec(workflow_id="echo", entry_node="answer", nodes={"answer": _echo_node}),
    "flaky": WorkflowSpec(workflow_id="flaky", entry_node="answer", nodes={"answer": _flaky_node}),
    "ask": WorkflowSpec(workflow_id="ask", entry_node="ask", nodes={"ask": _ask_node, "done": _done_node}),
    "ask_event": WorkflowSpec(workflow_id="ask_event", entry_node="ask", nodes={"ask": _ask_event_node, "done": _done_node}),
    "ask_named_event": WorkflowSpec(workflow_id="ask_named_event", entry_node="ask",
                                    nodes={"ask": _ask_named_event_node, "done": _done_node}),
}


def make_stores(kind: str, root: Path):
    if kind == "json":
        return JsonFileRunStore(root / "runs"), JsonlLedgerStore(root / "ledger")
    db = SqliteDatabase(root / "gateway.sqlite")
    return SqliteRunStore(db), SqliteLedgerStore(db)


def make_runtime(run_store, ledger_store) -> Runtime:
    registry = WorkflowRegistry()
    for spec in TARGETS.values():
        registry.register(spec)
    register_controller_bundle(registry)
    return Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry)


class Clock:
    """The controller's clock (the runtime's own deadline checks use real time;
    tests move time by setting this and waking the controller)."""

    def __init__(self, monkeypatch, now: str = T0) -> None:
        self.now = now
        monkeypatch.setattr(controller_mod, "_now_iso", lambda: self.now)

    def set(self, now: str) -> None:
        self.now = now


def request(
    *,
    request_id: Optional[str] = None,
    workflow_id: str = "echo",
    trigger: Optional[Dict[str, Any]] = None,
    input_data: Optional[Dict[str, Any]] = None,
    mode: str = "independent",
    retry: Optional[Dict[str, Any]] = None,
    workspace_root: str = "/tmp/automation-ws",
) -> Dict[str, Any]:
    req: Dict[str, Any] = {
        # Unique by default: occurrence ids derive from the automation id, and the
        # runtime keeps process-wide per-run-id state (effect cancellation marks).
        "request_id": request_id or f"req-{uuid.uuid4()}",
        "title": "Memory watch",
        "target": {
            "workflow_id": workflow_id,
            "bundle_ref": "fixtures@1.0.0",
            "flow_id": workflow_id,
            "input_data": input_data if input_data is not None else {"prompt": "check memory"},
        },
        "trigger": trigger or {"source_id": "schedule", "source_version": 1, "config": {"start_at": T0, "every": "2m"}},
        "context": {"mode": mode},
        "workspace_root": workspace_root,
    }
    if retry is not None:
        req["policy"] = {"retry": retry}
    return req


def create(runtime: Runtime, clock: Clock, **kwargs) -> str:
    automation_id, _ = create_automation(runtime, request(**kwargs), now=clock.now)
    return automation_id


def wake(runtime: Runtime, automation_id: str) -> None:
    """What a due deadline does: resume the controller's wake wait (timed out)."""
    state = runtime.get_state(automation_id)
    if state.status == RunStatus.WAITING and state.waiting and state.waiting.wait_key.endswith(":wake"):
        runtime.resume(
            workflow=controller_workflow_spec(),
            run_id=automation_id,
            wait_key=state.waiting.wait_key,
            payload={"timed_out": True},
            max_steps=0,
        )


def drive(runtime: Runtime, automation_id: str):
    return drive_automation(runtime, automation_id)


def at(runtime: Runtime, clock: Clock, automation_id: str, now: str):
    """Move the clock to `now`, wake the controller, drive it until it parks."""
    clock.set(now)
    wake(runtime, automation_id)
    return drive(runtime, automation_id)


def automation_state(runtime: Runtime, automation_id: str) -> Dict[str, Any]:
    return runtime.get_state(automation_id).vars["_runtime"]["automation"]


def children(runtime: Runtime, automation_id: str):
    return sorted(runtime.run_store.list_children(parent_run_id=automation_id), key=lambda r: r.created_at)
