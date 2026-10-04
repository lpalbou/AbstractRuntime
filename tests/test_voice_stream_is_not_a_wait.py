"""R13.1 (2026-10-04 watchdog incident): streamed speech is never a run wait.

Before: `stream_voice` parked a child run of the CALLER'S run on
`WAIT_EVENT abstractcore.voice.tts.stream:<uuid>` for the whole stream. A
client folded that waiting record into the run ("Waiting for an event ›
Streaming voice synthesis is running." + "Event routing unavailable"), and
after a gateway restart mid-stream nothing could ever resume it.

Now: nothing durable exists while audio streams; the child run is created
already COMPLETED with the outcome when the stream ends; a legacy wait left by
an older process is closed by `close_interrupted_voice_stream`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime import Effect, EffectType, Runtime, RunStatus, StepPlan, WorkflowSpec
from abstractruntime.integrations.abstractcore import (
    LEGACY_VOICE_STREAM_WAIT_KEY_PREFIX,
    get_abstractcore_run_facade,
    is_interrupted_voice_stream_wait,
)
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class _NoopTools:
    def execute(self, *, tool_calls):
        return {"mode": "executed", "results": []}


def _runtime(client: Any = None) -> Runtime:
    store = InMemoryArtifactStore()
    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=store,
        effect_handlers=build_effect_handlers(llm=_NoopTools(), tools=_NoopTools(), artifact_store=store),
    )
    if client is not None:
        rt._abstractcore_llm_client = client
    return rt


def _live_parent(rt: Runtime) -> str:
    """A parent that is itself still running (the incident: Read aloud during a live turn)."""

    def ask(run, ctx):
        return StepPlan(node_id="ask", effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "next?"}, result_key="a"), next_node="done")

    def done(run, ctx):
        return StepPlan(node_id="done", complete_output={"ok": True})

    wf = WorkflowSpec("wf_live_parent", "ask", {"ask": ask, "done": done})
    rid = rt.start(workflow=wf, session_id="sess-r13")
    rt.tick(workflow=wf, run_id=rid)
    return rid


def _children(rt: Runtime, parent_id: str) -> List[Any]:
    return [r for r in rt.run_store.list_runs(limit=1000) if getattr(r, "parent_run_id", None) == parent_id]


def _waiting_ledger_records(rt: Runtime, run_ids: List[str]) -> List[Dict[str, Any]]:
    out = []
    for rid in run_ids:
        for rec in rt.ledger_store.list(rid):
            if (rec.get("status") if isinstance(rec, dict) else getattr(rec, "status", None)) in ("waiting", RunStatus.WAITING):
                out.append(rec)
    return out


class _GatedStream:
    """A streaming client whose stream the test inspects between events."""

    def __init__(self, observe):
        self.observe = observe

    def stream_tts(self, *, text, output=None, params=None):
        yield {"type": "start", "ok": True}
        self.observe("after-start")
        yield {"type": "audio", "sequence": 0, "content_type": "audio/wav", "audio_b64": "UklGRg=="}
        self.observe("after-audio")
        yield {"type": "done", "ok": True, "chunks": 1, "audio_artifact": {"artifact_id": "a1", "content_type": "audio/wav"}}


def test_no_child_run_and_no_waiting_record_exist_while_speech_streams() -> None:
    seen: Dict[str, Any] = {}
    holder: Dict[str, Any] = {}

    def observe(label: str) -> None:
        rt, parent = holder["rt"], holder["parent"]
        kids = _children(rt, parent)
        seen[label] = {
            "children": [(k.run_id, k.status) for k in kids],
            "waiting_runs": [r.run_id for r in rt.run_store.list_runs(status=RunStatus.WAITING, limit=1000) if r.run_id != parent],
        }

    rt = _runtime(_GatedStream(observe))
    parent = _live_parent(rt)
    holder.update(rt=rt, parent=parent)
    facade = get_abstractcore_run_facade(rt)
    events = list(facade.stream_voice(parent, text="Read this aloud.", output={"provider": "fake", "format": "wav"}))

    assert seen["after-start"] == {"children": [], "waiting_runs": []}
    assert seen["after-audio"] == {"children": [], "waiting_runs": []}
    start = events[0]
    assert start["type"] == "runtime_start" and "wait_key" not in start
    done = events[-1]
    assert done["type"] == "done" and done["child_run_status"] == RunStatus.COMPLETED.value
    assert done["child_run_id"] == start["child_run_id"]
    kids = _children(rt, parent)
    assert [(k.run_id, k.status) for k in kids] == [(start["child_run_id"], RunStatus.COMPLETED)]
    assert _waiting_ledger_records(rt, [k.run_id for k in kids]) == []
    # The caller's run is untouched: still waiting on ITS OWN question, nothing else.
    assert rt.get_state(parent).waiting.reason.value == "user"


def test_an_abandoned_stream_leaves_a_completed_cancelled_child_not_a_wait() -> None:
    rt = _runtime(_GatedStream(lambda label: None))
    parent = _live_parent(rt)
    events = get_abstractcore_run_facade(rt).stream_voice(parent, text="x", output={"format": "wav"})
    first = next(events)
    next(events)  # start
    events.close()  # the client left
    kids = _children(rt, parent)
    assert [k.run_id for k in kids] == [first["child_run_id"]]
    assert kids[0].status == RunStatus.COMPLETED
    assert kids[0].output["result"]["errors"][0]["code"] == "cancelled"


def test_a_stream_that_dies_mid_way_leaves_no_run_behind() -> None:
    """A gateway killed mid-stream (the incident) must not leave a run that waits forever."""

    rt = _runtime(_GatedStream(lambda label: None))
    parent = _live_parent(rt)
    events = get_abstractcore_run_facade(rt).stream_voice(parent, text="x", output={"format": "wav"})
    next(events)
    next(events)
    # The process "dies" here: the generator is never closed or resumed.
    assert _children(rt, parent) == []
    assert [r.run_id for r in rt.run_store.list_runs(status=RunStatus.WAITING, limit=1000) if r.run_id != parent] == []
    events.close()


def _seed_legacy_stream_wait(rt: Runtime, parent: str) -> str:
    """Exactly what a pre-R13.1 runtime left on disk (ledger a2abbf65 of the incident)."""

    key = f"{LEGACY_VOICE_STREAM_WAIT_KEY_PREFIX}c8f04c59-7819-48b4-87b7-b03e2f921ab2"

    def wait(run, ctx):
        return StepPlan(
            node_id="wait",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={"wait_key": key, "resume_to_node": "done", "prompt": "Streaming voice synthesis is running.",
                         "allow_free_text": False, "details": {"mode": "abstractcore_voice_stream", "text": "Found it!"}},
                result_key="_abstractcore_result",
            ),
            next_node="done",
        )

    def done(run, ctx):
        return StepPlan(node_id="done", complete_output={"result": run.vars.get("_abstractcore_result")})

    wf = WorkflowSpec("wf_abstractcore_run_facade_tts_stream", "wait", {"wait": wait, "done": done})
    parent_state = rt.get_state(parent)
    child = rt.start(workflow=wf, session_id=parent_state.session_id, parent_run_id=parent)
    rt.tick(workflow=wf, run_id=child)
    assert rt.get_state(child).status == RunStatus.WAITING
    return child


def test_a_legacy_stream_wait_left_by_a_dead_process_is_closed_with_a_sentence() -> None:
    rt = _runtime()
    parent = _live_parent(rt)
    child = _seed_legacy_stream_wait(rt, parent)
    assert is_interrupted_voice_stream_wait(rt.get_state(child).waiting)

    reason = "Read aloud was interrupted when the gateway restarted; the audio was not finished."
    state = get_abstractcore_run_facade(rt).close_interrupted_voice_stream(child, reason=reason)

    assert state.status == RunStatus.COMPLETED and state.waiting is None
    err = state.output["result"]["errors"][0]
    assert err == {"message": reason, "code": "interrupted"}
    assert rt.get_state(parent).waiting.reason.value == "user"  # the caller's run is untouched


def test_close_interrupted_voice_stream_refuses_any_other_wait() -> None:
    rt = _runtime()
    parent = _live_parent(rt)  # waiting on ASK_USER
    with pytest.raises(ValueError, match="not waiting on a streamed-speech event"):
        get_abstractcore_run_facade(rt).close_interrupted_voice_stream(parent, reason="x")


# --- the wait-kind registry --------------------------------------------------------------
#
# Every place the runtime itself parks a run on an EVENT wait. A client renders
# an event wait as "Waiting for an event" and can only route it when the key is
# canonical (`evt:<scope>:<id>:<name>`, `core/event_keys.build_event_wait_key`).
# Each site is listed with WHY its key is not shown as an event card; a new
# site fails this test until it is classified here.
_EVENT_WAIT_SITES = {
    # user-authored workflow event nodes: canonical keys (build_event_wait_key)
    "visualflow_compiler/adapters/event_adapter.py": "canonical",
    "visualflow_compiler/adapters/effect_adapter.py": "canonical",
    # tool approvals: details.mode == "approval_required" -> the client's approval card, never an event card
    "integrations/abstractcore/effect_handlers.py": "approval",
    # an automation controller parks on automation:<id>:wake (its own root run; clients show the automation, not the wait)
    "automations/controller.py": "automation_controller",
    # an entity visit parks on visitor_input (the visit run; answered by the visitor's next message)
    "identity/visit_workflow.py": "entity_visit",
}


def _wait_event_sites(root: Path) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "WAIT_EVENT" and isinstance(node.value, ast.Name) and node.value.id == "EffectType":
                rel = path.relative_to(root).as_posix()
                found[rel] = found.get(rel, 0) + 1
            if isinstance(node, ast.keyword) and node.arg == "reason" and isinstance(node.value, ast.Attribute) and node.value.attr == "EVENT":
                rel = path.relative_to(root).as_posix()
                found[rel] = found.get(rel, 0) + 1
            # `wait_reason = WaitReason.EVENT` (a handler choosing the reason of the wait it returns)
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "EVENT"
                and any(isinstance(t, ast.Name) and "reason" in t.id for t in node.targets)
            ):
                rel = path.relative_to(root).as_posix()
                found[rel] = found.get(rel, 0) + 1
    return found


def test_every_runtime_event_wait_site_is_classified_and_streamed_speech_is_not_one() -> None:
    import abstractruntime

    root = Path(abstractruntime.__file__).parent
    sites = _wait_event_sites(root)
    # The runtime's own machinery (handlers, models, stores, the scheduler) names the
    # enum without creating a wait; only these files construct one.
    machinery = {"core/models.py", "core/runtime.py", "scheduler/scheduler.py", "visualflow_compiler/visual/executor.py"}
    creating = {f for f in sites if f not in machinery}
    assert "integrations/abstractcore/run_facade.py" not in creating, "streamed speech must never park a run on an event wait"
    unclassified = sorted(creating - set(_EVENT_WAIT_SITES))
    assert unclassified == [], f"new event-wait sites need a client presentation class: {unclassified}"
