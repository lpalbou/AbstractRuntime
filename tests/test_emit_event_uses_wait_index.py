"""emit_event must USE the store's waiter index, not scan the whole store.

The delivery semantics are pinned by the store-level parity tests
(test_event_wait_index.py). What this file pins is the WIRING — the failure
mode where the index exists, the parity tests are green, and `emit_event` quietly
keeps its `list_runs(WAITING, EVENT, 10_000)` because a decorator or a
signature change made the fast path unreachable. That regression is invisible
except as latency (0.17-0.25s per emit on the operator's store, twice per chat
turn), so it gets an assertion.
"""
from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime import JsonlLedgerStore, Runtime
from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.offloading import OffloadingRunStore


def _listener_spec() -> WorkflowSpec:
    def wait_node(run: RunState, ctx: object) -> StepPlan:
        del run, ctx
        return StepPlan(
            node_id="wait",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={"name": "ready", "scope": "session"},
                result_key="_temp.evt",
            ),
            next_node="done",
        )

    def done_node(run: RunState, ctx: object) -> StepPlan:
        del run, ctx
        return StepPlan(node_id="done", complete_output={"success": True})

    return WorkflowSpec(workflow_id="wf_listener", entry_node="wait", nodes={"wait": wait_node, "done": done_node})


def _emitter_spec() -> WorkflowSpec:
    def emit_node(run: RunState, ctx: object) -> StepPlan:
        del run, ctx
        return StepPlan(
            node_id="emit",
            effect=Effect(
                type=EffectType.EMIT_EVENT,
                payload={"name": "ready", "scope": "session", "payload": {"v": 1}},
                result_key="_temp.emit",
            ),
            next_node="fin",
        )

    def fin_node(run: RunState, ctx: object) -> StepPlan:
        del run, ctx
        return StepPlan(node_id="fin", complete_output={"success": True})

    return WorkflowSpec(workflow_id="wf_emitter", entry_node="emit", nodes={"emit": emit_node, "fin": fin_node})


class _Registry:
    def __init__(self, specs: List[WorkflowSpec]) -> None:
        self._by_id: Dict[str, WorkflowSpec] = {s.workflow_id: s for s in specs}

    def get(self, workflow_id: str) -> Any:
        return self._by_id.get(str(workflow_id))


def _runtime(tmp_path, *, offloaded: bool):
    inner = JsonFileRunStore(tmp_path)
    if offloaded:
        run_store: Any = OffloadingRunStore(inner, artifact_store=FileArtifactStore(tmp_path / "artifacts"))
    else:
        run_store = inner
    listener, emitter = _listener_spec(), _emitter_spec()
    rt = Runtime(
        run_store=run_store,
        ledger_store=JsonlLedgerStore(tmp_path),
        workflow_registry=_Registry([listener, emitter]),
    )
    return rt, run_store, inner, listener, emitter


def _park_listener(rt, listener, session_id: str) -> str:
    rid = rt.start(workflow=listener, vars={}, session_id=session_id)
    run = rt.tick(workflow=listener, run_id=rid, max_steps=10)
    assert str(getattr(run.status, "value", run.status)) == "waiting"
    return rid


def _emit(rt, emitter, session_id: str):
    eid = rt.start(workflow=emitter, vars={}, session_id=session_id)
    return rt.tick(workflow=emitter, run_id=eid, max_steps=10)


def test_emit_event_resumes_the_listener_without_scanning_the_store(tmp_path) -> None:
    rt, run_store, inner, listener, emitter = _runtime(tmp_path, offloaded=False)
    listener_id = _park_listener(rt, listener, "s1")

    scans: List[dict] = []
    real_list_runs = inner.list_runs

    def counting_list_runs(**kw):
        scans.append(dict(kw))
        return real_list_runs(**kw)

    inner.list_runs = counting_list_runs  # type: ignore[assignment]

    _emit(rt, emitter, "s1")

    resumed = run_store.load(listener_id)
    assert str(getattr(resumed.status, "value", resumed.status)) == "running", "the listener must be resumed"
    assert scans == [], f"emit_event fell back to a whole-store scan: {scans}"


def test_the_offloading_wrapper_does_not_hide_the_index(tmp_path) -> None:
    """The gateway's production wiring is OffloadingRunStore(JsonFileRunStore);
    an implicit passthrough is exactly how the fast path gets lost (the P1-4
    `probe_control` lesson)."""
    rt, run_store, inner, listener, emitter = _runtime(tmp_path, offloaded=True)
    listener_id = _park_listener(rt, listener, "s2")

    scans: List[dict] = []
    real_list_runs = inner.list_runs

    def counting_list_runs(**kw):
        scans.append(dict(kw))
        return real_list_runs(**kw)

    inner.list_runs = counting_list_runs  # type: ignore[assignment]

    _emit(rt, emitter, "s2")

    resumed = run_store.load(listener_id)
    assert str(getattr(resumed.status, "value", resumed.status)) == "running"
    assert scans == [], f"emit_event fell back to a whole-store scan through the wrapper: {scans}"


def test_a_store_without_the_index_still_delivers_through_the_scan(tmp_path) -> None:
    """Capability, not requirement: an older store keeps working."""
    rt, run_store, inner, listener, emitter = _runtime(tmp_path, offloaded=False)
    listener_id = _park_listener(rt, listener, "s3")

    hidden = _NoIndexStore(inner)
    assert not hasattr(hidden, "list_event_waiters"), "fixture must actually hide the index"
    rt._run_store = hidden  # type: ignore[attr-defined]

    _emit(rt, emitter, "s3")

    resumed = run_store.load(listener_id)
    assert str(getattr(resumed.status, "value", resumed.status)) == "running"


class _NoIndexStore:
    """A QueryableRunStore that predates the event-wait index.

    Delegation is EXPLICIT: Python 3.12's runtime_checkable isinstance uses
    `inspect.getattr_static`, so a `__getattr__` proxy is not a
    QueryableRunStore and emit_event would refuse it for the wrong reason.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def save(self, run: Any) -> None:
        self._inner.save(run)

    def load(self, run_id: str) -> Any:
        return self._inner.load(run_id)

    def delete(self, run_id: str) -> bool:
        return self._inner.delete(run_id)

    def list_runs(self, **kw: Any) -> Any:
        return self._inner.list_runs(**kw)

    def list_due_wait_until(self, **kw: Any) -> Any:
        return self._inner.list_due_wait_until(**kw)

    def list_children(self, **kw: Any) -> Any:
        return self._inner.list_children(**kw)

    def list_run_index(self, **kw: Any) -> Any:
        return self._inner.list_run_index(**kw)


def test_missing_listener_diagnostic_survives_the_index(tmp_path) -> None:
    """`available_listeners_in_session` is how a user debugs a name mismatch;
    the scan produced it as a side effect, so the index has to produce it
    explicitly."""
    rt, _run_store, _inner, listener, _emitter = _runtime(tmp_path, offloaded=False)
    _park_listener(rt, listener, "s4")

    def emit_wrong_name(run: RunState, ctx: object) -> StepPlan:
        del run, ctx
        return StepPlan(
            node_id="emit",
            effect=Effect(
                type=EffectType.EMIT_EVENT,
                payload={"name": "reddy", "scope": "session"},
                result_key="_temp.emit",
            ),
            next_node="fin",
        )

    def fin_node(run: RunState, ctx: object) -> StepPlan:
        del run, ctx
        return StepPlan(node_id="fin", complete_output={"success": True})

    typo = WorkflowSpec(workflow_id="wf_typo", entry_node="emit", nodes={"emit": emit_wrong_name, "fin": fin_node})
    rt._workflow_registry._by_id[typo.workflow_id] = typo  # type: ignore[attr-defined]

    eid = rt.start(workflow=typo, vars={}, session_id="s4")
    rt.tick(workflow=typo, run_id=eid, max_steps=10)

    records = rt.ledger_store.list(eid)
    emit_rec = next(
        r
        for r in records
        if (r.get("effect") or {}).get("type") == "emit_event"
        and str(r.get("status")) == "completed"
        and (r.get("effect") or {}).get("payload", {}).get("name") == "reddy"
    )
    result = emit_rec.get("result") or {}
    assert result.get("delivered") == 0
    assert result.get("available_listeners_in_session") == ["ready"], result
