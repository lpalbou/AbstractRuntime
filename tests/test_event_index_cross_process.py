"""The event-wait and mailbox indexes see what ANOTHER process saved (split mode, 2026-10-26).

`serve --no-runner` + `abstractgateway runner` share one run folder. The runner process
parks a run in WAITING(EVENT) by re-saving a run it already saved as active: no creation-log
line. Before the cross-process refresh, the API process's event-wait index (built earlier,
from one scan) never heard of the park, so an event the API process delivered missed the
waiter; the mailbox index had the same hole for a run that declared `events_mailbox` later.

The two-process tests run process B as a real subprocess on the same folder; process A is
this test. Each step is driven over B's stdin/stdout, so the order is exact: A builds its
indexes FIRST, then B parks, then A delivers.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

from abstractruntime import JsonlLedgerStore, Runtime
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus, StepPlan, WaitReason, WaitState
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.json_files import JsonFileRunStore

_SPECS = textwrap.dedent(
    """
    from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan
    from abstractruntime.core.spec import WorkflowSpec

    def listener_spec():
        def wait_node(run, ctx):
            return StepPlan(node_id="wait", effect=Effect(type=EffectType.WAIT_EVENT,
                            payload={"name": "ready", "scope": "session"}, result_key="_temp.evt"),
                            next_node="done")

        def done_node(run, ctx):
            return StepPlan(node_id="done", complete_output={"success": True})

        return WorkflowSpec(workflow_id="wf_listener", entry_node="wait", nodes={"wait": wait_node, "done": done_node})
    """
)

# Process B: starts the listener (RUNNING, `+id` line), waits for "park", ticks it into
# WAITING(EVENT) (a re-save of a run it saved as active: no log line), reports, exits.
_PROCESS_B = _SPECS + textwrap.dedent(
    """
    import sys
    from abstractruntime import JsonlLedgerStore, Runtime
    from abstractruntime.storage.json_files import JsonFileRunStore

    folder = sys.argv[1]
    spec = listener_spec()
    rt = Runtime(run_store=JsonFileRunStore(folder), ledger_store=JsonlLedgerStore(folder))
    rid = rt.start(workflow=spec, vars={}, session_id="s1")
    print("started", rid, flush=True)
    assert sys.stdin.readline().strip() == "park"
    run = rt.tick(workflow=spec, run_id=rid, max_steps=10)
    print("parked", str(getattr(run.status, "value", run.status)), flush=True)
    """
)

# Process B for the mailbox: an active run with no mailbox, then the SAME run re-saved
# declaring one (still active: no log line).
_PROCESS_B_MAILBOX = textwrap.dedent(
    """
    import sys
    from abstractruntime import RunState, RunStatus
    from abstractruntime.storage.json_files import JsonFileRunStore

    store = JsonFileRunStore(sys.argv[1])
    run = RunState.new(workflow_id="wf", entry_node="start")
    run.status = RunStatus.RUNNING
    store.save(run)
    print("started", run.run_id, flush=True)
    assert sys.stdin.readline().strip() == "declare"
    run.vars["events_mailbox"] = "agora:inbox"
    store.save(run)
    print("declared", flush=True)
    """
)


def _spawn(script: str, folder: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", script, str(folder)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _line(proc: subprocess.Popen) -> List[str]:
    out = proc.stdout.readline()
    if not out:
        raise AssertionError(f"process B ended early: {proc.stderr.read()}")
    return out.split()


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


def _listener_spec() -> WorkflowSpec:
    ns: Dict[str, Any] = {}
    exec(_SPECS, ns)  # noqa: S102 - the same spec process B runs
    return ns["listener_spec"]()


class _Registry:
    def __init__(self, specs: List[WorkflowSpec]) -> None:
        self._by_id = {s.workflow_id: s for s in specs}

    def get(self, workflow_id: str) -> Any:
        return self._by_id.get(str(workflow_id))


def _status(run: Any) -> str:
    return str(getattr(run.status, "value", run.status))


def test_a_run_parked_in_process_b_is_found_by_an_event_delivered_in_process_a(tmp_path: Path) -> None:
    store_a = JsonFileRunStore(tmp_path)
    emitter = _emitter_spec()
    rt_a = Runtime(
        run_store=store_a,
        ledger_store=JsonlLedgerStore(tmp_path),
        workflow_registry=_Registry([_listener_spec(), emitter]),
    )
    b = _spawn(_PROCESS_B, tmp_path)
    try:
        tag, listener_id = _line(b)
        assert tag == "started"
        # Process A delivers first: nobody waits yet, and A's index is now BUILT (the
        # pre-fix code never looked at disk again).
        first = rt_a.tick(workflow=emitter, run_id=rt_a.start(workflow=emitter, vars={}, session_id="s1"), max_steps=10)
        assert _status(first) == "completed"
        assert store_a.list_event_waiters(wait_keys=["evt:session:s1:ready"]) == []

        b.stdin.write("park\n")
        b.stdin.flush()
        assert _line(b) == ["parked", "waiting"]
        assert b.wait(timeout=30) == 0

        parked = store_a.load(listener_id)
        assert _status(parked) == "waiting" and parked.waiting.reason == WaitReason.EVENT

        t0 = time.monotonic()
        second = rt_a.tick(workflow=emitter, run_id=rt_a.start(workflow=emitter, vars={}, session_id="s1"), max_steps=10)
        assert _status(second) == "completed"
        resumed = store_a.load(listener_id)
        assert _status(resumed) == "running", "the event delivered in process A must resume the run parked by process B"
        assert resumed.waiting is None
        assert time.monotonic() - t0 < 2.0
        assert store_a.event_index_stats()["refiled"] >= 1
    finally:
        if b.poll() is None:
            b.kill()
        b.wait(timeout=10)


def test_a_mailbox_declared_in_process_b_is_found_by_process_a(tmp_path: Path) -> None:
    store_a = JsonFileRunStore(tmp_path)
    b = _spawn(_PROCESS_B_MAILBOX, tmp_path)
    try:
        tag, rid = _line(b)
        assert tag == "started"
        assert store_a.list_mailbox_runs(names=["agora:inbox"]) == []  # built now, empty
        b.stdin.write("declare\n")
        b.stdin.flush()
        assert _line(b) == ["declared"]
        assert b.wait(timeout=30) == 0
        assert [r.run_id for r in store_a.list_mailbox_runs(names=["agora:inbox"])] == [rid]
    finally:
        if b.poll() is None:
            b.kill()
        b.wait(timeout=10)


def _waiter(run: RunState, key: str) -> RunState:
    run.status = RunStatus.WAITING
    run.waiting = WaitState(reason=WaitReason.EVENT, wait_key=key, resume_to_node="next", result_key="evt")
    return run


def test_an_unmoved_folder_still_refreshes_within_the_floor(tmp_path: Path, monkeypatch) -> None:
    """A coarse-mtime filesystem (or an in-place edit) can leave the folder's mtime and the log
    unchanged: the lookup still re-reads the active runs at least every EVENT_INDEX_REFRESH_S."""
    a = JsonFileRunStore(tmp_path)
    other = JsonFileRunStore(tmp_path)  # another writer: its saves never reach a's indexes directly
    run = RunState.new(workflow_id="wf", entry_node="start")
    run.status = RunStatus.RUNNING
    other.save(run)
    assert a.list_event_waiters(wait_keys=["k"]) == []

    monkeypatch.setattr(a, "_xproc_change_mark", lambda: (1, 1))
    a._xproc_mark = (1, 1)
    a._xproc_at = time.monotonic()
    other.save(_waiter(run, "k"))
    assert a.list_event_waiters(wait_keys=["k"]) == [], "inside the floor and nothing moved: not re-read yet"
    a._xproc_at -= JsonFileRunStore.EVENT_INDEX_REFRESH_S
    assert [r.run_id for r in a.list_event_waiters(wait_keys=["k"])] == [run.run_id]


def test_an_unchanged_store_costs_no_run_file_stat(tmp_path: Path, monkeypatch) -> None:
    a = JsonFileRunStore(tmp_path)
    for _ in range(5):
        r = RunState.new(workflow_id="wf", entry_node="start")
        r.status = RunStatus.RUNNING
        a.save(r)
    a.save(_waiter(RunState.new(workflow_id="wf", entry_node="start"), "k"))
    assert len(a.list_event_waiters(wait_keys=["k"])) == 1
    calls: List[str] = []
    real = a._active_candidate_ids

    def counting(**kw: Any) -> Any:
        calls.append("x")
        return real(**kw)

    monkeypatch.setattr(a, "_active_candidate_ids", counting)
    for _ in range(3):
        assert len(a.list_event_waiters(wait_keys=["k"])) == 1
    assert calls == [], "nothing moved and the floor has not passed: no pass over the active runs"


def test_own_saves_are_not_re_read(tmp_path: Path) -> None:
    a = JsonFileRunStore(tmp_path)
    assert a.list_event_waiters(wait_keys=["k"]) == []
    for i in range(4):
        a.save(_waiter(RunState.new(workflow_id="wf", entry_node="start"), "k"))
    assert len(a.list_event_waiters(wait_keys=["k"])) == 4
    assert a.event_index_stats()["refiled"] == 0
