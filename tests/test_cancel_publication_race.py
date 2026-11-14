"""Cancellation wins the exact prepublication window, across real local stores."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from abstractruntime import Runtime, RunStatus, WorkflowSpec, StepPlan, Effect, EffectType, InMemoryRunStore, InMemoryLedgerStore
from abstractruntime.core.runtime import EffectOutcome
from abstractruntime.core.models import WaitState, WaitReason
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteRunStore, SqliteDatabase

pytestmark = pytest.mark.basic


def stores(kind, tmp_path):
    if kind == 'memory':
        store = InMemoryRunStore()
        return store, lambda: store
    if kind == 'json':
        return JsonFileRunStore(tmp_path), lambda: JsonFileRunStore(tmp_path)
    if kind == 'sqlite':
        path = tmp_path/'runs.sqlite'
        return SqliteRunStore(SqliteDatabase(path)), lambda: SqliteRunStore(SqliteDatabase(path))
    from abstractruntime.storage.offloading import OffloadingRunStore
    from abstractruntime.storage.artifacts import InMemoryArtifactStore
    return OffloadingRunStore(JsonFileRunStore(tmp_path), artifact_store=InMemoryArtifactStore()), lambda: JsonFileRunStore(tmp_path)


def workflow(mode):
    def node(run, ctx):
        if mode == 'pure':
            return StepPlan(node_id='n', complete_output={'done': True})
        return StepPlan(node_id='n', effect=Effect(type=EffectType.TOOL_CALLS, payload={}), next_node='n' if mode == 'wait' else None)
    return WorkflowSpec(workflow_id='publication-test', entry_node='n', nodes={'n': node})


@pytest.mark.parametrize('kind', ['memory','json','sqlite','offload'])
def test_cancel_in_missed_probe_window_is_not_overwritten(kind, tmp_path, monkeypatch):
    store, fresh = stores(kind, tmp_path)
    ledger = InMemoryLedgerStore()
    waiting = EffectOutcome.waiting(WaitState(reason=WaitReason.EVENT, wait_key='never', resume_to_node='n'))
    runtime = Runtime(run_store=store, ledger_store=ledger, effect_handlers={EffectType.TOOL_CALLS:lambda *args: waiting})
    wf = workflow('wait')
    run_id = runtime.start(workflow=wf, vars={})
    entered, release = threading.Event(), threading.Event()
    original = runtime._save_run_state
    candidate = []
    def before_lock(run):
        if run.run_id == run_id and run.status == RunStatus.WAITING:
            candidate.append(run)
            run._runtime_noncopyable_test_attachment = threading.Lock()
            entered.set()
            assert release.wait(10)
        return original(run)
    monkeypatch.setattr(runtime, '_save_run_state', before_lock)
    with ThreadPoolExecutor(1) as pool:
        ticking = pool.submit(runtime.tick, workflow=wf, run_id=run_id)
        try:
            assert entered.wait(10)
            cancelled = Runtime(run_store=store, ledger_store=ledger).cancel_run(run_id, reason='Human preemption', cancelled_by='operator')
            assert cancelled.status == RunStatus.CANCELLED and cancelled.waiting is None
            assert cancelled is not candidate[0]
        finally:
            release.set()
        assert ticking.result(10).status == RunStatus.CANCELLED
    persisted = fresh().load(run_id)
    assert persisted.status == RunStatus.CANCELLED and persisted.waiting is None
    assert persisted.error == 'Human preemption'
    assert persisted.vars['_runtime']['control']['cancelled_by'] == 'operator'
    candidate[0].status = RunStatus.RUNNING
    candidate[0].vars.setdefault('_runtime', {}).setdefault('control', {})['cancelled_by'] = 'stale candidate'
    final = fresh().load(run_id)
    assert final.status == RunStatus.CANCELLED
    assert final.vars['_runtime']['control']['cancelled_by'] == 'operator'
    assert not any(error.get('site') == 'tick' for error in runtime.health()['last_errors'])


@pytest.mark.parametrize('mode',['pure','effect'])
def test_cancel_before_completion_publication_appends_no_false_completion(mode, tmp_path, monkeypatch):
    import abstractruntime.core.runtime as module
    from contextlib import contextmanager
    store, ledger = InMemoryRunStore(), InMemoryLedgerStore()
    runtime = Runtime(run_store=store, ledger_store=ledger, effect_handlers={EffectType.TOOL_CALLS:lambda *args: EffectOutcome.completed({'done': True})})
    wf = workflow(mode)
    run_id = runtime.start(workflow=wf, vars={})
    entered, release = threading.Event(), threading.Event()
    original = module._RESUME_LOCKS.hold
    tick_thread = []
    @contextmanager
    def before_lock(key):
        if key == 'publication:' + run_id and threading.get_ident() in tick_thread and not entered.is_set():
            entered.set()
            assert release.wait(10)
        with original(key):
            yield
    monkeypatch.setattr(module._RESUME_LOCKS, 'hold', before_lock)
    def tick():
        tick_thread.append(threading.get_ident())
        return runtime.tick(workflow=wf, run_id=run_id)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(tick)
        try:
            assert entered.wait(10)
            Runtime(run_store=store, ledger_store=ledger).cancel_run(run_id)
        finally:
            release.set()
        assert future.result(10).status == RunStatus.CANCELLED
    rows = ledger.list(run_id)
    assert not any(str(row.get('node_id') or '').startswith('__completion') or str(row.get('idempotency_key') or '').startswith('system:completion') for row in rows)


def test_direct_effect_cancellation_is_persisted_to_json(tmp_path):
    store = JsonFileRunStore(tmp_path)
    runtime = Runtime(run_store=store, ledger_store=InMemoryLedgerStore(), effect_handlers={EffectType.TOOL_CALLS:lambda *args: EffectOutcome.cancelled({'reason':'direct cancellation'})})
    wf = workflow('effect')
    run_id = runtime.start(workflow=wf, vars={})
    assert runtime.tick(workflow=wf, run_id=run_id).status == RunStatus.CANCELLED
    assert JsonFileRunStore(tmp_path).load(run_id).status == RunStatus.CANCELLED


@pytest.mark.parametrize('operation',['pause','resume_paused','limits'])
def test_non_tick_public_api_returns_documented_shape_when_cancel_wins(operation, tmp_path, monkeypatch):
    store, ledger = JsonFileRunStore(tmp_path), InMemoryLedgerStore()
    runtime = Runtime(run_store=store, ledger_store=ledger)
    wf = workflow('pure')
    run_id = runtime.start(workflow=wf, vars={})
    if operation == 'resume_paused':
        runtime.pause_run(run_id)
    entered, release = threading.Event(), threading.Event()
    original = runtime._save_run_state
    def before_lock(run):
        entered.set()
        assert release.wait(10)
        return original(run)
    monkeypatch.setattr(runtime,'_save_run_state',before_lock)
    action = {'pause':lambda:runtime.pause_run(run_id), 'resume_paused':lambda:runtime.resume_run(run_id),
              'limits':lambda:runtime.update_limits(run_id, {'max_tokens':50})}[operation]
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(action)
        try:
            assert entered.wait(10)
            Runtime(run_store=store,ledger_store=ledger).cancel_run(run_id)
        finally:
            release.set()
        result = future.result(10)
    if operation == 'limits':
        assert result is None
    else:
        assert result.run_id == run_id and result.status == RunStatus.CANCELLED
    assert JsonFileRunStore(tmp_path).load(run_id).status == RunStatus.CANCELLED


def test_terminal_wait_resume_and_cancel_share_completion_publication(tmp_path, monkeypatch):
    import abstractruntime.core.runtime as module
    from contextlib import contextmanager
    store, ledger = JsonFileRunStore(tmp_path), InMemoryLedgerStore()
    waiting = EffectOutcome.waiting(WaitState(reason=WaitReason.EVENT,wait_key='answer',resume_to_node=None))
    runtime=Runtime(run_store=store,ledger_store=ledger,effect_handlers={EffectType.TOOL_CALLS:lambda *args:waiting})
    wf=workflow('effect'); run_id=runtime.start(workflow=wf,vars={})
    assert runtime.tick(workflow=wf,run_id=run_id).status == RunStatus.WAITING
    entered,release=threading.Event(),threading.Event()
    original=module._RESUME_LOCKS.hold; worker=[]
    @contextmanager
    def before_lock(key):
        if key == 'publication:'+run_id and threading.get_ident() in worker and not entered.is_set():
            entered.set(); assert release.wait(10)
        with original(key): yield
    monkeypatch.setattr(module._RESUME_LOCKS,'hold',before_lock)
    def resume():
        worker.append(threading.get_ident())
        return runtime.resume(workflow=wf,run_id=run_id,wait_key='answer',payload={'answer':'yes'})
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(resume)
        try:
            assert entered.wait(10)
            Runtime(run_store=store,ledger_store=ledger).cancel_run(run_id)
        finally: release.set()
        assert future.result(10).status == RunStatus.CANCELLED
    assert not any(str(row.get('idempotency_key') or '').startswith('system:completion:') for row in ledger.list(run_id))
    assert JsonFileRunStore(tmp_path).load(run_id).status == RunStatus.CANCELLED


def test_cancel_does_not_wait_for_blocked_provider_to_release(tmp_path):
    from abstractruntime.core.effect_cancellation import current_effect_cancel_event
    store, ledger=JsonFileRunStore(tmp_path),InMemoryLedgerStore()
    entered,release=threading.Event(),threading.Event(); events=[]
    def provider(*args):
        events.append(current_effect_cancel_event())
        entered.set(); assert release.wait(10)
        return EffectOutcome.completed({'late':'reply'})
    runtime=Runtime(run_store=store,ledger_store=ledger,effect_handlers={EffectType.TOOL_CALLS:provider})
    wf=workflow('effect'); run_id=runtime.start(workflow=wf,vars={})
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(runtime.tick,workflow=wf,run_id=run_id)
        try:
            assert entered.wait(10)
            result=Runtime(run_store=store,ledger_store=ledger).cancel_run(run_id)
            assert result.status == RunStatus.CANCELLED and events[0].is_set()
            assert not release.is_set()
        finally: release.set()
        assert future.result(10).status == RunStatus.CANCELLED


def test_foreign_memory_owner_rejection_never_returns_target_as_executing_run(tmp_path, monkeypatch):
    from abstractruntime.storage.artifacts import InMemoryArtifactStore
    store,ledger=JsonFileRunStore(tmp_path),InMemoryLedgerStore()
    runtime=Runtime(run_store=store,ledger_store=ledger,artifact_store=InMemoryArtifactStore())
    target=runtime.start(workflow=workflow('pure'),vars={})
    def note(run,ctx):
        return StepPlan(node_id='note',effect=Effect(type=EffectType.MEMORY_NOTE,payload={'note':'A note','target_run_id':target}),next_node=None)
    wf=WorkflowSpec(workflow_id='foreign-note',entry_node='note',nodes={'note':note})
    run_id=runtime.start(workflow=wf,vars={})
    original=runtime._save_run_state
    def cancel_target_before_save(candidate):
        if candidate.run_id == target:
            Runtime(run_store=store,ledger_store=ledger).cancel_run(target)
        return original(candidate)
    monkeypatch.setattr(runtime,'_save_run_state',cancel_target_before_save)
    state=runtime.tick(workflow=wf,run_id=run_id)
    assert state.run_id == run_id and state.status == RunStatus.FAILED
    assert target in str(state.error) and 'cancelled' in str(state.error)
    assert JsonFileRunStore(tmp_path).load(target).status == RunStatus.CANCELLED
