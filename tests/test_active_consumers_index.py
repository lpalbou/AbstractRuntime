"""The active-consumers index (`automations.consumers`, 2026-10-26): every "who reads this
stream?" answer comes from one snapshot per runtime, rebuilt once per change.

Benchmark: a generated store of 1,000 controller runs (200 active, the rest paused, archived or
finished) behind 10 feed connections. Regression: the index follows every automation command
(create, pause, resume, revise, archive, unarchive) at once, and a change outside the protocol
within the TTL.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from types import SimpleNamespace

import pytest

from automation_harness import TARGETS, Clock, drive, request
from abstractruntime import Runtime
from abstractruntime.automations import apply_automation_command, create_automation, register_controller_bundle
from abstractruntime.automations.consumers import ActiveConsumersIndex, active_consumers_index, invalidate_active_consumers
from abstractruntime.automations.ledger import definition_of
from abstractruntime.automations.models import automation_status
from abstractruntime.core.models import RunState, RunStatus, WaitReason, WaitState
from abstractruntime.email import email_trigger_consumers, stream_consumers
from abstractruntime.email.inbox import InMemoryEventInbox
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.triggers import TriggerConnectionStore, connection_principal
from abstractruntime.triggers.connections import connection_trigger_consumers, connection_usage, connections_usage

ALICE = connection_principal("t", "alice")
N_CONNECTIONS = 10
N_RUNS = 1_000
N_ACTIVE = 200


# --- the generated store -----------------------------------------------------------------


def _controller(i: int, *, kind: str, connection_id: str) -> RunState:
    """A controller run as `create_automation` lays it out (only the fields the index reads,
    plus a few KB of state, as a real controller carries)."""
    archived = kind == "archived"
    # An archived controller ends (`read_definition` -> end); a finished one completed or failed.
    status = {"finished": RunStatus.COMPLETED if i % 2 else RunStatus.FAILED, "archived": RunStatus.COMPLETED}.get(kind, RunStatus.WAITING)
    trigger = (
        {"source_id": "email.received", "source_version": 1, "config": {"account": "self"}, "binding_id": f"b{i}"}
        if i % 10 == 0
        else {"source_id": "feed.item", "source_version": 1, "config": {"connection_id": connection_id}, "binding_id": f"b{i}"}
    )
    run_id = str(uuid.UUID(int=i + 1))
    return RunState(
        run_id=run_id,
        workflow_id="abstractframework.automation-controller@1.0.0:controller",
        status=status,
        current_node="wait",
        vars={
            "_meta": {"automation": {"revision": 1, "trigger": trigger, "archived_at": "2026-01-01T00:00:00+00:00" if archived else None}},
            "_runtime": {"automation": {"paused": kind == "paused", "state_version": 3, "history": ["x" * 64] * 40}},
        },
        waiting=WaitState(reason=WaitReason.EVENT, wait_key=f"automation:wake:{run_id}") if status == RunStatus.WAITING else None,
    )


def _kind(i: int) -> str:
    if i < N_ACTIVE:
        return "active"
    return ("paused", "archived", "finished")[i % 3]


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    root = tmp_path_factory.mktemp("consumers-bench")
    writer = JsonFileRunStore(root / "runs")
    for i in range(N_RUNS):
        writer.save(_controller(i, kind=_kind(i), connection_id=f"tc_{i % N_CONNECTIONS}"))
    writer.list_run_index(limit=1_000_000)
    writer._persist_scan_memo()  # the scan sidecar a running host keeps current
    conns = TriggerConnectionStore(root / "plane", sealing_key_file=root / "sealing.key")
    for n in range(N_CONNECTIONS):
        conns.create(principal=ALICE, source_id="feed.item", source_version=1,
                     config={"url": f"https://news.example.test/{n}.xml", "every": "60s"}, label=f"feed {n}")
    listed = conns.list(principal=ALICE)
    assert len(listed) == N_CONNECTIONS
    return root, listed


def _brute_force(run_store, wanted):
    """The pre-index algorithm: every controller run loaded, every call."""
    out = []
    for row in run_store.list_run_index(limit=1_000_000, role="controller"):
        run = run_store.load(str(row["run_id"]))
        trigger = definition_of(run)["trigger"]
        if wanted(trigger) and automation_status(run) == "active":
            out.append(run.run_id)
    return sorted(out)


def test_benchmark_1000_controller_runs(big):
    root, conns = big
    conns = [type(c).from_dict({**c.to_dict(), "connection_id": f"tc_{n}"}) for n, c in enumerate(conns)]
    # COLD: a fresh store object over the folder (a restarted host: the scan sidecar is current,
    # nothing is loaded in memory) and no index yet. Best of 3 fresh hosts: the bound is on the
    # work, not on a loaded machine's scheduler.
    colds = []
    for _ in range(3):
        runtime = SimpleNamespace(run_store=JsonFileRunStore(root / "runs"), email_binding=None)
        index = active_consumers_index(runtime)
        t0 = time.perf_counter()
        usage = connections_usage(runtime, conns)
        colds.append((time.perf_counter() - t0) * 1000)
    cold_ms = min(colds)

    t0 = time.perf_counter()
    for _ in range(20):
        connections_usage(runtime, conns)
    warm_ms = (time.perf_counter() - t0) * 1000 / 20

    t0 = time.perf_counter()
    for _ in range(100):
        connection_usage(runtime, conns[3])  # a feed poll's usage check
    feed_ms = (time.perf_counter() - t0) * 1000 / 100

    assert cold_ms < 300, f"connections listing cold: {cold_ms:.1f} ms"
    assert warm_ms < 20, f"connections listing warm: {warm_ms:.3f} ms"
    assert feed_ms < 5, f"feed poll usage check warm: {feed_ms:.4f} ms"
    assert index.rebuilds == 1  # 1 + 20 listings + 100 feed checks: one pass over the controllers

    # The answers are the brute-force ones.
    store = runtime.run_store
    for conn, row in zip(conns, usage):
        expected = _brute_force(store, lambda t, cid=conn.connection_id: (t.get("config") or {}).get("connection_id") == cid)
        assert row == {"listening": "listening" if expected else "idle", "used_by": len(expected)}
        assert connection_trigger_consumers(runtime, connection_id=conn.connection_id) == expected
        assert stream_consumers(runtime, conn.stream) == expected
    assert sum(r["used_by"] for r in usage) == N_ACTIVE - N_ACTIVE // 10  # every 10th active one reads mail
    assert email_trigger_consumers(runtime) == _brute_force(store, lambda t: t.get("source_id") == "email.received")
    assert len(email_trigger_consumers(runtime)) == N_ACTIVE // 10
    assert index.rebuilds == 1

    # One change, one rebuild — however many callers ask afterwards.
    invalidate_active_consumers(store)
    for _ in range(50):
        connections_usage(runtime, conns)
        email_trigger_consumers(runtime)
    assert index.rebuilds == 2


# --- the index follows every command -------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    registry = WorkflowRegistry()
    for spec in TARGETS.values():
        registry.register(spec)
    register_controller_bundle(registry)
    runtime = Runtime(run_store=JsonFileRunStore(tmp_path / "runs"), ledger_store=JsonlLedgerStore(tmp_path / "ledger"),
                      workflow_registry=registry, artifact_store=FileArtifactStore(tmp_path / "artifacts"))
    runtime.set_event_inbox(InMemoryEventInbox())
    store = TriggerConnectionStore(tmp_path / "plane", sealing_key_file=tmp_path / "data" / "secrets" / "sealing.key")
    runtime.set_trigger_connection_store(store)
    return runtime, store, Clock(monkeypatch)


def _feed(store):
    return store.create(principal=ALICE, source_id="feed.item", source_version=1, config={"url": "https://x.test/f.xml", "every": "60s"})


def test_the_index_invalidates_on_each_automation_command(env):
    runtime, store, clock = env
    a, b = _feed(store), _feed(store)
    index = active_consumers_index(runtime)
    used = lambda: [u["used_by"] for u in connections_usage(runtime, [a, b])]  # noqa: E731
    assert used() == [0, 0]
    rebuilds = index.rebuilds

    def step(expected, command=None, **payload):
        nonlocal rebuilds
        if command is not None:
            out = apply_automation_command(runtime, automation_id=aid, command_id=str(uuid.uuid4()), type=command, payload=payload or None)
            assert out["status"] == "applied", out
        assert used() == expected, command
        assert used() == expected
        assert index.rebuilds == rebuilds + 1, f"{command}: exactly one rebuild per change"
        rebuilds = index.rebuilds

    req = request(trigger={"source_id": "feed.item", "source_version": 1, "config": {"connection_id": a.connection_id}})
    req.update({"tenant": "t", "user": "alice"})
    aid, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, aid)
    step([1, 0])
    step([0, 0], "automation.pause")
    step([1, 0], "automation.resume")
    step([0, 1], "automation.revise", changes={"trigger": {"source_id": "feed.item", "source_version": 1, "config": {"connection_id": b.connection_id}}})
    step([0, 0], "automation.archive")
    drive(runtime, aid)  # the archived controller ends
    step([0, 0], "automation.unarchive")  # back PAUSED
    drive(runtime, aid)
    step([0, 1], "automation.resume")


def _bound(runtime, store, clock):
    conn = _feed(store)
    req = request(trigger={"source_id": "feed.item", "source_version": 1, "config": {"connection_id": conn.connection_id}})
    req.update({"tenant": "t", "user": "alice"})
    aid, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, aid)
    return conn, aid


def test_a_controller_saved_outside_the_protocol_is_seen_at_once(env):
    runtime, store, clock = env
    conn, aid = _bound(runtime, store, clock)
    index = active_consumers_index(runtime)
    assert connection_usage(runtime, conn)["used_by"] == 1
    rebuilds = index.rebuilds
    # An ordinary save (nothing the index holds changed) bumps nothing.
    runtime.run_store.save(runtime.run_store.load(aid))
    assert connection_usage(runtime, conn)["used_by"] == 1 and index.rebuilds == rebuilds
    # The controller fails on its own (no decision, no command): its index row changed.
    run = runtime.run_store.load(aid)
    run.status, run.waiting = RunStatus.FAILED, None
    runtime.run_store.save(run)
    assert connection_usage(runtime, conn)["used_by"] == 0
    assert index.rebuilds == rebuilds + 1


def test_another_process_change_is_seen_within_the_ttl(env, tmp_path):
    runtime, store, clock = env
    conn, aid = _bound(runtime, store, clock)
    now = [1000.0]
    index = ActiveConsumersIndex(runtime, ttl_s=30.0, clock=lambda: now[0])
    runtime._active_consumers_index = index
    assert connection_usage(runtime, conn)["used_by"] == 1
    # Another process fails the controller: its file changes, nothing in this process hears it.
    path = tmp_path / "runs" / f"run_{aid}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(status="failed", waiting=None)
    tmp = path.with_name(path.name + ".foreign.tmp")
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    os.replace(tmp, path)
    now[0] += 29.0
    assert connection_usage(runtime, conn)["used_by"] == 1  # within the TTL: the snapshot stands
    assert index.rebuilds == 1
    now[0] += 1.5
    assert connection_usage(runtime, conn)["used_by"] == 0  # past it: rebuilt, truthful again
    # That rebuild's own scan noticed the changed row (a bump): the next call rebuilds once more.
    assert connection_usage(runtime, conn)["used_by"] == 0
    settled = index.rebuilds
    assert settled <= 3
    for _ in range(5):
        assert connection_usage(runtime, conn)["used_by"] == 0
    assert index.rebuilds == settled


def test_two_runtimes_over_one_store_invalidate_each_other(env, tmp_path):
    runtime, store, clock = env
    conn = _feed(store)
    other = SimpleNamespace(run_store=JsonFileRunStore(tmp_path / "runs"), email_binding=None)
    assert connection_usage(other, conn)["used_by"] == 0
    req = request(trigger={"source_id": "feed.item", "source_version": 1, "config": {"connection_id": conn.connection_id}})
    req.update({"tenant": "t", "user": "alice"})
    aid, _ = create_automation(runtime, req, now=clock.now)
    drive(runtime, aid)
    assert connection_usage(other, conn)["used_by"] == 1  # the generation is per store folder
