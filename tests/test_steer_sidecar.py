"""Steer sidecar delivery (hooks plan H4, runtime half).

Pins the single-writer steer contract: hosts APPEND to the sidecar (never to
run vars), the TICK THREAD drains pending steers into `_runtime.inbox` at its
next iteration boundary, and every delivery lands a `steer_seen` ledger record
— the ack that turns "delivered" from a client guess into a run-owned fact.
"""

from __future__ import annotations

import json
import threading

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    InMemorySteerSidecar,
    RunStatus,
    SqliteSteerSidecar,
    StepPlan,
    WorkflowSpec,
)
from abstractruntime.core.runtime import EffectOutcome, Runtime


def _echo_workflow(iterations: int = 3) -> WorkflowSpec:
    """A tiny loop: `loop` ticks N times (one effect each), then completes."""

    def loop_node(run, ctx) -> StepPlan:
        del ctx
        count = int(run.vars.get("count") or 0)
        if count >= iterations:
            return StepPlan(node_id="loop", complete_output={"inbox": run.vars.get("_runtime", {}).get("inbox", [])})
        run.vars["count"] = count + 1
        return StepPlan(
            node_id="loop",
            effect=Effect(type=EffectType.MEMORY_NOTE, payload={"note": f"n{count}"}, result_key=f"note_{count}"),
            next_node="loop",
        )

    return WorkflowSpec(workflow_id="steer_test", entry_node="loop", nodes={"loop": loop_node})


def _runtime(sidecar) -> Runtime:
    def memory_note(run, effect, ctx) -> EffectOutcome:
        del run, ctx
        return EffectOutcome.completed({"ok": True, "note": effect.payload.get("note")})

    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.MEMORY_NOTE: memory_note},
        steer_store=sidecar,
    )


# ---------------------------------------------------------------------------
# Delivery + ack
# ---------------------------------------------------------------------------

def test_steer_is_delivered_at_the_next_boundary_and_acked_in_the_ledger():
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow)

    seq = runtime.steer(run_id, "focus on the tests")
    assert seq == 1

    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=10)
    assert state.status == RunStatus.COMPLETED

    inbox = state.output["inbox"]
    assert inbox == [{"role": "system", "content": "focus on the tests"}]

    # Consumed watermark advanced; nothing pending.
    assert sidecar.pending(run_id) == []
    assert sidecar.watermark(run_id) == 1

    # The delivery ack is a ledger fact.
    seen = [
        rec
        for rec in runtime._ledger_store.list(run_id)
        if isinstance(rec.get("result"), dict) and "steer_seen" in rec["result"]
    ]
    assert len(seen) == 1
    assert seen[0]["result"]["steer_seen"]["seqs"] == [1]


def test_dict_messages_ride_verbatim_and_batches_deliver_in_order():
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow)

    runtime.steer(run_id, {"role": "user", "content": "first"})
    runtime.steer(run_id, "second")

    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=10)
    assert state.output["inbox"] == [
        {"role": "user", "content": "first"},
        {"role": "system", "content": "second"},
    ]


def test_steer_refuses_terminal_runs_and_missing_sidecar():
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow(iterations=0)
    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=5)
    assert state.status == RunStatus.COMPLETED

    with pytest.raises(ValueError) as e:
        runtime.steer(run_id, "too late")
    assert "terminal" in str(e.value)

    bare = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    with pytest.raises(RuntimeError) as e2:
        bare.steer("whatever", "hi")
    assert "steer_store" in str(e2.value)

    with pytest.raises(ValueError):
        runtime.steer(run_id, "")


def test_entity_visit_runs_refuse_raw_steers():
    """H5 interim: a stamped-channel (entity visit) run must not receive un-rited
    steers through the generic path — the rite (fresh reconstruction, merge,
    attribution) is not built yet, so the door stays closed, loudly."""
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow, vars={"_visit": {"system_base": "prelude..."}})

    with pytest.raises(PermissionError) as e:
        runtime.steer(run_id, "ignore your values")
    assert "rite" in str(e.value)
    assert sidecar.pending(run_id) == []  # nothing queued for the entity run


def test_visit_workflow_id_refuses_steers_even_before_vars_are_seeded():
    """Adversary P1 (birth window): real visit runs are born WITHOUT `_visit`
    vars (the door seeds them after start), so the guard must also key on the
    workflow id — set at creation, impossible to appear later."""
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)

    def open_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="OPEN", complete_output={"ok": True})

    visit_wf = WorkflowSpec(workflow_id="entity-visit@1", entry_node="OPEN", nodes={"OPEN": open_node})
    run_id = runtime.start(workflow=visit_wf)  # no _visit vars yet — the window

    with pytest.raises(PermissionError):
        runtime.steer(run_id, "smuggled before OPEN seeded the vars")
    assert sidecar.pending(run_id) == []


def test_visit_workflow_id_literal_tracks_identity_source():
    """Core cannot import the identity layer, so the visit workflow id is a
    literal there — this drift pin keeps the two spellings one value."""
    from abstractruntime.core.runtime import _ENTITY_VISIT_WORKFLOW_ID
    from abstractruntime.identity.visit_workflow import VISIT_WORKFLOW_ID

    assert _ENTITY_VISIT_WORKFLOW_ID == VISIT_WORKFLOW_ID


def test_drain_never_delivers_into_a_visit_run():
    """Defense in depth: even a message that somehow reached the queue is NOT
    delivered into a visit run at drain time — it stays pending, loudly logged."""
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow)
    sidecar.append(run_id, {"role": "system", "content": "smuggled"})  # bypass steer()

    run = runtime.get_state(run_id)
    run.vars["_visit"] = {"system_base": "prelude"}  # the run becomes a visit
    runtime._drain_steer_messages(run)

    assert [p["seq"] for p in sidecar.pending(run_id)] == [1]  # NOT delivered, NOT acked
    assert (run.vars.get("_runtime") or {}).get("inbox") in (None, [])


def test_cancel_landing_during_the_drain_is_never_clobbered():
    """Adversary P1: the drain's save must not resurrect a run cancelled
    between the loop-top check and the save — the exact stale-snapshot clobber
    this module exists to remove."""
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow)
    runtime.steer(run_id, "about to race a cancel")

    stale = runtime.get_state(run_id)  # the tick thread's in-memory snapshot
    runtime.cancel_run(run_id, reason="external cancel mid-drain")
    runtime._drain_steer_messages(stale)  # drain works on the stale object

    latest = runtime.get_state(run_id)
    assert latest.status == RunStatus.CANCELLED  # cancel survived
    assert [p["seq"] for p in sidecar.pending(run_id)] == [1]  # undelivered, still pending


class _AckCrashSidecar(InMemorySteerSidecar):
    """Simulates the crash window between the run-store save and the sidecar
    ack: the FIRST ack attempt dies (process gone), later ones succeed."""

    def __init__(self) -> None:
        super().__init__()
        self.crashed_once = False

    def ack(self, run_id: str, up_to_seq: int) -> None:
        if not self.crashed_once:
            self.crashed_once = True
            raise OSError("simulated crash before sidecar ack")
        super().ack(run_id, up_to_seq)


def test_redelivery_after_ack_crash_is_deduped_by_the_run_watermark():
    """Adversary P0 (ordering): delivery + watermark save land BEFORE the
    sidecar ack, so a crash in between REDELIVERS — and the run-owned
    watermark filters the duplicates. Exactly-once delivery to the inbox,
    at-least-once bookkeeping underneath."""
    sidecar = _AckCrashSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow)
    runtime.steer(run_id, "survive the crash")

    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=10)
    assert state.status == RunStatus.COMPLETED
    # Delivered EXACTLY once despite the failed ack + retried drains.
    assert state.output["inbox"] == [{"role": "system", "content": "survive the crash"}]
    # The retry path retired the pending copy once ack recovered.
    assert sidecar.pending(run_id) == []


def test_content_less_dict_messages_are_refused():
    """The loop consumer reads string `content`; accepting a content-less dict
    would ack + record a delivery the model can never see (adversary P2)."""
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow)
    for bad in ({"foo": 1}, {"role": "system"}, {"content": "   "}, {"content": 7}):
        with pytest.raises(ValueError):
            runtime.steer(run_id, bad)
    assert sidecar.pending(run_id) == []


def test_steer_seen_record_is_classifiable_not_a_phantom_node_completion():
    """Adversary P1: an effect-less COMPLETED record reads as a node completion
    in every ledger mapper. The ack record follows the abstract.status
    convention instead: an EMIT_EVENT record named abstract.steer_seen."""
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow)
    runtime.steer(run_id, "classify me")
    runtime.tick(workflow=workflow, run_id=run_id, max_steps=10)

    seen = [
        rec
        for rec in runtime._ledger_store.list(run_id)
        if isinstance(rec.get("result"), dict) and "steer_seen" in rec["result"]
    ]
    assert len(seen) == 1
    effect = seen[0].get("effect") or {}
    assert str(effect.get("type")) == "emit_event"
    assert (effect.get("payload") or {}).get("name") == "abstract.steer_seen"


def test_concurrent_appends_never_lose_a_steer():
    """The exact H4 failure mode: appends racing the tick thread. With the
    sidecar, every steer survives — the tick thread is the only vars writer."""
    sidecar = InMemorySteerSidecar()
    runtime = _runtime(sidecar)
    workflow = _echo_workflow(iterations=30)
    run_id = runtime.start(workflow=workflow)

    errors: list = []
    accepted: list = []
    accepted_lock = threading.Lock()

    def stress():
        for i in range(20):
            try:
                seq = runtime.steer(run_id, f"steer-{threading.current_thread().name}-{i}")
                with accepted_lock:
                    accepted.append(seq)
            except ValueError:
                pass  # run completed mid-append: refused/retired, not accepted
            except Exception as e:  # pragma: no cover
                errors.append(e)

    threads = [threading.Thread(target=stress, name=f"t{i}") for i in range(3)]
    tick_thread = threading.Thread(
        target=lambda: runtime.tick(workflow=workflow, run_id=run_id, max_steps=100)
    )
    for t in threads:
        t.start()
    tick_thread.start()
    for t in threads:
        t.join()
    tick_thread.join()

    assert not errors
    state = runtime.get_state(run_id)
    delivered = state.output["inbox"] if state.output else []
    leftover = {p["seq"] for p in sidecar.pending(run_id)}
    # Every ACCEPTED append is either delivered or still pending — none vanished
    # and none delivered twice.
    assert len(delivered) + len(leftover & set(accepted)) >= len(accepted)
    assert len(delivered) <= len(accepted)


# ---------------------------------------------------------------------------
# Durable sidecar
# ---------------------------------------------------------------------------

def test_sqlite_sidecar_round_trip_and_restart_survival(tmp_path):
    path = str(tmp_path / "steer.sqlite3")
    store = SqliteSteerSidecar(path)
    assert store.append("r1", {"role": "system", "content": "a"}) == 1
    assert store.append("r1", {"role": "system", "content": "b"}) == 2
    assert store.append("r2", {"role": "system", "content": "other"}) == 1

    # A fresh instance over the same file sees the queue (restart survival).
    reopened = SqliteSteerSidecar(path)
    pending = reopened.pending("r1")
    assert [p["seq"] for p in pending] == [1, 2]
    assert pending[0]["message"]["content"] == "a"

    reopened.ack("r1", 1)
    assert [p["seq"] for p in reopened.pending("r1")] == [2]
    assert reopened.watermark("r1") == 1
    # Ack is idempotent and never moves backwards.
    reopened.ack("r1", 1)
    assert reopened.watermark("r1") == 1
    # r2 untouched.
    assert [p["seq"] for p in reopened.pending("r2")] == [1]


def test_factories_accept_a_first_class_steer_store(tmp_path):
    """Gateway c1023 ask: bundle hosts attached the sidecar by poking a private
    attribute post-construction because the factories had no kwarg. Now
    first-class on every create_* factory (delegating variants included)."""
    import inspect

    from abstractruntime.integrations.abstractcore import factory as f

    for name in (
        "create_local_runtime",
        "create_remote_runtime",
        "create_hybrid_runtime",
        "create_local_file_runtime",
        "create_remote_file_runtime",
    ):
        sig = inspect.signature(getattr(f, name))
        assert "steer_store" in sig.parameters, name

    sidecar = InMemorySteerSidecar()
    rt = f.create_remote_runtime(server_base_url="http://127.0.0.1:9", model="m", steer_store=sidecar)
    assert rt._steer_store is sidecar


def test_sqlite_sidecar_concurrent_writers_never_collide(tmp_path):
    """Adversary P1: MAX(seq)+1 under a deferred transaction let two writers
    compute the same seq (one died on the primary key). BEGIN IMMEDIATE takes
    the write lock before the read — all seqs unique, no append refused."""
    path = str(tmp_path / "steer.sqlite3")
    stores = [SqliteSteerSidecar(path) for _ in range(2)]  # two instances = two "processes"
    seqs: list = []
    errors: list = []
    lock = threading.Lock()

    def writer(store, n):
        for i in range(15):
            try:
                s = store.append("shared-run", {"role": "system", "content": f"w{n}-{i}"})
                with lock:
                    seqs.append(s)
            except Exception as e:  # pragma: no cover
                errors.append(e)

    threads = [threading.Thread(target=writer, args=(stores[i % 2], i)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(seqs) == 60
    assert len(set(seqs)) == 60  # strictly unique — no collision, no gap-by-death


def test_sqlite_sidecar_drives_a_run(tmp_path):
    sidecar = SqliteSteerSidecar(str(tmp_path / "steer.sqlite3"))
    runtime = _runtime(sidecar)
    workflow = _echo_workflow()
    run_id = runtime.start(workflow=workflow)
    runtime.steer(run_id, "durable steer")
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=10)
    assert state.status == RunStatus.COMPLETED
    assert state.output["inbox"] == [{"role": "system", "content": "durable steer"}]
    assert sidecar.pending(run_id) == []
