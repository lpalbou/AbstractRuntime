"""The event-wait index must answer EXACTLY what the whole-store scan did.

`emit_event` used to find its listeners with `list_runs(status=WAITING,
wait_reason=EVENT, limit=10_000)` — a walk of every run file in the directory,
twice per agent chat turn. `JsonFileRunStore.list_event_waiters` answers the
same question in O(waiters). The bar these tests hold it to is equivalence with
the scan, not "it finds something": every case below computes the scan's answer
in the test and asserts the index reproduces it.
"""
from __future__ import annotations

from abstractruntime import RunState, RunStatus
from abstractruntime.core.models import WaitReason, WaitState
from abstractruntime.storage.json_files import JsonFileRunStore


def _waiter(*, wait_key: str, workflow_id: str = "wf", session_id: str | None = None) -> RunState:
    run = RunState.new(workflow_id=workflow_id, entry_node="start", session_id=session_id)
    run.status = RunStatus.WAITING
    run.waiting = WaitState(
        reason=WaitReason.EVENT,
        wait_key=wait_key,
        resume_to_node="next",
        result_key="evt",
    )
    return run


def _other(*, status: RunStatus, wait: WaitState | None = None) -> RunState:
    run = RunState.new(workflow_id="wf", entry_node="start")
    run.status = status
    run.waiting = wait
    return run


def _scan(store: JsonFileRunStore, keys: list[str]) -> list[str]:
    """What the pre-index code did: scan, then filter on wait_key."""
    rows = store.list_runs(status=RunStatus.WAITING, wait_reason=WaitReason.EVENT, limit=10_000)
    return [r.run_id for r in rows if getattr(r.waiting, "wait_key", None) in keys]


def _indexed(store: JsonFileRunStore, keys: list[str]) -> list[str]:
    return [r.run_id for r in store.list_event_waiters(wait_keys=keys, limit=10_000)]


def test_index_matches_the_scan_across_every_scope(tmp_path):
    store = JsonFileRunStore(tmp_path)

    keys = {
        "session": "evt:session:s1:ready",
        "session_other": "evt:session:s2:ready",
        "session_wildcard": "evt:session:s1:*",
        "workflow": "evt:workflow:wf:ready",
        "run": "evt:run:r1:ready",
        "global": "evt:global:global:ready",
        "same_key_twin": "evt:session:s1:ready",
    }
    waiters = {name: _waiter(wait_key=k, session_id="s1") for name, k in keys.items()}
    for run in waiters.values():
        store.save(run)

    # Decoys the scan would also have walked past.
    store.save(_other(status=RunStatus.RUNNING))
    store.save(_other(status=RunStatus.COMPLETED))
    store.save(_other(status=RunStatus.FAILED))
    store.save(_other(status=RunStatus.WAITING, wait=WaitState(reason=WaitReason.USER, wait_key="user:input")))
    store.save(_other(status=RunStatus.WAITING, wait=WaitState(reason=WaitReason.UNTIL, wait_key="until:x")))
    store.save(
        _other(status=RunStatus.WAITING, wait=WaitState(reason=WaitReason.SUBWORKFLOW, wait_key="subworkflow:abc"))
    )

    for probe in (
        ["evt:session:s1:ready"],
        ["evt:session:s1:ready", "evt:session:s1:*"],
        ["evt:workflow:wf:ready"],
        ["evt:run:r1:ready"],
        ["evt:global:global:ready"],
        ["evt:session:s2:ready"],
        ["evt:session:s1:nobody-waits-on-this"],
    ):
        assert sorted(_indexed(store, probe)) == sorted(_scan(store, probe)), probe

    # Two runs on the SAME key are both delivered to (at-least-once fan-out).
    both = set(_indexed(store, ["evt:session:s1:ready"]))
    assert both == {waiters["session"].run_id, waiters["same_key_twin"].run_id}


def test_index_orders_like_the_scan(tmp_path):
    store = JsonFileRunStore(tmp_path)
    key = "evt:session:s1:ready"
    for i in range(5):
        run = _waiter(wait_key=key)
        run.updated_at = f"2026-09-2{i}T00:00:00+00:00"
        store.save(run)
    assert _indexed(store, [key]) == _scan(store, [key])


def test_a_resumed_waiter_is_retracted_on_save(tmp_path):
    store = JsonFileRunStore(tmp_path)
    key = "evt:session:s1:ready"
    run = _waiter(wait_key=key)
    store.save(run)
    assert _indexed(store, [key]) == [run.run_id]

    run.status = RunStatus.RUNNING
    run.waiting = None
    store.save(run)

    assert _scan(store, [key]) == []
    assert _indexed(store, [key]) == []


def test_reparking_on_another_key_moves_the_entry(tmp_path):
    store = JsonFileRunStore(tmp_path)
    old, new = "evt:session:s1:first", "evt:session:s1:second"
    run = _waiter(wait_key=old)
    store.save(run)
    assert _indexed(store, [old]) == [run.run_id]

    run.waiting = WaitState(reason=WaitReason.EVENT, wait_key=new, resume_to_node="next")
    store.save(run)

    assert _indexed(store, [old]) == _scan(store, [old]) == []
    assert _indexed(store, [new]) == _scan(store, [new]) == [run.run_id]


def test_a_terminal_waiter_is_retracted(tmp_path):
    store = JsonFileRunStore(tmp_path)
    key = "evt:session:s1:ready"
    run = _waiter(wait_key=key)
    store.save(run)
    run.status = RunStatus.COMPLETED
    store.save(run)
    assert _indexed(store, [key]) == _scan(store, [key]) == []


def test_delete_drops_the_entry(tmp_path):
    store = JsonFileRunStore(tmp_path)
    key = "evt:session:s1:ready"
    run = _waiter(wait_key=key)
    store.save(run)
    assert _indexed(store, [key]) == [run.run_id]
    store.delete(run.run_id)
    assert _indexed(store, [key]) == _scan(store, [key]) == []


def test_a_restart_rebuilds_the_index_from_disk(tmp_path):
    """The index is in-memory; a process restart must heal it with one scan."""
    writer = JsonFileRunStore(tmp_path)
    key = "evt:session:s1:ready"
    run = _waiter(wait_key=key)
    writer.save(run)

    reader = JsonFileRunStore(tmp_path)  # fresh process
    assert _indexed(reader, [key]) == _scan(reader, [key]) == [run.run_id]


def test_a_stale_entry_is_verified_against_disk_never_resumed(tmp_path):
    """Index says 'waiter', disk says otherwise -> the run is NOT returned.

    This is the direction that would cause a wrong resume, so it is closed by
    re-loading and re-checking every candidate. (The opposite direction — a
    waiter another PROCESS parked, which this index has not seen — is the
    documented single-writer limit, healed by the restart rebuild above.)
    """
    store = JsonFileRunStore(tmp_path)
    key = "evt:session:s1:ready"
    run = _waiter(wait_key=key)
    store.save(run)
    assert _indexed(store, [key]) == [run.run_id]

    # Another writer moves the run on (what a split-runner deployment does).
    other = JsonFileRunStore(tmp_path)
    moved = other.load(run.run_id)
    assert moved is not None
    moved.status = RunStatus.COMPLETED
    moved.waiting = None
    other.save(moved)

    assert _scan(store, [key]) == []
    assert _indexed(store, [key]) == []


def test_lookup_is_O_waiters_not_O_store(tmp_path):
    """One directory walk at first use, then never again — the whole point."""
    store = JsonFileRunStore(tmp_path)
    key = "evt:session:s1:ready"
    store.save(_waiter(wait_key=key))
    for _ in range(50):
        store.save(_other(status=RunStatus.COMPLETED))

    calls = {"n": 0}
    original = store._scan_fields

    def counted(p):
        calls["n"] += 1
        return original(p)

    store._scan_fields = counted  # type: ignore[method-assign]

    store.list_event_waiters(wait_keys=[key], limit=10)
    first = calls["n"]
    assert first >= 2, "first use walks the directory once"
    for _ in range(5):
        store.list_event_waiters(wait_keys=[key], limit=10)
    assert calls["n"] == first, "later lookups never walk the store"

    store.save(_waiter(wait_key=key))
    assert len(store.list_event_waiters(wait_keys=[key], limit=10)) == 2
    assert calls["n"] == first, "a new waiter is indexed on save, not by re-walking"


def test_empty_and_unknown_keys_are_cheap_and_empty(tmp_path):
    store = JsonFileRunStore(tmp_path)
    assert store.list_event_waiters(wait_keys=[], limit=10) == []
    assert store.list_event_waiters(wait_keys=[""], limit=10) == []
    assert store.list_event_waiters(wait_keys=["evt:session:nobody:x"], limit=10) == []


def test_prefix_lookup_matches_the_scan(tmp_path):
    store = JsonFileRunStore(tmp_path)
    parked = {
        "evt:session:s1:alpha",
        "evt:session:s1:beta",
        "evt:session:s2:gamma",
        "evt:run:r1:alpha",
    }
    for k in parked:
        store.save(_waiter(wait_key=k))
    store.save(_other(status=RunStatus.WAITING, wait=WaitState(reason=WaitReason.USER, wait_key="evt:session:s1:nope")))

    prefix = "evt:session:s1:"
    got = {r.waiting.wait_key for r in store.list_event_waiters_by_prefix(prefix=prefix, limit=64)}
    expected = {
        r.waiting.wait_key
        for r in store.list_runs(status=RunStatus.WAITING, wait_reason=WaitReason.EVENT, limit=10_000)
        if str(getattr(r.waiting, "wait_key", "")).startswith(prefix)
    }
    assert got == expected == {"evt:session:s1:alpha", "evt:session:s1:beta"}
