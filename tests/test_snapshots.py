from pathlib import Path

from abstractruntime.core.models import RunState
from abstractruntime.storage.snapshots import InMemorySnapshotStore, JsonSnapshotStore, Snapshot


def test_in_memory_snapshot_store_roundtrip_and_search():
    store = InMemorySnapshotStore()

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={"x": 1})
    s1 = Snapshot.from_run(run=run, name="invoice:1", description="first", tags=["prod", "invoice"])
    s2 = Snapshot.from_run(run=run, name="notes", description="misc", tags=["dev"])

    store.save(s1)
    store.save(s2)

    loaded = store.load(s1.snapshot_id)
    assert loaded is not None
    assert loaded.name == "invoice:1"
    assert loaded.run_state["vars"]["x"] == 1

    by_tag = store.list(tag="invoice")
    assert len(by_tag) == 1
    assert by_tag[0].snapshot_id == s1.snapshot_id

    by_query = store.list(query="first")
    assert len(by_query) == 1
    assert by_query[0].snapshot_id == s1.snapshot_id


def test_json_snapshot_store_roundtrip(tmp_path: Path):
    store = JsonSnapshotStore(tmp_path)

    run = RunState.new(workflow_id="wf", entry_node="n1", vars={"x": 1})
    snap = Snapshot.from_run(run=run, name="invoice", description="hello", tags=["prod"])

    store.save(snap)
    loaded = store.load(snap.snapshot_id)

    assert loaded is not None
    assert loaded.snapshot_id == snap.snapshot_id
    assert loaded.run_state["workflow_id"] == "wf"

    listed = store.list(tag="prod")
    assert len(listed) == 1
    assert listed[0].snapshot_id == snap.snapshot_id

