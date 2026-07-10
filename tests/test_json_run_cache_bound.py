"""The JSON run store's cache is LRU-bounded (2026-07-11 incident review:
the unbounded cache retained every RunState a full directory scan ever
loaded — ~1.5GB RSS on a 3k-run dir, paid permanently by long-lived
serving processes). Pins: the bound holds under scans, aliasing survives
for cached entries (the ownership contract), and evicted entries re-load
correctly from disk.
"""

from __future__ import annotations

from pathlib import Path

from abstractruntime.core.models import RunState
from abstractruntime.storage.json_files import JsonFileRunStore


def _run(i: int) -> RunState:
    r = RunState.new(workflow_id="wf", entry_node="n1")
    r.vars = {"i": i, "payload": "x" * 200}
    return r


def test_cache_never_exceeds_bound_under_full_scans(tmp_path: Path) -> None:
    store = JsonFileRunStore(tmp_path, run_cache_max=8)
    ids = []
    for i in range(30):
        r = _run(i)
        store.save(r)
        ids.append(r.run_id)
    # A full listing loads every file; the cache must stay bounded.
    listed = store.list_runs(limit=1000)
    assert len(listed) == 30
    assert len(store._run_cache) <= 8

    # Evicted entries re-load correctly from disk (fresh object, same state).
    oldest = ids[0]
    loaded = store.load(oldest)
    assert loaded is not None
    assert loaded.vars["i"] == 0


def test_cached_load_still_aliases_saved_object(tmp_path: Path) -> None:
    """The ownership contract's aliasing survives for IN-CACHE entries:
    save() then load() returns the same object (mtime-validated)."""
    store = JsonFileRunStore(tmp_path, run_cache_max=8)
    r = _run(1)
    store.save(r)
    assert store.load(r.run_id) is r
