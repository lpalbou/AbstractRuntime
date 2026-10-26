"""`list_session_candidates`: a session's artifacts through the catalog indexes, never the
whole catalog (2026-10-26: the gateway's session artifact list read all 106k artifacts on its
event loop, 2 s per call, and the watchdog restarted the process twice)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from abstractruntime import FileArtifactStore, InMemoryArtifactStore


def _seed(store) -> dict:
    ids = {}
    ids["upload"] = store.store(b"u", content_type="text/plain", run_id="session_memory_s1").artifact_id
    ids["tagged"] = store.store(b"t", content_type="text/plain", run_id="run-a", tags={"session_id": "s1"}).artifact_id
    ids["shared"] = store.store(b"s", content_type="text/plain", run_id="run-b", tags={"shared": "user"}).artifact_id
    ids["other_session"] = store.store(b"o", content_type="text/plain", run_id="run-c", tags={"session_id": "s2"}).artifact_id
    ids["unrelated"] = store.store(b"x", content_type="text/plain", run_id="run-d").artifact_id
    return ids


@pytest.mark.parametrize("kind", ["file", "memory"])
def test_candidates_are_owner_rows_session_tag_and_shared_tag(tmp_path: Path, kind: str) -> None:
    store = FileArtifactStore(tmp_path) if kind == "file" else InMemoryArtifactStore()
    ids = _seed(store)
    got = {m.artifact_id for m in store.list_session_candidates("s1", owner_run_id="session_memory_s1")}
    assert got == {ids["upload"], ids["tagged"], ids["shared"]}
    # Without an owner run: the tags only.
    got = {m.artifact_id for m in store.list_session_candidates("s1")}
    assert got == {ids["tagged"], ids["shared"]}


def test_file_store_answers_from_indexes_never_a_catalog_scan(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    _seed(store)
    calls = []
    store.list_all = lambda **kw: calls.append(kw) or []  # type: ignore[method-assign]
    store.list_session_candidates("s1", owner_run_id="session_memory_s1")
    assert calls == []
    conn = sqlite3.connect(str(tmp_path / "artifacts" / "artifact_catalog.sqlite3"))
    from abstractruntime.storage.artifacts import _TAG_SESSION_SQL, _TAG_SHARED_SQL

    for where in (f"{_TAG_SESSION_SQL} = 's1'", f"{_TAG_SHARED_SQL} IS NOT NULL", "run_id = 'x'"):
        plan = " ".join(str(r[-1]) for r in conn.execute(f"EXPLAIN QUERY PLAN SELECT raw_json FROM artifact_catalog WHERE {where}"))
        assert "USING INDEX" in plan, (where, plan)


def _index_names(tmp_path: Path) -> set:
    conn = sqlite3.connect(str(tmp_path / "artifacts" / "artifact_catalog.sqlite3"))
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def test_indexes_are_built_by_the_first_lookup_never_by_the_open(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    _seed(store)
    tag_indexes = {"idx_artifact_catalog_tag_session", "idx_artifact_catalog_tag_shared"}
    # Opening (a gateway plane's first request may open it ON the event loop) builds nothing.
    assert not (tag_indexes & _index_names(tmp_path))
    store.list_session_candidates("s1")
    assert tag_indexes <= _index_names(tmp_path)
    # An existing catalog without them (every catalog before this change) gets them too.
    conn = sqlite3.connect(str(tmp_path / "artifacts" / "artifact_catalog.sqlite3"))
    conn.execute("DROP INDEX idx_artifact_catalog_tag_session")
    conn.execute("DROP INDEX idx_artifact_catalog_tag_shared")
    conn.commit()
    reopened = FileArtifactStore(tmp_path)
    assert not (tag_indexes & _index_names(tmp_path))
    reopened.list_session_candidates("s1")
    assert tag_indexes <= _index_names(tmp_path)


def test_a_catalog_error_raises_and_never_reads_every_artifact(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    _seed(store)
    store.list_session_candidates("s1")
    reads = []
    store._scan_metadata_files = lambda: reads.append(1) or []  # type: ignore[method-assign]
    store.list_all = lambda **kw: reads.append(2) or []  # type: ignore[method-assign]

    def broken():
        raise sqlite3.OperationalError("database disk image is malformed")

    store._connect_catalog = broken  # type: ignore[method-assign]
    with pytest.raises(sqlite3.OperationalError):
        store.list_session_candidates("s1", owner_run_id="session_memory_s1")
    assert reads == []


def test_benchmark_100k_catalog_session_lookup_under_50ms(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    db = tmp_path / "artifacts" / "artifact_catalog.sqlite3"
    rows = []
    for i in range(100_000):
        sid = f"s{i % 2000}"
        raw = {
            "artifact_id": f"a{i:06d}", "blob_id": None, "content_type": "text/plain", "size_bytes": 1,
            "created_at": f"2026-10-{1 + i % 25:02d}T00:00:00", "run_id": f"run-{i % 9000}",
            "tags": {"session_id": sid, "filename": f"f{i}.txt"}, "metadata": {"note": "y" * 200},
            "descriptor": {}, "access": {},
        }
        rows.append((raw["artifact_id"], raw["run_id"], raw["created_at"], json.dumps(raw["tags"]), json.dumps(raw)))
    conn = sqlite3.connect(str(db))
    conn.executemany(
        "INSERT INTO artifact_catalog (artifact_id, run_id, created_at, content_type, size_bytes, tags_json, metadata_json, "
        "descriptor_json, access_json, raw_json) VALUES (?, ?, ?, 'text/plain', 1, ?, '{}', '{}', '{}', ?)",
        rows,
    )
    conn.commit()
    store.list_session_candidates("s7")  # warm the page cache
    t = time.perf_counter()
    got = store.list_session_candidates("s7", owner_run_id="session_memory_s7")
    elapsed = time.perf_counter() - t
    assert len(got) == 50
    assert elapsed < 0.05, f"{elapsed * 1000:.1f} ms for one session of a 100k catalog"
