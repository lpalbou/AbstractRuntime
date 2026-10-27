"""The JSON-file ledger's row address is a seek, not a parse of the whole file (ADR 0011; backlog 0856 performance).

`JsonlLedgerStore` keeps a per-run line-offset index (`_JsonlRowIndex`), extended by its own appends (write-through)
and by reading only the bytes another writer appended since. Every case compares against `list()`, the definition
of the JSON-file row address (seq = 1-based position of the record in the order `list()` returns).

Mutants this file kills (each checked by hand when it was written):
- the index is not extended when another writer appends (cached by size, not re-read) -> the two-writer case fails;
- the index ignores a recreated file (inode / shrink check removed) -> the delete-and-recreate case fails;
- a concatenated line is indexed as one record -> the recovery case fails;
- `find_started_record` maps a byte hit to the wrong row (off-by-one in the window bisect) -> the started cases fail;
- `append` counts from the index while a torn trailing line is pending -> the torn-line case fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from abstractruntime.core.models import StepRecord, StepStatus
from abstractruntime.storage.base import LedgerStore
from abstractruntime.storage.json_files import JsonlLedgerStore

pytestmark = pytest.mark.basic

RID = "run-index"


def _step(i: int, status: StepStatus = StepStatus.STARTED, sid: str = "") -> StepRecord:
    return StepRecord(run_id=RID, step_id=sid or f"step-{i}", node_id=f"n{i}", status=status,
                      effect={"type": "tool_calls", "payload": {"i": i, "pad": "x" * (i % 7) * 50}})


def _assert_matches_list(store: JsonlLedgerStore) -> None:
    rows = store.list(RID)
    for s, row in enumerate(rows, start=1):
        assert row["seq"] == s
        assert store.get_record_unresolved(RID, s) == row, s
    assert store.get_record_unresolved(RID, len(rows) + 1) is None
    for row in rows:
        if row.get("status") == "started":
            expected = LedgerStore.find_started_record(store, RID, row["step_id"])
            assert store.find_started_record(RID, row["step_id"]) == expected
            assert store.find_started_record(RID, row["step_id"], before_seq=row["seq"] + 1) == expected
            assert store.find_started_record(RID, row["step_id"], before_seq=row["seq"]) is None


def test_point_reads_match_list_for_every_row(tmp_path: Path) -> None:
    store = JsonlLedgerStore(tmp_path)
    store._STARTED_SCAN_WINDOW_BYTES = 128  # exercise the doubling windows on a small file
    for i in range(1, 41):
        sid = f"step-{(i + 1) // 2}"
        store.append(_step(i, StepStatus.STARTED if i % 2 else StepStatus.COMPLETED, sid))
    _assert_matches_list(store)
    assert store.find_started_record(RID, "missing") is None


def test_a_second_writer_appending_between_reads(tmp_path: Path) -> None:
    a, b = JsonlLedgerStore(tmp_path), JsonlLedgerStore(tmp_path)
    for i in range(1, 4):
        assert a.append(_step(i)) == i
    assert a.get_record_unresolved(RID, 3)["node_id"] == "n3"  # A's index is current
    assert b.append(_step(4)) == 4 and b.append(_step(5)) == 5  # B appends behind A's back
    assert a.get_record_unresolved(RID, 5)["node_id"] == "n5"
    assert a.find_started_record(RID, "step-4")["seq"] == 4
    assert a.append(_step(6)) == 6
    assert b.get_record_unresolved(RID, 6)["node_id"] == "n6"
    _assert_matches_list(a)
    _assert_matches_list(b)


def test_a_deleted_and_recreated_ledger_is_reindexed(tmp_path: Path) -> None:
    a, b = JsonlLedgerStore(tmp_path), JsonlLedgerStore(tmp_path)
    for i in range(1, 6):
        a.append(_step(i))
    assert a.get_record_unresolved(RID, 5)["node_id"] == "n5"
    b.delete(RID)
    assert a.get_record_unresolved(RID, 1) is None
    b.append(_step(100))
    assert a.get_record_unresolved(RID, 1)["node_id"] == "n100"
    assert a.get_record_unresolved(RID, 2) is None
    _assert_matches_list(a)


def _line(node_id: str, length: int) -> str:
    base = {"run_id": RID, "step_id": f"s-{node_id}", "node_id": node_id, "status": "started", "pad": ""}
    base["pad"] = "x" * (length - 1 - len(json.dumps(base)))
    line = json.dumps(base) + "\n"
    assert len(line) == length
    return line


def test_a_ledger_recreated_behind_an_instance_is_reindexed(tmp_path: Path) -> None:
    """Rows of the recreated file start at byte offsets the old index also holds: only the file-identity check
    (inode, or a shrink) tells the index it is stale."""
    path = tmp_path / f"ledger_{RID}.jsonl"
    path.write_text("".join(_line(f"a{i}", 200) for i in range(1, 4)), encoding="utf-8")
    a = JsonlLedgerStore(tmp_path)
    assert a.get_record_unresolved(RID, 2)["node_id"] == "a2"  # offsets 0, 200, 400
    keep = tmp_path / "old-inode"
    path.rename(keep)  # keeps the old inode alive, so the new file cannot reuse it
    path.write_text("".join(_line(f"b{i}", 100) for i in range(1, 9)), encoding="utf-8")  # starts 0, 100, 200, ...
    assert a.get_record_unresolved(RID, 2)["node_id"] == "b2"
    assert a.get_record_unresolved(RID, 3)["node_id"] == "b3"
    _assert_matches_list(a)


def test_concatenated_blank_and_corrupt_lines_index_like_list(tmp_path: Path) -> None:
    path = tmp_path / f"ledger_{RID}.jsonl"
    path.write_text("\n{torn\n", encoding="utf-8")  # the file STARTS with a blank and a corrupt line
    store = JsonlLedgerStore(tmp_path)
    store.append(_step(1))
    two = [json.loads(json.dumps({"run_id": RID, "step_id": f"cat-{k}", "node_id": f"c{k}", "status": "started"}))
           for k in (1, 2)]
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(two[0]) + json.dumps(two[1]) + "\n")  # one line, two records (crash recovery)
        f.write("\n")                                           # blank
        f.write("{not json\n")                                  # corrupt: holds no record
        f.write("[1, 2]\n")                                     # not an object: holds no record
    assert store.append(_step(5)) == 4
    rows = store.list(RID)
    assert [r["node_id"] for r in rows] == ["n1", "c1", "c2", "n5"]
    _assert_matches_list(store)
    _assert_matches_list(JsonlLedgerStore(tmp_path))  # a cold instance indexes the same way


def test_a_torn_trailing_line_is_read_like_list(tmp_path: Path) -> None:
    store = JsonlLedgerStore(tmp_path)
    store.append(_step(1))
    assert store.get_record_unresolved(RID, 1)["node_id"] == "n1"
    path = tmp_path / f"ledger_{RID}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": RID, "step_id": "torn", "node_id": "t", "status": "started"}))  # no newline
    assert store.get_record_unresolved(RID, 2)["node_id"] == "t"  # list() reads it; so does the point read
    other = JsonlLedgerStore(tmp_path)
    assert other.append(_step(3)) == 3  # the torn line is ended first, so both stay records
    names: List[str] = [r["node_id"] for r in other.list(RID)]
    assert names == ["n1", "t", "n3"]
    _assert_matches_list(other)
    _assert_matches_list(store)


def test_a_crash_torn_half_line_never_swallows_the_next_append(tmp_path: Path) -> None:
    """Review 3a R9: a crashed writer leaves half a record without a newline; the next append ends that line first,
    so its record is listed and its seq resolves. Mutant: write without the leading newline -> the record is glued
    onto the fragment, disappears from list(), and get_record(seq) answers None or another row."""
    store = JsonlLedgerStore(tmp_path)
    for i in range(1, 4):
        store.append(_step(i))
    with (tmp_path / f"ledger_{RID}.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"run_id": "run-index", "step_id": "half')  # crash mid-write
    for writer in (store, JsonlLedgerStore(tmp_path)):
        seq = writer.append(_step(99 if writer is store else 100))
        got = writer.get_record(RID, seq)
        assert got is not None and got["node_id"] == ("n99" if writer is store else "n100"), (seq, got)
    assert [r["node_id"] for r in store.list(RID)] == ["n1", "n2", "n3", "n99", "n100"]
    _assert_matches_list(store)



def _persisting(tmp_path: Path) -> JsonlLedgerStore:
    store = JsonlLedgerStore(tmp_path)
    store._ROW_INDEX_PERSIST_BYTES = 1  # persist on every indexed append (the default waits for 256 KB / 1/8 file)
    store._ROW_INDEX_PERSIST_FRACTION = 1 << 30
    return store


def test_a_fresh_process_loads_the_persisted_index_instead_of_parsing(tmp_path: Path) -> None:
    """Architect C3: the row index is persisted beside the ledger (`ledger_<run>.idx`) so a fresh process (or an
    evicted run) does not re-parse the whole file. Mutants: never load the file -> `_row_index_load` answers None;
    load without the inode/size checks -> the recreated-ledger case reads the old offsets."""
    import os

    writer = _persisting(tmp_path)
    for i in range(1, 30):
        writer.append(_step(i))
    idx_file = tmp_path / f"ledger_{RID}.idx"
    assert idx_file.exists()
    fresh = JsonlLedgerStore(tmp_path)
    path = tmp_path / f"ledger_{RID}.jsonl"
    loaded = fresh._row_index_load(RID, path, path.stat())
    assert loaded is not None and list(loaded.offsets) == list(writer._row_index[RID].offsets)
    _assert_matches_list(fresh)

    # The ledger grew behind the persisted index: the tail is indexed on top of it.
    JsonlLedgerStore(tmp_path).append(_step(30))  # a writer without write-through of the persisted index
    _assert_matches_list(JsonlLedgerStore(tmp_path))

    # A garbage index is ignored and rebuilt.
    idx_file.write_bytes(b"garbage")
    assert JsonlLedgerStore(tmp_path)._row_index_load(RID, path, path.stat()) is None
    _assert_matches_list(JsonlLedgerStore(tmp_path))

    # An index of a previous file (the ledger was recreated: other inode, other rows) is never used, even when its
    # indexed size ends on a line break of the new file and its offsets are line starts there.
    path.write_text("".join(_line(f"a{i}", 200) for i in range(1, 4)), encoding="utf-8")  # offsets 0, 200, 400
    idx_file.unlink()
    _persisting(tmp_path).get_record_unresolved(RID, 1)  # builds + persists for this file (size 600)
    keep = tmp_path / "old.jsonl"
    path.rename(keep)  # keeps the old inode alive
    path.write_text("".join(_line(f"z{i}", 100) for i in range(1, 9)), encoding="utf-8")  # offsets 0, 100, 200, ...
    assert JsonlLedgerStore(tmp_path)._row_index_load(RID, path, path.stat()) is None
    assert JsonlLedgerStore(tmp_path).get_record_unresolved(RID, 2)["node_id"] == "z2"
    _assert_matches_list(JsonlLedgerStore(tmp_path))
    assert os.path.exists(idx_file)
    JsonlLedgerStore(tmp_path).delete(RID)
    assert not idx_file.exists()  # delete removes the index with the ledger


def test_a_rewritten_file_reads_by_persisted_seq_never_by_position(tmp_path: Path) -> None:
    """The row index finds a record by POSITION; the line carries its persisted `seq` (ADR 0010), which decides.
    Drop line 2 of a 4-row ledger: seq 3 is now at position 2 and is still found; seq 2 is gone (None, never the
    row that moved into its place). Mutant: skip the persisted-seq check -> get_record(2) answers the seq-3 row."""
    store = JsonlLedgerStore(tmp_path)
    for i in range(1, 5):
        store.append(_step(i))
    path = tmp_path / f"ledger_{RID}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(lines[0] + "".join(lines[2:]), encoding="utf-8")
    for reader in (store, JsonlLedgerStore(tmp_path)):
        assert reader.get_record_unresolved(RID, 2) is None
        assert reader.get_record_unresolved(RID, 3)["node_id"] == "n3"
        assert reader.get_record_unresolved(RID, 4)["node_id"] == "n4"
        assert reader.find_started_record(RID, "step-3")["seq"] == 3
        assert [r["seq"] for r in reader.list(RID)] == [1, 3, 4]
