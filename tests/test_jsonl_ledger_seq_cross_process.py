"""Two processes appending to one run's JSONL ledger never take the same seq (ADR-0010).

The seq is the durable cursor every reader resumes from; the JSONL store allocates it under an
advisory file lock spanning the head read and the write, so a split API process (or a CLI next
to a gateway) appending to the same run keeps the numbering dense and unique.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import pytest

from abstractruntime.core.models import StepRecord, StepStatus
from abstractruntime.storage.json_files import JsonlLedgerStore

RUN = "cross-process-run"


def _writer(base: str, tag: str, n: int, go) -> None:
    store = JsonlLedgerStore(base)
    go.wait(30)  # every process starts appending at the same moment
    for i in range(n):
        store.append(StepRecord(run_id=RUN, step_id=f"{tag}-{i}", node_id="n", status=StepStatus.COMPLETED))


@pytest.mark.skipif(sys.platform.startswith("win"), reason="advisory file locks are POSIX")
def test_processes_appending_to_one_run_get_dense_unique_seqs(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    go = ctx.Event()
    procs = [ctx.Process(target=_writer, args=(str(tmp_path), f"p{k}", 200, go)) for k in range(4)]
    for p in procs:
        p.start()
    time.sleep(1.5)  # let the spawned interpreters import and reach the barrier
    go.set()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    records = JsonlLedgerStore(tmp_path).list(RUN)
    assert len(records) == 800
    assert [r["seq"] for r in records] == list(range(1, 801))
    assert len({r["step_id"] for r in records}) == 800


@pytest.mark.skipif(sys.platform.startswith("win"), reason="advisory file locks are POSIX")
def test_a_second_store_cannot_append_between_the_head_read_and_the_write(tmp_path: Path) -> None:
    """Deterministic race: store A reads the head, then stalls; store B (another process's view
    of the same file: its own in-process lock) appends meanwhile. B must wait for A's line."""
    import threading

    a, b = JsonlLedgerStore(tmp_path), JsonlLedgerStore(tmp_path)
    a.append(StepRecord(run_id=RUN, step_id="first", node_id="n", status=StepStatus.COMPLETED))
    head_read = threading.Event()
    real = a._head_seq

    def slow_head(run_id, p):  # noqa: ANN001
        seq = real(run_id, p)
        head_read.set()
        time.sleep(0.4)
        return seq

    a._head_seq = slow_head  # type: ignore[method-assign]
    ta = threading.Thread(target=a.append, args=(StepRecord(run_id=RUN, step_id="a", node_id="n", status=StepStatus.COMPLETED),))
    ta.start()
    assert head_read.wait(5)
    b.append(StepRecord(run_id=RUN, step_id="b", node_id="n", status=StepStatus.COMPLETED))
    ta.join(5)
    records = JsonlLedgerStore(tmp_path).list(RUN)
    assert [(r["step_id"], r["seq"]) for r in records] == [("first", 1), ("a", 2), ("b", 3)]


def test_a_reader_hint_never_survives_another_store_recreating_the_ledger(tmp_path: Path) -> None:
    """Another store object (another process) deletes and recreates a run's ledger: a reader's
    remembered byte offset belongs to the old file and must not be used for the new one."""
    reader, writer = JsonlLedgerStore(tmp_path), JsonlLedgerStore(tmp_path)
    for i in range(3):
        writer.append(StepRecord(run_id=RUN, step_id=f"old-{'x' * 300}-{i}", node_id="n", status=StepStatus.COMPLETED))
    got, last = reader.list_after(run_id=RUN, after=0)
    assert len(got) == 3 and last == 3
    writer.delete(RUN)
    for i in range(6):
        writer.append(StepRecord(run_id=RUN, step_id=f"n{i}", node_id="n", status=StepStatus.COMPLETED))
    got, last = reader.list_after(run_id=RUN, after=3)
    assert [(r["seq"], r["step_id"]) for r in got] == [(4, "n3"), (5, "n4"), (6, "n5")]
    # The reader's own appends continue the recreated file's numbering.
    assert reader.append(StepRecord(run_id=RUN, step_id="mine", node_id="n", status=StepStatus.COMPLETED)) == 7
