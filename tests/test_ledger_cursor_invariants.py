"""The durable ledger cursor at the store level (gateway backlog 0082 I1-I4, ADR-0009).

- I1 replay completeness: `list(run_id)` returns every appended record, in append order, with
  `seq` 1..N, and `append` returns that seq;
- I2 streaming ⊆ replay: `list_after(after=k)` == `list()[k:]` for every k, paging with `limit`
  reads the same records, and a subscriber (ObservableLedgerStore) gets the same seq as `list()`;
- I3 cursor stability: a seq names the same record after re-opening the store and after a
  JSONL -> SQLite migration (`SqliteLedgerStore.import_records`), and appends continue after it;
- I4 recovery: a JSONL line holding two concatenated objects and lines written before `seq`
  existed are recovered and numbered by position; appends continue the numbering; `list_after`
  reads them the same.
Plus: 8 threads appending to one JSONL run get dense unique seqs 1..N.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from abstractruntime.core.models import StepRecord, StepStatus
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore
from abstractruntime.storage.json_files import JsonlLedgerStore
from abstractruntime.storage.ledger_chain import HashChainedLedgerStore
from abstractruntime.storage.observable import ObservableLedgerStore
from abstractruntime.storage.offloading import OffloadingLedgerStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

RUN_A = "11111111-aaaa-4000-8000-000000000001"
RUN_B = "22222222-bbbb-4000-8000-000000000002"


def _rec(run_id: str, i: int, *, big: bool = False) -> StepRecord:
    status = (StepStatus.STARTED, StepStatus.COMPLETED, StepStatus.WAITING)[i % 3]
    result = None if status == StepStatus.STARTED else {"i": i, "text": ("x" * 300_000) if big else f"r{i}"}
    return StepRecord(run_id=run_id, step_id=f"{run_id[:4]}-step-{i:03d}", node_id=f"n{i % 4}", status=status,
                      effect={"type": "llm_call", "payload": {"prompt": f"p{i}"}}, result=result,
                      idempotency_key=f"idem-{run_id[:4]}-{i}", started_at=f"2026-10-27T10:00:{i % 60:02d}+00:00")


def _make(kind: str, tmp_path: Path):
    """(store, reopen) where reopen() returns a fresh handle on the same persisted ledger, or None."""
    if kind == "jsonl":
        return JsonlLedgerStore(tmp_path / "led"), lambda: JsonlLedgerStore(tmp_path / "led")
    if kind == "sqlite":
        return SqliteLedgerStore(SqliteDatabase(tmp_path / "led.sqlite")), lambda: SqliteLedgerStore(SqliteDatabase(tmp_path / "led.sqlite"))
    if kind == "memory":
        return InMemoryLedgerStore(), None
    if kind == "observable(jsonl)":
        return ObservableLedgerStore(JsonlLedgerStore(tmp_path / "led")), lambda: JsonlLedgerStore(tmp_path / "led")
    if kind == "observable(sqlite)":
        return ObservableLedgerStore(SqliteLedgerStore(SqliteDatabase(tmp_path / "led.sqlite"))), None
    if kind == "chained(jsonl)":
        return HashChainedLedgerStore(JsonlLedgerStore(tmp_path / "led")), lambda: HashChainedLedgerStore(JsonlLedgerStore(tmp_path / "led"))
    if kind == "chained(sqlite)":
        return HashChainedLedgerStore(SqliteLedgerStore(SqliteDatabase(tmp_path / "led.sqlite"))), None
    if kind == "offloading(jsonl)":
        art = InMemoryArtifactStore()
        return (OffloadingLedgerStore(JsonlLedgerStore(tmp_path / "led"), artifact_store=art, max_inline_bytes=4096),
                lambda: OffloadingLedgerStore(JsonlLedgerStore(tmp_path / "led"), artifact_store=art, max_inline_bytes=4096))
    raise AssertionError(kind)


KINDS = ["jsonl", "sqlite", "memory", "observable(jsonl)", "observable(sqlite)", "chained(jsonl)", "chained(sqlite)",
         "offloading(jsonl)"]


def _fill(store, n: int = 25):
    """Append n records to each of two runs, interleaved; return {run: [returned seqs]}."""
    seqs = {RUN_A: [], RUN_B: []}
    for i in range(n):
        for run_id in (RUN_A, RUN_B) if i % 2 else (RUN_B, RUN_A):
            seqs[run_id].append(store.append(_rec(run_id, i, big=(i == 7))))
    return seqs


def _key(rec):
    return rec["step_id"], rec["seq"]


def _check_list_after(store, run_id, full):
    n = len(full)
    for k in range(n + 2):
        got, cursor = store.list_after(run_id=run_id, after=k)
        assert got == full[k:], (run_id, k)
        assert cursor == (full[-1]["seq"] if got else k)
    for limit in (1, 3, 7, 1000):
        paged, after = [], 0
        while True:
            page, after2 = store.list_after(run_id=run_id, after=after, limit=limit)
            assert len(page) <= limit
            if not page:
                assert after2 == after
                break
            paged.extend(page)
            after = after2
        assert paged == full
    # A reader that goes back (a reconnect from an older cursor) reads the same records.
    for k in (n, n // 2, 1, n - 1, 0, n // 3):
        assert store.list_after(run_id=run_id, after=k)[0] == full[k:]


# --------------------------------------------------------------------------- I1, I2


@pytest.mark.parametrize("kind", KINDS)
def test_i1_list_returns_every_record_in_order_with_dense_seq(kind, tmp_path):
    store, _ = _make(kind, tmp_path)
    seqs = _fill(store)
    for run_id in (RUN_A, RUN_B):
        records = store.list(run_id)
        assert [r["seq"] for r in records] == list(range(1, 26))
        assert seqs[run_id] == list(range(1, 26))  # append returns the record's seq
        assert [r["step_id"] for r in records] == [f"{run_id[:4]}-step-{i:03d}" for i in range(25)]


@pytest.mark.parametrize("kind", KINDS)
def test_i2_list_after_equals_list_suffix_and_pages(kind, tmp_path):
    store, _ = _make(kind, tmp_path)
    _fill(store)
    for run_id in (RUN_A, RUN_B):
        _check_list_after(store, run_id, store.list(run_id))
    assert store.list_after(run_id="no-such-run", after=0) == ([], 0)


@pytest.mark.parametrize("kind", ["observable(jsonl)", "observable(sqlite)"])
def test_i2_subscribers_get_the_seq_list_returns(kind, tmp_path):
    store, _ = _make(kind, tmp_path)
    got = []
    unsubscribe = store.subscribe(lambda rec: got.append((rec["run_id"], rec["step_id"], rec["seq"])))
    _fill(store, 10)
    unsubscribe()
    listed = [(r["run_id"], r["step_id"], r["seq"]) for run_id in (RUN_A, RUN_B) for r in store.list(run_id)]
    assert sorted(got) == sorted(listed)
    assert [x for x in got if x[0] == RUN_A] == [x for x in listed if x[0] == RUN_A]  # same order per run


# --------------------------------------------------------------------------- I3


@pytest.mark.parametrize("kind", ["jsonl", "sqlite", "observable(jsonl)", "chained(jsonl)", "offloading(jsonl)"])
def test_i3_a_seq_means_the_same_record_after_reopening(kind, tmp_path):
    store, reopen = _make(kind, tmp_path)
    _fill(store)
    before = {run_id: store.list(run_id) for run_id in (RUN_A, RUN_B)}
    again = reopen()
    for run_id, records in before.items():
        assert [_key(r) for r in again.list(run_id)] == [_key(r) for r in records]
        _check_list_after(again, run_id, again.list(run_id))
    # The re-opened handle continues the numbering (no head cache to rely on).
    assert again.append(_rec(RUN_A, 99)) == 26
    assert store.list(RUN_A)[-1]["seq"] == 26 and store.list(RUN_A)[-1]["step_id"] == f"{RUN_A[:4]}-step-099"
    # And the first handle (whose head cache is now stale) does not reuse 26.
    assert store.append(_rec(RUN_A, 100)) == 27
    assert [r["seq"] for r in again.list(RUN_A)] == list(range(1, 28))


def test_i3_a_seq_survives_a_jsonl_to_sqlite_migration(tmp_path):
    jsonl = JsonlLedgerStore(tmp_path / "led")
    _fill(jsonl)
    # A run whose file starts with pre-seq lines (numbered by position) and continues with seq.
    legacy = tmp_path / "led" / f"ledger_{RUN_B}.jsonl"
    lines = legacy.read_text(encoding="utf-8").splitlines()
    stripped = []
    for i, line in enumerate(lines):
        obj = json.loads(line)
        if i < 5:
            obj.pop("seq")
        stripped.append(json.dumps(obj))
    legacy.write_text("\n".join(stripped) + "\n", encoding="utf-8")
    jsonl = JsonlLedgerStore(tmp_path / "led")
    sqlite = SqliteLedgerStore(SqliteDatabase(tmp_path / "migrated.sqlite"))
    for run_id in (RUN_A, RUN_B):
        records = jsonl.list(run_id)
        assert [r["seq"] for r in records] == list(range(1, 26))
        assert sqlite.import_records(run_id, records) == 25
        migrated = sqlite.list(run_id)
        assert migrated == records  # same records, same seq
        for k in range(26):
            assert sqlite.list_after(run_id=run_id, after=k)[0] == jsonl.list_after(run_id=run_id, after=k)[0]
    assert sqlite.append(_rec(RUN_A, 50)) == 26  # the head was carried over
    with pytest.raises(ValueError):
        sqlite.import_records(RUN_A, jsonl.list(RUN_A))  # an import never merges


def test_i3_import_keeps_gaps_and_numbers_seqless_records_after_the_previous(tmp_path):
    sqlite = SqliteLedgerStore(SqliteDatabase(tmp_path / "m.sqlite"))
    recs = [{"run_id": RUN_A, "step_id": f"s{i}", "node_id": "n", "status": "completed"} for i in range(5)]
    recs[0]["seq"], recs[1]["seq"], recs[3]["seq"] = 1, 4, 9  # a gap kept; recs[2] -> 5, recs[4] -> 10
    sqlite.import_records(RUN_A, recs)
    assert [(r["step_id"], r["seq"]) for r in sqlite.list(RUN_A)] == [("s0", 1), ("s1", 4), ("s2", 5), ("s3", 9), ("s4", 10)]
    assert sqlite.list_after(run_id=RUN_A, after=4)[0][0]["step_id"] == "s2"
    assert sqlite.append(_rec(RUN_A, 1)) == 11


# --------------------------------------------------------------------------- I4


def _line(run_id, i, **extra):
    return json.dumps({"run_id": run_id, "step_id": f"legacy-{i}", "node_id": "n", "status": "completed",
                       "effect": None, "result": {"i": i}, **extra})


def test_i4_concatenated_and_pre_seq_lines_are_recovered_and_numbered_by_position(tmp_path, caplog):
    base = tmp_path / "led"
    base.mkdir()
    p = base / f"ledger_{RUN_A}.jsonl"
    p.write_text(
        _line(RUN_A, 0) + "\n"                       # pre-seq: 1
        + _line(RUN_A, 1) + _line(RUN_A, 2) + "\n"   # two objects on one line: 2, 3
        + "\n"                                       # a blank line is not a record
        + _line(RUN_A, 3) + "\n",                    # pre-seq: 4
        encoding="utf-8")
    store = JsonlLedgerStore(base)
    with caplog.at_level("WARNING"):
        records = store.list(RUN_A)
    assert [(r["step_id"], r["seq"]) for r in records] == [(f"legacy-{i}", i + 1) for i in range(4)]
    assert "#FALLBACK" in caplog.text  # recovery is observable
    _check_list_after(store, RUN_A, records)
    # Appends continue the numbering, from this handle and from a fresh one (no head cache).
    assert store.append(_rec(RUN_A, 4)) == 5
    assert JsonlLedgerStore(base).append(_rec(RUN_A, 5)) == 6
    records = store.list(RUN_A)
    assert [r["seq"] for r in records] == [1, 2, 3, 4, 5, 6]
    assert json.loads(p.read_text().splitlines()[-1])["seq"] == 6  # persisted on the line
    _check_list_after(store, RUN_A, records)
    _check_list_after(JsonlLedgerStore(base), RUN_A, records)


def test_i4_a_line_mixing_seq_and_pre_seq_objects_reads_the_same_everywhere(tmp_path):
    base = tmp_path / "led"
    base.mkdir()
    p = base / f"ledger_{RUN_B}.jsonl"
    p.write_text(_line(RUN_B, 0, seq=1) + "\n" + _line(RUN_B, 1, seq=2) + _line(RUN_B, 2) + "\n" + _line(RUN_B, 3) + "\n",
                 encoding="utf-8")
    store = JsonlLedgerStore(base)
    records = store.list(RUN_B)
    assert [(r["step_id"], r["seq"]) for r in records] == [("legacy-0", 1), ("legacy-1", 2), ("legacy-2", 3), ("legacy-3", 4)]
    _check_list_after(store, RUN_B, records)
    assert store.append(_rec(RUN_B, 4)) == 5


def test_i4_a_torn_line_followed_by_a_whole_record_keeps_the_whole_record(tmp_path):
    """A crash mid-write leaves a torn prefix; the next append lands on the same line."""
    base = tmp_path / "led"
    base.mkdir()
    p = base / f"ledger_{RUN_A}.jsonl"
    torn = _line(RUN_A, 1)[:25]
    p.write_text(_line(RUN_A, 0) + "\n" + torn, encoding="utf-8")
    store = JsonlLedgerStore(base)
    assert [r["step_id"] for r in store.list(RUN_A)] == ["legacy-0"]
    seq = store.append(_rec(RUN_A, 2))
    records = store.list(RUN_A)
    assert [r["step_id"] for r in records] == ["legacy-0", f"{RUN_A[:4]}-step-002"]
    assert records[-1]["seq"] == seq == 2
    _check_list_after(store, RUN_A, records)


# --------------------------------------------------------------------------- concurrency


def test_concurrent_appends_from_8_threads_get_dense_unique_seqs(tmp_path):
    store = JsonlLedgerStore(tmp_path / "led")
    per_thread = 40
    returned, errors = [], []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(t):
        try:
            barrier.wait()
            for i in range(per_thread):
                s = store.append(_rec(RUN_A, t * 1000 + i))
                with lock:
                    returned.append(s)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    n = 8 * per_thread
    assert sorted(returned) == list(range(1, n + 1))
    records = store.list(RUN_A)
    assert [r["seq"] for r in records] == list(range(1, n + 1))
    assert len({r["step_id"] for r in records}) == n
    # The persisted lines carry the same seqs (a fresh reader agrees, line by line).
    lines = [json.loads(x) for x in (tmp_path / "led" / f"ledger_{RUN_A}.jsonl").read_text().splitlines()]
    assert [x["seq"] for x in lines] == list(range(1, n + 1))
    _check_list_after(JsonlLedgerStore(tmp_path / "led"), RUN_A, records)


def test_concurrent_appends_through_two_store_handles_get_dense_unique_seqs(tmp_path):
    """Two handles on one directory (as two processes would be): the file lock, not the handle's
    in-process lock, keeps the seqs unique."""
    stores = [JsonlLedgerStore(tmp_path / "led"), JsonlLedgerStore(tmp_path / "led")]
    returned = []
    lock = threading.Lock()

    def worker(t):
        for i in range(30):
            s = stores[t % 2].append(_rec(RUN_B, t * 1000 + i))
            with lock:
                returned.append(s)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(returned) == list(range(1, 121))
    assert [r["seq"] for r in JsonlLedgerStore(tmp_path / "led").list(RUN_B)] == list(range(1, 121))
