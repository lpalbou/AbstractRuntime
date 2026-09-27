"""A root start that names a session stays O(session), not O(store) (review 45 H1).

`Runtime.start` asks `session_attribution` on every root start with a session
id (the discussion anchor). On the gateway's default JSON store that was a
whole-directory scan: 277 ms median at 20k runs. The attribution now reads the
store's session index (`session_kinds`); the budget is <= 2 ms median over a
plain start at 20,000 runs, on both stores.
"""

from __future__ import annotations

import json
import statistics
import time

import pytest

from abstractruntime import Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.storage.in_memory import InMemoryLedgerStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore

N_RUNS = 20_000
WF = WorkflowSpec(workflow_id="wf", entry_node="n", nodes={"n": lambda r, c: StepPlan(node_id="n", complete_output={})})


def _doc(i: int) -> dict:
    ts = f"2026-09-01T00:00:{i % 60:02d}+00:00"
    return {"run_id": f"r{i}", "workflow_id": "wf", "status": "completed", "current_node": "n",
            "vars": {"prompt": "p"}, "session_id": f"s{i % 5000}", "parent_run_id": None,
            "created_at": ts, "updated_at": ts}


def _json_store(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for i in range(N_RUNS):
        (runs / f"run_r{i}.json").write_text(json.dumps(_doc(i)))
    return JsonFileRunStore(runs)


def _sqlite_store(tmp_path):
    store = SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))
    conn = store._db.connection()
    cols = SqliteRunStore._RUN_COLUMNS
    rows = []
    for i in range(N_RUNS):
        d = _doc(i)
        row = SqliteRunStore._run_row(RunState(run_id=d["run_id"], workflow_id="wf", status=RunStatus.COMPLETED,
                                               current_node="n", vars=d["vars"], session_id=d["session_id"],
                                               created_at=d["created_at"], updated_at=d["updated_at"]))
        rows.append(tuple(row[c] for c in cols))
    with conn:
        conn.executemany(f"INSERT INTO runs ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)});", rows)
    return store


def _median_ms(fn, n: int = 25) -> float:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


@pytest.mark.parametrize("make", [_json_store, _sqlite_store], ids=["json", "sqlite"])
def test_a_session_start_costs_at_most_2ms_more_than_a_plain_start(make, tmp_path) -> None:
    store = make(tmp_path)
    rt = Runtime(run_store=store, ledger_store=InMemoryLedgerStore())
    rt.start(workflow=WF, vars={}, session_id="s7")  # warm-up: builds the session index once
    rt.start(workflow=WF, vars={})
    baseline = _median_ms(lambda: rt.start(workflow=WF, vars={}))
    in_session = _median_ms(lambda: rt.start(workflow=WF, vars={}, session_id="s7"))
    assert in_session - baseline <= 2.0, f"session start {in_session:.2f} ms vs plain {baseline:.2f} ms"


def test_json_session_index_follows_saves_creates_and_deletes(tmp_path) -> None:
    store = JsonFileRunStore(tmp_path / "runs")
    store.save(RunState(run_id="a", workflow_id="wf", status=RunStatus.COMPLETED, current_node="n", vars={}, session_id="s"))
    assert store.session_kinds("s") == frozenset({"chat"})  # builds the index
    store.create_if_absent(RunState.new(workflow_id="wf", entry_node="n", session_id="s", run_id="b",
                                        vars={"_meta": {"discussion": {"discussion_root_run_id": "b"}}}))
    assert store.session_kinds("s") == frozenset({"chat", "discussion"})
    assert {r["run_id"] for r in store.list_run_index(session_id="s")} == {"a", "b"}
    store.delete("b")
    assert store.session_kinds("s") == frozenset({"chat"})
    assert [r["run_id"] for r in store.list_run_index(session_id="s")] == ["a"]
    # A fresh store (restart) rebuilds the same view from disk.
    assert JsonFileRunStore(tmp_path / "runs").session_kinds("s") == frozenset({"chat"})
    assert store.session_kinds("nobody") == frozenset()


def test_a_discussion_created_through_another_store_object_is_seen(tmp_path) -> None:
    """Two store objects on one folder (standby / multi-worker gateway): A has
    built its session index; the discussion root is created through B; a start
    through A is restamped read-only and the history through A sees it."""
    from abstractruntime import session_chat_messages

    ws = tmp_path / "ws"
    ws.mkdir()
    a = JsonFileRunStore(tmp_path / "runs")
    b = JsonFileRunStore(tmp_path / "runs")
    a.save(RunState(run_id="c0", workflow_id="wf", status=RunStatus.COMPLETED, current_node="n", vars={}, session_id="other"))
    a.warm_session_index()
    assert a.session_kinds("disc-s") == frozenset()

    disc = {"automation_id": "auto-1", "occurrence_index": 1, "discussion_root_run_id": "root"}
    seed = [{"role": "user", "content": "tick 1"}, {"role": "assistant", "content": "mem 41%"}]
    b.create_if_absent(RunState(run_id="root", workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                                session_id="disc-s", output={"answer": "because"},
                                vars={"prompt": "why?", "context": {"messages": []}, "workspace_root": str(ws),
                                      "workspace_read_only": True, "_meta": {"discussion": {**disc, "seed_messages": seed}}}))

    rt = Runtime(run_store=a, ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=WF, vars={"prompt": "and now?", "workspace_read_only": False}, session_id="disc-s")
    vars_ = a.load(rid).vars
    assert vars_["workspace_read_only"] is True and vars_["workspace_root"] == str(ws)
    assert vars_["_meta"]["discussion"]["discussion_root_run_id"] == "root"
    history = session_chat_messages(run_store=a, session_id="disc-s", strict=True)
    assert [m["content"] for m in history][:4] == ["tick 1", "mem 41%", "why?", "because"]
    # Deletes through B reach A's children/session indexes too.
    b.delete("root")
    b.delete(rid)
    assert a.list_run_index(session_id="disc-s") == []
    assert a.session_kinds("disc-s") == frozenset()


def test_the_journal_check_costs_under_a_tenth_of_a_millisecond(tmp_path) -> None:
    store = _json_store(tmp_path)
    store.warm_session_index()
    store.session_kinds("s7")
    per_lookup = _median_ms(lambda: store.session_kinds("s7"), n=200)
    assert per_lookup < 0.1, f"session_kinds {per_lookup:.3f} ms per lookup"
