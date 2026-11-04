"""Track record and reputation (ADR 0052 §5-§6, framework backlog 1119).

Each rule has a test that goes red when the rule is removed: the counted-run filter (turn roots, reserved
workflows, origin), the signal of each source, the task kind, the trust formula, the evidence window, the
session scope, the token fold and determinism.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from abstractruntime.core.models import RunState, RunStatus, StepRecord, StepStatus
from abstractruntime.identity.digital_identity import identity_fingerprint, identity_ref, record_default_identity, stamp_run_identity_ref
from abstractruntime.identity.track_record import (
    COMPLETION_RATE_ONLY,
    EVIDENCE_WINDOW,
    FEEDBACK_SIGNALS,
    SIGNALS,
    OutcomeEvidence,
    TrackRecordError,
    compute_reputation,
    entity_track_record,
    ledger_token_counts,
    session_track_record,
    task_kind_of,
    token_counts_of_records,
    trust_of,
)
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore

pytestmark = pytest.mark.basic

MANIFEST = {
    "entity_id": "entity:castor",
    "slug": "castor",
    "home_id": "home-1a2b3c4d",
    "spark_hash": "f" * 64,
    "name": "Castor",
}
AUTOMATION = "auto-1"


def _ts(minute: int) -> str:
    return f"2026-11-04T10:{minute:02d}:00+00:00"


def _origin(kind: str) -> Dict[str, Any]:
    ref: Dict[str, Any] = {}
    if kind == "trigger":
        ref = {"automation_id": AUTOMATION}
    if kind == "agent":
        ref = {"origin_run_id": "r-parent"}
    return {"kind": kind, "actor": "default:laurent", "client": "console", "ref": ref}


def _run(
    run_id: str,
    *,
    minute: int,
    status: RunStatus = RunStatus.COMPLETED,
    session: str = "s1",
    workflow: str = "chat-flow",
    origin: Optional[str] = "person",
    parent: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    stamp: Optional[Dict[str, Any]] = None,
    extra_vars: Optional[Dict[str, Any]] = None,
) -> RunState:
    meta_ns: Dict[str, Any] = dict(meta or {})
    if origin is not None:
        meta_ns["origin"] = _origin(origin)
    vars_obj: Dict[str, Any] = {"_meta": meta_ns, **(extra_vars or {})}
    if stamp is not None:
        vars_obj["_runtime"] = {"entity": stamp}
    return RunState(
        run_id=run_id,
        workflow_id=workflow,
        status=status,
        current_node="end",
        vars=vars_obj,
        created_at=_ts(minute),
        updated_at=_ts(minute + 1),
        session_id=session,
        parent_run_id=parent,
        actor_id="gateway",
    )


def _occurrence_meta(index: int) -> Dict[str, Any]:
    return {"occurrence": {"automation_id": AUTOMATION, "occurrence_index": index, "role": "occurrence", "session_kind": "automation"}}


def _llm(run_id: str, step: str, *, inp: int, out: int, usage_keys: str = "new") -> StepRecord:
    usage = (
        {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
        if usage_keys == "new"
        else {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}
    )
    return StepRecord(
        run_id=run_id,
        step_id=step,
        node_id="llm",
        status=StepStatus.COMPLETED,
        effect={"type": "llm_call", "payload": {}, "result_key": "_tmp"},
        result={"usage": usage},
    )


def _decision(index: int, run_id: str, status: str, minute: int) -> StepRecord:
    payload = {"run_id": run_id, "index": index, "status": status, "attempts": 1, "finished_at": _ts(minute)}
    return StepRecord(
        run_id=AUTOMATION,
        step_id=f"completed-{index}",
        node_id="record_outcome",
        status=StepStatus.COMPLETED,
        effect={"type": "emit_event", "payload": {"name": "automation.completed", "payload": payload}},
        idempotency_key=f"automation:completed:{AUTOMATION}:{index}",
    )


def _stores(kind: str, tmp_path: Path):
    if kind == "sqlite":
        db = SqliteDatabase(tmp_path / "runtime_castor.sqlite3")
        return SqliteRunStore(db), SqliteLedgerStore(db)
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs"), JsonlLedgerStore(tmp_path / "runs")
    return InMemoryRunStore(), InMemoryLedgerStore()


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "castor"
    home.mkdir(exist_ok=True)
    (home / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    return home


def _fixture(run_store, ledger) -> None:
    """3 sessions, 7 counted turn roots (4 completed, 2 failed, 1 cancelled), 2 sub-runs, 1 owner run,
    1 occurrence completed, 1 occurrence failed after a retried attempt, and runs that never count."""
    runs = [
        _run("r1", minute=1, session="s1"),
        _run("r2", minute=3, session="s1"),
        _run("r3", minute=5, session="s2", workflow="visit-flow", stamp={"phase": "visit", "entity_id": "entity:castor"}),
        _run("r4", minute=7, session="s2", origin="trigger"),
        _run("r5", minute=9, session="s3", status=RunStatus.FAILED),
        _run("r6", minute=11, session="s3", status=RunStatus.FAILED),
        _run("r7", minute=13, session="s3", status=RunStatus.CANCELLED),
        # sub-runs of r1 and r2: never counted, their tokens count with their root
        _run("r1-sub", minute=2, session="s1", parent="r1", status=RunStatus.FAILED),
        _run("r2-sub", minute=4, session="s1", parent="r2"),
        # the session memory owner run: reserved workflow id, never counted (even with a person origin)
        _run("session_memory_s1", minute=0, session="s1", workflow="__session_memory__"),
        # runs another run or the system started, and a run with no origin: never counted
        _run("r-agent", minute=15, session="s1", origin="agent"),
        _run("r-system", minute=16, session="s2", origin="system"),
        _run("r-legacy", minute=17, session="s2", origin=None),
        # the automation controller: never a turn root
        _run(AUTOMATION, minute=0, session="auto", workflow="automation-controller", origin=None, meta={"automation": {"title": "t"}}),
        # occurrences: #1 completed; #2 failed after a retried attempt (one occurrence, two runs)
        _run("o1", minute=20, session="s2", workflow="digest-flow", origin="trigger", meta=_occurrence_meta(1)),
        _run("o2a1", minute=22, session="s3", workflow="digest-flow", origin="trigger", status=RunStatus.FAILED, meta=_occurrence_meta(2)),
        _run("o2a2", minute=24, session="s3", workflow="digest-flow", origin="trigger", status=RunStatus.FAILED, meta=_occurrence_meta(2)),
    ]
    for run in runs:
        run_store.save(run)
    ledger.append(_llm("r1", "a", inp=100, out=10))
    ledger.append(_llm("r1-sub", "a", inp=50, out=5, usage_keys="legacy"))
    ledger.append(_llm("r2", "a", inp=7, out=3))
    ledger.append(_llm("o2a1", "a", inp=20, out=2))
    ledger.append(_llm("o2a2", "a", inp=30, out=3))
    ledger.append(_llm("r-agent", "a", inp=999, out=999))  # not counted: its tokens never appear
    ledger.append(_llm("session_memory_s1", "a", inp=888, out=888))
    ledger.append(_decision(1, "o1", "completed", 21))
    ledger.append(_decision(2, "o2a2", "failed", 25))


def _comparable(record) -> Dict[str, Any]:
    out = record.to_dict()
    out.pop("computed_at")
    return out


# --- counted runs and signals ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["sqlite", "json", "memory"])
def test_fixture_counts(tmp_path: Path, backend: str) -> None:
    run_store, ledger = _stores(backend, tmp_path)
    _fixture(run_store, ledger)
    record = entity_track_record(_home(tmp_path), run_store, ledger, book_head="f00d")
    assert (record.sessions, record.runs, record.successes, record.failures, record.cancelled, record.in_progress) == (3, 9, 5, 3, 1, 0)
    assert record.subject == "entity:castor"
    assert record.assessed == 0
    assert record.basis == COMPLETION_RATE_ONLY
    # r1 + r1-sub + r2 (new and legacy usage keys) + both attempts of occurrence 2; never r-agent / the owner run.
    assert (record.tokens.input, record.tokens.output, record.tokens.total) == (207, 23, 230)


def test_counted_kinds_and_signals(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    _fixture(run_store, ledger)
    record = entity_track_record(_home(tmp_path), run_store, ledger, book_head="f00d")
    kinds = {k: (v.runs, v.successes, v.failures, v.cancelled, v.tokens_total) for k, v in record.by_task_kind.items()}
    assert kinds == {
        "chat/chat-flow": (6, 3, 2, 1, 175),
        "visit/visit-flow": (1, 1, 0, 0, 0),
        "occurrence/digest-flow": (2, 1, 1, 0, 55),
    }
    signals = {(e.run_id, e.outcome, e.signal) for e in record.recent_outcomes}
    assert ("o1", "success", "occurrence_completed") in signals
    assert ("o2a2", "failure", "occurrence_failed") in signals  # the decision names the last attempt
    assert not any(e.run_id == "o2a1" for e in record.recent_outcomes)
    assert ("r5", "failure", "run_failed") in signals
    assert ("r1", "success", "run_completed") in signals
    assert not any(e.run_id == "r7" for e in record.recent_outcomes)  # cancelled is not an outcome


def test_occurrence_without_decision_is_in_progress(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    run_store.save(_run("o9", minute=1, workflow="digest-flow", origin="trigger", status=RunStatus.FAILED, meta=_occurrence_meta(9)))
    record = session_track_record("s1", run_store, ledger)
    assert (record.runs, record.successes, record.failures, record.in_progress) == (1, 0, 0, 1)


def test_occurrence_decision_naming_a_foreign_run_fails_loudly(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    run_store.save(_run("o1", minute=1, workflow="digest-flow", origin="trigger", meta=_occurrence_meta(1)))
    ledger.append(_decision(1, "somewhere-else", "completed", 2))
    with pytest.raises(TrackRecordError, match="somewhere-else"):
        session_track_record("s1", run_store, ledger)


def test_running_and_waiting_runs_are_in_progress(tmp_path: Path) -> None:
    run_store, ledger = _stores("memory", tmp_path)
    run_store.save(_run("a", minute=1, status=RunStatus.RUNNING))
    run_store.save(_run("b", minute=2, status=RunStatus.WAITING))
    record = session_track_record("s1", run_store, ledger)
    assert (record.runs, record.in_progress, record.successes, record.failures) == (2, 2, 0, 0)


def test_task_done_and_work_order_verdicts_count_once_each(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    home = _home(tmp_path)
    events = [
        {"event": "added", "task_id": "t1", "title": "x", "at": _ts(1)},
        {"event": "added", "task_id": "t2", "title": "y", "at": _ts(2)},
        {"event": "added", "task_id": "t3", "title": "z", "at": _ts(3)},
        {"event": "status", "task_id": "t1", "status": "taken", "at": _ts(4)},
        {"event": "status", "task_id": "t1", "status": "done", "at": _ts(5)},
        {"event": "status", "task_id": "t2", "status": "done", "at": _ts(6)},
        {"event": "status", "task_id": "t2", "status": "pending", "at": _ts(7)},  # reopened: not done
        {"event": "status", "task_id": "t3", "status": "parked", "at": _ts(8)},
    ]
    (home / "task_inbox.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events) + '{"event": "sta', encoding="utf-8")
    (home / "work_order.done.md").write_text(
        # The order's own text quotes a verdict-looking line: not the writer's layout, never read as a verdict.
        "Fix the build.\n\nThe old log said:\n[2026-11-04T09:00:00+00:00] done: copied from a log\n\n"
        "---\n[2026-11-04T11:00:00+00:00] done: the build is green\n"
        "Write the doc.\n\n---\n[2026-11-04T12:00:00+00:00] blocked: no access to the wiki\n",
        encoding="utf-8",
    )
    record = entity_track_record(home, run_store, ledger, book_head=None)
    assert (record.runs, record.sessions, record.successes, record.failures) == (0, 0, 2, 1)
    assert [(e.signal, e.outcome, e.run_id, e.task_kind) for e in record.recent_outcomes] == [
        ("task_done", "success", None, "work/task_inbox"),
        ("work_order_done", "success", None, "work/work_order"),
        ("work_order_blocked", "failure", None, "work/work_order"),
    ]
    assert record.by_task_kind["work/task_inbox"].runs == 0


def test_task_inbox_torn_line_in_the_middle_fails_loudly(tmp_path: Path) -> None:
    run_store, ledger = _stores("memory", tmp_path)
    home = _home(tmp_path)
    (home / "task_inbox.jsonl").write_text('{"event": "add\n{"event": "added", "task_id": "t1", "at": "x"}\n', encoding="utf-8")
    with pytest.raises(TrackRecordError, match="line 1"):
        entity_track_record(home, run_store, ledger, book_head=None)


# --- task kind ------------------------------------------------------------------------------------------


def test_task_kind_from_ids_only() -> None:
    misleading = {"prompt": "occurrence visit work: please do the discussion task", "_temp": {"text": "visit/work"}}
    chat = _run("c", minute=1, workflow="wf-a", extra_vars=misleading)
    assert task_kind_of(chat, role=None) == "chat/wf-a"
    assert task_kind_of(chat, role="occurrence") == "occurrence/wf-a"
    assert task_kind_of(chat, role="discussion") == "discussion/wf-a"
    stamped = _run("v", minute=1, workflow="wf-v", stamp={"phase": "work"}, extra_vars=misleading)
    assert task_kind_of(stamped, role="occurrence") == "work/wf-v"  # the stamp's phase wins over the role
    legacy_stamp = _run("v1", minute=1, workflow="wf-v", stamp={"entity_id": "entity:castor"})
    assert task_kind_of(legacy_stamp, role=None) == "visit/wf-v"  # a phase-less (v1) stamp is a visit
    aliased = _run("v2", minute=1, workflow="wf-v", stamp={"phase": "own_time"})
    assert task_kind_of(aliased, role=None) == "personal/wf-v"
    with pytest.raises(TrackRecordError):
        task_kind_of(_run("v3", minute=1, stamp={"phase": "holiday"}), role=None)


# --- trust ----------------------------------------------------------------------------------------------


def test_trust_formula() -> None:
    assert trust_of(4, 1) == pytest.approx(5 / 7)
    assert trust_of(0, 0) == 0.5
    assert trust_of(0, 3) == pytest.approx(1 / 5)
    assert trust_of(10, 0) == pytest.approx(11 / 12)


def test_reputation_from_fixture(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    _fixture(run_store, ledger)
    reputation = compute_reputation(entity_track_record(_home(tmp_path), run_store, ledger, book_head="f00d"))
    chat = reputation.by_task_kind["chat/chat-flow"]
    assert (chat.successes, chat.failures, chat.support) == (3, 2, 5)  # the cancelled run is not support
    assert chat.trust == pytest.approx(4 / 7)
    visit = reputation.by_task_kind["visit/visit-flow"]
    assert visit.trust == pytest.approx(2 / 3)
    occ = reputation.by_task_kind["occurrence/digest-flow"]
    assert occ.trust == pytest.approx(0.5) and occ.support == 2
    weighted = (4 / 7 * 5 + 2 / 3 * 1 + 0.5 * 2) / 8
    assert reputation.overall.trust == pytest.approx(weighted)
    assert reputation.overall.support == 8
    assert reputation.overall.describe() == f"trust {weighted:.2f} over 8 outcomes"
    assert reputation.basis == COMPLETION_RATE_ONLY
    assert reputation.to_dict()["overall"]["text"].startswith("trust ")


def test_no_outcomes_reads_half_with_no_support(tmp_path: Path) -> None:
    run_store, ledger = _stores("memory", tmp_path)
    run_store.save(_run("a", minute=1, status=RunStatus.CANCELLED))
    reputation = compute_reputation(session_track_record("s1", run_store, ledger))
    cell = reputation.by_task_kind["chat/chat-flow"]
    assert (cell.trust, cell.support) == (0.5, 0)
    assert cell.describe() == "no experience yet"
    assert (reputation.overall.trust, reputation.overall.support) == (0.5, 0)
    assert reputation.overall.describe() == "no experience yet"
    empty = compute_reputation(session_track_record("nobody", run_store, ledger))
    assert (empty.overall.trust, empty.overall.support, empty.by_task_kind, empty.evidence) == (0.5, 0, {}, ())


def test_evidence_is_the_last_twenty_in_time_order(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    for i in range(25):
        run_store.save(_run(f"r{i:02d}", minute=i * 2, status=RunStatus.COMPLETED if i % 3 else RunStatus.FAILED))
    record = session_track_record("s1", run_store, ledger)
    evidence = compute_reputation(record).evidence
    assert len(evidence) == EVIDENCE_WINDOW == 20
    assert [e.run_id for e in evidence] == [f"r{i:02d}" for i in range(5, 25)]
    assert [e.at for e in evidence] == sorted(e.at for e in evidence)
    assert {e.signal for e in evidence} == {"run_completed", "run_failed"}
    assert record.successes + record.failures == 25  # the counts keep everything; only the evidence is windowed


# --- scope ----------------------------------------------------------------------------------------------


def test_session_scope(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    _fixture(run_store, ledger)
    s1 = session_track_record("s1", run_store, ledger)
    assert s1.subject == "session:s1"
    assert s1.fingerprint is None
    assert (s1.sessions, s1.runs, s1.successes, s1.failures) == (1, 2, 2, 0)
    assert s1.tokens.total == 175
    s3 = session_track_record("s3", run_store, ledger)
    assert (s3.runs, s3.successes, s3.failures, s3.cancelled) == (4, 0, 3, 1)


def test_since_keeps_what_started_at_or_after(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    _fixture(run_store, ledger)
    record = entity_track_record(_home(tmp_path), run_store, ledger, book_head="f00d", since=_ts(11))
    assert (record.runs, record.successes, record.failures, record.cancelled) == (4, 1, 2, 1)
    assert record.since == _ts(11)


# --- tokens ---------------------------------------------------------------------------------------------


def test_token_fold_matches_metrics_many(tmp_path: Path) -> None:
    sqlite_store, sqlite_ledger = _stores("sqlite", tmp_path)
    json_store, json_ledger = _stores("json", tmp_path)
    for ledger in (sqlite_ledger, json_ledger):
        ledger.append(_llm("x", "a", inp=11, out=4))
        ledger.append(_llm("x", "b", inp=5, out=6, usage_keys="legacy"))
        ledger.append(StepRecord(run_id="x", step_id="c", node_id="llm", status=StepStatus.STARTED,
                                 effect={"type": "llm_call", "payload": {}}, result={"usage": {"input_tokens": 500}}))
        ledger.append(StepRecord(run_id="x", step_id="d", node_id="t", status=StepStatus.COMPLETED,
                                 effect={"type": "tool_calls", "payload": {}}, result={"usage": {"input_tokens": 700}}))
    metrics = sqlite_ledger.metrics_many(["x"])["x"]
    assert (metrics["tokens_input"], metrics["tokens_output"], metrics["tokens_total"]) == (16, 10, 26)
    assert ledger_token_counts(sqlite_ledger, ["x"]) == ledger_token_counts(json_ledger, ["x"])
    assert token_counts_of_records(json_ledger.list("x")).total == 26


# --- determinism ----------------------------------------------------------------------------------------


def test_deterministic_across_reads_and_backends(tmp_path: Path) -> None:
    results = []
    for backend in ("sqlite", "json", "memory"):
        sub = tmp_path / backend
        sub.mkdir()
        run_store, ledger = _stores(backend, sub)
        _fixture(run_store, ledger)
        home = _home(sub)
        first = entity_track_record(home, run_store, ledger, book_head=None)
        second = entity_track_record(home, run_store, ledger, book_head=None)
        assert _comparable(first) == _comparable(second)
        assert compute_reputation(first).to_dict()["by_task_kind"] == compute_reputation(second).to_dict()["by_task_kind"]
        results.append(_comparable(first))
    assert results[0] == results[1] == results[2]


# --- fingerprint and identity ref (Q1's helpers, ADR §3-§4) ----------------------------------------------


def test_entity_record_carries_the_identity_fingerprint_and_ref(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    home = _home(tmp_path)
    record = entity_track_record(home, run_store, ledger, book_head="f00d")
    assert record.fingerprint == identity_fingerprint(MANIFEST, book_head="f00d")
    assert record.identity_ref is None  # no identity.json yet: a ref is never invented
    record_default_identity(home, entity_id=MANIFEST["entity_id"], gateway_default_voice=None, provenance="test")
    record = entity_track_record(home, run_store, ledger, book_head="f00d")
    assert record.identity_ref == identity_ref("entity:castor", 1)
    assert record.to_dict()["identity_ref"] == {"of": "entity:castor", "rev": 1}
    assert compute_reputation(record).identity_ref == record.identity_ref
    with pytest.raises(TrackRecordError, match="home_id"):
        (home / "manifest.json").write_text(json.dumps({k: v for k, v in MANIFEST.items() if k != "home_id"}), encoding="utf-8")
        entity_track_record(home, run_store, ledger, book_head=None)


def test_session_record_carries_its_newest_runs_ref(tmp_path: Path) -> None:
    run_store, ledger = _stores("memory", tmp_path)
    older, newer = _run("a", minute=1), _run("b", minute=5)
    stamp_run_identity_ref(older.vars, identity_ref("session:s1", 1))
    stamp_run_identity_ref(newer.vars, identity_ref("session:s1", 2))
    run_store.save(older)
    run_store.save(newer)
    run_store.save(_run("c", minute=9))  # carries none: the newest ref still wins
    record = session_track_record("s1", run_store, ledger)
    assert record.identity_ref == identity_ref("session:s1", 2)
    assert record.fingerprint is None
    assert session_track_record("s2", run_store, ledger).identity_ref is None
    foreign = _run("d", minute=12, session="s3")
    stamp_run_identity_ref(foreign.vars, identity_ref("session:elsewhere", 1))
    run_store.save(foreign)
    with pytest.raises(TrackRecordError, match="another subject"):
        session_track_record("s3", run_store, ledger)


# --- room for external feedback (operator ruling 2026-11-04; intake = backlog 1120) ---------------------


def test_feedback_signals_are_typed_but_produced_by_no_source(tmp_path: Path) -> None:
    assert FEEDBACK_SIGNALS == ("feedback_user", "feedback_agent")
    assert set(FEEDBACK_SIGNALS) <= set(SIGNALS)
    judged = OutcomeEvidence(run_id="r1", task_kind="chat/chat-flow", outcome="failure", at=_ts(30),
                             signal="feedback_user", by="default:laurent")
    assert judged.by == "default:laurent"
    run_store, ledger = _stores("sqlite", tmp_path)
    _fixture(run_store, ledger)
    record = entity_track_record(_home(tmp_path), run_store, ledger, book_head="f00d")
    assert record.assessed == 0
    assert not any(e.signal in FEEDBACK_SIGNALS or e.by is not None for e in record.recent_outcomes)
    assert record.basis == compute_reputation(record).basis == "completion-rate only"
