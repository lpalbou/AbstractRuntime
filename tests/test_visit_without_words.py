"""A visit with no word exchanged leaves nothing (operator 2026-11-11 00:40, verbatim: "(a visit opened and closed
without a word) that should not create a memory if no word was shared" and "a zero-turn visit ... no and it should
not even be remembered, fix it").

Pinned through the real durable visit workflow on a real home:

- open + close with zero turns (an operator close, the idle deadline, the pause freeze): no memory row of any kind
  (every table of the memory store, the journal position, the diary), no model call at all (no look-back, no
  authoring), and the life does not count it - the track record has no run and the acquaintances met nobody. The
  ledger keeps the run's open and close rows (the ledger is the record of the run; memory is reconstruction).
- open + one line + close: the records of today (an episode, the look-back call), counted once in the track record
  and once per visitor in the acquaintances.
- `visit_without_words` reads visit runs only and fails loudly on a malformed turn count.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import EffectType, RunStatus  # noqa: E402
from abstractruntime.core.run_origin import origin_for, stamp_origin  # noqa: E402
from abstractruntime.identity.acquaintances import entity_acquaintances  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402
from abstractruntime.identity.track_record import (  # noqa: E402
    TrackRecordError,
    entity_track_record,
    is_wordless_visit,
)
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    VISIT_WORKFLOW_ID,
    VISITOR_WAIT_KEY,
    build_visit_workflow,
    visit_without_words,
)
from test_visit_workflow import _make_home, _ScriptedLLMHandler  # noqa: E402

VISITOR = "person:albou"
ACTOR = "default:albou"
VISIT_KIND = f"visit/{VISIT_WORKFLOW_ID}"
# What a close of a visit with words writes; a wordless close writes none of them.
FORMING_EFFECTS = {"llm_call", "memory_form", "memory_appraise", "diary_write", "memory_access"}


def _home(tmp_path: Path, slug: str) -> Path:
    home_dir = _make_home(tmp_path, slug=slug)
    manifest_path = home_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"slug": slug, "home_id": "home-test", "spark_hash": "a" * 64})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return home_dir


def _memory_rows(home_dir: Path) -> Dict[str, int]:
    """Row count of every table of the home's memory store (read-only)."""
    con = sqlite3.connect(f"file:{home_dir / 'memory.sqlite3'}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
    finally:
        con.close()


def _open(home_dir: Path, llm: _ScriptedLLMHandler, *, idle_seconds: float = 3600):
    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm}, fence=LifecycleFence.admit(home_dir))
    wf = build_visit_workflow(
        ert.home, participants=[VISITOR], idle_seconds=idle_seconds,
        model_info={"provider": "test", "model": "scripted"}, visit_id="visit-w0",
    )
    return ert, wf


def _start(ert: Any, wf: Any) -> str:
    """Start and park the visit as the door does: the stamp's participants and a person origin."""
    run_vars: Dict[str, Any] = {
        "_runtime": {"entity": {"phase": "visit", "participants": [VISITOR, ert.entity_id], "visit_id": "visit-w0"}},
    }
    stamp_origin(run_vars, origin_for("person", actor=ACTOR, client="test"))
    run_id = ert.runtime.start(workflow=wf, vars=run_vars, session_id="visit-w0")
    state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
    assert state.status == RunStatus.WAITING and state.waiting.wait_key == VISITOR_WAIT_KEY
    return run_id


def _close(ert: Any, wf: Any, run_id: str, how: str) -> Any:
    if how == "idle":
        time.sleep(0.05)  # nobody spoke; the idle deadline passes
        return ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=200)
    payload: Dict[str, Any] = {"kind": "close", "closed_by": "operator", "reason": "closed without a word"}
    if how == "pause":
        payload.update({"closed_by": "pause", "skip_reflection": True})
    return ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY, payload=payload, max_steps=200)


@pytest.mark.parametrize("how", ["operator", "idle", "pause"])
def test_wordless_visit_leaves_no_memory_runs_no_model_and_is_not_counted(tmp_path: Path, how: str) -> None:
    home_dir = _home(tmp_path, slug="quietling")
    rows_before = _memory_rows(home_dir)
    llm = _ScriptedLLMHandler([])  # nothing scripted: any model call would show in llm.calls
    ert, wf = _open(home_dir, llm, idle_seconds=0.01 if how == "idle" else 3600)
    try:
        seq_before = ert.home.ms.current_seq()
        diary_before = len(ert.home.diary.list_entries())
        run_id = _start(ert, wf)
        state = _close(ert, wf, run_id, how)

        assert state.status == RunStatus.COMPLETED
        assert state.output["turns"] == 0
        assert state.output["reflection_notices"] == []
        assert "reflection_pending" not in state.output  # nothing owed to a next open either
        # No model call of any kind: no look-back, no authoring.
        assert llm.calls == []
        # No memory: the journal did not move, no diary line.
        assert ert.home.ms.current_seq() == seq_before
        assert len(ert.home.diary.list_entries()) == diary_before
        # The ledger keeps the run's open (the park) and its close; nothing that forms or calls a model ran.
        ledger = ert.runtime.get_ledger(run_id)
        effects = [str(((r.get("effect") or {}).get("type")) or "") for r in ledger if isinstance(r, dict)]
        assert "wait_event" in effects
        assert any(r.get("node_id") == "DONE" and r.get("status") == "completed" for r in ledger)
        assert not FORMING_EFFECTS & set(effects)
        # The life does not count it: no run in the track record, nobody met.
        run = ert.run_store.load(run_id)
        assert visit_without_words(run) is True
        record = entity_track_record(home_dir, ert.run_store, ert.ledger_store, book_head=ert.home.diary.head())
        assert (record.runs, record.successes, record.in_progress, dict(record.by_task_kind)) == (0, 0, 0, {})
        met = entity_acquaintances(home_dir, ert.run_store)
        assert (met.humans, met.entities) == ((), ())
    finally:
        ert.close()
    # Every table of the memory store, row for row (bindings, events, traces, valence, versions, tasks...).
    assert _memory_rows(home_dir) == rows_before


def test_wordless_visit_still_open_is_not_counted_until_its_first_line(tmp_path: Path) -> None:
    home_dir = _home(tmp_path, slug="waitling")
    llm = _ScriptedLLMHandler(["Hello."])
    ert, wf = _open(home_dir, llm)
    try:
        run_id = _start(ert, wf)
        met = entity_acquaintances(home_dir, ert.run_store)
        assert met.humans == ()
        assert entity_track_record(home_dir, ert.run_store, ert.ledger_store, book_head=None).runs == 0
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Hi.", "speaker": VISITOR}, max_steps=100,
        )
        assert state.status == RunStatus.WAITING  # parked again, one turn in
        met = entity_acquaintances(home_dir, ert.run_store)
        assert {a.id: a.interactions for a in met.humans} == {VISITOR: 1, ACTOR: 1}
        record = entity_track_record(home_dir, ert.run_store, ert.ledger_store, book_head=None)
        assert (record.runs, record.in_progress) == (1, 1)
    finally:
        ert.close()


def test_one_line_visit_forms_todays_records_and_counts_once(tmp_path: Path) -> None:
    from abstractmemory import TripleQuery

    home_dir = _home(tmp_path, slug="talkling")
    llm = _ScriptedLLMHandler(["Hello Laurent.", "A short visit; I keep it."])
    ert, wf = _open(home_dir, llm)
    try:
        seq_before = ert.home.ms.current_seq()
        run_id = _start(ert, wf)
        ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Hello.", "speaker": VISITOR}, max_steps=100,
        )
        state = _close(ert, wf, run_id, "operator")
        assert state.status == RunStatus.COMPLETED
        assert state.output["turns"] == 1
        # The records of today: the turn's episode formed and the close ran its look-back model call.
        rows = ert.home.ms.query(TripleQuery(scope="life", owner_id=ert.entity_id, limit=0))
        kinds = [(a.attributes or {}).get("record_kind") for a in rows if isinstance(a.attributes, dict)]
        assert kinds.count("episode") == 1
        assert ert.home.ms.current_seq() > seq_before
        assert any(c.get("turn_id") == "t-reflect" for c in llm.calls)
        # Counted once: one completed visit, one interaction per visitor id.
        assert visit_without_words(ert.run_store.load(run_id)) is False
        record = entity_track_record(home_dir, ert.run_store, ert.ledger_store, book_head=ert.home.diary.head())
        assert (record.runs, record.successes) == (1, 1)
        assert record.by_task_kind[VISIT_KIND].runs == 1
        met = entity_acquaintances(home_dir, ert.run_store)
        assert {a.id: a.interactions for a in met.humans} == {VISITOR: 1, ACTOR: 1}
    finally:
        ert.close()


def test_visit_without_words_reads_visit_runs_only_and_fails_loudly() -> None:
    def run(workflow_id: str, visit: Any) -> SimpleNamespace:
        return SimpleNamespace(run_id="r1", workflow_id=workflow_id, vars={} if visit is None else {"_visit": visit})

    assert visit_without_words(run(VISIT_WORKFLOW_ID, {"turn_n": 0})) is True
    assert visit_without_words(run(VISIT_WORKFLOW_ID, {})) is True  # opened, no turn yet (or a refused prelude)
    assert visit_without_words(run(VISIT_WORKFLOW_ID, None)) is True  # started, not yet ticked
    assert visit_without_words(run(VISIT_WORKFLOW_ID, {"turn_n": 1})) is False
    # Another workflow is never a visit, whatever its vars say.
    assert visit_without_words(run("chat-flow", {"turn_n": 0})) is False
    for bad in ("0", 1.0, True, None):
        with pytest.raises(ValueError, match="turn_n is not an integer"):
            visit_without_words(run(VISIT_WORKFLOW_ID, {"turn_n": bad}))
        with pytest.raises(TrackRecordError, match="turn_n is not an integer"):
            is_wordless_visit(run(VISIT_WORKFLOW_ID, {"turn_n": bad}))
