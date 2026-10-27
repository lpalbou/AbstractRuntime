"""The runtime publishes the executing step to its handler (ADR 0009 decision 3; `core.step_context`).

Real `Runtime` runs on real temporary stores (SQLite and JSON files), no mocks.

Mutants this file kills (each checked by hand when it was written):
- `_execute_effect_with_retry` does not install `effect_step_context` -> the handler sees None and every case fails;
- the context carries `seq=None` (the appended seq dropped) or a guessed seq (`len(ledger) + 1`) -> the `get_record`
  resolution fails or points at another row;
- `effect_step_context` does not restore the previous value -> the parent handler, after driving a child run, sees
  the child's step (or None) and the nesting case fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.core.step_context import current_step_context
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore, SqliteRunStore

pytestmark = pytest.mark.basic


def _stores(kind: str, tmp_path: Path):
    if kind == "sqlite":
        db = SqliteDatabase(tmp_path / "runtime.sqlite3")
        return SqliteRunStore(db), SqliteLedgerStore(db)
    return JsonFileRunStore(tmp_path / "runs"), JsonlLedgerStore(tmp_path / "runs")


def _one_effect_workflow(workflow_id: str, payload: Dict[str, Any]) -> WorkflowSpec:
    def node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("out") is not None:
            return StepPlan(node_id="n1", complete_output={"ok": True})
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.TOOL_CALLS, payload=dict(payload), result_key="out"),
            next_node="n1",
        )

    return WorkflowSpec(workflow_id=workflow_id, entry_node="n1", nodes={"n1": node})


@pytest.mark.parametrize("kind", ["sqlite", "jsonl"])
def test_handler_sees_its_own_started_row(kind: str, tmp_path: Path) -> None:
    run_store, ledger = _stores(kind, tmp_path)
    seen: List[Any] = []
    payload = {"tool_calls": [{"name": "noop", "arguments": {"marker": "verbatim-42"}}], "mode": "x"}

    def handler(run: RunState, effect: Effect, default_next_node: Optional[str] = None) -> EffectOutcome:
        seen.append(current_step_context(run.run_id))
        seen.append(current_step_context("some-other-run"))
        return EffectOutcome.completed({"results": []})

    wf = _one_effect_workflow("wf-step-ctx", payload)
    rt = Runtime(run_store=run_store, ledger_store=ledger, effect_handlers={EffectType.TOOL_CALLS: handler})
    # Rows of another run precede ours in the same store: the seq is per run, never a global count.
    first = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=first)
    seen.clear()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)

    ctx, other = seen[0], seen[1]
    assert other is None
    assert ctx is not None and ctx.run_id == run_id and ctx.node_id == "n1" and ctx.attempt == 1
    assert ctx.effect_type == "tool_calls" and isinstance(ctx.seq, int)
    row = ledger.get_record(run_id, ctx.seq)
    assert row is not None and row["status"] == "started" and row["step_id"] == ctx.step_id
    assert row["seq"] == ctx.seq
    assert row["effect"]["payload"]["tool_calls"][0]["arguments"] == {"marker": "verbatim-42"}
    assert current_step_context() is None  # restored after the step


def test_handler_called_directly_sees_none() -> None:
    seen: List[Any] = []

    def handler(run: RunState, effect: Effect, default_next_node: Optional[str] = None) -> EffectOutcome:
        seen.append(current_step_context(run.run_id))
        return EffectOutcome.completed({})

    run = RunState.new(workflow_id="wf", entry_node="n1")
    handler(run, Effect(type=EffectType.TOOL_CALLS, payload={}), None)
    assert seen == [None]


def test_child_run_handler_sees_its_own_run_and_the_parent_context_is_restored(tmp_path: Path) -> None:
    run_store, ledger = _stores("sqlite", tmp_path)
    seen: Dict[str, Any] = {}
    child_wf = _one_effect_workflow("wf-child", {"tool_calls": [], "mode": "child"})

    def child_handler(run: RunState, effect: Effect, default_next_node: Optional[str] = None) -> EffectOutcome:
        seen["child"] = current_step_context(run.run_id)
        seen["child_sees_parent"] = current_step_context(run.parent_run_id)
        return EffectOutcome.completed({"results": []})

    def parent_handler(run: RunState, effect: Effect, default_next_node: Optional[str] = None) -> EffectOutcome:
        seen["parent_before"] = current_step_context(run.run_id)
        nested = Runtime(run_store=run_store, ledger_store=ledger,
                         effect_handlers={EffectType.TOOL_CALLS: child_handler})
        child_id = nested.start(workflow=child_wf, parent_run_id=run.run_id)
        nested.tick(workflow=child_wf, run_id=child_id)
        seen["child_id"] = child_id
        seen["parent_after"] = current_step_context(run.run_id)
        return EffectOutcome.completed({"results": []})

    parent_wf = _one_effect_workflow("wf-parent", {"tool_calls": [], "mode": "parent"})
    rt = Runtime(run_store=run_store, ledger_store=ledger, effect_handlers={EffectType.TOOL_CALLS: parent_handler})
    parent_id = rt.start(workflow=parent_wf)
    rt.tick(workflow=parent_wf, run_id=parent_id)

    assert seen["child"] is not None and seen["child"].run_id == seen["child_id"]
    assert seen["child_sees_parent"] is None  # the enclosing step is another run's: not this write's source
    child_row = ledger.get_record(seen["child_id"], seen["child"].seq)
    assert child_row["status"] == "started" and child_row["effect"]["payload"]["mode"] == "child"
    assert seen["parent_before"] is not None and seen["parent_after"] == seen["parent_before"]
    parent_row = ledger.get_record(parent_id, seen["parent_after"].seq)
    assert parent_row["status"] == "started" and parent_row["effect"]["payload"]["mode"] == "parent"


class _ContextRecordingExecutor:
    """A tool executor that records the step context its tool runs under (what a memory-writing tool reads)."""

    def __init__(self) -> None:
        self.seen: List[Any] = []

    def execute(self, *, tool_calls):
        self.seen.append(current_step_context())
        return {"mode": "executed", "results": [
            {"name": tc["name"], "call_id": tc.get("call_id"), "success": True, "output": "ok"} for tc in tool_calls
        ]}


@pytest.mark.parametrize("kind", ["sqlite", "jsonl"])
def test_a_tool_run_by_an_approval_resume_sees_the_waiting_steps_started_row(kind: str, tmp_path: Path) -> None:
    """Approved tools run inside `Runtime.resume`, not `_execute_effect_with_retry`: the resume publishes the
    TOOL_CALLS step that waited (its STARTED row), so a tool writing memory names the same row it would have named
    without the approval gate. Mutant: drop the `effect_step_context` on the approval path -> the tool sees None."""
    from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
    from abstractruntime.integrations.abstractcore.tool_executor import ApprovalToolExecutor, ToolApprovalPolicy

    run_store, ledger = _stores(kind, tmp_path)
    delegate = _ContextRecordingExecutor()
    tools = ApprovalToolExecutor(delegate=delegate, policy=ToolApprovalPolicy(
        auto_approve_tools=set(), require_approval_tools={"note_capture"}))
    rt = Runtime(run_store=run_store, ledger_store=ledger,
                 effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    rt.set_tool_executor_for_resume(tools)

    def tools_node(run: RunState, ctx: Any) -> StepPlan:
        return StepPlan(node_id="tools", next_node="done", effect=Effect(
            type=EffectType.TOOL_CALLS, result_key="tool_results",
            payload={"tool_calls": [{"name": "note_capture", "arguments": {"text": "Met Dana"}, "call_id": "c0"}]}))

    def done_node(run: RunState, ctx: Any) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="wf-approval-ctx", entry_node="tools", nodes={"tools": tools_node, "done": done_node})
    run_id = rt.start(workflow=wf)
    waiting = rt.tick(workflow=wf, run_id=run_id)
    assert waiting.status.value == "waiting" and delegate.seen == []
    done = rt.resume(workflow=wf, run_id=run_id, wait_key=waiting.waiting.wait_key, payload={"approved": True})
    assert done.status.value == "completed", done.error

    (ctx,) = delegate.seen
    assert ctx is not None and ctx.run_id == run_id and ctx.effect_type == "tool_calls" and ctx.node_id == "tools"
    waiting_row = next(r for r in ledger.list(run_id) if r["status"] == "waiting")
    assert ctx.step_id == waiting_row["step_id"]
    row = ledger.get_record(run_id, ctx.seq)
    assert row["status"] == "started" and row["step_id"] == ctx.step_id and row["seq"] < waiting_row["seq"]
    assert row["effect"]["payload"]["tool_calls"][0]["arguments"] == {"text": "Met Dana"}
    assert current_step_context() is None


@pytest.mark.parametrize("kind", ["sqlite", "jsonl"])
def test_a_step_digest_vouches_for_the_started_row_and_an_edit_is_a_mismatch(kind: str, tmp_path: Path) -> None:
    """Architect A4: the step context carries the sha256 of the STARTED record as written; reconstruction of a ref
    carrying it verifies the row, and an in-place edit of the row (an UPDATE on SQLite, a rewritten line on JSONL)
    answers `mismatch`, never `ok` with the edited bytes. Mutants: no digest in the context -> nothing to verify;
    skip the verification -> `ok` after the edit."""
    import json
    import sqlite3

    from abstractruntime.integrations.abstractmemory.memory_facade import MemoryFacade
    from abstractruntime.storage.serialize import record_digest

    run_store, ledger = _stores(kind, tmp_path)
    seen: List[Any] = []

    def handler(run: RunState, effect: Effect, default_next_node: Optional[str] = None) -> EffectOutcome:
        seen.append(current_step_context(run.run_id))
        return EffectOutcome.completed({"results": []})

    wf = _one_effect_workflow("wf-digest", {"tool_calls": [{"name": "t", "arguments": {"say": "the truth"}}]})
    rt = Runtime(run_store=run_store, ledger_store=ledger, effect_handlers={EffectType.TOOL_CALLS: handler})
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)
    (ctx,) = seen
    row = ledger.get_record(run_id, ctx.seq)
    assert ctx.digest and ctx.digest == record_digest(row)
    ref = {"kind": "step", "run_id": run_id, "seq": ctx.seq, "step_id": ctx.step_id, "digest": ctx.digest}
    first = MemoryFacade(ledger_store=ledger).reconstruct_source(ref)
    assert first["status"] == "ok" and first["metadata"]["digest_verified"] is True

    edited = json.loads(json.dumps({k: v for k, v in row.items() if k != "seq"}))
    edited["effect"]["payload"]["tool_calls"][0]["arguments"]["say"] = "a lie"
    if kind == "sqlite":
        with sqlite3.connect(tmp_path / "runtime.sqlite3") as conn:
            conn.execute("UPDATE ledger SET record_json = ? WHERE run_id = ? AND seq = ?",
                         (json.dumps(edited), run_id, ctx.seq))
        reader = ledger
    else:
        path = tmp_path / "runs" / f"ledger_{run_id}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[ctx.seq - 1] = json.dumps(edited) + "\n"
        path.write_text("".join(lines), encoding="utf-8")
        reader = JsonlLedgerStore(tmp_path / "runs")
    second = MemoryFacade(ledger_store=reader).reconstruct_source(ref)
    assert second["status"] == "mismatch" and second["expected_digest"] == ctx.digest
    assert second["content"]["effect"]["payload"]["tool_calls"][0]["arguments"]["say"] == "a lie"
