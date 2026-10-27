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
