from __future__ import annotations

from typing import Any

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.memory.active_memory import add_critical_insight, ensure_active_memory
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_memory_compact_structured_effect_archives_overflow_and_appends_key_history() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store)
    artifact_store = InMemoryArtifactStore()
    runtime.set_artifact_store(artifact_store)

    vars: dict[str, Any] = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {},
        "_runtime": {"memory_spans": []},
        "_temp": {},
        "_limits": {},
    }

    times = [
        "2025-01-01T00:00:00+00:00",
        "2025-01-01T00:00:01+00:00",
        "2025-01-01T00:00:02+00:00",
        "2025-01-01T00:00:03+00:00",
        "2025-01-01T00:00:04+00:00",
        "2025-01-01T00:00:05+00:00",
    ]

    def now_iso() -> str:
        return times.pop(0)

    ensure_active_memory(vars, now_iso=now_iso)
    for i in range(5):
        add_critical_insight(vars, text=f"insight {i}", now_iso=now_iso)

    def compact_node(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="compact",
            effect=Effect(
                type=EffectType.MEMORY_COMPACT_STRUCTURED,
                payload={"components": ["critical_insights"], "preserve": {"critical_insights": 2}},
                result_key="_temp.compact",
            ),
            next_node="done",
        )

    def done_node(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"result": run.vars.get("_temp", {}).get("compact")})

    wf = WorkflowSpec(
        workflow_id="wf_memory_compact_structured",
        entry_node="compact",
        nodes={"compact": compact_node, "done": done_node},
    )

    run_id = runtime.start(workflow=wf, vars=vars)
    state = runtime.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.COMPLETED

    runtime_ns = state.vars.get("_runtime") if isinstance(state.vars.get("_runtime"), dict) else {}
    mem = runtime_ns.get("active_memory") if isinstance(runtime_ns.get("active_memory"), dict) else {}
    insights = mem.get("critical_insights")
    assert isinstance(insights, list)
    assert len(insights) == 2

    spans = runtime_ns.get("memory_spans")
    assert isinstance(spans, list)
    active_spans = [s for s in spans if isinstance(s, dict) and s.get("kind") == "active_memory_span"]
    assert active_spans
    span = next(s for s in active_spans if s.get("component_id") == "critical_insights")
    span_id = span.get("artifact_id")
    assert isinstance(span_id, str) and span_id

    payload = artifact_store.load_json(span_id)
    assert isinstance(payload, dict)
    assert payload.get("kind") == "active_memory_span"
    assert payload.get("component_id") == "critical_insights"
    archived_items = payload.get("items")
    assert isinstance(archived_items, list)
    assert len(archived_items) == 3

    history = mem.get("key_history")
    assert isinstance(history, list)
    assert any(
        isinstance(h, dict)
        and h.get("kind") == "active_memory_compact"
        and span_id in str(h.get("summary") or "")
        for h in history
    )

