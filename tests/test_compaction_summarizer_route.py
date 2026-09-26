"""MEMORY_COMPACT summarizes a run with the run's OWN provider/model (M2
review S7, 2026-09-26): summarizing a run pinned to another model with the
default loaded the default (possibly a second 15 GB model) just for the
summary. MUTANT: drop the route kwargs in Runtime -> RED."""
from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime import (Effect, EffectType, InMemoryArtifactStore, InMemoryLedgerStore, InMemoryRunStore,
                             StepPlan, WorkflowSpec)
from abstractruntime.core.runtime import Runtime


class _RecordingSummarizer:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def summarize_chat_history(self, messages, *, preserve_recent=6, focus=None, compression_mode="standard",
                               provider=None, model=None):
        self.calls.append({"provider": provider, "model": model})
        return {"summary": "s", "key_points": [], "confidence": 1.0}


class _LegacySummarizer:
    def __init__(self):
        self.calls = 0

    def summarize_chat_history(self, messages, *, preserve_recent=6, focus=None, compression_mode="standard"):
        self.calls += 1
        return {"summary": "s", "key_points": [], "confidence": 1.0}


def _run(summarizer, route):
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), chat_summarizer=summarizer)
    rt.set_artifact_store(InMemoryArtifactStore())

    def compact(run, ctx):
        return StepPlan(node_id="compact", effect=Effect(type=EffectType.MEMORY_COMPACT,
                                                         payload={"preserve_recent": 1}), next_node="done")

    def done(run, ctx):
        return StepPlan(node_id="done", complete_output={})

    wf = WorkflowSpec(workflow_id="compact_route", entry_node="compact", nodes={"compact": compact, "done": done})
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]
    vars_ = {"context": {"messages": messages}}
    if route:
        vars_["_runtime"] = dict(route)
    run_id = rt.start(workflow=wf, vars=vars_)
    for _ in range(5):
        state = rt.get_state(run_id)
        if str(getattr(state.status, "value", state.status)) in ("completed", "failed"):
            break
        rt.tick(workflow=wf, run_id=run_id)
    state = rt.get_state(run_id)
    assert str(getattr(state.status, "value", state.status)) == "completed", getattr(state, "error", None)
    return state


def test_a_pinned_run_is_summarized_by_its_own_model():
    s = _RecordingSummarizer()
    _run(s, {"provider": "mlx", "model": "vendor/RUN"})
    assert s.calls == [{"provider": "mlx", "model": "vendor/RUN"}]


def test_a_run_without_a_route_uses_the_default_and_legacy_summarizers_still_work():
    s = _RecordingSummarizer()
    _run(s, None)
    assert s.calls == [{"provider": None, "model": None}]
    legacy = _LegacySummarizer()
    _run(legacy, {"provider": "mlx", "model": "vendor/RUN"})
    assert legacy.calls == 1
