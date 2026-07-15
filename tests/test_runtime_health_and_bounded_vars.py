"""RuntimeHealth counters + bounded run-vars growth (backlogs 0054 + 0053).

The runtime's self-knowledge was logger.warning folklore; a 24/7 resident's
state grew without ceilings. These tests pin the pulse and the caps:

- health(): counters increment from their triggering conditions (steps,
  retries, failures, waits, resumes, replay reuses, absorbed failures),
  tick durations bucket, snapshot is JSON-safe and isolated;
- vars-size gauge refreshes on effectful ticks and counts threshold
  crossings (warning once per run, counter every crossing);
- node-trace entries above the inline cap offload their leaves (artifact
  refs) or truncate WITH the #TRUNCATION label — small entries untouched;
- _runtime.inbox and evidence_warnings drop-oldest WITH counters;
- list_stalled_waits reports aged waits oldest-first, [] on non-queryable
  stores.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus, StepPlan
from abstractruntime.core.runtime import (
    EVIDENCE_WARNINGS_MAX,
    INBOX_MAX_MESSAGES,
    NODE_TRACE_ENTRY_INLINE_CAP,
    EffectOutcome,
    Runtime,
)
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.steer_sidecar import InMemorySteerSidecar


def _runtime(handler, *, artifact_store: Any = None, steer_store: Any = None):
    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: handler},
        artifact_store=artifact_store,
        steer_store=steer_store,
    )


def _one_effect_workflow(payload: Optional[dict] = None) -> WorkflowSpec:
    def node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("out") is not None:
            return StepPlan(node_id="n1", complete_output={"ok": True})
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.LLM_CALL, payload=dict(payload or {"prompt": "x"}), result_key="out"),
            next_node="n1",
        )

    return WorkflowSpec(workflow_id="wf-health", entry_node="n1", nodes={"n1": node})


# ---------------------------------------------------------------------------
# 0054: counters
# ---------------------------------------------------------------------------


def test_health_counts_steps_ticks_and_completions() -> None:
    def ok(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": "hi"})

    rt = _runtime(ok)
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)

    h = rt.health()
    assert h["counters"]["effect_steps_total"] == 1
    assert h["counters"]["ticks_total"] >= 1
    assert sum(h["tick_duration_buckets"].values()) == h["counters"]["ticks_total"]
    assert h["counters"].get("effect_failures_total", 0) == 0
    json.dumps(h)  # snapshot must be JSON-safe

    # Snapshot isolation: mutating the copy never touches live state.
    h["counters"]["effect_steps_total"] = 999
    assert rt.health()["counters"]["effect_steps_total"] == 1


def test_health_counts_retries_and_failures() -> None:
    from abstractruntime.core.policy import RetryPolicy

    calls = {"n": 0}

    def flaky(run, effect, default_next_node=None):
        calls["n"] += 1
        return EffectOutcome.failed("boom")

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: flaky},
        effect_policy=RetryPolicy(llm_max_attempts=3, tool_max_attempts=1),
    )
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.FAILED

    h = rt.health()["counters"]
    assert h["effect_retries_total"] == 2, "attempts 2 and 3 are retries"
    assert h["effect_failures_total"] == 1


def test_health_counts_waits_and_resumes() -> None:
    from abstractruntime.core.models import WaitReason, WaitState

    def waits(run, effect, default_next_node=None):
        if run.vars.get("resumed"):
            return EffectOutcome.completed({"content": "after"})
        return EffectOutcome.waiting(
            WaitState(reason=WaitReason.EVENT, wait_key="k1", resume_to_node="n1", result_key="out")
        )

    rt = _runtime(waits)
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    state = rt.tick(workflow=wf, run_id=run_id)
    assert state.status == RunStatus.WAITING
    assert rt.health()["counters"]["waits_entered_total"] == 1

    loaded = rt.get_state(run_id)
    loaded.vars["resumed"] = True
    rt._run_store.save(loaded)
    rt.resume(workflow=wf, run_id=run_id, wait_key="k1", payload={"go": True})
    assert rt.health()["counters"]["resumes_total"] == 1


def test_health_counts_steer_drains() -> None:
    def ok(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": "hi"})

    sidecar = InMemorySteerSidecar()
    rt = _runtime(ok, steer_store=sidecar)
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    sidecar.append(run_id, {"role": "system", "content": "steer!"})
    rt.tick(workflow=wf, run_id=run_id)

    h = rt.health()["counters"]
    assert h["steer_drains_total"] == 1
    assert h["steer_messages_delivered_total"] == 1


# ---------------------------------------------------------------------------
# 0053: vars gauge + caps
# ---------------------------------------------------------------------------


def test_vars_gauge_refreshes_on_effectful_ticks() -> None:
    def ok(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": "hi"})

    rt = _runtime(ok)
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)
    gauges = rt.health()["gauges"]
    assert gauges.get("vars_bytes_last", 0) > 0
    assert gauges.get("vars_bytes_last_max", 0) >= gauges["vars_bytes_last"]


def test_vars_threshold_crossing_counts_and_warns_once(caplog) -> None:
    import logging

    def fat(run, effect, default_next_node=None):
        run.vars["blob"] = "x" * (2 * 1024 * 1024)
        return EffectOutcome.completed({"content": "hi"})

    def node(run: RunState, ctx: Any) -> StepPlan:
        step = int(run.vars.get("step") or 0)
        if step >= 2:
            return StepPlan(node_id="n1", complete_output={"ok": True})
        run.vars["step"] = step + 1
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.LLM_CALL, payload={"prompt": "x"}, result_key=f"out{step}"),
            next_node="n1",
        )

    rt = _runtime(fat)
    rt.VARS_BYTES_WARN_THRESHOLD = 1024 * 1024  # instance override for the pin
    wf = WorkflowSpec(workflow_id="wf-fat", entry_node="n1", nodes={"n1": node})
    run_id = rt.start(workflow=wf)
    with caplog.at_level(logging.WARNING, logger="abstractruntime.core.runtime"):
        # Two EFFECTFUL ticks on the same run (one step each): the counter
        # counts both crossings, the log fires only once per run.
        rt.tick(workflow=wf, run_id=run_id, max_steps=1)
        rt.tick(workflow=wf, run_id=run_id, max_steps=1)
    h = rt.health()
    assert h["counters"].get("vars_bytes_threshold_crossings_total", 0) >= 2
    assert h["gauges"]["vars_bytes_last"] > 1024 * 1024
    warn_lines = [r for r in caplog.records if "vars serialized to" in str(r.getMessage())]
    assert len(warn_lines) == 1, "the warning fires ONCE per run; the counter keeps counting"


def test_node_trace_entry_above_cap_offloads_with_artifact_store() -> None:
    big = "y" * (NODE_TRACE_ENTRY_INLINE_CAP + 4096)

    def fat_result(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": big})

    artifacts = InMemoryArtifactStore()
    rt = _runtime(fat_result, artifact_store=artifacts)
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)

    traces = rt.get_node_traces(run_id)
    entry = traces["n1"]["steps"][0]
    assert entry.get("trace_bounded") == "offloaded"
    raw = json.dumps(entry)
    assert big not in raw, "the oversized leaf must not rest inline in vars"
    assert "$artifact" in raw, "the leaf offloads to the artifact store"
    # The run-vars copy of the RESULT (result_key) stays untouched.
    assert rt.get_state(run_id).vars["out"]["content"] == big


def test_node_trace_entry_truncates_with_label_without_store() -> None:
    big = "z" * (NODE_TRACE_ENTRY_INLINE_CAP + 4096)

    def fat_result(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": big})

    rt = _runtime(fat_result, artifact_store=None)
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)

    entry = rt.get_node_traces(run_id)["n1"]["steps"][0]
    assert entry.get("trace_bounded") == "truncated"
    assert "#TRUNCATION" in json.dumps(entry)


def test_small_trace_entries_are_untouched() -> None:
    def ok(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": "small"})

    rt = _runtime(ok, artifact_store=InMemoryArtifactStore())
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)
    entry = rt.get_node_traces(run_id)["n1"]["steps"][0]
    assert "trace_bounded" not in entry
    assert entry["result"]["content"] == "small"


def test_full_inbox_backpressures_steers_instead_of_destroying_them() -> None:
    """Adversary P2-2 fold: the first cut drop-oldested at delivery —
    destroying operator words UNREAD while `steer_seen` claimed them all.
    Now a full inbox DEFERS: undelivered steers stay pending in the sidecar
    (watermark redelivery), the ack covers only what was delivered, and
    nothing is ever destroyed."""

    def ok(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": "hi"})

    sidecar = InMemorySteerSidecar()
    rt = _runtime(ok, steer_store=sidecar)
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)

    # Fill the inbox to one below the cap, then send three steers.
    run = rt.get_state(run_id)
    runtime_ns = run.vars.setdefault("_runtime", {})
    runtime_ns["inbox"] = [{"role": "system", "content": f"old-{i}"} for i in range(INBOX_MAX_MESSAGES - 1)]
    rt._run_store.save(run)
    for i in range(3):
        sidecar.append(run_id, {"role": "system", "content": f"new-{i}"})
    rt.tick(workflow=wf, run_id=run_id)

    state = rt.get_state(run_id)
    inbox = state.vars["_runtime"]["inbox"]
    assert len(inbox) == INBOX_MAX_MESSAGES, "delivery stops at the cap"
    assert inbox[0]["content"] == "old-0", "nothing was destroyed"
    assert inbox[-1]["content"] == "new-0", "headroom delivered oldest-pending first"
    pending = sidecar.pending(run_id)
    assert [p["message"]["content"] for p in pending] == ["new-1", "new-2"], (
        "undelivered steers stay PENDING for the next boundary"
    )
    assert rt.health()["counters"]["inbox_backpressure_total"] >= 1

    # A zero-headroom drain delivers nothing and acks nothing.
    rt.tick(workflow=wf, run_id=run_id)
    assert len(sidecar.pending(run_id)) == 2, "still pending, never dropped"


def test_evidence_warnings_cap(monkeypatch) -> None:
    from abstractruntime.core.runtime import _ensure_runtime_namespace

    def ok(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": "hi"})

    rt = _runtime(ok, artifact_store=InMemoryArtifactStore())
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    run = rt.get_state(run_id)
    ns = _ensure_runtime_namespace(run.vars)
    ns["evidence_warnings"] = [{"error": f"w{i}"} for i in range(EVIDENCE_WARNINGS_MAX)]

    # Force the capture path to raise so the warning-append site runs
    # (the method imports EvidenceRecorder at call time).
    class _BoomRecorder:
        def __init__(self, *a, **k):
            raise RuntimeError("evidence boom")

    monkeypatch.setattr("abstractruntime.evidence.EvidenceRecorder", _BoomRecorder)
    rt._maybe_record_tool_evidence(
        run=run,
        node_id="n1",
        effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [{"name": "web_search"}]}),
        tool_results={"results": [{}]},
    )
    warnings = run.vars["_runtime"]["evidence_warnings"]
    assert len(warnings) <= EVIDENCE_WARNINGS_MAX, "drop-oldest keeps the list bounded"
    assert int(run.vars["_runtime"].get("evidence_warnings_dropped") or 0) >= 1


# ---------------------------------------------------------------------------
# 0054: stalled waits
# ---------------------------------------------------------------------------


def test_list_stalled_waits_reports_aged_waits() -> None:
    from abstractruntime.core.models import WaitReason, WaitState

    def waits(run, effect, default_next_node=None):
        return EffectOutcome.waiting(
            WaitState(reason=WaitReason.EVENT, wait_key="stalled", resume_to_node="n1", result_key="out")
        )

    rt = _runtime(waits)
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)

    # Fresh wait: below the age threshold.
    assert rt.list_stalled_waits(older_than_s=3600) == []

    # Age the run by rewriting updated_at.
    run = rt.get_state(run_id)
    run.updated_at = "2020-01-01T00:00:00+00:00"
    rt._run_store.save(run)

    rows = rt.list_stalled_waits(older_than_s=300)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run_id
    assert row["wait_reason"] == "event"
    assert row["wait_key"] == "stalled"
    assert row["age_s"] > 300
    assert row["paused"] is False


def test_stalled_waits_oldest_survive_a_full_window() -> None:
    """Adversary P1-1 fold: with newest-first candidate windows, the OLDEST
    waits — the definition of stalled — fell off the end once waiting runs
    exceeded the window, and the board read [] exactly when it mattered.
    The ordered index path must surface the oldest regardless of how many
    fresh waits exist."""
    from abstractruntime.core.models import WaitReason, WaitState

    def waits(run, effect, default_next_node=None):
        return EffectOutcome.waiting(
            WaitState(reason=WaitReason.EVENT, wait_key=f"k-{run.run_id[:6]}", resume_to_node="n1", result_key="out")
        )

    rt = _runtime(waits)
    wf = _one_effect_workflow()

    stalled_ids = []
    for i in range(3):
        rid = rt.start(workflow=wf)
        rt.tick(workflow=wf, run_id=rid)
        run = rt.get_state(rid)
        run.updated_at = f"2020-01-01T00:00:0{i}+00:00"
        rt._run_store.save(run)
        stalled_ids.append(rid)
    for _ in range(30):
        rid = rt.start(workflow=wf)
        rt.tick(workflow=wf, run_id=rid)

    rows = rt.list_stalled_waits(older_than_s=300, limit=2)
    assert len(rows) == 2
    assert {r["run_id"] for r in rows} <= set(stalled_ids), "the OLDEST stalls win the window"
    assert rows[0]["age_s"] >= rows[1]["age_s"]
    assert all(r["wait_key"] for r in rows), "wait_key resolves through the winner's document load"


def test_health_ring_captures_effect_failures() -> None:
    """Adversary P3 fold: the last-errors ring had one producer
    (steer_drain); effect failures — the dominant breakage class — must
    reach the ring that exists to answer 'what broke last'."""
    from abstractruntime.core.policy import RetryPolicy

    def broken(run, effect, default_next_node=None):
        return EffectOutcome.failed("provider exploded: quota")

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: broken},
        effect_policy=RetryPolicy(llm_max_attempts=1, tool_max_attempts=1),
    )
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)
    ring = rt.health()["last_errors"]
    assert any(e["site"] == "effect" and "provider exploded" in e["message"] for e in ring)


def test_oversized_error_string_is_bounded_on_the_offload_path() -> None:
    """Adversary P1-2 fold: top-level `error` is a string, not a subtree —
    it escaped the offload cap entirely (a 100KB provider error stamped
    'offloaded' at 102KB; failed steps are exactly the entries that loop)."""
    from abstractruntime.core.policy import RetryPolicy
    from abstractruntime.core.runtime import NODE_TRACE_ENTRY_INLINE_CAP

    big_error = "E" * (NODE_TRACE_ENTRY_INLINE_CAP + 8192)

    def broken(run, effect, default_next_node=None):
        return EffectOutcome.failed(big_error)

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: broken},
        artifact_store=InMemoryArtifactStore(),
        effect_policy=RetryPolicy(llm_max_attempts=1, tool_max_attempts=1),
    )
    wf = _one_effect_workflow()
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)

    entry = rt.get_node_traces(run_id)["n1"]["steps"][0]
    assert entry.get("trace_bounded") == "offloaded"
    raw = json.dumps(entry)
    assert big_error not in raw, "the error string must be bounded too"
    assert len(raw) < NODE_TRACE_ENTRY_INLINE_CAP, "the whole entry lands under the cap"


def test_phantom_tool_calls_never_render_from_offloaded_traces() -> None:
    """Adversary P2-1 fold: a bounded trace whose tool_calls subtree became
    an `$artifact` ref must not extract as a phantom call (count 1, name
    None) — consumers skip refs and say so."""
    from abstractruntime.rendering.agent_trace_report import render_agent_trace_markdown

    scratchpad = {
        "node_traces": {
            "agent": {
                "node_id": "agent",
                "steps": [
                    {
                        "ts": "2026-07-14T00:00:00+00:00",
                        "node_id": "agent",
                        "status": "completed",
                        "effect": {"type": "tool_calls", "payload": {"tool_calls": {"$artifact": "abc123"}}, "result_key": "r"},
                        "result": {"results": {"$artifact": "def456"}},
                        "trace_bounded": "offloaded",
                    }
                ],
            }
        }
    }
    report = render_agent_trace_markdown(scratchpad)
    assert "Tool: `None`" not in report and "Tool: None" not in report
    assert "offloaded to artifact" in report


def test_list_stalled_waits_empty_on_non_queryable_store() -> None:
    class _MinimalStore:
        def __init__(self):
            self._runs = {}

        def save(self, run):
            self._runs[run.run_id] = run

        def load(self, run_id):
            return self._runs.get(run_id)

    def ok(run, effect, default_next_node=None):
        return EffectOutcome.completed({"content": "hi"})

    rt = Runtime(
        run_store=_MinimalStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: ok},
    )
    assert rt.list_stalled_waits() == []
