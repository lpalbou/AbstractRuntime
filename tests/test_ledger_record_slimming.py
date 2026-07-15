"""Terminal-record slimming + offload wiring (backlog 0067-M).

The ledger used to persist the same conversation-sized bytes 3-4x per LLM
effect: STARTED payload, the terminal re-append's payload, and the result's
observability copies (`_provider_request`, `_runtime_observability`). These
tests pin the slimming discipline end to end:

- oversized terminal payload fields become verified `$slim` markers; the
  STARTED record keeps the bytes; sub-threshold fields stay inline;
- result observability copies dedup ONLY when provably reconstructable
  (byte-verified); decorated wire bytes stay verbatim (B3 fidelity);
- crash-replay reuse rehydrates markers so vars stay byte-identical to the
  live path;
- history reconstruction resolves slimmed payloads;
- the run-vars copy of a result is never touched (spine-copy discipline);
- durable factories + the entity runtime write through OffloadingLedgerStore
  and reads come back rehydrated.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pytest

from abstractruntime.core.models import (
    Effect,
    EffectType,
    RunState,
    RunStatus,
    StepPlan,
    StepRecord,
    StepStatus,
)
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.ledger_slim import (
    SLIM_FIELD_THRESHOLD_BYTES,
    SLIM_MARKER_KEY,
    build_started_payload_index,
    capture_started_payload_digests,
    contains_slim_marker,
    is_slim_marker,
    resolve_slim_tree,
    resolve_slim_value,
    slim_result_metadata,
    slim_terminal_effect,
)


def _slim_effect(effect, step_id):
    """Mirror the runtime flow: digests captured at STARTED-append time."""
    return slim_terminal_effect(
        effect, step_id=step_id, started_digests=capture_started_payload_digests(effect)
    )

BIG = "x" * (SLIM_FIELD_THRESHOLD_BYTES + 512)
BIG_MESSAGES = [{"role": "user", "content": BIG}, {"role": "assistant", "content": "ok"}]


# ---------------------------------------------------------------------------
# Unit: slim_terminal_effect
# ---------------------------------------------------------------------------


def test_oversized_payload_field_becomes_a_verified_marker() -> None:
    effect = {"type": "llm_call", "payload": {"messages": BIG_MESSAGES, "temperature": 0.2}, "result_key": "out"}
    slimmed = _slim_effect(effect, step_id="step-1")

    assert slimmed is not effect, "oversized field must produce a spine copy"
    assert effect["payload"]["messages"] is BIG_MESSAGES, "original object untouched"
    marker = slimmed["payload"]["messages"]
    assert is_slim_marker(marker)
    body = marker[SLIM_MARKER_KEY]
    assert body["step_id"] == "step-1"
    assert body["field"] == "messages"
    assert slimmed["payload"]["temperature"] == 0.2, "sub-threshold fields stay inline"

    index = {"step-1": effect["payload"]}
    assert resolve_slim_value(marker, index) == BIG_MESSAGES


def test_small_payload_is_returned_unchanged() -> None:
    effect = {"type": "tool_calls", "payload": {"tool_calls": [{"name": "read_file"}]}, "result_key": "r"}
    assert _slim_effect(effect, step_id="s") is effect


def test_resolution_refuses_a_tampered_source() -> None:
    effect = {"type": "llm_call", "payload": {"messages": BIG_MESSAGES}, "result_key": None}
    slimmed = _slim_effect(effect, step_id="step-t")
    marker = slimmed["payload"]["messages"]

    tampered_index = {"step-t": {"messages": [{"role": "user", "content": "different"}]}}
    resolved = resolve_slim_value(marker, tampered_index)
    assert is_slim_marker(resolved), "sha mismatch must keep the marker, never a wrong payload"


# ---------------------------------------------------------------------------
# Unit: slim_result_metadata
# ---------------------------------------------------------------------------


def _llm_result_with_copies(payload: Dict[str, Any], *, provider_messages: Any) -> Dict[str, Any]:
    return {
        "content": "the answer",
        "metadata": {
            "_runtime_observability": {
                "llm_generate_kwargs": {
                    "prompt": "",
                    "messages": json.loads(json.dumps(payload["messages"])),
                    "params": {"temperature": 0.1},
                }
            },
            "_provider_request": {
                "transport": "local",
                "payload": {"messages": provider_messages, "params": {}},
            },
        },
    }


def test_observability_kwargs_dedup_when_byte_identical() -> None:
    payload = {"messages": BIG_MESSAGES, "system_prompt": "be brief"}
    result = _llm_result_with_copies(payload, provider_messages=[{"role": "user", "content": "decorated: " + BIG}])

    slimmed = slim_result_metadata(result, effect_payload=payload, step_id="s9", started_digests=capture_started_payload_digests({"payload": payload}))
    assert slimmed is not result

    kwargs = slimmed["metadata"]["_runtime_observability"]["llm_generate_kwargs"]
    assert is_slim_marker(kwargs["messages"])
    # Decorated provider bytes differ from the payload -> kept verbatim (B3).
    preq_msgs = slimmed["metadata"]["_provider_request"]["payload"]["messages"]
    assert not is_slim_marker(preq_msgs)
    assert preq_msgs[0]["content"].startswith("decorated: ")

    # The ORIGINAL result object is untouched (it flows into run vars).
    assert result["metadata"]["_runtime_observability"]["llm_generate_kwargs"]["messages"] == BIG_MESSAGES


def test_fabricated_provider_request_dedups_via_layout() -> None:
    payload = {"messages": BIG_MESSAGES, "system_prompt": "be brief", "prompt": ""}
    fabricated = [{"role": "system", "content": "be brief"}] + [dict(m) for m in BIG_MESSAGES]
    result = _llm_result_with_copies(payload, provider_messages=fabricated)

    slimmed = slim_result_metadata(result, effect_payload=payload, step_id="s10", started_digests=capture_started_payload_digests({"payload": payload}))
    marker = slimmed["metadata"]["_provider_request"]["payload"]["messages"]
    assert is_slim_marker(marker)
    assert marker[SLIM_MARKER_KEY]["kind"] == "started_messages_layout"

    index = {"s10": payload}
    assert resolve_slim_tree(slimmed, index)["metadata"]["_provider_request"]["payload"]["messages"] == fabricated


# ---------------------------------------------------------------------------
# Runtime integration: terminal appends slim, replay rehydrates
# ---------------------------------------------------------------------------


def _one_effect_workflow(payload: Dict[str, Any], *, effect_type: EffectType = EffectType.LLM_CALL) -> WorkflowSpec:
    def node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("out") is not None:
            return StepPlan(node_id="n1", complete_output={"ok": True})
        return StepPlan(
            node_id="n1",
            effect=Effect(type=effect_type, payload=payload, result_key="out"),
            next_node="n1",
        )

    return WorkflowSpec(workflow_id="wf-slim", entry_node="n1", nodes={"n1": node})


def _make_runtime(handler_result: Dict[str, Any]):
    run_store = InMemoryRunStore()
    ledger = InMemoryLedgerStore()
    calls = {"n": 0}

    def handler(run: RunState, effect: Effect, default_next_node: Optional[str] = None):
        from abstractruntime.core.runtime import EffectOutcome

        calls["n"] += 1
        return EffectOutcome.completed(json.loads(json.dumps(handler_result)))

    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger,
        effect_handlers={EffectType.LLM_CALL: handler},
    )
    return rt, run_store, ledger, calls


def test_terminal_record_is_slim_and_started_keeps_the_bytes() -> None:
    payload = {"messages": BIG_MESSAGES, "temperature": 0.0}
    result = {"content": "hi", "metadata": {}}
    rt, run_store, ledger, _calls = _make_runtime(result)
    wf = _one_effect_workflow(payload)

    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)

    records = ledger.list(run_id)
    started = [r for r in records if r.get("status") == "started" and (r.get("effect") or {}).get("type") == "llm_call"]
    completed = [r for r in records if r.get("status") == "completed" and (r.get("effect") or {}).get("type") == "llm_call"]
    assert started and completed

    # The runtime may decorate the payload (grounding envelope message), so
    # the STARTED record is the byte truth the marker must resolve back to.
    started_messages = started[0]["effect"]["payload"]["messages"]
    assert BIG_MESSAGES[0] in started_messages, "STARTED keeps the original bytes"
    marker = completed[0]["effect"]["payload"]["messages"]
    assert is_slim_marker(marker), "terminal re-append drops the duplicate"
    assert completed[0]["effect"]["payload"]["temperature"] == 0.0

    index = build_started_payload_index(records)
    assert resolve_slim_value(marker, index) == started_messages

    # Ledger bytes: the duplicate conversation is gone from the terminal record.
    started_bytes = len(json.dumps(started[0]))
    completed_bytes = len(json.dumps(completed[0]))
    assert completed_bytes < started_bytes / 4


def test_crash_replay_reuses_a_rehydrated_result() -> None:
    payload = {"messages": BIG_MESSAGES, "system_prompt": "be brief"}
    fat_result = _llm_result_with_copies(payload, provider_messages=[dict(m) for m in BIG_MESSAGES])
    rt, run_store, ledger, calls = _make_runtime(fat_result)
    wf = _one_effect_workflow(payload)

    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)
    assert calls["n"] == 1
    live_vars_result = run_store.load(run_id).vars["out"]
    assert live_vars_result["metadata"]["_runtime_observability"]["llm_generate_kwargs"]["messages"] == BIG_MESSAGES

    # Simulate the crash window: effect completed + terminal record landed,
    # but the run-state save did not. Replay must reuse (not re-execute) and
    # the reused result must be byte-identical to the live path.
    crashed = run_store.load(run_id)
    crashed.vars.pop("out", None)
    crashed.current_node = "n1"
    from abstractruntime.core.models import RunStatus

    crashed.status = RunStatus.RUNNING
    crashed.output = None
    seq_before = ((crashed.vars.get("_runtime") or {}).get("effect_seq"))
    runtime_ns = crashed.vars.setdefault("_runtime", {})
    if isinstance(seq_before, int) and seq_before > 0:
        runtime_ns["effect_seq"] = seq_before - 1
    run_store.save(crashed)

    rt.tick(workflow=wf, run_id=run_id)
    assert calls["n"] == 1, "crash-replay must reuse the completed result"

    replayed = run_store.load(run_id).vars["out"]
    assert replayed == live_vars_result, "rehydrated replay result must be byte-identical to the live path"
    assert not contains_slim_marker(replayed)


def test_history_bundle_resolves_slimmed_records() -> None:
    from abstractruntime.history_bundle import (
        _extract_flow_end_output_from_ledger,
        _extract_repl_stats_from_ledger,
    )

    big_calls = [{"name": "write_file", "arguments": {"content": BIG}}, {"name": "read_file", "arguments": {}}]
    started = StepRecord.start(
        run=RunState.new(workflow_id="wf", entry_node="n", vars={}),
        node_id="n",
        effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": big_calls}),
        idempotency_key="k1",
    )
    started_dict = json.loads(json.dumps({**started.__dict__, "status": "started"}, default=str))

    terminal = StepRecord(**{**started.__dict__})
    terminal.finish_success({"results": [{"name": "write_file", "success": True}, {"name": "read_file", "success": True}]})
    slimmed_effect = slim_terminal_effect(terminal.effect, step_id=terminal.step_id, started_digests=capture_started_payload_digests(terminal.effect))
    terminal_dict = json.loads(json.dumps({**terminal.__dict__, "effect": slimmed_effect, "status": "completed"}, default=str))
    assert is_slim_marker(terminal_dict["effect"]["payload"]["tool_calls"])

    stats = _extract_repl_stats_from_ledger([started_dict, terminal_dict])
    assert stats["tool_calls"] == 2, "stats must count through the marker"

    big_message = "answer " + BIG
    a_started = StepRecord.start(
        run=RunState.new(workflow_id="wf", entry_node="n", vars={}),
        node_id="answer",
        effect=Effect(type=EffectType.ANSWER_USER, payload={"message": big_message}),
        idempotency_key="k2",
    )
    a_started_dict = json.loads(json.dumps({**a_started.__dict__, "status": "started"}, default=str))
    a_terminal = StepRecord(**{**a_started.__dict__})
    a_terminal.finish_success({"delivered": True})
    a_slim = slim_terminal_effect(a_terminal.effect, step_id=a_terminal.step_id, started_digests=capture_started_payload_digests(a_terminal.effect))
    a_terminal_dict = json.loads(json.dumps({**a_terminal.__dict__, "effect": a_slim, "status": "completed"}, default=str))
    assert is_slim_marker(a_terminal_dict["effect"]["payload"]["message"])

    text, _meta = _extract_flow_end_output_from_ledger([a_started_dict, a_terminal_dict])
    assert text == big_message.strip()


# ---------------------------------------------------------------------------
# Offloading wiring
# ---------------------------------------------------------------------------


def test_offloading_reads_keep_refs_but_replay_rehydrates(tmp_path) -> None:
    """Read-shape contract (adversary P1-2/P2-1 fold): list() serves refs
    exactly as disk holds them (rehydrating every read measured 113x time /
    593x bytes on offload-heavy ledgers and broke chain verification);
    find_completed_result — the crash-replay path where byte-identity is a
    correctness requirement — rehydrates the offloader's own refs."""
    from abstractruntime.storage.artifacts import InMemoryArtifactStore, artifact_ref
    from abstractruntime.storage.offloading import OffloadingLedgerStore

    artifacts = InMemoryArtifactStore()
    store = OffloadingLedgerStore(InMemoryLedgerStore(), artifact_store=artifacts, max_inline_bytes=64)

    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    rec = StepRecord.start(
        run=run,
        node_id="n",
        effect=Effect(type=EffectType.LLM_CALL, payload={"messages": [{"role": "user", "content": "y" * 500}]}),
        idempotency_key="k-off",
    )
    rec.finish_success({"content": "z" * 500, "media": [artifact_ref("artifact-authored-by-handler")]})
    store.append(rec)

    read_back = store.list(run.run_id)[0]
    raw = json.dumps(read_back)
    assert "$artifact" in raw, "oversized values are refs on the read surface"
    assert ("y" * 500) not in raw and ("z" * 500) not in raw

    found = store.find_completed_result(run.run_id, "k-off")
    assert found is not None and found["content"] == "z" * 500, "crash-replay reuse rehydrates"
    # Handler-authored refs are the caller's contract: never rehydrated.
    assert found["media"][0] == {"$artifact": "artifact-authored-by-handler"}


def test_durable_factory_composition_serves_subscribe_and_offloads(tmp_path) -> None:
    """End-to-end behavior pin (adversary P0: OffloadingLedgerStore
    satisfies the observable Protocol by METHOD PRESENCE, so the old
    presence check skipped the real Observable wrapper and subscribe_ledger
    raised on every durable deployment — silently killing host live-ledger
    features). The composition must SERVE subscriptions, notify with the
    pre-offload record, and still offload durable bytes."""
    from abstractruntime.integrations.abstractcore.factory import _compose_durable_ledger
    from abstractruntime.storage.artifacts import FileArtifactStore, InMemoryArtifactStore
    from abstractruntime.storage.json_files import JsonlLedgerStore
    from abstractruntime.storage.observable import ObservableLedgerStore
    from abstractruntime.storage.offloading import OffloadingLedgerStore

    durable = _compose_durable_ledger(JsonlLedgerStore(tmp_path), FileArtifactStore(tmp_path))
    assert isinstance(durable, ObservableLedgerStore)
    assert isinstance(durable._inner, OffloadingLedgerStore)

    seen: list = []
    durable.subscribe(lambda record: seen.append(record))

    run = RunState.new(workflow_id="wf", entry_node="n", vars={})
    rec = StepRecord.start(
        run=run,
        node_id="n",
        effect=Effect(type=EffectType.LLM_CALL, payload={"messages": [{"role": "user", "content": "s" * 500}]}),
        idempotency_key="k-sub",
    )
    durable.append(rec)
    assert len(seen) == 1, "subscribe_ledger must WORK on the durable composition"
    assert seen[0]["effect"]["payload"]["messages"][0]["content"] == "s" * 500, (
        "subscribers receive the record before offloading touches it"
    )

    # In-memory shapes skip offloading but still gain observability.
    memory = _compose_durable_ledger(InMemoryLedgerStore(), FileArtifactStore(tmp_path))
    assert isinstance(memory, ObservableLedgerStore)
    assert isinstance(memory._inner, InMemoryLedgerStore)

    mem_artifacts = _compose_durable_ledger(JsonlLedgerStore(tmp_path), InMemoryArtifactStore())
    assert isinstance(mem_artifacts, ObservableLedgerStore)
    assert isinstance(mem_artifacts._inner, JsonlLedgerStore), "memory artifacts gain nothing from refs"

    # A caller-supplied Offloading store is WRAPPED (never trusted for
    # observability); a genuine observable store is respected as-is.
    pre_offloaded = OffloadingLedgerStore(JsonlLedgerStore(tmp_path), artifact_store=FileArtifactStore(tmp_path))
    composed = _compose_durable_ledger(pre_offloaded, FileArtifactStore(tmp_path))
    assert isinstance(composed, ObservableLedgerStore)
    composed.subscribe(lambda record: None)
    already_observable = _compose_durable_ledger(composed, FileArtifactStore(tmp_path))
    assert already_observable is composed


def test_mutated_payload_field_keeps_verbatim_bytes(tmp_path) -> None:
    """Adversary P1-1: a handler mutating an oversized payload field IN
    PLACE between the STARTED and terminal appends must not produce an
    unresolvable marker — the terminal record is the only holder of the
    execution-time bytes and keeps them verbatim."""
    run_store = InMemoryRunStore()
    ledger = InMemoryLedgerStore()
    payload = {"blob": BIG, "note": "small"}

    def node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("out") is not None:
            return StepPlan(node_id="n1", complete_output={"ok": True})
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.TOOL_CALLS, payload=payload, result_key="out"),
            next_node="n1",
        )

    from abstractruntime.core.runtime import EffectOutcome

    def mutating_handler(run: RunState, effect: Effect, default_next_node: Any = None) -> EffectOutcome:
        effect.payload["blob"] = "MUTATED " + BIG  # contract violation, deliberately
        return EffectOutcome.completed({"results": []})

    wf = WorkflowSpec(workflow_id="wf-mut", entry_node="n1", nodes={"n1": node})
    rt = Runtime(run_store=run_store, ledger_store=ledger, effect_handlers={EffectType.TOOL_CALLS: mutating_handler})
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)

    records = ledger.list(run_id)
    completed = [r for r in records if str(getattr(r.get("status"), "value", r.get("status"))) == "completed" and r.get("effect")]
    terminal_blob = completed[0]["effect"]["payload"]["blob"]
    assert not is_slim_marker(terminal_blob), "mutated bytes must never become a marker"
    assert terminal_blob.startswith("MUTATED "), "the execution-time truth is preserved verbatim"


def test_replay_never_resolves_marker_shaped_data(tmp_path) -> None:
    """Adversary P2-2: a tool result that ECHOES a slimmed record carries
    valid-looking markers as DATA; crash-replay rehydration is targeted to
    the runtime's own metadata paths, so the echo survives byte-identical."""
    from abstractruntime.core.runtime import EffectOutcome

    run_store = InMemoryRunStore()
    ledger = InMemoryLedgerStore()
    payload = {"messages": BIG_MESSAGES}
    echoed_marker = {SLIM_MARKER_KEY: {"v": 1, "kind": "started_payload_field", "step_id": "someone-else", "field": "messages", "sha256": "0" * 64, "bytes": 5}}
    calls = {"n": 0}

    def node(run: RunState, ctx: Any) -> StepPlan:
        if run.vars.get("out") is not None:
            return StepPlan(node_id="n1", complete_output={"ok": True})
        return StepPlan(
            node_id="n1",
            effect=Effect(type=EffectType.LLM_CALL, payload=payload, result_key="out"),
            next_node="n1",
        )

    def handler(run: RunState, effect: Effect, default_next_node: Any = None) -> EffectOutcome:
        calls["n"] += 1
        return EffectOutcome.completed({"content": "ok", "echo": json.loads(json.dumps(echoed_marker))})

    wf = WorkflowSpec(workflow_id="wf-echo", entry_node="n1", nodes={"n1": node})
    rt = Runtime(run_store=run_store, ledger_store=ledger, effect_handlers={EffectType.LLM_CALL: handler})
    run_id = rt.start(workflow=wf)
    rt.tick(workflow=wf, run_id=run_id)
    live = run_store.load(run_id).vars["out"]
    assert live["echo"] == echoed_marker

    # Crash window: result landed, save didn't.
    crashed = run_store.load(run_id)
    crashed.vars.pop("out", None)
    crashed.current_node = "n1"
    crashed.status = RunStatus.RUNNING
    crashed.output = None
    runtime_ns = crashed.vars.setdefault("_runtime", {})
    seq_before = runtime_ns.get("effect_seq")
    if isinstance(seq_before, int) and seq_before > 0:
        runtime_ns["effect_seq"] = seq_before - 1
    run_store.save(crashed)

    rt.tick(workflow=wf, run_id=run_id)
    assert calls["n"] == 1, "replay must reuse, not re-execute"
    replayed = run_store.load(run_id).vars["out"]
    assert replayed == live, "marker-shaped DATA must survive replay untouched"


def test_entity_runtime_writes_through_offloading(tmp_path) -> None:
    pytest.importorskip("abstractmemory")
    import copy

    import yaml
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    from abstractruntime.identity.entity_runtime import open_entity_runtime
    from abstractruntime.storage.offloading import OffloadingLedgerStore

    home_dir = tmp_path / "entities" / "castor"
    home_dir.mkdir(parents=True, exist_ok=True)
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Castor"
    spark["spark"] = 1
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": "entity:castor@home-test"}), encoding="utf-8")
    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    journal = SQLiteJournal(home_dir / "memory.sqlite3")
    engram(MemorySystem(store=store, journal=journal), spark, owner_id="entity:castor@home-test")
    store.close()
    journal.close()

    ert = open_entity_runtime(home_dir)
    try:
        assert isinstance(ert.runtime._ledger_store, OffloadingLedgerStore)
        assert ert.runtime._ledger_store.inner is ert.ledger_store
    finally:
        ert.close()
