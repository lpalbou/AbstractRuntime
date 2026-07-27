from __future__ import annotations

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryArtifactStore,
    InMemoryLedgerStore,
    InMemoryRunStore,
    RUN_HISTORY_BUNDLE_VERSION_V1,
    RunState,
    RunStatus,
    Runtime,
    StepPlan,
    WorkflowSpec,
    export_run_history_bundle,
    persist_workflow_snapshot,
)
from abstractruntime.core.models import StepRecord, StepStatus
from abstractruntime.storage.artifacts import build_artifact_descriptor_payload


pytestmark = pytest.mark.basic


class IndexedOnlyRunStore(InMemoryRunStore):
    def list_runs(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("history_bundle should use list_run_index for session turns when available")


def test_persist_workflow_snapshot_persists_ref_and_artifact() -> None:
    run_store = InMemoryRunStore()
    artifact_store = InMemoryArtifactStore()

    wf = WorkflowSpec(workflow_id="wf", entry_node="done", nodes={"done": lambda run, ctx: StepPlan(node_id="done", complete_output={"ok": True})})
    rt = Runtime(run_store=run_store, ledger_store=InMemoryLedgerStore(), effect_handlers={})
    run_id = rt.start(workflow=wf, vars={"context": {"messages": []}})

    ref = persist_workflow_snapshot(
        run_store=run_store,
        artifact_store=artifact_store,
        run_id=run_id,
        workflow_id="wf",
        snapshot={"kind": "unit", "wf": "wf"},
        format="unit_json",
    )

    assert isinstance(ref, dict)
    assert ref.get("workflow_id") == "wf"
    assert ref.get("format") == "unit_json"
    assert isinstance(ref.get("sha256"), str) and ref["sha256"]
    assert isinstance(ref.get("artifact_id"), str) and ref["artifact_id"]

    reloaded = run_store.load(run_id)
    assert reloaded is not None
    runtime_ns = (reloaded.vars or {}).get("_runtime") if isinstance(reloaded.vars, dict) else None
    assert isinstance(runtime_ns, dict)
    assert runtime_ns.get("workflow_snapshot", {}).get("artifact_id") == ref["artifact_id"]

    stored = artifact_store.load_json(ref["artifact_id"])
    assert stored == {"kind": "unit", "wf": "wf"}


def test_export_run_history_bundle_filters_input_data_and_tails_ledgers() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    artifact_store = InMemoryArtifactStore()

    run_id = "run_1"
    run_store.save(
        RunState(
            run_id=run_id,
            workflow_id="wf",
            status=RunStatus.COMPLETED,
            current_node="done",
            vars={
                "prompt": "hi",
                "context": {"messages": [{"role": "user", "content": "hi"}], "attachments": []},
                "_runtime": {"secret": True},
            },
            output={"response": "ok"},
            error=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            actor_id="tester",
            session_id="sess1",
            parent_run_id=None,
            waiting=None,
        )
    )
    for i in range(5):
        ledger_store.append(
            StepRecord(
                run_id=run_id,
                step_id=f"s{i+1}",
                node_id="n",
                status=StepStatus.COMPLETED,
                effect={"type": "llm_call", "payload": {"i": i + 1}, "result_key": None},
                result={
                    "ok": True,
                    "i": i + 1,
                    "metadata": (
                        {
                            "_runtime_resolved_action": {
                                "kind": "generate",
                                "action_id": "generate_text",
                                "modality": "text",
                                "task": "text_generation",
                            }
                        }
                        if i == 4
                        else {}
                    ),
                },
                error=None,
                started_at=f"2026-01-01T00:00:0{i}+00:00",
                ended_at=f"2026-01-01T00:00:0{i}+00:00",
                actor_id="tester",
                session_id="sess1",
                attempt=1,
                idempotency_key=None,
                prev_hash=None,
                record_hash=None,
                signature=None,
            )
        )

    bundle = export_run_history_bundle(
        run_id=run_id,
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        include_subruns=False,
        include_session=True,
        session_turn_limit=50,
        ledger_mode="tail",
        ledger_max_items=2,
    )

    assert bundle.get("version") == RUN_HISTORY_BUNDLE_VERSION_V1
    assert bundle.get("root_run_id") == run_id
    assert bundle.get("run", {}).get("run_id") == run_id
    assert bundle.get("run", {}).get("status") in {RunStatus.RUNNING.value, RunStatus.COMPLETED.value, RunStatus.WAITING.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}

    # Input data: private namespaces are filtered out.
    input_data = bundle.get("input_data") or {}
    assert "_runtime" not in input_data
    assert input_data.get("prompt") == "hi"

    ledgers = bundle.get("ledgers") or {}
    assert run_id in ledgers
    ledger = ledgers[run_id]
    assert ledger.get("total") >= 1
    assert ledger.get("cursor_start") == 4
    assert len(ledger.get("items") or []) <= 2
    items = ledger.get("items") or []
    if len(items) == 2:
        assert items[0]["cursor"] == ledger["cursor_start"]
        assert items[1]["cursor"] == ledger["cursor_start"] + 1

    # Session turns: should include the run (chat-like vars shape).
    turns = (bundle.get("session") or {}).get("turns") or []
    assert any(t.get("run_id") == run_id and t.get("kind") == "chat" for t in turns)
    turn0 = next((t for t in turns if t.get("run_id") == run_id), None)
    assert turn0 is not None
    assert turn0.get("answer") == "ok"
    resolved_actions = bundle.get("resolved_actions") or []
    assert resolved_actions[-1]["action_id"] == "generate_text"
    assert resolved_actions[-1]["run_id"] == run_id
    stats = turn0.get("stats") or {}
    assert isinstance(stats, dict)
    assert stats.get("llm_calls") == 5
    assert stats.get("tool_calls") == 0
    assert stats.get("duration_ms") == 4000


def test_export_run_history_bundle_uses_indexed_session_turns_and_replay_artifacts() -> None:
    run_store = IndexedOnlyRunStore()
    ledger_store = InMemoryLedgerStore()
    artifact_store = InMemoryArtifactStore()

    run_id = "run_indexed"
    run_store.save(
        RunState(
            run_id=run_id,
            workflow_id="wf_music",
            status=RunStatus.COMPLETED,
            current_node="music_generation",
            vars={
                "context": {"messages": [{"role": "user", "content": "make a calm synth loop"}]},
            },
            output={"response": "created music"},
            error=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:10+00:00",
            actor_id="tester",
            session_id="sess-index",
            parent_run_id=None,
            waiting=None,
        )
    )
    descriptor, metadata = build_artifact_descriptor_payload(
        semantic_kind="music",
        render_kind="audio",
        modality="music",
        task="music_generation",
        session_id="sess-index",
        workflow_id="wf_music",
        node_id="music_generation",
        turn_id="turn-1",
        run_id=run_id,
        producer={"provider": "unit-provider", "model": "music-1"},
        generation={"prompt": "make a calm synth loop", "params": {"duration_seconds": 30}},
        media={"kind": "audio", "duration_seconds": 30.0, "sample_rate": 44100},
    )
    artifact_store.store(
        b"not really audio",
        content_type="audio/wav",
        run_id=run_id,
        artifact_id="artifact_music",
        tags={"source": "unit"},
        metadata=metadata,
        descriptor=descriptor,
    )

    bundle = export_run_history_bundle(
        run_id=run_id,
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        include_subruns=False,
        include_session=True,
        session_turn_limit=20,
        ledger_mode="tail",
        ledger_max_items=20,
    )

    ledger_artifacts = (bundle.get("ledgers") or {}).get(run_id, {}).get("artifacts") or []
    assert len(ledger_artifacts) == 1
    assert ledger_artifacts[0]["artifact_id"] == "artifact_music"
    assert ledger_artifacts[0]["descriptor"]["semantic_kind"] == "music"
    assert ledger_artifacts[0]["descriptor"]["render_kind"] == "audio"
    assert ledger_artifacts[0]["descriptor"]["producer"]["provider"] == "unit-provider"
    assert ledger_artifacts[0]["descriptor"]["generation"]["prompt"] == "make a calm synth loop"
    assert ledger_artifacts[0]["descriptor"]["media"]["duration_seconds"] == 30.0

    turns = (bundle.get("session") or {}).get("turns") or []
    assert [t.get("run_id") for t in turns] == [run_id]
    assert turns[0]["prompt"] == "make a calm synth loop"
    assert turns[0]["answer"] == "created music"
    assert turns[0]["artifacts"][0]["artifact_id"] == "artifact_music"


def test_export_run_history_bundle_keeps_replay_artifacts_beyond_old_compact_limit() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    artifact_store = InMemoryArtifactStore()

    run_id = "run_many_artifacts"
    run_store.save(
        RunState(
            run_id=run_id,
            workflow_id="wf_many_media",
            status=RunStatus.WAITING,
            current_node="ask",
            vars={"context": {"messages": [{"role": "user", "content": "generate a media sequence"}]}},
            output={},
            error=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:01:00+00:00",
            actor_id="tester",
            session_id="sess-many-artifacts",
            parent_run_id=None,
            waiting=None,
        )
    )
    for i in range(60):
        is_audio = i % 2 == 1
        artifact_store.store(
            f"payload-{i}".encode("utf-8"),
            content_type="audio/wav" if is_audio else "image/png",
            run_id=run_id,
            artifact_id=f"artifact_{i:02d}",
            tags={
                "kind": "generated_media",
                "modality": "voice" if is_audio else "image",
                "task": "tts" if is_audio else "image_generation",
                "workflow_id": "wf_many_media",
                "node_id": "node-4" if is_audio else "node-3",
                "session_id": "sess-many-artifacts",
            },
        )

    bundle = export_run_history_bundle(
        run_id=run_id,
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        include_subruns=False,
        include_session=True,
        session_turn_limit=10,
        ledger_mode="tail",
        ledger_max_items=20,
    )

    ledger_artifacts = (bundle.get("ledgers") or {}).get(run_id, {}).get("artifacts") or []
    turn_artifacts = ((bundle.get("session") or {}).get("turns") or [{}])[0].get("artifacts") or []
    assert len(ledger_artifacts) == 60
    assert len(turn_artifacts) == 60
    assert {a.get("artifact_id") for a in turn_artifacts} >= {"artifact_00", "artifact_08", "artifact_59"}


def test_export_run_history_bundle_strips_runtime_metadata_from_session_turn_prompt() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    artifact_store = InMemoryArtifactStore()

    run_id = "run_with_grounded_prompt"
    prompt = (
        '<runtime_metadata>{"country":"FR","display":"[2026-05-24 20:21:42 FR]",'
        '"local_datetime":"2026-05-24T20:21:42+02:00","timezone":"Europe/Paris","user":"albou"}</runtime_metadata>\n'
        "who are you ?"
    )
    run_store.save(
        RunState(
            run_id=run_id,
            workflow_id="dialogue@dev:044df8e2",
            status=RunStatus.WAITING,
            current_node="node-2",
            vars={"context": {"messages": [{"role": "user", "content": prompt}]}},
            output={},
            error=None,
            created_at="2026-05-24T18:21:42+00:00",
            updated_at="2026-05-24T18:21:42+00:00",
            actor_id="tester",
            session_id="sess-runtime-meta",
            parent_run_id=None,
            waiting=None,
        )
    )

    bundle = export_run_history_bundle(
        run_id=run_id,
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        include_subruns=False,
        include_session=True,
        session_turn_limit=10,
        ledger_mode="tail",
        ledger_max_items=20,
    )

    turns = (bundle.get("session") or {}).get("turns") or []
    assert len(turns) == 1
    assert turns[0]["prompt"] == "who are you ?"
    assert turns[0]["prompt_metadata"]["country"] == "FR"
    assert turns[0]["prompt_metadata"]["timezone"] == "Europe/Paris"
    assert turns[0]["prompt_metadata"]["user"] == "albou"


def test_export_run_history_bundle_replay_uses_latest_user_message_and_excludes_future_turns() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    artifact_store = InMemoryArtifactStore()

    prior_run_id = "run_prior_chat"
    selected_run_id = "run_waiting_selected"
    future_run_id = "run_cancelled_future"
    selected_prompt = (
        '<runtime_metadata>{"country":"FR","display":"[2026-05-24 20:21:42 FR]",'
        '"timezone":"Europe/Paris","user":"albou"}</runtime_metadata>\n'
        "what should happen now?"
    )
    run_store.save(
        RunState(
            run_id=selected_run_id,
            workflow_id="dialogue@dev:044df8e2",
            status=RunStatus.WAITING,
            current_node="node-2",
            vars={
                "context": {
                    "messages": [
                        {"role": "user", "content": "who are you ?"},
                        {"role": "assistant", "content": "I am an assistant."},
                        {"role": "user", "content": selected_prompt},
                    ]
                }
            },
            output={},
            error=None,
            created_at="2026-05-24T18:21:42+00:00",
            updated_at="2026-05-24T18:21:42+00:00",
            actor_id="tester",
            session_id="sess-replay-bound",
            parent_run_id=None,
            waiting=None,
        )
    )
    run_store.save(
        RunState(
            run_id=future_run_id,
            workflow_id="dialogue@dev:044df8e2",
            status=RunStatus.CANCELLED,
            current_node="node-2",
            vars={"context": {"messages": [{"role": "user", "content": "future duplicated question"}]}},
            output={},
            error=None,
            created_at="2026-05-24T18:25:00+00:00",
            updated_at="2026-05-24T18:25:01+00:00",
            actor_id="tester",
            session_id="sess-replay-bound",
            parent_run_id=None,
            waiting=None,
        )
    )
    # Saved after the selected/future runs on purpose: replay order must be
    # chronological, not insertion order, and must support `Z` timestamps.
    run_store.save(
        RunState(
            run_id=prior_run_id,
            workflow_id="dialogue@dev:044df8e2",
            status=RunStatus.COMPLETED,
            current_node="node-2",
            vars={"context": {"messages": [{"role": "user", "content": "earlier question"}]}},
            output={"response": "earlier answer"},
            error=None,
            created_at="2026-05-24T18:10:00Z",
            updated_at="2026-05-24T18:10:03Z",
            actor_id="tester",
            session_id="sess-replay-bound",
            parent_run_id=None,
            waiting=None,
        )
    )

    bundle = export_run_history_bundle(
        run_id=selected_run_id,
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        include_subruns=False,
        include_session=True,
        session_turn_limit=10,
        ledger_mode="tail",
        ledger_max_items=20,
    )

    turns = (bundle.get("session") or {}).get("turns") or []
    assert [t.get("run_id") for t in turns] == [prior_run_id, selected_run_id]
    assert [t.get("prompt") for t in turns] == ["earlier question", "what should happen now?"]
    assert turns[1]["prompt_metadata"]["timezone"] == "Europe/Paris"


def test_resume_appends_resume_record_to_ledger() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()

    def start(run, ctx) -> StepPlan:
        return StepPlan(
            node_id="start",
            effect=Effect(type=EffectType.ASK_USER, payload={"prompt": "name?"}, result_key="answer"),
            next_node="done",
        )

    def done(run, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"answer": run.vars.get("answer")})

    wf = WorkflowSpec(workflow_id="wf_wait", entry_node="start", nodes={"start": start, "done": done})
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, effect_handlers={})

    run_id = rt.start(workflow=wf, vars={})
    st = rt.tick(workflow=wf, run_id=run_id, max_steps=5)
    assert st.status == RunStatus.WAITING

    rt.resume(workflow=wf, run_id=run_id, wait_key=st.waiting.wait_key if st.waiting else None, payload={"response": "alice"}, max_steps=5)

    records = ledger_store.list(run_id)
    assert any(isinstance(r, dict) and isinstance(r.get("effect"), dict) and r["effect"].get("type") == "resume" for r in records)


# ---------------------------------------------------------------------------
# Replay profile + in-band warnings (replay-integrity audit, 2026-07-25)
# ---------------------------------------------------------------------------


def _mk_run(run_store, run_id: str = "run_rp", session_id: str = "sess_rp") -> None:
    run_store.save(
        RunState(
            run_id=run_id,
            workflow_id="wf",
            status=RunStatus.COMPLETED,
            current_node="done",
            vars={"prompt": "hi"},
            output={"response": "ok"},
            error=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            actor_id="tester",
            session_id=session_id,
            parent_run_id=None,
            waiting=None,
        )
    )


def _mk_llm_record(run_id: str, step_id: str) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        step_id=step_id,
        node_id="reason",
        status=StepStatus.COMPLETED,
        effect={
            "type": "llm_call",
            "payload": {
                "messages": [{"role": "user", "content": "u" * 500}],
                "system_prompt": "s" * 400,
                "prompt": "p" * 100,
                "temperature": 0.2,
            },
            "result_key": None,
        },
        result={
            "content": "the answer",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "metadata": {
                "_runtime_observability": {"llm_generate_kwargs": {"messages": [{"role": "user", "content": "u" * 500}]}},
                "_provider_request": {"payload": {"messages": [{"role": "user", "content": "u" * 500}]}},
                "model": "m1",
            },
        },
        error=None,
        started_at="2026-01-01T00:00:01+00:00",
        ended_at="2026-01-01T00:00:02+00:00",
        actor_id="tester",
        session_id="sess_rp",
        attempt=1,
        idempotency_key=None,
        prev_hash=None,
        record_hash=None,
        signature=None,
    )


def test_replay_profile_drops_request_side_with_omitted_markers() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    _mk_run(run_store)
    ledger_store.append(_mk_llm_record("run_rp", "s1"))

    bundle = export_run_history_bundle(
        run_id="run_rp", run_store=run_store, ledger_store=ledger_store, detail="replay"
    )
    assert bundle["detail"] == "replay"
    rec = bundle["ledgers"]["run_rp"]["items"][0]["record"]
    payload = rec["effect"]["payload"]
    for f in ("messages", "system_prompt", "prompt"):
        marker = payload[f]
        assert isinstance(marker, dict) and "$omitted" in marker, f"payload.{f} must carry an $omitted marker"
        assert marker["$omitted"]["profile"] == "replay"
        assert isinstance(marker["$omitted"]["bytes"], int) and marker["$omitted"]["bytes"] > 0
    # Non-request fields stay.
    assert payload["temperature"] == 0.2
    md = rec["result"]["metadata"]
    assert "$omitted" in md["_runtime_observability"]
    assert "$omitted" in md["_provider_request"]
    # Transcript-read fields are untouched.
    assert rec["result"]["content"] == "the answer"
    assert rec["result"]["usage"]["input_tokens"] == 10
    assert md["model"] == "m1"
    # Timeline skipped in the projection (never read by a transcript fold).
    assert bundle["timeline"] == []


def test_replay_profile_never_mutates_the_stored_records() -> None:
    """Projection is spine-copy only — the ledger store's objects stay intact.

    InMemoryLedgerStore shares record objects with the caller; a projection
    that mutated them would poison every LATER full-detail export (the
    JsonFileRunStore memo-hygiene class).
    """
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    _mk_run(run_store)
    ledger_store.append(_mk_llm_record("run_rp", "s1"))

    export_run_history_bundle(run_id="run_rp", run_store=run_store, ledger_store=ledger_store, detail="replay")
    full = export_run_history_bundle(run_id="run_rp", run_store=run_store, ledger_store=ledger_store)
    assert full["detail"] == "full"
    payload = full["ledgers"]["run_rp"]["items"][0]["record"]["effect"]["payload"]
    assert payload["messages"] == [{"role": "user", "content": "u" * 500}]
    assert payload["system_prompt"] == "s" * 400
    md = full["ledgers"]["run_rp"]["items"][0]["record"]["result"]["metadata"]
    assert "llm_generate_kwargs" in md["_runtime_observability"]


def test_replay_profile_rejects_unknown_detail() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    _mk_run(run_store)
    with pytest.raises(ValueError):
        export_run_history_bundle(
            run_id="run_rp", run_store=run_store, ledger_store=ledger_store, detail="compact"
        )


def test_bundle_warnings_are_in_band_never_silent() -> None:
    """Every degradation the export survives must be visible in the bundle.

    Operator ruling (2026-07-25): silent omission is the class that broke
    replay — a bundle that cannot be complete must say so in-band.
    """
    run_store = InMemoryRunStore()

    class FailingLedgerStore(InMemoryLedgerStore):
        def list(self, run_id):  # type: ignore[override]
            raise RuntimeError("disk went away")

    _mk_run(run_store)
    bundle = export_run_history_bundle(
        run_id="run_rp", run_store=run_store, ledger_store=FailingLedgerStore()
    )
    codes = [w["code"] for w in bundle["warnings"]]
    assert "ledger_read_failed" in codes
    w = next(w for w in bundle["warnings"] if w["code"] == "ledger_read_failed")
    assert w["run_id"] == "run_rp"
    assert "disk went away" in w["detail"]


def test_bundle_tail_window_truncation_is_reported() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    _mk_run(run_store)
    for i in range(7):
        ledger_store.append(_mk_llm_record("run_rp", f"s{i}"))

    bundle = export_run_history_bundle(
        run_id="run_rp",
        run_store=run_store,
        ledger_store=ledger_store,
        ledger_mode="tail",
        ledger_max_items=3,
    )
    codes = [w["code"] for w in bundle["warnings"]]
    assert "ledger_tail_window" in codes
    w = next(w for w in bundle["warnings"] if w["code"] == "ledger_tail_window")
    assert w["total"] == 7 and w["window"] == 3
    # Untruncated exports carry NO tail warning.
    bundle_full = export_run_history_bundle(
        run_id="run_rp", run_store=run_store, ledger_store=ledger_store, ledger_mode="full"
    )
    assert all(w["code"] != "ledger_tail_window" for w in bundle_full["warnings"])


def test_bundle_subtree_discovery_failure_is_reported() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    _mk_run(run_store)

    def _boom(**kwargs):
        raise RuntimeError("children index corrupt")

    run_store.list_children = _boom  # type: ignore[attr-defined]
    bundle = export_run_history_bundle(run_id="run_rp", run_store=run_store, ledger_store=ledger_store)
    codes = [w["code"] for w in bundle["warnings"]]
    assert "subtree_discovery_failed" in codes


def test_replay_profile_preserves_every_field_the_fold_reads() -> None:
    """Pin detail=replay against code-tui's read-field list (commons c5556).

    The fold reads: envelope (run_id/node_id/status/started_at/step_id),
    effect.type, payload.tool_calls on tool_calls records, subworkflow
    binding fields, result.content/reasoning/usage/wait/output/results.
    Everything on that list must survive the projection VERBATIM.
    """
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    _mk_run(run_store)

    tool_calls = [{"name": "read_file", "call_id": "c1", "arguments": {"path": "a.txt"}}]
    ledger_store.append(
        StepRecord(
            run_id="run_rp",
            step_id="t1",
            node_id="act",
            status=StepStatus.COMPLETED,
            effect={"type": "tool_calls", "payload": {"tool_calls": tool_calls}, "result_key": None},
            result={
                "results": [{"name": "read_file", "call_id": "c1", "success": True, "output": "text"}],
                "wait": {"reason": "user", "wait_key": "w1", "prompt": "approve?", "details": {"mode": "approval_required"}},
                "reasoning": "cycle gist",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
            error=None,
            started_at="2026-01-01T00:00:03+00:00",
            ended_at="2026-01-01T00:00:04+00:00",
            actor_id="tester",
            session_id="sess_rp",
            attempt=1,
            idempotency_key=None,
            prev_hash=None,
            record_hash=None,
            signature=None,
        )
    )
    ledger_store.append(
        StepRecord(
            run_id="run_rp",
            step_id="sw1",
            node_id="agent",
            status=StepStatus.COMPLETED,
            effect={
                "type": "start_subworkflow",
                "payload": {"sub_workflow_id": "child-wf", "wrap_as_tool_result": True},
                "result_key": None,
            },
            result={"output": {"answer": "done"}},
            error=None,
            started_at="2026-01-01T00:00:05+00:00",
            ended_at="2026-01-01T00:00:06+00:00",
            actor_id="tester",
            session_id="sess_rp",
            attempt=1,
            idempotency_key=None,
            prev_hash=None,
            record_hash=None,
            signature=None,
        )
    )

    bundle = export_run_history_bundle(
        run_id="run_rp", run_store=run_store, ledger_store=ledger_store, detail="replay"
    )
    items = bundle["ledgers"]["run_rp"]["items"]
    recs = {it["record"]["step_id"]: it["record"] for it in items}

    tc = recs["t1"]
    assert tc["effect"]["type"] == "tool_calls"
    assert tc["effect"]["payload"]["tool_calls"] == tool_calls, "tool cards lose names/args if this drops"
    assert tc["result"]["results"][0]["output"] == "text"
    assert tc["result"]["wait"]["prompt"] == "approve?"
    assert tc["result"]["reasoning"] == "cycle gist"
    assert tc["result"]["usage"]["input_tokens"] == 3
    assert tc["node_id"] == "act" and tc["started_at"]

    sw = recs["sw1"]
    assert sw["effect"]["payload"]["sub_workflow_id"] == "child-wf"
    assert sw["effect"]["payload"]["wrap_as_tool_result"] is True
    assert sw["result"]["output"]["answer"] == "done"
