from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler, make_tool_calls_handler
from abstractruntime.integrations.abstractcore.session_attachments import (
    dedup_messages_view,
    execute_open_attachment,
    list_session_attachments,
    render_session_attachments_system_message,
    session_memory_owner_run_id,
)
from abstractruntime.storage.artifacts import FileArtifactStore, InMemoryArtifactStore


@pytest.mark.basic
def test_session_memory_owner_run_id_matches_gateway_behavior() -> None:
    assert session_memory_owner_run_id("s1") == "session_memory_s1"
    assert session_memory_owner_run_id("a-b_c") == "session_memory_a-b_c"

    rid = session_memory_owner_run_id("hello world")
    assert rid.startswith("session_memory_sha_")
    assert len(rid) == len("session_memory_sha_") + 32


@pytest.mark.basic
def test_session_attachments_index_rendering_is_bounded() -> None:
    entries = []
    for i in range(50):
        entries.append(
            {
                "handle": f"docs/file{i}.md",
                "artifact_id": f"aid{i}",
                "sha256": "a" * 64,
                "content_type": "text/markdown",
                "size_bytes": 123,
                "created_at": "2026-01-01T00:00:00+00:00",
                "tags": {},
            }
        )

    msg = render_session_attachments_system_message(entries, max_entries=20, max_chars=220)
    assert msg.startswith("Session attachments")
    assert len(msg) <= 220


@pytest.mark.basic
def test_dedup_messages_view_stubs_duplicate_read_file_outputs() -> None:
    tool_msg = (
        "[read_file]: File: docs/notes.txt (2 lines)\n\n"
        "1: hello\n"
        "2: world\n"
    )
    messages = [{"role": "tool", "content": tool_msg}, {"role": "tool", "content": tool_msg}]
    session_attachments = [{"handle": "docs/notes.txt", "artifact_id": "a1", "sha256": "b" * 64}]

    rewritten = dedup_messages_view(messages, session_attachments=session_attachments)
    assert rewritten[0]["content"] == tool_msg
    assert "(duplicate)" in rewritten[1]["content"]
    assert "open_attachment" in rewritten[1]["content"]


@pytest.mark.basic
def test_open_attachment_reads_bounded_ranges() -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"hello\nworld\nthird\n"
    sha = hashlib.sha256(content).hexdigest()
    meta = store.store(
        content,
        content_type="text/plain",
        run_id=rid,
        tags={"kind": "attachment", "path": "notes.txt", "filename": "notes.txt", "session_id": sid, "sha256": sha},
    )

    ok, out, err = execute_open_attachment(
        artifact_store=store,
        session_id=sid,
        artifact_id=meta.artifact_id,
        handle=None,
        expected_sha256=None,
        start_line=2,
        end_line=2,
        max_chars=2000,
    )
    assert ok is True
    assert err is None
    assert isinstance(out, dict)
    rendered = out.get("rendered")
    assert isinstance(rendered, str)
    assert "\n2: world" in rendered

    ok2, out2, _err2 = execute_open_attachment(
        artifact_store=store,
        session_id=sid,
        artifact_id=meta.artifact_id,
        handle=None,
        expected_sha256=None,
        start_line=1,
        end_line=None,
        max_chars=40,
    )
    assert ok2 is True
    assert isinstance(out2, dict)
    assert out2.get("truncated") is True


@pytest.mark.basic
def test_llm_call_handler_injects_session_attachment_index_when_enabled() -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"hello\nworld\n"
    sha = hashlib.sha256(content).hexdigest()
    store.store(
        content,
        content_type="text/plain",
        run_id=rid,
        tags={"kind": "attachment", "path": "notes.txt", "filename": "notes.txt", "session_id": sid, "sha256": sha},
    )

    captured: dict = {}

    class _StubLLM:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return {"content": "ok", "metadata": {}}

    run = RunState.new(workflow_id="wf", entry_node="n1", session_id=sid, vars={})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "tools": [{"name": "open_attachment", "parameters": {}}],
            "params": {"temperature": 0.0},
        },
    )

    handler = make_llm_call_handler(llm=_StubLLM(), artifact_store=store)
    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    msgs = captured.get("messages")
    assert isinstance(msgs, list) and msgs
    assert msgs[0].get("role") == "system"
    assert "Session attachments" in str(msgs[0].get("content") or "")


@pytest.mark.integration
def test_open_attachment_persists_across_restart_file_store(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    store1 = FileArtifactStore(base)
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"hello\nworld\n"
    sha = hashlib.sha256(content).hexdigest()
    meta = store1.store(
        content,
        content_type="text/plain",
        run_id=rid,
        tags={"kind": "attachment", "path": "notes.txt", "filename": "notes.txt", "session_id": sid, "sha256": sha},
    )

    class _NoopTools:
        def execute(self, *, tool_calls):
            raise AssertionError("ToolExecutor should not be invoked for open_attachment-only batches")

    handler1 = make_tool_calls_handler(tools=_NoopTools(), artifact_store=store1)
    run = RunState.new(workflow_id="wf", entry_node="n1", session_id=sid, vars={})
    eff = Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"name": "open_attachment", "arguments": {"artifact_id": meta.artifact_id, "max_chars": 200}}]},
    )
    out1 = handler1(run, eff, None)
    assert out1.status == "completed"
    res1 = out1.result or {}
    assert res1.get("mode") == "executed"
    results1 = res1.get("results") or []
    assert isinstance(results1, list) and results1
    assert results1[0].get("success") is True

    # Restart simulation: new store instance (same base dir) can still load the artifact.
    store2 = FileArtifactStore(base)
    handler2 = make_tool_calls_handler(tools=_NoopTools(), artifact_store=store2)
    out2 = handler2(run, eff, None)
    assert out2.status == "completed"
    res2 = out2.result or {}
    results2 = res2.get("results") or []
    assert isinstance(results2, list) and results2
    assert results2[0].get("success") is True


@pytest.mark.e2e
def test_e2e_open_attachment_tool_call_lmstudio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Level C: LMStudio tool-calling + session attachment index injection + open_attachment."""
    if os.environ.get("ABSTRACT_E2E_LMSTUDIO") != "1":
        pytest.skip("Set ABSTRACT_E2E_LMSTUDIO=1 to run this test.")

    import httpx

    base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-next-80b")
    try:
        r = httpx.get(base_url.rstrip("/") + "/models", timeout=2.0)
        if r.status_code != 200:
            pytest.skip(f"LMStudio not reachable at {base_url!r}")
    except Exception:
        pytest.skip(f"LMStudio not reachable at {base_url!r}")

    monkeypatch.setenv("LMSTUDIO_BASE_URL", base_url)

    from abstractruntime import RunStatus, StepPlan, WorkflowSpec
    from abstractruntime.integrations.abstractcore import create_local_runtime
    from abstractruntime.integrations.abstractcore.default_tools import filter_tool_specs
    from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

    base = tmp_path / "rt"
    run_store = JsonFileRunStore(base)
    ledger_store = JsonlLedgerStore(base)
    artifact_store = FileArtifactStore(base)

    sid = "s1"
    owner = session_memory_owner_run_id(sid)
    content = b"hello\nworld\n"
    sha = hashlib.sha256(content).hexdigest()
    meta = artifact_store.store(
        content,
        content_type="text/plain",
        run_id=owner,
        tags={"kind": "attachment", "path": "notes.txt", "filename": "notes.txt", "session_id": sid, "sha256": sha},
    )
    assert meta.artifact_id

    rt = create_local_runtime(
        provider="lmstudio",
        model=model,
        llm_kwargs={"base_url": base_url},
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
    )

    tool_specs = filter_tool_specs(["open_attachment"])

    prompt = (
        "Call the tool `open_attachment` exactly once to open the attachment for handle '@notes.txt' "
        "with start_line=1 and end_line=1. Do not write any other text."
    )

    def llm_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="LLM",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": prompt,
                    "tools": tool_specs,
                    "provider": "lmstudio",
                    "model": model,
                    "params": {"temperature": 0.0},
                },
                result_key="llm_response",
            ),
            next_node="TOOLS",
        )

    def tools_node(run, ctx) -> StepPlan:
        del ctx
        resp = run.vars.get("llm_response")
        tool_calls = resp.get("tool_calls") if isinstance(resp, dict) else None
        if not isinstance(tool_calls, list):
            tool_calls = []
        return StepPlan(
            node_id="TOOLS",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": tool_calls, "allowed_tools": ["open_attachment"]},
                result_key="tool_results",
            ),
            next_node="DONE",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="DONE", complete_output={"tool_results": run.vars.get("tool_results"), "llm": run.vars.get("llm_response")})

    wf = WorkflowSpec(workflow_id="e2e_open_attachment_lmstudio", entry_node="LLM", nodes={"LLM": llm_node, "TOOLS": tools_node, "DONE": done_node})

    run_id = rt.start(workflow=wf, session_id=sid)
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=8)
    assert state.status == RunStatus.COMPLETED

    # Validate injection occurred (captured in runtime observability payload).
    llm_resp = state.output.get("llm") if isinstance(state.output, dict) else None
    assert isinstance(llm_resp, dict)
    meta_resp = llm_resp.get("metadata") if isinstance(llm_resp.get("metadata"), dict) else {}
    obs = meta_resp.get("_runtime_observability") if isinstance(meta_resp.get("_runtime_observability"), dict) else {}
    captured = obs.get("llm_generate_kwargs") if isinstance(obs.get("llm_generate_kwargs"), dict) else {}
    msgs = captured.get("messages")
    assert isinstance(msgs, list) and msgs
    assert msgs[0].get("role") == "system"
    assert "Session attachments" in str(msgs[0].get("content") or "")

    # Validate tool produced the expected excerpt.
    tool_results = state.output.get("tool_results") if isinstance(state.output, dict) else None
    assert isinstance(tool_results, dict)
    results = tool_results.get("results") or []
    assert isinstance(results, list) and results
    r0 = results[0]
    assert r0.get("success") is True
    out = r0.get("output") or {}
    assert isinstance(out, dict)
    rendered = out.get("rendered")
    assert isinstance(rendered, str)
    assert "1: hello" in rendered

