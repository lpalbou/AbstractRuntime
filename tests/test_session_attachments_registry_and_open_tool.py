from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from abstractcore.tools.common_tools import read_file

from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler, make_tool_calls_handler
from abstractruntime.integrations.abstractcore.session_attachments import (
    dedup_messages_view,
    execute_open_attachment,
    list_session_attachments,
    render_session_attachments_system_message,
    session_memory_owner_run_id,
)
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.storage.in_memory import InMemoryRunStore
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
    assert msg.startswith("Stored session attachments")
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

    ok3, out3, _err3 = execute_open_attachment(
        artifact_store=store,
        session_id=sid,
        artifact_id=meta.artifact_id,
        handle=None,
        expected_sha256=None,
        start_line=1,
        end_line=None,
        max_chars=0,  # no cap
    )
    assert ok3 is True
    assert isinstance(out3, dict)
    assert out3.get("truncated") is False
    assert "3: third" in str(out3.get("rendered") or "")


@pytest.mark.basic
def test_open_attachment_small_preview_expands_to_full_by_default() -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)

    content_str = "\n".join([f"line{i}" for i in range(1, 31)]) + "\n"
    content = content_str.encode("utf-8")
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
        start_line=1,
        end_line=20,  # common "preview" pattern from models
        max_chars=8000,  # default budget
    )
    assert ok is True
    assert err is None
    assert isinstance(out, dict)
    rendered = out.get("rendered")
    assert isinstance(rendered, str)
    assert "lines 1-30" in rendered
    assert "30: line30" in rendered


@pytest.mark.basic
def test_open_attachment_falls_back_from_invalid_artifact_id_to_handle() -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"hello\nworld\n"
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
        artifact_id="notes.txt",  # common model mistake: passes handle as artifact_id
        handle=None,
        expected_sha256=None,
        start_line=1,
        end_line=1,
        max_chars=2000,
    )
    assert ok is True
    assert err is None
    assert isinstance(out, dict)
    assert out.get("artifact_id") == meta.artifact_id
    assert "1: hello" in str(out.get("rendered") or "")


@pytest.mark.basic
def test_open_attachment_binary_returns_media_ref() -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"\xff\xd8\xff\xdb\x00\x00fakejpg"
    sha = hashlib.sha256(content).hexdigest()
    meta = store.store(
        content,
        content_type="image/jpeg",
        run_id=rid,
        tags={"kind": "attachment", "path": "tmp.jpg", "filename": "tmp.jpg", "session_id": sid, "sha256": sha},
    )

    ok, out, err = execute_open_attachment(
        artifact_store=store,
        session_id=sid,
        artifact_id=meta.artifact_id,
        handle=None,
        expected_sha256=None,
        start_line=1,
        end_line=None,
        max_chars=2000,
    )
    assert ok is True
    assert err is None
    assert isinstance(out, dict)
    assert out.get("content_type") == "image/jpeg"
    media = out.get("media")
    assert isinstance(media, list) and media
    m0 = media[0]
    assert isinstance(m0, dict)
    assert m0.get("$artifact") == meta.artifact_id


@pytest.mark.basic
def test_open_attachment_pdf_extracts_text_when_media_stack_available() -> None:
    pytest.importorskip("pymupdf")
    pytest.importorskip("pymupdf4llm")
    pytest.importorskip("abstractcore.media.auto_handler")

    import pymupdf as fitz  # type: ignore[import-not-found]

    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF")
    content = doc.tobytes()
    doc.close()

    sha = hashlib.sha256(content).hexdigest()
    meta = store.store(
        content,
        content_type="application/pdf",
        run_id=rid,
        tags={"kind": "attachment", "path": "doc.pdf", "filename": "doc.pdf", "session_id": sid, "sha256": sha},
    )

    ok, out, err = execute_open_attachment(
        artifact_store=store,
        session_id=sid,
        artifact_id=meta.artifact_id,
        handle=None,
        expected_sha256=None,
        start_line=1,
        end_line=None,
        max_chars=20_000,
    )
    assert ok is True
    assert err is None
    assert isinstance(out, dict)
    assert out.get("derived_from_content_type") == "application/pdf"
    rendered = str(out.get("rendered") or "")
    assert "Hello PDF" in rendered
    content_text = str(out.get("content_text") or "")
    assert "Hello PDF" in content_text


@pytest.mark.basic
def test_tool_calls_read_file_registers_session_attachment_and_open_attachment_works_in_order(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    run_store = InMemoryRunStore()
    sid = "s1"

    (tmp_path / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")

    # Workspace policy must be present in run.vars for path rewriting + relative handle stability.
    run = RunState.new(workflow_id="wf", entry_node="n1", session_id=sid, vars={"workspace_root": str(tmp_path)})

    handler = make_tool_calls_handler(tools=MappingToolExecutor.from_tools([read_file]), artifact_store=store, run_store=run_store)
    eff = Effect(
        type=EffectType.TOOL_CALLS,
        payload={
            "tool_calls": [
                {"name": "read_file", "arguments": {"file_path": "notes.txt", "start_line": 1, "end_line": 2}},
                {"name": "open_attachment", "arguments": {"handle": "@notes.txt", "start_line": 2, "end_line": 2, "max_chars": 4000}},
            ],
            "allowed_tools": ["read_file", "open_attachment"],
        },
    )
    out = handler(run, eff, None)
    assert out.status == "completed"
    res = out.result or {}
    assert res.get("mode") == "executed"
    results = res.get("results") or []
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].get("name") == "read_file"
    assert results[0].get("success") is True
    assert results[1].get("name") == "open_attachment"
    assert results[1].get("success") is True
    rendered = ((results[1].get("output") or {}) if isinstance(results[1].get("output"), dict) else {}).get("rendered")
    assert isinstance(rendered, str) and "2: world" in rendered

    # Attachment is stored under the session memory owner run with a stable virtual handle.
    owner = session_memory_owner_run_id(sid)
    metas = list_session_attachments(artifact_store=store, session_id=sid)
    assert any(str(m.get("artifact_id") or "").strip() for m in metas)
    assert any(m.get("handle") == "notes.txt" for m in metas)
    assert run_store.load(owner) is not None


@pytest.mark.basic
def test_tool_calls_open_attachment_binary_enqueues_pending_media_and_llm_call_consumes(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"\xff\xd8\xff\xdb\x00\x00fakejpg"
    sha = hashlib.sha256(content).hexdigest()
    meta = store.store(
        content,
        content_type="image/jpeg",
        run_id=rid,
        tags={"kind": "attachment", "path": "tmp.jpg", "filename": "tmp.jpg", "session_id": sid, "sha256": sha},
    )

    class _NoopTools:
        def execute(self, *, tool_calls):
            raise AssertionError("ToolExecutor should not be invoked for open_attachment-only batches")

    run = RunState.new(workflow_id="wf", entry_node="n1", session_id=sid, vars={"_runtime": {}})
    handler_tools = make_tool_calls_handler(tools=_NoopTools(), artifact_store=store)
    eff_tools = Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"name": "open_attachment", "arguments": {"handle": "@tmp.jpg"}}]},
    )
    out_tools = handler_tools(run, eff_tools, None)
    assert out_tools.status == "completed"

    rt_ns = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
    assert isinstance(rt_ns, dict)
    pending = rt_ns.get("pending_media")
    assert isinstance(pending, list) and pending
    assert any(isinstance(it, dict) and it.get("$artifact") == meta.artifact_id for it in pending)

    captured: dict = {}

    class _StubLLM:
        def generate(self, **kwargs):
            captured.update(kwargs)
            # Validate that the resolved media file exists during the call.
            media = kwargs.get("media")
            assert isinstance(media, list) and len(media) == 1
            p = Path(str(media[0]))
            assert p.exists()
            assert p.read_bytes() == content
            return {"content": "ok", "metadata": {}}

    handler_llm = make_llm_call_handler(llm=_StubLLM(), artifact_store=store)
    eff_llm = Effect(type=EffectType.LLM_CALL, payload={"prompt": "ok", "params": {"temperature": 0.0}})
    out_llm = handler_llm(run, eff_llm, None)
    assert out_llm.status == "completed"

    rt_ns2 = run.vars.get("_runtime") if isinstance(run.vars, dict) else None
    assert isinstance(rt_ns2, dict)
    assert rt_ns2.get("pending_media") == []


@pytest.mark.integration
def test_read_file_registered_attachment_persists_across_restart_file_store(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    store1 = FileArtifactStore(base)
    run_store = InMemoryRunStore()
    sid = "s1"

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")

    run = RunState.new(workflow_id="wf", entry_node="n1", session_id=sid, vars={"workspace_root": str(ws)})

    handler1 = make_tool_calls_handler(tools=MappingToolExecutor.from_tools([read_file]), artifact_store=store1, run_store=run_store)
    eff1 = Effect(
        type=EffectType.TOOL_CALLS,
        payload={"tool_calls": [{"name": "read_file", "arguments": {"file_path": "notes.txt"}}], "allowed_tools": ["read_file"]},
    )
    out1 = handler1(run, eff1, None)
    assert out1.status == "completed"

    # Restart simulation: new store instance can open the attachment created by read_file.
    store2 = FileArtifactStore(base)
    ok, out, err = execute_open_attachment(
        artifact_store=store2,
        session_id=sid,
        artifact_id=None,
        handle="@notes.txt",
        expected_sha256=None,
        start_line=1,
        end_line=1,
        max_chars=2000,
    )
    assert ok is True
    assert err is None
    assert isinstance(out, dict)
    assert "1: hello" in str(out.get("rendered") or "")


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
    assert "Stored session attachments" in str(msgs[0].get("content") or "")


@pytest.mark.basic
def test_llm_call_handler_injects_active_attachments_when_media_present(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"hello\n"
    sha = hashlib.sha256(content).hexdigest()
    meta = store.store(
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
            "media": [{"$artifact": meta.artifact_id, "filename": "notes.txt", "source_path": "notes.txt", "sha256": sha, "content_type": "text/plain"}],
            "params": {"temperature": 0.0},
        },
    )

    handler = make_llm_call_handler(llm=_StubLLM(), artifact_store=store)
    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    msgs = captured.get("messages")
    assert isinstance(msgs, list) and msgs
    assert msgs[0].get("role") == "system"
    content = str(msgs[0].get("content") or "")
    assert content.startswith("Active attachments")
    assert "notes.txt" in content
    assert "@notes.txt" not in content


@pytest.mark.basic
def test_llm_call_handler_inlines_active_text_attachments_into_messages_and_removes_media() -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"hello\nworld\n"
    sha = hashlib.sha256(content).hexdigest()
    meta = store.store(
        content,
        content_type="text/plain",
        run_id=rid,
        tags={"kind": "attachment", "path": "/Users/albou/Downloads/notes.txt", "filename": "notes.txt", "session_id": sid, "sha256": sha},
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
            "messages": [{"role": "user", "content": "Summarize attachments."}],
            "media": [
                {
                    "$artifact": meta.artifact_id,
                    "source_path": "/Users/albou/Downloads/notes.txt",
                    "sha256": sha,
                    "content_type": "text/plain",
                }
            ],
            "tools": [
                {"name": "ask_user", "description": "Ask.", "parameters": {}},
                {"name": "open_attachment", "description": "Open.", "parameters": {}},
            ],
            "params": {"temperature": 0.0},
        },
    )

    handler = make_llm_call_handler(llm=_StubLLM(), artifact_store=store)
    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    # Small active text attachment should be inlined into messages and removed from provider media.
    assert captured.get("media") is None

    msgs = captured.get("messages")
    assert isinstance(msgs, list) and msgs
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    assert user_msgs and isinstance(user_msgs[-1].get("content"), str)
    user_text = str(user_msgs[-1].get("content") or "")
    assert "--- Content from notes.txt ---" in user_text
    assert "hello" in user_text and "world" in user_text

    sys_text = "\n".join([str(m.get("content") or "") for m in msgs if isinstance(m, dict) and m.get("role") == "system"])
    assert "Active attachments" in sys_text
    assert "notes.txt" in sys_text
    assert "@notes.txt" not in sys_text
    assert "/Users/" not in sys_text
    assert "Stored session attachments" not in sys_text


@pytest.mark.basic
def test_llm_call_use_context_does_not_persist_inlined_attachment_blocks() -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"hello\nworld\n"
    sha = hashlib.sha256(content).hexdigest()
    meta = store.store(
        content,
        content_type="text/plain",
        run_id=rid,
        tags={"kind": "attachment", "path": "/Users/albou/Downloads/notes.txt", "filename": "notes.txt", "session_id": sid, "sha256": sha},
    )

    class _StubLLM:
        def generate(self, **kwargs):  # type: ignore[no-untyped-def]
            return {"content": "ok", "metadata": {}}

    run = RunState.new(workflow_id="wf", entry_node="n1", session_id=sid, vars={"context": {}})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "messages": [{"role": "user", "content": "Summarize attachments."}],
            "media": [
                {
                    "$artifact": meta.artifact_id,
                    "source_path": "/Users/albou/Downloads/notes.txt",
                    "sha256": sha,
                    "content_type": "text/plain",
                }
            ],
            "include_context": True,
            "params": {"temperature": 0.0},
        },
    )

    handler = make_llm_call_handler(llm=_StubLLM(), artifact_store=store)
    outcome = handler(run, effect, None)
    assert outcome.status == "completed"

    ctx = run.vars.get("context")
    assert isinstance(ctx, dict)
    msgs_any = ctx.get("messages")
    assert isinstance(msgs_any, list) and msgs_any
    assert msgs_any[0].get("role") == "user"
    assert msgs_any[0].get("content") == "Summarize attachments."
    assert "--- Content from" not in str(msgs_any[0].get("content") or "")


@pytest.mark.basic
def test_open_attachment_not_found_returns_suggestions() -> None:
    store = InMemoryArtifactStore()
    sid = "s1"
    rid = session_memory_owner_run_id(sid)
    content = b"hello\nworld\n"
    sha = hashlib.sha256(content).hexdigest()
    store.store(
        content,
        content_type="text/plain",
        run_id=rid,
        tags={
            "kind": "attachment",
            "path": "mnemosyne/memory/notes/2025/03/Aletheia-to-my-son-09.md",
            "filename": "Aletheia-to-my-son-09.md",
            "session_id": sid,
            "sha256": sha,
        },
    )

    ok, out, err = execute_open_attachment(
        artifact_store=store,
        session_id=sid,
        artifact_id=None,
        handle="@mnemosyne/memory/notes/2025/12/Aletheia-to-my-son.md",
        expected_sha256=None,
        start_line=1,
        end_line=5,
        max_chars=2000,
    )
    assert ok is False
    assert err == "attachment not found"
    assert isinstance(out, dict)
    suggestions = out.get("suggestions")
    assert isinstance(suggestions, list) and suggestions
    assert any(
        isinstance(s, dict) and str(s.get("handle") or "").endswith("Aletheia-to-my-son-09.md") for s in suggestions
    )


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
        tool_executor=MappingToolExecutor.from_tools([read_file]),
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
    assert "Stored session attachments" in str(msgs[0].get("content") or "")

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


@pytest.mark.e2e
def test_e2e_open_attachment_media_ref_lmstudio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Level C: LMStudio tool-calling; open_attachment returns media refs for binary attachments."""
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
    content = b"\xff\xd8\xff\xdb\x00\x00fakejpg"
    sha = hashlib.sha256(content).hexdigest()
    meta = artifact_store.store(
        content,
        content_type="image/jpeg",
        run_id=owner,
        tags={"kind": "attachment", "path": "tmp.jpg", "filename": "tmp.jpg", "session_id": sid, "sha256": sha},
    )
    assert meta.artifact_id

    rt = create_local_runtime(
        provider="lmstudio",
        model=model,
        llm_kwargs={"base_url": base_url},
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        tool_executor=MappingToolExecutor.from_tools([read_file]),
    )

    tool_specs = filter_tool_specs(["open_attachment"])
    prompt = "Call `open_attachment` exactly once with handle='@tmp.jpg'. Do not write any other text."

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
        return StepPlan(node_id="DONE", complete_output={"tool_results": run.vars.get("tool_results")})

    wf = WorkflowSpec(workflow_id="e2e_open_attachment_media_ref_lmstudio", entry_node="LLM", nodes={"LLM": llm_node, "TOOLS": tools_node, "DONE": done_node})

    run_id = rt.start(workflow=wf, session_id=sid)
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=8)
    assert state.status == RunStatus.COMPLETED

    tool_results = state.output.get("tool_results") if isinstance(state.output, dict) else None
    assert isinstance(tool_results, dict)
    results = tool_results.get("results") or []
    assert isinstance(results, list) and results
    r0 = results[0]
    assert r0.get("success") is True
    out0 = r0.get("output") or {}
    assert isinstance(out0, dict)
    assert out0.get("content_type") == "image/jpeg"
    media = out0.get("media")
    assert isinstance(media, list) and media
    m0 = media[0]
    assert isinstance(m0, dict)
    assert m0.get("$artifact") == meta.artifact_id


@pytest.mark.e2e
def test_e2e_read_file_registers_attachment_then_open_attachment_lmstudio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Level C: LMStudio native tools; read_file registers attachment; open_attachment works next call."""
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

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")

    base = tmp_path / "rt"
    run_store = JsonFileRunStore(base)
    ledger_store = JsonlLedgerStore(base)
    artifact_store = FileArtifactStore(base)

    sid = "s1"
    rt = create_local_runtime(
        provider="lmstudio",
        model=model,
        llm_kwargs={"base_url": base_url},
        run_store=run_store,
        ledger_store=ledger_store,
        artifact_store=artifact_store,
        tool_executor=MappingToolExecutor.from_tools([read_file]),
    )

    tool_specs_read = filter_tool_specs(["read_file"])
    tool_specs_open = filter_tool_specs(["open_attachment"])

    prompt1 = "Call the tool `read_file` exactly once with file_path='notes.txt'. Do not write any other text."
    prompt2 = (
        "Call the tool `open_attachment` exactly once with handle='@notes.txt', start_line=2 and end_line=2. "
        "Do not write any other text."
    )

    def llm1_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="LLM1",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": prompt1,
                    "tools": tool_specs_read,
                    "provider": "lmstudio",
                    "model": model,
                    "params": {"temperature": 0.0},
                },
                result_key="llm1",
            ),
            next_node="TOOLS1",
        )

    def tools1_node(run, ctx) -> StepPlan:
        del ctx
        resp = run.vars.get("llm1")
        tool_calls = resp.get("tool_calls") if isinstance(resp, dict) else None
        if not isinstance(tool_calls, list):
            tool_calls = []
        return StepPlan(
            node_id="TOOLS1",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": tool_calls, "allowed_tools": ["read_file"]}, result_key="tools1"),
            next_node="LLM2",
        )

    def llm2_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(
            node_id="LLM2",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": prompt2,
                    "tools": tool_specs_open,
                    "provider": "lmstudio",
                    "model": model,
                    "params": {"temperature": 0.0},
                },
                result_key="llm2",
            ),
            next_node="TOOLS2",
        )

    def tools2_node(run, ctx) -> StepPlan:
        del ctx
        resp = run.vars.get("llm2")
        tool_calls = resp.get("tool_calls") if isinstance(resp, dict) else None
        if not isinstance(tool_calls, list):
            tool_calls = []
        return StepPlan(
            node_id="TOOLS2",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": tool_calls, "allowed_tools": ["open_attachment"]}, result_key="tools2"),
            next_node="DONE",
        )

    def done_node(run, ctx) -> StepPlan:
        del ctx
        return StepPlan(node_id="DONE", complete_output={"llm1": run.vars.get("llm1"), "tools1": run.vars.get("tools1"), "llm2": run.vars.get("llm2"), "tools2": run.vars.get("tools2")})

    wf = WorkflowSpec(
        workflow_id="e2e_read_file_registers_attachment_lmstudio",
        entry_node="LLM1",
        nodes={"LLM1": llm1_node, "TOOLS1": tools1_node, "LLM2": llm2_node, "TOOLS2": tools2_node, "DONE": done_node},
    )

    run_id = rt.start(workflow=wf, session_id=sid, vars={"workspace_root": str(ws)})
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=12)
    assert state.status == RunStatus.COMPLETED

    # Ensure `read_file` actually executed and triggered attachment registration.
    tools1 = state.output.get("tools1") if isinstance(state.output, dict) else None
    assert isinstance(tools1, dict)
    res1 = tools1.get("results") or []
    assert isinstance(res1, list) and res1
    assert res1[0].get("name") == "read_file"
    assert res1[0].get("success") is True

    # Validate second LLM call injection includes the new attachment handle.
    llm2 = state.output.get("llm2") if isinstance(state.output, dict) else None
    assert isinstance(llm2, dict)
    meta_resp = llm2.get("metadata") if isinstance(llm2.get("metadata"), dict) else {}
    obs = meta_resp.get("_runtime_observability") if isinstance(meta_resp.get("_runtime_observability"), dict) else {}
    captured = obs.get("llm_generate_kwargs") if isinstance(obs.get("llm_generate_kwargs"), dict) else {}
    msgs = captured.get("messages")
    assert isinstance(msgs, list) and msgs
    msg0 = str(msgs[0].get("content") or "")
    assert "notes.txt" in msg0
    assert "@notes.txt" not in msg0

    # Validate tool output contains the expected line.
    tools2 = state.output.get("tools2") if isinstance(state.output, dict) else None
    assert isinstance(tools2, dict)
    res2 = tools2.get("results") or []
    assert isinstance(res2, list) and res2
    r0 = res2[0]
    assert r0.get("name") == "open_attachment"
    assert r0.get("success") is True
    out = r0.get("output") or {}
    assert isinstance(out, dict)
    assert "2: world" in str(out.get("rendered") or "")
