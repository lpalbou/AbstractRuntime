from __future__ import annotations

import hashlib
from pathlib import Path

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.session_attachments import session_memory_owner_run_id
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryRunStore


class _FailingReadFileExecutor:
    """ToolExecutor stub that always fails read_file (simulates missing on-disk file)."""

    def __init__(self) -> None:
        self.last_calls: list[dict] = []

    def execute(self, *, tool_calls: list[dict]) -> dict:
        self.last_calls = list(tool_calls or [])
        results: list[dict] = []
        for tc in self.last_calls:
            results.append(
                {
                    "call_id": str(tc.get("call_id") or ""),
                    "name": str(tc.get("name") or ""),
                    "success": False,
                    "output": None,
                    "error": "File does not exist",
                }
            )
        return {"mode": "executed", "results": results}


def _run_state(*, run_id: str, session_id: str, workspace_root: Path) -> RunState:
    now = "2026-01-18T00:00:00+00:00"
    return RunState(
        run_id=run_id,
        workflow_id="wf_test",
        status=RunStatus.RUNNING,
        current_node="node",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {"memory_spans": []},
            "_temp": {},
            "_limits": {},
            "workspace_root": str(workspace_root),
            "workspace_access_mode": "workspace_only",
        },
        waiting=None,
        output={"messages": []},
        error=None,
        created_at=now,
        updated_at=now,
        actor_id=None,
        session_id=session_id,
        parent_run_id=None,
    )


def test_read_file_falls_back_to_session_attachment_on_missing_file(tmp_path: Path) -> None:
    sid = "s1"
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)

    artifact_store = InMemoryArtifactStore()
    run_store = InMemoryRunStore()

    # Store an attachment as if it was uploaded by a client (no on-disk file required).
    content = b"line1\nline2\n"
    sha256 = hashlib.sha256(content).hexdigest()
    session_run_id = session_memory_owner_run_id(sid)
    meta = artifact_store.store(
        content,
        content_type="text/plain",
        run_id=session_run_id,
        tags={
            "kind": "attachment",
            "source": "upload",
            "path": "bfi-test.md",
            "filename": "bfi-test.md",
            "session_id": sid,
            "sha256": sha256,
        },
    )

    executor = _FailingReadFileExecutor()
    handler = make_tool_calls_handler(tools=executor, artifact_store=artifact_store, run_store=run_store)

    run = _run_state(run_id="r1", session_id=sid, workspace_root=ws)
    effect = Effect(
        type=EffectType.TOOL_CALLS,
        payload={
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": "read_file",
                    "arguments": {"file_path": "bfi-test.md", "start_line": 1, "end_line": 2},
                }
            ]
        },
    )

    out = handler(run, effect, None)
    assert out.status == "completed"
    assert isinstance(out.result, dict)
    results = out.result.get("results")
    assert isinstance(results, list) and len(results) == 1

    # Workspace scope rewrite should convert relative file_path into an absolute path under workspace_root.
    assert executor.last_calls and isinstance(executor.last_calls[0].get("arguments"), dict)
    rewritten_fp = str(executor.last_calls[0]["arguments"].get("file_path") or "")
    assert rewritten_fp.startswith(str(ws)), rewritten_fp

    r0 = results[0]
    assert isinstance(r0, dict)
    assert r0.get("success") is True
    output = r0.get("output")
    assert isinstance(output, str) and output.strip()

    assert f"File: {rewritten_fp} (2 lines)" in output
    assert meta.artifact_id in output
    assert "1: line1" in output
    assert "2: line2" in output


def test_out_of_workspace_file_tools_use_session_attachments_without_executor_calls(tmp_path: Path) -> None:
    sid = "s2"
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)

    outside = (tmp_path / "outside.txt").resolve()
    outside_handle = str(outside)

    artifact_store = InMemoryArtifactStore()
    run_store = InMemoryRunStore()

    content = b"line1\nline2\n"
    sha256 = hashlib.sha256(content).hexdigest()
    session_run_id = session_memory_owner_run_id(sid)
    meta = artifact_store.store(
        content,
        content_type="text/plain",
        run_id=session_run_id,
        tags={
            "kind": "attachment",
            "source": "local",
            "path": outside_handle,
            "filename": "outside.txt",
            "session_id": sid,
            "sha256": sha256,
        },
    )

    executor = _FailingReadFileExecutor()
    handler = make_tool_calls_handler(tools=executor, artifact_store=artifact_store, run_store=run_store)
    run = _run_state(run_id="r2", session_id=sid, workspace_root=ws)

    effect = Effect(
        type=EffectType.TOOL_CALLS,
        payload={
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": "read_file",
                    "arguments": {"file_path": outside_handle, "start_line": 1, "end_line": 2},
                },
                {
                    "call_id": "c2",
                    "name": "list_files",
                    "arguments": {"directory_path": str(tmp_path), "pattern": "*.txt", "recursive": False, "head_limit": 10},
                },
                {
                    "call_id": "c3",
                    "name": "search_files",
                    "arguments": {"pattern": "line2", "path": outside_handle, "head_limit": 5, "max_hits": 5},
                },
                {
                    "call_id": "c4",
                    "name": "skim_files",
                    "arguments": {"paths": [outside_handle], "head_lines": 2, "tail_lines": 0},
                },
                {
                    "call_id": "c5",
                    "name": "skim_folders",
                    "arguments": {"paths": [str(tmp_path)]},
                },
            ]
        },
    )

    out = handler(run, effect, None)
    assert out.status == "completed"
    assert isinstance(out.result, dict)
    results = out.result.get("results")
    assert isinstance(results, list) and len(results) == 5

    # These should be served from session attachments at rewrite-time; no host tool execution required.
    assert executor.last_calls == []

    r_read = results[0]
    assert isinstance(r_read, dict) and r_read.get("success") is True
    read_out = r_read.get("output")
    assert isinstance(read_out, str) and f"File: {outside_handle} (2 lines)" in read_out
    assert meta.artifact_id in read_out

    r_list = results[1]
    assert isinstance(r_list, dict) and r_list.get("success") is True
    list_out = r_list.get("output")
    assert isinstance(list_out, str) and "outside.txt" in list_out

    r_search = results[2]
    assert isinstance(r_search, dict) and r_search.get("success") is True
    search_out = r_search.get("output")
    assert isinstance(search_out, str) and "2: line2" in search_out

    r_skim = results[3]
    assert isinstance(r_skim, dict) and r_skim.get("success") is True
    skim_out = r_skim.get("output")
    assert isinstance(skim_out, str) and "Attachment:" in skim_out and "outside.txt" in skim_out

    r_skim_folders = results[4]
    assert isinstance(r_skim_folders, dict) and r_skim_folders.get("success") is True
    skim_folders_out = r_skim_folders.get("output")
    assert isinstance(skim_folders_out, str) and "outside.txt" in skim_folders_out
