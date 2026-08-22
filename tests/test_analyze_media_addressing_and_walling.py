"""`analyze_media` addresses BOTH namespaces, and rides the workspace wall.

Two defects measured on 2026-08-21 across 14 gateway sessions:

1. ADDRESSING. `analyze_media` stats the path it is given, so a session
   attachment — addressed by the artifact id or the display filename that the
   runtime itself puts in the model's system message — always failed with
   "File '<x>' does not exist". 9 of the 10 `analyze_media` calls in the
   corpus died this way; one run retried the same live artifact id five times
   while `open_attachment` resolved that id in the same turn.

2. WALLING. `analyze_media` was absent from `rewrite_tool_arguments`'s tool
   list, whose fall-through returns the arguments unchanged. Under
   `workspace_only`, `read_file "/etc/hosts"` raised and `analyze_media
   "/etc/hosts"` passed through — on the one file-reading tool that ships
   BYTES to a possibly-remote vision provider.

The two fixes meet at one seam: an attachment resolves to a path this process
wrote, which must NOT then be walled. That is expressed as a predicate
(`_is_system_produced_media_path`) rather than an ordering rule between two
call sites, so a third rewrite cannot silently break it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.session_attachments import (
    attachment_media_dir,
    session_memory_owner_run_id,
)
from abstractruntime.integrations.abstractcore.workspace_scoped_tools import (
    WorkspaceScope,
    rewrite_tool_arguments,
)
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.in_memory import InMemoryRunStore

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000100" "05fe02fe" "dccc59e70000000049454e44ae426082"
)


class _RecordingExecutor:
    """Records the arguments the tool actually received."""

    def __init__(self) -> None:
        self.last_calls: list[dict] = []

    def execute(self, *, tool_calls: list[dict]) -> dict:
        self.last_calls = list(tool_calls or [])
        return {
            "mode": "executed",
            "results": [
                {
                    "call_id": str(tc.get("call_id") or ""),
                    "name": str(tc.get("name") or ""),
                    "success": True,
                    "output": "a description",
                    "error": None,
                }
                for tc in self.last_calls
            ],
        }


def _run_state(*, session_id: str, workspace_root: Path) -> RunState:
    now = "2026-08-21T00:00:00+00:00"
    return RunState(
        run_id="r1",
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


def _scope(tmp_path: Path) -> WorkspaceScope:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    scope = WorkspaceScope.from_input_data({"workspace_root": str(ws)})
    assert scope is not None
    return scope


def _store_attachment(store: InMemoryArtifactStore, sid: str, filename: str):
    return store.store(
        PNG_1PX,
        content_type="image/png",
        run_id=session_memory_owner_run_id(sid),
        tags={
            "kind": "attachment",
            "source": "upload",
            "path": filename,
            "filename": filename,
            "session_id": sid,
            "sha256": hashlib.sha256(PNG_1PX).hexdigest(),
        },
    )


def _analyze(handler, run, file_path: str):
    return handler(
        run,
        Effect(
            type=EffectType.TOOL_CALLS,
            payload={
                "tool_calls": [
                    {"call_id": "c1", "name": "analyze_media",
                     "arguments": {"file_path": file_path, "question": "what is this?"}}
                ]
            },
        ),
        None,
    )


# --------------------------------------------------------------------------
# 1. Addressing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("addressed_as", ["artifact_id", "filename"])
def test_attachment_is_resolved_to_real_bytes(tmp_path: Path, addressed_as: str) -> None:
    sid = "s-media"
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    store = InMemoryArtifactStore()
    meta = _store_attachment(store, sid, "Screenshot 2026-08-21 at 4.49.30 AM.png")

    executor = _RecordingExecutor()
    handler = make_tool_calls_handler(tools=executor, artifact_store=store, run_store=InMemoryRunStore())
    query = meta.artifact_id if addressed_as == "artifact_id" else "Screenshot 2026-08-21 at 4.49.30 AM.png"

    out = _analyze(handler, _run_state(session_id=sid, workspace_root=ws), query)
    assert out.status == "completed"

    seen = str(executor.last_calls[0]["arguments"]["file_path"])
    assert Path(seen).is_file(), f"the tool must receive a real file, got {seen!r}"
    assert Path(seen).read_bytes() == PNG_1PX, "the bytes must be the attachment's own"
    assert Path(seen).suffix == ".png", "the suffix must survive: analyze_media gates on it"


def test_a_genuinely_absent_path_is_left_alone(tmp_path: Path) -> None:
    """The miss case must not invent a resolution — the file namespace answers."""
    sid = "s-miss"
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    store = InMemoryArtifactStore()
    _store_attachment(store, sid, "present.png")

    executor = _RecordingExecutor()
    handler = make_tool_calls_handler(tools=executor, artifact_store=store, run_store=InMemoryRunStore())
    out = _analyze(handler, _run_state(session_id=sid, workspace_root=ws), "absent.png")
    assert out.status == "completed"
    seen = str(executor.last_calls[0]["arguments"]["file_path"])
    assert seen.startswith(str(ws)), "an unresolvable name stays a workspace path"


def test_an_ambiguous_name_is_not_guessed(tmp_path: Path) -> None:
    """Two attachments, one name: refuse to resolve rather than pick one."""
    sid = "s-dupe"
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    store = InMemoryArtifactStore()
    _store_attachment(store, sid, "shot.png")
    store.store(
        PNG_1PX + b"\x00",
        content_type="image/png",
        run_id=session_memory_owner_run_id(sid),
        tags={"kind": "attachment", "source": "upload", "path": "shot.png",
              "filename": "shot.png", "session_id": sid},
    )

    executor = _RecordingExecutor()
    handler = make_tool_calls_handler(tools=executor, artifact_store=store, run_store=InMemoryRunStore())
    _analyze(handler, _run_state(session_id=sid, workspace_root=ws), "shot.png")
    seen = str(executor.last_calls[0]["arguments"]["file_path"])
    assert seen.startswith(str(ws)), "an ambiguous name must not resolve to either artifact"


def test_the_result_names_the_exact_call_that_shows_the_image(tmp_path: Path) -> None:
    """A reading is one model's summary; the caller can do better than trust it.

    When the image IS a session attachment, the host appends the exact
    `open_attachment(...)` that puts it in front of the model. The tool itself
    cannot say this — whether these bytes are attachable is a session fact core
    does not have, and `open_attachment` resolves session attachments only, so
    the same advice on a plain disk file would be a dead end.
    """
    sid = "s-hint"
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    store = InMemoryArtifactStore()
    meta = _store_attachment(store, sid, "card.png")

    executor = _RecordingExecutor()
    handler = make_tool_calls_handler(tools=executor, artifact_store=store, run_store=InMemoryRunStore())
    out = _analyze(handler, _run_state(session_id=sid, workspace_root=ws), "card.png")
    results = out.result.get("results")
    text = str(results[0].get("output") or "")
    assert f'open_attachment(artifact_id="{meta.artifact_id}")' in text, text
    assert "see it yourself" in text


def test_a_plain_file_gets_no_attachment_hint(tmp_path: Path) -> None:
    """The dead-end case: nothing to open, so nothing is offered."""
    sid = "s-plain"
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "on_disk.png").write_bytes(PNG_1PX)
    store = InMemoryArtifactStore()

    executor = _RecordingExecutor()
    handler = make_tool_calls_handler(tools=executor, artifact_store=store, run_store=InMemoryRunStore())
    out = _analyze(handler, _run_state(session_id=sid, workspace_root=ws), "on_disk.png")
    text = str((out.result.get("results") or [{}])[0].get("output") or "")
    assert "open_attachment(" not in text, text


# --------------------------------------------------------------------------
# 2. Walling
# --------------------------------------------------------------------------

def test_absolute_path_outside_the_workspace_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        rewrite_tool_arguments(
            tool_name="analyze_media", args={"file_path": "/etc/hosts"}, scope=_scope(tmp_path)
        )


def test_relative_path_resolves_under_the_workspace_root(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    (tmp_path / "ws" / "shot.png").write_bytes(PNG_1PX)
    out = rewrite_tool_arguments(
        tool_name="analyze_media", args={"file_path": "shot.png"}, scope=scope
    )
    assert out["file_path"] == str((tmp_path / "ws" / "shot.png").resolve())


def test_path_alias_is_folded_before_walling(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    (tmp_path / "ws" / "shot.png").write_bytes(PNG_1PX)
    out = rewrite_tool_arguments(
        tool_name="analyze_media", args={"path": "shot.png"}, scope=scope
    )
    assert out["file_path"] == str((tmp_path / "ws" / "shot.png").resolve())


def test_system_produced_media_paths_are_not_walled(tmp_path: Path) -> None:
    """The seam: bytes THIS PROCESS wrote are not a user-filesystem read.

    Without this, fixing the wall would break the fix for the addressing —
    and the browser probe's screenshot handoff, which analyze_media has
    consumed since before either change.
    """
    scope = _scope(tmp_path)

    materialized = Path(attachment_media_dir()) / "deadbeef.png"
    materialized.write_bytes(PNG_1PX)
    out = rewrite_tool_arguments(
        tool_name="analyze_media", args={"file_path": str(materialized)}, scope=scope
    )
    assert out["file_path"] == str(materialized)

    from abstractcore.tools.browser_tools import _shared_screenshot_dir

    shot = Path(_shared_screenshot_dir()) / "probe_abc.png"
    shot.write_bytes(PNG_1PX)
    out = rewrite_tool_arguments(
        tool_name="analyze_media", args={"file_path": str(shot)}, scope=scope
    )
    assert out["file_path"] == str(shot)
