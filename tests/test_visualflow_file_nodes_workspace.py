from __future__ import annotations

from pathlib import Path

from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json


def test_write_file_uses_run_workspace_after_upstream_execution_node(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    relpath = "reports/deep-research.md"

    vf = load_visualflow_json(
        {
            "id": "test-write-file-workspace",
            "name": "test-write-file-workspace",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "build_markdown",
                    "type": "code",
                    "data": {
                        "nodeType": "code",
                        "codeBody": "return '# Deep Research\\n\\nA workspace-scoped report.\\n'",
                    },
                },
                {
                    "id": "write_markdown",
                    "type": "write_file",
                    "data": {
                        "nodeType": "write_file",
                        "pinDefaults": {"file_path": relpath},
                    },
                },
                {"id": "read_markdown", "type": "read_file", "data": {"nodeType": "read_file"}},
                {
                    "id": "end",
                    "type": "on_flow_end",
                    "data": {
                        "nodeType": "on_flow_end",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "content", "label": "content", "type": "string"},
                        ],
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "build_markdown", "targetHandle": "exec-in"},
                {
                    "source": "build_markdown",
                    "sourceHandle": "exec-out",
                    "target": "write_markdown",
                    "targetHandle": "exec-in",
                },
                {
                    "source": "write_markdown",
                    "sourceHandle": "exec-out",
                    "target": "read_markdown",
                    "targetHandle": "exec-in",
                },
                {"source": "read_markdown", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
                {
                    "source": "build_markdown",
                    "sourceHandle": "output",
                    "target": "write_markdown",
                    "targetHandle": "content",
                },
                {
                    "source": "write_markdown",
                    "sourceHandle": "file_path",
                    "target": "read_markdown",
                    "targetHandle": "file_path",
                },
                {"source": "read_markdown", "sourceHandle": "content", "target": "end", "targetHandle": "content"},
            ],
        }
    )

    workflow = compile_flow(visual_to_flow(vf))
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(
        workflow=workflow,
        vars={"workspace_root": str(workspace), "workspace_access_mode": "workspace_only"},
    )
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=20)

    assert state.status.value == "completed"
    assert (workspace / relpath).read_text(encoding="utf-8") == "# Deep Research\n\nA workspace-scoped report.\n"
    assert state.output == {"content": "# Deep Research\n\nA workspace-scoped report.\n", "success": True}


def test_write_pdf_and_read_pdf_use_run_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    relpath = "reports/deep-research.pdf"

    vf = load_visualflow_json(
        {
            "id": "test-pdf-workspace",
            "name": "test-pdf-workspace",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "build_markdown",
                    "type": "code",
                    "data": {
                        "nodeType": "code",
                        "codeBody": (
                            "return '# Deep Research\\n\\n'"
                            "+ '#### 2.1 Tool use and external action\\n\\n'"
                            "+ 'A workspace-scoped PDF report.\\n\\n'"
                            "+ '##### 2.1.1 Source handling\\n\\n'"
                            "+ '- Source A\\n\\n'"
                            "+ '###### Implementation note\\n\\n'"
                            "+ 'No raw markdown heading markers should leak into the PDF text.\\n'"
                        ),
                    },
                },
                {
                    "id": "write_pdf",
                    "type": "write_pdf",
                    "data": {
                        "nodeType": "write_pdf",
                        "pinDefaults": {"file_path": relpath, "title": "Deep Research"},
                    },
                },
                {"id": "read_pdf", "type": "read_pdf", "data": {"nodeType": "read_pdf"}},
                {
                    "id": "end",
                    "type": "on_flow_end",
                    "data": {
                        "nodeType": "on_flow_end",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "content", "label": "content", "type": "string"},
                            {"id": "pages", "label": "pages", "type": "number"},
                            {"id": "metadata", "label": "metadata", "type": "object"},
                            {"id": "pdf_path", "label": "pdf_path", "type": "string"},
                            {"id": "sha256", "label": "sha256", "type": "string"},
                        ],
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "build_markdown", "targetHandle": "exec-in"},
                {"source": "build_markdown", "sourceHandle": "exec-out", "target": "write_pdf", "targetHandle": "exec-in"},
                {"source": "write_pdf", "sourceHandle": "exec-out", "target": "read_pdf", "targetHandle": "exec-in"},
                {"source": "read_pdf", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
                {"source": "build_markdown", "sourceHandle": "output", "target": "write_pdf", "targetHandle": "content"},
                {"source": "write_pdf", "sourceHandle": "file_path", "target": "read_pdf", "targetHandle": "file_path"},
                {"source": "read_pdf", "sourceHandle": "content", "target": "end", "targetHandle": "content"},
                {"source": "read_pdf", "sourceHandle": "pages", "target": "end", "targetHandle": "pages"},
                {"source": "read_pdf", "sourceHandle": "metadata", "target": "end", "targetHandle": "metadata"},
                {"source": "write_pdf", "sourceHandle": "file_path", "target": "end", "targetHandle": "pdf_path"},
                {"source": "write_pdf", "sourceHandle": "sha256", "target": "end", "targetHandle": "sha256"},
            ],
        }
    )

    workflow = compile_flow(visual_to_flow(vf))
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(
        workflow=workflow,
        vars={"workspace_root": str(workspace), "workspace_access_mode": "workspace_only"},
    )
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=20)

    report = workspace / relpath
    assert state.status.value == "completed"
    assert report.read_bytes().startswith(b"%PDF-")
    assert "Deep Research" in state.output["content"]
    assert "2.1 Tool use and external action" in state.output["content"]
    assert "2.1.1 Source handling" in state.output["content"]
    assert "Implementation note" in state.output["content"]
    assert "#### 2.1 Tool use" not in state.output["content"]
    assert "##### 2.1.1 Source" not in state.output["content"]
    assert "###### Implementation" not in state.output["content"]
    assert "workspace-scoped PDF report" in state.output["content"]
    assert state.output["pages"] >= 1
    assert state.output["metadata"]["content_type"] == "application/pdf"
    assert state.output["pdf_path"] == relpath
    assert len(state.output["sha256"]) == 64


def test_read_pdf_reports_explicit_truncation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    relpath = "reports/truncated.pdf"

    vf = load_visualflow_json(
        {
            "id": "test-pdf-truncation",
            "name": "test-pdf-truncation",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "build_markdown",
                    "type": "code",
                    "data": {
                        "nodeType": "code",
                        "codeBody": "return '# Long Report\\n\\n' + ('abcdef ' * 80)",
                    },
                },
                {
                    "id": "write_pdf",
                    "type": "write_pdf",
                    "data": {"nodeType": "write_pdf", "pinDefaults": {"file_path": relpath}},
                },
                {
                    "id": "read_pdf",
                    "type": "read_pdf",
                    "data": {"nodeType": "read_pdf", "pinDefaults": {"max_chars": 32}},
                },
                {
                    "id": "end",
                    "type": "on_flow_end",
                    "data": {
                        "nodeType": "on_flow_end",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "content", "label": "content", "type": "string"},
                            {"id": "truncated", "label": "truncated", "type": "boolean"},
                            {"id": "warnings", "label": "warnings", "type": "array"},
                        ],
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "build_markdown", "targetHandle": "exec-in"},
                {"source": "build_markdown", "sourceHandle": "exec-out", "target": "write_pdf", "targetHandle": "exec-in"},
                {"source": "write_pdf", "sourceHandle": "exec-out", "target": "read_pdf", "targetHandle": "exec-in"},
                {"source": "read_pdf", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
                {"source": "build_markdown", "sourceHandle": "output", "target": "write_pdf", "targetHandle": "content"},
                {"source": "write_pdf", "sourceHandle": "file_path", "target": "read_pdf", "targetHandle": "file_path"},
                {"source": "read_pdf", "sourceHandle": "content", "target": "end", "targetHandle": "content"},
                {"source": "read_pdf", "sourceHandle": "truncated", "target": "end", "targetHandle": "truncated"},
                {"source": "read_pdf", "sourceHandle": "warnings", "target": "end", "targetHandle": "warnings"},
            ],
        }
    )

    workflow = compile_flow(visual_to_flow(vf))
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(
        workflow=workflow,
        vars={"workspace_root": str(workspace), "workspace_access_mode": "workspace_only"},
    )
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=20)

    assert state.status.value == "completed"
    assert len(state.output["content"]) == 32
    assert state.output["truncated"] is True
    assert any("#TRUNCATION" in str(item) for item in state.output["warnings"])


def test_read_file_without_workspace_scope_falls_back_to_process_cwd(tmp_path: Path, monkeypatch) -> None:
    relpath = "notes/local.txt"
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    (tmp_path / relpath).write_text("local fallback\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    vf = load_visualflow_json(
        {
            "id": "test-read-file-cwd-fallback",
            "name": "test-read-file-cwd-fallback",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "read_file",
                    "type": "read_file",
                    "data": {"nodeType": "read_file", "pinDefaults": {"file_path": relpath}},
                },
                {
                    "id": "end",
                    "type": "on_flow_end",
                    "data": {
                        "nodeType": "on_flow_end",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "content", "label": "content", "type": "string"},
                        ],
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "read_file", "targetHandle": "exec-in"},
                {"source": "read_file", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
                {"source": "read_file", "sourceHandle": "content", "target": "end", "targetHandle": "content"},
            ],
        }
    )

    workflow = compile_flow(visual_to_flow(vf))
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(workflow=workflow, vars={})
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=20)

    assert state.status.value == "completed"
    assert state.output == {"content": "local fallback\n", "success": True}


def test_write_file_all_except_ignored_canonicalizes_absolute_paths_under_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "outside"
    workspace.mkdir(parents=True, exist_ok=True)
    abs_path = workspace / "notes" / "secret.txt"

    vf = load_visualflow_json(
        {
            "id": "test-write-file-abs-canonical",
            "name": "test-write-file-abs-canonical",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "write_file",
                    "type": "write_file",
                    "data": {
                        "nodeType": "write_file",
                        "pinDefaults": {"file_path": str(abs_path), "content": "secret\n"},
                    },
                },
                {
                    "id": "end",
                    "type": "on_flow_end",
                    "data": {
                        "nodeType": "on_flow_end",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "file_path", "label": "file_path", "type": "string"},
                        ],
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "write_file", "targetHandle": "exec-in"},
                {"source": "write_file", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
                {"source": "write_file", "sourceHandle": "file_path", "target": "end", "targetHandle": "file_path"},
            ],
        }
    )

    workflow = compile_flow(visual_to_flow(vf))
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    run_id = runtime.start(
        workflow=workflow,
        vars={"workspace_root": str(workspace), "workspace_access_mode": "all_except_ignored"},
    )
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=20)

    assert state.status.value == "completed"
    assert state.output["file_path"] == "notes/secret.txt"
    assert abs_path.read_text(encoding="utf-8") == "secret\n"


def test_list_folder_files_filters_workspace_entries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "keep.md").write_text("# keep\n", encoding="utf-8")
    (workspace / "docs" / "skip.json").write_text('{"skip":true}\n', encoding="utf-8")
    (workspace / "docs" / "nested").mkdir()
    (workspace / "docs" / "nested" / "deep.md").write_text("# deep\n", encoding="utf-8")

    vf = load_visualflow_json(
        {
            "id": "test-list-folder-files",
            "name": "test-list-folder-files",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "list_folder",
                    "type": "list_folder_files",
                    "data": {
                        "nodeType": "list_folder_files",
                        "pinDefaults": {
                            "folder_path": "docs",
                            "recursive": True,
                            "extensions": ["md"],
                        },
                    },
                },
                {
                    "id": "end",
                    "type": "on_flow_end",
                    "data": {
                        "nodeType": "on_flow_end",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "files", "label": "files", "type": "array"},
                            {"id": "count", "label": "count", "type": "number"},
                            {"id": "folder_path", "label": "folder_path", "type": "string"},
                        ],
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "list_folder", "targetHandle": "exec-in"},
                {"source": "list_folder", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
                {"source": "list_folder", "sourceHandle": "files", "target": "end", "targetHandle": "files"},
                {"source": "list_folder", "sourceHandle": "count", "target": "end", "targetHandle": "count"},
                {"source": "list_folder", "sourceHandle": "folder_path", "target": "end", "targetHandle": "folder_path"},
            ],
        }
    )

    workflow = compile_flow(visual_to_flow(vf))
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    run_id = runtime.start(
        workflow=workflow,
        vars={"workspace_root": str(workspace), "workspace_access_mode": "workspace_only"},
    )
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=20)

    assert state.status.value == "completed"
    assert state.output["folder_path"] == "docs"
    assert state.output["count"] == 2
    assert state.output["files"] == ["docs/keep.md", "docs/nested/deep.md"]


def test_import_read_and_export_artifact_round_trip(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source").mkdir()
    (workspace / "exports").mkdir()
    (workspace / "source" / "notes.txt").write_text("artifact round trip\n", encoding="utf-8")

    vf = load_visualflow_json(
        {
            "id": "test-import-read-export-artifact",
            "name": "test-import-read-export-artifact",
            "nodes": [
                {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "import_file",
                    "type": "import_workspace_file",
                    "data": {"nodeType": "import_workspace_file", "pinDefaults": {"file_path": "source/notes.txt"}},
                },
                {"id": "read_artifact", "type": "read_artifact", "data": {"nodeType": "read_artifact"}},
                {
                    "id": "export_artifact",
                    "type": "export_artifact",
                    "data": {"nodeType": "export_artifact", "pinDefaults": {"file_path": "exports/copy.txt"}},
                },
                {
                    "id": "end",
                    "type": "on_flow_end",
                    "data": {
                        "nodeType": "on_flow_end",
                        "inputs": [
                            {"id": "exec-in", "label": "", "type": "execution"},
                            {"id": "text", "label": "text", "type": "string"},
                            {"id": "artifact_id", "label": "artifact_id", "type": "string"},
                            {"id": "export_path", "label": "export_path", "type": "string"},
                            {"id": "content_type", "label": "content_type", "type": "string"},
                        ],
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "import_file", "targetHandle": "exec-in"},
                {"source": "import_file", "sourceHandle": "exec-out", "target": "read_artifact", "targetHandle": "exec-in"},
                {"source": "read_artifact", "sourceHandle": "exec-out", "target": "export_artifact", "targetHandle": "exec-in"},
                {"source": "export_artifact", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
                {"source": "import_file", "sourceHandle": "artifact", "target": "read_artifact", "targetHandle": "artifact"},
                {"source": "import_file", "sourceHandle": "artifact", "target": "export_artifact", "targetHandle": "artifact"},
                {"source": "read_artifact", "sourceHandle": "text", "target": "end", "targetHandle": "text"},
                {"source": "read_artifact", "sourceHandle": "artifact_id", "target": "end", "targetHandle": "artifact_id"},
                {"source": "export_artifact", "sourceHandle": "file_path", "target": "end", "targetHandle": "export_path"},
                {"source": "export_artifact", "sourceHandle": "content_type", "target": "end", "targetHandle": "content_type"},
            ],
        }
    )

    workflow = compile_flow(visual_to_flow(vf))
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        artifact_store=InMemoryArtifactStore(),
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={"workspace_root": str(workspace), "workspace_access_mode": "workspace_only"},
        session_id="artifact-session",
    )
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=20)

    assert state.status.value == "completed"
    assert state.output["text"] == "artifact round trip\n"
    assert state.output["artifact_id"]
    assert state.output["export_path"] == "exports/copy.txt"
    assert state.output["content_type"] == "text/plain"
    assert (workspace / "exports" / "copy.txt").read_text(encoding="utf-8") == "artifact round trip\n"
