"""Tests for artifact storage.

Tests cover:
- InMemoryArtifactStore: store, load, delete, list_by_run
- FileArtifactStore: persistence, content-addressed IDs
- Convenience methods: store_text, store_json, load_text, load_json
- Artifact references: artifact_ref, is_artifact_ref, resolve_artifact
- Integration with Runtime and workflows
"""

import json
import tempfile
from pathlib import Path

import pytest

from abstractruntime import (
    Artifact,
    ArtifactMetadata,
    ArtifactStore,
    InMemoryArtifactStore,
    FileArtifactStore,
    artifact_ref,
    is_artifact_ref,
    get_artifact_id,
    resolve_artifact,
    Runtime,
    RunState,
    RunStatus,
    StepPlan,
    Effect,
    EffectType,
    WorkflowSpec,
    InMemoryRunStore,
    InMemoryLedgerStore,
    create_scheduled_runtime,
)


# -----------------------------------------------------------------------------
# Tests: InMemoryArtifactStore
# -----------------------------------------------------------------------------


class TestInMemoryArtifactStore:
    """Tests for in-memory artifact storage."""

    def test_store_and_load_bytes(self):
        """Can store and load binary content."""
        store = InMemoryArtifactStore()
        content = b"Hello, World!"

        metadata = store.store(content, content_type="text/plain")

        assert metadata.artifact_id is not None
        assert metadata.content_type == "text/plain"
        assert metadata.size_bytes == len(content)

        artifact = store.load(metadata.artifact_id)
        assert artifact is not None
        assert artifact.content == content
        assert artifact.content_type == "text/plain"

    def test_content_addressed_id(self):
        """Same content produces same artifact ID."""
        store = InMemoryArtifactStore()
        content = b"Deterministic content"

        meta1 = store.store(content)
        meta2 = store.store(content)

        assert meta1.artifact_id == meta2.artifact_id

    def test_different_content_different_id(self):
        """Different content produces different artifact IDs."""
        store = InMemoryArtifactStore()

        meta1 = store.store(b"Content A")
        meta2 = store.store(b"Content B")

        assert meta1.artifact_id != meta2.artifact_id

    def test_explicit_artifact_id(self):
        """Can provide explicit artifact ID."""
        store = InMemoryArtifactStore()

        metadata = store.store(b"content", artifact_id="my-custom-id")

        assert metadata.artifact_id == "my-custom-id"
        assert store.exists("my-custom-id")

    def test_exists(self):
        """Can check if artifact exists."""
        store = InMemoryArtifactStore()

        assert not store.exists("nonexistent")

        metadata = store.store(b"content")
        assert store.exists(metadata.artifact_id)

    def test_delete(self):
        """Can delete artifacts."""
        store = InMemoryArtifactStore()

        metadata = store.store(b"content")
        assert store.exists(metadata.artifact_id)

        deleted = store.delete(metadata.artifact_id)
        assert deleted is True
        assert not store.exists(metadata.artifact_id)

        # Delete nonexistent returns False
        deleted = store.delete("nonexistent")
        assert deleted is False

    def test_get_metadata(self):
        """Can get metadata without loading content."""
        store = InMemoryArtifactStore()
        content = b"x" * 10000  # 10KB

        metadata = store.store(content, content_type="application/octet-stream")

        loaded_meta = store.get_metadata(metadata.artifact_id)
        assert loaded_meta is not None
        assert loaded_meta.artifact_id == metadata.artifact_id
        assert loaded_meta.size_bytes == 10000

    def test_list_by_run(self):
        """Can list artifacts by run ID."""
        store = InMemoryArtifactStore()

        # Store artifacts for different runs
        store.store(b"run1-a", run_id="run-1")
        store.store(b"run1-b", run_id="run-1")
        store.store(b"run2-a", run_id="run-2")
        store.store(b"no-run")

        run1_artifacts = store.list_by_run("run-1")
        assert len(run1_artifacts) == 2

        run2_artifacts = store.list_by_run("run-2")
        assert len(run2_artifacts) == 1

        run3_artifacts = store.list_by_run("run-3")
        assert len(run3_artifacts) == 0

    def test_tags(self):
        """Can store and retrieve tags."""
        store = InMemoryArtifactStore()

        metadata = store.store(
            b"content",
            tags={"source": "llm", "model": "gpt-4"},
        )

        loaded = store.get_metadata(metadata.artifact_id)
        assert loaded.tags["source"] == "llm"
        assert loaded.tags["model"] == "gpt-4"


# -----------------------------------------------------------------------------
# Tests: Convenience Methods
# -----------------------------------------------------------------------------


class TestArtifactConvenienceMethods:
    """Tests for store_text, store_json, etc."""

    def test_store_text(self):
        """Can store and load text."""
        store = InMemoryArtifactStore()

        metadata = store.store_text("Hello, World!")

        assert metadata.content_type == "text/plain"

        text = store.load_text(metadata.artifact_id)
        assert text == "Hello, World!"

    def test_store_json(self):
        """Can store and load JSON."""
        store = InMemoryArtifactStore()
        data = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}

        metadata = store.store_json(data)

        assert metadata.content_type == "application/json"

        loaded = store.load_json(metadata.artifact_id)
        assert loaded == data

    def test_store_text_with_encoding(self):
        """Can store text with custom encoding."""
        store = InMemoryArtifactStore()

        # UTF-8 with special characters
        text = "Héllo, 世界! 🌍"
        metadata = store.store_text(text)

        loaded = store.load_text(metadata.artifact_id)
        assert loaded == text

    def test_load_nonexistent_returns_none(self):
        """Loading nonexistent artifact returns None."""
        store = InMemoryArtifactStore()

        assert store.load("nonexistent") is None
        assert store.load_text("nonexistent") is None
        assert store.load_json("nonexistent") is None


# -----------------------------------------------------------------------------
# Tests: FileArtifactStore
# -----------------------------------------------------------------------------


class TestFileArtifactStore:
    """Tests for file-based artifact storage."""

    def test_store_and_load(self):
        """Can store and load from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            content = b"Persistent content"

            metadata = store.store(content, content_type="text/plain")

            # Verify files exist
            artifacts_dir = Path(tmpdir) / "artifacts"
            assert (artifacts_dir / f"{metadata.artifact_id}.bin").exists()
            assert (artifacts_dir / f"{metadata.artifact_id}.meta").exists()

            # Load and verify
            artifact = store.load(metadata.artifact_id)
            assert artifact.content == content

    def test_persistence_across_instances(self):
        """Artifacts persist across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Store with first instance
            store1 = FileArtifactStore(tmpdir)
            metadata = store1.store_json({"persistent": True})
            artifact_id = metadata.artifact_id

            # Load with second instance
            store2 = FileArtifactStore(tmpdir)
            data = store2.load_json(artifact_id)

            assert data == {"persistent": True}

    def test_same_content_different_runs_do_not_collide(self):
        """Same bytes stored for different runs must not overwrite metadata (run-scoped ids)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            m1 = store.store(b"same-content", run_id="run-1")
            m2 = store.store(b"same-content", run_id="run-2")

            assert m1.artifact_id != m2.artifact_id

            run1 = store.list_by_run("run-1")
            run2 = store.list_by_run("run-2")
            assert {m.artifact_id for m in run1} == {m1.artifact_id}
            assert {m.artifact_id for m in run2} == {m2.artifact_id}

            meta1 = store.get_metadata(m1.artifact_id)
            meta2 = store.get_metadata(m2.artifact_id)
            assert meta1 is not None and meta1.run_id == "run-1"
            assert meta2 is not None and meta2.run_id == "run-2"

    def test_delete_removes_files(self):
        """Delete removes both content and metadata files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            metadata = store.store(b"content")

            artifacts_dir = Path(tmpdir) / "artifacts"
            content_path = artifacts_dir / f"{metadata.artifact_id}.bin"
            meta_path = artifacts_dir / f"{metadata.artifact_id}.meta"

            assert content_path.exists()
            assert meta_path.exists()

            store.delete(metadata.artifact_id)

            assert not content_path.exists()
            assert not meta_path.exists()

    def test_list_by_run(self):
        """Can list artifacts by run ID from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            store.store(b"a", run_id="run-1")
            store.store(b"b", run_id="run-1")
            store.store(b"c", run_id="run-2")

            run1 = store.list_by_run("run-1")
            assert len(run1) == 2


# -----------------------------------------------------------------------------
# Tests: Artifact References
# -----------------------------------------------------------------------------


class TestArtifactReferences:
    """Tests for artifact reference helpers."""

    def test_artifact_ref(self):
        """Can create artifact reference."""
        ref = artifact_ref("abc123")
        assert ref == {"$artifact": "abc123"}

    def test_is_artifact_ref(self):
        """Can detect artifact references."""
        assert is_artifact_ref({"$artifact": "abc123"})
        assert not is_artifact_ref({"other": "value"})
        assert not is_artifact_ref("string")
        assert not is_artifact_ref(123)
        assert not is_artifact_ref(None)

    def test_get_artifact_id(self):
        """Can extract artifact ID from reference."""
        ref = artifact_ref("my-id")
        assert get_artifact_id(ref) == "my-id"

    def test_resolve_artifact(self):
        """Can resolve artifact reference to content."""
        store = InMemoryArtifactStore()
        metadata = store.store_json({"resolved": True})

        ref = artifact_ref(metadata.artifact_id)
        artifact = resolve_artifact(ref, store)

        assert artifact is not None
        assert artifact.as_json() == {"resolved": True}

    def test_resolve_nonexistent(self):
        """Resolving nonexistent artifact returns None."""
        store = InMemoryArtifactStore()
        ref = artifact_ref("nonexistent")

        artifact = resolve_artifact(ref, store)
        assert artifact is None


# -----------------------------------------------------------------------------
# Tests: Integration with Runtime
# -----------------------------------------------------------------------------


class TestArtifactRuntimeIntegration:
    """Tests for artifact store integration with Runtime."""

    def test_runtime_has_artifact_store(self):
        """Runtime can be configured with artifact store."""
        artifact_store = InMemoryArtifactStore()

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            artifact_store=artifact_store,
        )

        assert runtime.artifact_store is artifact_store

    def test_set_artifact_store(self):
        """Can set artifact store after construction."""
        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
        )

        assert runtime.artifact_store is None

        artifact_store = InMemoryArtifactStore()
        runtime.set_artifact_store(artifact_store)

        assert runtime.artifact_store is artifact_store

    def test_scheduled_runtime_with_artifact_store(self):
        """ScheduledRuntime can be created with artifact store."""
        artifact_store = InMemoryArtifactStore()

        sr = create_scheduled_runtime(
            artifact_store=artifact_store,
            auto_start=False,
        )

        assert sr.runtime.artifact_store is artifact_store


# -----------------------------------------------------------------------------
# Tests: Workflow Integration
# -----------------------------------------------------------------------------


class TestArtifactWorkflowIntegration:
    """Tests for using artifacts in workflows."""

    def test_store_large_result_as_artifact(self):
        """Workflow can store large results as artifacts."""
        artifact_store = InMemoryArtifactStore()

        # Simulate a workflow that produces large output
        large_data = {"items": [f"item-{i}" for i in range(1000)]}

        def process_node(run: RunState, ctx) -> StepPlan:
            # Store large result as artifact
            metadata = artifact_store.store_json(large_data, run_id=run.run_id)

            # Store only the reference in vars
            return StepPlan(
                node_id="process",
                complete_output={
                    "result_ref": artifact_ref(metadata.artifact_id),
                    "artifact_id": metadata.artifact_id,
                },
            )

        workflow = WorkflowSpec(
            workflow_id="artifact_workflow",
            entry_node="process",
            nodes={"process": process_node},
        )

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            artifact_store=artifact_store,
        )

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED

        # Output contains reference, not the large data
        assert is_artifact_ref(state.output["result_ref"])

        # Can resolve the reference to get the data
        ref = state.output["result_ref"]
        artifact = resolve_artifact(ref, artifact_store)
        assert artifact.as_json() == large_data

        # Artifact is associated with the run
        run_artifacts = artifact_store.list_by_run(run_id)
        assert len(run_artifacts) == 1

    def test_pass_artifact_between_nodes(self):
        """Artifacts can be passed between workflow nodes."""
        artifact_store = InMemoryArtifactStore()

        def producer_node(run: RunState, ctx) -> StepPlan:
            data = {"produced": "data", "size": 1000}
            metadata = artifact_store.store_json(data, run_id=run.run_id)
            run.vars["data_ref"] = artifact_ref(metadata.artifact_id)
            return StepPlan(node_id="producer", next_node="consumer")

        def consumer_node(run: RunState, ctx) -> StepPlan:
            ref = run.vars["data_ref"]
            artifact = resolve_artifact(ref, artifact_store)
            data = artifact.as_json()

            return StepPlan(
                node_id="consumer",
                complete_output={"consumed": data["produced"], "size": data["size"]},
            )

        workflow = WorkflowSpec(
            workflow_id="producer_consumer",
            entry_node="producer",
            nodes={"producer": producer_node, "consumer": consumer_node},
        )

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            artifact_store=artifact_store,
        )

        run_id = runtime.start(workflow=workflow)
        state = runtime.tick(workflow=workflow, run_id=run_id)

        assert state.status == RunStatus.COMPLETED
        assert state.output["consumed"] == "data"
        assert state.output["size"] == 1000


# -----------------------------------------------------------------------------
# Tests: Edge Cases
# -----------------------------------------------------------------------------


class TestArtifactEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self):
        """Can store empty content."""
        store = InMemoryArtifactStore()

        metadata = store.store(b"")
        assert metadata.size_bytes == 0

        artifact = store.load(metadata.artifact_id)
        assert artifact.content == b""

    def test_large_content(self):
        """Can store large content."""
        store = InMemoryArtifactStore()

        # 1MB of data
        content = b"x" * (1024 * 1024)
        metadata = store.store(content)

        assert metadata.size_bytes == 1024 * 1024

        artifact = store.load(metadata.artifact_id)
        assert len(artifact.content) == 1024 * 1024

    def test_binary_content(self):
        """Can store binary content with null bytes."""
        store = InMemoryArtifactStore()

        content = bytes(range(256))  # All byte values 0-255
        metadata = store.store(content, content_type="application/octet-stream")

        artifact = store.load(metadata.artifact_id)
        assert artifact.content == content

    def test_unicode_in_json(self):
        """Can store JSON with unicode characters."""
        store = InMemoryArtifactStore()

        data = {
            "emoji": "🎉🚀",
            "chinese": "你好世界",
            "arabic": "مرحبا",
            "math": "∑∏∫",
        }

        metadata = store.store_json(data)
        loaded = store.load_json(metadata.artifact_id)

        assert loaded == data


# -----------------------------------------------------------------------------
# Tests: New Functionality
# -----------------------------------------------------------------------------


class TestArtifactListAll:
    """Tests for list_all functionality."""

    def test_list_all_empty(self):
        """list_all returns empty list when no artifacts."""
        store = InMemoryArtifactStore()
        assert store.list_all() == []

    def test_list_all_returns_all(self):
        """list_all returns all artifacts."""
        store = InMemoryArtifactStore()

        store.store(b"a")
        store.store(b"b")
        store.store(b"c")

        all_artifacts = store.list_all()
        assert len(all_artifacts) == 3

    def test_list_all_respects_limit(self):
        """list_all respects limit parameter."""
        store = InMemoryArtifactStore()

        for i in range(10):
            store.store(f"content-{i}".encode())

        limited = store.list_all(limit=5)
        assert len(limited) == 5

    def test_list_all_file_store(self):
        """list_all works with file store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            store.store(b"a")
            store.store(b"b")

            all_artifacts = store.list_all()
            assert len(all_artifacts) == 2


class TestArtifactSearch:
    """Tests for metadata-based search() helper."""

    def test_search_filters(self):
        store = InMemoryArtifactStore()

        a = store.store(b"a", run_id="run-1", content_type="text/plain", tags={"kind": "note"})
        store.store(b"b", run_id="run-1", content_type="application/json", tags={"kind": "data"})
        store.store(b"c", run_id="run-2", content_type="text/plain", tags={"kind": "note"})

        res = store.search(run_id="run-1", content_type="text/plain")
        assert [m.artifact_id for m in res] == [a.artifact_id]

        res = store.search(tags={"kind": "note"})
        assert {m.artifact_id for m in res} == {a.artifact_id, store.search(run_id="run-2")[0].artifact_id}

    def test_search_file_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            meta = store.store(b"x", run_id="run-1", content_type="text/plain", tags={"k": "v"})
            res = store.search(run_id="run-1", tags={"k": "v"})
            assert [m.artifact_id for m in res] == [meta.artifact_id]


class TestArtifactDeleteByRun:
    """Tests for delete_by_run functionality."""

    def test_delete_by_run(self):
        """delete_by_run removes all artifacts for a run."""
        store = InMemoryArtifactStore()

        store.store(b"a", run_id="run-1")
        store.store(b"b", run_id="run-1")
        store.store(b"c", run_id="run-2")

        deleted = store.delete_by_run("run-1")
        assert deleted == 2

        # run-1 artifacts gone
        assert len(store.list_by_run("run-1")) == 0

        # run-2 artifact still exists
        assert len(store.list_by_run("run-2")) == 1

    def test_delete_by_run_nonexistent(self):
        """delete_by_run returns 0 for nonexistent run."""
        store = InMemoryArtifactStore()
        deleted = store.delete_by_run("nonexistent")
        assert deleted == 0


class TestArtifactIdValidation:
    """Tests for artifact ID validation."""

    def test_valid_artifact_ids(self):
        """Valid artifact IDs are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            # These should all work
            store.store(b"a", artifact_id="abc123")
            store.store(b"b", artifact_id="my-artifact")
            store.store(b"c", artifact_id="test_artifact")
            store.store(b"d", artifact_id="ABC-123_test")

            assert store.exists("abc123")
            assert store.exists("my-artifact")
            assert store.exists("test_artifact")
            assert store.exists("ABC-123_test")

    def test_invalid_artifact_id_path_traversal(self):
        """Path traversal attempts are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            with pytest.raises(ValueError, match="Invalid artifact_id"):
                store.store(b"evil", artifact_id="../../../etc/passwd")

    def test_invalid_artifact_id_special_chars(self):
        """Special characters in artifact ID are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            with pytest.raises(ValueError, match="Invalid artifact_id"):
                store.store(b"evil", artifact_id="test/file")

            with pytest.raises(ValueError, match="Invalid artifact_id"):
                store.store(b"evil", artifact_id="test.file")

    def test_empty_artifact_id_rejected(self):
        """Empty artifact ID is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            with pytest.raises(ValueError, match="cannot be empty"):
                store.store(b"content", artifact_id="")


class TestComputeArtifactId:
    """Tests for compute_artifact_id function."""

    def test_compute_artifact_id_deterministic(self):
        """Same content produces same ID."""
        from abstractruntime import compute_artifact_id

        content = b"test content"
        id1 = compute_artifact_id(content)
        id2 = compute_artifact_id(content)

        assert id1 == id2
        assert len(id1) == 32  # First 32 chars of SHA-256

    def test_compute_artifact_id_different_content(self):
        """Different content produces different IDs."""
        from abstractruntime import compute_artifact_id

        id1 = compute_artifact_id(b"content a")
        id2 = compute_artifact_id(b"content b")

        assert id1 != id2

    def test_compute_artifact_id_namespaced_by_run_id(self):
        """Same content produces different IDs when run_id differs (run-scoped addressing)."""
        from abstractruntime import compute_artifact_id

        content = b"same bytes"
        id1 = compute_artifact_id(content, run_id="run-1")
        id2 = compute_artifact_id(content, run_id="run-2")
        id3 = compute_artifact_id(content, run_id="run-1")

        assert id1 != id2
        assert id1 == id3

    def test_compute_artifact_id_check_before_store(self):
        """Can check if content exists before storing."""
        from abstractruntime import compute_artifact_id

        store = InMemoryArtifactStore()
        content = b"check this content"

        # Compute ID first
        artifact_id = compute_artifact_id(content)

        # Check if exists
        assert not store.exists(artifact_id)

        # Store it
        store.store(content)

        # Now it exists
        assert store.exists(artifact_id)
