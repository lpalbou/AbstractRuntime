"""Tests for artifact storage.

Tests cover:
- InMemoryArtifactStore: store, load, delete, list_by_run
- FileArtifactStore: persistence, content-addressed IDs
- Convenience methods: store_text, store_json, load_text, load_json
- Artifact references: artifact_ref, is_artifact_ref, resolve_artifact
- Integration with Runtime and workflows
"""

import json
import io
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from abstractruntime import (
    Artifact,
    ArtifactDescriptor,
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


def _wav_bytes(*, seconds: float = 0.25, sample_rate: int = 8000, channels: int = 1) -> bytes:
    frames = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\0\0" * frames * channels)
    return buf.getvalue()


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

    def test_code_artifact_descriptor_from_mime_and_tags(self):
        """Code artifacts are classified separately from generic text."""
        store = InMemoryArtifactStore()

        metadata = store.store(
            b"print('hello')\n",
            content_type="text/x-python",
            tags={"kind": "source_code"},
        )

        assert metadata.descriptor.render_kind == "code"
        assert metadata.descriptor.semantic_kind == "code"


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
            assert (artifacts_dir / "refs" / f"{metadata.artifact_id}.meta").exists()
            assert isinstance(metadata.blob_id, str) and metadata.blob_id
            assert (artifacts_dir / "blobs" / f"{metadata.blob_id}.bin").exists()

            # Load and verify
            artifact = store.load(metadata.artifact_id)
            assert artifact.content == content

    def test_content_path_is_public_for_file_backed_store(self):
        """File stores expose a stable content path; memory stores do not."""
        memory_store = InMemoryArtifactStore()
        memory_meta = memory_store.store(b"in memory", content_type="text/plain")
        assert memory_store.content_path(memory_meta.artifact_id) is None

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            metadata = store.store(b"Persistent content", content_type="text/plain", run_id="run-1")
            path = store.content_path(metadata.artifact_id)
            assert path is not None
            assert path.exists()
            assert path.read_bytes() == b"Persistent content"

            restarted = FileArtifactStore(tmpdir)
            restarted_path = restarted.content_path(metadata.artifact_id)
            assert restarted_path is not None
            assert restarted_path.exists()
            assert restarted_path.read_bytes() == b"Persistent content"

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
            assert m1.blob_id == m2.blob_id

            run1 = store.list_by_run("run-1")
            run2 = store.list_by_run("run-2")
            assert {m.artifact_id for m in run1} == {m1.artifact_id}
            assert {m.artifact_id for m in run2} == {m2.artifact_id}

            meta1 = store.get_metadata(m1.artifact_id)
            meta2 = store.get_metadata(m2.artifact_id)
            assert meta1 is not None and meta1.run_id == "run-1"
            assert meta2 is not None and meta2.run_id == "run-2"

    def test_delete_removes_files(self):
        """Delete removes ref metadata; blobs are cleaned up by gc()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            metadata = store.store(b"content")

            artifacts_dir = Path(tmpdir) / "artifacts"
            blob_path = artifacts_dir / "blobs" / f"{metadata.blob_id}.bin"
            meta_path = artifacts_dir / "refs" / f"{metadata.artifact_id}.meta"

            assert blob_path.exists()
            assert meta_path.exists()

            store.delete(metadata.artifact_id)

            assert not meta_path.exists()
            assert blob_path.exists()

            report = store.gc(dry_run=False)
            assert report.get("blobs_deleted") == 1
            assert not blob_path.exists()

    def test_list_by_run(self):
        """Can list artifacts by run ID from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            store.store(b"a", run_id="run-1")
            store.store(b"b", run_id="run-1")
            store.store(b"c", run_id="run-2")

            run1 = store.list_by_run("run-1")
            assert len(run1) == 2

    def test_gc_respects_cross_run_refs(self):
        """GC must not delete a shared blob while another run still references it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)

            m1 = store.store(b"shared", run_id="run-1")
            m2 = store.store(b"shared", run_id="run-2")
            assert m1.blob_id == m2.blob_id

            artifacts_dir = Path(tmpdir) / "artifacts"
            blob_path = artifacts_dir / "blobs" / f"{m1.blob_id}.bin"
            assert blob_path.exists()

            deleted = store.delete_by_run("run-1")
            assert deleted == 1

            report = store.gc(dry_run=False)
            assert report.get("blobs_deleted") == 0
            assert blob_path.exists()

            deleted = store.delete_by_run("run-2")
            assert deleted == 1

            report = store.gc(dry_run=False)
            assert report.get("blobs_deleted") == 1
            assert not blob_path.exists()


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


class TestArtifactDescriptorAndCatalog:
    """Tests for ArtifactDescriptorV1-compatible metadata and catalog behavior."""

    def test_store_descriptor_metadata_and_access_in_memory(self):
        store = InMemoryArtifactStore()

        meta = store.store(
            b"voice",
            content_type="audio/wav",
            run_id="run-1",
            tags={"session_id": "s1", "workflow_id": "wf", "node_id": "node-1"},
            metadata={"provider": "p", "model": "m"},
            descriptor={
                "semantic_kind": "voice",
                "render_kind": "audio",
                "generation": {"input_text": "hello"},
            },
        )

        assert meta.descriptor.schema_version == 1
        assert meta.descriptor.semantic_kind == "voice"
        assert meta.descriptor.render_kind == "audio"
        assert meta.descriptor.session_id == "s1"
        assert meta.metadata == {"provider": "p", "model": "m"}

        updated = store.record_access(
            meta.artifact_id,
            action="preview",
            actor_id="actor",
            session_id="s1",
            run_id="run-1",
            at="2026-06-06T00:00:00+00:00",
        )
        assert updated is not None
        assert updated.access.access_count == 1
        assert updated.access.preview_count == 1
        assert updated.access.last_action == "preview"
        assert updated.access.last_actor_id == "actor"
        stats = store.stats(facets=["semantic_kind"], run_id="run-1")
        assert stats["total"] == 1
        assert stats["total_bytes"] == len(b"voice")
        assert stats["facets"]["semantic_kind"] == {"voice": 1}

    def test_runtime_tags_project_to_descriptor_for_voice_and_music(self):
        store = InMemoryArtifactStore()

        voice = store.store(b"v", content_type="audio/wav", tags={"modality": "voice", "task": "tts"})
        music = store.store(b"m", content_type="audio/wav", tags={"modality": "music", "task": "music_generation"})

        assert voice.descriptor.render_kind == "audio"
        assert voice.descriptor.semantic_kind == "voice"
        assert voice.descriptor.classification_source == "runtime_tags"
        assert music.descriptor.render_kind == "audio"
        assert music.descriptor.semantic_kind == "music"

    def test_file_store_persists_descriptor_access_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            meta = store.store(
                _wav_bytes(seconds=0.5, sample_rate=8000),
                content_type="audio/wav",
                run_id="run-music",
                tags={
                    "session_id": "s-music",
                    "workflow_id": "wf_music",
                    "node_id": "call",
                    "modality": "music",
                    "task": "music_generation",
                },
                metadata={"provider": "abstractmusic", "model": "acestep"},
            )
            store.record_access(meta.artifact_id, action="download", session_id="s-music")

            restarted = FileArtifactStore(tmpdir)
            loaded = restarted.get_metadata(meta.artifact_id)

            assert loaded is not None
            assert loaded.descriptor.semantic_kind == "music"
            assert loaded.descriptor.render_kind == "audio"
            assert loaded.descriptor.session_id == "s-music"
            assert loaded.descriptor.workflow_id == "wf_music"
            assert loaded.descriptor.media["duration_s"] == pytest.approx(0.5)
            assert loaded.descriptor.media["sample_rate"] == 8000
            assert loaded.access.access_count == 1
            assert loaded.access.download_count == 1
            assert loaded.metadata["provider"] == "abstractmusic"

            by_kind = restarted.search(semantic_kind="music", session_id="s-music", limit=0)
            assert [m.artifact_id for m in by_kind] == [meta.artifact_id]
            assert restarted.count(semantic_kind="music") == 1
            assert restarted.facet_counts("semantic_kind") == {"music": 1}
            stats = restarted.stats(facets=["semantic_kind", "render_kind"], semantic_kind="music", session_id="s-music")
            assert stats["total"] == 1
            assert stats["total_bytes"] == meta.size_bytes
            assert stats["facets"]["semantic_kind"] == {"music": 1}
            assert stats["facets"]["render_kind"] == {"audio": 1}
            assert stats["source"] == "artifact_catalog"

    def test_update_metadata_reindexes_file_catalog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            meta = store.store(b"x", content_type="text/plain", tags={"session_id": "s1"})

            updated = store.update_metadata(
                meta.artifact_id,
                descriptor=ArtifactDescriptor(
                    semantic_kind="evidence",
                    render_kind="text",
                    workflow_id="wf",
                    node_id="n1",
                ),
                metadata={"source_url": "https://example.test"},
            )

            assert updated is not None
            assert updated.descriptor.semantic_kind == "evidence"
            assert store.search(semantic_kind="evidence")[0].artifact_id == meta.artifact_id
            assert store.search(workflow_id="wf")[0].artifact_id == meta.artifact_id
            assert store.get_metadata(meta.artifact_id).metadata["source_url"] == "https://example.test"

    def test_descriptor_update_preserves_existing_media_facts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            meta = store.store(
                _wav_bytes(seconds=0.25, sample_rate=8000),
                content_type="audio/wav",
                tags={"modality": "music", "task": "music_generation"},
            )

            updated = store.update_metadata(
                meta.artifact_id,
                descriptor={"workflow_id": "wf-music", "node_id": "call"},
            )

            assert updated is not None
            assert updated.descriptor.workflow_id == "wf-music"
            assert updated.descriptor.media["duration_s"] == pytest.approx(0.25)
            restarted = FileArtifactStore(tmpdir)
            reloaded = restarted.get_metadata(meta.artifact_id)
            assert reloaded is not None
            assert reloaded.descriptor.media["sample_rate"] == 8000
            assert restarted.search(workflow_id="wf-music")[0].artifact_id == meta.artifact_id

    def test_file_store_access_updates_are_thread_locked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(tmpdir)
            meta = store.store(b"x", content_type="text/plain", run_id="run-1")

            def touch(i: int) -> None:
                updated = store.record_access(
                    meta.artifact_id,
                    action="preview",
                    actor_id=f"actor-{i}",
                    run_id="run-1",
                )
                assert updated is not None

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(touch, range(20)))

            loaded = store.get_metadata(meta.artifact_id)
            assert loaded is not None
            assert loaded.access.access_count == 20
            assert loaded.access.preview_count == 20
            assert store.count(run_id="run-1") == 1

    def test_legacy_metadata_file_projects_descriptor_on_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            refs = base / "artifacts" / "refs"
            blobs = base / "artifacts" / "blobs"
            refs.mkdir(parents=True)
            blobs.mkdir(parents=True)
            content = b"legacy"
            blob_id = "abc123"
            (blobs / f"{blob_id}.bin").write_bytes(content)
            legacy = {
                "artifact_id": "legacy1",
                "blob_id": blob_id,
                "content_type": "audio/wav",
                "size_bytes": len(content),
                "created_at": "2026-06-06T00:00:00+00:00",
                "run_id": "run-legacy",
                "tags": {"modality": "voice", "task": "tts", "session_id": "s-legacy"},
            }
            (refs / "legacy1.meta").write_text(json.dumps(legacy), encoding="utf-8")

            store = FileArtifactStore(tmpdir)
            meta = store.get_metadata("legacy1")

            assert meta is not None
            assert meta.descriptor.semantic_kind == "voice"
            assert meta.descriptor.render_kind == "audio"
            assert meta.descriptor.session_id == "s-legacy"
            assert store.search(semantic_kind="voice")[0].artifact_id == "legacy1"


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
                store.store(b"evil", artifact_id="../../../SENSITIVE_FILE")

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
