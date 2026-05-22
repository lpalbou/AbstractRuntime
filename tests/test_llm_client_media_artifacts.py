"""Tests for artifact-backed media resolution in the AbstractCore LLM client."""

import os

from abstractruntime.integrations.abstractcore.llm_client import _resolve_media_artifacts
from abstractruntime.storage.artifacts import InMemoryArtifactStore


def test_resolve_media_artifact_creates_file_path() -> None:
    store = InMemoryArtifactStore()
    meta = store.store(
        b"test-image-bytes",
        content_type="image/png",
        run_id="run-1",
        tags={"filename": "sushi.png"},
    )

    media = [{"$artifact": meta.artifact_id, "content_type": "image/png", "filename": "sushi.png"}]
    resolved = _resolve_media_artifacts(media, artifact_store=store)

    assert isinstance(resolved, list)
    assert resolved
    item = resolved[0]
    assert isinstance(item, dict)
    assert item.get("$artifact") == meta.artifact_id
    assert item.get("artifact_id") == meta.artifact_id
    assert item.get("content_type") == "image/png"
    assert item.get("type") == "image"
    path = str(item.get("file_path") or "")
    assert os.path.exists(path)
    os.remove(path)


def test_resolve_media_artifact_without_content_type_still_preserves_role() -> None:
    store = InMemoryArtifactStore()
    meta = store.store(
        b"test-image-bytes",
        run_id="run-2",
        tags={"filename": "mask.bin"},
    )

    resolved = _resolve_media_artifacts(
        [{"$artifact": meta.artifact_id, "filename": "mask.bin", "role": "mask"}],
        artifact_store=store,
    )

    item = resolved[0]
    assert item.get("$artifact") == meta.artifact_id
    assert item.get("artifact_id") == meta.artifact_id
    assert item.get("role") == "mask"
    assert "type" not in item
    path = str(item.get("file_path") or "")
    assert os.path.exists(path)
    os.remove(path)
