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
    assert isinstance(item, str)
    assert os.path.exists(item)
    os.remove(item)
