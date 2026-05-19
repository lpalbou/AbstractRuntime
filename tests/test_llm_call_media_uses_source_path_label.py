from __future__ import annotations

from pathlib import Path


def test_llm_call_media_materializes_artifact_using_source_path_label(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.effect_handlers import _resolve_llm_call_media
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    store = InMemoryArtifactStore()
    meta = store.store(
        b"hello\n",
        content_type="text/plain",
        run_id=None,
        tags={"kind": "attachment", "source_path": "mnemosyne/memory/Core/Values.md"},
    )

    resolved, err = _resolve_llm_call_media(
        [{"$artifact": str(meta.artifact_id), "source_path": "mnemosyne/memory/Core/Values.md"}],
        artifact_store=store,
        temp_dir=tmp_path,
    )

    assert err is None
    assert isinstance(resolved, list) and len(resolved) == 1

    item = resolved[0]
    assert isinstance(item, dict)
    out_path = Path(str(item.get("file_path") or ""))
    assert out_path.is_file()
    assert out_path.read_bytes() == b"hello\n"
    short = str(meta.artifact_id)[:8]
    assert out_path.name == f"Values__{short}.md"
    assert item.get("$artifact") == str(meta.artifact_id)
    assert item.get("artifact_id") == str(meta.artifact_id)
    assert item.get("content_type") == "text/plain"
