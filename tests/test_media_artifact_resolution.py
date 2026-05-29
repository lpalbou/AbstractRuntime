from __future__ import annotations

from pathlib import Path


def test_resolved_audio_artifact_preserves_content_type_for_transcription(tmp_path: Path) -> None:
    from abstractcore.providers.base import BaseProvider
    from abstractruntime.integrations.abstractcore.llm_client import _is_audio_media_item, _resolve_media_artifacts

    content_path = tmp_path / "artifact-content-without-extension"
    content_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    class Store:
        def _content_path(self, artifact_id: str) -> Path:
            assert artifact_id == "audio-1"
            return content_path

    resolved = _resolve_media_artifacts(
        [{"$artifact": "audio-1", "content_type": "audio/wav"}],
        artifact_store=Store(),
    )

    assert isinstance(resolved, list)
    assert resolved == [
        {
            "$artifact": "audio-1",
            "artifact_id": "audio-1",
            "content_type": "audio/wav",
            "file_path": str(content_path),
            "type": "audio",
        }
    ]
    assert _is_audio_media_item(resolved[0]) is True
    assert BaseProvider._media_type(resolved[0]) == "audio"


def test_effect_handler_materialized_audio_artifact_preserves_content_type(tmp_path: Path) -> None:
    from abstractcore.providers.base import BaseProvider
    from abstractruntime.integrations.abstractcore.effect_handlers import _resolve_llm_call_media

    class Artifact:
        content = b"RIFF\x00\x00\x00\x00WAVE"
        content_type = None
        metadata = None

    class Store:
        def load(self, artifact_id: str) -> Artifact:
            assert artifact_id == "audio-1"
            return Artifact()

    resolved, error = _resolve_llm_call_media(
        [{"$artifact": "audio-1", "content_type": "audio/wav"}],
        artifact_store=Store(),
        temp_dir=tmp_path,
    )

    assert error is None
    assert isinstance(resolved, list)
    assert len(resolved) == 1
    item = resolved[0]
    assert item["$artifact"] == "audio-1"
    assert item["artifact_id"] == "audio-1"
    assert item["content_type"] == "audio/wav"
    assert item["type"] == "audio"
    assert Path(item["file_path"]).suffix == ".wav"
    assert BaseProvider._media_type(item) == "audio"


def test_effect_handler_materialized_image_artifacts_preserve_roles(tmp_path: Path) -> None:
    from abstractruntime.integrations.abstractcore.effect_handlers import _resolve_llm_call_media

    class Artifact:
        content = PNG_1X1 = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
            b"\x1f\x15\xc4\x89"
        )
        content_type = "image/png"
        metadata = None

    class Store:
        def load(self, artifact_id: str) -> Artifact:
            assert artifact_id in {"source-img", "mask-img"}
            return Artifact()

    resolved, error = _resolve_llm_call_media(
        [
            {"$artifact": "source-img", "content_type": "image/png", "role": "source"},
            {"$artifact": "mask-img", "content_type": "image/png", "role": "mask"},
        ],
        artifact_store=Store(),
        temp_dir=tmp_path,
    )

    assert error is None
    assert isinstance(resolved, list)
    assert [item.get("role") for item in resolved] == ["source", "mask"]
    assert all(item.get("type") == "image" for item in resolved)
    assert all(Path(str(item.get("file_path") or "")).is_file() for item in resolved)
