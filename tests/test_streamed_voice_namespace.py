import io
import wave

import pytest

from abstractruntime.integrations.abstractcore.llm_client import _finalize_tts_stream_artifact
from abstractruntime.storage.artifacts import FileArtifactStore, InMemoryArtifactStore


def wav(frames):
    data = io.BytesIO()
    with wave.open(data, "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
        out.writeframes(frames)
    return data.getvalue()


@pytest.mark.parametrize("file_backed", [False, True])
def test_stream_recording_has_voice_namespace_and_preserves_playback_and_provenance(tmp_path, file_backed):
    store = FileArtifactStore(tmp_path) if file_backed else InMemoryArtifactStore()
    tags = {"kind": "generated_media", "modality": "voice", "task": "tts", "session_id": "chat", "request_id": "request"}
    audio = _finalize_tts_stream_artifact(
        segments=[wav(b"\x01\x00" * 20), wav(b"\x02\x00" * 30)],
        artifact_store=store, run_id="run", tags=tags, text="Hello", spec={"modality": "voice", "task": "tts"},
        provider="test", model="test-voice", metadata={"stream": {"chunk_format": "wav-segment", "chunks": 2}})
    artifact_id = audio["artifact_ref"]["$artifact"]
    loaded = store.load(artifact_id)
    assert loaded.metadata.tags["path"].startswith("voice/")
    assert loaded.metadata.tags["purpose"] == "voice_playback"
    assert loaded.metadata.tags["artifact_space"] == "voice"
    assert loaded.metadata.tags["session_id"] == "chat" and loaded.metadata.run_id == "run"
    assert loaded.metadata.descriptor.producer["provider"] == "test"
    with wave.open(io.BytesIO(loaded.content), "rb") as output:
        assert output.getnframes() == 50
        assert output.readframes(50) == b"\x01\x00" * 20 + b"\x02\x00" * 30
    assert "purpose" not in tags  # caller-owned provenance is not mutated
