"""Voice transport in the session-turn projection (ADR 0047).

Two groups:

1. CHARACTERIZATION (pass today): a spoken question + read-aloud answer, built in the
   exact shapes the live gateway writes (run c5ad4cce, 2026-11-01): the playback artifact
   is stored under the TURN run (not the TTS child), the TTS child is
   `wf_abstractcore_run_facade_tts_stream` with the artifact ref in its output, the
   dictation upload sits under `session_memory_<sid>` with only a `client:voice/dictation/`
   path (no `purpose` tag — the live gateway predates the stamping), and the turn's input
   context carries `messages` + `task` only. The projection DROPS transport and types nothing.

2. CONTRACT (xfail, strict): the proposed typed `voice` block on each session turn —
   `untracked/voice-stall/abstraction-server.md`. These go red (XPASS) the moment an
   implementer ships the block, which is the signal to delete the markers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractruntime import InMemoryArtifactStore, InMemoryLedgerStore, InMemoryRunStore, export_run_history_bundle
from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.history_bundle import is_voice_transport

pytestmark = pytest.mark.basic

SID = "watch-sess"
TURN = "turn-spoken"
OWNER = f"session_memory_{SID}"
TTS_CHILD = "tts-child-1"
TTS_CANCELLED = "tts-child-cancelled"
STT_CHILD = "stt-child-1"
DICTATION = "artifact_dictation"
TRANSCRIPT = "artifact_transcript"
PLAYBACK = "artifact_playback"
IMAGE = "artifact_image"
VOICE_SCHEMA = "abstractruntime.session_turn_voice.v1"


def _run(run_id: str, *, workflow_id: str, vars: Dict[str, Any], output: Any, parent: Optional[str],
         created_at: str, status: RunStatus = RunStatus.COMPLETED) -> RunState:
    return RunState(
        run_id=run_id, workflow_id=workflow_id, status=status, current_node="done", vars=vars, output=output,
        error=None, created_at=created_at, updated_at=created_at, actor_id="gateway", session_id=SID,
        parent_run_id=parent, waiting=None,
    )


def _tts_result(artifact_id: Optional[str], *, cancelled: bool = False) -> Dict[str, Any]:
    """The child output `run_facade.stream_voice._final_result_for_event` records (live shape)."""
    if cancelled:
        return {"content": None, "outputs": {}, "errors": [{"message": "TTS stream was cancelled.", "code": "cancelled"}],
                "metadata": {"streaming": True, "cancelled": True, "terminal_event": {"type": "cancelled"}}}
    return {
        "content": None,
        "outputs": {"voice": [{
            "modality": "voice", "task": "tts", "content_type": "audio/wav", "format": "wav",
            "provider": "supertonic", "model": "supertonic-3",
            "artifact_ref": {"$artifact": artifact_id, "artifact_id": artifact_id, "content_type": "audio/wav", "size_bytes": 3},
            "metadata": {"stream": {"chunks": 3, "chunk_format": "wav-segment"}, "terminal_event": {"type": "done", "ok": True}},
            "artifact_id": artifact_id, "size_bytes": 3,
        }]},
        "metadata": {"streaming": True, "terminal_event": {"type": "done", "ok": True}},
    }


def _spoken_turn(*, context_voice: Optional[Dict[str, Any]] = None, playback_in_store: bool = True) -> Tuple[Any, Any, Any]:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    artifacts = InMemoryArtifactStore()
    image_att = {"$artifact": IMAGE, "target": "client", "source_path": "client:photo.png", "content_type": "image/png"}

    # The session-memory owner run: dictation upload + its transcript live here (gateway `_store_session_artifact`).
    run_store.save(_run(OWNER, workflow_id="__session_memory__", vars={"context": {"task": "", "messages": []}},
                        output={"messages": []}, parent=None, created_at="2026-11-01T20:15:34+00:00"))
    artifacts.store(b"wav", content_type="audio/wav", run_id=OWNER, artifact_id=DICTATION,
                    tags={"kind": "attachment", "target": "client", "source": "upload", "modality": "audio",
                          "path": "client:voice/dictation/4e6a43dc.wav", "filename": "ask.wav", "session_id": SID})
    artifacts.store(b"Dim on the plate sclerosis.", content_type="text/plain; charset=utf-8", run_id=OWNER, artifact_id=TRANSCRIPT,
                    tags={"kind": "derived_text", "modality": "audio", "task": "stt", "request_id": "req-stt-1",
                          "session_id": SID, "source_audio_artifact": DICTATION})
    run_store.save(_run(STT_CHILD, workflow_id="wf_abstractcore_run_facade_transcription", vars={"_runtime": {}},
                        output={"result": {"content": "Dim on the plate sclerosis.", "outputs": {}, "metadata": {"media_only": True}}},
                        parent=OWNER, created_at="2026-11-01T20:18:12.3+00:00"))

    # The turn root: text only, plus one real attachment. `context.voice` is the PROPOSED provenance field.
    ctx: Dict[str, Any] = {"messages": [{"role": "user", "content": "Dim on the plate sclerosis."}],
                           "task": "Dim on the plate sclerosis.", "attachments": [image_att]}
    if context_voice is not None:
        ctx["voice"] = context_voice
    run_store.save(_run(TURN, workflow_id="basic-agent@0.0.5:81795ea9", vars={"prompt": "Dim on the plate sclerosis.", "context": ctx},
                        output={"response": "Multiple sclerosis is..."}, parent=None, created_at="2026-11-01T20:18:13.5+00:00"))
    artifacts.store(b"png", content_type="image/png", run_id=TURN, artifact_id=IMAGE,
                    tags={"kind": "attachment", "modality": "image", "path": "client:photo.png", "session_id": SID})

    # Read-aloud: the TTS child is a child of the TURN; its artifact is stored under the TURN run id (live fact:
    # the gateway's trace_metadata.run_id is the parent, run_facade only `setdefault`s the child id).
    run_store.save(_run(TTS_CHILD, workflow_id="wf_abstractcore_run_facade_tts_stream", vars={"_runtime": {}, "result": _tts_result(PLAYBACK)},
                        output={"result": _tts_result(PLAYBACK)}, parent=TURN, created_at="2026-11-01T21:55:53.2+00:00"))
    run_store.save(_run(TTS_CANCELLED, workflow_id="wf_abstractcore_run_facade_tts_stream", vars={"_runtime": {}, "result": _tts_result(None, cancelled=True)},
                        output={"result": _tts_result(None, cancelled=True)}, parent=TURN, created_at="2026-11-01T21:56:10.0+00:00"))
    if playback_in_store:
        artifacts.store(b"tts", content_type="audio/wav", run_id=TURN, artifact_id=PLAYBACK,
                        tags={"kind": "generated_media", "source": "llm_call", "modality": "voice", "task": "tts", "model": "supertonic-3",
                              "purpose": "voice_playback", "artifact_space": "voice", "filename": f"tts-{PLAYBACK}.wav",
                              "path": f"voice/tts-{PLAYBACK}.wav", "session_id": SID, "parent_run_id": TURN, "request_id": "req-tts-1"},
                        metadata={"capability_metadata": {"stream": {"chunk_format": "wav-segment", "chunks": 3}}})
    return run_store, ledger_store, artifacts


def _export(stores: Tuple[Any, Any, Any], *, detail: str = "full") -> Dict[str, Any]:
    run_store, ledger_store, artifacts = stores
    return export_run_history_bundle(
        run_id=TURN, run_store=run_store, ledger_store=ledger_store, artifact_store=artifacts,
        include_subruns=True, include_session=True, session_turn_limit=20, ledger_mode="tail", ledger_max_items=50, detail=detail,
    )


def _turn(bundle: Dict[str, Any]) -> Dict[str, Any]:
    turns: List[Dict[str, Any]] = (bundle.get("session") or {}).get("turns") or []
    assert [t["run_id"] for t in turns] == [TURN], turns  # the session-memory owner and the children are never turns
    return turns[0]


# --------------------------------------------------------------------------- characterization (today)


@pytest.mark.parametrize("detail", ["full", "replay"])
def test_voice_projection_today_drops_transport_and_types_nothing(detail: str) -> None:
    bundle = _export(_spoken_turn(), detail=detail)
    turn = _turn(bundle)
    assert bundle["version"] == 1 and bundle["detail"] == detail
    # The conversation projection: only the photo survives, in both lists.
    assert [a["$artifact"] for a in turn["attachments"]] == [IMAGE]
    assert [a["artifact_id"] for a in turn["artifacts"]] == [IMAGE]
    # TODAY'S HOLE: nothing says this question was spoken or this answer was read aloud.
    assert "voice" not in turn
    # The exact key set, so a voice field (or any other) cannot slip in unseen. `prompt_author`, `author` (turn
    # authors, 6706188 + 3e8e08a) and `repeat_of` (1abcc26), all 2026-11-08, are conversation fields, not transport,
    # and are pinned here on purpose. `notices` (turn.notices, 2026-11-09: what the answering turn formed, beside the
    # answer) is a conversation field too.
    assert set(turn) == {"run_id", "workflow_id", "kind", "status", "created_at", "updated_at", "prompt", "prompt_verbatim",
                         "prompt_metadata", "attachments", "answer", "answer_meta", "stats", "artifacts",
                         "prompt_author", "author", "repeat_of", "notices"}


def test_voice_projection_today_archive_keeps_playback_and_the_ledger_can_derive_it() -> None:
    bundle = _export(_spoken_turn())
    # The run bundle's own artifacts section is the archive: the playback stays, under the TURN run.
    archive = bundle["ledgers"][TURN]["artifacts"]
    assert sorted(a["artifact_id"] for a in archive) == [IMAGE, PLAYBACK]
    assert next(a for a in archive if a["artifact_id"] == PLAYBACK)["run_id"] == TURN
    # Derivability: the TTS children are in the subtree and their recorded output names the artifact.
    assert {TTS_CHILD, TTS_CANCELLED} <= set(bundle["ledgers"])
    assert bundle["ledgers"][TTS_CHILD]["artifacts"] == []  # the child owns NO artifact row; the ref lives in its output
    run_store = _spoken_turn()[0]
    child = run_store.load(TTS_CHILD)
    assert child.output["result"]["outputs"]["voice"][0]["artifact_ref"]["$artifact"] == PLAYBACK
    assert run_store.load(TTS_CANCELLED).output["result"]["outputs"] == {}


def test_voice_projection_today_dictation_is_unlinked_and_the_path_rule_is_load_bearing() -> None:
    run_store, _ledgers, artifacts = _spoken_turn()
    turn_vars = run_store.load(TURN).vars
    # The turn's input never names the recording or its transcript: linking by session + time would be a guess.
    assert set(turn_vars["context"]) == {"messages", "task", "attachments"}
    assert DICTATION not in str(turn_vars)
    # Live uploads carry NO purpose tag (the running gateway predates `dictation_tags`): the reserved path decides.
    dictation_tags = artifacts.get_metadata(DICTATION).tags
    assert "purpose" not in dictation_tags and is_voice_transport(dictation_tags)
    assert is_voice_transport(artifacts.get_metadata(PLAYBACK).tags)
    # Divergence pinned: a LEGACY stream (no purpose tag, provenance only in metadata) is transport for the
    # gateway (`voice_artifacts.is_voice_playback`) but NOT for the runtime rule, which reads tags alone.
    assert not is_voice_transport({"kind": "generated_media", "modality": "voice", "task": "tts"})


# --------------------------------------------------------------------------- contract (proposed, red today)

XFAIL = dict(strict=True, reason="proposed contract: typed session-turn `voice` block "
                                 "(untracked/voice-stall/abstraction-server.md, section A)")


@pytest.mark.xfail(**XFAIL)
def test_voice_projection_contract_playback_is_typed_from_the_tts_children() -> None:
    turn = _turn(_export(_spoken_turn()))
    voice = turn["voice"]
    assert voice["schema"] == VOICE_SCHEMA
    assert voice["dictation"] is None  # no provenance in the input → null, never guessed from the session
    # One entry per terminal TTS child of the turn, oldest first; cancelled readings are listed, not hidden.
    assert [p["child_run_id"] for p in voice["playback"]] == [TTS_CHILD, TTS_CANCELLED]
    done, cancelled = voice["playback"]
    assert done == {
        "child_run_id": TTS_CHILD, "artifact_id": PLAYBACK, "run_id": TURN, "created_at": "2026-11-01T21:55:53.2+00:00",
        "status": "completed", "content_type": "audio/wav", "provider": "supertonic", "model": "supertonic-3",
        "voice": None, "duration_s": None, "available": True,
    }
    assert cancelled["artifact_id"] is None and cancelled["status"] == "cancelled" and cancelled["available"] is False
    # `attachments` / `artifacts` keep excluding transport: the block is the ONLY place it appears.
    assert [a["artifact_id"] for a in turn["artifacts"]] == [IMAGE]


@pytest.mark.xfail(**XFAIL)
def test_voice_projection_contract_dictation_comes_from_the_turn_input_not_from_time() -> None:
    provenance = {"dictation": {"$artifact": DICTATION, "run_id": OWNER, "request_id": "req-stt-1",
                                "transcript": {"$artifact": TRANSCRIPT}}}
    turn = _turn(_export(_spoken_turn(context_voice=provenance)))
    assert turn["voice"]["dictation"] == {
        "artifact_id": DICTATION, "run_id": OWNER, "content_type": "audio/wav", "duration_s": None,
        "request_id": "req-stt-1", "transcript_artifact_id": TRANSCRIPT, "available": True,
    }
    # The recording is still not an attachment of the turn.
    assert [a["$artifact"] for a in turn["attachments"]] == [IMAGE]


@pytest.mark.xfail(**XFAIL)
def test_voice_projection_contract_missing_playback_artifact_is_reported_not_dropped() -> None:
    turn = _turn(_export(_spoken_turn(playback_in_store=False)))
    done = turn["voice"]["playback"][0]
    assert done["artifact_id"] == PLAYBACK and done["status"] == "completed" and done["available"] is False
