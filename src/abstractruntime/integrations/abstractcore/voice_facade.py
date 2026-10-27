"""Public host-facing voice library facade for Runtime integrations.

Hosts manage the voices AbstractCore can speak with through this facade instead of reaching
into AbstractCore or AbstractVoice directly:

- languages each voice engine accepts (`list_languages`);
- cloned voices, including designed and built-in ones: list (optionally by language), read,
  rename / replace the reference transcript, delete;
- create a voice from a recording (`clone_voice`) or from a written description
  (`design_voice`);
- the capability catalog (`compatibility_catalog`): which engine and model supports which
  feature (speed, one_shot, design...) and how (native, emulated, conditional).

The runtime's AbstractCore client runs each call in process (local clients) or against a
remote AbstractCore server's `/v1` routes (remote client). Every refusal is a
`VoiceLibraryError` with an HTTP-like `status_code`. Listing voices for synthesis stays with
`AbstractCoreDiscoveryFacade.get_voice_catalog(..., language=)`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .voice_library import VoiceLibraryError

_RUNTIME_ABSTRACTCORE_CLIENT_ATTR = "_abstractcore_llm_client"


def _client_from_runtime(runtime: Any) -> Any:
    client = getattr(runtime, _RUNTIME_ABSTRACTCORE_CLIENT_ATTR, None)
    if client is None:
        raise RuntimeError(
            "Runtime is not wired to AbstractCore. Use an AbstractCore runtime factory, then obtain the facade via "
            "`abstractruntime.integrations.abstractcore.get_abstractcore_voice_facade(runtime)`."
        )
    if not callable(getattr(client, "voice_library", None)):
        raise TypeError("Runtime AbstractCore client does not implement voice_library(op, **kwargs).")
    return client


class AbstractCoreVoiceFacade:
    """Voice library operations bound to an AbstractCore-wired runtime (or a client)."""

    def __init__(self, runtime: Any = None, *, client: Any = None):
        self._client = client if client is not None else _client_from_runtime(runtime)

    @classmethod
    def from_runtime(cls, runtime: Any) -> "AbstractCoreVoiceFacade":
        return cls(runtime)

    def _call(self, op: str, **kwargs: Any) -> Any:
        return self._client.voice_library(op, **kwargs)

    def list_languages(self, *, kind: Optional[str] = None, provider: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """`{"languages": {kind: {provider: {provider, languages, open, declared}}}}`; `kind` is
        `tts`, `stt` or `cloning`. No model is loaded."""
        return self._call("list_languages", kind=kind, provider=provider, **kwargs)

    def list_cloned_voices(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """`{"voices": [...], "count": n}`: cloned voices (designed and built-in ones included,
        `tags.designed` / `tags.builtin`); `language` keeps the voices that can speak it,
        native speakers first."""
        return self._call("list_cloned_voices", provider=provider, model=model, language=language, **kwargs)

    def get_cloned_voice(self, voice_id: str, **kwargs: Any) -> Dict[str, Any]:
        """One cloned voice's stored record; 404 for an unknown id."""
        return self._call("get_cloned_voice", voice_id=voice_id, **kwargs)

    def update_cloned_voice(
        self,
        voice_id: str,
        *,
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Rename a cloned voice and/or replace its reference transcript; both fields are checked
        before either is applied. Returns the updated record. Built-in voices are read-only (422)."""
        return self._call("update_cloned_voice", voice_id=voice_id, name=name, reference_text=reference_text, **kwargs)

    def delete_cloned_voice(self, voice_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Delete a cloned voice and its reference audio. Built-in voices are read-only (422)."""
        return self._call("delete_cloned_voice", voice_id=voice_id, **kwargs)

    def design_voice(
        self,
        description: str,
        *,
        name: Optional[str] = None,
        language: Optional[str] = None,
        text: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a reusable voice from a written description (Qwen3-TTS VoiceDesign, then cloned
        with Qwen3-TTS Base); stored as a cloned voice. Needs `abstractvoice[qwen3-tts]` and
        `abstractvoice[stt]` where the voice capability runs (501 otherwise)."""
        return self._call("design_voice", description=description, name=name, language=language, text=text, **kwargs)

    def clone_voice(
        self,
        audio: bytes,
        *,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a cloned voice from a reference recording (`provider` = the cloning engine, e.g.
        `qwen3-tts`, `omnivoice`; `language` = the recording's language, AbstractVoice 0.19.0);
        returns the stored record, flat, with its `voice_id` (same shape locally and remotely)."""
        return self._call(
            "clone_voice",
            audio=audio,
            filename=filename,
            content_type=content_type,
            name=name,
            reference_text=reference_text,
            provider=provider,
            model=model,
            language=language,
            **kwargs,
        )

    def deploy_voice_to_library(self, voice_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Copy one of this runtime's personal voices into its first library, where every runtime
        sharing that library lists and uses it; returns the library voice's record. Needs a
        personal store (`voice_store_dir`) and AbstractVoice 0.19.0; local runtimes only (501
        remote). Who may deploy is the host's decision."""
        return self._call("deploy_voice_to_library", voice_id=voice_id, **kwargs)

    def compatibility_catalog(self, **kwargs: Any) -> Dict[str, Any]:
        """The capability catalog: per kind (tts, stt, cloning), provider, model and surface, the
        support level of each feature (`native`, `emulated`, `conditional`, `unsupported`)."""
        return self._call("compatibility_catalog", **kwargs)


def get_abstractcore_voice_facade(runtime: Any) -> AbstractCoreVoiceFacade:
    """Return the public voice library facade bound to a runtime."""

    return AbstractCoreVoiceFacade.from_runtime(runtime)


__all__ = ["AbstractCoreVoiceFacade", "VoiceLibraryError", "get_abstractcore_voice_facade"]
