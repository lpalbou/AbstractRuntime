"""The voice library facade: languages, cloned voices, design, cloning and the capability catalog.

Local clients call AbstractCore's voice capability in process; the remote client calls a remote
AbstractCore server's `/v1` voice routes. Both answer the same shapes and refuse with
`VoiceLibraryError(status_code, detail)`. Also: the voice catalog's `language` filter and
`one_shot` on remote buffered TTS.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from abstractruntime.integrations.abstractcore import (
    AbstractCoreVoiceFacade,
    LocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
    VoiceLibraryError,
    get_abstractcore_voice_facade,
)
from abstractruntime.integrations.abstractcore import discovery_queries as dq
from abstractruntime.integrations.abstractcore import voice_library as vl


# ---------------------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------------------


class _FakeVoice:
    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.voices = {"v1": {"voice_id": "v1", "name": "Ada", "engine": "qwen3-tts", "reference_text": "hi"}}

    def list_languages(self, *, kind=None, provider=None):
        self.calls.append(("list_languages", kind, provider))
        return {"tts": {"qwen3-tts": {"languages": ["en", "fr"]}}}

    def list_cloned_voices(self, *, provider=None, model=None, language=None):
        self.calls.append(("list_cloned_voices", provider, model, language))
        return list(self.voices.values())

    def get_cloned_voice(self, voice_id):
        if voice_id not in self.voices:
            raise KeyError(f"Unknown cloned voice: {voice_id}")
        return dict(self.voices[voice_id])

    def rename_cloned_voice(self, voice_id, new_name):
        self.calls.append(("rename", voice_id, new_name))
        self.voices[voice_id]["name"] = new_name
        return dict(self.voices[voice_id])

    def set_cloned_voice_reference_text(self, voice_id, reference_text):
        self.calls.append(("reference_text", voice_id, reference_text))
        self.voices[voice_id]["reference_text"] = reference_text
        return dict(self.voices[voice_id])

    def delete_cloned_voice(self, voice_id):
        if voice_id.startswith("builtin"):
            raise ValueError("Built-in voices are read-only.")
        self.calls.append(("delete", voice_id))
        return {"voice_id": voice_id, "deleted": True}

    def design_voice(self, description, *, name=None, language=None, text=None):
        self.calls.append(("design", description, name, language, text))
        return {"voice_id": "d1", "name": name or "Designed"}

    def clone(self, audio, **kwargs):
        self.calls.append(("clone", bytes(audio), kwargs))
        self.voices["c1"] = {"voice_id": "c1", "name": kwargs.get("name"), "engine": kwargs.get("provider")}
        return "c1"  # the plugin may answer a bare id

    def compatibility_catalog(self):
        return {"providers": {"tts": {"qwen3-tts": {}}}}

    def voice_catalog(self, **kwargs):
        self.calls.append(("voice_catalog", kwargs))
        return {"profiles": [], "voices": [{"id": "v1", "provider": "qwen3-tts"}]}


@pytest.fixture
def fake_voice(monkeypatch: pytest.MonkeyPatch) -> _FakeVoice:
    voice = _FakeVoice()
    seen: List[Dict[str, Any]] = []

    class _FakeRegistry:
        def __init__(self, owner, preferred_backends=None):
            seen.append(dict(owner.config))
            self.voice = voice

    import abstractcore.capabilities as caps

    monkeypatch.setattr(caps, "CapabilityRegistry", _FakeRegistry)
    voice.owner_configs = seen  # type: ignore[attr-defined]
    return voice


def _local_facade() -> AbstractCoreVoiceFacade:
    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)  # no model load
    return AbstractCoreVoiceFacade(client=client)


def test_local_reads(fake_voice: _FakeVoice) -> None:
    facade = _local_facade()
    assert facade.list_languages(kind="tts")["languages"]["tts"]["qwen3-tts"]["languages"] == ["en", "fr"]
    listed = facade.list_cloned_voices(language="fr")
    assert listed["count"] == 1 and listed["voices"][0]["voice_id"] == "v1"
    assert ("list_cloned_voices", None, None, "fr") in fake_voice.calls
    assert facade.get_cloned_voice("v1")["name"] == "Ada"
    assert facade.compatibility_catalog()["providers"]["tts"] == {"qwen3-tts": {}}


def test_local_update_checks_before_changing(fake_voice: _FakeVoice) -> None:
    facade = _local_facade()
    out = facade.update_cloned_voice("v1", name="Grace", reference_text="hello there")
    assert out["name"] == "Grace" and out["reference_text"] == "hello there"
    with pytest.raises(VoiceLibraryError) as unknown:
        facade.update_cloned_voice("nope", name="X")
    assert unknown.value.status_code == 404
    with pytest.raises(VoiceLibraryError) as empty:
        facade.update_cloned_voice("v1", name="Z", reference_text="  ")
    assert empty.value.status_code == 422
    assert fake_voice.voices["v1"]["name"] == "Grace"  # nothing applied on a refused update
    with pytest.raises(VoiceLibraryError) as nothing:
        facade.update_cloned_voice("v1")
    assert nothing.value.status_code == 422


def test_local_delete_design_and_clone(fake_voice: _FakeVoice) -> None:
    facade = _local_facade()
    assert facade.delete_cloned_voice("v1")["deleted"] is True
    with pytest.raises(VoiceLibraryError) as builtin:
        facade.delete_cloned_voice("builtin-ada")
    assert builtin.value.status_code == 422
    assert facade.design_voice("a warm low voice", name="Warm", language="en")["voice_id"] == "d1"
    with pytest.raises(VoiceLibraryError) as blank:
        facade.design_voice("   ")
    assert blank.value.status_code == 422
    record = facade.clone_voice(b"RIFF....", name="Mine", provider="qwen3-tts", reference_text="hello")
    assert record["voice_id"] == "c1" and record["ok"] is True and record["name"] == "Mine"
    _name, audio, kwargs = [c for c in fake_voice.calls if c[0] == "clone"][0]
    assert audio == b"RIFF...." and kwargs["provider"] == "qwen3-tts" and kwargs["cloning_engine"] == "qwen3-tts"
    with pytest.raises(VoiceLibraryError) as no_audio:
        facade.clone_voice(b"")
    assert no_audio.value.status_code == 422


def test_local_capability_errors_map_to_501(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.capabilities.errors import CapabilityUnavailableError

    class _NoVoice:
        def design_voice(self, *a, **k):
            raise CapabilityUnavailableError(capability="voice", reason="no backend")

        def list_languages(self, **k):
            raise RuntimeError("Qwen3-TTS is not installed: install it with pip install abstractvoice[qwen3-tts]")

    monkeypatch.setattr(vl, "_local_voice", lambda key: _NoVoice())
    facade = _local_facade()
    for call in (lambda: facade.design_voice("x"), lambda: facade.list_languages()):
        with pytest.raises(VoiceLibraryError) as err:
            call()
        assert err.value.status_code == 501


def test_local_threads_the_host_openai_key(fake_voice: _FakeVoice) -> None:
    _local_facade().list_languages(voice_openai_api_key="sk-host")
    assert fake_voice.owner_configs[-1]["voice_openai_api_key"] == "sk-host"  # type: ignore[attr-defined]


def test_unknown_operation_is_refused() -> None:
    with pytest.raises(VoiceLibraryError) as err:
        vl.local_voice_library("rename_everything", {})
    assert err.value.status_code == 422


# ---------------------------------------------------------------------------------------
# Language filter on the voice catalog (local)
# ---------------------------------------------------------------------------------------


def test_local_catalog_passes_language_only_when_asked(fake_voice: _FakeVoice) -> None:
    dq.local_get_voice_catalog(provider="qwen3-tts", language="fr")
    dq.local_get_voice_catalog(provider="qwen3-tts")
    catalog_calls = [c[1] for c in fake_voice.calls if c[0] == "voice_catalog"]
    assert catalog_calls[0]["language"] == "fr"
    assert "language" not in catalog_calls[1]


# ---------------------------------------------------------------------------------------
# Remote
# ---------------------------------------------------------------------------------------


class _Response:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _HTTPError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"HTTP {status_code}")
        self.response = _Response(status_code, {"detail": detail})


class _Sender:
    def __init__(self, answers: Dict[tuple, Any]):
        self.answers = answers
        self.calls: List[tuple] = []

    def _answer(self, method: str, url: str) -> Any:
        path = url.split("/v1", 1)[1].split("?", 1)[0]
        answer = self.answers.get((method, path))
        if isinstance(answer, Exception):
            raise answer
        return answer if answer is not None else {}

    def get(self, url, *, headers, timeout):
        self.calls.append(("GET", url, None, headers))
        return self._answer("GET", url)

    def post(self, url, *, headers, json, timeout, **_kw):
        self.calls.append(("POST", url, json, headers))
        return self._answer("POST", url)

    def request(self, method, url, *, headers, json=None, timeout):
        self.calls.append((method, url, json, headers))
        return self._answer(method, url)

    def post_multipart(self, url, *, headers, data, files, timeout):
        self.calls.append(("MULTIPART", url, {"data": data, "files": files}, headers))
        return self._answer("POST", url)


def _remote(answers: Dict[tuple, Any]) -> tuple[AbstractCoreVoiceFacade, _Sender]:
    sender = _Sender(answers)
    client = RemoteAbstractCoreLLMClient(server_base_url="http://core.example:8000", model="m", request_sender=sender)
    return AbstractCoreVoiceFacade(client=client), sender


def test_remote_reads_call_the_v1_routes() -> None:
    facade, sender = _remote(
        {
            ("GET", "/audio/languages"): {"languages": {"tts": {}}, "source": "abstractvoice"},
            ("GET", "/voice/clones"): {"voices": [{"voice_id": "v1"}], "count": 1},
            ("GET", "/voice/clones/v%2F1"): {"voice_id": "v/1"},
            ("GET", "/audio/voices"): {"compatibility_catalog": {"providers": {"tts": {}}}},
        }
    )
    assert facade.list_languages(kind="tts", provider="qwen3-tts") == {"languages": {"tts": {}}}
    assert sender.calls[-1][1] == "http://core.example:8000/v1/audio/languages?kind=tts&provider=qwen3-tts"
    assert facade.list_cloned_voices(language="fr")["count"] == 1
    assert sender.calls[-1][1].endswith("/v1/voice/clones?language=fr")
    assert facade.get_cloned_voice("v/1")["voice_id"] == "v/1"  # the id is escaped in the path
    assert facade.compatibility_catalog() == {"providers": {"tts": {}}}


def test_remote_changes_use_patch_delete_post_and_multipart() -> None:
    facade, sender = _remote(
        {
            ("PATCH", "/voice/clones/v1"): {"voice_id": "v1", "name": "Grace"},
            ("DELETE", "/voice/clones/v1"): {"voice_id": "v1", "deleted": True},
            ("POST", "/voice/design"): {"voice_id": "d1"},
            ("POST", "/voice/clone"): {"voice_id": "c1", "ok": True},
        }
    )
    assert facade.update_cloned_voice("v1", name="Grace")["name"] == "Grace"
    assert sender.calls[-1][0] == "PATCH" and sender.calls[-1][2] == {"name": "Grace"}
    assert facade.delete_cloned_voice("v1")["deleted"] is True
    assert sender.calls[-1][0] == "DELETE"
    facade.design_voice("a warm voice", language="en")
    assert sender.calls[-1][2] == {"description": "a warm voice", "language": "en"}
    assert facade.clone_voice(b"RIFF", filename="me.wav", name="Mine", provider="qwen3-tts")["voice_id"] == "c1"
    method, url, body, _headers = sender.calls[-1]
    assert method == "MULTIPART" and url.endswith("/v1/voice/clone")
    assert body["data"] == {"name": "Mine", "provider": "qwen3-tts"}
    assert body["files"]["file"] == ("me.wav", b"RIFF", "audio/wav")


def test_remote_never_sends_the_host_openai_key_and_forwards_the_provider_key() -> None:
    facade, sender = _remote({("GET", "/audio/languages"): {"languages": {}}})
    facade.list_languages(voice_openai_api_key="sk-host", provider_api_key="pk-123")
    headers = sender.calls[-1][3]
    assert "sk-host" not in repr(headers) and "pk-123" in repr(headers)


@pytest.mark.parametrize("status,expected", [(401, 401), (404, 404), (422, 422), (501, 501), (500, 502)])
def test_remote_errors_keep_their_status(status: int, expected: int) -> None:
    facade, _sender = _remote({("DELETE", "/voice/clones/v1"): _HTTPError(status, "refused")})
    with pytest.raises(VoiceLibraryError) as err:
        facade.delete_cloned_voice("v1")
    assert err.value.status_code == expected and "refused" in err.value.detail


def test_remote_error_reads_the_core_server_error_shape() -> None:
    class _CoreError(Exception):
        def __init__(self) -> None:
            super().__init__("Client error '404 Not Found'")
            self.response = _Response(404, {"error": {"message": "Unknown voice_id: nope", "type": "http_error"}})

    facade, _sender = _remote({("GET", "/voice/clones/nope"): _CoreError()})
    with pytest.raises(VoiceLibraryError) as err:
        facade.get_cloned_voice("nope")
    assert err.value.status_code == 404 and "Unknown voice_id: nope" in err.value.detail


def test_remote_unreachable_is_502() -> None:
    facade, _sender = _remote({("GET", "/voice/clones"): ConnectionError("refused")})
    with pytest.raises(VoiceLibraryError) as err:
        facade.list_cloned_voices()
    assert err.value.status_code == 502


def test_remote_compatibility_catalog_missing_is_501_not_empty() -> None:
    facade, _sender = _remote({("GET", "/audio/voices"): {"voices": []}})
    with pytest.raises(VoiceLibraryError) as err:
        facade.compatibility_catalog()
    assert err.value.status_code == 501


def test_remote_change_without_a_request_capable_sender_fails_loudly() -> None:
    class _OldSender(_Sender):
        request = None  # type: ignore[assignment]

    sender = _OldSender({})
    client = RemoteAbstractCoreLLMClient(server_base_url="http://core.example:8000", model="m", request_sender=sender)
    with pytest.raises(VoiceLibraryError) as err:
        AbstractCoreVoiceFacade(client=client).delete_cloned_voice("v1")
    assert err.value.status_code == 501


def test_remote_catalog_forwards_language() -> None:
    sender = _Sender({("GET", "/audio/voices"): {"voices": []}})
    client = RemoteAbstractCoreLLMClient(server_base_url="http://core.example:8000", model="m", request_sender=sender)
    client.get_voice_catalog(provider="qwen3-tts", language="fr")
    assert "language=fr" in sender.calls[-1][1]
    client.get_voice_catalog(provider="qwen3-tts")
    assert "language" not in sender.calls[-1][1]


def test_remote_buffered_tts_forwards_one_shot_and_the_stream_does_not() -> None:
    import inspect

    from abstractruntime.integrations.abstractcore import llm_client

    tts = inspect.getsource(llm_client.RemoteAbstractCoreLLMClient._remote_tts)
    stream = inspect.getsource(llm_client.RemoteAbstractCoreLLMClient._remote_tts_stream)
    assert '"one_shot"' in tts
    assert '"one_shot"' not in stream  # AbstractCore refuses one_shot on the stream route


def test_facade_from_runtime_requires_a_wired_client() -> None:
    class _Runtime:
        pass

    with pytest.raises(RuntimeError):
        get_abstractcore_voice_facade(_Runtime())
    runtime = _Runtime()
    runtime._abstractcore_llm_client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    assert isinstance(get_abstractcore_voice_facade(runtime), AbstractCoreVoiceFacade)
