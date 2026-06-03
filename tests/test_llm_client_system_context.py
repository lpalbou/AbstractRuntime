from __future__ import annotations

import json
import re

import pytest

from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient, RemoteAbstractCoreLLMClient


pytestmark = pytest.mark.basic


_RUNTIME_METADATA_RE = re.compile(
    r"^\s*<runtime_metadata>\s*(?P<payload>.*?)\s*</runtime_metadata>\s*",
    re.IGNORECASE | re.DOTALL,
)


def _runtime_metadata_from_text(value: str) -> dict:
    match = _RUNTIME_METADATA_RE.match(value)
    assert match
    payload = json.loads(match.group("payload"))
    assert isinstance(payload, dict)
    return payload


def test_local_llm_client_surfaces_runtime_metadata_to_user_turn(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.handler import UniversalToolHandler

    monkeypatch.setenv("ABSTRACT_COUNTRY", "FR")

    class _DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerateResponse(content="ok", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._artifact_store = None
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    result = client.generate(prompt="hello")

    assert client._llm.calls, "expected a provider call"
    prompt_sent = str(client._llm.calls[0].get("prompt") or "")
    runtime_meta = _runtime_metadata_from_text(prompt_sent)
    assert runtime_meta["country"] == "FR"
    assert "local_datetime" in runtime_meta
    assert prompt_sent.rstrip().endswith("hello")
    assert isinstance(result.get("metadata"), dict)
    assert result["metadata"]["runtime_grounding"]["country"] == "FR"
    assert result["metadata"]["runtime_grounding"]["prompt_injected"] is True
    payload = result["metadata"]["_provider_request"]["payload"]
    assert payload["messages"][0]["role"] == "user"
    request_meta = _runtime_metadata_from_text(str(payload["messages"][0]["content"] or ""))
    assert request_meta["country"] == "FR"


def test_local_llm_client_uses_browser_context_for_prompt_grounding(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.handler import UniversalToolHandler

    monkeypatch.setenv("ABSTRACT_COUNTRY", "FR")
    monkeypatch.setenv("TZ", "Europe/Paris")

    class _DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerateResponse(content="ok", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._artifact_store = None
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    result = client.generate(
        prompt="what time is it?",
        params={
            "trace_metadata": {
                "user_id": "alice",
                "client_context": {
                    "source": "browser",
                    "local_datetime": "2026-06-01T11:22:33-04:00",
                    "timezone": "America/New_York",
                    "locale": "en-US",
                    "country": "US",
                    "timezone_offset_minutes": -240,
                },
            }
        },
    )

    prompt_sent = str(client._llm.calls[0].get("prompt") or "")
    runtime_meta = _runtime_metadata_from_text(prompt_sent)
    assert runtime_meta["local_datetime"] == "2026-06-01T11:22:33-04:00"
    assert runtime_meta["timezone"] == "America/New_York"
    assert runtime_meta["country"] == "US"
    assert runtime_meta["user"] == "alice"
    assert isinstance(result.get("metadata"), dict)
    grounding = result["metadata"]["runtime_grounding"]
    assert grounding["source"] == "browser_untrusted"
    assert grounding["server_context"]["country"] == "FR"
    assert grounding["server_context"]["source"] == "abstractruntime"


def test_browser_timezone_wins_over_locale_country_for_grounding(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.handler import UniversalToolHandler

    monkeypatch.setenv("ABSTRACT_COUNTRY", "US")

    class _DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerateResponse(content="ok", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._artifact_store = None
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    client.generate(
        prompt="where am I grounded?",
        params={
            "trace_metadata": {
                "client_context": {
                    "local_datetime": "2026-06-01T23:07:00+02:00",
                    "timezone": "Europe/Paris",
                    "locale": "en-US",
                    "locale_country": "US",
                },
            }
        },
    )

    prompt_sent = str(client._llm.calls[0].get("prompt") or "")
    runtime_meta = _runtime_metadata_from_text(prompt_sent)
    assert runtime_meta["timezone"] == "Europe/Paris"
    assert runtime_meta["country"] == "FR"


def test_local_llm_client_strips_legacy_grounding_from_system_prompt(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.handler import UniversalToolHandler

    monkeypatch.setenv("ABSTRACT_COUNTRY", "FR")

    class _DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerateResponse(content="ok", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._artifact_store = None
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    sys_in = "Grounding: 1999/12/31|23:59|FR\n\nBase system prompt."
    client.generate(prompt="hello", system_prompt=sys_in)

    sys = str(client._llm.calls[0].get("system_prompt") or "")
    assert not sys.startswith("Grounding:")
    assert sys.strip() == "Base system prompt."
    prompt_sent = str(client._llm.calls[0].get("prompt") or "")
    runtime_meta = _runtime_metadata_from_text(prompt_sent)
    assert runtime_meta["country"] == "FR"


def test_local_llm_client_drops_recent_tool_activity_system_messages(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.handler import UniversalToolHandler

    class _DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerateResponse(content="ok", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._artifact_store = None
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    messages = [
        {"role": "system", "content": "Recent tool activity (auto):\n- turn 1: tools: —"},
        {"role": "user", "content": "hello"},
    ]

    client.generate(prompt="", messages=messages)

    assert client._llm.calls, "expected a provider call"
    sent = client._llm.calls[0].get("messages")
    assert isinstance(sent, list)
    assert all(
        not (m.get("role") == "system" and str(m.get("content") or "").lstrip().startswith("Recent tool activity"))
        for m in sent
        if isinstance(m, dict)
    )


def test_remote_llm_client_surfaces_runtime_metadata_to_user_turn(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACT_COUNTRY", "FR")

    class StubSender:
        def __init__(self):
            self.calls = []

        def post(self, url, *, headers, json, timeout):
            self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return {
                "model": json["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    sender = StubSender()
    client = RemoteAbstractCoreLLMClient(
        server_base_url="http://localhost:8080",
        model="openai-compatible/default",
        request_sender=sender,
        headers={"X-Test": "1"},
    )

    client.generate(prompt="hello", params={"max_tokens": 5})

    body = sender.calls[0]["json"]
    assert body["messages"][0]["role"] == "user"
    runtime_meta = _runtime_metadata_from_text(str(body["messages"][0]["content"] or ""))
    assert runtime_meta["country"] == "FR"


def test_runtime_metadata_envelope_is_per_call() -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    g1 = llm_client._mark_grounding_prompt_injected(
        {
            "local_datetime": "2000-01-01T00:00:01+01:00",
            "country": "FR",
            "display": "[2000-01-01 00:00:01 FR]",
        },
        True,
    )
    g2 = llm_client._mark_grounding_prompt_injected(
        {
            "local_datetime": "2000-01-01T00:00:02+01:00",
            "country": "FR",
            "display": "[2000-01-01 00:00:02 FR]",
        },
        True,
    )

    a, _ = llm_client._normalize_turn_grounding(prompt="hello", messages=None, grounding=g1)
    b, _ = llm_client._normalize_turn_grounding(prompt="hello", messages=None, grounding=g2)

    assert _runtime_metadata_from_text(a)["local_datetime"].endswith("00:00:01+01:00")
    assert _runtime_metadata_from_text(b)["local_datetime"].endswith("00:00:02+01:00")
    assert a != b


def test_media_only_turn_grounding_does_not_inject_prompt_text() -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    prompt, messages = llm_client._normalize_turn_grounding(
        prompt="hello",
        messages=None,
        grounding=None,
    )

    assert prompt == "hello"
    assert messages is None


def test_media_turns_do_not_inject_runtime_metadata_into_user_text(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.handler import UniversalToolHandler

    monkeypatch.setenv("ABSTRACT_COUNTRY", "FR")

    class _DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_capabilities(self) -> list[str]:
            return []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerateResponse(content="ok", finish_reason="stop", gen_time=1.0)

    client = object.__new__(LocalAbstractCoreLLMClient)
    client._provider = "dummy"
    client._model = "openai/gpt-oss-20b"
    client._artifact_store = None
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    result = client.generate(prompt="describe this picture", media=[{"content": b"png", "content_type": "image/png", "type": "image"}])

    assert client._llm.calls, "expected a provider call"
    prompt_sent = str(client._llm.calls[0].get("prompt") or "")
    assert prompt_sent == "describe this picture"
    assert "<runtime_metadata>" not in prompt_sent
    assert isinstance(result.get("metadata"), dict)
    assert result["metadata"]["runtime_grounding"]["country"] == "FR"
    assert result["metadata"]["runtime_grounding"]["prompt_injected"] is False


def test_runtime_metadata_echo_is_sanitized_from_response_text() -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    result = {
        "content": '<runtime_metadata>{"country":"FR"}</runtime_metadata>\nhello',
        "text": {"content": '<runtime_metadata>{"country":"FR"}</runtime_metadata>\nhello'},
    }

    llm_client._sanitize_runtime_grounding_echoes(result)

    assert result["content"] == "hello"
    assert result["text"]["content"] == "hello"


def test_country_detection_ignores_encoding_only_locale_values(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    monkeypatch.delenv("ABSTRACT_COUNTRY", raising=False)
    monkeypatch.delenv("ABSTRACTFRAMEWORK_COUNTRY", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    monkeypatch.setenv("LC_CTYPE", "UTF-8")

    monkeypatch.setattr(llm_client.locale, "getlocale", lambda: (None, None))
    monkeypatch.setattr(llm_client, "_detect_timezone_name", lambda: None)
    monkeypatch.setattr(llm_client, "_ZONEINFO_TAB_CANDIDATES", [])

    assert llm_client._detect_country() == "XX"


def test_country_detection_parses_locale_region_from_lang_env(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    monkeypatch.delenv("ABSTRACT_COUNTRY", raising=False)
    monkeypatch.delenv("ABSTRACTFRAMEWORK_COUNTRY", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_CTYPE", raising=False)
    monkeypatch.setenv("LANG", "fr_FR.UTF-8")

    monkeypatch.setattr(llm_client.locale, "getlocale", lambda: (None, None))

    assert llm_client._detect_country() == "FR"


def test_country_detection_prefers_timezone_over_inherited_locale(monkeypatch, tmp_path) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    monkeypatch.delenv("ABSTRACT_COUNTRY", raising=False)
    monkeypatch.delenv("ABSTRACTFRAMEWORK_COUNTRY", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_CTYPE", raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")

    monkeypatch.setattr(llm_client.locale, "getlocale", lambda: ("en_US", "UTF-8"))
    monkeypatch.setattr(llm_client, "_detect_timezone_name", lambda: "Europe/Paris")

    zone_tab = tmp_path / "zone.tab"
    zone_tab.write_text("# test\nFR\t+4852+00220\tEurope/Paris\n", encoding="utf-8")
    monkeypatch.setattr(llm_client, "_ZONEINFO_TAB_CANDIDATES", [str(zone_tab)])

    assert llm_client._detect_country() == "FR"


def test_country_detection_falls_back_to_timezone_zone_tab(monkeypatch, tmp_path) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    monkeypatch.delenv("ABSTRACT_COUNTRY", raising=False)
    monkeypatch.delenv("ABSTRACTFRAMEWORK_COUNTRY", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    monkeypatch.delenv("LC_CTYPE", raising=False)

    monkeypatch.setattr(llm_client.locale, "getlocale", lambda: (None, None))
    monkeypatch.setattr(llm_client, "_detect_timezone_name", lambda: "Europe/Paris")

    zone_tab = tmp_path / "zone.tab"
    zone_tab.write_text("# test\nFR\t+4852+00220\tEurope/Paris\n", encoding="utf-8")
    monkeypatch.setattr(llm_client, "_ZONEINFO_TAB_CANDIDATES", [str(zone_tab)])

    assert llm_client._detect_country() == "FR"
