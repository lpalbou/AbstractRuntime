from __future__ import annotations

import re

import pytest

from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient, RemoteAbstractCoreLLMClient


pytestmark = pytest.mark.basic


_HEADER_RE = re.compile(
    r"^Grounding:\s*\d{4}/\d{2}/\d{2}\|\d{2}:\d{2}\|[A-Z]{2}$",
    re.IGNORECASE,
)


def test_local_llm_client_injects_date_and_country_into_user_prompt(monkeypatch) -> None:
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
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    result = client.generate(prompt="hello")

    assert client._llm.calls, "expected a provider call"
    prompt_sent = str(client._llm.calls[0].get("prompt") or "")
    first = prompt_sent.splitlines()[0].strip()
    assert _HEADER_RE.match(first)
    assert first.endswith("|FR")
    assert isinstance(result.get("metadata"), dict)
    payload = result["metadata"]["_provider_request"]["payload"]
    assert payload["messages"][0]["role"] == "user"
    assert str(payload["messages"][0]["content"] or "").splitlines()[0].strip().endswith("|FR")


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
    client._llm = _DummyLLM()
    client._tool_handler = UniversalToolHandler(client._model)

    sys_in = "Grounding: 1999/12/31|23:59|FR\n\nBase system prompt."
    client.generate(prompt="hello", system_prompt=sys_in)

    sys = str(client._llm.calls[0].get("system_prompt") or "")
    assert not sys.startswith("Grounding:")
    assert sys.strip() == "Base system prompt."
    prompt_sent = str(client._llm.calls[0].get("prompt") or "")
    first = prompt_sent.splitlines()[0].strip()
    assert _HEADER_RE.match(first)
    assert first.endswith("|FR")


def test_remote_llm_client_injects_system_context_into_user_prompt(monkeypatch) -> None:
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
    first = str(body["messages"][0]["content"] or "").splitlines()[0].strip()
    assert _HEADER_RE.match(first)
    assert first.endswith("|FR")


def test_system_context_header_is_per_call(monkeypatch) -> None:
    import abstractruntime.integrations.abstractcore.llm_client as llm_client

    counter = {"n": 0}

    def _fake_header() -> str:
        counter["n"] += 1
        # keep the same format so other consumers remain compatible
        return f"Grounding: 2000/01/01|00:0{counter['n']}|FR"

    monkeypatch.setattr(llm_client, "_system_context_header", _fake_header)

    a, _ = llm_client._inject_turn_grounding(prompt="hello", messages=None)
    b, _ = llm_client._inject_turn_grounding(prompt="hello", messages=None)

    assert a.splitlines()[0].strip() != b.splitlines()[0].strip()
    assert counter["n"] == 2


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
