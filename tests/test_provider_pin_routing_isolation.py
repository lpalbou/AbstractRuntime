"""Per-call provider pins must reach the provider the author pinned.

Routing defect 2026-07-31 (36-run live benchmark wave): every bundle run that
pinned `lmstudio` / `ollama` / `openai` was served by the gateway's DEFAULT
endpoint profile instead. The pool passed its construction kwargs -- which carry
the default endpoint profile's `base_url` and `api_key` -- to every client it
built, so `create_llm("lmstudio", base_url="<relay>/v1", api_key="<relay key>")`
produced an LM Studio client aimed at the relay. Nothing in the ledger showed it:
the record declared the REQUEST, so the misroute survived 36 runs.

These tests pin the two invariants that make that impossible to reintroduce:
  1. connection-scoped kwargs travel only with the identity they belong to;
  2. an LLM result discloses the endpoint it was ACTUALLY served by.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from abstractruntime.integrations.abstractcore import llm_client as llm_client_mod
from abstractruntime.integrations.abstractcore.llm_client import (
    MultiLocalAbstractCoreLLMClient,
    _models_agree,
    _split_connection_scoped_llm_kwargs,
    _stamp_effective_route,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8317/v1"
DEFAULT_API_KEY = "default-profile-key"


class _FakeCoreLLM:
    def __init__(self, provider: str, model: str, **kwargs: Any) -> None:
        self.provider = provider
        self.model = model
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.kwargs = dict(kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"content": "ok", "model": self.model}


class _FakePooledClient:
    """Stands in for LocalAbstractCoreLLMClient without touching a provider."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        artifact_store: Optional[Any] = None,
        **_ignored: Any,
    ) -> None:
        self._provider = provider
        self._model = model
        self._llm_kwargs = dict(llm_kwargs or {})
        self._artifact_store = artifact_store
        self._llm = _FakeCoreLLM(provider, model, **self._llm_kwargs)
        self.generate_calls: List[Dict[str, Any]] = []

    def generate(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.generate_calls.append({"prompt": prompt, "params": dict(params or {})})
        return {"content": "ok", "model": self._model}

    def set_on_token(self, callback: Any) -> None:  # pragma: no cover - pool fan-out
        self._on_token = callback


@pytest.fixture()
def pool(monkeypatch: pytest.MonkeyPatch) -> MultiLocalAbstractCoreLLMClient:
    monkeypatch.setattr(llm_client_mod, "LocalAbstractCoreLLMClient", _FakePooledClient)
    return MultiLocalAbstractCoreLLMClient(
        provider="openai-compatible",
        model="gpt-5.4",
        llm_kwargs={
            "base_url": DEFAULT_BASE_URL,
            "api_key": DEFAULT_API_KEY,
            "enable_tracing": True,
            "max_traces": 50,
        },
    )


def test_default_provider_keeps_its_configured_endpoint(pool: MultiLocalAbstractCoreLLMClient) -> None:
    client = pool._get_client("openai-compatible", "gpt-5.4")
    assert client._llm_kwargs["base_url"] == DEFAULT_BASE_URL
    assert client._llm_kwargs["api_key"] == DEFAULT_API_KEY


@pytest.mark.parametrize(
    ("provider", "model"),
    [("lmstudio", "qwen/qwen3.6-27b"), ("ollama", "gemma3:1b"), ("openai", "gpt-5.6-sol"), ("anthropic", "claude-x")],
)
def test_pinned_provider_never_inherits_the_default_endpoint(
    pool: MultiLocalAbstractCoreLLMClient, provider: str, model: str
) -> None:
    client = pool._get_client(provider, model)
    assert "base_url" not in client._llm_kwargs, f"{provider} inherited the default endpoint address"
    assert "api_key" not in client._llm_kwargs, f"{provider} inherited the default endpoint credential"


def test_pinned_provider_still_inherits_provider_agnostic_kwargs(pool: MultiLocalAbstractCoreLLMClient) -> None:
    client = pool._get_client("lmstudio", "qwen/qwen3.6-27b")
    assert client._llm_kwargs["enable_tracing"] is True
    assert client._llm_kwargs["max_traces"] == 50


def test_explicit_per_call_endpoint_override_still_wins(pool: MultiLocalAbstractCoreLLMClient) -> None:
    client = pool._get_client(
        "openai-compatible",
        "llama-3",
        llm_kwargs_override={"base_url": "https://other.example.test/v1", "api_key": "other-key"},
    )
    assert client._llm_kwargs["base_url"] == "https://other.example.test/v1"
    assert client._llm_kwargs["api_key"] == "other-key"


def test_generate_routes_the_pin_and_discloses_the_endpoint(pool: MultiLocalAbstractCoreLLMClient) -> None:
    result = pool.generate(prompt="hi", params={"_provider": "lmstudio", "_model": "qwen/qwen3.6-27b"})
    route = result["route"]
    assert route["provider"] == "lmstudio"
    assert route["model"] == "qwen/qwen3.6-27b"
    assert route["base_url"] != DEFAULT_BASE_URL
    assert route["mismatch"] is False


def test_split_keeps_connection_kwargs_apart() -> None:
    shared, connection = _split_connection_scoped_llm_kwargs(
        {"base_url": "u", "api_key": "k", "enable_tracing": True, "timeout": 30}
    )
    assert shared == {"enable_tracing": True, "timeout": 30}
    assert connection == {"base_url": "u", "api_key": "k"}


def test_route_disclosure_flags_a_served_model_that_is_not_the_pinned_one() -> None:
    client = _FakePooledClient(provider="openai", model="gpt-5.6-sol", llm_kwargs={"base_url": DEFAULT_BASE_URL})
    result = _stamp_effective_route(
        {"content": "x", "model": "gpt-5.4-mini-2026-03-17"},
        requested_provider="openai",
        requested_model="gpt-5.6-sol",
        client=client,
    )
    assert result["route"]["mismatch"] is True
    assert result["route"]["served_model"] == "gpt-5.4-mini-2026-03-17"


@pytest.mark.parametrize(
    ("requested", "served", "agree"),
    [
        ("gpt-5.4-mini", "gpt-5.4-mini-2026-03-17", True),
        ("qwen/qwen3.6-27b", "qwen3.6-27b", True),
        ("gpt-5.6-sol", "gpt-5.4-mini", False),
        ("qwen/qwen3.6-27b", "gpt-5.4-mini-2026-03-17", False),
    ],
)
def test_model_agreement_tolerates_snapshots_not_substitutions(requested: str, served: str, agree: bool) -> None:
    assert _models_agree(requested, served) is agree
