"""Pooled LLM clients must strip construction-time prompt_cache_key (agency-parity 0221).

MultiLocalAbstractCoreLLMClient pools one provider instance per (provider, model) across all
runs/sessions of a runtime; a construction-time instance-default key would stamp one session's
cache identity onto every tenant's traffic. Session-scoped keys are injected per call by the
LLM_CALL handler instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def test_pooled_client_strips_prompt_cache_key(monkeypatch):
    from abstractruntime.integrations.abstractcore import llm_client as mod

    captured: Dict[str, Any] = {}

    class _StubLocal:
        def __init__(self, *, provider: str, model: str, llm_kwargs: Optional[Dict[str, Any]] = None, **kw):
            captured["llm_kwargs"] = dict(llm_kwargs or {})
            self._llm = object()

    monkeypatch.setattr(mod, "LocalAbstractCoreLLMClient", _StubLocal)

    mod.MultiLocalAbstractCoreLLMClient(
        provider="lmstudio",
        model="stub-model",
        llm_kwargs={"prompt_cache_key": "sess-abc", "temperature": 0.1},
    )

    assert "prompt_cache_key" not in captured["llm_kwargs"]
    assert captured["llm_kwargs"].get("temperature") == 0.1
