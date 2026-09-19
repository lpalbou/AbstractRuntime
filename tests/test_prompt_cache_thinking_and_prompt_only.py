"""Regression pins (2026-09-17): two ways the runtime made a session cache unreachable.

Both were found on one live AbstractAssistant session (`mlx-community/Qwen3.8-27B-4bit`)
where every turn paid ~30 s for a ~5.5k-token context that never changed.

1. THE PREFIX WAS PLANNED WITHOUT THE THINKING REQUEST. The runtime prepares a
   (system, tools) bloc chain and forks it into the session key, then calls
   `generate(thinking=...)`. Qwen3.8 renders the effort level as a sentence at the HEAD
   of the system block, so the prepared prefix and the real prompt agreed on 3 tokens.
   AbstractCore owns that rewrite; the runtime's job is to hand it the same request.

2. PROMPT-ONLY CALLS WERE APPENDED. `messages=None` tells AbstractCore "the cache is my
   context, append this fragment". A runtime llm_call re-sends everything every time, so
   the assistant's `route_call` stacked its whole prompt onto one key each turn
   (6790 → 10130 → 13480 cached tokens), never reused a token, and showed the router
   every earlier routing request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from abstractruntime.integrations.abstractcore import llm_client
from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient

_SYS = "SYSTEM"
_TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]
_ATTRIBUTION = {
    "session_id": "sess-1",
    "run_id": "run-1",
    "workflow_id": "wf-1",
    "node_id": "route_call",
    "namespace": "session",
}


class _Provider:
    """Local control-plane fake. `accepts_thinking` selects the prepare signature."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []
        self.meta: Dict[str, Dict[str, Any]] = {}
        self.generate_calls: List[Dict[str, Any]] = []

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_operation(self, operation: str) -> bool:
        return True

    def prompt_cache_key_meta(self, key: Any) -> Dict[str, Any]:
        return dict(self.meta.get(str(key)) or {})

    def prompt_cache_update_key_meta(self, key: Any, **updates: Any) -> bool:
        entry = self.meta.setdefault(str(key), {})
        entry.update({k: v for k, v in updates.items() if v is not None})
        return True

    def _prepare(self, namespace: str, modules: List[Dict[str, Any]], thinking: Any) -> Dict[str, Any]:
        seed = f"seed|{thinking!r}"
        derived = []
        for m in modules:
            raw = json.dumps(m, sort_keys=True, default=str)
            seed = hashlib.sha256((seed + raw).encode()).hexdigest()
            derived.append(
                {"module_id": m.get("module_id"), "module_hash": seed, "cache_key": f"{namespace}:{seed[:16]}"}
            )
        self.calls.append(("prepare_modules", thinking))
        return {"supported": True, "modules": derived, "final_cache_key": derived[-1]["cache_key"]}

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        thinking: Any = None,
    ) -> Dict[str, Any]:
        return self._prepare(namespace, modules, thinking)

    def prompt_cache_clear(self, key: Optional[str] = None) -> bool:
        self.calls.append(("clear", key))
        self.meta.pop(str(key), None)
        return True

    def prompt_cache_fork(self, from_key: str, to_key: str, **kwargs: Any) -> bool:
        self.calls.append(("fork", from_key, to_key))
        self.meta.setdefault(str(to_key), {})["forked_from"] = str(from_key)
        return True

    def generate(self, **kwargs: Any) -> Dict[str, Any]:
        self.generate_calls.append(dict(kwargs))
        return {"content": "ok"}


class _LegacyProvider(_Provider):
    """An AbstractCore that predates `prepare_modules(thinking=...)`."""

    def prompt_cache_prepare_modules(  # type: ignore[override]
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
    ) -> Dict[str, Any]:
        return self._prepare(namespace, modules, None)


def _client(provider: Any) -> LocalAbstractCoreLLMClient:
    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    client._provider = "mlx"  # type: ignore[attr-defined]
    client._model = "test-model"  # type: ignore[attr-defined]
    client._llm = provider  # type: ignore[attr-defined]
    client._llm_kwargs = {}  # type: ignore[attr-defined]
    client._artifact_store = None  # type: ignore[attr-defined]
    client._prompt_cache_state_lock = threading.Lock()  # type: ignore[attr-defined]
    client._prompt_cache_state = {}  # type: ignore[attr-defined]
    return client


# --------------------------------------------------------------------------
# 1. The thinking request reaches the bloc planner.
# --------------------------------------------------------------------------


def test_prepare_hands_the_thinking_request_to_abstractcore() -> None:
    provider = _Provider()
    _client(provider)._maybe_prepare_prompt_cache(
        prompt_cache_key="session:k", system_prompt=_SYS, tools=_TOOLS, messages=None, thinking="minimal"
    )
    assert provider.calls[0] == ("prepare_modules", "minimal")
    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]


def test_generate_prepares_under_the_same_thinking_it_generates_with() -> None:
    provider = _Provider()
    _client(provider).generate(
        prompt="",
        messages=[{"role": "user", "content": "hi"}],
        system_prompt=_SYS,
        tools=_TOOLS,
        params={
            "thinking": "low",
            "prompt_cache_key": "session:k",
            "_prompt_cache_attribution": dict(_ATTRIBUTION),
        },
    )
    assert provider.calls[0] == ("prepare_modules", "low")
    assert provider.generate_calls[-1]["thinking"] == "low"


def test_a_changed_effort_level_reforks_the_session_key() -> None:
    """The level rewrites the head of the prompt, so the old session cache is a prefix of
    nothing: a different final prefix key must trigger exactly one re-fork."""
    provider = _Provider()
    client = _client(provider)
    for level in ("low", "low", "xhigh"):
        client._maybe_prepare_prompt_cache(
            prompt_cache_key="session:k", system_prompt=_SYS, tools=_TOOLS, messages=None, thinking=level
        )
    assert [c[0] for c in provider.calls] == [
        "prepare_modules", "clear", "fork",   # low: first fork
        "prepare_modules",                    # low again: already forked from this prefix
        "prepare_modules", "clear", "fork",   # xhigh: new prefix identity
    ]


def test_an_abstractcore_without_the_parameter_still_prepares_and_says_so(caplog) -> None:
    """Passing an unknown kwarg would raise TypeError inside a blanket `except` and the
    prefix cache would vanish without a word. Degrade to the old behaviour LOUDLY."""
    provider = _LegacyProvider()
    with caplog.at_level(logging.WARNING):
        _client(provider)._maybe_prepare_prompt_cache(
            prompt_cache_key="session:k", system_prompt=_SYS, tools=_TOOLS, messages=None, thinking="low"
        )
    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]
    assert any("#FALLBACK" in r.getMessage() and "thinking" in r.getMessage() for r in caplog.records)


def test_no_thinking_request_means_no_kwarg_and_no_warning(caplog) -> None:
    provider = _LegacyProvider()
    with caplog.at_level(logging.WARNING):
        _client(provider)._maybe_prepare_prompt_cache(
            prompt_cache_key="session:k", system_prompt=_SYS, tools=_TOOLS, messages=None
        )
    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]
    assert not any("#FALLBACK" in r.getMessage() for r in caplog.records)


def test_callable_accepts_kwarg_reads_signatures_not_names() -> None:
    assert llm_client._callable_accepts_kwarg(_Provider().prompt_cache_prepare_modules, "thinking")
    assert not llm_client._callable_accepts_kwarg(_LegacyProvider().prompt_cache_prepare_modules, "thinking")
    assert llm_client._callable_accepts_kwarg(lambda **kw: None, "thinking")
    assert not llm_client._callable_accepts_kwarg(object(), "thinking")


# --------------------------------------------------------------------------
# 2. Prompt-only calls under a runtime-derived key are full-context.
# --------------------------------------------------------------------------


def test_prompt_only_call_under_a_derived_key_is_sent_full_context() -> None:
    provider = _Provider()
    _client(provider).generate(
        prompt="Route this request",
        system_prompt=_SYS,
        params={"prompt_cache_key": "session:k", "_prompt_cache_attribution": dict(_ATTRIBUTION)},
    )
    sent = provider.generate_calls[-1]
    # `[]`, not None: AbstractCore reads None as "append this fragment to the cache".
    assert sent["messages"] == [] and sent["messages"] is not None
    assert sent["prompt"].rstrip().endswith("Route this request")


class _RemoteStyleProvider(_Provider):
    """A provider with no in-process cache to plan into (Ollama, OpenAI-compatible…)."""

    def prompt_cache_supports_operation(self, operation: str) -> bool:
        return False


def test_a_provider_without_a_local_cache_keeps_its_call_shape() -> None:
    """The rewrite exists for the in-process APPEND lane only. Ollama routes on
    `messages is not None` (`/api/chat` vs `/api/generate`), so rewriting its call shape
    would switch endpoint and template for a cache it does not have."""
    provider = _RemoteStyleProvider()
    _client(provider).generate(
        prompt="Route this request",
        system_prompt=_SYS,
        params={"prompt_cache_key": "session:k", "_prompt_cache_attribution": dict(_ATTRIBUTION)},
    )
    assert provider.generate_calls[-1]["messages"] is None


def test_a_caller_chosen_key_keeps_append_semantics() -> None:
    """No attribution rider = the key was not derived by the runtime. A host that picked
    its own key and sends prompt-only may really want KV-session append; leave it alone."""
    provider = _Provider()
    _client(provider).generate(prompt="next fragment", params={"prompt_cache_key": "my-kv-session"})
    assert provider.generate_calls[-1]["messages"] is None


def test_no_cache_key_leaves_the_call_shape_alone() -> None:
    provider = _Provider()
    _client(provider).generate(prompt="hello", params={"_prompt_cache_attribution": dict(_ATTRIBUTION)})
    assert provider.generate_calls[-1]["messages"] is None


def test_a_transcript_call_is_passed_through_unchanged() -> None:
    provider = _Provider()
    messages = [{"role": "user", "content": "hi"}]
    _client(provider).generate(
        prompt="",
        messages=messages,
        params={"prompt_cache_key": "session:k", "_prompt_cache_attribution": dict(_ATTRIBUTION)},
    )
    sent = provider.generate_calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["user"]
    assert sent[0]["content"].rstrip().endswith("hi")


# --------------------------------------------------------------------------
# 3. `thinking` survives the host-facing control plane (local and remote).
# --------------------------------------------------------------------------


class _Sender:
    def __init__(self) -> None:
        self.posts: List[Tuple[str, Dict[str, Any]]] = []

    def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        self.posts.append((url, dict(json)))
        return {"supported": True, "operation": "prepare_modules", "final_cache_key": "ns:k"}


_MODULES = [{"module_id": "system", "system_prompt": _SYS}]


def test_local_control_plane_forwards_thinking_to_the_provider() -> None:
    """The wrapper used to `_ = kwargs` — a host naming the level planned without it."""
    provider = _Provider()
    provider.get_prompt_cache_capabilities = lambda: {  # type: ignore[attr-defined]
        "supported": True, "mode": "local_control_plane", "supports_prepare_modules": True,
    }
    result = _client(provider).prompt_cache_prepare_modules(
        namespace="ns", modules=list(_MODULES), thinking="xhigh"
    )
    assert result["supported"] is True
    assert provider.calls == [("prepare_modules", "xhigh")]


def test_local_control_plane_refuses_rather_than_plan_a_prefix_that_cannot_match() -> None:
    provider = _LegacyProvider()
    provider.get_prompt_cache_capabilities = lambda: {  # type: ignore[attr-defined]
        "supported": True, "mode": "local_control_plane", "supports_prepare_modules": True,
    }
    result = _client(provider).prompt_cache_prepare_modules(
        namespace="ns", modules=list(_MODULES), thinking="xhigh"
    )
    assert result["supported"] is False and "thinking" in str(result.get("error"))
    assert provider.calls == []
    # ...while a host that names no level is served exactly as before.
    assert _client(provider).prompt_cache_prepare_modules(namespace="ns", modules=list(_MODULES))["supported"] is True


def test_remote_control_plane_puts_thinking_on_the_wire() -> None:
    from abstractruntime.integrations.abstractcore.llm_client import RemoteAbstractCoreLLMClient

    sender = _Sender()
    client = RemoteAbstractCoreLLMClient(server_base_url="http://endpoint", model="m", request_sender=sender)
    client.prompt_cache_prepare_modules(namespace="ns", modules=list(_MODULES), thinking="low", provider="stub", model="m")
    client.prompt_cache_prepare_modules(namespace="ns", modules=list(_MODULES), provider="stub", model="m")
    assert sender.posts[0][0].endswith("/acore/prompt_cache/prepare_modules")
    assert sender.posts[0][1]["thinking"] == "low"
    assert "thinking" not in sender.posts[1][1]


# --------------------------------------------------------------------------
# 4. A session keeps its cache key when its workflow is merely republished.
# --------------------------------------------------------------------------


def test_republishing_a_workflow_does_not_change_a_sessions_cache_key() -> None:
    """Operator's rule: asking a question in an existing session resumes with its cache
    unless the cache was purged. The key hashed the bundle VERSION, so a republish (a
    desktop client does one at launch) sent the same session to a different key."""
    from abstractruntime.integrations.abstractcore.effect_handlers import _derive_prompt_cache_key as key

    common = dict(namespace="session", session_id="sess-1", provider="mlx", model="m")
    before = key(workflow_id="__catalog__v2__tenant__YWJz@0.0.3:c53b1579", node_id="route_call", **common)
    after = key(workflow_id="__catalog__v2__tenant__YWJz@0.0.7:c53b1579", node_id="route_call", **common)
    assert before == after

    sub_before = key(workflow_id="visual_react_agent___catalog__v2__tenant__YWJz_0_0_3_c53b1579_assistant_agent", node_id="reason", **common)
    sub_after = key(workflow_id="visual_react_agent___catalog__v2__tenant__YWJz_0_0_7_c53b1579_assistant_agent", node_id="reason", **common)
    assert sub_before == sub_after


def test_everything_that_really_identifies_a_cache_still_separates_keys() -> None:
    from abstractruntime.integrations.abstractcore.effect_handlers import _derive_prompt_cache_key as key

    base = dict(namespace="session", session_id="sess-1", provider="mlx", model="m",
                workflow_id="assistant@0.0.3:c53b1579", node_id="route_call")
    variants = [
        {**base, "session_id": "sess-2"},
        {**base, "model": "other"},
        {**base, "provider": "huggingface"},
        {**base, "node_id": "reason"},
        {**base, "workflow_id": "assistant@0.0.3:ffffffff"},   # another flow of the bundle
        {**base, "workflow_id": "basic-agent@0.0.3:c53b1579"},  # another bundle
    ]
    keys = {key(**base)} | {key(**v) for v in variants}
    assert len(keys) == 1 + len(variants)
