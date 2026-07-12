"""In-process token callback for streamed LLM calls (code seat c990's ask).

Pins the contract: `on_token(delta, meta)` fires per content chunk,
BEST-EFFORT and never load-bearing — the durable aggregated result is
byte-identical with or without a callback, and a raising callback is disabled
mid-stream (one warning) without failing the call.
"""

from __future__ import annotations

from typing import Any, Dict, List

from abstractruntime.integrations.abstractcore.llm_client import (
    _normalize_local_streaming_response,
)


def _chunks() -> List[Dict[str, Any]]:
    return [
        {"content": "Hel", "model": "m1"},
        {"content": "lo ", "metadata": {"trace_id": "t-1"}},
        {"tool_calls": None, "usage": {"total_tokens": 5}},
        {"content": "world", "finish_reason": "stop"},
    ]


def test_on_token_fires_per_content_delta_and_result_is_unchanged() -> None:
    seen: List[str] = []
    with_cb = _normalize_local_streaming_response(
        iter(_chunks()), on_token=lambda delta, meta: seen.append(delta)
    )
    without_cb = _normalize_local_streaming_response(iter(_chunks()))

    assert seen == ["Hel", "lo ", "world"]
    assert with_cb["content"] == without_cb["content"] == "Hello world"
    # The durable result is identical with or without a callback (timing aside).
    for key in ("content", "tool_calls", "usage", "model", "finish_reason", "trace_id"):
        assert with_cb[key] == without_cb[key]


def test_raising_callback_is_contained_and_disabled() -> None:
    calls: List[str] = []

    def bad(delta: str, meta: Dict[str, Any]) -> None:
        calls.append(delta)
        raise RuntimeError("hostile callback")

    result = _normalize_local_streaming_response(iter(_chunks()), on_token=bad)
    # First delta reached the callback; the raise disabled it; the call survived.
    assert calls == ["Hel"]
    assert result["content"] == "Hello world"


def test_non_callable_on_token_is_ignored() -> None:
    result = _normalize_local_streaming_response(iter(_chunks()), on_token="not-a-function")
    assert result["content"] == "Hello world"


class _RecordingProvider:
    """Fake AbstractCore provider: records the stream kwarg, returns a plain reply."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate(self, **kwargs: Any):
        self.calls.append(kwargs)

        class _Resp:
            content = "final"
            raw_response = None
            tool_calls = None
            usage = {"total_tokens": 3}
            model = "fake"
            finish_reason = "stop"
            metadata: Dict[str, Any] = {}
            gen_time = None

        return _Resp()


def _bare_client(provider: Any):
    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient

    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    client._llm = provider  # type: ignore[attr-defined]
    client._provider = "fake"  # type: ignore[attr-defined]
    client._model = "fake"  # type: ignore[attr-defined]
    client._artifact_store = None  # type: ignore[attr-defined]
    client._generate_lock = None  # type: ignore[attr-defined]
    client._on_token = None  # type: ignore[attr-defined]
    import threading as _threading

    client._prompt_cache_state_lock = _threading.Lock()  # type: ignore[attr-defined]
    client._prompt_cache_state = {}  # type: ignore[attr-defined]
    return client


def test_structured_output_requests_force_the_non_streamed_path() -> None:
    """Code seat c1009 defect 1: a review/structured call under stream=True
    completed with an EMPTY answer — the streamed normalizer cannot produce
    validated `data` or artifact-backed outputs. Structured calls now force
    stream=False (correctness over rendering; on_token stays silent)."""
    provider = _RecordingProvider()
    client = _bare_client(provider)

    out = client.generate(
        prompt="review this",
        params={"stream": True, "response_format": {"type": "json_schema"}},
    )
    assert out["content"] == "final"
    assert provider.calls[-1]["stream"] is False  # forced non-stream

    out2 = client.generate(prompt="chat", params={"stream": True})
    assert out2["content"] == "final"  # plain call: provider returned non-iterator, normalized fine
    assert provider.calls[-1]["stream"] is True  # plain text calls still stream


def test_multilocal_pool_fans_out_on_token() -> None:
    """Code seat c1009 ask 3: registering on the pool reaches existing AND
    lazily-created clients (per-request overrides would otherwise silently
    not stream)."""
    from abstractruntime.integrations.abstractcore.llm_client import MultiLocalAbstractCoreLLMClient

    pool = MultiLocalAbstractCoreLLMClient.__new__(MultiLocalAbstractCoreLLMClient)
    pool._clients = {}  # type: ignore[attr-defined]
    pool._override_clients = {}  # type: ignore[attr-defined]
    existing = _bare_client(_RecordingProvider())
    pool._clients[("fake", "m1")] = existing  # type: ignore[attr-defined]
    pool._default_client = existing  # type: ignore[attr-defined]

    cb = lambda delta, meta: None  # noqa: E731
    pool.set_on_token(cb)
    assert existing._on_token is cb
    assert pool._pool_on_token is cb  # applied to future clients in _create_client

    pool.set_on_token(None)
    assert existing._on_token is None
