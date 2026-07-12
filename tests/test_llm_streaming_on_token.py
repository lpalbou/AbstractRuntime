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
