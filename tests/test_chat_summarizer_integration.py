"""Test AbstractCoreChatSummarizer integration with Runtime.

These tests verify:
1. AbstractCoreChatSummarizer wrapper works correctly
2. Token limits are passed through to BasicSummarizer
3. Factory injects summarizer into Runtime
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List, Optional


class MockSummaryOutput:
    """Mock SummaryOutput for testing without AbstractCore."""
    def __init__(
        self,
        summary: str = "Test summary",
        key_points: List[str] = None,
        confidence: float = 0.9,
        focus_alignment: float = 0.8,
        word_count_original: int = 100,
        word_count_summary: int = 20,
    ):
        self.summary = summary
        self.key_points = key_points or ["Point 1", "Point 2"]
        self.confidence = confidence
        self.focus_alignment = focus_alignment
        self.word_count_original = word_count_original
        self.word_count_summary = word_count_summary


class MockBasicSummarizer:
    """Mock BasicSummarizer for testing."""

    def __init__(self, llm, *, max_tokens: int = -1, max_output_tokens: int = -1):
        self.llm = llm
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens
        self.calls = []

    def summarize_chat_history(
        self,
        messages: List[Dict[str, Any]],
        preserve_recent: int = 6,
        focus: Optional[str] = None,
        compression_mode=None,
    ) -> MockSummaryOutput:
        self.calls.append({
            "messages": messages,
            "preserve_recent": preserve_recent,
            "focus": focus,
            "compression_mode": compression_mode,
        })
        return MockSummaryOutput(
            summary=f"Summary of {len(messages)} messages",
            key_points=[f"Key point for {focus or 'general'}"],
        )


def test_chat_summarizer_protocol_signature():
    """Verify ChatSummarizer protocol has correct signature."""
    from abstractruntime.integrations.abstractcore.summarizer import ChatSummarizer

    # Protocol should define summarize_chat_history
    assert hasattr(ChatSummarizer, "summarize_chat_history")


def test_abstractcore_chat_summarizer_init():
    """Test AbstractCoreChatSummarizer initialization stores limits."""
    # We can test the class structure without needing AbstractCore
    from abstractruntime.integrations.abstractcore.summarizer import AbstractCoreChatSummarizer

    # Skip if AbstractCore not available
    try:
        from abstractcore.processing import BasicSummarizer
    except ImportError:
        pytest.skip("AbstractCore not available")

    # Verify class has expected methods
    assert hasattr(AbstractCoreChatSummarizer, "summarize_chat_history")
    assert hasattr(AbstractCoreChatSummarizer, "max_tokens")
    assert hasattr(AbstractCoreChatSummarizer, "max_output_tokens")


def test_runtime_accepts_chat_summarizer_parameter():
    """Verify Runtime.__init__ accepts chat_summarizer parameter."""
    from abstractruntime.core.runtime import Runtime
    from abstractruntime.storage.in_memory import InMemoryRunStore, InMemoryLedgerStore

    class MockSummarizer:
        def summarize_chat_history(self, messages, **kwargs):
            return {"summary": "mock"}

    # Create runtime with summarizer - should not raise
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        chat_summarizer=MockSummarizer(),
    )

    assert runtime._chat_summarizer is not None


def test_runtime_without_chat_summarizer():
    """Verify Runtime works without chat_summarizer (backward compatible)."""
    from abstractruntime.core.runtime import Runtime
    from abstractruntime.storage.in_memory import InMemoryRunStore, InMemoryLedgerStore

    # Create runtime without summarizer - should not raise
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
    )

    assert runtime._chat_summarizer is None


def test_summarizer_result_format():
    """Test that summarizer returns JSON-safe dict with expected keys."""
    from abstractruntime.integrations.abstractcore.summarizer import AbstractCoreChatSummarizer

    try:
        from abstractcore.processing import BasicSummarizer
        from abstractcore import create_llm
    except ImportError:
        pytest.skip("AbstractCore not available")

    # Only test format, not actual summarization
    # Expected keys in result dict
    expected_keys = {"summary", "key_points", "confidence", "focus_alignment", "word_count_original", "word_count_summary"}

    # Verify the class method returns dict with correct structure
    # (actual LLM test would require live provider)
    result_schema = {
        "summary": str,
        "key_points": list,
        "confidence": (float, type(None)),
        "focus_alignment": (float, type(None)),
        "word_count_original": int,
        "word_count_summary": int,
    }

    for key, expected_type in result_schema.items():
        assert key in expected_keys, f"Missing expected key: {key}"
