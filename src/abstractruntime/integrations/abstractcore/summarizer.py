"""abstractruntime.integrations.abstractcore.summarizer

Integration with AbstractCore's BasicSummarizer for chat compaction.

This module provides a wrapper around AbstractCore's BasicSummarizer
that respects environment token limits (max_tokens, max_output_tokens)
for adaptive chunking during conversation compaction.

Design:
- The kernel (runtime.py) uses this via dependency injection
- BasicSummarizer handles adaptive chunking based on max_tokens
- When max_tokens == -1: Uses model's full capability (AUTO mode)
- When max_tokens > 0: Uses explicit limit (environment constraint)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class ChatSummarizer(Protocol):
    """Protocol for chat history summarization.

    This protocol allows the runtime kernel to use summarization
    without directly importing AbstractCore.
    """

    def summarize_chat_history(
        self,
        messages: List[Dict[str, Any]],
        *,
        preserve_recent: int = 6,
        focus: Optional[str] = None,
        compression_mode: str = "standard",
    ) -> Dict[str, Any]:
        """Summarize chat history with adaptive chunking.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            preserve_recent: Number of recent messages to keep intact (default 6)
            focus: Optional focus for summarization
            compression_mode: How aggressively to compress (light|standard|heavy)

        Returns:
            Dict with keys: summary, key_points, confidence, focus_alignment
        """
        ...


class AbstractCoreChatSummarizer:
    """Wrapper around AbstractCore's BasicSummarizer for runtime integration.

    This class:
    - Wraps BasicSummarizer with token limits from RuntimeConfig
    - Handles adaptive chunking automatically via BasicSummarizer
    - Returns JSON-safe dicts for storage in RunState.vars

    Example:
        >>> summarizer = AbstractCoreChatSummarizer(
        ...     llm=llm_instance,
        ...     max_tokens=32768,
        ...     max_output_tokens=4096,
        ... )
        >>> result = summarizer.summarize_chat_history(messages)
        >>> print(result["summary"])
    """

    def __init__(
        self,
        llm,
        *,
        max_tokens: int = -1,
        max_output_tokens: int = -1,
    ):
        """Initialize the summarizer with token limits.

        Args:
            llm: AbstractCore LLM instance (from create_llm or provider)
            max_tokens: Maximum context tokens. -1 = AUTO (use model capability)
            max_output_tokens: Maximum output tokens. -1 = AUTO
        """
        from abstractcore.processing import BasicSummarizer

        self._summarizer = BasicSummarizer(
            llm=llm,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
        )
        self._max_tokens = max_tokens
        self._max_output_tokens = max_output_tokens

    def summarize_chat_history(
        self,
        messages: List[Dict[str, Any]],
        *,
        preserve_recent: int = 6,
        focus: Optional[str] = None,
        compression_mode: str = "standard",
    ) -> Dict[str, Any]:
        """Summarize chat history with adaptive chunking.

        When max_tokens > 0 and messages exceed the limit, BasicSummarizer
        automatically uses map-reduce chunking to process the content.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            preserve_recent: Number of recent messages to keep intact (default 6)
            focus: Optional focus for summarization (e.g., "key decisions")
            compression_mode: How aggressively to compress (light|standard|heavy)

        Returns:
            Dict with keys:
                - summary: The summarized text
                - key_points: List of key points extracted
                - confidence: Confidence score (0-1)
                - focus_alignment: How well summary addresses focus (0-1)
        """
        from abstractcore.processing import CompressionMode

        # Map string to enum
        mode_map = {
            "light": CompressionMode.LIGHT,
            "standard": CompressionMode.STANDARD,
            "heavy": CompressionMode.HEAVY,
        }
        mode = mode_map.get(compression_mode.lower(), CompressionMode.STANDARD)

        # Call BasicSummarizer - it handles adaptive chunking internally
        result = self._summarizer.summarize_chat_history(
            messages=messages,
            preserve_recent=preserve_recent,
            focus=focus,
            compression_mode=mode,
        )

        # Return as JSON-safe dict for storage in RunState.vars
        return {
            "summary": result.summary,
            "key_points": list(result.key_points) if result.key_points else [],
            "confidence": float(result.confidence) if result.confidence else None,
            "focus_alignment": float(result.focus_alignment) if result.focus_alignment else None,
            "word_count_original": result.word_count_original,
            "word_count_summary": result.word_count_summary,
        }

    @property
    def max_tokens(self) -> int:
        """Current max_tokens setting."""
        return self._max_tokens

    @property
    def max_output_tokens(self) -> int:
        """Current max_output_tokens setting."""
        return self._max_output_tokens
