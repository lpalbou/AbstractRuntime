"""abstractruntime.core.config

Runtime configuration for resource limits and model capabilities.

This module provides a RuntimeConfig dataclass that centralizes configuration
for runtime resource limits (iterations, tokens, history) and model capabilities.
The config is used to initialize the `_limits` namespace in RunState.vars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for runtime resource limits and model capabilities.

    This configuration is used by the Runtime to:
    1. Initialize the `_limits` namespace in RunState.vars when starting a run
    2. Provide model capability information for resource tracking
    3. Configure warning thresholds for proactive notifications

    Attributes:
        max_iterations: Maximum number of reasoning iterations (default: 25)
        warn_iterations_pct: Percentage threshold for iteration warnings (default: 80)
        max_tokens: Maximum context window tokens (None = use model capabilities)
        max_output_tokens: Maximum tokens for LLM response (None = provider default)
        warn_tokens_pct: Percentage threshold for token warnings (default: 80)
        max_history_messages: Maximum conversation history messages (-1 = unlimited)
        provider: Default provider id for this Runtime (best-effort; used for run metadata)
        model: Default model id for this Runtime (best-effort; used for run metadata)
        model_capabilities: Dict of model capabilities from LLM provider

    Example:
        >>> config = RuntimeConfig(max_iterations=50, max_tokens=65536)
        >>> limits = config.to_limits_dict()
        >>> limits["max_iterations"]
        50
    """

    # Iteration control
    max_iterations: int = 25
    warn_iterations_pct: int = 80

    # Token/context window management
    max_tokens: Optional[int] = None  # None = query from model capabilities
    max_output_tokens: Optional[int] = None  # None = use provider default
    warn_tokens_pct: int = 80

    # History management
    max_history_messages: int = -1  # -1 = unlimited (send all messages)

    # Default routing metadata (optional; depends on how the Runtime was constructed)
    provider: Optional[str] = None
    model: Optional[str] = None

    # Model capabilities (populated from LLM client)
    model_capabilities: Dict[str, Any] = field(default_factory=dict)

    def to_limits_dict(self) -> Dict[str, Any]:
        """Convert to _limits namespace dict for RunState.vars.

        Returns:
            Dict with canonical limit values for storage in RunState.vars["_limits"].
            Uses model_capabilities as fallback for max_tokens if not explicitly set.
        """
        max_output_tokens = self.max_output_tokens
        if max_output_tokens is None:
            # Best-effort: persist the provider/model default so agent logic can reason about
            # output-size constraints (e.g., chunk large tool arguments like file contents).
            max_output_tokens = self.model_capabilities.get("max_output_tokens")
        return {
            # Iteration control
            "max_iterations": self.max_iterations,
            "current_iteration": 0,

            # Token management
            "max_tokens": self.max_tokens or self.model_capabilities.get("max_tokens", 32768),
            "max_output_tokens": max_output_tokens,
            "estimated_tokens_used": 0,

            # History management
            "max_history_messages": self.max_history_messages,

            # Warning thresholds
            "warn_iterations_pct": self.warn_iterations_pct,
            "warn_tokens_pct": self.warn_tokens_pct,
        }

    def with_capabilities(self, capabilities: Dict[str, Any]) -> "RuntimeConfig":
        """Create a new RuntimeConfig with updated model capabilities.

        This is useful for merging model capabilities from an LLM client
        into an existing configuration.

        Args:
            capabilities: Dict of model capabilities (e.g., from get_model_capabilities())

        Returns:
            New RuntimeConfig with merged capabilities
        """
        return RuntimeConfig(
            max_iterations=self.max_iterations,
            warn_iterations_pct=self.warn_iterations_pct,
            max_tokens=self.max_tokens,
            max_output_tokens=self.max_output_tokens,
            warn_tokens_pct=self.warn_tokens_pct,
            max_history_messages=self.max_history_messages,
            provider=self.provider,
            model=self.model,
            model_capabilities=capabilities,
        )
