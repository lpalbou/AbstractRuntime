"""abstractruntime.core.config

Runtime configuration for resource limits and model capabilities.

This module provides a RuntimeConfig dataclass that centralizes configuration
for runtime resource limits (iterations, tokens, history) and model capabilities.
The config is used to initialize the `_limits` namespace in RunState.vars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .vars import DEFAULT_MAX_TOKENS

# Truncation policy: keep mechanisms, but default to disabled.
# A positive value enables a conservative auto-cap for `max_input_tokens` when callers do not
# explicitly set an input budget. `-1` disables this cap (no automatic truncation).
DEFAULT_RECOMMENDED_MAX_INPUT_TOKENS = -1


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
    max_input_tokens: Optional[int] = None  # None = auto-calculate from max_tokens/max_output_tokens
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
        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = self.model_capabilities.get("max_tokens")
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS

        max_output_tokens = self.max_output_tokens
        if max_output_tokens is None:
            # Best-effort: persist the provider/model default so agent logic can reason about
            # output-size constraints (e.g., chunk large tool arguments like file contents).
            max_output_tokens = self.model_capabilities.get("max_output_tokens")
        # If capabilities are unavailable and max_output_tokens is unset, keep it as None
        # (meaning: provider default). Do not force a conservative output cap here.

        # ADR-0008 alignment:
        # - max_tokens: total context window size
        # - max_output_tokens: output budget
        # - max_input_tokens: explicit or derived input budget (may be smaller than max_tokens-max_output_tokens)
        #
        # Constraint: max_input_tokens + max_output_tokens + delta <= max_tokens
        delta = 256
        effective_max_input_tokens = self.max_input_tokens

        try:
            max_tokens_int = int(max_tokens) if max_tokens is not None else None
        except Exception:
            max_tokens_int = None
        try:
            max_output_int = int(max_output_tokens) if max_output_tokens is not None else None
        except Exception:
            max_output_int = None

        if (
            max_tokens_int is not None
            and max_tokens_int > 0
            and max_output_int is not None
            and max_output_int >= 0
            and effective_max_input_tokens is not None
        ):
            # If callers explicitly set max_input_tokens, clamp it to the context-window constraint.
            max_allowed_in = max(0, int(max_tokens_int) - int(max_output_int) - int(delta))
            try:
                effective_max_input_tokens = int(effective_max_input_tokens)
            except Exception:
                effective_max_input_tokens = max_allowed_in
            if effective_max_input_tokens < 0:
                effective_max_input_tokens = 0
            if effective_max_input_tokens > max_allowed_in:
                effective_max_input_tokens = max_allowed_in

        # Optional conservative auto-cap (disabled by default with -1).
        if (
            self.max_input_tokens is None
            and effective_max_input_tokens is not None
            and isinstance(DEFAULT_RECOMMENDED_MAX_INPUT_TOKENS, int)
            and DEFAULT_RECOMMENDED_MAX_INPUT_TOKENS > 0
        ):
            try:
                effective_max_input_tokens = min(int(effective_max_input_tokens), int(DEFAULT_RECOMMENDED_MAX_INPUT_TOKENS))
            except Exception:
                pass

        return {
            # Iteration control
            "max_iterations": self.max_iterations,
            "current_iteration": 0,

            # Token management
            "max_tokens": max_tokens,
            "max_output_tokens": max_output_tokens,
            "max_input_tokens": effective_max_input_tokens,
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
            max_input_tokens=self.max_input_tokens,
            warn_tokens_pct=self.warn_tokens_pct,
            max_history_messages=self.max_history_messages,
            provider=self.provider,
            model=self.model,
            model_capabilities=capabilities,
        )
