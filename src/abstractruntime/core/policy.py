"""abstractruntime.core.policy

Effect execution policies for retry and idempotency.

Policies control:
- How many times to retry a failed effect
- Backoff timing between retries
- Idempotency keys for deduplication on crash recovery
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from .models import Effect, RunState


class EffectPolicy(Protocol):
    """Protocol for effect execution policies.
    
    Implementations control retry behavior and idempotency.
    """

    def max_attempts(self, effect: Effect) -> int:
        """Maximum number of attempts for an effect.
        
        Args:
            effect: The effect being executed.
            
        Returns:
            Maximum attempts (1 = no retries, 2 = one retry, etc.)
        """
        ...

    def backoff_seconds(self, *, effect: Effect, attempt: int) -> float:
        """Seconds to wait before retry.
        
        Args:
            effect: The effect being retried.
            attempt: Current attempt number (1-indexed).
            
        Returns:
            Seconds to wait before next attempt.
        """
        ...

    def idempotency_key(
        self, *, run: RunState, node_id: str, effect: Effect
    ) -> str:
        """Compute idempotency key for an effect.
        
        Effects with the same idempotency key are considered duplicates.
        If a prior completed result exists for this key, it will be reused.
        
        Args:
            run: Current run state.
            node_id: Current node ID.
            effect: The effect being executed.
            
        Returns:
            Idempotency key string.
        """
        ...


@dataclass
class DefaultEffectPolicy:
    """Default effect policy with configurable retry and idempotency.
    
    Attributes:
        default_max_attempts: Default max attempts for all effects.
        default_backoff_base: Base backoff in seconds (exponential).
        default_backoff_max: Maximum backoff in seconds.
        effect_max_attempts: Per-effect-type max attempts override.
    """

    default_max_attempts: int = 1  # No retries by default
    default_backoff_base: float = 1.0
    default_backoff_max: float = 60.0
    effect_max_attempts: Dict[str, int] = None  # type: ignore

    def __post_init__(self):
        if self.effect_max_attempts is None:
            self.effect_max_attempts = {}

    def max_attempts(self, effect: Effect) -> int:
        """Get max attempts for an effect type."""
        effect_type = effect.type.value
        return self.effect_max_attempts.get(effect_type, self.default_max_attempts)

    def backoff_seconds(self, *, effect: Effect, attempt: int) -> float:
        """Exponential backoff capped at max."""
        # Exponential: base * 2^(attempt-1), capped at max
        delay = self.default_backoff_base * (2 ** (attempt - 1))
        return min(delay, self.default_backoff_max)

    def idempotency_key(
        self, *, run: RunState, node_id: str, effect: Effect
    ) -> str:
        """Compute idempotency key from run_id, node_id, and effect.
        
        The key is a hash of:
        - run_id: Unique to this run
        - node_id: Current node
        - effect type and payload: What we're doing
        
        This ensures the same effect at the same point in the same run
        gets the same key, enabling deduplication on restart.
        """
        key_data = {
            "run_id": run.run_id,
            "node_id": node_id,
            "effect_type": effect.type.value,
            "effect_payload": effect.payload,
        }
        key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(key_json.encode()).hexdigest()[:32]


class RetryPolicy(DefaultEffectPolicy):
    """Policy with retries enabled for LLM and tool calls."""

    def __init__(
        self,
        *,
        llm_max_attempts: int = 3,
        tool_max_attempts: int = 2,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ):
        super().__init__(
            default_max_attempts=1,
            default_backoff_base=backoff_base,
            default_backoff_max=backoff_max,
            effect_max_attempts={
                "llm_call": llm_max_attempts,
                "tool_calls": tool_max_attempts,
            },
        )


class NoRetryPolicy(DefaultEffectPolicy):
    """Policy with no retries (fail immediately)."""

    def __init__(self):
        super().__init__(default_max_attempts=1)


def compute_idempotency_key(
    *, run_id: str, node_id: str, effect: Effect
) -> str:
    """Standalone function to compute idempotency key.
    
    Useful when you need to compute a key without a full policy.
    """
    key_data = {
        "run_id": run_id,
        "node_id": node_id,
        "effect_type": effect.type.value,
        "effect_payload": effect.payload,
    }
    key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key_json.encode()).hexdigest()[:32]
