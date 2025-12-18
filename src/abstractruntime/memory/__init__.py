"""abstractruntime.memory

Runtime-owned memory utilities.

The runtime stores *everything* durably (RunStore/LedgerStore/ArtifactStore), but
only a selected view is sent to the LLM as **active context**:

    RunState.vars["context"]["messages"]

This package provides minimal, JSON-safe helpers to:
- list and filter archived spans (metadata/time range)
- rehydrate archived spans back into active context deterministically
- derive the LLM-visible view from active context under simple limits

Semantic retrieval and graph-level memory live in AbstractMemory/AbstractFlow.
"""

from .active_context import ActiveContextPolicy, TimeRange

__all__ = ["ActiveContextPolicy", "TimeRange"]

