"""abstractruntime.integrations.abstractcore

AbstractCore integration package.

Provides:
- LLM clients (local + remote)
- Tool executors (executed + passthrough)
- Effect handlers wiring
- Convenience runtime factories for local/remote/hybrid modes
- RuntimeConfig for limits and model capabilities

Importing this module is the explicit opt-in to an AbstractCore dependency.
"""

from ...core.config import RuntimeConfig
from .llm_client import (
    AbstractCoreLLMClient,
    LocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
)
from .tool_executor import AbstractCoreToolExecutor, MappingToolExecutor, PassthroughToolExecutor, ToolExecutor
from .effect_handlers import build_effect_handlers
from .factory import (
    create_hybrid_runtime,
    create_local_file_runtime,
    create_local_runtime,
    create_remote_file_runtime,
    create_remote_runtime,
)
from .observability import attach_global_event_bus_bridge, emit_step_record

__all__ = [
    "AbstractCoreLLMClient",
    "LocalAbstractCoreLLMClient",
    "RemoteAbstractCoreLLMClient",
    "RuntimeConfig",
    "ToolExecutor",
    "MappingToolExecutor",
    "AbstractCoreToolExecutor",
    "PassthroughToolExecutor",

    "build_effect_handlers",
    "create_local_runtime",
    "create_remote_runtime",
    "create_hybrid_runtime",
    "create_local_file_runtime",
    "create_remote_file_runtime",
    "attach_global_event_bus_bridge",
    "emit_step_record",
]
