"""abstractruntime.integrations.abstractcore

AbstractCore integration package.

Provides:
- LLM clients (local + remote)
- Tool executors (executed + passthrough)
- Effect handlers wiring
- Convenience runtime factories for local/remote/hybrid modes

Importing this module is the explicit opt-in to an AbstractCore dependency.
"""

from .llm_client import (
    AbstractCoreLLMClient,
    LocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
)
from .tool_executor import AbstractCoreToolExecutor, PassthroughToolExecutor, ToolExecutor
from .effect_handlers import build_effect_handlers
from .factory import (
    create_hybrid_runtime,
    create_local_file_runtime,
    create_local_runtime,
    create_remote_file_runtime,
    create_remote_runtime,
)

__all__ = [
    "AbstractCoreLLMClient",
    "LocalAbstractCoreLLMClient",
    "RemoteAbstractCoreLLMClient",
    "ToolExecutor",
    "AbstractCoreToolExecutor",
    "PassthroughToolExecutor",
    "build_effect_handlers",
    "create_local_runtime",
    "create_remote_runtime",
    "create_hybrid_runtime",
    "create_local_file_runtime",
    "create_remote_file_runtime",
]

