"""abstractruntime.integrations.abstractcore

AbstractCore integration package.

Provides:
- LLM clients (local + remote)
- Tool executors (executed + passthrough)
- Effect handlers wiring
- Convenience runtime factories for local/remote/hybrid modes
- Public discovery facade for provider/media/catalog snapshot queries
- Public host facade for prompt-cache, durable bloc/KV, and model-residency control operations
- Public durable run facade for run-scoped AbstractCore child runs
- RuntimeConfig for limits and model capabilities

Importing this module is the explicit opt-in to an AbstractCore dependency.
"""

from ...core.config import RuntimeConfig
from .llm_client import (
    AbstractCoreLLMClient,
    AbstractCoreControlClient,
    LocalAbstractCoreLLMClient,
    MultiLocalAbstractCoreLLMClient,
    RemoteAbstractCoreLLMClient,
)
from .embeddings_client import AbstractCoreEmbeddingsClient, EmbeddingsResult
from .host_facade import (
    AbstractCoreHostFacade,
    get_abstractcore_host_facade,
)
from .discovery_facade import (
    AbstractCoreDiscoveryFacade,
    get_abstractcore_discovery_facade,
)
from .run_facade import (
    AbstractCoreRunFacade,
    get_abstractcore_run_facade,
)
from .tool_executor import (
    AbstractCoreToolExecutor,
    ApprovalToolExecutor,
    MappingToolExecutor,
    PassthroughToolExecutor,
    ToolApprovalPolicy,
    ToolExecutor,
)
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
    "AbstractCoreControlClient",
    "AbstractCoreDiscoveryFacade",
    "AbstractCoreHostFacade",
    "AbstractCoreRunFacade",
    "LocalAbstractCoreLLMClient",
    "MultiLocalAbstractCoreLLMClient",
    "RemoteAbstractCoreLLMClient",
    "AbstractCoreEmbeddingsClient",
    "EmbeddingsResult",
    "RuntimeConfig",
    "ToolExecutor",
    "MappingToolExecutor",
    "AbstractCoreToolExecutor",
    "PassthroughToolExecutor",
    "ToolApprovalPolicy",
    "ApprovalToolExecutor",

    "build_effect_handlers",
    "get_abstractcore_discovery_facade",
    "get_abstractcore_host_facade",
    "get_abstractcore_run_facade",
    "create_local_runtime",
    "create_remote_runtime",
    "create_hybrid_runtime",
    "create_local_file_runtime",
    "create_remote_file_runtime",
    "attach_global_event_bus_bridge",
    "emit_step_record",
]
