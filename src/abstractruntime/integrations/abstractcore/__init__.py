"""abstractruntime.integrations.abstractcore

AbstractCore integration package.

Provides:
- LLM clients (local + remote)
- Tool executors (executed + passthrough)
- Effect handlers wiring
- Convenience runtime factories for local/remote/hybrid modes
- Public discovery facade for provider/media/catalog snapshot queries
- Public host facade for prompt-cache, durable bloc/KV, and model-residency control operations
- Public config facade for AbstractCore capability-default routes and config API keys
- Public structured-output facade (`structured_facade`): `parse_response_format` / `ResponseFormatError` for hosts
- Public email facade (`email_facade`): AbstractCore's mail library re-exported for hosts
- Public Telegram host wrappers for TDLib bootstrap/global-client/send parity
- Public durable run facade for run-scoped AbstractCore LLM/tool child runs, including outbound comms sends
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
from .config_facade import (
    capability_default_config_file,
    capability_default_specs,
    clear_capability_default,
    list_capability_defaults,
    read_config_api_key,
    set_capability_default,
)
from .structured_facade import parse_response_format
from .discovery_facade import (
    AbstractCoreDiscoveryFacade,
    get_abstractcore_discovery_facade,
)
from .run_facade import (
    AbstractCoreRunFacade,
    get_abstractcore_run_facade,
)
from .telegram_facade import (
    TelegramTdlibNotAvailable,
    bootstrap_telegram_auth_from_env,
    get_global_telegram_client,
    send_telegram_message,
    stop_global_telegram_client,
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
    "TelegramTdlibNotAvailable",
    "RuntimeConfig",
    "ToolExecutor",
    "MappingToolExecutor",
    "AbstractCoreToolExecutor",
    "PassthroughToolExecutor",
    "ToolApprovalPolicy",
    "ApprovalToolExecutor",

    "build_effect_handlers",
    "bootstrap_telegram_auth_from_env",
    "get_abstractcore_discovery_facade",
    "get_abstractcore_host_facade",
    "capability_default_config_file",
    "capability_default_specs",
    "clear_capability_default",
    "list_capability_defaults",
    "read_config_api_key",
    "set_capability_default",
    "parse_response_format",
    "ResponseFormatError",
    "get_global_telegram_client",
    "get_abstractcore_run_facade",
    "create_local_runtime",
    "create_remote_runtime",
    "create_hybrid_runtime",
    "create_local_file_runtime",
    "create_remote_file_runtime",
    "send_telegram_message",
    "stop_global_telegram_client",
    "attach_global_event_bus_bridge",
    "emit_step_record",
]


def __getattr__(name: str):
    # AbstractCore's ResponseFormatError, resolved on first use: the structured
    # module needs AbstractCore >= 2.25.0, which importing this package must not.
    if name == "ResponseFormatError":
        from . import structured_facade

        return structured_facade.ResponseFormatError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
