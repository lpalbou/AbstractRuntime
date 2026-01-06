"""abstractruntime.integrations.abstractcore.factory

Convenience constructors for a Runtime wired to AbstractCore.

These helpers implement the three supported execution modes:
- local: in-process LLM + local tool execution
- remote: HTTP to AbstractCore server + tool passthrough
- hybrid: HTTP to AbstractCore server + local tool execution

The caller supplies storage backends (in-memory or file-based).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ...core.config import RuntimeConfig
from ...core.runtime import Runtime
from ...storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from ...storage.json_files import JsonFileRunStore, JsonlLedgerStore
from ...storage.base import LedgerStore, RunStore
from ...storage.artifacts import FileArtifactStore, InMemoryArtifactStore, ArtifactStore
from ...storage.observable import ObservableLedgerStore, ObservableLedgerStoreProtocol

from .effect_handlers import build_effect_handlers
from .llm_client import MultiLocalAbstractCoreLLMClient, RemoteAbstractCoreLLMClient
from .tool_executor import AbstractCoreToolExecutor, PassthroughToolExecutor, ToolExecutor
from .summarizer import AbstractCoreChatSummarizer
from .constants import DEFAULT_LLM_TIMEOUT_S, DEFAULT_TOOL_TIMEOUT_S


def _default_in_memory_stores() -> tuple[RunStore, LedgerStore]:
    return InMemoryRunStore(), InMemoryLedgerStore()


def _default_file_stores(*, base_dir: str | Path) -> tuple[RunStore, LedgerStore]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return JsonFileRunStore(base), JsonlLedgerStore(base)

def _ensure_observable_ledger(ledger_store: LedgerStore) -> LedgerStore:
    """Wrap a LedgerStore so Runtime.subscribe_ledger() is available (in-process).

    Why:
    - Real-time UI/UX often needs "step started" signals *before* a blocking effect
      (LLM/tool HTTP) returns.
    - The runtime kernel stays transport-agnostic; this is an optional decorator.
    """
    if isinstance(ledger_store, ObservableLedgerStoreProtocol):
        return ledger_store
    return ObservableLedgerStore(ledger_store)


def create_local_runtime(
    *,
    provider: str,
    model: str,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    tool_executor: Optional[ToolExecutor] = None,
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    context: Optional[Any] = None,
    effect_policy: Optional[Any] = None,
    config: Optional[RuntimeConfig] = None,
    artifact_store: Optional[ArtifactStore] = None,
) -> Runtime:
    """Create a runtime with local LLM execution via AbstractCore.

    Args:
        provider: LLM provider (e.g., "ollama", "openai")
        model: Model name
        llm_kwargs: Additional kwargs for LLM client
        run_store: Storage for run state (default: in-memory)
        ledger_store: Storage for ledger (default: in-memory)
        tool_executor: Optional custom tool executor. If not provided, defaults
            to `AbstractCoreToolExecutor()` (AbstractCore global tool registry).
        context: Optional context object
        effect_policy: Optional effect policy (retry, etc.)
        config: Optional RuntimeConfig for limits and model capabilities.
            If not provided, model capabilities are queried from the LLM client.

    Note:
        For durable execution, tool callables should never be stored in `RunState.vars`
        or passed in effect payloads. Prefer `MappingToolExecutor.from_tools([...])`.
    """
    if run_store is None or ledger_store is None:
        run_store, ledger_store = _default_in_memory_stores()
    ledger_store = _ensure_observable_ledger(ledger_store)

    if artifact_store is None:
        artifact_store = InMemoryArtifactStore()

    # Runtime authority: default LLM timeout for orchestrated workflows.
    #
    # We set this here (in the runtime layer) rather than relying on AbstractCore global config,
    # so workflow behavior is consistent and controlled by the orchestrator.
    effective_llm_kwargs: Dict[str, Any] = dict(llm_kwargs or {})
    effective_llm_kwargs.setdefault("timeout", DEFAULT_LLM_TIMEOUT_S)

    llm_client = MultiLocalAbstractCoreLLMClient(provider=provider, model=model, llm_kwargs=effective_llm_kwargs)
    tools = tool_executor or AbstractCoreToolExecutor(timeout_s=tool_timeout_s)
    # Orchestrator policy: enforce tool execution timeout at the runtime layer.
    try:
        setter = getattr(tools, "set_timeout_s", None)
        if callable(setter):
            setter(tool_timeout_s)
    except Exception:
        pass
    handlers = build_effect_handlers(llm=llm_client, tools=tools)

    # Query model capabilities and merge into config
    capabilities = llm_client.get_model_capabilities()
    if config is None:
        config = RuntimeConfig(
            provider=str(provider).strip() if isinstance(provider, str) and str(provider).strip() else None,
            model=str(model).strip() if isinstance(model, str) and str(model).strip() else None,
            model_capabilities=capabilities,
        )
    else:
        # Merge capabilities into provided config
        config = config.with_capabilities(capabilities)

    # Create chat summarizer with token limits from config
    # This enables adaptive chunking during MEMORY_COMPACT
    summarizer = AbstractCoreChatSummarizer(
        llm=llm_client._llm,  # Use the underlying AbstractCore LLM instance
        max_tokens=config.max_tokens if config.max_tokens is not None else -1,
        max_output_tokens=config.max_output_tokens if config.max_output_tokens is not None else -1,
    )

    return Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=handlers,
        context=context,
        effect_policy=effect_policy,
        config=config,
        artifact_store=artifact_store,
        chat_summarizer=summarizer,
    )


def create_remote_runtime(
    *,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    tool_executor: Optional[ToolExecutor] = None,
    context: Optional[Any] = None,
    artifact_store: Optional[ArtifactStore] = None,
) -> Runtime:
    if run_store is None or ledger_store is None:
        run_store, ledger_store = _default_in_memory_stores()
    ledger_store = _ensure_observable_ledger(ledger_store)

    if artifact_store is None:
        artifact_store = InMemoryArtifactStore()

    llm_client = RemoteAbstractCoreLLMClient(
        server_base_url=server_base_url,
        model=model,
        headers=headers,
        timeout_s=timeout_s,
    )
    tools = tool_executor or PassthroughToolExecutor()
    handlers = build_effect_handlers(llm=llm_client, tools=tools)

    return Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=handlers,
        context=context,
        artifact_store=artifact_store,
    )


def create_hybrid_runtime(
    *,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    context: Optional[Any] = None,
    artifact_store: Optional[ArtifactStore] = None,
) -> Runtime:
    """Remote LLM via AbstractCore server, local tool execution."""

    if run_store is None or ledger_store is None:
        run_store, ledger_store = _default_in_memory_stores()
    ledger_store = _ensure_observable_ledger(ledger_store)

    if artifact_store is None:
        artifact_store = InMemoryArtifactStore()

    llm_client = RemoteAbstractCoreLLMClient(
        server_base_url=server_base_url,
        model=model,
        headers=headers,
        timeout_s=timeout_s,
    )
    tools = AbstractCoreToolExecutor(timeout_s=tool_timeout_s)
    try:
        setter = getattr(tools, "set_timeout_s", None)
        if callable(setter):
            setter(tool_timeout_s)
    except Exception:
        pass
    handlers = build_effect_handlers(llm=llm_client, tools=tools)

    return Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=handlers,
        context=context,
        artifact_store=artifact_store,
    )


def create_local_file_runtime(
    *,
    base_dir: str | Path,
    provider: str,
    model: str,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    context: Optional[Any] = None,
    config: Optional[RuntimeConfig] = None,
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
) -> Runtime:
    run_store, ledger_store = _default_file_stores(base_dir=base_dir)
    artifact_store = FileArtifactStore(base_dir)
    return create_local_runtime(
        provider=provider,
        model=model,
        llm_kwargs=llm_kwargs,
        run_store=run_store,
        ledger_store=ledger_store,
        context=context,
        config=config,
        artifact_store=artifact_store,
        tool_timeout_s=tool_timeout_s,
    )


def create_remote_file_runtime(
    *,
    base_dir: str | Path,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    context: Optional[Any] = None,
) -> Runtime:
    run_store, ledger_store = _default_file_stores(base_dir=base_dir)
    artifact_store = FileArtifactStore(base_dir)
    return create_remote_runtime(
        server_base_url=server_base_url,
        model=model,
        headers=headers,
        timeout_s=timeout_s,
        run_store=run_store,
        ledger_store=ledger_store,
        context=context,
        artifact_store=artifact_store,
    )
