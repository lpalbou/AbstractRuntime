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
from ...core.policy import RetryPolicy
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

def _compose_durable_ledger(ledger_store: LedgerStore, artifact_store: ArtifactStore) -> LedgerStore:
    """ONE composition site for the factory ledger stack (backlog 0067-M).

    Target shape for durable stores: Observable(Offloading(raw)) —
    Observable OUTERMOST so `Runtime.subscribe_ledger` is served here and
    live subscribers receive the record BEFORE offloading touches it;
    Offloading beneath so durable bytes stay bounded. In-memory ledgers or
    memory-only artifact stores gain nothing from refs and skip offloading.

    WHY explicit composition instead of chained helpers (2026-07-14
    adversary P0): `OffloadingLedgerStore` satisfies the runtime-checkable
    ObservableLedgerStoreProtocol by METHOD PRESENCE (its `subscribe`
    delegates and raises when the inner store can't), so a presence check
    after offload-wrapping concluded "already observable" and never added
    the real Observable — `subscribe_ledger` then raised RuntimeError on
    every durable deployment, silently killing host live-ledger features.
    A caller-supplied OffloadingLedgerStore is therefore wrapped, never
    trusted for observability; a caller-supplied genuine observable store
    is respected as-is (host owns its composition).
    """
    from ...storage.artifacts import InMemoryArtifactStore as _MemArtifacts
    from ...storage.in_memory import InMemoryLedgerStore as _MemLedger
    from ...storage.offloading import OffloadingLedgerStore as _Offloading

    if isinstance(ledger_store, _Offloading):
        return ObservableLedgerStore(ledger_store)
    if isinstance(ledger_store, ObservableLedgerStoreProtocol):
        return ledger_store
    if (
        isinstance(ledger_store, _MemLedger)
        or isinstance(artifact_store, _MemArtifacts)
    ):
        return ObservableLedgerStore(ledger_store)
    return ObservableLedgerStore(_Offloading(ledger_store, artifact_store=artifact_store))


def register_shell_session_teardown(runtime: Runtime) -> None:
    """Close a run's persistent shell sessions when the run reaches a terminal state.

    Backlog 0220: shell sessions are process-local resources namespaced by run id
    (`namespaced_session_id`); this hook guarantees an approved session never outlives
    the run it was approved for — including explicit cancel, which flows through the
    same terminal seam. Safe to call on any runtime (no-op when shell tools are unused
    or abstractcore is absent).
    """
    try:
        from abstractcore.tools.shell_session import get_shell_session_registry
    except Exception:  # pragma: no cover - abstractcore always present in this integration
        return

    def _close_run_sessions(run: Any) -> None:
        run_id = str(getattr(run, "run_id", "") or "").strip()
        if run_id:
            get_shell_session_registry().close_namespace(run_id)

    add = getattr(runtime, "add_terminal_hook", None)
    if callable(add):
        add(_close_run_sessions)


def _attach_runtime_abstractcore_client(runtime: Runtime, llm_client: Any) -> None:
    """Attach the internal client consumed by `get_abstractcore_host_facade(runtime)`.

    The attribute intentionally stays private so `Runtime` does not grow an
    AbstractCore-specific public API. Hosts should bind the public facade
    through `abstractruntime.integrations.abstractcore.get_abstractcore_host_facade(...)`.
    """

    try:
        setattr(runtime, "_abstractcore_llm_client", llm_client)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to attach the AbstractCore control client to the runtime. "
            "The public host facade would be unavailable for this runtime instance."
        ) from exc


def create_local_runtime(
    *,
    provider: str,
    model: str,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    bloc_root_dir: Optional[str | Path] = None,
    prompt_cache_export_root_dir: Optional[str | Path] = None,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    tool_executor: Optional[ToolExecutor] = None,
    tool_timeout_s: Optional[float] = None,
    context: Optional[Any] = None,
    effect_policy: Optional[Any] = None,
    config: Optional[RuntimeConfig] = None,
    artifact_store: Optional[ArtifactStore] = None,
    extra_effect_handlers: Optional[Dict[Any, Any]] = None,
    core_config_file: Optional[str | Path] = None,
    capability_defaults: Optional[Any] = None,
    steer_store: Optional[Any] = None,
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

    if artifact_store is None:
        artifact_store = InMemoryArtifactStore()
    ledger_store = _compose_durable_ledger(ledger_store, artifact_store)

    # Runtime authority: choose default timeouts for orchestrated workflows.
    #
    # Note: local providers may ignore timeouts (e.g. MLX) and should warn explicitly when doing so.
    # We still pass a timeout through the runtime for future-proofing and consistent orchestration
    # policy across providers.
    default_llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    default_tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
    try:
        from abstractcore.config.manager import get_config_manager  # type: ignore

        cfg_mgr = get_config_manager()
        default_llm_timeout_s = float(cfg_mgr.get_default_timeout())
        default_tool_timeout_s = float(cfg_mgr.get_tool_timeout())
    except Exception:
        pass

    resolved_tool_timeout_s = float(tool_timeout_s) if tool_timeout_s is not None else float(default_tool_timeout_s)

    effective_llm_kwargs: Dict[str, Any] = dict(llm_kwargs or {})
    effective_llm_kwargs.setdefault("timeout", float(default_llm_timeout_s))

    llm_client = MultiLocalAbstractCoreLLMClient(
        provider=provider,
        model=model,
        llm_kwargs=effective_llm_kwargs,
        artifact_store=artifact_store,
        bloc_root_dir=bloc_root_dir,
        prompt_cache_export_root_dir=prompt_cache_export_root_dir,
        core_config_file=core_config_file,
        capability_defaults=capability_defaults,
    )
    tools = tool_executor or AbstractCoreToolExecutor(timeout_s=resolved_tool_timeout_s)
    # Orchestrator policy: enforce tool execution timeout at the runtime layer.
    try:
        setter = getattr(tools, "set_timeout_s", None)
        if callable(setter):
            setter(resolved_tool_timeout_s)
    except Exception:
        pass
    handlers = build_effect_handlers(llm=llm_client, tools=tools, artifact_store=artifact_store, run_store=run_store)
    if extra_effect_handlers:
        handlers.update(dict(extra_effect_handlers))

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

    # Default to retries for transient LLM failures (backlog 0217). A single provider hiccup
    # (rate limit, connection reset) should not fail an entire durable run; an LLM_CALL is
    # side-effect-free, so re-issuing it is safe. TOOL_CALLS are NOT retried by default
    # (tool_max_attempts=1): a batch that raised mid-execution may have already applied a side
    # effect (write_file/execute_command), and the idempotency key only makes REPLAY of a
    # *completed* result safe, not re-execution of a partially-applied one. Callers can override
    # via an explicit `effect_policy`.
    effective_policy = (
        effect_policy
        if effect_policy is not None
        else RetryPolicy(llm_max_attempts=3, tool_max_attempts=1)
    )

    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=handlers,
        context=context,
        effect_policy=effective_policy,
        config=config,
        artifact_store=artifact_store,
        chat_summarizer=summarizer,
        steer_store=steer_store,
    )
    # Best-effort: expose the tool executor for approval-style TOOL_CALLS resumes.
    try:  # pragma: no cover
        setter = getattr(rt, "set_tool_executor_for_resume", None)
        if callable(setter):
            setter(tools)
    except Exception:
        pass
    register_shell_session_teardown(rt)
    _attach_runtime_abstractcore_client(rt, llm_client)
    return rt


def create_remote_runtime(
    *,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: Optional[float] = None,
    core_config_file: Optional[str | Path] = None,
    capability_defaults: Optional[Any] = None,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    tool_executor: Optional[ToolExecutor] = None,
    context: Optional[Any] = None,
    artifact_store: Optional[ArtifactStore] = None,
    effect_policy: Optional[Any] = None,
    steer_store: Optional[Any] = None,
) -> Runtime:
    if run_store is None or ledger_store is None:
        run_store, ledger_store = _default_in_memory_stores()

    if artifact_store is None:
        artifact_store = InMemoryArtifactStore()
    ledger_store = _compose_durable_ledger(ledger_store, artifact_store)

    resolved_timeout_s = float(timeout_s) if timeout_s is not None else float(DEFAULT_LLM_TIMEOUT_S)
    if timeout_s is None:
        try:
            from abstractcore.config.manager import get_config_manager  # type: ignore

            resolved_timeout_s = float(get_config_manager().get_default_timeout())
        except Exception:
            pass

    llm_client = RemoteAbstractCoreLLMClient(
        server_base_url=server_base_url,
        model=model,
        headers=headers,
        timeout_s=resolved_timeout_s,
        artifact_store=artifact_store,
        core_config_file=core_config_file,
        capability_defaults=capability_defaults,
    )
    tools = tool_executor or PassthroughToolExecutor()
    handlers = build_effect_handlers(llm=llm_client, tools=tools, artifact_store=artifact_store, run_store=run_store)
    # Match create_local_runtime: default to retries for transient LLM failures (backlog 0217).
    # Remote LLM calls are equally side-effect-free; tools are NOT retried (tool_max_attempts=1) for
    # the same partial-application reason. Previously the remote path silently had NO retries.
    effective_policy = (
        effect_policy
        if effect_policy is not None
        else RetryPolicy(llm_max_attempts=3, tool_max_attempts=1)
    )
    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=handlers,
        context=context,
        artifact_store=artifact_store,
        effect_policy=effective_policy,
        steer_store=steer_store,
    )
    try:  # pragma: no cover
        setter = getattr(rt, "set_tool_executor_for_resume", None)
        if callable(setter):
            setter(tools)
    except Exception:
        pass
    register_shell_session_teardown(rt)
    _attach_runtime_abstractcore_client(rt, llm_client)
    return rt


def create_hybrid_runtime(
    *,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: Optional[float] = None,
    tool_timeout_s: Optional[float] = None,
    core_config_file: Optional[str | Path] = None,
    capability_defaults: Optional[Any] = None,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    context: Optional[Any] = None,
    artifact_store: Optional[ArtifactStore] = None,
    steer_store: Optional[Any] = None,
) -> Runtime:
    """Remote LLM via AbstractCore server, local tool execution."""

    if run_store is None or ledger_store is None:
        run_store, ledger_store = _default_in_memory_stores()

    if artifact_store is None:
        artifact_store = InMemoryArtifactStore()
    ledger_store = _compose_durable_ledger(ledger_store, artifact_store)

    default_llm_timeout_s = float(DEFAULT_LLM_TIMEOUT_S)
    default_tool_timeout_s = float(DEFAULT_TOOL_TIMEOUT_S)
    if timeout_s is None or tool_timeout_s is None:
        try:
            from abstractcore.config.manager import get_config_manager  # type: ignore

            cfg_mgr = get_config_manager()
            default_llm_timeout_s = float(cfg_mgr.get_default_timeout())
            default_tool_timeout_s = float(cfg_mgr.get_tool_timeout())
        except Exception:
            pass

    resolved_timeout_s = float(timeout_s) if timeout_s is not None else float(default_llm_timeout_s)
    resolved_tool_timeout_s = float(tool_timeout_s) if tool_timeout_s is not None else float(default_tool_timeout_s)

    llm_client = RemoteAbstractCoreLLMClient(
        server_base_url=server_base_url,
        model=model,
        headers=headers,
        timeout_s=resolved_timeout_s,
        artifact_store=artifact_store,
        core_config_file=core_config_file,
        capability_defaults=capability_defaults,
    )
    tools = AbstractCoreToolExecutor(timeout_s=resolved_tool_timeout_s)
    try:
        setter = getattr(tools, "set_timeout_s", None)
        if callable(setter):
            setter(resolved_tool_timeout_s)
    except Exception:
        pass
    handlers = build_effect_handlers(llm=llm_client, tools=tools, artifact_store=artifact_store, run_store=run_store)

    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers=handlers,
        context=context,
        artifact_store=artifact_store,
        steer_store=steer_store,
    )
    try:  # pragma: no cover
        setter = getattr(rt, "set_tool_executor_for_resume", None)
        if callable(setter):
            setter(tools)
    except Exception:
        pass
    register_shell_session_teardown(rt)
    _attach_runtime_abstractcore_client(rt, llm_client)
    return rt


def create_local_file_runtime(
    *,
    base_dir: str | Path,
    provider: str,
    model: str,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    bloc_root_dir: Optional[str | Path] = None,
    prompt_cache_export_root_dir: Optional[str | Path] = None,
    context: Optional[Any] = None,
    config: Optional[RuntimeConfig] = None,
    tool_timeout_s: Optional[float] = None,
    steer_store: Optional[Any] = None,
) -> Runtime:
    run_store, ledger_store = _default_file_stores(base_dir=base_dir)
    artifact_store = FileArtifactStore(base_dir)
    return create_local_runtime(
        provider=provider,
        model=model,
        llm_kwargs=llm_kwargs,
        bloc_root_dir=Path(bloc_root_dir) if bloc_root_dir is not None else (Path(base_dir) / "blocs"),
        prompt_cache_export_root_dir=(
            Path(prompt_cache_export_root_dir)
            if prompt_cache_export_root_dir is not None
            else (Path(base_dir) / "prompt_cache_exports")
        ),
        run_store=run_store,
        ledger_store=ledger_store,
        context=context,
        config=config,
        artifact_store=artifact_store,
        tool_timeout_s=tool_timeout_s,
        steer_store=steer_store,
    )


def create_remote_file_runtime(
    *,
    base_dir: str | Path,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: Optional[float] = None,
    context: Optional[Any] = None,
    steer_store: Optional[Any] = None,
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
        steer_store=steer_store,
    )
