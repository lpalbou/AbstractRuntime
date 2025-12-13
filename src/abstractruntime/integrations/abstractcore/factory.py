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

from ...core.runtime import Runtime
from ...storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from ...storage.json_files import JsonFileRunStore, JsonlLedgerStore
from ...storage.base import LedgerStore, RunStore

from .effect_handlers import build_effect_handlers
from .llm_client import LocalAbstractCoreLLMClient, RemoteAbstractCoreLLMClient
from .tool_executor import AbstractCoreToolExecutor, PassthroughToolExecutor, ToolExecutor


def _default_in_memory_stores() -> tuple[RunStore, LedgerStore]:
    return InMemoryRunStore(), InMemoryLedgerStore()


def _default_file_stores(*, base_dir: str | Path) -> tuple[RunStore, LedgerStore]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return JsonFileRunStore(base), JsonlLedgerStore(base)


def create_local_runtime(
    *,
    provider: str,
    model: str,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    tool_executor: Optional[ToolExecutor] = None,
    context: Optional[Any] = None,
    effect_policy: Optional[Any] = None,
) -> Runtime:
    if run_store is None or ledger_store is None:
        run_store, ledger_store = _default_in_memory_stores()

    llm_client = LocalAbstractCoreLLMClient(provider=provider, model=model, llm_kwargs=llm_kwargs)
    tools = tool_executor or AbstractCoreToolExecutor()
    handlers = build_effect_handlers(llm=llm_client, tools=tools)

    return Runtime(run_store=run_store, ledger_store=ledger_store, effect_handlers=handlers, context=context, effect_policy=effect_policy)


def create_remote_runtime(
    *,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 60.0,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    tool_executor: Optional[ToolExecutor] = None,
    context: Optional[Any] = None,
) -> Runtime:
    if run_store is None or ledger_store is None:
        run_store, ledger_store = _default_in_memory_stores()

    llm_client = RemoteAbstractCoreLLMClient(
        server_base_url=server_base_url,
        model=model,
        headers=headers,
        timeout_s=timeout_s,
    )
    tools = tool_executor or PassthroughToolExecutor()
    handlers = build_effect_handlers(llm=llm_client, tools=tools)

    return Runtime(run_store=run_store, ledger_store=ledger_store, effect_handlers=handlers, context=context)


def create_hybrid_runtime(
    *,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 60.0,
    run_store: Optional[RunStore] = None,
    ledger_store: Optional[LedgerStore] = None,
    context: Optional[Any] = None,
) -> Runtime:
    """Remote LLM via AbstractCore server, local tool execution."""

    if run_store is None or ledger_store is None:
        run_store, ledger_store = _default_in_memory_stores()

    llm_client = RemoteAbstractCoreLLMClient(
        server_base_url=server_base_url,
        model=model,
        headers=headers,
        timeout_s=timeout_s,
    )
    tools = AbstractCoreToolExecutor()
    handlers = build_effect_handlers(llm=llm_client, tools=tools)

    return Runtime(run_store=run_store, ledger_store=ledger_store, effect_handlers=handlers, context=context)


def create_local_file_runtime(
    *,
    base_dir: str | Path,
    provider: str,
    model: str,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    context: Optional[Any] = None,
) -> Runtime:
    run_store, ledger_store = _default_file_stores(base_dir=base_dir)
    return create_local_runtime(
        provider=provider,
        model=model,
        llm_kwargs=llm_kwargs,
        run_store=run_store,
        ledger_store=ledger_store,
        context=context,
    )


def create_remote_file_runtime(
    *,
    base_dir: str | Path,
    server_base_url: str,
    model: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 60.0,
    context: Optional[Any] = None,
) -> Runtime:
    run_store, ledger_store = _default_file_stores(base_dir=base_dir)
    return create_remote_runtime(
        server_base_url=server_base_url,
        model=model,
        headers=headers,
        timeout_s=timeout_s,
        run_store=run_store,
        ledger_store=ledger_store,
        context=context,
    )

