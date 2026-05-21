"""Public host-facing AbstractCore control facade for Runtime integrations.

Hosts should import this module instead of reaching through
`runtime._abstractcore_llm_client` directly.

Scope:
- prompt-cache control operations
- host-local prompt-cache export/import admin operations
- durable bloc/KV prompt-cache operations
- durable bloc/KV lifecycle operations
- model-residency control operations
- host-local email/comms helper operations

Non-goals:
- durable Runtime effect execution (use `AbstractCoreRunFacade` / `get_abstractcore_run_facade(...)`)
- generated media / TTS / STT helpers
- provider-private prompt-cache export/import behavior
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

_RUNTIME_ABSTRACTCORE_CLIENT_ATTR = "_abstractcore_llm_client"
_HOST_CONTROL_METHODS = (
    "get_prompt_cache_capabilities",
    "get_prompt_cache_stats",
    "prompt_cache_set",
    "prompt_cache_update",
    "prompt_cache_fork",
    "prompt_cache_clear",
    "prompt_cache_prepare_modules",
    "list_prompt_cache_exports",
    "prompt_cache_export",
    "prompt_cache_import",
    "upsert_text_bloc",
    "get_bloc_record",
    "list_blocs",
    "get_bloc_kv_manifest",
    "ensure_bloc_kv_artifact",
    "load_bloc_kv_artifact",
    "list_bloc_kv_artifacts",
    "delete_bloc_kv_artifact",
    "prune_bloc_kv_artifacts",
    "delete_bloc",
    "list_model_residency",
    "load_model_residency",
    "unload_model_residency",
)


class AbstractCoreHostControlClient(Protocol):
    """Duck-typed control client contract implemented by AbstractCore LLM clients."""

    def get_prompt_cache_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        ...

    def get_prompt_cache_stats(self, **kwargs: Any) -> Dict[str, Any]:
        ...

    def prompt_cache_set(
        self,
        *,
        key: str,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_fork(
        self,
        *,
        from_key: str,
        to_key: str,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_clear(
        self,
        *,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_prompt_cache_exports(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_export(
        self,
        *,
        name: str,
        key: str,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prompt_cache_import(
        self,
        *,
        name: str,
        key: Optional[str] = None,
        make_default: bool = True,
        clear_existing: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def upsert_text_bloc(
        self,
        *,
        path: str,
        content: str,
        sha256: Optional[str] = None,
        content_sha256: Optional[str] = None,
        media_type: str = "text",
        size_bytes: Optional[int] = None,
        mtime_ns: Optional[int] = None,
        format: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        relpath_base: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def get_bloc_record(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_blocs(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def get_bloc_kv_manifest(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def ensure_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def load_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        stable_cache_key: Optional[str] = None,
        key: Optional[str] = None,
        make_default: bool = False,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def delete_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def prune_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def delete_bloc(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        delete_kv: bool = True,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def load_model_residency(
        self,
        *,
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    def unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


def _coerce_control_client(client: Any, *, source: str) -> AbstractCoreHostControlClient:
    missing = [name for name in _HOST_CONTROL_METHODS if not callable(getattr(client, name, None))]
    if missing:
        methods = ", ".join(missing)
        raise TypeError(
            f"{source} does not implement the AbstractCore host control contract. Missing methods: {methods}."
        )
    return client


def _client_from_runtime(runtime: Any) -> AbstractCoreHostControlClient:
    client = getattr(runtime, _RUNTIME_ABSTRACTCORE_CLIENT_ATTR, None)
    if client is None:
        raise RuntimeError(
            "Runtime is not wired to AbstractCore host controls. "
            "Use an AbstractCore runtime factory, then obtain the facade via "
            "`abstractruntime.integrations.abstractcore.get_abstractcore_host_facade(runtime)`."
        )
    return _coerce_control_client(client, source="Runtime AbstractCore client")


class AbstractCoreHostFacade:
    """Public host-facing control surface for AbstractCore-backed runtimes.

    This facade is intentionally narrow and operator-scoped. Its methods are
    synchronous helpers and do not create durable Runtime history on their own.
    Most methods delegate through the configured Runtime client; the small
    comms/email helpers intentionally remain host-local wrappers over
    AbstractCore's public tool modules in phase 1.
    """

    __slots__ = ("_client",)

    def __init__(self, runtime: Any):
        self._client = _client_from_runtime(runtime)

    @classmethod
    def from_runtime(cls, runtime: Any) -> "AbstractCoreHostFacade":
        """Bind the public host facade to an AbstractCore-wired runtime."""

        return cls(runtime)

    def get_prompt_cache_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        return self._client.get_prompt_cache_capabilities(**kwargs)

    def get_prompt_cache_stats(self, **kwargs: Any) -> Dict[str, Any]:
        return self._client.get_prompt_cache_stats(**kwargs)

    def prompt_cache_set(
        self,
        *,
        key: str,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_set(
            key=key,
            make_default=make_default,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_update(
        self,
        *,
        key: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_update(
            key=key,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_fork(
        self,
        *,
        from_key: str,
        to_key: str,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_fork(
            from_key=from_key,
            to_key=to_key,
            make_default=make_default,
            ttl_s=ttl_s,
            **kwargs,
        )

    def prompt_cache_clear(
        self,
        *,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_clear(key=key, **kwargs)

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_prepare_modules(
            namespace=namespace,
            modules=modules,
            make_default=make_default,
            ttl_s=ttl_s,
            version=version,
            **kwargs,
        )

    def list_prompt_cache_exports(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_prompt_cache_exports(provider=provider, model=model, **kwargs)

    def prompt_cache_export(
        self,
        *,
        name: str,
        key: str,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_export(
            name=name,
            key=key,
            q8=q8,
            meta=meta,
            provider=provider,
            model=model,
            **kwargs,
        )

    def prompt_cache_import(
        self,
        *,
        name: str,
        key: Optional[str] = None,
        make_default: bool = True,
        clear_existing: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prompt_cache_import(
            name=name,
            key=key,
            make_default=make_default,
            clear_existing=clear_existing,
            provider=provider,
            model=model,
            **kwargs,
        )

    def upsert_text_bloc(
        self,
        *,
        path: str,
        content: str,
        sha256: Optional[str] = None,
        content_sha256: Optional[str] = None,
        media_type: str = "text",
        size_bytes: Optional[int] = None,
        mtime_ns: Optional[int] = None,
        format: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        relpath_base: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.upsert_text_bloc(
            path=path,
            content=content,
            sha256=sha256,
            content_sha256=content_sha256,
            media_type=media_type,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            format=format,
            estimated_tokens=estimated_tokens,
            relpath_base=relpath_base,
            summary=summary,
            keywords=keywords,
            **kwargs,
        )

    def get_bloc_record(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.get_bloc_record(sha256=sha256, bloc_id=bloc_id, **kwargs)

    def list_blocs(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_blocs(sha256=sha256, bloc_id=bloc_id, **kwargs)

    def get_bloc_kv_manifest(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.get_bloc_kv_manifest(
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            **kwargs,
        )

    def ensure_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.ensure_bloc_kv_artifact(
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            force_rebuild=force_rebuild,
            debug=debug,
            **kwargs,
        )

    def load_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        stable_cache_key: Optional[str] = None,
        key: Optional[str] = None,
        make_default: bool = False,
        force_rebuild: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.load_bloc_kv_artifact(
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            stable_cache_key=stable_cache_key,
            key=key,
            make_default=make_default,
            force_rebuild=force_rebuild,
            debug=debug,
            **kwargs,
        )

    def list_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_bloc_kv_artifacts(
            sha256=sha256,
            bloc_id=bloc_id,
            provider=provider,
            model=model,
            **kwargs,
        )

    def delete_bloc_kv_artifact(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.delete_bloc_kv_artifact(
            sha256=sha256,
            bloc_id=bloc_id,
            artifact_path=artifact_path,
            provider=provider,
            model=model,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
            debug=debug,
            **kwargs,
        )

    def prune_bloc_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.prune_bloc_kv_artifacts(
            sha256=sha256,
            bloc_id=bloc_id,
            provider=provider,
            model=model,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
            debug=debug,
            **kwargs,
        )

    def delete_bloc(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        delete_kv: bool = True,
        clear_loaded: bool = False,
        force: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.delete_bloc(
            sha256=sha256,
            bloc_id=bloc_id,
            delete_kv=delete_kv,
            clear_loaded=clear_loaded,
            force=force,
            dry_run=dry_run,
            **kwargs,
        )

    def list_model_residency(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.list_model_residency(
            task=task,
            provider=provider,
            model=model,
            **kwargs,
        )

    def load_model_residency(
        self,
        *,
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        pin: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.load_model_residency(
            task=task,
            provider=provider,
            model=model,
            options=options,
            pin=pin,
            **kwargs,
        )

    def unload_model_residency(
        self,
        *,
        task: Optional[str] = None,
        runtime_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._client.unload_model_residency(
            task=task,
            runtime_id=runtime_id,
            provider=provider,
            model=model,
            options=options,
            **kwargs,
        )

    def list_email_accounts(self) -> Dict[str, Any]:
        from .comms_facade import list_email_accounts

        return list_email_accounts()

    def list_emails(
        self,
        *,
        account: Optional[str] = None,
        mailbox: Optional[str] = None,
        since: Optional[str] = None,
        status: str = "all",
        limit: int = 20,
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        from .comms_facade import list_emails

        return list_emails(
            account=account,
            mailbox=mailbox,
            since=since,
            status=status,
            limit=limit,
            timeout_s=timeout_s,
        )

    def read_email(
        self,
        *,
        uid: str,
        account: Optional[str] = None,
        mailbox: Optional[str] = None,
        timeout_s: float = 30.0,
        max_body_chars: int = 20000,
    ) -> Dict[str, Any]:
        from .comms_facade import read_email

        return read_email(
            uid=uid,
            account=account,
            mailbox=mailbox,
            timeout_s=timeout_s,
            max_body_chars=max_body_chars,
        )

    def send_email(
        self,
        to: Any,
        subject: str,
        *,
        account: Optional[str] = None,
        body_text: Optional[str] = None,
        body_html: Optional[str] = None,
        cc: Any = None,
        bcc: Any = None,
        timeout_s: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Host-local operator send helper.

        If the outbound send belongs to a workflow/run, use
        `get_abstractcore_run_facade(runtime).send_email(...)` instead so
        Runtime authors the durable child-run truth.
        """
        from .comms_facade import send_email

        return send_email(
            to=to,
            subject=subject,
            account=account,
            body_text=body_text,
            body_html=body_html,
            cc=cc,
            bcc=bcc,
            timeout_s=timeout_s,
            headers=headers,
        )


def get_abstractcore_host_facade(runtime: Any) -> AbstractCoreHostFacade:
    """Return the public AbstractCore host facade bound to a runtime."""

    return AbstractCoreHostFacade.from_runtime(runtime)


__all__ = [
    "AbstractCoreHostFacade",
    "get_abstractcore_host_facade",
]
