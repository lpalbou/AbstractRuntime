"""Runtime-owned AbstractCore configuration facade for hosts.

Hosts (e.g. AbstractGateway) should import this module instead of reaching into
`abstractcore.config.manager` / `abstractcore.config.capability_defaults`
directly. This keeps the host -> Runtime -> AbstractCore boundary clean: the
host never imports `abstractcore` itself.

Scope:
- read/write/clear AbstractCore capability-default routes, either against the
  process-default AbstractCore config or an explicit `config_file` path (hosts
  use scoped per-principal config files)
- resolve the effective AbstractCore config file path
- enumerate capability-default route specs (the catalog of known routes)
- read a provider API key from a specific AbstractCore config file
- read AbstractCore's stored mail (IMAP/SMTP) and maintenance-triage settings

Non-goals:
- durable Runtime effect execution (use `AbstractCoreRunFacade`)
- provider-connection *policy* / discovery (kept in the host)

Design mirrors `comms_facade.py`: free functions that lazily import AbstractCore
so importing this module stays light and the dependency error is actionable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_PathLike = Union[str, Path]


def _configuration_manager(
    config_file: Optional[_PathLike] = None,
    *,
    apply_env: bool = True,
):
    """Return a ConfigurationManager bound to `config_file` (or the default)."""
    try:
        from abstractcore.config.manager import ConfigurationManager
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "AbstractCore configuration support is unavailable. Install a Runtime "
            "environment with AbstractCore (the `integrations/abstractcore` opt-in)."
        ) from exc
    if config_file is not None:
        return ConfigurationManager(config_file=Path(config_file), apply_env=apply_env)
    return ConfigurationManager(apply_env=apply_env)


def list_capability_defaults(
    *,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> List[Dict[str, Any]]:
    """Return the capability-default routes for the given (or default) config."""
    return list(_configuration_manager(config_file, apply_env=apply_env).list_capability_defaults())


def capability_default_config_file(
    *,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> str:
    """Return the resolved config-file path a ConfigurationManager would use."""
    return str(_configuration_manager(config_file, apply_env=apply_env).config_file)


def capability_default_config_path(
    *,
    config_file: Optional[_PathLike] = None,
) -> str:
    """WHERE the AbstractCore config lives, without loading it.

    Same answer as `capability_default_config_file`, at `stat` cost instead of
    a full load: a host that only wants to know whether the store CHANGED must
    not re-read and re-parse it to find out where it is (measured 133us for the
    loading form against 1.7us for the stat it guards). Use the loading form
    only when the manager's env side effects are actually wanted.
    """
    try:
        from abstractcore.config.manager import resolve_config_file
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "AbstractCore configuration support is unavailable. Install a Runtime "
            "environment with AbstractCore (the `integrations/abstractcore` opt-in)."
        ) from exc
    return str(resolve_config_file(None, config_file))


def capability_defaults_seed_marker(
    *,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> Optional[str]:
    """The provenance marker of the fresh-install seed, or ``None``.

    AbstractCore seeds its recommended capability routes only when no config
    file has EVER existed there (operator ruling 2026-08-01) and stamps
    `capability_defaults.seeded` (`"recommended-v1"`) on the result. The routes
    themselves are ordinary rows -- overridable, clearable, always beaten by a
    request pin -- so the marker changes no behaviour; it exists so a surface
    can say "recommended default" instead of implying an operator chose the
    value. Hosts read it through here rather than importing AbstractCore.
    """
    try:
        return _configuration_manager(config_file, apply_env=apply_env).config.capability_defaults.seeded or None
    except Exception:
        return None


def set_capability_default(
    kind: str,
    modality: Optional[str] = None,
    *,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    reasoning: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> bool:
    """Persist one capability-default route. Returns True on success.

    A FAILURE PROPAGATES ITS REASON. AbstractCore raises
    `CapabilityDefaultWriteError` (a `ValueError`) naming the route and the
    underlying cause; the facade deliberately does not catch it, so the
    Gateway's 400 and the CLI's error line both say WHY instead of the
    reason-free "Failed to set capability default <route>" they printed
    before 2026-08-01.
    """
    return bool(
        _configuration_manager(config_file, apply_env=apply_env).set_capability_default(
            kind,
            modality,
            task=task,
            provider=provider,
            model=model,
            base_url=base_url,
            reasoning=reasoning,
            options=options,
        )
    )


def clear_capability_default(
    kind: str,
    modality: Optional[str] = None,
    *,
    task: Optional[str] = None,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> bool:
    """Clear one capability-default route. Returns True on success.

    Raises `CapabilityDefaultWriteError` (a `ValueError`) on failure, for the
    same reason as `set_capability_default`.
    """
    return bool(
        _configuration_manager(config_file, apply_env=apply_env).clear_capability_default(
            kind,
            modality,
            task,
        )
    )


def apply_recommended_capability_defaults(
    *,
    only: Optional[List[str]] = None,
    force: bool = False,
    dry_run: bool = False,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> Dict[str, Any]:
    """Bring a store's capability routes to AbstractCore's recommendation.

    THE SAME ACTION AT EVERY SURFACE. AbstractCore's fresh-install seed refuses
    to touch an existing store, which left "make my machine match the
    recommendation" unanswerable from the Gateway and both console-TUIs. The
    decision (apply / kept yours / already / overwrite) is AbstractCore's, so
    hosts call it rather than re-deriving it: a Gateway button and the CLI must
    not be able to disagree about whether a route was overruled.

    Returns the per-route before/after report; raises `ValueError` on an unknown
    `only` selector.
    """
    return dict(
        _configuration_manager(config_file, apply_env=apply_env).apply_recommended_capability_defaults(
            only=list(only) if only else None,
            force=bool(force),
            dry_run=bool(dry_run),
        )
    )


def recommended_capability_selectors() -> List[str]:
    """The `only=` vocabulary (`["image", "text", "voice"]`), from AbstractCore.

    Returns `[]` when AbstractCore is unavailable, so a surface that only
    labels a control degrades to silence instead of a wrong list.
    """
    try:
        from abstractcore.config.capability_defaults import RECOMMENDED_SELECTORS

        return sorted(str(name) for name in RECOMMENDED_SELECTORS)
    except Exception:
        return []


def capability_default_specs() -> Dict[str, Dict[str, Any]]:
    """Return {route_key -> spec dict} for all known capability-default routes."""
    try:
        from abstractcore.config.capability_defaults import capability_default_specs_dict
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "AbstractCore capability-default specs are unavailable. Install a Runtime "
            "environment with AbstractCore (the `integrations/abstractcore` opt-in)."
        ) from exc
    return dict(capability_default_specs_dict())


def list_llm_provider_names() -> list[str]:
    """Provider ids AbstractCore can build a TEXT client for.

    The authority is AbstractCore's provider registry -- the same list its
    "Unknown provider: x. Available providers: ..." error prints -- so a caller
    validating a text route default asks the component that will refuse it,
    rather than keeping a second list that drifts. Media/plugin backends
    (`mlx-gen`, `supertonic`, `faster-whisper`, ...) are deliberately NOT here:
    they are valid on media routes and this list answers a text question.

    Returns `[]` when the registry is unavailable, so a caller that only wants
    to WARN degrades to silence instead of to a false alarm.
    """
    try:
        from abstractcore.providers.registry import get_provider_registry

        return [str(name) for name in get_provider_registry().list_provider_names()]
    except Exception:
        return []


def list_provider_profiles(
    *,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> List[Dict[str, Any]]:
    """Every persisted provider endpoint profile, WITH its secrets.

    The rows a host needs to actually resolve `endpoint:<id>` itself, so
    `to_dict()` (api_key, api_key_env_var) rather than the redacted
    `public_dict()`. Only the process holding the store gets these; a host that
    only renders them uses the redacted rows its own layer builds.
    """
    manager = _configuration_manager(config_file, apply_env=apply_env)
    return [
        profile.to_dict()
        for profile in sorted(manager.config.provider_profiles.profiles.values(), key=lambda p: p.id.lower())
    ]


def set_provider_profile(
    profile_id: str,
    *,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
    **fields: Any,
) -> Dict[str, Any]:
    """Create or update one provider endpoint profile in the AbstractCore store."""
    manager = _configuration_manager(config_file, apply_env=apply_env)
    profile = manager.set_provider_profile(profile_id, **fields)
    return profile.to_dict()


def delete_provider_profile(
    profile_id: str,
    *,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> bool:
    """Delete one provider endpoint profile from the AbstractCore store."""
    return bool(_configuration_manager(config_file, apply_env=apply_env).delete_provider_profile(profile_id))


def default_config_document() -> Dict[str, Any]:
    """AbstractCore's UNTOUCHED config document -- what a store nobody edited holds.

    The BASELINE of a three-way merge between two config stores: a section that
    equals this one was never configured by that store's writer, so it must not
    be allowed to win a conflict against a value someone actually chose. Built
    from `AbstractCoreConfig.default()`, so it tracks the dataclasses rather
    than a copy of them that drifts.
    """
    try:
        from dataclasses import asdict

        from abstractcore.config.manager import AbstractCoreConfig
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "AbstractCore configuration support is unavailable. Install a Runtime "
            "environment with AbstractCore (the `integrations/abstractcore` opt-in)."
        ) from exc
    config = AbstractCoreConfig.default()
    return {
        "audio_strategy_explicit": False,
        "vision": asdict(config.vision),
        "audio": asdict(config.audio),
        "video": asdict(config.video),
        "embeddings": asdict(config.embeddings),
        "app_defaults": asdict(config.app_defaults),
        "default_models": asdict(config.default_models),
        "capability_defaults": config.capability_defaults.to_dict(),
        "provider_profiles": config.provider_profiles.to_dict(),
        "api_keys": asdict(config.api_keys),
        "server": asdict(config.server),
        "cache": asdict(config.cache),
        "logging": asdict(config.logging),
        "streaming": asdict(config.streaming),
        "timeouts": asdict(config.timeouts),
        "offline": asdict(config.offline),
        "maintenance": asdict(config.maintenance),
        "email": asdict(config.email),
    }


def merge_config_documents(
    baseline: Optional[Dict[str, Any]],
    mine: Dict[str, Any],
    disk: Dict[str, Any],
) -> Dict[str, Any]:
    """AbstractCore's own three-way store merge, for hosts that reconcile stores.

    The same function every AbstractCore save publishes through
    (`manager.merge_store_documents`): a field changed against the baseline
    wins, a field left at the baseline yields to the other document. Hosts must
    not restate these rules -- a second merge implementation is a second
    opinion about the same store.
    """
    try:
        from abstractcore.config.manager import merge_store_documents
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "AbstractCore configuration support is unavailable. Install a Runtime "
            "environment with AbstractCore (the `integrations/abstractcore` opt-in)."
        ) from exc
    return merge_store_documents(baseline, mine, disk)


def read_config_api_key(config_file: _PathLike, attr: str) -> str:
    """Read a provider API key attribute from an AbstractCore config file.

    Never applies environment side effects (`apply_env=False`) and never raises
    on a missing file/attr; returns an empty string instead.
    """
    path = Path(config_file)
    if not path.exists():
        return ""
    try:
        manager = _configuration_manager(path, apply_env=False)
        value = getattr(manager.config.api_keys, attr, None)
        return str(value or "").strip()
    except Exception:
        return ""


def read_email_settings(*, config_file: Optional[_PathLike] = None) -> Dict[str, Any]:
    """Return AbstractCore's stored IMAP/SMTP settings, or `{}`.

    AbstractCore holds the host's mail connection in its `email` config section
    and its own comms tools resolve from it. A host that polls the same mailbox
    must read the same store rather than keep a second copy, so this is the one
    door to it. Secrets are NOT returned: the section names the environment
    variable a password is read from, which is what travels here.

    Never raises; an unavailable or unreadable config reads as `{}`.
    """
    try:
        section = getattr(_configuration_manager(config_file, apply_env=False).config, "email", None)
    except Exception:
        return {}
    if section is None:
        return {}
    out: Dict[str, Any] = {}
    for name in (
        "smtp_host",
        "smtp_port",
        "smtp_username",
        "smtp_password_env_var",
        "smtp_use_starttls",
        "from_email",
        "reply_to",
        "imap_host",
        "imap_port",
        "imap_username",
        "imap_password_env_var",
        "imap_folder",
    ):
        value = getattr(section, name, None)
        if value is not None and value != "":
            out[name] = value
    return out


def read_maintenance_settings(*, config_file: Optional[_PathLike] = None) -> Dict[str, Any]:
    """Return AbstractCore's stored maintenance-triage LLM settings, or `{}`.

    AbstractCore holds the triage assistant's provider settings in its
    `maintenance` config section. A host that runs the same assistant reads them
    here rather than keeping a second copy under its own environment names.

    Never raises; an unavailable or unreadable config reads as `{}`.
    """
    try:
        section = getattr(_configuration_manager(config_file, apply_env=False).config, "maintenance", None)
    except Exception:
        return {}
    if section is None:
        return {}
    out: Dict[str, Any] = {}
    for name in (
        "triage_llm_enabled",
        "triage_llm_base_url",
        "triage_llm_model",
        "triage_llm_temperature",
        "triage_llm_max_tokens",
        "triage_llm_timeout_s",
    ):
        value = getattr(section, name, None)
        if value is not None and value != "":
            out[name] = value
    return out


# ---------------------------------------------------------------------------
# Model weights: is the configured model actually ON this machine?
# ---------------------------------------------------------------------------
#
# A capability default names a provider and a model; whether that model's
# WEIGHTS exist locally is a different question, and AbstractCore owns the one
# answer (`abstractcore.config.model_materializer`). These wrappers exist so a
# host asks the same materializer the AbstractCore CLI and console-TUI ask,
# instead of shelling out to `lms`/`ollama` and inventing a second opinion.
#
# Everything crossing this boundary is JSON-safe: hosts never see AbstractCore
# dataclasses, so a host payload is a serialization, not a re-derivation.


def _model_materializer():
    try:
        from abstractcore.config import model_materializer
    except Exception as exc:  # pragma: no cover - exercised through facade behavior tests
        raise RuntimeError(
            "AbstractCore model availability support is unavailable. Install a Runtime "
            "environment with AbstractCore (the `integrations/abstractcore` opt-in)."
        ) from exc
    return model_materializer


def probe_model_presence(provider: str, model: str, *, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Are `model`'s weights present locally for `provider`?

    Returns `{"provider", "artifact", "status", "downloadable", ...}` where
    status is one of installed / absent / unknown / not_applicable. Reads only:
    this never downloads and never contacts a model hub.
    """
    return dict(_model_materializer().probe(provider, model, base_url=base_url).to_dict())


def model_availability(
    *,
    config_file: Optional[_PathLike] = None,
    apply_env: bool = True,
) -> Dict[str, Any]:
    """Capability-default routes annotated with local weight availability.

    The `routes` are exactly what `list_capability_defaults()` returns, each
    carrying an extra `availability` object and, where the served model id is
    not the download reference (quantization pinned), a `download_artifact`.
    `recommended` is the fresh-install set with the same annotation, which is
    what a "download the missing defaults" surface acts on.
    """
    materializer = _model_materializer()
    manager = _configuration_manager(config_file, apply_env=apply_env)
    # ONE SWEEP, so the grid and the "N of 3 present" summary in the same
    # payload are computed from ONE reading of each provider's library --
    # they cannot contradict each other, and a wedged provider CLI costs its
    # timeout once instead of once per probe.
    with materializer.presence_sweep():
        routes = materializer.annotate_route_availability(manager.list_capability_defaults())
        recommended = materializer.recommended_plan()
    return {
        "ok": True,
        "version": 1,
        "config_file": str(manager.config_file),
        "routes": routes,
        "recommended": recommended,
        "providers": materializer.supported_providers(),
    }


def split_model_artifact(artifact: str) -> tuple:
    """`"qwen/qwen3.5-9b@4bit"` -> `("qwen/qwen3.5-9b", "4bit")`.

    Re-exported, never reimplemented: AbstractCore's materializer is the ONE
    place that knows the `@quant` convention, and a host that split on `@`
    itself would be a second copy of it -- one that disagrees the first time a
    model id legitimately contains an `@`.
    """
    return _model_materializer().split_artifact(artifact)


def model_presence_sweep() -> Any:
    """Context manager: read each provider's model library once inside the block.

    A host that annotates a grid AND asks for the recommended plan is building
    ONE payload out of two calls. Wrapping both in a sweep makes them one
    consistent snapshot -- a download landing between them can no longer make
    the banner and the rows disagree -- and costs a wedged provider CLI its
    timeout once rather than once per probe. Outside a sweep nothing is cached.
    """
    return _model_materializer().presence_sweep()


def annotate_model_availability(routes: Any) -> List[Dict[str, Any]]:
    """Annotate ALREADY-RESOLVED capability rows with local weight availability.

    A host that resolves its own route rows (a Gateway overlaying per-principal
    and per-runtime config on the install store) must annotate THOSE rows, not
    a second read of the local config -- otherwise the availability column and
    the provider/model columns would describe different stores.
    """
    return list(_model_materializer().annotate_route_availability(routes or []))


def recommended_model_plan() -> Dict[str, Any]:
    """The recommended fresh-install set, probed. No downloads, no hub calls.

    `{"recommended": [...], "total", "installed", "absent", "unknown",
    "would_download": [...]}` -- the payload behind `--dry-run`. It answers ONE
    question, "is the starter-kit model on this disk?", so a surface that
    reports it as work to do must first pass it through
    `mark_recommended_route_gaps`.
    """
    return dict(_model_materializer().recommended_plan())


def mark_recommended_route_gaps(plan: Dict[str, Any], routes: Any) -> Dict[str, Any]:
    """Split a `recommended_model_plan()` into ADVICE and GAPS, in place.

    `plan["gaps"]` is the subset of `would_download` whose route has nothing
    else serving it -- the only part a console may present as work to do. A
    host that resolves its own route rows passes THOSE rows, for the same
    reason it annotates them itself (see `annotate_model_availability`).

    AbstractCore owns the judgement; this is the door it comes through. A host
    re-deriving "is this route answered?" from its own row fields is exactly
    the drift this seam exists to prevent.
    """
    return dict(_model_materializer().mark_recommended_route_gaps(plan, routes or []))


def recommended_model_downloads() -> List[Dict[str, str]]:
    """The `{route, provider, artifact}` triples the recommended set fetches.

    The artifact is the EXACT weights reference (`qwen/qwen3.5-9b@4bit`), which
    differs from the route's served model id whenever a quantization is pinned.
    """
    return [dict(item) for item in _model_materializer().recommended_downloads()]


def download_model_artifact(
    provider: str,
    artifact: str,
    *,
    progress_cb: Optional[Any] = None,
    base_url: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the provider's own download tool once, for ONE artifact.

    Blocking and explicit: a host calls this from a worker thread in response to
    a human action, never on a render path. `progress_cb` receives AbstractCore
    `DownloadProgress` objects; `dry_run` resolves the command and stops.
    """
    outcome = _model_materializer().download(
        provider,
        artifact,
        progress_cb=progress_cb,
        base_url=base_url,
        dry_run=dry_run,
    )
    return dict(outcome.to_dict())


__all__ = [
    "list_capability_defaults",
    "capability_default_config_file",
    "capability_default_config_path",
    "capability_defaults_seed_marker",
    "set_capability_default",
    "clear_capability_default",
    "capability_default_specs",
    "list_llm_provider_names",
    "split_model_artifact",
    "model_presence_sweep",
    "read_config_api_key",
    "read_email_settings",
    "read_maintenance_settings",
    "probe_model_presence",
    "model_availability",
    "annotate_model_availability",
    "recommended_model_plan",
    "mark_recommended_route_gaps",
    "recommended_model_downloads",
    "download_model_artifact",
]
