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
    """Persist one capability-default route. Returns True on success."""
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
    """Clear one capability-default route. Returns True on success."""
    return bool(
        _configuration_manager(config_file, apply_env=apply_env).clear_capability_default(
            kind,
            modality,
            task,
        )
    )


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


__all__ = [
    "list_capability_defaults",
    "capability_default_config_file",
    "set_capability_default",
    "clear_capability_default",
    "capability_default_specs",
    "read_config_api_key",
]
