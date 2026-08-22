"""Runtime-owned facade over AbstractCore's data-home registry.

Hosts (e.g. AbstractGateway) import THIS module instead of reaching into
`abstractcore.utils.data_registry` directly — the ruled host -> Runtime ->
AbstractCore boundary (gateway backlog 0059; `config_facade.py` precedent;
shipped for the Data & Caches console lane, commons c1771, 2026-07-14).

Thin pass-throughs, zero logic: signatures, return shapes, and the refusal
lattice (`DataRegistryError` naming owner + rule verbatim) are core's —
this module adds nothing and must never grow policy. Lazy imports keep the
module light and the missing-dependency error actionable (the facade-family
contract).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _data_registry():
    try:
        from abstractcore.utils import data_registry
    except Exception as exc:  # pragma: no cover - exercised via facade behavior tests
        raise RuntimeError(
            "AbstractCore data-registry support is unavailable. Install a Runtime "
            "environment with AbstractCore (the `integrations/abstractcore` opt-in)."
        ) from exc
    return data_registry


def ensure_data_home_registered(
    name: str,
    *,
    path: str,
    kind: str,
    owner: str,
    safe_to_purge: bool,
    description: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Best-effort registration for hot paths — never raises (core contract)."""
    return _data_registry().ensure_data_home_registered(
        name,
        path=path,
        kind=kind,
        owner=owner,
        safe_to_purge=safe_to_purge,
        description=description,
        meta=meta,
    )


def list_data_homes(*, include_sizes: bool = False) -> List[Dict[str, Any]]:
    """All registered rows (JSON-ready dicts; live sizes when requested)."""
    return _data_registry().list_data_homes(include_sizes=include_sizes)


def purge_data_home(name: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """Purge a registered, owner-declared-safe home's CONTENTS.

    Core's refusal lattice propagates verbatim (`DataRegistryError` naming
    owner + rule): unknown names refuse, `safe_to_purge=False` refuses, the
    home directory itself survives, symlinks are never followed.
    `dry_run=True` returns the would-purge accounting without deleting.
    """
    return _data_registry().purge_data_home(name, dry_run=dry_run)


def unregister_data_home(name: str) -> bool:
    """Remove ONE registry ROW (never touches disk). The stale-registration
    cleanup lane (gateway console 'Forget', 2026-08-19): rows whose path
    was deleted or whose data root moved. A row whose path still exists
    re-registers at the owner's next boot — callers should refuse those."""
    return bool(_data_registry().unregister_data_home(name))


__all__ = [
    "ensure_data_home_registered",
    "list_data_homes",
    "purge_data_home",
    "unregister_data_home",
]
