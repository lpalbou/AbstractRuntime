"""Trigger-source registry (automations contract C).

Sources come from two places:

- the built-ins `schedule@1` and `manual@1`, shipped in this package. They are
  a REQUIRED seam: if one cannot be loaded, or no longer describes itself as
  expected, discovery raises `TriggerRegistryError` — automations must not
  start on a runtime that silently lost its scheduler;
- third-party packages, through the entry-point group
  `abstractruntime.trigger_sources` (`name = "module:AdapterClass"`). A broken
  third-party source (import error, bad descriptor, entry-point name different
  from `descriptor.id`) is listed with `available: False` and a reason, and is
  never selectable; it does not take the registry down.

The built-ins are also declared in that entry-point group by this package's
`pyproject.toml`, so the group is the one list hosts can inspect.
"""

from __future__ import annotations

import importlib
import threading
from importlib import metadata
from typing import Any, Dict, List, Optional, Tuple

from .protocol import TriggerAdapter

ENTRY_POINT_GROUP = "abstractruntime.trigger_sources"

BUILTIN_TRIGGER_SOURCES: Dict[str, str] = {
    "schedule": "abstractruntime.triggers.schedule:ScheduleTriggerAdapter",
    "manual": "abstractruntime.triggers.manual:ManualTriggerAdapter",
}

_ADAPTER_METHODS = ("validate", "initial_state", "prepare", "admit", "rearm", "normalize")
_DESCRIPTOR_KEYS = ("id", "version", "label", "config_schema", "event_schema", "capabilities")


class TriggerRegistryError(RuntimeError):
    """A required trigger source (a built-in) is missing or broken."""


class UnknownTriggerSource(LookupError):
    """No available trigger source matches `source_id@source_version`."""

    reason_code = "unknown_trigger_source"

    def __init__(self, source_id: Any, source_version: Any, detail: str = "") -> None:
        self.source_id = source_id
        self.source_version = source_version
        msg = f"unknown trigger source {source_id!r}@{source_version!r}"
        super().__init__(f"{msg}: {detail}" if detail else msg)


def _load_target(target: str) -> Any:
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ValueError(f"entry point target must be 'module:Attribute', got {target!r}")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _instantiate(obj: Any) -> Any:
    return obj() if isinstance(obj, type) else obj


def _check_adapter(name: str, adapter: Any) -> Dict[str, Any]:
    """Validate an adapter's shape; returns its descriptor or raises ValueError."""
    missing = [m for m in _ADAPTER_METHODS if not callable(getattr(adapter, m, None))]
    if missing:
        raise ValueError(f"adapter lacks method(s) {missing}")
    descriptor = getattr(adapter, "descriptor", None)
    if not isinstance(descriptor, dict):
        raise ValueError("adapter has no descriptor dict")
    absent = [k for k in _DESCRIPTOR_KEYS if k not in descriptor]
    if absent:
        raise ValueError(f"descriptor lacks {absent}")
    version = descriptor.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"descriptor.version must be an integer >= 1, got {version!r}")
    if descriptor.get("id") != name:
        raise ValueError(f"entry point name {name!r} differs from descriptor.id {descriptor.get('id')!r}")
    kind = (descriptor.get("capabilities") or {}).get("kind") if isinstance(descriptor.get("capabilities"), dict) else None
    if kind not in ("time", "manual", "event"):
        raise ValueError(f"descriptor.capabilities.kind must be time|manual|event, got {kind!r}")
    return descriptor


def _entry_points() -> List[Tuple[str, str]]:
    """(name, target) pairs registered in the group, in a stable order."""
    eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    return sorted({(ep.name, ep.value) for ep in eps})


class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._adapters: Dict[Tuple[str, int], Any] = {}

    def reset(self) -> None:
        with self._lock:
            self._rows = None
            self._adapters = {}

    def _discover(self) -> None:
        rows: List[Dict[str, Any]] = []
        adapters: Dict[Tuple[str, int], Any] = {}

        for name, target in BUILTIN_TRIGGER_SOURCES.items():
            try:
                adapter = _instantiate(_load_target(target))
                descriptor = _check_adapter(name, adapter)
            except Exception as exc:  # a required seam: fail loudly
                raise TriggerRegistryError(
                    f"built-in trigger source {name!r} ({target}) is missing or broken: {exc}"
                ) from exc
            adapters[(descriptor["id"], int(descriptor["version"]))] = adapter
            rows.append({"descriptor": descriptor, "available": True})

        for name, target in _entry_points():
            if name in BUILTIN_TRIGGER_SOURCES:
                if target != BUILTIN_TRIGGER_SOURCES[name]:
                    rows.append({
                        "descriptor": None,
                        "name": name,
                        "available": False,
                        "unavailable_reason": f"entry point {name!r} ({target}) conflicts with the built-in source",
                    })
                continue
            try:
                adapter = _instantiate(_load_target(target))
                descriptor = _check_adapter(name, adapter)
                key = (descriptor["id"], int(descriptor["version"]))
                if key in adapters:
                    raise ValueError(f"duplicate trigger source {key[0]}@{key[1]}")
            except Exception as exc:
                rows.append({
                    "descriptor": None,
                    "name": name,
                    "available": False,
                    "unavailable_reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            adapters[key] = adapter
            rows.append({"descriptor": descriptor, "available": True})

        self._rows = rows
        self._adapters = adapters

    def rows(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._rows is None:
                self._discover()
            return [dict(r) for r in self._rows or []]

    def adapter(self, source_id: Any, source_version: Any) -> Any:
        with self._lock:
            if self._rows is None:
                self._discover()
            try:
                key = (str(source_id), int(source_version))
            except (TypeError, ValueError):
                raise UnknownTriggerSource(source_id, source_version, "source_version must be an integer")
            adapter = self._adapters.get(key)
        if adapter is None:
            raise UnknownTriggerSource(source_id, source_version)
        return adapter


_REGISTRY = _Registry()


def trigger_sources() -> List[Dict[str, Any]]:
    """Every discovered source: `{descriptor|None, available, unavailable_reason?, name?}`.

    Raises TriggerRegistryError when a built-in source is missing or broken.
    """
    return _REGISTRY.rows()


def get_trigger_adapter(source_id: Any, source_version: Any) -> TriggerAdapter:
    """The available adapter for `source_id@source_version`; raises UnknownTriggerSource."""
    return _REGISTRY.adapter(source_id, source_version)


def reset_trigger_registry() -> None:
    """Forget cached discovery (tests, or after installing a source package)."""
    _REGISTRY.reset()


__all__ = [
    "BUILTIN_TRIGGER_SOURCES",
    "ENTRY_POINT_GROUP",
    "TriggerRegistryError",
    "UnknownTriggerSource",
    "get_trigger_adapter",
    "reset_trigger_registry",
    "trigger_sources",
]
