"""Resolve the explicit tool ceiling across a run tree.

Approval policy decides *whether to ask*. It cannot grant a tool outside this
ceiling. A missing ``allowed_tools`` key is unrestricted; an explicit empty
list denies every tool. No separate persisted policy namespace is required.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .models import RunState
from ..storage.base import RunStore


class ToolScopeError(ValueError):
    """The effective tool ceiling cannot safely be established."""


def resolve_tool_scope(
    run: RunState,
    *,
    run_store: Optional[RunStore] = None,
    local_scope: Optional[Mapping[str, Any]] = None,
) -> Optional[frozenset[str]]:
    """Intersect a node/child scope with the run and every recorded ancestor.

    ``local_scope`` is an existing effect payload or child ``_runtime`` dict;
    only its ``allowed_tools`` key is read. Malformed explicit lists and an
    incomplete/cyclic ancestry fail closed, rather than becoming unrestricted.
    Runtime supplies its store transiently to handlers for backwards-compatible
    handler construction without an explicit ``run_store`` argument.
    """
    ceiling: Optional[frozenset[str]] = None

    def include(scope: Any) -> None:
        nonlocal ceiling
        if not isinstance(scope, Mapping) or "allowed_tools" not in scope:
            return
        raw = scope["allowed_tools"]
        if not isinstance(raw, list):
            raise ToolScopeError("allowed_tools must be a list when explicitly set")
        names = frozenset(name.strip() for name in raw if isinstance(name, str) and name.strip())
        ceiling = names if ceiling is None else ceiling & names

    include(local_scope)
    store = run_store if run_store is not None else getattr(run, "_runtime_run_store", None)
    current = run
    seen: set[str] = set()
    for _ in range(256):
        current_id = str(current.run_id)
        if current_id in seen:
            raise ToolScopeError("Cannot resolve tool scope: cyclic run ancestry")
        seen.add(current_id)
        vars0 = current.vars if isinstance(current.vars, dict) else {}
        include(vars0.get("_runtime"))
        parent_id = current.parent_run_id
        if parent_id is None or parent_id == "":
            return ceiling
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise ToolScopeError("Cannot resolve tool scope: invalid parent run id")
        if store is None:
            raise ToolScopeError("Cannot resolve tool scope: parent run store is unavailable")
        try:
            parent = store.load(parent_id)
        except Exception as exc:
            raise ToolScopeError("Cannot resolve tool scope: parent run could not be read") from exc
        if parent is None or parent.run_id != parent_id:
            raise ToolScopeError("Cannot resolve tool scope: parent run is missing or mismatched")
        current = parent
    raise ToolScopeError("Cannot resolve tool scope: run ancestry exceeds 256 levels")
