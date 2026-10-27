"""The client side of a replay view's change: `apply_delta`, `diff_views`, cursor tokens.

A delta (`ReplayView.apply`'s return, a `delta` event on the gateway's view stream) carries
the view's scalars whole (`status`, `display_status`, `interaction`, `open_waits`, `usage`,
`warnings`, `cursor`) and the list items that changed (`runs`, `messages`, `activities`,
`artifacts`), each keyed by `id` (`run_id` for runs). Items are never removed: an item only
changes. `apply_delta(view, delta)` therefore replaces the scalars, upserts the items and
re-sorts each list by its `order`, and `apply_delta(old, diff_views(old, new)) == new`.
A delta's `changes` (typed change entries for a push dispatcher, root backlog 1058) describe the
same change and are ignored here: a client's view is the scalars and the lists.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

SCALAR_KEYS = ("version", "root_run_id", "workflow", "session_id", "status", "display_status", "cursor", "interaction", "open_waits", "usage", "warnings")
LIST_KEYS = ("runs", "messages", "activities", "artifacts")


def _item_id(key: str, item: Dict[str, Any]) -> str:
    return str(item.get("run_id") if key == "runs" else item.get("id"))


def _order_key(key: str, item: Dict[str, Any]) -> tuple:
    """Sort by `order` (compared element by element, a prefix first: a run row's `order` is its
    path from the root, so a parent sorts before its children), then by id."""
    order = item.get("order") or ()
    return (tuple(str(x) if not isinstance(x, int) else f"{x:012d}" for x in order), _item_id(key, item))


def apply_delta(view: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """Return `view` with `delta` applied (neither argument is modified)."""
    out = copy.deepcopy(view) if view else {k: [] for k in LIST_KEYS}
    for k in SCALAR_KEYS:
        if k in delta:
            out[k] = copy.deepcopy(delta[k])
    for k in LIST_KEYS:
        items = delta.get(k)
        if not items:
            out.setdefault(k, [])
            continue
        by_id = {_item_id(k, it): it for it in out.get(k) or []}
        for it in items:
            by_id[_item_id(k, it)] = copy.deepcopy(it)
        out[k] = sorted(by_id.values(), key=lambda it, k=k: _order_key(k, it))
    return out


def diff_views(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    """The delta that turns `old` into `new` (every scalar, the changed items)."""
    delta: Dict[str, Any] = {k: copy.deepcopy(new.get(k)) for k in SCALAR_KEYS if k in new}
    for k in LIST_KEYS:
        before = {_item_id(k, it): it for it in (old or {}).get(k) or []}
        delta[k] = [copy.deepcopy(it) for it in new.get(k) or [] if before.get(_item_id(k, it)) != it]
    return delta


def parse_cursor_token(token: Any) -> Optional[Dict[str, int]]:
    """`"<run id prefix>:<seq>.…"` (a view's `cursor.token`, its ETag and SSE event id) -> {prefix: seq}.

    None when the token is not one (a client's own id, a damaged header)."""
    if not isinstance(token, str) or not token.strip():
        return None
    out: Dict[str, int] = {}
    raw = token.strip()
    if raw.startswith("W/"):
        raw = raw[2:]
    for part in raw.strip('"').split("."):
        prefix, sep, seq = part.partition(":")
        if not sep or not prefix.strip():
            return None
        try:
            n = int(seq)
        except ValueError:
            return None
        if n < 0:
            return None
        out[prefix.strip()] = n
    return out or None
