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
import re
from typing import Any, Dict, Iterable, List, Optional

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
    """Return `view` with `delta` applied (neither argument is modified).

    Cost follows the delta, not the view: the result shares every unchanged item with `view`
    (items are replaced, never edited in place), copies only the delta's items, and re-sorts a
    list only when an item is new or moved."""
    out: Dict[str, Any] = dict(view) if view else {k: [] for k in LIST_KEYS}
    for k in SCALAR_KEYS:
        if k in delta:
            out[k] = copy.deepcopy(delta[k])
    for k in LIST_KEYS:
        items = delta.get(k)
        current = out.get(k) or []
        if not items:
            out[k] = current
            continue
        rows = list(current)
        where = {_item_id(k, it): i for i, it in enumerate(rows)}
        resort = False
        for it in items:
            new_it = copy.deepcopy(it)
            i = where.get(_item_id(k, it))
            if i is None:
                where[_item_id(k, it)] = len(rows)
                rows.append(new_it)
                resort = True
            else:
                if rows[i].get("order") != new_it.get("order"):
                    resort = True
                rows[i] = new_it
        if resort:
            rows.sort(key=lambda it, k=k: _order_key(k, it))
        out[k] = rows
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


_TOKEN_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def _token_id(run_id: str) -> str:
    return _TOKEN_UNSAFE.sub("", str(run_id or ""))


def token_run_names(run_ids: Iterable[str]) -> Dict[str, str]:
    """run id -> the name its cursor token part uses: the run id's letters, digits, `_` and `-`,
    cut to the shortest length (8 at least) that no other run of the tree shares."""
    ids = list(dict.fromkeys(run_ids))
    safe = {rid: _token_id(rid) for rid in ids}
    length = 8
    longest = max((len(v) for v in safe.values()), default=0)
    while length < longest and len({v[:length] for v in safe.values()}) < len(ids):
        length += 4
    return {rid: (safe[rid][:length] or "_") for rid in ids}


def resolve_cursor_token(token: Any, run_ids: Iterable[str]) -> Optional[Dict[str, int]]:
    """A cursor token -> {run id: seq} over `run_ids` (a view's runs); None when the token is not
    one of this tree's (a name matching no run, or more than one)."""
    parsed = parse_cursor_token(token)
    if not parsed:
        return None
    safe = {rid: _token_id(rid) for rid in run_ids}
    out: Dict[str, int] = {}
    for name, seq in parsed.items():
        hits = [rid for rid, v in safe.items() if v.startswith(name)]
        if len(hits) != 1 or hits[0] in out:
            return None
        out[hits[0]] = seq
    return out
