"""Cursor tokens name each run of the tree unambiguously (ADR 0010).

A token (`<run name>:<seq>.…`) is a view's ETag and SSE id; a host maps it back to the tree's
runs to rebuild the view at that cursor. Run names are run-id prefixes long enough that no
two runs of the tree share one, made of header-safe characters only.
"""
from __future__ import annotations

from abstractruntime.core.models import StepRecord, StepStatus
from abstractruntime.replay import parse_cursor_token, replay, resolve_cursor_token, token_run_names


def _rec(run_id: str, seq: int, **kw):
    return {"run_id": run_id, "step_id": f"s{seq}", "node_id": "n", "status": "completed", "seq": seq, **kw}


def test_runs_sharing_an_8_character_prefix_get_longer_names() -> None:
    root, a, b = "run-0000-root", "child-00-aaaa", "child-00-bbbb"
    names = token_run_names([root, a, b])
    assert len(set(names.values())) == 3 and names[root].startswith("run-0000")
    assert names[a] != names[b] and len(names[a]) >= 9
    assert resolve_cursor_token(f"{names[root]}:2.{names[a]}:1.{names[b]}:5", [root, a, b]) == {root: 2, a: 1, b: 5}
    # The short, ambiguous name is not this tree's token.
    assert resolve_cursor_token("child-00:1", [root, a, b]) is None


def test_unsafe_characters_never_reach_the_token() -> None:
    rid = 'run:1.2 "x",y'
    names = token_run_names([rid])
    assert all(c not in names[rid] for c in ':." ,')
    view = replay({rid: [_rec(rid, 1), _rec(rid, 2)]}, root_run_id=rid)
    token = view.cursor["token"]
    assert parse_cursor_token(token) is not None and resolve_cursor_token(token, view.run_ids) == {rid: 2}


def test_a_view_token_resolves_to_every_run_seq() -> None:
    root, child = "aaaaaaaa-root", "aaaaaaaa-child"
    wait = {"wait": {"reason": "subworkflow", "wait_key": f"subworkflow:{child}"}}
    view = replay({root: [_rec(root, 1, status="waiting", effect={"type": "start_subworkflow"}, result=wait)],
                   child: [_rec(child, 1), _rec(child, 2)]}, root_run_id=root)
    assert resolve_cursor_token(view.cursor["token"], view.run_ids) == {root: 1, child: 2}
