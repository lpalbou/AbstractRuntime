"""Backlog 0938: a third-party run store that cannot attribute a session is refused, never started as a
plain chat. Three shapes: no index; an index whose rows carry no `session_kind` (a store written before
automations v1); a full index."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from abstractruntime.core.run_attribution import SessionAttributionError, session_attribution, store_session_kinds

pytestmark = pytest.mark.basic


class _NoIndex:
    pass


class _IndexWithoutKinds:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def list_run_index(self, *, session_id: str, limit: int) -> List[Dict[str, Any]]:
        return [r for r in self.rows if r["session_id"] == session_id][:limit]


def test_a_store_without_an_index_is_refused_naming_what_is_missing() -> None:
    with pytest.raises(SessionAttributionError) as info:
        store_session_kinds(_NoIndex(), "s1")
    assert "session_kinds()" in str(info.value) and "list_run_index()" in str(info.value)


def test_an_index_without_session_kind_is_refused_not_started_as_a_chat() -> None:
    store = _IndexWithoutKinds([{"run_id": "r1", "session_id": "discussion-session:a1:1:r0", "status": "completed"}])
    with pytest.raises(SessionAttributionError) as info:
        session_attribution(store, "discussion-session:a1:1:r0")
    assert "session_kind" in str(info.value)


def test_a_full_index_attributes_and_a_new_session_is_none() -> None:
    store = _IndexWithoutKinds([{"run_id": "r1", "session_id": "s1", "session_kind": "chat", "role": "root"}])
    assert store_session_kinds(store, "s1") == frozenset({"chat"})
    assert session_attribution(store, "s1") == {"kind": "chat"}
    assert session_attribution(store, "s-new") is None  # no runs yet: nothing to attribute
