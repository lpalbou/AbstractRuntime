"""An authored summary is never stamped mechanical (live finding 2026-11-09: the visit episode formed with
digest_method "mechanical-v2" BEFORE its authored title/summary were merged, so memory read the model-authored
summary as mechanical and queued a `digest.rewrite` for the sleep - tokens spent rewriting authored words).

The label follows the digest's words: the summaries capability wrote the summary -> `authored:summaries`; only a truly
mechanical digest (the authoring call failed, the digest is the provisional mechanical one) keeps `mechanical-v2`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pytest

pytest.importorskip("abstractmemory")
sys.path.insert(0, str(Path(__file__).parent))

from test_memory_authoring import (  # noqa: E402
    SUMMARY,
    TITLE,
    _Capability,
    _close_visit,
    _rows,
    _session,
    _tasks,
    _visit,
    _workspace_home,
)


def _rewrites_for(tasks: List[Any], record_id: str) -> List[Any]:
    return [t for t in tasks if t["kind"] == "digest.rewrite" and t["target_id"] == record_id]


def _authored_summaries(capability_payloads: List[Any]) -> Any:
    import json

    from abstractruntime.core.runtime import EffectOutcome

    def summaries(run: Any, effect: Any, dnn: Any = None) -> Any:
        capability_payloads.append(dict(effect.payload))
        return EffectOutcome.completed({"content": json.dumps({"title": TITLE, "summary": SUMMARY}), "model": "sum-1"})

    return summaries


def test_an_authored_visit_episode_says_authored_and_is_not_queued_for_repair(tmp_path: Path) -> None:
    from abstractmemory import MECHANICAL_DIGEST_METHODS

    from abstractruntime.identity.chat import AUTHORED_SUMMARIES_METHOD

    calls: List[Any] = []
    ert, _home_dir, episodes = _visit(tmp_path, {"summaries": _authored_summaries(calls)})
    try:
        assert calls, "the visit called the summaries capability"
        ep = episodes[0]
        assert str(ep.object) == SUMMARY and "pending" not in ep.attributes
        assert ep.attributes["digest_method"] == AUTHORED_SUMMARIES_METHOD == "authored:summaries", ep.attributes
        assert ep.attributes["digest_method"] not in MECHANICAL_DIGEST_METHODS
        # Memory's record is NOT repairable: no digest.rewrite queued at formation, no rewrite debt counted.
        assert _rewrites_for(list(ert.home.ms.tasks.tasks()), str(ep.subject)) == []
        debt = ert.home.ms.understanding_debt(scopes=[("life", ert.home.entity_id)])
        assert "episode" not in (debt["debt"].get("digest.rewrite") or {}), debt
    finally:
        ert.close()


def test_a_failed_visit_authoring_keeps_the_mechanical_label(tmp_path: Path) -> None:
    from abstractruntime.core.runtime import EffectOutcome

    ert, _home_dir, episodes = _visit(
        tmp_path, {"summaries": lambda run, effect, dnn=None: EffectOutcome.failed("endpoint refused")})
    try:
        ep = episodes[0]
        assert ep.attributes["pending"]["summary"] is True
        assert ep.attributes["digest_method"] == "mechanical-v2", "the provisional digest IS the mechanical one"
        assert "metadata_authored" not in ep.attributes
    finally:
        ert.close()


def test_an_authored_chat_episode_says_authored_and_is_not_queued_for_repair(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Hello there."])
    s.llm_for_capability = lambda name: _Capability() if name == "summaries" else None
    s.turn("hi")
    home.close()
    ep = _rows(home_dir, "episode")[-1]
    assert str(ep.object) == SUMMARY
    assert ep.attributes["digest_method"] == "authored:summaries", ep.attributes
    assert _rewrites_for(_tasks(home_dir, "digest.rewrite"), str(ep.subject)) == []


def test_a_chat_episode_without_a_summaries_model_keeps_the_mechanical_label(tmp_path: Path) -> None:
    home_dir = _workspace_home(tmp_path)
    home, s = _session(home_dir, ["Hello there."])
    s.llm_for_capability = None
    s.turn("hi")
    home.close()
    ep = _rows(home_dir, "episode")[-1]
    assert ep.attributes["digest_method"] == "mechanical-v2" and ep.attributes["pending"]["summary"] is True


def test_a_title_only_authoring_leaves_the_digest_label_alone() -> None:
    from abstractruntime.identity.chat import authored_digest_attributes

    attrs = {"digest_method": "mechanical-floor-v1", "phase": "visit"}
    assert authored_digest_attributes(attrs, ["title"]) == attrs
    assert authored_digest_attributes(attrs, ["title", "summary"])["digest_method"] == "authored:summaries"
    assert attrs["digest_method"] == "mechanical-floor-v1", "the caller's dict is not mutated"


def test_the_visit_close_summary_says_authored(tmp_path: Path) -> None:
    import json

    from abstractruntime.core.runtime import EffectOutcome

    def summaries(run: Any, effect: Any, dnn: Any = None) -> Any:
        return EffectOutcome.completed({"content": json.dumps({"title": "A title", "summary": "What it holds."})})

    ert, _home_dir, rows = _close_visit(tmp_path, {"summaries": summaries})
    try:
        summary = rows["summary"][0]
        assert str(summary.object) == "What it holds."
        assert summary.attributes["digest_method"] == "authored:summaries", summary.attributes
        assert _rewrites_for(list(ert.home.ms.tasks.tasks()), str(summary.subject)) == []
    finally:
        ert.close()
