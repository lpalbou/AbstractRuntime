"""Root 1145 step 1 (operator 2026-11-08): "there should be no activity, either by human or by entities, that
does not go through the runtime ... to enable retrospective analysis / search, including to rehydrate
context/memory". Every tool result during entity activity rests WHOLE as an artifact of the entity's home
(content-addressed, no size cut, with tool / arguments / turn or tick id / created_at), and the result text the
model reads names it (`[artifact: <id>]`) so the verbatim and the memory records can reference it.

Red without the artifact write: the visit lane leaves no artifacts for its two calls, the personal-time fetch
leaves no whole page, and no result carries an artifact id.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunState  # noqa: E402
from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402


def _make_home(tmp_path: Path, slug: str = "durable") -> Path:
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = slug.capitalize()
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


class _ScriptedLLM:
    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, messages: List[Dict[str, str]], system_prompt: str) -> Any:
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})

        class _R:
            content = self.replies.pop(0)

        return _R()


def _artifact_line(artifact_id: str) -> str:
    return f"\n[artifact: {artifact_id}]"


def _home_artifact_texts(home_dir: Path) -> List[str]:
    return [
        p.read_text(encoding="utf-8", errors="ignore")
        for p in (home_dir / "artifacts").rglob("*")
        if p.is_file()
    ]


# ------------------------------------------------------------------ the ledgered (effect) lane: a visit turn


def test_visit_tool_round_keeps_search_and_read_file_whole_as_home_artifacts(tmp_path: Path, monkeypatch) -> None:
    import abstractcore.tools.common_tools as ct

    def fake_search(*, query: str, num_results: int = 5) -> str:
        return json.dumps({"results": [
            {"title": "Beaver dams", "snippet": "built from wood and mud", "url": "https://example.org/beavers"},
        ]})

    monkeypatch.setattr(ct, "web_search", fake_search)

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    try:
        # A workspace file LONGER than the text read cap (24,000 chars): the shown result is a labeled view,
        # the artifact must be the whole file, byte for byte.
        notes = home_dir / "workspace" / "notes"
        notes.mkdir(parents=True)
        plan = notes / "plan.md"
        plan.write_text("".join(f"line {i}: the plan for the river bend, step by step.\n" for i in range(600)), encoding="utf-8")
        file_bytes = plan.read_bytes()
        assert len(file_bytes) > 24_000

        run = RunState.new(workflow_id="entity-visit", entry_node="TOOLS", session_id="visit-7")
        run.current_node = "TOOLS"
        out = home.handlers[EffectType.ENTITY_TOOLS_EXECUTE](
            run,
            Effect(type=EffectType.ENTITY_TOOLS_EXECUTE, payload={
                "phase": "visit",
                "turn_id": "t-0003",
                "tool_calls": [
                    {"name": "web_search", "arguments": {"query": "beaver dams"}},
                    {"name": "read_file", "arguments": {"path": "notes/plan.md"}},
                ],
            }),
            None,
        )
        assert out.status == "completed", out.error
        results = {r["name"]: r for r in out.result["results"]}
        assert set(results) == {"web_search", "read_file"}

        # 1. The search output, whole, equals the artifact byte for byte; the result text names it.
        search = results["web_search"]
        search_id = search["artifact_id"]
        assert search_id and search["result"].endswith(_artifact_line(search_id))
        shown_search = search["result"][: -len(_artifact_line(search_id))]
        assert "Beaver dams" in shown_search and "https://example.org/beavers" in shown_search
        assert home.artifacts.load(search_id).content == shown_search.encode("utf-8")

        # 2. The file read: the shown result is a labeled view, the artifact is the whole file.
        read = results["read_file"]
        read_id = read["artifact_id"]
        assert read_id and read["result"].endswith(_artifact_line(read_id))
        assert "#TRUNCATION" in read["result"], "the shown text is bounded"
        assert home.artifacts.load(read_id).content == file_bytes, "the artifact is the whole file, byte for byte"

        # Metadata: tool, arguments, turn id, created_at, lane, run id; tags indexed by tool and lane.
        meta = home.artifacts.get_metadata(read_id)
        assert meta.tags["kind"] == "tool_result" and meta.tags["tool"] == "read_file" and meta.tags["lane"] == "effect"
        assert meta.metadata["arguments"]["body"].startswith("notes/plan.md")
        assert meta.metadata["turn_id"] == "t-0003" and meta.metadata["run_id"] == run.run_id
        assert meta.metadata["created_at"] and meta.metadata["entity_id"] == home.entity_id
        assert meta.metadata["shown_is_view"] is True and meta.metadata["bytes"] == len(file_bytes)
        assert meta.content_type.startswith("text/plain")
        assert home.artifacts.get_metadata(search_id).metadata["shown_is_view"] is False

        # Both are the run's artifacts (what a run view lists), and the message the model reads names both.
        assert sorted(m.artifact_id for m in home.artifacts.list_by_run(run.run_id)) == sorted([search_id, read_id])
        assert out.result["results_message"].count("[artifact: ") == 2
    finally:
        home.close()


# ------------------------------------------------------------------ the home-direct lane: a personal-time tick


def test_personal_time_fetch_keeps_the_whole_page_as_a_home_artifact(tmp_path: Path, monkeypatch) -> None:
    import abstractcore.tools.common_tools as ct

    page = "".join(f"Paragraph {i}: the river bends twice before the dam; the beavers rebuild every spring. " for i in range(250))
    assert len(page) > 20_000

    def fake_fetch(**kwargs: Any) -> Dict[str, Any]:
        assert kwargs["method"] == "GET"
        return {"title": "The river bend", "content": page, "success": True}

    monkeypatch.setattr(ct, "fetch_url", fake_fetch)

    home_dir = _make_home(tmp_path, "wanderer")
    home = open_home(home_dir)
    try:
        llm = _ScriptedLLM([
            "I want to read that page.\n```tool name=fetch_url\nhttps://example.org/river\n```",
            "I read about the river bend; the beavers rebuild every spring.",
            "(unused)",
            "(unused)",
        ])
        session = ChatSession(
            home, llm, participants=[home.entity_id], phase="personal", enable_tools=True,
            context_window=20000, out=lambda s: None,
        )
        _reply, report = session.turn("Your own time begins.")

        assert report.tools == ["fetch_url"]
        detail = report.tool_details[0]
        artifact_id = detail["artifact_id"]
        assert artifact_id, "the fetch left an artifact"
        # The model saw a bounded, labeled view ending with the artifact line...
        assert "#TRUNCATION" in detail["result"] and detail["result"].endswith(f"[artifact: {artifact_id}]")
        # ...and the artifact is the whole page.
        kept = home.artifacts.load(artifact_id).content.decode("utf-8")
        assert kept.startswith("Fetched: https://example.org/river\nTitle: The river bend\n\n")
        # (the fetch trims surrounding whitespace, the only normalization it applies)
        assert page.strip() in kept, "the page is kept whole"
        meta = home.artifacts.get_metadata(artifact_id)
        assert meta.metadata["lane"] == "personal" and meta.metadata["tool"] == "fetch_url"
        assert meta.metadata["tick_id"] == meta.metadata["turn_id"] and meta.metadata["turn_id"]
        assert meta.metadata["run_id"] == f"chat-{session.session_id}"
        assert meta.metadata["arguments"]["body"] == "https://example.org/river"
        # The turn's verbatim (another artifact of the same home) names the page's artifact.
        others = [t for t in _home_artifact_texts(home_dir) if page.strip() not in t]
        assert any(f"[artifact: {artifact_id}]" in t for t in others), "the verbatim references the artifact"
    finally:
        home.close()


# ------------------------------------------------------------------ a lane without a store (the gateway today)


def test_without_an_artifact_store_nothing_is_stored_and_no_line_is_added() -> None:
    from abstractruntime.identity.tools import ToolElection, execute_tool_elections

    election = ToolElection(name="web_search", body="beaver dams", args={})
    message, notices = execute_tool_elections(
        [election], diary_store=None, diary_read_effect=lambda x: {}, web_search_fn=lambda q: "RESULT: wood",
    )
    assert election.result == "RESULT: wood" and election.artifact_id is None
    assert "[artifact: " not in message and notices == []
