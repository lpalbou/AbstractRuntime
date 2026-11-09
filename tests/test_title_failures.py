"""A missing title is recorded, never guessed (operator ruling 2026-11-09: "we shouldn't need a fallback. if it fails,
it should add information to a temporary error file in untracked/ so we know what's failing and how to fix it at the
source").

When an elected block (interest, lesson, realization) carries no usable `title:` first line, or a public diary entry
carries no `gist:`, the record takes NO title from his words (AbstractMemory requires a title, so it carries the
visibly-untitled label "untitled <kind> [<key>]"; its words whole, as before) and ONE JSON line is appended to
`<workspace>/untracked/title-failures/<YYYY-MM-DD>.jsonl` (the workspace = the parent of the data dir that holds the
home). When no such folder resolves, the same line goes to the home's own-time log after a `#TITLE_FAILURE` marker.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.identity.reflection import parse_elected_title
from abstractruntime.identity.title_failures import untitled_label

FIELDS = {
    "observed_at", "entity", "record_id", "kind", "raw_block", "template_id", "template_version",
    "provider", "model", "reason",
}
GOOD = "Patience as a practice"


def test_reasons_are_structural_and_the_words_stay_whole() -> None:
    # missing: no `title:` line at all - no title, the words whole (never his first line as a guess).
    assert parse_elected_title("Dr. Smith taught me patience.\nIt matters.") == (
        None, "Dr. Smith taught me patience. It matters.", "missing")
    # empty: the label names nothing.
    assert parse_elected_title("title:\nThe words.") == (None, "The words.", "empty")
    # multi_line: two title lines - which one he meant is not ours to pick.
    assert parse_elected_title("title: One\ntitle: Two\nWords.") == (None, "One Two Words.", "multi_line")
    # malformed: the title line is not the first line.
    assert parse_elected_title("Words first.\ntitle: Late") == (None, "Words first. Late", "malformed")
    # too_long: past the bound the title is not a title; its words stay in the body.
    long_title = "x" * 201
    assert parse_elected_title(f"title: {long_title}\nRest.") == (None, f"{long_title} Rest.", "too_long")
    # A good title line: verbatim, out of the words, no reason.
    assert parse_elected_title(f"title: {GOOD}\nThe words.") == (GOOD, "The words.", None)
    assert parse_elected_title(f"title: {GOOD}") == (GOOD, GOOD, None)


def _workspace_home(tmp_path: Path) -> Path:
    """A home inside a workspace that holds `untracked/` (<ws>/data/entities/<slug>)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _make_home

    (tmp_path / "ws" / "untracked").mkdir(parents=True)
    return _make_home(tmp_path / "ws" / "data")


def _failure_lines(tmp_path: Path) -> List[Dict[str, Any]]:
    folder = tmp_path / "ws" / "untracked" / "title-failures"
    if not folder.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(folder.glob("*.jsonl")):
        out.extend(json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    return out


def _stored(home_dir: Path) -> Dict[str, List[tuple]]:
    from abstractmemory import SQLiteTripleStore, TripleQuery

    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    try:
        rows = list(store.query(TripleQuery(predicate="dcterms:abstract", limit=0)))
    finally:
        store.close()
    out: Dict[str, List[tuple]] = {}
    for a in rows:
        attrs = a.attributes if isinstance(a.attributes, dict) else {}
        out.setdefault(str(attrs.get("record_kind") or ""), []).append(
            (str(attrs.get("title") or ""), str(a.object), str(a.subject)))
    return out


def _session(home_dir: Path, replies: List[str]):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_lookback_offers import _Recorder

    from abstractruntime.identity.chat import ChatSession, open_home

    home = open_home(home_dir)
    s = ChatSession(home, _Recorder(replies), participants=["person:laurent"], context_window=20000,
                    out=lambda _s: None, phase="visit", model_info={"provider": "lmstudio", "model": "qwen-test"})
    return home, s


@pytest.mark.parametrize("site", ["lookback", "midturn"])
def test_an_untitled_lesson_is_stored_without_a_title_and_recorded(tmp_path: Path, site: str) -> None:
    pytest.importorskip("abstractmemory")
    home_dir = _workspace_home(tmp_path)
    block = "Dr. Smith taught me patience.\nIt matters to how I teach."
    reply = f"Looking back.\n\n```lesson\n{block}\n```\n\n```lesson\ntitle: {GOOD}\nWait before answering.\n```\nBye."
    home, s = _session(home_dir, [reply, "A quiet close."] if site == "midturn" else ["Hello.", reply])
    s.turn("hi there")
    if site == "lookback":
        s.reflect()
    home.close()
    lessons = _stored(home_dir)["lesson"]
    untitled = [r for r in lessons if r[1] == "Dr. Smith taught me patience. It matters to how I teach."]
    # No title from his words (AbstractMemory requires one, so the visibly-untitled label stands in: the kind and a
    # key of the block, none of his words); the words whole.
    assert len(untitled) == 1 and untitled[0][0] == untitled_label("lesson", block), lessons
    assert untitled[0][0].startswith("untitled lesson [") and "Smith" not in untitled[0][0]
    assert (GOOD, "Wait before answering.") in [(t, w) for t, w, _id in lessons]  # a good title: verbatim
    lines = _failure_lines(tmp_path)
    assert len(lines) == 1, lines  # the good title wrote nothing
    line = lines[0]
    assert set(line) >= FIELDS, sorted(line)
    assert line["kind"] == "lesson" and line["reason"] == "missing"
    assert line["raw_block"] == block  # the block he wrote, whole
    assert line["entity"] == "entity:liveling@home-test"
    assert line["record_id"] and line["record_id"] == untitled[0][2]
    assert line["provider"] == "lmstudio" and line["model"] == "qwen-test"
    assert line["template_id"] and line["template_version"]
    assert line["observed_at"].endswith("+00:00")
    assert line["stored_title"] == untitled[0][0]
    assert (tmp_path / "ws" / "untracked" / "title-failures" / f"{line['observed_at'][:10]}.jsonl").is_file()


def test_unresolvable_folder_writes_the_marker_to_the_own_time_log(tmp_path: Path) -> None:
    pytest.importorskip("abstractmemory")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _make_home

    home_dir = _make_home(tmp_path / "nowhere")  # no untracked/ beside the data dir
    home, s = _session(home_dir, ["Hello.", "Back.\n```interest\nwhy rivers bend\n```\n"])
    s.turn("hi")
    s.reflect()
    home.close()
    log = (home_dir / "own_time.log").read_text(encoding="utf-8")
    marked = [ln for ln in log.splitlines() if ln.startswith("#TITLE_FAILURE ")]
    assert len(marked) == 1, log
    line = json.loads(marked[0][len("#TITLE_FAILURE "):])
    assert set(line) >= FIELDS and line["kind"] == "interest" and line["reason"] == "missing"
    assert line["raw_block"] == "why rivers bend"
    assert not list(tmp_path.rglob("title-failures"))


def _project(home_dir: Path, gist: Any, text: str) -> str:
    from abstractruntime.integrations.abstractmemory.identity_support import project_diary_entry

    captured: Dict[str, Any] = {}

    class _MS:
        def remember_many(self, records, **kwargs):
            captured["records"] = list(records)
            return ["ex:rec-1"]

    project_diary_entry(
        _MS(), entity_id="entity:x", entry_id="diary_q1abcdef", kind="question", visibility="normal", gist=gist,
        text=text, written_at="2026-11-08T10:00:00+00:00", turn_id="t-1", origin={}, home_dir=home_dir,
    )
    return str(captured["records"][0].title)


def test_a_diary_entry_without_a_gist_takes_no_title_from_its_text(tmp_path: Path) -> None:
    home_dir = tmp_path / "ws" / "data" / "entities" / "x"
    home_dir.mkdir(parents=True)
    (tmp_path / "ws" / "untracked").mkdir()
    title = _project(home_dir, None, "Why do rivers bend?\nMore prose.")
    assert "rivers" not in title and title.startswith("Diary entry (question) — 2026-11-08"), title
    lines = _failure_lines(tmp_path)
    assert len(lines) == 1 and lines[0]["kind"] == "diary" and lines[0]["reason"] == "missing"
    assert lines[0]["raw_block"] == "Why do rivers bend?\nMore prose." and lines[0]["record_id"] == "ex:rec-1"
    # His gist is his title: no failure.
    assert _project(home_dir, "Why rivers bend", "prose").endswith(": Why rivers bend")
    assert len(_failure_lines(tmp_path)) == 1


def test_a_placeholder_is_marked_and_never_rendered_as_his_title(tmp_path: Path) -> None:
    """Codex review 2026-11-09 #9 (interim until memory stores a missing title as None): the placeholder record says
    so in its attributes (title_missing, title_failure_reason), and the recall shelf shows no title for it."""
    pytest.importorskip("abstractmemory")
    from abstractmemory import SQLiteTripleStore, TripleQuery

    from abstractruntime.identity.chat import _memories_block
    from abstractruntime.identity.title_failures import is_placeholder_title

    home_dir = _workspace_home(tmp_path)
    block = "why rivers bend"
    home, s = _session(home_dir, ["Hello.", f"Back.\n```interest\n{block}\n```\n"])
    s.turn("hi")
    s.reflect()
    home.close()
    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    try:
        rows = [a for a in store.query(TripleQuery(predicate="dcterms:abstract", limit=0))
                if (a.attributes or {}).get("record_kind") == "interest"]
    finally:
        store.close()
    attrs = rows[-1].attributes
    assert attrs["title"] == untitled_label("interest", block) and is_placeholder_title(attrs["title"])
    assert attrs["title_missing"] is True and attrs["title_failure_reason"] == "missing"
    shelf = _memories_block([
        {"kind": "interest", "title": untitled_label("interest", block), "digest": block, "admission": "stimulus"},
        {"kind": "lesson", "title": GOOD, "digest": "Wait before answering.", "admission": "stimulus"},
    ], 7)
    assert "untitled interest" not in shelf and f"{block} (" in shelf
    assert f"{GOOD}: Wait before answering." in shelf
    assert not is_placeholder_title("untitled thoughts [abc123]") and not is_placeholder_title("Untitled song")
