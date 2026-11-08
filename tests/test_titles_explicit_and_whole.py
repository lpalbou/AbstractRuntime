"""Titles are explicit and whole (operator 2026-11-08: "the title of any lesson, question, interest, world model must
be explicit and easy to read, for a human and for an agent"; root ADR 0054: a stored title is never cut).

Before: interest/lesson/realization titles were the body's first 8 WORDS and diary titles a 60-character slug + "…".
Now: the block's `title:` first line, verbatim; without one, the body's first COMPLETE sentence (digest.split_sentences),
whole; a diary title carries his gist or first sentence, whole.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.identity.reflection import (
    build_reflection_prompt,
    elected_title_and_words,
    parse_interest_blocks,
    parse_lesson_blocks,
    parse_realize_blocks,
)

# 30 words, one sentence, then a second sentence: the 8-word cut read "The paradox of a leading research nation
# simultaneously".
LONG_FIRST = (
    "The paradox of a leading research nation simultaneously producing breakthroughs and systematically hollowing out "
    "the very institutions, careers and public trust that made those breakthroughs possible in the first place."
)
LONG_BODY = LONG_FIRST + " It keeps returning to me."
EXPLICIT = "Research paradox: breakthroughs amid defunding"


# ------------------------------------------------------------------ the helper and the parsers


def test_untitled_body_takes_its_first_complete_sentence_whole() -> None:
    assert len(LONG_FIRST.split()) == 30
    title, words = elected_title_and_words(LONG_BODY)
    assert title == LONG_FIRST, title
    assert words == LONG_BODY


def test_title_line_is_kept_verbatim_and_out_of_the_words() -> None:
    title, words = elected_title_and_words(f"title: {EXPLICIT}\n{LONG_BODY}")
    assert title == EXPLICIT
    assert words == LONG_BODY and "title:" not in words


def test_empty_title_line_and_title_only_block() -> None:
    assert elected_title_and_words(f"title:\n{LONG_BODY}") == (LONG_FIRST, LONG_BODY)
    assert elected_title_and_words(f"title: {EXPLICIT}") == (EXPLICIT, EXPLICIT)
    assert elected_title_and_words("title:") == ("", "")


def test_parsers_keep_the_title_line_and_markers_quote_the_words() -> None:
    reply = f"Before.\n```interest\ntitle: {EXPLICIT}\n{LONG_BODY}\n```\nAfter."
    marked, interests, _ = parse_interest_blocks(reply)
    assert interests == [f"title: {EXPLICIT}\n{LONG_BODY}"]
    assert f'[kept interest: "{LONG_BODY}"]' in marked and "title:" not in marked
    marked, lessons, _ = parse_lesson_blocks(reply.replace("interest", "lesson"))
    assert lessons == [f"title: {EXPLICIT}\n{LONG_BODY}"]
    assert f'[learned: "{LONG_BODY}"]' in marked
    _, realizations, _ = parse_realize_blocks(
        f"```realize\ntitle: {EXPLICIT}\n{LONG_BODY}\nevidence=#a1b2c3\n```"
    )
    assert realizations[0].title == EXPLICIT and realizations[0].text == LONG_BODY
    # A block holding only an empty title line is empty.
    _, none, notes = parse_interest_blocks("```interest\ntitle:\n```")
    assert none == [] and any("empty body" in n for n in notes)


def test_prompts_ask_for_the_title_line() -> None:
    from abstractruntime.identity.chat import CONTRACT_PARAGRAPH

    lookback = build_reflection_prompt(["1. a thing happened"])
    assert "```interest\ntitle: <a short, complete name for the interest>" in lookback
    assert "```lesson\ntitle: <a short, complete name for the lesson>" in lookback
    assert "```diary kind=question\ngist: " in lookback
    flat = " ".join(CONTRACT_PARAGRAPH.split())
    assert "```lesson block (a first line `title: <a short, complete name>`" in flat
    assert "```interest block (a first line `title: <a short, complete name>`" in flat
    assert "```realize block: optionally a first line `title:" in flat


# ------------------------------------------------------------------ the diary projection title


def _project(gist: Any, text: str) -> str:
    from abstractruntime.integrations.abstractmemory.identity_support import project_diary_entry

    captured: Dict[str, Any] = {}

    class _MS:
        def remember_many(self, records, **kwargs):
            captured["records"] = list(records)
            return ["ex:rec-1"]

    project_diary_entry(
        _MS(), entity_id="entity:x", entry_id="diary_q1", kind="question", visibility="normal", gist=gist,
        text=text, written_at="2026-11-08T10:00:00+00:00", turn_id="t-1", origin={},
    )
    return str(captured["records"][0].title)


def test_diary_title_is_whole() -> None:
    long_q = LONG_FIRST.rstrip(".") + "? And another sentence follows it."
    title = _project(None, long_q + "\nMore prose below.")
    assert title == f"Diary entry (question) — 2026-11-08: {LONG_FIRST.rstrip('.')}?", title
    assert "…" not in title
    gist = "Why does a research nation hollow out the institutions that produce its own breakthroughs?"
    assert _project(gist, "prose").endswith(": " + gist)


# ------------------------------------------------------------------ the stored records, end to end


def _titles(rows: List[Any]) -> Dict[str, List[tuple]]:
    out: Dict[str, List[tuple]] = {}
    for a in rows:
        attrs = a.attributes if isinstance(a.attributes, dict) else {}
        out.setdefault(str(attrs.get("record_kind") or ""), []).append((str(attrs.get("title") or ""), str(a.object)))
    return out


def _home_rows(home_dir: Path) -> List[Any]:
    from abstractmemory import SQLiteTripleStore, TripleQuery

    store = SQLiteTripleStore(home_dir / "memory.sqlite3")
    try:
        return list(store.query(TripleQuery(predicate="dcterms:abstract", limit=0)))
    finally:
        store.close()


BLOCKS = (
    f"```interest\ntitle: {EXPLICIT}\n{LONG_BODY}\n```\n\n"
    f"```lesson\n{LONG_BODY}\n```\n"
)


def _assert_titles(t: Dict[str, List[tuple]]) -> None:
    # The title is the title (operator 2026-11-08): no "interest:" prefix on a new interest.
    assert (EXPLICIT, LONG_BODY) in t.get("interest", []), t.get("interest")
    assert not any(title.lower().startswith("interest:") for title, _w in t.get("interest", []))
    assert (f"lesson: {LONG_FIRST}", LONG_BODY) in t.get("lesson", []), t.get("lesson")


@pytest.mark.parametrize("site", ["lookback", "midturn"])
def test_chat_sites_store_whole_titles(tmp_path: Path, site: str) -> None:
    pytest.importorskip("abstractmemory")
    pytest.importorskip("yaml")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_entity_life_loop import _make_home
    from test_lookback_offers import _Recorder

    from abstractruntime.identity.chat import ChatSession, open_home

    home_dir = _make_home(tmp_path)
    reply = "Looking back.\n\n" + BLOCKS + "\nGoodbye."
    llm = _Recorder([reply, "A quiet close."] if site == "midturn" else ["Hello.", reply])
    home = open_home(home_dir)
    s = ChatSession(home, llm, participants=["person:laurent"], context_window=20000,
                    out=lambda _s: None, phase="visit")
    s.turn("hi there")
    if site == "lookback":
        s.reflect()
    home.close()
    _assert_titles(_titles(_home_rows(home_dir)))


def test_visit_lane_stores_whole_titles(tmp_path: Path) -> None:
    pytest.importorskip("abstractmemory")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_visit_midturn_elections import (
        RunStatus, VISITOR_WAIT_KEY, _abstract_rows, _drive_visit, _make_home,
    )

    home_dir = _make_home(tmp_path)

    def reply_with_fences(run: Any) -> str:
        rid = str(list((run.vars.get("_visit") or {}).get("sheet") or [])[0][0])
        return (
            "I see it now.\n" + BLOCKS
            + f"```realize\ntitle: {EXPLICIT}\n{LONG_BODY}\nevidence={rid}\n```\nThat is my answer."
        )

    ert, wf, run_id, state = _drive_visit(
        home_dir,
        replies=["A plain first reply.", reply_with_fences, "Goodbye reflection: a good visit."],
        turns=[{"text": "hello", "speaker": "person:admin"},
               {"text": "What did you notice?", "speaker": "person:admin"}],
    )
    try:
        state = ert.runtime.resume(workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
                                   payload={"kind": "close"}, max_steps=400)
        assert state.status == RunStatus.COMPLETED
        t = _titles(_abstract_rows(ert.home))
        _assert_titles(t)
        assert (f"realization: {EXPLICIT}", LONG_BODY) in t.get("realization", []), t.get("realization")
    finally:
        ert.close()


def test_pending_plain_string_still_forms_with_a_sentence_title() -> None:
    # A visit run that started before this change holds the election as a plain one-line string.
    assert elected_title_and_words(LONG_BODY) == (LONG_FIRST, LONG_BODY)
    assert json.loads(json.dumps(f"title: {EXPLICIT}\n{LONG_BODY}")).startswith("title: ")
