"""The model look-back offers interests, lessons, questions and problems at
every visit close, every own-time day close and the end of every
personal-time tick (operator 2026-11-08: "the end visit is the way to
reflect on the discussion!!!! that's the root cause issue!!! and also at
every tick in personal time, the same thing should happen!").

The 2026-07-21 change (fc2fc84) made the own-time day close a zero-LLM
summary and removed these offers from the look-back prompt; the mid-turn
offer that replaced them produced 0 records in 17 homes. These tests pin
that the offers are back in the prompt and that a model answer carrying the
blocks forms the records through the same parsers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("abstractmemory")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).parent))
from test_entity_life_loop import _make_home  # noqa: E402

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.life import LifeLoop  # noqa: E402
from abstractruntime.identity.reflection import build_reflection_prompt  # noqa: E402

LOOKBACK_REPLY = """Looking back, the tide question stayed with me.

```interest
how tides keep time without a clock
```

```lesson
Reading the source first saves a day of guessing.
```

```diary kind=question
Why does the evening feel shorter than the morning?
```

Goodbye for now."""


class _Recorder:
    """A model double: replies in order, records every prompt it was sent."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.prompts: List[str] = []

    def generate(self, *, messages, system_prompt):
        self.prompts.append(str(messages[-1]["content"]))

        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0) if self.replies else "A quiet thought."
        return r


def _kinds(home_dir: Path) -> Dict[str, List[str]]:
    """record_kind -> digests, and the open questions, read from the store."""
    from abstractmemory import SQLiteJournal, SQLiteTripleStore, TripleQuery
    from abstractmemory.diary import open_questions

    import json

    entity_id = json.loads((home_dir / "manifest.json").read_text())["entity_id"]
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    try:
        out: Dict[str, List[str]] = {}
        for a in store.query(TripleQuery(predicate="dcterms:abstract", limit=0)):
            attrs = a.attributes if isinstance(a.attributes, dict) else {}
            out.setdefault(str(attrs.get("record_kind") or ""), []).append(str(a.object))
        questions: List[str] = []
        for scope in ("self", "diary", "life"):
            for a in open_questions(store, scope=scope, owner_id=entity_id, journal=journal, limit=0):
                questions.append(str(a.object))
        out["open_question"] = questions
        return out
    finally:
        store.close()
        journal.close()


def _has(texts: List[str], words: str) -> bool:
    return any(words in t for t in texts)


# ------------------------------------------------------------------ prompt


@pytest.mark.parametrize("moment", ["session", "tick"])
def test_lookback_prompt_offers_interest_lesson_question_problem(moment: str) -> None:
    prompt = build_reflection_prompt(["1. a thing happened"], moment=moment)
    assert "```feel" in prompt
    assert "```interest" in prompt and "up to two" in prompt
    assert "```lesson" in prompt
    assert "```diary kind=question" in prompt and "kind=problem" in prompt
    assert "IN THE MOMENT" not in prompt and "no longer asks" not in prompt


def test_tick_prompt_is_framed_for_the_tick() -> None:
    tick = build_reflection_prompt(["1. x"], moment="tick")
    session = build_reflection_prompt(["1. x"])
    assert tick.startswith("This stretch of your own time is done.")
    assert "goodbye" not in tick, "a tick is not a goodbye; his time continues"
    assert session.startswith("The session is ending.")
    assert "goodbye" in session
    with pytest.raises(ValueError):
        build_reflection_prompt(["1. x"], moment="day")


# ----------------------------------------------------------- the three sites


def test_visit_close_lookback_forms_the_offered_records(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _Recorder(["Hello, good to see you.", LOOKBACK_REPLY])
    home = open_home(home_dir)
    s = ChatSession(home, llm, participants=["person:laurent"], context_window=20000,
                    out=lambda _s: None, phase="visit")
    s.turn("hi there")
    result = s.reflect()
    home.close()
    assert result is not None
    assert len(llm.prompts) == 2 and "```interest" in llm.prompts[1]
    assert result["interests"] and result["lessons"] and result["diary_entries"] == 1
    kinds = _kinds(home_dir)
    assert _has(kinds.get("interest", []), "how tides keep time")
    assert _has(kinds.get("lesson", []), "Reading the source first")
    assert _has(kinds["open_question"], "evening feel shorter"), kinds["open_question"]


def test_own_time_day_close_runs_one_model_lookback(tmp_path: Path) -> None:
    """The own-time close is no longer the zero-LLM mechanical summary: one
    model call, and its blocks reach the parsers."""
    home_dir = _make_home(tmp_path)
    llm = _Recorder(["A morning thought.", LOOKBACK_REPLY])
    home = open_home(home_dir)
    s = ChatSession(home, llm, participants=[home.entity_id], context_window=20000,
                    out=lambda _s: None, phase="personal")
    s.turn("(cue) morning")
    before = len(llm.prompts)
    result = s.reflect()
    home.close()
    assert len(llm.prompts) == before + 1, "exactly one look-back call at the day close"
    assert llm.prompts[-1].startswith("The session is ending.")
    assert result is not None and result["session_record_id"]
    assert not any("mechanical close" in n for n in result["notices"])
    kinds = _kinds(home_dir)
    assert _has(kinds.get("interest", []), "how tides keep time")
    assert _has(kinds.get("lesson", []), "Reading the source first")
    assert _has(kinds["open_question"], "evening feel shorter")


def _personal_factory(home_dir: Path, llm: Any):
    def _open() -> ChatSession:
        home = open_home(home_dir)
        return ChatSession(home, llm, participants=[home.entity_id], context_window=20000,
                           enable_workspace=True, out=lambda _s: None, phase="personal")

    return _open


def test_every_personal_tick_ends_with_a_lookback(tmp_path: Path) -> None:
    """Two personal ticks then the day close: tick, look-back, tick,
    look-back, close look-back. The tick look-back's blocks form records."""
    home_dir = _make_home(tmp_path)
    tick_lookback = LOOKBACK_REPLY.replace("Goodbye for now.", "")
    llm = _Recorder(["First tick.", tick_lookback, "Second tick.", "Nothing to keep.",
                     "The day, seen whole: nothing more."])
    lines: List[str] = []
    loop = LifeLoop(_personal_factory(home_dir, llm), tick_seconds=0, ticks_per_day=2,
                    max_ticks=2, state_home=home_dir, sleep_fn=lambda _s: None,
                    out=lines.append)
    report = loop.run()
    assert report.ticks == 2
    starts = [p.split("\n", 1)[0] for p in llm.prompts]
    tick_prompts = [i for i, p in enumerate(llm.prompts)
                    if p.startswith("This stretch of your own time is done.")]
    assert tick_prompts == [1, 3], starts
    assert llm.prompts[4].startswith("The session is ending."), starts
    assert not any("look-back failed" in ln for ln in lines), lines
    kinds = _kinds(home_dir)
    assert _has(kinds.get("interest", []), "how tides keep time")
    assert _has(kinds["open_question"], "evening feel shorter")
    # Each tick's look-back covers only that tick's records.
    assert "First tick" not in llm.prompts[3]


def test_rest_tick_leaves_the_lookback_to_the_day_close(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _Recorder(["Enough for today.\n```rest\nquiet\n```", LOOKBACK_REPLY])
    loop = LifeLoop(_personal_factory(home_dir, llm), tick_seconds=0, ticks_per_day=4,
                    max_ticks=1, state_home=home_dir, sleep_fn=lambda _s: None,
                    out=lambda _s: None)
    loop.run()
    assert len(llm.prompts) == 2
    assert llm.prompts[1].startswith("The session is ending."), "one look-back, at the close"


def test_work_ticks_have_no_tick_lookback(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    llm = _Recorder(["Working.", "Still working.", "Close."])

    def _open() -> ChatSession:
        home = open_home(home_dir)
        return ChatSession(home, llm, participants=[home.entity_id], context_window=20000,
                           out=lambda _s: None, phase="work")

    loop = LifeLoop(_open, tick_seconds=0, ticks_per_day=2, max_ticks=2,
                    sleep_fn=lambda _s: None, out=lambda _s: None)
    loop.run()
    assert not any(p.startswith("This stretch of your own time is done.") for p in llm.prompts)
