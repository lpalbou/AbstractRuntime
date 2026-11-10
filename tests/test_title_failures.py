"""A missing title is never guessed (operator rulings 2026-11-09: "we shouldn't need a fallback. if it fails, it
should add information to a temporary error file in untracked/"; "the title + summary must ALWAYS exist").

The title comes from the block's `title:` first line, else the summaries capability model authors it in the
formation step (tests/test_memory_authoring.py). When that step fails - here: no capability model in the session -
the record forms PENDING (title None, memory's marker and title.author task) and ONE JSON line is appended to
`<workspace>/untracked/authoring-failures/<YYYY-MM-DD>.jsonl` (the workspace = the parent of the data dir that holds
the home); when no such folder resolves, the same line goes to the home's own-time log after `#AUTHORING_FAILURE`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractruntime.identity.reflection import parse_elected_title

FIELDS = {
    "observed_at", "entity", "record_id", "kind", "raw_block", "template_id", "template_version",
    "provider", "model", "reason", "missing",
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
    folder = tmp_path / "ws" / "untracked" / "authoring-failures"
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

    home = open_home(home_dir, fence=None)
    s = ChatSession(home, _Recorder(replies), participants=["person:laurent"], context_window=20000,
                    out=lambda _s: None, phase="visit", model_info={"provider": "lmstudio", "model": "qwen-test"}, fence=None)
    return home, s


@pytest.mark.parametrize("site", ["lookback", "midturn"])
def test_an_untitled_lesson_whose_authoring_fails_is_pending_and_recorded(tmp_path: Path, site: str) -> None:
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
    # No title from his words, no placeholder: the title is pending (None); the words whole.
    assert len(untitled) == 1 and untitled[0][0] == "", lessons
    assert (GOOD, "Wait before answering.") in [(t, w) for t, w, _id in lessons]  # a good title: verbatim
    lines = [ln for ln in _failure_lines(tmp_path) if ln["kind"] == "lesson"]
    assert len(lines) == 1, lines  # the good title wrote nothing
    line = lines[0]
    assert set(line) >= FIELDS, sorted(line)
    assert line["missing"] == "title" and line["reason"] == "unavailable" and line["block_reason"] == "missing"
    assert line["raw_block"] == block  # the block he wrote, whole
    assert line["entity"] == "entity:liveling@home-test"
    assert line["record_id"] and line["record_id"] == untitled[0][2]
    assert line["template_id"] and line["template_version"]
    assert line["observed_at"].endswith("+00:00")
    assert (tmp_path / "ws" / "untracked" / "authoring-failures" / f"{line['observed_at'][:10]}.jsonl").is_file()


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
    marked = [json.loads(ln[len("#AUTHORING_FAILURE "):]) for ln in log.splitlines()
              if ln.startswith("#AUTHORING_FAILURE ")]
    interest = [ln for ln in marked if ln["kind"] == "interest"]
    assert len(interest) == 1, log
    line = interest[0]
    assert set(line) >= FIELDS and line["missing"] == "title" and line["block_reason"] == "missing"
    assert line["raw_block"] == "why rivers bend"
    assert not list(tmp_path.rglob("authoring-failures"))


def _project(gist: Any, text: str) -> Any:
    from abstractruntime.integrations.abstractmemory.identity_support import project_diary_entry

    captured: Dict[str, Any] = {}

    class _MS:
        def remember_many(self, records, **kwargs):
            captured["records"] = list(records)
            return ["ex:rec-1"]

    project_diary_entry(
        _MS(), entity_id="entity:x", entry_id="diary_q1abcdef", kind="question", visibility="normal", gist=gist,
        text=text, written_at="2026-11-08T10:00:00+00:00", turn_id="t-1", origin={},
    )
    return captured["records"][0]


def test_a_diary_entry_without_a_gist_takes_no_title_from_its_text() -> None:
    record = _project(None, "Why do rivers bend?\nMore prose.")
    # No authored metadata reached the projection (this lane authored nothing): pending, never his first line as a
    # title, never a template standing in (Codex review 2 #10).
    # The summary too is still to be authored (Codex review 3 #9): pending, his first line only its PROVISIONAL words.
    assert record.title in (None, "") and tuple(record.pending) == ("title", "summary")
    assert record.digest == "Why do rivers bend?"
    gisted = _project("Why rivers bend", "prose")
    assert gisted.title == "Why rivers bend" and tuple(gisted.pending) == ("summary",), "his gist, whole, no template"


def test_a_legacy_placeholder_title_is_never_rendered_as_his() -> None:
    """The 8e72975 placeholder ("untitled <kind> [<hex>]", retired once memory stored a pending title as None) may
    still stand on records formed that afternoon: the recall shelf shows no title for it (the sleep repairs it)."""
    from abstractruntime.identity.chat import _memories_block
    from abstractruntime.identity.title_failures import is_placeholder_title

    shelf = _memories_block([
        {"kind": "interest", "title": "untitled interest [3fa2c1]", "digest": "why rivers bend", "admission": "stimulus"},
        {"kind": "lesson", "title": GOOD, "digest": "Wait before answering.", "admission": "stimulus"},
    ], 7)
    assert "untitled interest" not in shelf and "why rivers bend (" in shelf
    assert f"{GOOD}: Wait before answering." in shelf
    assert is_placeholder_title("untitled lesson [abc123]")
    assert not is_placeholder_title("untitled thoughts [abc123]") and not is_placeholder_title("Untitled song")


def test_memorys_birth_labels_are_placeholders_never_words() -> None:
    """Memory's birth labels for an unnamed spark item (`trait-N`, `value-N`, `purpose-N`, `limit-N`; castor's two
    identity traits are both "trait-0", 2026-07-07) are template labels, not his words: exact formats only."""
    from abstractruntime.identity.title_failures import is_placeholder_title

    for label in ("trait-0", "trait-12", "value-3", "purpose-0", "limit-1"):
        assert is_placeholder_title(label), label
    for words in ("trait-0 of patience", "Trait-0", "trait-", "trait-x", "my trait-0", "values-1", "trait 0",
                  "Ask before assuming", "limit-1a"):
        assert not is_placeholder_title(words), words


def test_the_sleep_repairs_birth_labelled_identity_records(tmp_path: Path) -> None:
    """The sleep's repair declares a birth-labelled identity record pending and the drainer authors its title (castor's
    shape: two `trait` records titled "trait-0", one an honesty limit)."""
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.authoring import declare_missing_metadata, drain_authoring
    from abstractruntime.identity.chat import open_home

    sys.path.insert(0, str(Path(__file__).parent))
    from test_memory_authoring import TITLE, _Capability, _rows

    home_dir = _workspace_home(tmp_path)
    home = open_home(home_dir, fence=None)
    try:
        eid = home.entity_id
        seeded = []
        for key, digest, attrs in (
                ("t0", "Ask before assuming; verify before asserting.", {"precedence": 0, "spark_version": 1}),
                ("l0", "Report failures and degradations as they happened.", {"precedence": 0, "spark_version": 1,
                                                                             "trait_class": "limit"})):
            seeded += home.ms.remember_many([MemoryRecordInput(kind="trait", title="trait-0", digest=digest,
                                                               attributes=attrs)],
                                            scope="self", owner_id=eid, idempotency_key=f"legacy-birth-{key}")
        receipt = declare_missing_metadata(home.ms, scopes=[("self", eid)])
        assert receipt["declared"] >= 2 and receipt["refused"] == 0, receipt
        drained = drain_authoring(home.ms, llm_for_capability=lambda name: _Capability(), artifacts=home.artifacts,
                                  home_dir=home_dir, entity=eid, holder="test", cap=8, window="nap")
    finally:
        home.close()
    assert drained["authored"] >= 2, drained
    traits = [r for r in _rows(home_dir, "trait") if str(r.subject) in {str(g) for g in seeded}]
    assert len(traits) >= 2 and all(r.attributes.get("title") == TITLE for r in traits), \
        [r.attributes.get("title") for r in traits]
