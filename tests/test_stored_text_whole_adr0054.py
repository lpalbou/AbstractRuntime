"""Root ADR 0054: stored text is never truncated (operator, 2026-11-08: "the text stored should never be
truncated, otherwise it's not a verbatim; the text shown is UI/app dependent").

Every test here is RED against the fixed-length cut it replaces (a 1,000-character input stored at 200, 80 or
120 characters, or an attention item cut at 120 / 280 / 2,000) and GREEN now that the write paths store the
whole text. A screen or a prompt may still fold; the one prompt fold kept here (the day-open cue) is VISIBLE
(" …" at a word boundary), never a silent character cut.
"""
from __future__ import annotations

import json
from typing import Any, Dict

# ~1,250 characters, one line, no sentence end: the digest's sentence selector finds nothing that fits a
# budget, the old fallbacks cut it at 200 / 80 / 120 characters mid-word.
LONG_LINE = " ".join(f"word{i:03d}" for i in range(150))
assert len(LONG_LINE) > 1000


# -------------------------------------------------------------- identity/digest.py (stored digest)


def test_digest_fallback_carries_the_whole_side_not_200_characters() -> None:
    from abstractruntime.identity.digest import mechanical_digest_v2

    _title, digest, _kw = mechanical_digest_v2(LONG_LINE, "Noted.", "nerin", speaker="person:t")
    assert LONG_LINE in digest, "no whole sentence fit the user budget: the side rides the stored digest whole"
    _title, digest, _kw = mechanical_digest_v2("Hello there.", LONG_LINE, "nerin", speaker="person:t")
    assert LONG_LINE in digest, "same for the reply side"


# -------------------------------------------------------------- identity/reflection.py (the marked, stored reply)


def test_interest_marker_carries_the_words_whole() -> None:
    from abstractruntime.identity.reflection import parse_interest_blocks

    marked, interests, notices = parse_interest_blocks("I noticed something.\n```interest\n" + LONG_LINE + "\n```")
    assert interests == [LONG_LINE]
    assert f'[kept interest: "{LONG_LINE}"]' in marked
    assert notices == []


def test_lesson_marker_carries_the_words_whole() -> None:
    from abstractruntime.identity.reflection import parse_lesson_blocks

    marked, lessons, notices = parse_lesson_blocks("```lesson\n" + LONG_LINE + "\n```")
    assert lessons == [LONG_LINE]
    assert f'[learned: "{LONG_LINE}"]' in marked
    assert notices == []


def test_topic_marker_carries_the_words_whole() -> None:
    from abstractruntime.identity.reflection import MAX_TOPIC_WORDS, parse_topic_blocks

    topic = " ".join("x" * 30 for _ in range(MAX_TOPIC_WORDS))  # under the word cap, past the old 80 characters
    assert len(topic) > 80
    marked, topics, _notices = parse_topic_blocks("```topic\n" + topic + "\n```")
    assert topics == [topic]
    assert f'[topic: "{topic}"]' in marked


# -------------------------------------------------------------- identity/act_only.py (capture meta: run vars + sheet)


def test_diary_capture_gist_is_the_whole_first_line() -> None:
    from abstractruntime.identity.act_only import capture_diary_elections

    class _Out:
        status = "completed"
        result = {"entry_id": "diary_0000aaaa", "projected_record_id": "ex:p1", "warnings": []}

    class _Run:
        run_id = "run-whole"
        vars: Dict[str, Any] = {}

    _marked, entries, _warnings = capture_diary_elections(
        "kept.\n```diary kind=note\n" + LONG_LINE + "\nA second line stays in the book.\n```",
        run=_Run(), turn_id="t-whole", diary_write_handler=lambda run, effect, dnn: _Out(),
    )
    assert len(entries) == 1
    assert entries[0]["gist"] == LONG_LINE, "the first LINE whole - a line is structure, 120 characters was a cut"
    assert json.dumps(entries[0])  # still JSON-safe for run vars


# -------------------------------------------------------------- identity/life.py (the day-open cue: a visible prompt fold)


def test_day_open_cue_folds_a_long_question_visibly_never_silently(tmp_path) -> None:
    from abstractruntime.identity.diary import DiaryEntry, DiaryStore
    from abstractruntime.identity.life import DRIVE_GIST_MAX_WORDS, standing_state_note
    from abstractruntime.storage.sqlite import SqliteDatabase, SqliteLedgerStore

    (tmp_path / "manifest.json").write_text(json.dumps({"entity_id": "entity:desk@t"}), encoding="utf-8")
    db = SqliteDatabase(str(tmp_path / "home.sqlite3"))
    diary = DiaryStore(entity_id="entity:desk@t", ledger_store=SqliteLedgerStore(db))
    diary.append_entry(DiaryEntry(
        entry_id="diary_q1", author="entity:desk@t", kind="question", text=LONG_LINE, gist=None,
        visibility="self", written_at="2026-11-08T10:00:00Z",
    ))
    db.close()

    offer = standing_state_note(tmp_path, rotation_key=0)
    words = LONG_LINE.split()
    assert '"' + " ".join(words[:DRIVE_GIST_MAX_WORDS]) + ' …"' in offer, offer
    assert LONG_LINE[:120] + '"' not in offer, "the old silent 120-character cut"
    assert "diary_read diary_q1" in offer, "the reread command opens the whole entry"


# -------------------------------------------------------------- automations (notify title/body, attention item)


def test_notify_title_and_body_are_stored_whole() -> None:
    from abstractruntime.automations.attention import notify_payload
    from abstractruntime.automations.models import validate_notify

    long_body = LONG_LINE + " " + LONG_LINE  # past the old 2,000-character body cut
    assert len(long_body) > 2000
    stored = validate_notify({"always": True, "title": LONG_LINE, "body": long_body})
    assert stored["title"] == LONG_LINE and stored["body"] == long_body
    assert notify_payload(True, title=LONG_LINE, answer=long_body) == {"title": LONG_LINE, "body": long_body}
    assert notify_payload({"title": LONG_LINE, "body": long_body}, title="t", answer="") == {
        "title": LONG_LINE, "body": long_body,
    }
