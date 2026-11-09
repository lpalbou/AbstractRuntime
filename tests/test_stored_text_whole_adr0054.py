"""Root ADR 0054: stored text is never truncated (operator, 2026-11-08: "the text stored should never be
truncated, otherwise it's not a verbatim; the text shown is UI/app dependent").

Every test here is RED against the fixed-length cut it replaces (a 1,000-character input stored at 200, 80 or
120 characters, or an attention item cut at 120 / 280 / 2,000) and GREEN now that the write paths store the
whole text. A screen or a prompt may still fold; the one prompt fold kept here (the day-open cue) is VISIBLE
(" …" at a word boundary), never a silent character cut.
"""
from __future__ import annotations

from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402
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
        result = {"entry_id": "diary_0000aaaa", "projected_record_id": "ex:p1", "warnings": [], "title": None}

    class _Run:
        run_id = "run-whole"
        vars: Dict[str, Any] = {}

    _marked, entries, _warnings = capture_diary_elections(
        "kept.\n```diary kind=note\n" + LONG_LINE + "\nA second line stays in the book.\n```",
        run=_Run(), turn_id="t-whole", diary_write_handler=lambda run, effect, dnn: _Out(),
    )
    assert len(entries) == 1
    # Codex review 3 #9: the stored meta names the record's title as formed (None while it is still being written),
    # never his first line standing in - nothing of the entry is cut or copied into run vars.
    assert entries[0]["title"] is None and "gist" not in entries[0]
    assert LONG_LINE not in json.dumps(entries[0])  # still JSON-safe for run vars


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


# -------------------------------------------------------------- round 2 (2026-11-08): no caps anywhere on stored text

LONG_TEXT = ("The river bends twice before the dam and the beavers rebuild every spring. " * 40).strip()
assert len(LONG_TEXT) > 2000


def test_automation_title_has_no_length_cap() -> None:
    from datetime import datetime, timezone

    from abstractruntime.automations.models import validate_title
    from abstractruntime.automations.wakes import validate_schedule_run

    assert validate_title("  " + LONG_LINE + "  ") == LONG_LINE
    now = datetime.now(timezone.utc).isoformat()
    assert validate_schedule_run({"after_s": 60, "note": "x", "title": LONG_LINE}, now=now)["title"] == LONG_LINE
    # The note / task text had a 20,000-character refusal; it is stored whole too (it is the prompt).
    long_note = LONG_TEXT * 10
    assert len(long_note) > 20_000
    assert validate_schedule_run({"after_s": 60, "note": long_note}, now=now)["note"] == long_note


def test_publish_title_body_and_subject_have_no_length_cap() -> None:
    from abstractruntime.core.publish import validate_publish

    out = validate_publish({"channel": "kitchen", "title": LONG_LINE, "body": LONG_TEXT, "subject": LONG_LINE})
    assert out["title"] == LONG_LINE and out["body"] == LONG_TEXT and out["subject"] == LONG_LINE


def test_mechanical_title_is_the_chosen_sentence_whole() -> None:
    from abstractruntime.identity.digest import mechanical_digest_v2

    sentence = (
        "The river bends twice before the old dam, and the beavers rebuild it every spring with willow, "
        "mud and stones carried from the north bank where the wooden bridge once stood."
    )
    assert len(sentence.split()) > 14
    title, _digest, _kw = mechanical_digest_v2(sentence, "Noted.", "nerin", speaker="person:t")
    assert title == f"exchange: {sentence}"
    assert "…" not in title


def test_feel_marker_carries_the_sheet_words_and_the_reason_whole() -> None:
    from abstractruntime.identity.reflection import parse_feel_blocks

    desc = " ".join(f"part{i}" for i in range(60))
    assert len(desc) > 300
    reply = "```feel\n" + f'target=1 feeling=+2 reason="{LONG_LINE}"\n' + "```"
    marked, elections, _notices = parse_feel_blocks(reply, [f"1. {desc}"])
    assert len(elections) == 1
    assert f'[felt: about "{desc}" +2 - "{LONG_LINE}"]' in marked


def test_realize_marker_carries_the_realization_whole() -> None:
    from abstractruntime.identity.reflection import parse_realize_blocks

    marked, realizations, _notices = parse_realize_blocks("```realize\n" + LONG_LINE + "\nevidence=#tag1\n```")
    assert len(realizations) == 1 and realizations[0].text == LONG_LINE
    assert f'[realized: "{LONG_LINE}" - held for sleep]' in marked


def test_fallback_notices_quote_the_skipped_line_whole() -> None:
    from abstractruntime.identity.reflection import parse_feel_blocks, parse_topic_blocks

    bad_feel = "this is not a feel line " + LONG_LINE
    _m, _e, notices = parse_feel_blocks("```feel\n" + bad_feel + "\n```")
    assert any(bad_feel in n for n in notices), notices
    bad_topic = "diary_" + LONG_LINE.replace(" ", "")
    _m, _t, notices = parse_topic_blocks("```topic\n" + bad_topic + "\n```")
    assert any(bad_topic in n for n in notices), notices


def test_loop_status_day_cause_detail_and_next_cue_are_whole(tmp_path) -> None:
    from abstractruntime.identity.life import parse_next_cue, read_loop_status, write_loop_status

    write_loop_status(tmp_path, "day", day_kind="personal", day_cause={"cause": "drives", "detail": LONG_LINE}, fence=LifecycleFence.admit(tmp_path))
    assert read_loop_status(tmp_path)["day_cause"]["detail"] == LONG_LINE
    assert parse_next_cue("Done for now.\nnext: " + LONG_LINE) == LONG_LINE


def test_failed_occurrence_attention_body_is_the_whole_error(tmp_path, monkeypatch) -> None:
    from automation_harness import TARGETS, Clock, create, drive, make_stores
    from abstractruntime import Runtime, StepPlan, WorkflowSpec
    from abstractruntime.automations import list_attention, register_controller_bundle
    from abstractruntime.scheduler.registry import WorkflowRegistry

    error = "E: " + LONG_TEXT  # past the old 2,000-character cut

    def _fail(run, ctx):
        return StepPlan(node_id="answer", complete_output={"success": False, "error": error})

    registry = WorkflowRegistry()
    for spec in TARGETS.values():
        registry.register(spec)
    registry.register(WorkflowSpec(workflow_id="longfail", entry_node="answer", nodes={"answer": _fail}))
    register_controller_bundle(registry)
    run_store, ledger_store = make_stores("json", tmp_path)
    runtime = Runtime(run_store=run_store, ledger_store=ledger_store, workflow_registry=registry)
    clock = Clock(monkeypatch)
    aid = create(runtime, clock, workflow_id="longfail", input_data={"prompt": "p"}, retry={"max_attempts": 1})
    drive(runtime, aid)
    (item,) = list_attention(runtime.ledger_store, aid)["items"]
    assert item["kind"] == "failure" and item["body"] == error


def test_chat_sheet_line_and_tool_arg_are_whole(tmp_path) -> None:
    from abstractruntime.identity.chat import ChatSession, open_home
    from test_tool_results_are_artifacts import _ScriptedLLM, _make_home

    query = " ".join(f"topic{i}" for i in range(60))
    assert len(query) > 300
    home = open_home(_make_home(tmp_path, "sheetling"), fence=None)
    try:
        llm = _ScriptedLLM([
            "Let me look.\n```tool name=web_search\n" + query + "\n```",
            "```diary kind=note\n" + LONG_LINE + "\nA second line stays in the book.\n```\nKept.",
            "(unused)",
        ])
        session = ChatSession(
            home, llm, participants=["agent:t"], context_window=20000,
            web_search_fn=lambda q: "RESULT: wood", out=lambda s: None, fence=None,
        )
        _reply, report = session.turn("Tell me about the dam.")
        assert report.tool_details[0]["arg"] == query, "the probe's tool argument is whole"
        # Codex review 3 #9: the sheet names the diary record by its title - here still being written (no summaries
        # model in this session) - never his first line standing in.
        assert any(line == "you kept a diary entry (note) whose title is still being written"
                   for _rid, line in session.session_sheet), session.session_sheet
    finally:
        home.close()
