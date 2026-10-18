"""Keyword search over a plane's conversations (conversation_search.ConversationSearch).

Runs are saved in the real run stores (json files and SQLite) the way chat turns are, then
searched: exact words and substrings in questions and answers, accents and case ignored, a typo
still found (trigram fuzzy pass), one hit per conversation, incremental reindexing of a changed
conversation, the time budget reported, and the caller's visible set honoured.
"""

from __future__ import annotations

import pytest

from abstractruntime import RunState, RunStatus
from abstractruntime.conversation_search import ConversationSearch, normalize
from abstractruntime.storage.in_memory import InMemoryLedgerStore
from abstractruntime.storage.json_files import JsonFileRunStore
from abstractruntime.storage.sqlite import SqliteDatabase, SqliteRunStore


def make_store(kind, tmp_path):
    if kind == "json":
        return JsonFileRunStore(tmp_path / "runs")
    return SqliteRunStore(SqliteDatabase(tmp_path / "runs.sqlite"))


def save(store, run_id, minute, session, prompt, answer):
    ts = f"2026-10-18T10:{minute:02d}:00+00:00"
    store.save(RunState(run_id=run_id, workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                        vars={"prompt": prompt, "context": {"messages": []}}, output={"answer": answer},
                        session_id=session, created_at=ts, updated_at=ts))


def populate(store):
    save(store, "r1", 1, "s-km", "Compare overall survival by treatment arm in the ADTTE", "Arm B: HR 0.78 (Kaplan–Meier).")
    save(store, "r2", 2, "s-km", "Now split it by age group", "Under 65: HR 0.71; over 65: HR 0.86.")
    save(store, "r3", 3, "s-qc", "Did the nightly assay QC pass?", "Yes: all 96 wells within range, CV 3.1 %.")
    save(store, "r4", 4, "s-fr", "Quelle est la météo à Paris ?", "Il fait beau, 21 °C.")
    save(store, "r5", 5, "s-ai", "What is AI safety?", "A field about making AI systems behave.")


@pytest.fixture(params=["json", "sqlite"])
def index(request, tmp_path):
    store = make_store(request.param, tmp_path)
    populate(store)
    idx = ConversationSearch(tmp_path / "search.sqlite3", run_store=store, ledger_store=InMemoryLedgerStore())
    report = idx.sync()
    assert report.complete and report.total_turns == 5 and report.reindexed_sessions == 4
    return idx, store


def test_a_word_in_a_question_finds_its_conversation(index):
    idx, _ = index
    hits = idx.search("survival")
    assert [h.session_id for h in hits] == ["s-km"]
    assert hits[0].matched_in == "question" and hits[0].run_id == "r1"
    start, length = hits[0].highlights[0]
    assert hits[0].snippet[start:start + length].lower() == "survival"


def test_an_answer_and_a_substring_match(index):
    idx, _ = index
    hits = idx.search("wells")
    assert [h.session_id for h in hits] == ["s-qc"] and hits[0].matched_in == "answer"
    assert [h.session_id for h in idx.search("kaplan")] == ["s-km"]            # inside "Kaplan–Meier"
    assert [h.session_id for h in idx.search("surviv")] == ["s-km"]            # a word's start


def test_every_word_must_match_and_case_and_accents_do_not_matter(index):
    idx, _ = index
    assert [h.session_id for h in idx.search("AGE group")] == ["s-km"]
    assert idx.search("age wells") == []                                       # not one turn has both
    assert [h.session_id for h in idx.search("meteo paris")] == ["s-fr"]       # météo, without the accent
    assert normalize("Météo À  Paris") == "meteo a paris"


def test_short_words_match_too(index):
    idx, _ = index
    assert [h.session_id for h in idx.search("AI")] == ["s-ai"]


def test_a_typo_still_finds_its_conversation(index):
    idx, _ = index
    hits = idx.search("survivl")
    assert hits and hits[0].session_id == "s-km" and hits[0].fuzzy
    assert idx.search("zzzz qqqq") == []


def test_one_hit_per_conversation_its_best_turn(index):
    idx, _ = index
    hits = idx.search("HR")
    assert [h.session_id for h in hits] == ["s-km"]


def test_a_changed_conversation_is_reindexed(index):
    idx, store = index
    save(store, "r6", 6, "s-qc", "And the plate P07?", "P07 was re-read: control CV 9 %.")
    report = idx.sync()
    assert report.reindexed_sessions == 1 and report.total_turns == 6
    assert [h.session_id for h in idx.search("P07")] == ["s-qc"]
    assert idx.sync().reindexed_sessions == 0                                   # nothing changed since


def test_the_time_budget_is_reported(tmp_path):
    store = make_store("sqlite", tmp_path)
    populate(store)
    idx = ConversationSearch(tmp_path / "search.sqlite3", run_store=store, ledger_store=InMemoryLedgerStore())
    first = idx.sync(budget_s=0)
    assert not first.complete and first.pending_sessions == 4
    assert idx.sync().complete


def test_only_the_callers_visible_sessions(index):
    idx, _ = index
    assert idx.search("survival", sessions={"s-qc"}) == []
    assert [h.session_id for h in idx.search("survival", sessions={"s-km"})] == ["s-km"]


# --- regressions found in review (2026-10-18) -------------------------------------------------------

def fresh(tmp_path, runs):
    store = make_store("sqlite", tmp_path)
    for i, (rid, sid, p, a) in enumerate(runs):
        save(store, rid, i % 60, sid, p, a)
    idx = ConversationSearch(tmp_path / "search.sqlite3", run_store=store, ledger_store=InMemoryLedgerStore())
    assert idx.sync(budget_s=60).complete
    return idx, store


def test_a_filter_never_empties_a_page_with_matches_further_down(tmp_path):
    runs = [(f"r{i}", f"s{i}", f"budget question {i}", "an answer") for i in range(120)]
    idx, _ = fresh(tmp_path, runs + [("rz", "s-arch", "budget question archived", "x")])
    assert [h.session_id for h in idx.search("budget", limit=5, sessions={"s-arch"})] == ["s-arch"]
    assert "s-arch" not in [h.session_id for h in idx.search("budget", limit=200, exclude={"s-arch"})]
    assert len(idx.search("budget", limit=50)) == 50


def test_an_exact_match_always_comes_before_a_fuzzy_one(tmp_path):
    idx, _ = fresh(tmp_path, [("a", "s-fuzzy", "surviving the vivid rival survivor", "x"),
                              ("b", "s-exact", "overall survival by arm", "x")])
    hits = idx.search("survival", limit=5)
    assert hits[0].session_id == "s-exact" and not hits[0].fuzzy


def test_highlights_survive_an_ellipsis_and_folded_letters(tmp_path):
    long_answer = "Intro words. " * 30 + "The Kaplan curve shows it. " + "More words follow. " * 20
    idx, _ = fresh(tmp_path, [("a", "s1", "show me", long_answer), ("b", "s2", "Straße und Größe", "x"),
                              ("c", "s3", "👩‍💻 survival notes", "x")])
    h = idx.search("curve")[0]
    assert h.snippet.startswith("…") and h.snippet.endswith("…")
    start, n = h.highlights[0]
    assert h.snippet[start:start + n].lower() == "curve"
    g = idx.search("strasse")[0]
    start, n = g.highlights[0]
    assert g.snippet[start:start + n] == "Straße"                 # ß → ss, marked on the source
    e = idx.search("survival")[0]
    start, n = e.highlights[0]
    assert e.snippet[start:start + n] == "survival"               # offsets in Unicode scalars, after an emoji


def test_a_typo_marks_the_word_it_resembles(tmp_path):
    idx, _ = fresh(tmp_path, [("a", "s1", "overall survival by arm", "x")])
    h = idx.search("survivl")[0]
    assert h.fuzzy and h.matched_in == "question"
    start, n = h.highlights[0]
    assert h.snippet[start:start + n] == "survival"


def test_short_words_are_found_beyond_the_first_rows(tmp_path):
    runs = [(f"r{i}", f"s{i}", "il fait beau", "oui") for i in range(600)] + [("rz", "s-ai", "is ai safe", "maybe")]
    idx, _ = fresh(tmp_path, runs)
    assert [h.session_id for h in idx.search("ai")] == ["s-ai"]


def test_scripts_without_spaces_and_with_vowel_signs(tmp_path):
    idx, _ = fresh(tmp_path, [("a", "s-ja", "東京に行きました", "x"), ("b", "s-hi", "दुनिया का सबसे ऊँचा पर्वत", "x")])
    assert [h.session_id for h in idx.search("東京")] == ["s-ja"]
    assert [h.session_id for h in idx.search("दुनिया")] == ["s-hi"]


def test_a_deleted_run_leaves_the_index(tmp_path):
    idx, store = fresh(tmp_path, [("a", "s1", "secret hunter2 question", "x"), ("b", "s2", "other", "y")])
    assert idx.search("hunter2")
    store.delete_run("a") if hasattr(store, "delete_run") else store.delete("a")
    idx.sync()
    assert idx.search("hunter2") == []


def test_automation_sessions_are_not_indexed(tmp_path):
    store = make_store("sqlite", tmp_path)
    ts = "2026-10-18T10:00:00+00:00"
    store.save(RunState(run_id="occ", workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                        vars={"prompt": "nightly tick", "context": {"messages": []},
                              "_meta": {"occurrence": {"automation_id": "a1", "occurrence_index": 1, "attempt": 1, "role": "occurrence",
                                                       "session_kind": "automation"}}},
                        output={"answer": "mem 41%"}, session_id="s-auto", parent_run_id="a1", created_at=ts, updated_at=ts))
    idx = ConversationSearch(tmp_path / "search.sqlite3", run_store=store, ledger_store=InMemoryLedgerStore())
    idx.sync()
    assert idx.search("nightly") == []


# --- artifacts: what a turn made or was given --------------------------------------------------------

from abstractruntime.storage.artifacts import FileArtifactStore, build_artifact_descriptor_payload  # noqa: E402


def media_plane(tmp_path):
    store = make_store("sqlite", tmp_path)
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    ts = "2026-10-18T10:{:02d}:00+00:00"

    def turn(rid, minute, sid, prompt, answer, attachments=None, parent=None):
        ctx = {"messages": [], **({"attachments": attachments} if attachments else {})}
        store.save(RunState(run_id=rid, workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                            vars={"prompt": prompt, "context": ctx}, output={"answer": answer},
                            session_id=sid, parent_run_id=parent, created_at=ts.format(minute), updated_at=ts.format(minute)))

    def generated(aid, rid, sid, modality, prompt, purpose=None):
        descriptor, metadata = build_artifact_descriptor_payload(semantic_kind=modality, render_kind=modality, modality=modality,
                                                                  session_id=sid, run_id=rid, generation={"prompt": prompt})
        tags = {"kind": "generated_media", "modality": modality, "run_id": rid, "session_id": sid}
        if purpose:
            tags["purpose"] = purpose
        artifacts.store(b"x" + aid.encode(), content_type="image/png" if modality == "image" else "audio/wav", run_id=rid,
                        artifact_id=aid, tags=tags, metadata=metadata, descriptor=descriptor)

    turn("t1", 1, "s-img", "Draw something for the kitchen", "Here is your picture.")
    generated("img1", "t1", "s-img", "image", "A realistic red apple resting on a wooden table")
    turn("t2", 2, "s-img", "Now a pear", "Here it is.")
    turn("child", 3, "s-img", "", "", parent="t2")                         # the agent's own run, under t2
    generated("img2", "child", "s-img", "image", "A green pear in soft morning light")
    turns_attachment = artifacts.store(b"%PDF", content_type="application/pdf", run_id="session_memory_s-doc",
                                       tags={"kind": "attachment", "session_id": "s-doc"}, metadata={"filename": "budget-2026.pdf"})
    turn("t3", 4, "s-doc", "Summarise this", "It is a budget.", attachments=[{"$artifact": turns_attachment.artifact_id, "filename": "budget-2026.pdf"}])
    generated("tts", "t3", "s-doc", "voice", "Spoken dragonfly words", purpose="voice_playback")   # read aloud: not media
    artifacts.store(b"RIFF", content_type="audio/wav", run_id="session_memory_s-doc",       # a voice question's upload
                    tags={"kind": "attachment", "session_id": "s-doc"}, metadata={"filename": "ask.wav"})
    idx = ConversationSearch(tmp_path / "search.sqlite3", run_store=store, ledger_store=InMemoryLedgerStore(), artifact_store=artifacts)
    assert idx.sync().complete
    return idx


def test_a_generated_image_is_found_by_its_prompt(tmp_path):
    idx = media_plane(tmp_path)
    hit = idx.search("apple table")[0]
    assert hit.session_id == "s-img" and hit.run_id == "t1" and hit.matched_in == "artifact"
    assert hit.artifact["artifact_id"] == "img1" and hit.artifact["kind"] == "image" and hit.artifact["run_id"] == "t1"
    start, n = hit.highlights[0]
    assert hit.snippet[start:start + n] == "apple"


def test_media_made_in_an_agents_run_belongs_to_its_turn(tmp_path):
    idx = media_plane(tmp_path)
    hit = idx.search("pear morning")[0]
    assert hit.run_id == "t2" and hit.artifact["artifact_id"] == "img2" and hit.artifact["run_id"] == "child"


def test_an_attachment_is_found_by_its_file_name(tmp_path):
    idx = media_plane(tmp_path)
    hit = idx.search("budget-2026")[0]
    assert hit.session_id == "s-doc" and hit.matched_in == "artifact" and hit.artifact["name"] == "budget-2026.pdf"


def test_read_aloud_audio_and_voice_question_uploads_are_not_media(tmp_path):
    idx = media_plane(tmp_path)
    assert idx.search("dragonfly") == []                                      # read-aloud audio is not media
    assert idx.search("ask.wav") == []                                        # nor a voice question's upload
    assert idx.search("budget")[0].artifact["artifact_id"] != "tts"


def test_a_question_match_comes_before_an_artifact_match(tmp_path):
    idx = media_plane(tmp_path)
    assert idx.search("kitchen")[0].matched_in == "question"
    assert idx.search("image apple")[0].artifact["artifact_id"] == "img1"      # its kind is searchable too


def test_media_without_text_to_find_is_not_indexed(tmp_path):
    store = make_store("sqlite", tmp_path)
    artifacts = FileArtifactStore(tmp_path / "artifacts")
    ts = "2026-10-18T10:01:00+00:00"
    store.save(RunState(run_id="t1", workflow_id="wf", status=RunStatus.COMPLETED, current_node="n",
                        vars={"prompt": "Say hello", "context": {"messages": []}}, output={"answer": "Hello."},
                        session_id="s1", created_at=ts, updated_at=ts))
    descriptor, metadata = build_artifact_descriptor_payload(semantic_kind="voice", render_kind="audio", modality="voice", session_id="s1", run_id="t1")
    artifacts.store(b"RIFF", content_type="audio/wav", run_id="t1", artifact_id="old-tts",                  # no prompt kept
                    tags={"kind": "generated_media", "modality": "voice", "run_id": "t1", "session_id": "s1"},
                    metadata={**metadata, "filename": "tts.wav"}, descriptor=descriptor)
    idx = ConversationSearch(tmp_path / "search.sqlite3", run_store=store, ledger_store=InMemoryLedgerStore(), artifact_store=artifacts)
    idx.sync()
    assert idx.search("voice") == [] and idx.search("tts") == []
