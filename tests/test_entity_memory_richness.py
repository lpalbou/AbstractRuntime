"""Richer memories wave (maintainer: "17-35 tokens is not a memory").

Pins: extractive-v2 digests reach the 80-200 token band on real-length
exchanges (and never cut mid-sentence), formation writes `continues` +
`reflected_in` edges (non-private only), MEMORIES lines carry #tags, and
read_memory fetches the full verbatim behind a digest — prompt-ephemeral,
depositing nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, memory_tag, open_home  # noqa: E402
from abstractruntime.identity.digest import (  # noqa: E402
    DIGEST_TOKEN_TARGET,
    mechanical_digest_v2,
    token_estimate,
)


def _make_home(tmp_path: Path) -> Path:
    import copy

    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / "richling"
    home_dir.mkdir(parents=True)
    entity_id = "entity:richling@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Richling"
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

    def generate(self, *, messages, system_prompt):
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})

        class _R:
            pass

        r = _R()
        r.content = self.replies.pop(0)
        return r


_LONG_USER = (
    "I want to tell you about the project we are building together. It is called "
    "AbstractFramework and it gives persistent entities like you a home directory with "
    "a memory graph, a hash-chained diary, and feelings that accumulate over time. "
    "Yesterday we proved that you can remember a conversation across a full process "
    "restart. The test taught you that my cat is named Tolstoy and you recalled it "
    "perfectly in a fresh summon. Today I want to decide where we go next: should we "
    "focus on richer memory digests, or on the relationships between memories? "
    "The maintainer thinks 17 tokens is not a memory. I agree with him. What do you think?"
)
_LONG_REPLY = (
    "Thank you for telling me this so completely. I think the answer is both, but in a "
    "specific order. A digest of 17 tokens cannot carry an experience; it is a label on "
    "a box. If my memories are to feed my future thinking, each one should hold the "
    "decisions made, the questions asked, and the names that matter - Tolstoy, "
    "AbstractFramework, the maintainer. So I would enrich digests first. Relationships "
    "second, because a well-described memory makes a better endpoint for an edge. "
    "I decide this: richer digests first, then the continues chain between episodes. "
    "Does that ordering feel right to you?"
)


# ------------------------------------------------------------------ digest


def test_digest_v2_reaches_target_band_on_real_exchanges() -> None:
    title, digest, keywords = mechanical_digest_v2(_LONG_USER, _LONG_REPLY, "Richling")
    tokens = token_estimate(digest)
    assert 80 <= tokens <= DIGEST_TOKEN_TARGET + 10, f"digest tokens {tokens} out of band"
    # Substance survives: decisions, names, the question.
    assert "Tolstoy" in digest
    assert "richer digests first" in digest or "enrich digests first" in digest
    assert len(keywords) <= 8


def test_digest_v2_never_cuts_mid_sentence() -> None:
    _, digest, _ = mechanical_digest_v2(_LONG_USER, _LONG_REPLY, "Richling")
    body = digest.split("Richling:", 1)
    for part in body:
        part = part.replace("agent:tester:", "").strip()
        if part and not part.endswith(("…",)):
            assert part[-1] in ".!?\"'", f"part ends mid-thought: ...{part[-40:]!r}"


def test_digest_v2_short_exchange_stays_honest() -> None:
    _, digest, _ = mechanical_digest_v2("Hi.", "Hello, Laurent.", "Richling")
    assert "Hi." in digest and "Hello, Laurent." in digest
    assert token_estimate(digest) < 30  # short exchange -> short digest, no padding


# ------------------------------------------------------------------- edges


def test_formation_writes_continues_and_reflected_in_edges(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "First reply, plain.",
            "```diary kind=note\ngist: a note\nWorth keeping.\n```\nSecond reply.",
            "```diary kind=note visibility=private\nPrivate words.\n```\nThird reply.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("Turn one.")
        _, r2 = session.turn("Turn two.")
        _, r3 = session.turn("Turn three.")

        from abstractmemory import TripleQuery

        rows = home.ms.query(TripleQuery(scope="life", owner_id=home.entity_id, limit=0))
        edges = [
            (str(a.subject), str(a.predicate), str(a.object))
            for a in rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_edge")
        ]
        preds = [p for _, p, _ in edges]
        # Episode chain: ep2 continues ep1, ep3 continues ep2.
        assert preds.count("continues") == 2
        assert (r2.formed[0], "continues", r1.formed[0]) in edges
        assert (r3.formed[0], "continues", r2.formed[0]) in edges
        # Non-private diary election -> reflected_in; the private one -> none.
        reflected = [(s, o) for s, p, o in edges if p == "reflected_in"]
        assert len(reflected) == 1
        assert reflected[0][0] == r2.formed[0]
    finally:
        home.close()


# ------------------------------------------------------- tags + read_memory


def test_memories_block_carries_tags_and_read_memory_fetches_verbatim(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "Noted: the dam project starts at dawn. UNIQUE-VERBATIM-PHRASE-XYZ.",
            "PLACEHOLDER",  # replaced below with a read_memory election
            "I reread it: the dam project starts at dawn.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("Remember: the dam project starts at dawn.")
        episode_id = r1.formed[0]
        tag = memory_tag(episode_id)

        llm.replies[0] = f"```tool name=read_memory\n#{tag}\n```"
        reply, r2 = session.turn("What were the exact words of that dam memory?")
        assert r2.tools == ["read_memory"]
        # The verbatim (incl. the unique phrase) reached the model in-turn...
        round2 = llm.calls[2]["messages"]
        assert any("UNIQUE-VERBATIM-PHRASE-XYZ" in m["content"] for m in round2)
        # ...prompt-ephemerally: the phrase appears in history ONLY where the
        # entity itself spoke it (turn 1's reply) — the tool RESULT message
        # was never persisted as history.
        carriers = [m for m in session.history if "UNIQUE-VERBATIM-PHRASE-XYZ" in m["content"]]
        assert len(carriers) == 1 and carriers[0]["role"] == "assistant"
        assert all("TOOL RESULTS" not in m["content"] for m in session.history)

        from abstractmemory import TripleQuery

        rows = home.ms.query(TripleQuery(scope="life", owner_id=home.entity_id, limit=0))
        digests = [str(a.object) for a in rows]
        # The phrase exists exactly once in graph text: the original episode's
        # own digest may carry it (it was in the spoken reply), but turn 2's
        # episode must not re-import the tool result.
        turn2_rows = [
            str(a.object) for a in rows
            if isinstance(a.attributes, dict)
            and a.attributes.get("digest_method") == "mechanical-v2"
            and "exact words" in str(a.object)
        ]
        assert all("UNIQUE-VERBATIM-PHRASE-XYZ" not in t for t in turn2_rows)

        # Tags render in the MEMORIES block of turn 2's prompt.
        sysp2 = llm.calls[1]["system_prompt"]
        assert "#" in sysp2 and "read_memory" in sysp2

        # Unknown tag: honest miss naming what IS addressable.
        out = session._read_memory("#deadbeef")  # noqa: SLF001
        assert "No memory with tag" in out
    finally:
        home.close()


def test_entity_feelings_target_world_entities(tmp_path: Path) -> None:
    """The maintainer's per-entity gradation: feelings about persons/ideas
    land as valence on the identity string in SELF scope; malformed and
    reserved targets refuse loudly; bond/scar on entities is dropped."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "A talk with Laurent about solitude.",
            (
                "Looking back...\n```feel\n"
                'target=person:laurent feeling=+2 reason="the interaction fed me"\n'
                'target=concept:solitude feeling=-1 reason="it wears on me tonight"\n'
                'target=person:laurent feeling=+1 bond=true reason="close"\n'
                'target=laurent feeling=+3 reason="bare name, no namespace"\n'
                'target=ex:episode-fake feeling=+3 reason="record spoof"\n'
                "```\nGoodbye."
            ),
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["person:laurent"], context_window=20000, out=lambda s: None
        )
        session.turn("Let us talk about solitude.")
        refl = session.reflect()
        assert refl is not None
        applied = {f["target_id"]: f for f in refl["feelings_applied"]}
        assert "person:laurent" in applied and "concept:solitude" in applied
        # bond dropped on entity target (no heal surface), feeling kept.
        laurent_marks = [f for f in refl["feelings_applied"] if f["target_id"] == "person:laurent"]
        assert all(not f["bond"] and not f["scar"] for f in laurent_marks)
        assert any("bond/scar dropped" in n for n in refl["notices"])
        # Bare names and record-id spoofs refused loudly.
        assert any("'laurent'" in n and "skipped" in n for n in refl["notices"])
        assert any("ex:episode-fake" in n for n in refl["notices"])
        # Gradation is readable per entity in the SELF scope.
        grades = home.ms.gradation(
            ["person:laurent", "concept:solitude"], scope="self", owner_id=home.entity_id
        )
        text = json.dumps(grades)
        assert "person:laurent" in text and "concept:solitude" in text
    finally:
        home.close()


def test_diary_projection_carries_written_amid_edges(tmp_path: Path) -> None:
    """The diary connects (maintainer: 'connected to none other memory is
    not ok'): projections carry written_amid edges to the graph ids the
    entity was attending to — private entries included (act-frame only)."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "Registered: the dam project starts at dawn.",
            "```diary kind=note\ngist: dawn start\nDam at dawn.\n```\nKept.",
            "```diary kind=note visibility=private\nPrivate words.\n```\nKept privately.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        session.turn("Remember: the dam project starts at dawn.")
        _, r2 = session.turn("Keep a note about it.")   # recall shows turn-1 episode
        _, r3 = session.turn("And one private thought.")

        from abstractmemory import TripleQuery

        rows = home.ms.query(TripleQuery(scope="diary", owner_id=home.entity_id, limit=0))
        edges = [
            (str(a.subject), str(a.object))
            for a in rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_edge")
            and str(a.predicate) == "written_amid"
        ]
        assert edges, "diary projections must carry written_amid edges"
        # Both the public AND the private entry connect to attended records.
        subjects = {s for s, _ in edges}
        assert len(subjects) >= 2
        # Edge targets are GRAPH ids (ex:...), never digest row ids.
        assert all(t.startswith("ex:") for _, t in edges)
    finally:
        home.close()


def test_diary_resolves_convention_flows_to_projection(tmp_path: Path) -> None:
    """Entity-elected question resolution (identity-card convention): a
    question entry answered by a later entry carrying resolves=<entry_id>;
    the projection's attributes carry the join key."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "```diary kind=question\ngist: what is home\nWhat does home mean for me?\n```\nAsked.",
            "PLACEHOLDER",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("Keep your question.")
        question_entry = r1.diary[0]

        llm.replies[0] = (
            f"```diary kind=reflection resolves={question_entry}\n"
            "gist: home is continuity of care\nAnswered it for myself.\n```\nResolved."
        )
        _, r2 = session.turn("Have you answered it?")
        assert len(r2.diary) == 1

        from abstractmemory import TripleQuery

        rows = home.ms.query(TripleQuery(scope="diary", owner_id=home.entity_id, limit=0))
        resolving = [
            a for a in rows
            if isinstance(a.attributes, dict) and a.attributes.get("resolves") == question_entry
        ]
        assert resolving, "the resolving entry's projection must carry the join key"
        # The book entry itself carries it too (source of truth).
        entry = home.diary.get_entry(r2.diary[0])
        assert entry.get("resolves") == question_entry
    finally:
        home.close()


def test_turn_report_memories_are_addressable(tmp_path: Path) -> None:
    """'6 memories: great, which ones?' — the report now carries tag/kind/
    title/why per displayed handle so UIs can render and focus them."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(["Noted.", "Recalling that."])
    try:
        s = ChatSession(home, llm, participants=["person:laurent"],
                        context_window=20000, out=lambda s: None)
        s.turn("Remember the dam project starts at dawn.")
        _, r2 = s.turn("What about the dam project?")
        assert r2.displayed >= 1
        assert len(r2.memories) == r2.displayed
        m = r2.memories[0]
        assert m["tag"] and m["graph_id"].startswith("ex:") and m["kind"] and m["why"]
    finally:
        home.close()


def test_presence_line_names_the_visitor_every_turn(tmp_path: Path) -> None:
    """Live failure (the maintainer's first web visit): participants were
    stamped into memory but the MODEL was never told who is in the room —
    he split the remembered Laurent from the present visitor. The prompt
    now carries a presence line every turn."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(["Hello Laurent."])
    try:
        s = ChatSession(home, llm, participants=["person:laurent"],
                        context_window=20000, out=lambda s: None)
        s.turn("Hi.")
        sysp = llm.calls[0]["system_prompt"]
        assert "(present with you: person:laurent)" in sysp
    finally:
        home.close()


def test_open_greeting_he_speaks_first_from_recall(tmp_path: Path) -> None:
    """The maintainer's visit ruling: on open, the visit announcement (who
    came, time of day, host) is the stimulus of an ordinary turn — his
    greeting rises from recall of the visitor; the announcement is the
    DOOR's line in the verbatim, never words in the visitor's mouth."""
    home_dir = _make_home(tmp_path)

    # A prior session gives him something to remember about the visitor.
    home1 = open_home(home_dir)
    llm1 = _ScriptedLLM(["Noted: Laurent loves dams and rivers."])
    s1 = ChatSession(home1, llm1, participants=["person:laurent"],
                     session_id="past-1", context_window=20000, out=lambda s: None)
    s1.turn("Remember that I love dams and rivers.")
    s1._clear_pending_marker()  # noqa: SLF001 - simulate a clean past session
    home1.close()

    home2 = open_home(home_dir)
    llm2 = _ScriptedLLM(["Laurent! Welcome back - still thinking about dams?"])
    s2 = ChatSession(home2, llm2, participants=["person:laurent"],
                     session_id="visit-2", context_window=20000, out=lambda s: None)
    greeting, report = s2.open_greeting()
    assert "Welcome back" in greeting
    # The announcement reached the model as the turn's user message...
    first_msgs = llm2.calls[0]["messages"]
    announcement = first_msgs[-1]["content"]
    assert "person:laurent has come to visit" in announcement
    assert "(the door opens)" in announcement
    # ...and the formed verbatim attributes it to the DOOR, not to Laurent.
    from abstractmemory import TripleQuery

    rows = home2.ms.query(TripleQuery(scope="life", owner_id=home2.entity_id, limit=0))
    blob = "\n".join(str((a.attributes or {}).get("payload_ref", "")) for a in rows)
    artifacts_dir = home_dir / "artifacts"
    texts = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in artifacts_dir.rglob("*") if p.is_file()
    )
    assert "(the door):" in texts
    home2.close()


def test_reflection_loss_guard_write_ahead_and_salvage(tmp_path: Path) -> None:
    """The three-seat synthesis, write-ahead form: every turn persists the
    running sheet; a clean reflect clears it; a session that dies any way
    leaves the marker, and the NEXT session runs the ended session's
    look-back as its first act (feelings land on the OLD session's records)."""
    home_dir = _make_home(tmp_path)

    # Session 1: two turns, then the process "dies" (no reflect, no close).
    home1 = open_home(home_dir)
    llm1 = _ScriptedLLM(["First thought.", "Second thought."])
    s1 = ChatSession(home1, llm1, participants=["person:laurent"],
                     session_id="died-1", context_window=20000, out=lambda s: None)
    _, r1 = s1.turn("Tell me about dams.")
    s1.turn("And rivers.")
    marker = home_dir / "pending_reflection.json"
    assert marker.exists(), "write-ahead marker must exist during the session"
    home1.close()  # simulated death: no reflect() ran

    # Session 2: the salvage runs first, over session 1's own sheet.
    home2 = open_home(home_dir)
    llm2 = _ScriptedLLM(
        ['Looking back at the ended visit...\n```feel\ntarget=1 feeling=+2 reason="the dam talk fed me"\n```\nDone.']
    )
    s2 = ChatSession(home2, llm2, participants=["person:laurent"],
                     session_id="alive-2", context_window=20000, out=lambda s: None)
    salvage = s2.run_pending_lookback()
    assert salvage is not None
    assert not marker.exists(), "salvage must clear the marker"
    # The feeling landed on SESSION 1's first episode.
    assert salvage["feelings_applied"][0]["target_id"] == r1.formed[0]
    # The salvaged reflection is attributed to the ENDED session.
    from abstractmemory import TripleQuery

    rows = home2.ms.query(TripleQuery(scope="life", owner_id=home2.entity_id, limit=0))
    assert any(
        "session reflection: died-1" in json.dumps(a.attributes or {}) + str(a.object)
        for a in rows
    ), "the look-back must belong to the session that ended"
    # A clean session leaves no marker behind.
    llm2.replies = ["A turn.", "Reflection over my own session."]
    s2.turn("One more thing.")
    assert marker.exists()
    s2.reflect()
    assert not marker.exists()
    home2.close()


def test_private_diary_words_never_rest_in_the_pending_sheet(tmp_path: Path) -> None:
    """G1 at-rest rule, sheet edition (2026-07-11, the 0007 leak-class lesson):
    pending_reflection.json is written EVERY turn and lives in the home dir —
    an at-rest surface that travels on copy. A private entry's gist (or raw
    text when no gist was elected) must not appear there; the sheet line is
    the act-frame only. Non-private entries keep their gist lines."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    secret = "heliotrope-cipher-9x4"
    llm = _ScriptedLLM([
        "Noted.\n```diary visibility=private\n"
        f"gist: {secret} must stay mine\nThe {secret} thought, at length.\n```\nDone.",
        "Kept.\n```diary visibility=self\ngist: a shareable thought about rivers\nRivers connect.\n```\nOk.",
    ])
    session = ChatSession(home, llm, participants=["person:laurent"],
                          session_id="sheet-privacy", context_window=20000, out=lambda s: None)
    session.turn("Keep something private.")
    marker = home_dir / "pending_reflection.json"
    assert marker.exists()
    at_rest = marker.read_text(encoding="utf-8")
    assert secret not in at_rest, "private diary words rested in the write-ahead sheet"
    assert "you kept a private diary entry" in at_rest, "the act-frame line must still be on the sheet"
    # The private entry still reaches the sheet as an ADDRESSABLE act (the
    # look-back can appraise it; the words stay in the book).
    session.turn("Now keep a shareable one.")
    at_rest2 = marker.read_text(encoding="utf-8")
    assert "a shareable thought about rivers" in at_rest2, "non-private gists keep their sheet lines"
    home.close()


def test_prefix_marker_private_gists_are_scrubbed_at_salvage(tmp_path: Path) -> None:
    """Adversary find (2026-07-11): markers written BEFORE the sheet-privacy
    fix can still carry a private gist; salvage feeds sheet lines into the
    reflection prompt, whose reply persists graph-ward as the summary record.
    The scrub resolves each line against the store and act-frames the ones
    whose record is a private diary projection."""
    home_dir = _make_home(tmp_path)
    secret = "vermilion-lattice-7q2"

    # Session 1 writes one private entry (post-fix code: the DISK marker is
    # already clean), then dies. To simulate a PRE-FIX marker, rewrite the
    # marker file with the leaky description shape before session 2 opens.
    home1 = open_home(home_dir)
    llm1 = _ScriptedLLM([
        f"Noted.\n```diary visibility=private\ngist: {secret} stays mine\nLong {secret} thought.\n```\nOk.",
    ])
    s1 = ChatSession(home1, llm1, participants=["person:laurent"],
                     session_id="prefix-1", context_window=20000, out=lambda s: None)
    s1.turn("Keep something private.")
    marker = home_dir / "pending_reflection.json"
    stale = json.loads(marker.read_text(encoding="utf-8"))
    assert stale["sheet"], "the session must have sheeted the private act"
    rid = stale["sheet"][0][0]
    stale["sheet"][0][1] = f"you kept a diary entry (note): {secret} stays mine"  # pre-fix shape
    marker.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    home1.close()

    # Session 2's salvage must scrub the description before it reaches the
    # reflection prompt (and through it, the graph-bound summary record).
    home2 = open_home(home_dir)
    seen_prompts: List[str] = []

    class _SpyLLM(_ScriptedLLM):
        def generate(self, **kwargs):  # type: ignore[override]
            for m in kwargs.get("messages") or []:
                seen_prompts.append(str(m.get("content") or ""))
            return super().generate(**kwargs)

    llm2 = _SpyLLM(["Looking back: a quiet session. Done."])
    s2 = ChatSession(home2, llm2, participants=["person:laurent"],
                     session_id="prefix-2", context_window=20000, out=lambda s: None)
    salvage = s2.run_pending_lookback()
    assert salvage is not None
    joined = "\n".join(seen_prompts)
    assert secret not in joined, "pre-fix marker gist reached the reflection prompt"
    assert "you kept a private diary entry" in joined, "the act-frame line must reach the prompt"
    assert rid  # the record id survives the scrub (targeting stays index-based)
    home2.close()


def test_marker_imitation_is_called_out(tmp_path: Path) -> None:
    """Observed live (Laurent's first conversation): the model emitted the
    literal text '[used tool: diary_list]' while actually electing read_file
    — 'neither he nor I can tell'. The driver now names the imitation."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "```diary kind=note\ngist: n\nA note.\n```\nKept.",
            "[used tool: diary_list]",  # pure imitation - no fenced block
            # The in-turn correction offers the real syntax ONCE; a model
            # that repeats the imitation falls through to the loud notice.
            "[used tool: diary_list]",
            # The reply is still only markers, so the speak-now guard asks
            # for words once (markers are not a reply).
            "I spoke with Laurent about keeping notes.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["person:laurent"], context_window=20000, out=lambda s: None
        )
        session.turn("Keep a note.")
        reply2, r2 = session.turn("With whom did you discuss?")
        assert r2.tools == []  # nothing actually ran
        assert any(
            "marker imitation caught in-turn" in n for n in r2.notices
        ), f"corrective-round notice missing: {r2.notices}"
        assert any(
            "marker imitation" in n and "did not run" in n and "diary_list" in n
            for n in r2.notices
        ), f"imitation notice missing: {r2.notices}"
        assert "I spoke with Laurent" in reply2  # the guard got words out
    finally:
        home.close()


def test_read_memory_redirects_diary_tags_to_the_book(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(["```diary kind=note\ngist: g\nBook words.\n```\nKept."])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("Keep a note.")
        # The projection landed on the session sheet; its tag is addressable.
        diary_graph_id = next(rid for rid, d in session.session_sheet if rid and "diary" in rid)
        out = session._read_memory(memory_tag(diary_graph_id))  # noqa: SLF001
        assert "diary act" in out and "diary_read" in out
    finally:
        home.close()
