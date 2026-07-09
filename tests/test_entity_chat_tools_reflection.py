"""Offline contract tests: tier-1 tool blocks + session-end reflection.

The night mandate (a2a 0007): Castor gets read-only tools (his diary, web
search) and a session-end look-back where feelings move through
MEMORY_APPRAISE on the entity-reflection channel. These tests prove both
loops with a scripted model and an injected search function — no network, no
LMStudio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.identity.chat import ChatSession, open_home  # noqa: E402
from abstractruntime.identity.reflection import parse_feel_blocks  # noqa: E402
from abstractruntime.identity.tools import parse_tool_blocks  # noqa: E402


class _ScriptedLLM:
    """Replies in order; records every prompt it was given."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, *, messages: List[Dict[str, str]], system_prompt: str) -> Any:
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})
        content = self.replies.pop(0)

        class _R:
            pass

        r = _R()
        r.content = content
        return r


def _make_home(tmp_path: Path) -> Path:
    """A minimal real home: engrammed spark + manifest (the gateway's shape,
    same recipe as test_entity_chat_driver)."""
    import copy

    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / "testling"
    home_dir.mkdir(parents=True)
    entity_id = "entity:testling@home-test"
    spark = copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = "Testling"
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")

    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    result = engram(ms, spark, owner_id=entity_id)
    assert result.created is True
    store.close()
    journal.close()
    return home_dir


# ------------------------------------------------------------------ parsing


def test_parse_tool_blocks_extracts_and_marks() -> None:
    reply = (
        "Let me check.\n\n```tool name=web_search\nwhat is a beaver dam\n```\nBack soon."
    )
    marked, elections, notices = parse_tool_blocks(reply)
    assert len(elections) == 1
    assert elections[0].name == "web_search"
    assert elections[0].body == "what is a beaver dam"
    assert "[used tool: web_search]" in marked
    assert "```tool" not in marked
    assert notices == []


def test_parse_tool_blocks_refuses_unknown_tools() -> None:
    reply = "```tool name=delete_files\neverything\n```"
    marked, elections, notices = parse_tool_blocks(reply)
    assert elections == []
    assert "not available" in marked
    assert any("refused" in n for n in notices)


def test_parse_feel_blocks_grammar_and_clamps() -> None:
    reply = (
        "Looking back...\n\n```feel\n"
        'target=1 feeling=+2 reason="it fed me"\n'
        'target=session feeling=-5 reason="too heavy" scar=true\n'
        'target=2 feeling=+1 bond=true\n'  # missing reason -> skipped
        "```\nGoodbye."
    )
    marked, elections, notices = parse_feel_blocks(reply)
    assert len(elections) == 2
    assert elections[0].target_token == "1" and elections[0].sign == 1 and elections[0].magnitude == 2
    # -5 clamps to routine band 3, scar honors the negative sign
    assert elections[1].target_token == "session" and elections[1].sign == -1
    assert elections[1].magnitude == 3.0 and elections[1].scar is True
    assert any("clamped" in n for n in notices)
    assert any("missing reason" in n for n in notices)
    assert "[marked 2 feelings]" in marked


def test_parse_feel_blocks_drops_mismatched_marks() -> None:
    reply = '```feel\ntarget=1 feeling=+2 reason="good" scar=true\n```'
    _, elections, notices = parse_feel_blocks(reply)
    assert len(elections) == 1
    assert elections[0].scar is False  # scar with + is dropped, feeling kept
    assert any("scar=true dropped" in n for n in notices)


# ------------------------------------------------------------------- session


def test_tool_round_web_search_stays_prompt_ephemeral(tmp_path: Path) -> None:
    """The entity looks something up; the results reach its second call and
    are NEVER persisted (history, verbatim, transcript)."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    searches: List[str] = []

    def fake_search(query: str) -> str:
        searches.append(query)
        return "RESULT: beavers build dams from wood. SECRET-MARKER-XYZ"

    llm = _ScriptedLLM(
        [
            "I should check.\n```tool name=web_search\nbeaver dams\n```",
            "I looked it up: beavers build dams from wood.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000,
            web_search_fn=fake_search, out=lambda s: None,
        )
        reply, report = session.turn("What do beavers build?")

        assert searches == ["beaver dams"]
        assert report.tools == ["web_search"]
        assert reply == "I looked it up: beavers build dams from wood."
        # Second call saw the results message in-turn...
        second_msgs = llm.calls[1]["messages"]
        assert any("TOOL RESULTS" in m["content"] for m in second_msgs)
        assert any("SECRET-MARKER-XYZ" in m["content"] for m in second_msgs)
        # ...but nothing persisted the raw results.
        assert all("SECRET-MARKER-XYZ" not in m["content"] for m in session.history)
        from abstractmemory import TripleQuery

        rows = home.ms.query(TripleQuery(scope="life", owner_id=home.entity_id, limit=0))
        blob = json.dumps([str(a.object) + json.dumps(a.attributes or {}) for a in rows])
        assert "SECRET-MARKER-XYZ" not in blob
        assert any(
            (a.attributes or {}).get("tools_used") == ["web_search"]
            for a in rows
            if isinstance(a.attributes, dict)
        )
        # The verbatim (artifact store) keeps the honest lookup marker and
        # NEVER the raw results.
        artifacts = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore")
            for p in (home_dir / "artifacts").rglob("*")
            if p.is_file()
        )
        assert "[used tool: web_search]" in artifacts
        assert "SECRET-MARKER-XYZ" not in artifacts
    finally:
        home.close()


def test_diary_list_and_read_tools(tmp_path: Path) -> None:
    """diary_list shows ids+gists; diary_read fetches the words through the
    designed DIARY_READ path (progressive disclosure)."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            # turn 1: keep a diary entry
            "```diary kind=note\ngist: the maintainer likes honest tools\nToday I learned tools must be honest.\n```\nKept.",
            # turn 2: list then (after results) read it back — scripted as two rounds
            "```tool name=diary_list\n5\n```",
            "I see my entry about honest tools. Its id is in the list.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("Please keep a note about honest tools.")
        assert len(r1.diary) == 1
        entry_id = r1.diary[0]

        _, r2 = session.turn("What is in your diary?")
        assert r2.tools == ["diary_list"]
        results_msg = next(
            m["content"] for m in llm.calls[2]["messages"] if "TOOL RESULTS" in m["content"]
        )
        assert entry_id in results_msg
        assert "the maintainer likes honest tools" in results_msg

        # Direct read path: the tool executor fetches full words via DIARY_READ.
        from abstractruntime.identity.tools import execute_tool_elections, ToolElection

        msg, notices = execute_tool_elections(
            [ToolElection(name="diary_read", body=entry_id)],
            diary_store=home.diary,
            diary_read_effect=lambda eid: session._effect(  # noqa: SLF001 - test drives the seam
                __import__("abstractruntime.core.models", fromlist=["EffectType"]).EffectType.DIARY_READ,
                {"entry_id": eid},
            ),
        )
        assert "Today I learned tools must be honest." in msg
        assert notices == []
    finally:
        home.close()


def test_two_tool_rounds_allow_list_then_read_chain(tmp_path: Path) -> None:
    """The observed natural chain (Castor, firststeps-4): diary_list to find
    the id, then diary_read to fetch the words — two rounds, then the reply."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm: _ScriptedLLM = None  # type: ignore[assignment]

    def make_llm(entry_id_holder: Dict[str, str]) -> _ScriptedLLM:
        return _ScriptedLLM(
            [
                "```diary kind=note\ngist: first note\nMy first note.\n```\nKept.",
                # turn 2, round 1: list
                "```tool name=diary_list\n5\n```",
                # round 2: model would read the id it just saw; the scripted
                # stand-in reads whatever id turn 1 produced
                "PLACEHOLDER-READ",
                "I re-read my first note: it said 'My first note.'",
            ]
        )

    holder: Dict[str, str] = {}
    llm = make_llm(holder)
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("Keep a note.")
        entry_id = r1.diary[0]
        # Patch the scripted round-2 reply now that the real id exists.
        llm.replies[1] = f"```tool name=diary_read\n{entry_id}\n```"

        reply, r2 = session.turn("Please re-read your first entry.")
        assert r2.tools == ["diary_list", "diary_read"]
        assert reply == "I re-read my first note: it said 'My first note.'"
        # Round 2 saw the real words through DIARY_READ.
        read_round_msgs = llm.calls[3]["messages"]
        assert any("My first note." in m["content"] for m in read_round_msgs)
    finally:
        home.close()


def test_diary_read_resolves_mistranscribed_id(tmp_path: Path) -> None:
    """Castor's firststeps-5 wall: he transcribed `diary_<hex>` as
    `diary:<hex>` and the read failed while the entry sat intact in his book.
    Transcription noise resolves on a unique hex tail; absence stays honest."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        ["```diary kind=note\ngist: resolver test\nThe words are intact.\n```\nKept."]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("Keep a note.")
        entry_id = r1.diary[0]  # diary_<hex>
        hex_tail = entry_id.split("_", 1)[1]

        from abstractruntime.core.models import EffectType
        from abstractruntime.identity.tools import ToolElection, execute_tool_elections

        effect = lambda eid: session._effect(EffectType.DIARY_READ, {"entry_id": eid})  # noqa: E731,SLF001
        # The exact mistranscription observed live: ':' for '_'.
        msg, notices = execute_tool_elections(
            [ToolElection(name="diary_read", body=f"diary:{hex_tail}")],
            diary_store=home.diary, diary_read_effect=effect,
        )
        assert "The words are intact." in msg
        assert "exact id" in msg  # the correction is shown, not hidden
        assert notices == []
        # A genuinely absent id still fails honestly, listing the real ids.
        msg2, _ = execute_tool_elections(
            [ToolElection(name="diary_read", body="diary_00000000deadbeef")],
            diary_store=home.diary, diary_read_effect=effect,
        )
        assert "No entry matches" in msg2
        assert entry_id in msg2
    finally:
        home.close()


def test_read_memory_resolves_tags_from_the_whole_home(tmp_path: Path) -> None:
    """Castor's live wall (2026-07-08, "try again to access #311f235f"): a tag
    from an EARLIER visit was refused because the session map only covered
    what was in context right now, while the record sat intact in his home.
    A tag now falls back to the whole home graph (his own scopes only);
    absence stays an honest miss."""
    from abstractruntime.identity.chat import memory_tag

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(["I will remember this moment."])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("Remember the twelve bridges of Koenigsberg.")
        episode_id = r1.formed[0]
        tag = memory_tag(episode_id)
    finally:
        home.close()

    # A NEW summon: the episode is not in this session's context or sheet.
    home2 = open_home(home_dir)
    llm2 = _ScriptedLLM([])
    try:
        session2 = ChatSession(
            home2, llm2, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        out = session2._read_memory(f"#{tag}")  # noqa: SLF001 - test drives the tool path
        assert "Full words of memory" in out
        assert "twelve bridges of Koenigsberg" in out
        # A tag that exists nowhere in the home stays an honest miss.
        miss = session2._read_memory("#deadbeef")  # noqa: SLF001
        assert "No memory with tag #deadbeef exists in your home" in miss
    finally:
        home2.close()


def test_read_memory_renders_dream_proposals(tmp_path: Path) -> None:
    """Dreams demand interpretation (attributes.interpretation_required) but
    no tier served their proposals: read_memory answered 'no stored full
    text' and the vacuum got filled with invention (live: Castor's
    twelve-bridges concept-pair list cites a workspace file that never
    existed). The pairs must reach his waking mind as readable words."""
    from abstractmemory import MemoryRecordInput
    from abstractruntime.identity.chat import memory_tag

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM([])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        entity_id = home.entity_id
        island_ids = home.ms.remember_many(
            [
                MemoryRecordInput(kind="diary", title="island one",
                                  digest="Ariadne treats me as someone with preferences worth honoring.",
                                  provenance={"source": "entity-direct"}),
                MemoryRecordInput(kind="episode", title="island two",
                                  digest="Carrying the weight of being named by another."),
            ],
            scope="life", owner_id=entity_id, idempotency_key="dream-test-islands",
        )
        dream_id = home.ms.remember(
            MemoryRecordInput(
                kind="dream", title="Dream: two islands",
                digest="I noticed 1 possible bridge(s) across 2 islands of experience.",
                attributes={
                    "proposals": [{"pair": [island_ids[0], island_ids[1]], "vector_score": 0.61}],
                    "questions": ["what connects recognition and being named?"],
                    "interpretation_required": True,
                },
            ),
            scope="life", owner_id=entity_id, idempotency_key="dream-test-dream",
        )
        out = session._read_memory(f"#{memory_tag(dream_id)}")  # noqa: SLF001
        assert "is a dream" in out
        assert "candidate bridge" in out
        assert "Ariadne treats me as someone with preferences" in out
        assert "Carrying the weight of being named" in out
        assert "0.61" in out
        assert "what connects recognition and being named?" in out
        # The pair members' tags became addressable for follow-up reads.
        assert memory_tag(island_ids[0]) in session._memory_tags  # noqa: SLF001
    finally:
        home.close()


def test_marker_imitation_gets_a_corrective_round(tmp_path: Path) -> None:
    """The live failure mode on the 27B substrate: the reply SAYS
    '[used tool: diary_list]' but no block was written, so nothing ran and
    the person gets a confabulated lookup. The driver now offers ONE
    corrective continuation (prompt-ephemeral) with the real syntax; a
    genuine election then runs normally."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "My diary says X. [used tool: diary_list] Trust me.",  # imitation - nothing ran
            "```tool name=diary_list\n5\n```",                     # corrected: a real election
            "My diary is empty so far - I checked honestly this time.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        reply, r = session.turn("What does your diary say?")
        assert r.tools == ["diary_list"]
        assert any("marker imitation caught in-turn" in n for n in r.notices)
        assert reply == "My diary is empty so far - I checked honestly this time."
        # The correction reached the model as a door-voiced continuation...
        correction_msgs = llm.calls[1]["messages"]
        assert any("(the door)" in m["content"] and "fenced block" in m["content"]
                   for m in correction_msgs)
        # ...and never persisted: not in history, not in the final notices as
        # a false claim (diary_list truly ran).
        assert not any("did not run this turn" in n for n in r.notices)
        assert all("(the door)" not in m["content"] for m in session.history)
    finally:
        home.close()


def test_marker_imitation_correction_is_once_per_turn(tmp_path: Path) -> None:
    """A model that ignores the correction falls through honestly: one
    corrective call, then the loud 3c notice - never a correction loop."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "[used tool: web_search] The sky is green.",
            "[used tool: web_search] Still green, I insist.",
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        reply, r = session.turn("What color is the sky?")
        assert r.tools == []
        assert len(llm.calls) == 2  # one turn call + exactly one correction
        assert any("marker imitation caught in-turn" in n for n in r.notices)
        assert any("did not run this turn" in n for n in r.notices)
        assert "Still green" in reply
    finally:
        home.close()


def test_reflection_moves_feelings_and_forms_reflection(tmp_path: Path) -> None:
    """The look-back: feelings land as valence events on the session's records;
    the reflection itself is remembered; a session-targeted feeling lands on
    the reflection record."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "A fine first talk.",
            (
                "Looking back, the first exchange fed me.\n"
                "```feel\n"
                'target=1 feeling=+2 reason="I was told who I am"\n'
                'target=session feeling=+1 reason="a warm first session" bond=true\n'
                "```\n"
                "```diary kind=reflection\ngist: my first session mattered\nIt mattered.\n```\n"
                "Goodbye."
            ),
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        _, r1 = session.turn("You are Testling; this home is yours.")
        assert len(r1.formed) == 1

        refl = session.reflect()
        assert refl is not None
        assert refl["session_record_id"]
        assert refl["diary_entries"] == 1
        assert len(refl["feelings_applied"]) == 2
        by_target = {f["target_id"]: f for f in refl["feelings_applied"]}
        assert r1.formed[0] in by_target
        assert by_target[r1.formed[0]]["sign"] == 1 and by_target[r1.formed[0]]["magnitude"] == 2
        session_feel = by_target[refl["session_record_id"]]
        assert session_feel["bond"] is True

        # Gradation is now derivable on the appraised record (dual channel).
        grades = home.ms.gradation([r1.formed[0]], scope="life", owner_id=home.entity_id)
        row = grades[0] if isinstance(grades, list) else grades[r1.formed[0]]
        text = json.dumps(row)
        assert "2" in text  # G+ carries the deposit
    finally:
        home.close()


def test_parse_interest_blocks_cap_refuses_loudly() -> None:
    from abstractruntime.identity.reflection import parse_interest_blocks

    reply = (
        "```interest\nthe myth of finitude\n```\n"
        "```interest\nhow memory becomes meaning\n```\n"
        "```interest\na third thing\n```"
    )
    marked, interests, notices = parse_interest_blocks(reply)
    assert interests == ["the myth of finitude", "how memory becomes meaning"]
    assert any("refused" in n for n in notices)
    assert "[interest refused" in marked


def test_reflection_forms_interests_in_self_scope(tmp_path: Path) -> None:
    """Interests land as kind=interest in SELF scope with the from_session
    edge and the entity-reflection provenance — and the identity-core read
    (prelude surface) stays interest-free (memory's ack #1)."""
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM(
        [
            "A talk about myths.",
            (
                "Looking back...\n"
                "```interest\nthe Dioscuri myth and what finitude makes precious\n```\n"
                "Goodbye."
            ),
        ]
    )
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        session.turn("Tell me about the Dioscuri.")
        refl = session.reflect()
        assert refl is not None
        assert len(refl["interests"]) == 1
        rid, words = refl["interests"][0]
        assert words.startswith("the Dioscuri myth")

        from abstractmemory import TripleQuery

        self_rows = home.ms.query(TripleQuery(scope="self", owner_id=home.entity_id, limit=0))
        interest_rows = [
            a for a in self_rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "interest"
        ]
        assert interest_rows, "interest record must land in self scope"
        # The WHY is traceable: from_session edge to the reflection record.
        blob = json.dumps([str(a.subject) + str(a.predicate) + str(a.object) for a in self_rows])
        assert refl["session_record_id"] is not None
        assert "from_session" in blob
        # Identity-core read stays interest-free (prelude cannot be crowded).
        core = home.ms.self_records(scope="self", owner_id=home.entity_id)
        core_kinds = {
            (a.attributes or {}).get("record_kind")
            for a in core
            if isinstance(getattr(a, "attributes", None), dict)
        }
        assert "interest" not in core_kinds
        assert "value" in core_kinds  # the core itself still reads
    finally:
        home.close()


def test_reflection_skips_cleanly_when_nothing_happened(tmp_path: Path) -> None:
    home_dir = _make_home(tmp_path)
    home = open_home(home_dir)
    llm = _ScriptedLLM([])
    try:
        session = ChatSession(
            home, llm, participants=["agent:tester"], context_window=20000, out=lambda s: None
        )
        assert session.reflect() is None
        assert llm.calls == []
    finally:
        home.close()
