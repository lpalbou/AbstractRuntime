"""The entity's `topics` tool (identity/topics.py `topics_text`, through HomeMemoryReader.topics; operator 2026-11-10:
topics help the entity "to have its knowledge organized by topic, to know what it knows, and let it delve deeper into
subparts of its mind/memory").

It reads what memory serves - `abstractmemory.community_index.communities(system, scopes=, owner_id=,
version=None)` (AbstractMemory d550551) - never a cache of its own. The rendering cases use `_FakeCommunities`, a
scripted payload v2 over the real records of a real home (two topics, one untitled, one provisional member), so each
case is exact; the last tests run memory's REAL pass and REAL read end to end. Ranking by meaning goes through the
store's vector query with a deterministic test embedder. Each test goes red without the piece it names."""

from __future__ import annotations

import hashlib
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_entity_life_loop import _make_home  # noqa: E402

EID = "entity:liveling@home-test"
LADDER = [("self", EID), ("diary", EID), ("life", EID)]


class _HashEmbedder:
    """Bag of hashed words: texts sharing words are close, texts sharing none are ~orthogonal."""

    dim = 256

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in str(text).lower().split():
                tok = tok.strip(".,;:!?()[]\"'")
                if len(tok) > 3:
                    vec[int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big") % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


RECORDS = [  # (key, kind, title, digest, observed_at)
    ("dam-1", "episode", "Checking the dam spillway", "We checked the spillway gauges of the dam together.",
     "2026-11-01T10:00:00+00:00"),
    ("dam-2", "episode", "The dam inspection report", "The inspection report of the dam spillway came back clean.",
     "2026-11-05T10:00:00+00:00"),
    ("dam-3", "lesson", "Gauges drift in winter", "Spillway gauges drift when the winter cold sets in.",
     "2026-11-03T10:00:00+00:00"),
    ("gar-1", "episode", "Planting the garden sensors", "Planted moisture sensors along the garden beds.",
     "2026-11-02T10:00:00+00:00"),
    ("gar-2", "episode", "Watering schedule", "The garden moisture sensors now drive the watering schedule.",
     "2026-11-06T10:00:00+00:00"),
]


def _home(tmp_path: Path, *, embedder: Any = "hash") -> Any:
    from abstractmemory import MemoryRecordInput

    from abstractruntime.identity.chat import open_home

    home_dir = _make_home(tmp_path)
    home = open_home(home_dir, embedder=_HashEmbedder(), fence=None)
    ids: Dict[str, str] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key, kind, title, digest, at in RECORDS:
            with mock.patch("abstractmemory.records.utc_now_iso", return_value=at):  # the formation's clock
                ids[key] = home.ms.remember_many(
                    [MemoryRecordInput(kind=kind, title=title, digest=digest)],
                    scope="life", owner_id=home.entity_id, idempotency_key=key, turn_id=key)[0]
    home.close()
    home = open_home(home_dir, embedder=_HashEmbedder() if embedder == "hash" else embedder, fence=None)
    home.test_ids = ids
    return home


class _FakeCommunities:
    """memory's `communities` (frozen signature), serving payload v2 over the home's real records."""

    def __init__(self, ids: Dict[str, str], *, first_sleep_pending: bool = False, version: int = 2) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.payload = {
            "version": version, "version_seq": 41, "first_sleep_pending": first_sleep_pending,
            "algorithm": "louvain-warm-v1",
            "communities": [] if first_sleep_pending else [
                {"id": "ex:community-r1v41", "key": "ex:community-aaaa1111dam0", "title": "The dam and its gauges",
                 "summary": "How the dam's spillway was checked, and why its gauges drift.", "pending": [],
                 "verdict": None, "size": 3, "kinds": {"episode": 2, "lesson": 1},
                 "members": [{"id": ids["dam-1"], "via": "detected"}, {"id": ids["dam-2"], "via": "detected"}],
                 "provisional_ids": [ids["dam-3"]], "cohesion": 0.7, "layout": {},
                 "facts": {"span": {"first": "2026-11-01T10:00:00+00:00", "last": "2026-11-05T10:00:00+00:00"}}},
                {"id": "ex:community-r2v41", "key": "ex:community-bbbb2222gar0", "title": None,
                 "summary": None, "pending": ["title", "summary"], "verdict": None, "size": 2,
                 "kinds": {"episode": 2},
                 "members": [{"id": ids["gar-1"], "via": "detected"}, {"id": ids["gar-2"], "via": "attached"}],
                 "provisional_ids": [], "cohesion": 0.6, "layout": {},
                 "facts": {"span": {"first": "2026-11-02T10:00:00+00:00", "last": "2026-11-06T10:00:00+00:00"}}},
            ],
            "links": [],
        }

    def communities(self, system, *, scopes, owner_id, version=None):
        self.calls.append({"scopes": [tuple(s) for s in scopes], "owner_id": owner_id, "version": version})
        return self.payload


@pytest.fixture
def served(tmp_path: Path, monkeypatch):
    import abstractmemory.community_index as community_index

    from abstractruntime.identity.memory_reader import HomeMemoryReader

    home = _home(tmp_path)
    fake = _FakeCommunities(home.test_ids)
    monkeypatch.setattr(community_index, "communities", fake.communities)
    try:
        yield home, HomeMemoryReader(home), fake
    finally:
        home.close()


# ------------------------------------------------------------------------------------------------ the answers


def test_the_list_shows_each_topic_title_summary_size_and_newest_memory(served) -> None:
    home, reader, fake = served
    scans: List[int] = []
    real_scan = reader.digest_assertions_all
    reader.digest_assertions_all = lambda: scans.append(1) or real_scan()
    text = reader.topics("")
    assert scans == [], "the list is memory's one read (its facts carry the newest date): no scan of his digests"
    assert fake.calls == [{"scopes": LADDER, "owner_id": EID, "version": None}], "his scopes, his owner, current"
    assert "2 topics over 5 memories" in text
    assert ("- #1111dam0 The dam and its gauges (3 memories, newest 2026-11-05) - How the dam's spillway was "
            "checked, and why its gauges drift.") in text
    # A topic not titled yet says so, honestly - never a title cut from its members.
    assert "- #2222gar0 (untitled yet - 2 memories) - (its summary is not written yet" in text
    assert text.index("#1111dam0") < text.index("#2222gar0"), "by size"
    assert "read_memory #tag" in text


def test_a_topic_tag_shows_its_memories_newest_first_with_their_tags(served) -> None:
    from abstractruntime.identity.memory_reader import memory_tag

    home, reader, _fake = served
    ids = home.test_ids
    text = reader.topics("#1111dam0")
    assert text.startswith("Topic #1111dam0: The dam and its gauges (3 memories)")
    lines = [ln for ln in text.splitlines() if ln.startswith("- #")]
    assert lines == [
        f"- #{memory_tag(ids['dam-2'])} [episode 2026-11-05] The dam inspection report",
        f"- #{memory_tag(ids['dam-3'])} [lesson 2026-11-03] Gauges drift in winter (placed since your last sleep)",
        f"- #{memory_tag(ids['dam-1'])} [episode 2026-11-01] Checking the dam spillway",
    ], text
    # The #tags are read_memory's at once (registered in the reader's map).
    assert reader.tag_map[memory_tag(ids["dam-2"])] == ids["dam-2"]
    assert "Full words" in reader.read_memory(f"#{memory_tag(ids['dam-2'])}") or "born as these words" in \
        reader.read_memory(f"#{memory_tag(ids['dam-2'])}")


def test_a_memory_tag_shows_the_topic_it_belongs_to(served) -> None:
    from abstractruntime.identity.memory_reader import memory_tag

    home, reader, _fake = served
    tag = memory_tag(home.test_ids["gar-2"])
    text = reader.topics(f"#{tag}")
    assert text.startswith("Topic #2222gar0: (untitled yet - 2 memories)")
    assert f"#{tag} is one of the memories of this topic." in text


def test_an_unknown_tag_is_said_and_the_list_follows(served) -> None:
    _home_, reader, _fake = served
    text = reader.topics("#deadbeef")
    assert text.startswith("No topic or memory with tag #deadbeef in your topics.")
    assert "2 topics over 5 memories" in text


def test_words_find_the_closest_topic_by_meaning(served) -> None:
    """Meaning, not letters: the store's vector query ranks the topics' memories (memory's VectorMatrix)."""
    from abstractruntime.identity.memory_reader import memory_tag

    home, reader, _fake = served
    text = reader.topics("moisture sensors watering")
    assert text.startswith('The topic closest to "moisture sensors watering":\nTopic #2222gar0:'), text
    members = [ln for ln in text.splitlines() if ln.startswith("- #")]
    assert members[0].startswith(f"- #{memory_tag(home.test_ids['gar-2'])} [episode 2026-11-06] Watering schedule")
    assert "(closeness " in members[0]
    assert "Also close: #1111dam0 The dam and its gauges" not in text or text.index("Also close") > 0
    text = reader.topics("spillway gauges")
    assert "Topic #1111dam0: The dam and its gauges" in text.splitlines()[1]


def test_a_topic_tag_with_words_ranks_its_memories_by_meaning(served) -> None:
    from abstractruntime.identity.memory_reader import memory_tag

    home, reader, _fake = served
    text = reader.topics("#1111dam0 winter cold drift")
    members = [ln for ln in text.splitlines() if ln.startswith("- #")]
    assert 'Its memories closest to "winter cold drift" by meaning:' in text
    assert members[0].startswith(f"- #{memory_tag(home.test_ids['dam-3'])} [lesson 2026-11-03] Gauges drift")


def test_the_tool_reads_what_memory_serves_every_time(served) -> None:
    """No cache of its own: a new version memory serves is what the next call shows - under the same #tag (a
    topic's tag is its stable key's, never the version's record id)."""
    _home_, reader, fake = served
    assert "The dam and its gauges" in reader.topics("")
    fake.payload["communities"][0].update(title="Dam safety", id="ex:community-r1v57")
    text = reader.topics("#1111dam0")
    assert text.startswith("Topic #1111dam0: Dam safety") and len(fake.calls) == 2


# ------------------------------------------------------------------------------------------------ the edges


def test_before_the_first_sleep_the_tool_says_when_topics_form(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory.community_index as community_index

    from abstractruntime.identity.memory_reader import HomeMemoryReader

    home = _home(tmp_path)
    try:
        monkeypatch.setattr(community_index, "communities",
                            _FakeCommunities(home.test_ids, first_sleep_pending=True).communities)
        assert "during your next sleep" in HomeMemoryReader(home).topics("")
        empty = _FakeCommunities(home.test_ids)
        empty.payload["communities"] = []
        monkeypatch.setattr(community_index, "communities", empty.communities)
        assert HomeMemoryReader(home).topics("") == ("Your memory holds no topics: there is not yet enough in it to "
                                                     "group by subject.")
    finally:
        home.close()


def test_without_an_embedder_words_get_the_list_and_tags_get_newest_first(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory.community_index as community_index

    from abstractruntime.identity.memory_reader import HomeMemoryReader

    home = _home(tmp_path, embedder=None)
    try:
        monkeypatch.setattr(community_index, "communities", _FakeCommunities(home.test_ids).communities)
        reader = HomeMemoryReader(home)
        text = reader.topics("spillway gauges")
        assert text.startswith("(no embedder serves this session, so your topics cannot be matched by meaning")
        assert "2 topics over 5 memories" in text
        text = reader.topics("#1111dam0 winter")
        assert "cannot be ranked by meaning - newest first instead" in text and "Its memories, newest first:" in text
    finally:
        home.close()


def test_the_tool_runs_through_the_descriptor_and_fails_loudly_on_a_payload_it_cannot_read(served) -> None:
    from abstractruntime.identity.tools import ToolElection, execute_tool_elections

    _home_, reader, fake = served
    e = ToolElection(name="topics", body="")
    message, notices = execute_tool_elections([e], diary_store=None, diary_read_effect=lambda _i: {},
                                              topics_fn=reader.topics)
    assert "[topics]" in message and "The dam and its gauges" in e.result and notices == []
    fake.payload["version"] = 1
    e = ToolElection(name="topics", body="")
    _message, notices = execute_tool_elections([e], diary_store=None, diary_read_effect=lambda _i: {},
                                               topics_fn=reader.topics)
    assert "memory served communities payload version 1; this runtime reads version 2" in e.result
    assert any(n.startswith("#FALLBACK tool topics failed") for n in notices)


def test_a_memory_without_the_communities_verb_fails_the_call_loudly(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory.community_index as community_index

    from abstractruntime.identity.memory_reader import HomeMemoryReader
    from abstractruntime.identity.tools import ToolElection, execute_tool_elections

    home = _home(tmp_path)
    try:
        monkeypatch.delattr(community_index, "communities")
        e = ToolElection(name="topics", body="")
        _m, notices = execute_tool_elections([e], diary_store=None, diary_read_effect=lambda _i: {},
                                             topics_fn=HomeMemoryReader(home).topics)
        assert "the topics call failed" in e.result and "communities" in e.result
        assert any(n.startswith("#FALLBACK tool topics failed") for n in notices)
    finally:
        home.close()


def test_an_unwired_lane_says_so_never_silently(served) -> None:
    from abstractruntime.identity.tools import ToolElection, execute_tool_elections

    e = ToolElection(name="topics", body="")
    _m, notices = execute_tool_elections([e], diary_store=None, diary_read_effect=lambda _i: {})
    assert e.result == "(the topics tool is not enabled in this session)"
    assert "#FALLBACK topics elected but no resolver wired" in notices


# ------------------------------------------------------------------------------------------------ the surfaces


def test_topics_is_a_tier1_read_tool_taught_in_the_prompt_and_declared_natively() -> None:
    from abstractruntime.identity import tools as T
    from abstractruntime.identity.tool_policy import resolve_tool_grant  # noqa: F401 - the grant derives from tier1

    assert "topics" in T.TIER1_TOOL_NAMES
    d = T.TOOL_DESCRIPTORS["topics"]
    assert d.tier == "tier1" and d.mutating is False and d.body_optional is True
    assert d.capability_class == "tier1_self" and d.required == ()
    assert "```tool name=topics\n" in T.TOOLS_CONTRACT_PARAGRAPH
    flat = " ".join(T.TOOLS_CONTRACT_PARAGRAPH.split())
    assert "topics shows what you know, organized: your memories grouped by subject" in flat
    spec = next(s for s in T.native_tool_specs(T.TIER1_TOOL_NAMES) if s["name"] == "topics")
    assert spec["parameters"]["required"] == [] and "topic" in spec["parameters"]["properties"]
    elections, _markers, _notices = T.native_tool_elections([{"name": "topics", "arguments": {"topic": "the dam"}}])
    assert [(x.name, x.body) for x in elections] == [("topics", "the dam")]
    _marked, parsed, _notices = T.parse_tool_blocks("```tool name=topics\n```")
    assert [(x.name, x.body) for x in parsed] == [("topics", "")], parsed


def test_the_effect_lane_runs_topics_wired(tmp_path: Path, monkeypatch) -> None:
    """ENTITY_TOOLS_EXECUTE (the durable lanes): topics is granted AND wired - never "not enabled"."""
    import abstractmemory.community_index as community_index

    from test_entity_brain_effects import _dispatch, _result

    from abstractruntime.core.models import EffectType

    home = _home(tmp_path)
    try:
        monkeypatch.setattr(community_index, "communities", _FakeCommunities(home.test_ids).communities)
        out = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "visit", "tool_calls": [{"name": "topics", "arguments": {}}]}))
        assert out["tools_ran"] == ["topics"]
        assert "The dam and its gauges" in out["results"][0]["result"]
    finally:
        home.close()


def test_the_chat_lane_runs_topics_wired(tmp_path: Path, monkeypatch) -> None:
    """ChatSession (own time, hosted chat): a fenced topics block in a reply runs through the session's reader."""
    import abstractmemory.community_index as community_index

    from test_entity_recent_memories import _ScriptedLLM

    from abstractruntime.identity.chat import ChatSession

    home = _home(tmp_path)
    monkeypatch.setattr(community_index, "communities", _FakeCommunities(home.test_ids).communities)
    s = ChatSession(home, _ScriptedLLM(["```tool name=topics\n```", "I know about the dam."]),
                    participants=["agent:tester"], context_window=20000, out=lambda _s: None, fence=None)
    try:
        _reply, report = s.turn("What do you know?")
        detail = next(d for d in report.tool_details if d["name"] == "topics")
        assert "The dam and its gauges" in str(detail.get("result") or ""), report.tool_details
        assert not [n for n in report.notices if "topics" in n], report.notices
    finally:
        s.home.close()


# ------------------------------------------------------------------------------------------------ a topic is not a memory


class _TitledCapability:
    model_info = {"provider": "fake", "model": "summarizer"}

    def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
        return response_model(title="Waterworks of the valley", summary="The valley's water, kept and measured.")


def test_a_topic_record_is_never_served_as_a_memory(tmp_path: Path, monkeypatch) -> None:
    """Memory's seam (AbstractMemory aac9476 keeps community records off recall, probes, vectors and FTS; the
    runtime's own abstracts read is its to filter): with a REAL community record in the home, search_memory never
    lists it and read_memory never resolves its tag - while the topics tool lists it."""
    from abstractmemory.communities import communities_pass, community_records

    from abstractruntime.identity.authoring import capability_formation_author
    from abstractruntime.identity.memory_reader import HomeMemoryReader, memory_tag

    pytest.importorskip("numpy")
    home = _home(tmp_path)  # formed with the hash embedder: every record vectored
    try:
        cap = _TitledCapability()
        receipt = communities_pass(home.ms, scopes=LADDER, owner_id=EID,
                                   author=capability_formation_author(lambda _n: cap, home.artifacts),
                                   cancelled=lambda: False)
        assert receipt["version_seq"] is not None, receipt
        # (one author answer for every topic: the first keeps the title, a sibling's collision forms it pending)
        titled = [c for c in community_records(home.ms, scopes=LADDER, owner_id=EID) if c.title]
        assert [c.title for c in titled] == ["Waterworks of the valley"], titled
        topic = titled[0]
        reader = HomeMemoryReader(home)
        found = reader.search_memory("Waterworks of the valley")
        assert "[community" not in found and "Nothing in your memory graph" in found, found
        assert "No memory with tag" in reader.read_memory(f"#{memory_tag(topic.record_id)}")
        # The topics tool lists it, through memory's real read.
        assert f"#{memory_tag(topic.key)} Waterworks of the valley" in reader.topics("")
    finally:
        home.close()


def test_the_tool_end_to_end_on_memory_s_real_pass_and_read(tmp_path: Path) -> None:
    """Memory's real pass forms the topics, its real read serves them, the tool shows them: every authored title in
    the list, and a topic's #tag lists exactly the memories memory placed in it, each with its read_memory #tag."""
    from abstractmemory.communities import communities_pass
    from abstractmemory.community_index import communities

    from abstractruntime.identity.authoring import capability_formation_author
    from abstractruntime.identity.memory_reader import HomeMemoryReader, memory_tag

    pytest.importorskip("numpy")

    class _Numbered:
        model_info = {"provider": "fake", "model": "summarizer"}
        n = 0

        def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
            _Numbered.n += 1
            return response_model(title=f"Topic number {_Numbered.n}", summary=f"What topic {_Numbered.n} holds.")

    home = _home(tmp_path)
    try:
        cap = _Numbered()
        receipt = communities_pass(home.ms, scopes=LADDER, owner_id=EID,
                                   author=capability_formation_author(lambda _n: cap, home.artifacts),
                                   cancelled=lambda: False)
        assert receipt["version_seq"] is not None and receipt["authored"] == receipt["communities"] >= 1, receipt
        served = communities(home.ms, scopes=LADDER, owner_id=EID)["communities"]
        reader = HomeMemoryReader(home)
        listing = reader.topics("")
        for c in served:
            assert f"#{memory_tag(c['key'])} {c['title']} ({c['size']} memories, newest " in listing, listing
        biggest = served[0]
        drill = reader.topics(f"#{memory_tag(biggest['key'])}")
        shown = {ln.split()[1].lstrip("#") for ln in drill.splitlines() if ln.startswith("- #")}
        expected = {memory_tag(m["id"]) for m in biggest["members"]} | {memory_tag(p) for p in biggest["provisional_ids"]}
        assert shown == {t for t in expected if t in shown} and shown, (shown, expected)
        assert len(shown) == min(len(expected), 20)
        assert all(reader.tag_map[t] for t in shown)
    finally:
        home.close()
