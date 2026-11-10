"""Three fields the visit ledger records for a host's view of a turn (the Entity app's live activity panel,
abstractentity a82aa10, root backlog 1196). Each is additive: a reader without it degrades to what it showed before.

1. The RECALL record: every handle in `result.handles` carries `topic`, the stable key of the topic (memory
   community) its record is in now, or None. ONE membership-only read per recall (memory's `community_keys_of`, over
   the handles' graph ids): never memory's communities payload (it computes every topic's facts) and never one read
   per handle.
2. A search_memory result carries `hits` (int): the graph and book matches its text counts
   (`ToolElection.hits`; the ENTITY_TOOLS_EXECUTE result entry). The gateway's visit TOOL_CALLS executor builds the
   visit lane's entries itself and copies it from the election, as it copies `artifact_id`.
3. The `reason` model-call record: `effect.payload.iteration` (from 1) and `effect.payload.max_iterations` (the
   turn's cap) - "step 2 of 50".

Each test goes red without the piece it names."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

sys.path.insert(0, str(Path(__file__).parent))

from test_topics_tool import EID, _FakeCommunities, _home  # noqa: E402

from abstractruntime.core.models import Effect, EffectType, RunStatus  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.core.vars import DEFAULT_MAX_ITERATIONS  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.lifecycle_fence import LifecycleFence  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    HARVEST_NODE,
    VISITOR_WAIT_KEY,
    ReactMiddle,
    build_visit_workflow,
)

TOPIC_DAM = "ex:community-aaaa1111dam0"


class _FakeKeys:
    """memory's `community_keys_of` (frozen signature) answering from `_FakeCommunities`' scripted payload with
    memory's rule (community_index.community_memberships_of): a member's topic, else a provisional record's, else None."""

    def __init__(self, ids: Dict[str, str]) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.topic_of: Dict[str, str] = {}
        topics = _FakeCommunities(ids).payload["communities"]
        for c in topics:
            self.topic_of.update({m["id"]: c["key"] for m in c["members"]})
        for c in topics:
            for record_id in c["provisional_ids"]:
                self.topic_of.setdefault(record_id, c["key"])

    def community_keys_of(self, system: Any, record_ids: Any, *, scopes: Any, owner_id: str) -> Dict[str, Any]:
        asked = list(record_ids)
        self.calls.append({"record_ids": asked, "scopes": [tuple(s) for s in scopes], "owner_id": owner_id})
        return {rid: self.topic_of.get(rid) for rid in asked}


def _count_payload_reads(monkeypatch, community_index: Any) -> List[Dict[str, Any]]:
    """Every call of memory's `communities` (the payload read the stamp must never make), still answered."""
    calls: List[Dict[str, Any]] = []
    real = community_index.communities

    def _counted(system: Any, *, scopes: Any, owner_id: str, version: Any = None) -> Dict[str, Any]:
        calls.append({"scopes": [tuple(s) for s in scopes], "owner_id": owner_id})
        return real(system, scopes=scopes, owner_id=owner_id, version=version)

    monkeypatch.setattr(community_index, "communities", _counted)
    return calls


def _graph_ids(handles: List[Dict[str, Any]]) -> List[str]:
    """The handles' graph ids (`provenance.record_id`), once each, in order: what the stamp asks memory about."""
    ids = (str((h.get("provenance") or {}).get("record_id") or "") for h in handles)
    return [g for g in dict.fromkeys(ids) if g]


class _ScriptedLLM:
    def __init__(self, replies: List[Dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        self.calls.append(json.loads(json.dumps(effect.payload or {}, default=str)))
        return EffectOutcome.completed(dict(self.replies.pop(0) if self.replies else {"content": "..."}))


def _tools(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
    results = [{"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "a page",
                "error": None} for tc in ((effect.payload or {}).get("tool_calls") or [])]
    return EffectOutcome.completed({"mode": "executed", "results": results})


def _react_visit(home_dir: Path, replies: List[Dict[str, Any]], *, start_vars: Dict[str, Any] | None = None):
    pytest.importorskip("abstractagent")
    from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn
    from abstractagent.logic.react import ReActLogic
    from abstractcore.tools import ToolDefinition

    llm = _ScriptedLLM(replies)
    ert = open_entity_runtime(home_dir, fence=LifecycleFence.admit(home_dir),
                              extra_handlers={EffectType.LLM_CALL: llm, EffectType.TOOL_CALLS: _tools})
    react = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={})]),
        workflow_id="entity-visit-react", provider="stub", model="stub",
        allowed_tools=["read_file"], final_next_node=HARVEST_NODE,
    )
    wf = build_visit_workflow(
        ert.home, participants=["person:albou"], idle_seconds=3600,
        model_info={"provider": "test", "model": "scripted"}, visit_id="visit-f1",
        react_middle=ReactMiddle(nodes=react.nodes, entry="reason", reset_turn=reset_react_turn),
    )
    rid = ert.runtime.start(workflow=wf, vars=dict(start_vars or {}), session_id="visit-f")
    assert ert.runtime.tick(workflow=wf, run_id=rid, max_steps=50).status == RunStatus.WAITING
    return ert, wf, rid, llm


def _say(ert, wf, rid, text: str):
    state = ert.runtime.resume(workflow=wf, run_id=rid, wait_key=VISITOR_WAIT_KEY,
                               payload={"text": text, "speaker": "person:albou"}, max_steps=200)
    assert state.status == RunStatus.WAITING, (state.status, state.error)
    return state


def _records(ert, rid: str) -> List[Dict[str, Any]]:
    return [r if isinstance(r, dict) else dict(r) for r in ert.ledger_store.list(rid)]


# ------------------------------------------------------------------------------------------- 1. topics on recall


def test_the_recall_record_names_each_memory_s_topic_with_one_membership_read(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory.community_index as community_index

    home = _home(tmp_path)
    ids, home_dir = dict(home.test_ids), home.home_dir
    home.close()
    fake = _FakeKeys(ids)
    monkeypatch.setattr(community_index, "community_keys_of", fake.community_keys_of)
    monkeypatch.setattr(community_index, "communities", _FakeCommunities(ids).communities)
    payload_reads = _count_payload_reads(monkeypatch, community_index)
    ert, wf, rid, llm = _react_visit(home_dir, [{"content": "The gauges were fine.", "tool_calls": []}])
    try:
        _say(ert, wf, rid, "Do you remember the dam spillway gauges?")
        recalls = [r for r in _records(ert, rid)
                   if r["node_id"] == "RECALL" and r["status"] == "completed"]
        assert len(recalls) == 1
        handles = recalls[0]["result"]["handles"]
        by_graph = {(h.get("provenance") or {}).get("record_id"): h for h in handles}
        # The cue surfaced the dam memories: members (dam-1, dam-2) and the provisional one (dam-3).
        surfaced = [k for k in ("dam-1", "dam-2", "dam-3") if ids[k] in by_graph]
        assert surfaced, [h.get("title") for h in handles]
        for k in surfaced:
            assert by_graph[ids[k]]["topic"] == TOPIC_DAM, by_graph[ids[k]]
        # Every handle carries the field: the key, or None for a record in no topic (identity, here).
        assert all("topic" in h for h in handles), [h for h in handles if "topic" not in h]
        assert {h["topic"] for h in handles} <= {TOPIC_DAM, "ex:community-bbbb2222gar0", None}
        assert any(h["topic"] is None for h in handles if h.get("admission") == "self")
        # ONE membership read for the recall, over the handles' graph ids and the entity's own ladder; no payload read.
        assert len(fake.calls) == 1, fake.calls
        assert fake.calls[0]["record_ids"] == _graph_ids(handles)
        assert fake.calls[0]["owner_id"] == ert.entity_id
        assert sorted(fake.calls[0]["scopes"]) == sorted([("self", EID), ("diary", EID), ("life", EID)])
        assert payload_reads == []
        # Display only: the model's prompt never sees a topic key.
        assert TOPIC_DAM not in json.dumps(llm.calls)
    finally:
        ert.close()


def test_the_stamp_costs_one_membership_read_per_recall_on_memory_s_real_topics(tmp_path: Path, monkeypatch) -> None:
    """Memory's REAL pass forms the topics and its REAL `community_keys_of` answers: over two turns (two recalls) the
    stamp makes exactly one `community_keys_of` call per recall, with that recall's graph ids, and never reads the
    communities payload; each handle's topic is the one that payload gives (members, then provisional ids)."""
    import abstractmemory.community_index as community_index
    from abstractmemory.communities import communities_pass

    from abstractruntime.identity.authoring import capability_formation_author

    pytest.importorskip("numpy")

    class _Titler:
        model_info = {"provider": "fake", "model": "summarizer"}

        def generate(self, *, prompt: str, system_prompt: str, response_model: Any) -> Any:
            return response_model(title="A topic", summary="What it holds.")

    ladder = [("self", EID), ("diary", EID), ("life", EID)]
    home = _home(tmp_path)
    try:
        receipt = communities_pass(home.ms, scopes=ladder, owner_id=EID,
                                   author=capability_formation_author(lambda _n: _Titler(), home.artifacts,
                                                                      should_stop=lambda: False,
                                                                      entity_name=home.name),
                                   cancelled=lambda: False)
        assert receipt["version_seq"] is not None and receipt["communities"] >= 1, receipt
        expected: Dict[str, str] = {}
        served = community_index.communities(home.ms, scopes=ladder, owner_id=EID)["communities"]
        for c in served:
            expected.update({m["id"]: c["key"] for m in c["members"]})
        for c in served:
            for record_id in c["provisional_ids"]:
                expected.setdefault(record_id, c["key"])
        home_dir = home.home_dir
    finally:
        home.close()

    keys_calls: List[List[str]] = []
    real_keys_of = community_index.community_keys_of

    def _counted_keys_of(system: Any, record_ids: Any, *, scopes: Any, owner_id: str) -> Dict[str, Any]:
        asked = list(record_ids)
        keys_calls.append(asked)
        return real_keys_of(system, asked, scopes=scopes, owner_id=owner_id)

    monkeypatch.setattr(community_index, "community_keys_of", _counted_keys_of)
    payload_reads = _count_payload_reads(monkeypatch, community_index)
    ert, wf, rid, _llm = _react_visit(home_dir, [{"content": "The gauges were fine.", "tool_calls": []},
                                                 {"content": "The beds are wet.", "tool_calls": []}])
    try:
        _say(ert, wf, rid, "Do you remember the dam spillway gauges?")
        _say(ert, wf, rid, "And the garden moisture sensors?")
        recalls = [r["result"] for r in _records(ert, rid) if r["node_id"] == "RECALL" and r["status"] == "completed"]
        assert len(recalls) == 2
        assert keys_calls == [_graph_ids(r["handles"]) for r in recalls], keys_calls
        assert payload_reads == [], "the stamp read memory's communities payload"
        stamped = [h for r in recalls for h in r["handles"]]
        assert any(h["topic"] is not None for h in stamped), stamped
        for h in stamped:
            graph_id = (h.get("provenance") or {}).get("record_id")
            assert h["topic"] == (expected.get(graph_id) if graph_id else None), (h.get("title"), h["topic"])
        assert not any("#FALLBACK recall topics" in w for r in recalls for w in r.get("warnings") or [])
    finally:
        ert.close()


def test_a_topics_read_that_fails_leaves_the_recall_whole_and_says_why(tmp_path: Path, monkeypatch) -> None:
    import abstractmemory.community_index as community_index

    home = _home(tmp_path)
    home_dir = home.home_dir
    home.close()

    def _broken(*_a: Any, **_k: Any) -> Dict[str, Any]:
        raise community_index.CommunitySchemaError("community record ex:c1 carries no attributes.community_key")

    monkeypatch.setattr(community_index, "community_keys_of", _broken)
    ert, wf, rid, _llm = _react_visit(home_dir, [{"content": "Yes.", "tool_calls": []}])
    try:
        _say(ert, wf, rid, "Do you remember the dam spillway gauges?")
        result = next(r for r in _records(ert, rid)
                      if r["node_id"] == "RECALL" and r["status"] == "completed")["result"]
        assert result["handles"], "the recall itself stands"
        assert all("topic" not in h for h in result["handles"])
        assert any("#FALLBACK recall topics unavailable" in w and "community_key" in w
                   for w in result.get("warnings") or []), result.get("warnings")
    finally:
        ert.close()


# --------------------------------------------------------------------------------------- 2. search_memory hits


def test_a_memory_search_result_carries_its_hit_count(tmp_path: Path) -> None:
    from test_entity_brain_effects import _dispatch, _result

    from abstractruntime.identity.memory_reader import HomeMemoryReader, SearchMemoryText
    from abstractruntime.identity.tools import ToolElection, execute_tool_elections

    home = _home(tmp_path)
    try:
        reader = HomeMemoryReader(home)
        text = reader.search_memory("spillway")
        assert isinstance(text, SearchMemoryText) and text.hits == 3  # dam-1, dam-2, dam-3 (graph); no book entry
        assert "Your graph: 3 match(es)" in text
        assert reader.search_memory("zeppelin").hits == 0
        assert not hasattr(reader.search_memory(""), "hits")  # no words: no search ran, no count

        election = ToolElection(name="search_memory", body="garden")
        execute_tool_elections([election], diary_store=home.diary, diary_read_effect=lambda _i: {},
                               search_memory_fn=reader.search_memory)
        assert election.hits == 2 and type(election.result) is str  # gar-1, gar-2; the result stays plain text
        other = ToolElection(name="recent_memories", body="")
        execute_tool_elections([other], diary_store=home.diary, diary_read_effect=lambda _i: {},
                               recent_memories_fn=reader.recent_memories)
        assert other.hits is None

        out = _result(_dispatch(home.handlers, EffectType.ENTITY_TOOLS_EXECUTE, {
            "phase": "visit", "tool_calls": [{"name": "search_memory", "arguments": {"query": "spillway"}},
                                             {"name": "recent_memories", "arguments": {}}]}))
        assert out["tools_ran"] == ["search_memory", "recent_memories"]
        assert out["results"][0]["hits"] == 3 and "Your graph: 3 match(es)" in out["results"][0]["result"]
        assert "hits" not in out["results"][1]
    finally:
        home.close()


# ------------------------------------------------------------------------------- 3. the loop position on reason


def _reason_payloads(ert, rid: str) -> List[Dict[str, Any]]:
    return [(r.get("effect") or {}).get("payload") or {} for r in _records(ert, rid)
            if r["node_id"] == "reason" and r["status"] == "started"
            and (r.get("effect") or {}).get("type") == "llm_call"]


def test_the_reason_record_says_which_step_of_how_many(tmp_path: Path) -> None:
    home = _home(tmp_path)
    home_dir = home.home_dir
    home.close()
    ert, wf, rid, llm = _react_visit(home_dir, [
        {"content": "", "tool_calls": [{"name": "read_file", "arguments": {"path": "notes.txt"}, "call_id": "c1"}]},
        {"content": "I read it.", "tool_calls": []},
        {"content": "Hello again.", "tool_calls": []},
    ])
    try:
        _say(ert, wf, rid, "Read notes.txt please.")
        _say(ert, wf, rid, "Thanks.")
        steps = [(p.get("iteration"), p.get("max_iterations")) for p in _reason_payloads(ert, rid)]
        # Turn 1: two model calls of the loop; turn 2 starts at step 1 again.
        assert steps == [(1, DEFAULT_MAX_ITERATIONS), (2, DEFAULT_MAX_ITERATIONS), (1, DEFAULT_MAX_ITERATIONS)]
        # The started and the terminal record of a step carry the same position.
        done = [((r.get("effect") or {}).get("payload") or {}).get("iteration") for r in _records(ert, rid)
                if r["node_id"] == "reason" and r["status"] == "completed"]
        assert done == [1, 2, 1]
        # Metadata only: the prompt the model reads never says it.
        assert not any("max_iterations" in json.dumps(c.get("messages")) for c in llm.calls)
    finally:
        ert.close()


def test_the_reason_record_carries_a_door_set_cap(tmp_path: Path) -> None:
    home = _home(tmp_path)
    home_dir = home.home_dir
    home.close()
    ert, wf, rid, _llm = _react_visit(home_dir, [{"content": "Fine.", "tool_calls": []}],
                                      start_vars={"_limits": {"max_iterations": 7}})
    try:
        _say(ert, wf, rid, "How are you?")
        assert [(p.get("iteration"), p.get("max_iterations")) for p in _reason_payloads(ert, rid)] == [(1, 7)]
    finally:
        ert.close()
