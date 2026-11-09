"""Who speaks in a turn (entity mentions, 2026-11-08): `author` on session turns, on the history
replayed to the next agent run, and on the replay view's messages (core.turn_author)."""

from __future__ import annotations

import pytest

from abstractruntime import InMemoryLedgerStore, InMemoryRunStore, RunState, RunStatus, session_chat_messages
from abstractruntime.core.turn_author import (
    answer_author,
    author_for,
    prompt_author,
    speaker_line,
    stamp_turn_authors,
)
from abstractruntime.history_bundle import _best_effort_session_turns
from abstractruntime.replay import replay

KEYS = {"kind", "name", "entity_id", "identity_ref", "avatar"}

NERIN = author_for(
    "entity", "Nerin", entity_id="entity:nerin", identity_ref="entity:nerin#2",
    avatar={"style": "watch-halo", "color": "#336699"},
)
ALICE = author_for("person", "Alice Martin")


def _run(run_id, prompt, answer, *, created_at, vars_extra=None, workflow_id="basic-agent@0.0.5:81795ea9"):
    vars_ = {"prompt": prompt, "context": {"task": prompt, "messages": []}}
    vars_.update(vars_extra or {})
    return RunState(
        run_id=run_id, workflow_id=workflow_id, status=RunStatus.COMPLETED, current_node="done",
        vars=vars_, output={"answer": answer}, created_at=created_at, updated_at=created_at,
        actor_id="gateway", session_id="sess-1",
    )


def _mention_run():
    v = {"_meta": {"origin": {"kind": "person", "actor": "default:alice", "client": "mention", "ref": {}}}}
    stamp_turn_authors(v, author=NERIN, prompt_author=ALICE)
    return _run("run-mention", "@nerin what do you think?", "I think it holds.",
                created_at="2026-11-08T10:01:00+00:00", vars_extra=v, workflow_id="entity-mention")


def _agent_run():
    v = {
        "_meta": {"origin": {"kind": "person", "actor": "default:alice", "client": "assistant", "ref": {}}},
        "workflow_selection": {"workflow_id": "basic-agent@0.0.5:81795ea9", "name": "Basic Agent"},
    }
    return _run("run-agent", "hello", "Hi there.", created_at="2026-11-08T10:00:00+00:00", vars_extra=v)


def _store():
    rs = InMemoryRunStore()
    rs.save(_agent_run())
    rs.save(_mention_run())
    return rs


def test_author_shape_has_every_key_with_nulls_present():
    a = author_for("agent", "Basic Agent")
    assert a == {"kind": "agent", "name": "Basic Agent", "entity_id": None, "identity_ref": None, "avatar": None}
    with pytest.raises(ValueError):
        author_for("model", "x")
    # No identity, no avatar: the keys stay, null.
    bare = author_for("entity", "nerin", entity_id="entity:nerin")
    assert set(bare) == KEYS and bare["identity_ref"] is None and bare["avatar"] is None


def test_fallbacks_read_structured_fields_only():
    # Agent: the entrypoint name the host resolved, never a model name.
    assert answer_author({"workflow_selection": {"name": "Basic Agent"}, "provider": "mlx", "model": "Qwen"})["name"] == "Basic Agent"
    # No selection: the workflow id's bundle id.
    assert answer_author({}, workflow_id="abstractassistant-orchestrator@0.0.17:7d177c93")["name"] == "abstractassistant-orchestrator"
    # A summon run (a finalized, door-signed entity stamp) is the entity's turn.
    signed = {"entity_id": "entity:hypnos", "channel": "summon", "session_id": "s1", "nonce": "n" * 32,
              "run_id": "run-1", "sig": "f" * 64}
    summon = answer_author({"_runtime": {"entity": signed, "identity_ref": {"of": "entity:hypnos", "rev": 3}}})
    assert summon["kind"] == "entity" and summon["entity_id"] == "entity:hypnos" and summon["identity_ref"] == "entity:hypnos#3"


@pytest.mark.parametrize("missing", ["sig", "run_id", "nonce", "channel", "session_id"])
def test_an_unsigned_or_provisional_entity_stamp_never_makes_an_entity_turn(missing):
    """Review 2026-11-09 finding 3 (rt_probe PROBE_D): `{"_runtime": {"entity": {"entity_id":
    "entity:castor"}}}`, which any client could send, labelled the agent's reply as castor's."""
    probe = answer_author({"_runtime": {"entity": {"entity_id": "entity:castor"}}}, workflow_id="basic-agent@1:x")
    assert probe["kind"] == "agent" and probe["name"] == "basic-agent" and probe["entity_id"] is None
    stamp = {"entity_id": "entity:castor", "channel": "summon", "session_id": "s1", "nonce": "n", "run_id": "r", "sig": "s"}
    stamp[missing] = None
    assert answer_author({"_runtime": {"entity": stamp}}, workflow_id="basic-agent@1:x")["kind"] == "agent"
    # Person: the origin's actor; a run older than origins: name null.
    assert prompt_author({"_meta": {"origin": {"kind": "person", "actor": "default:bob"}}}) == author_for("person", "bob")
    assert prompt_author({}) == author_for("person", None)
    assert prompt_author({"_meta": {"origin": {"kind": "trigger", "client": "automation"}}})["kind"] == "agent"


def test_session_turns_carry_prompt_and_answer_authors():
    turns = _best_effort_session_turns(run_store=_store(), ledger_store=InMemoryLedgerStore(), artifact_store=None,
                                       session_id="sess-1", limit=10)
    by_id = {t["run_id"]: t for t in turns}
    assert by_id["run-mention"]["author"] == NERIN
    assert by_id["run-mention"]["prompt_author"] == ALICE
    assert by_id["run-agent"]["author"] == author_for("agent", "Basic Agent")
    assert by_id["run-agent"]["prompt_author"] == author_for("person", "alice")


def test_next_agent_run_history_names_the_entity_as_speaker():
    history = session_chat_messages(run_store=_store(), session_id="sess-1")
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    agent_reply, mention_user, entity_reply = history[1], history[2], history[3]
    # The entity's reply: structured author AND the plain rendering the model reads.
    assert entity_reply["metadata"]["author"] == NERIN
    assert entity_reply["content"] == "Nerin (entity): I think it holds."
    assert mention_user["metadata"]["author"] == ALICE
    # An agent's own reply keeps its exact bytes (prefix caches) and names the agent.
    assert agent_reply["content"] == "Hi there."
    assert agent_reply["metadata"]["author"]["kind"] == "agent"
    assert speaker_line(author_for("agent", "x"), "t") == "t" and speaker_line(None, "t") == "t"


def test_replay_view_messages_carry_the_turns_authors():
    run = _mention_run()
    view = replay({"run-mention": []}, {"run-mention": run}, root_run_id="run-mention").to_json()
    roles = {m["role"]: m for m in view["messages"]}
    assert roles["user"]["author"] == ALICE
    assert roles["assistant"]["author"] == NERIN
    agent = _agent_run()
    view = replay({"run-agent": []}, {"run-agent": agent}, root_run_id="run-agent").to_json()
    for m in view["messages"]:
        assert set(m["author"]) == KEYS
    assert {m["role"]: m["author"]["kind"] for m in view["messages"]} == {"user": "person", "assistant": "agent"}


def test_replay_view_relabels_messages_folded_before_the_snapshot():
    from abstractruntime.replay import apply

    view = replay({"run-mention": []}, {}, root_run_id="run-mention")
    rec = {"run_id": "run-mention", "seq": 1, "step_id": "s1", "node_id": "answer", "status": "completed",
           "effect": {"type": "answer_user", "payload": {"message": "I think it holds."}},
           "result": {"message": "I think it holds."}, "started_at": "2026-11-08T10:01:01+00:00"}
    apply(view, [rec])
    apply(view, [], runs_by_id={"run-mention": _mention_run()})
    msgs = view.to_json()["messages"]
    assert msgs and all(m["author"] in (NERIN, ALICE) for m in msgs)


def test_history_bundle_run_section_carries_the_authors():
    from abstractruntime import export_run_history_bundle

    rs = _store()
    bundle = export_run_history_bundle(run_id="run-mention", run_store=rs, ledger_store=InMemoryLedgerStore(), artifact_store=None)
    assert bundle["run"]["author"] == NERIN and bundle["run"]["prompt_author"] == ALICE


def test_a_repeated_prompt_is_marked_repeat_of_and_kept_whole_for_replay():
    rs = InMemoryRunStore()
    rs.save(_agent_run())
    run = _mention_run()
    run.vars["_meta"]["repeat_of"] = "run-agent"
    rs.save(run)
    turns = {t["run_id"]: t for t in _best_effort_session_turns(
        run_store=rs, ledger_store=InMemoryLedgerStore(), artifact_store=None, session_id="sess-1", limit=10)}
    assert turns["run-mention"]["repeat_of"] == "run-agent" and turns["run-agent"]["repeat_of"] is None
    # Replay keeps the pair whole (the entity's reply is never lost) and marks the repeat.
    history = session_chat_messages(run_store=rs, session_id="sess-1")
    assert len(history) == 4 and history[2]["metadata"]["repeat_of"] == "run-agent"
    assert "repeat_of" not in history[0]["metadata"]
    # The replay view's input message names the turn it repeats; the answer is not marked.
    view = replay({"run-mention": []}, {"run-mention": run}, root_run_id="run-mention").to_json()
    by_role = {m["role"]: m for m in view["messages"]}
    assert by_role["user"]["repeat_of"] == "run-agent" and "repeat_of" not in by_role["assistant"]
    from abstractruntime import export_run_history_bundle

    bundle = export_run_history_bundle(run_id="run-mention", run_store=rs, ledger_store=InMemoryLedgerStore(), artifact_store=None)
    assert bundle["run"]["repeat_of"] == "run-agent"


def test_a_replayed_answer_carries_its_own_time_after_the_question():
    rs = InMemoryRunStore()
    run = _mention_run()
    run.updated_at = "2026-11-08T10:01:09+00:00"
    rs.save(run)
    user, answer = session_chat_messages(run_store=rs, session_id="sess-1")
    assert user["metadata"]["ts"] == "2026-11-08T10:01:00+00:00"
    assert answer["metadata"]["ts"] == "2026-11-08T10:01:09+00:00"


def test_an_entity_rereading_its_own_session_reads_its_replies_bare():
    """Review 2026-11-09 REVIEW-1 #7: a re-summoned entity replayed its own earlier replies as
    "Nerin (entity): ..." (the model may copy the prefix, and the bytes no longer match what it
    said, so every prefix cache of the session missed). The label is for OTHER readers only."""
    rs = _store()
    own = session_chat_messages(run_store=rs, session_id="sess-1", reader_entity_id="entity:nerin")
    # Byte-identical to what the entity said (prompt-cache safe); the author stays structured.
    assert own[3]["content"] == "I think it holds."
    assert own[3]["metadata"]["author"] == NERIN
    # Every other message is unchanged by the reader.
    plain = session_chat_messages(run_store=rs, session_id="sess-1")
    assert [m["content"] for m in own[:3]] == [m["content"] for m in plain[:3]]
    # Another reader (an agent, another entity, nobody named) still sees who spoke.
    for reader in (None, "entity:castor", ""):
        other = session_chat_messages(run_store=rs, session_id="sess-1", reader_entity_id=reader)
        assert other[3]["content"] == "Nerin (entity): I think it holds."
    # The legacy id spelling (entity:<name>@<home_id>) and the bare name are the same reader.
    for same in ("entity:nerin@6f1c2a", "nerin", "Entity:Nerin"):
        assert speaker_line(NERIN, "t", reader_entity_id=same) == "t"
    assert speaker_line(NERIN, "t", reader_entity_id="entity:castor") == "Nerin (entity): t"


def test_an_entitys_own_reply_replays_byte_for_byte_whitespace_included():
    """Codex review 2026-11-09 finding 10: the answer was .strip()-ed before the reader-aware label, so
    "  indented answer\\n\\n" replayed as "indented answer" - not the bytes the entity said."""
    rs = InMemoryRunStore()
    run = _mention_run()
    run.output = {"answer": "  indented answer\n\n- a list item\n\n"}
    rs.save(run)
    own = session_chat_messages(run_store=rs, session_id="sess-1", reader_entity_id="entity:nerin")
    assert own[1]["content"] == "  indented answer\n\n- a list item\n\n"
    other = session_chat_messages(run_store=rs, session_id="sess-1")
    assert other[1]["content"] == "Nerin (entity):   indented answer\n\n- a list item\n\n"
    # A whitespace-only answer is still no answer (the pair is skipped whole).
    run.output = {"answer": "  \n "}
    rs.save(run)
    assert session_chat_messages(run_store=rs, session_id="sess-1", reader_entity_id="entity:nerin") == []
