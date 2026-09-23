"""Conversational prompt-cache byte-stability (mission A, 2026-09-22).

THE INVARIANT: the bytes a user turn is SENT with are the bytes it is STORED
with, so turn N's rendered prompt stays an exact byte-PREFIX of turn N+1's.
That prefix relation is the single precondition every prefix cache needs
(provider prompt caching, KV reuse, mlx-vlm's APC — a hybrid model can only
restore an EXACTLY stored prefix).

What used to break it, measured through the real stack on the MLX native lane
(`untracked/missionA/replay_runtime.py`, 4B pair, 8 turns): the runtime stamped
a `<runtime_metadata>` envelope into the user turn at the PAYLOAD boundary on
every call and stored none of it, so the SAME turn was replayed one turn later
without the envelope. The divergence sat at the first byte of the previous user
message: 185-388 tokens re-prefilled EVERY turn, growing with the conversation.
After: turn N's prompt is a full prefix of turn N+1's and 105-121 tokens are fed
(the reply, the new question, and the template).

These tests need no model: they pin the byte contract at each seam that can
break it, and the end-to-end prefix relation over the session replay path.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_client
from abstractruntime import (
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunState,
    RunStatus,
    session_chat_messages,
)
from abstractruntime.turn_grounding import (
    message_carries_grounding_envelope,
    stamp_user_turn_grounding,
    strip_turn_grounding,
)

pytestmark = pytest.mark.basic


def _grounding(second: int) -> Dict[str, Any]:
    return llm_client._mark_grounding_prompt_injected(
        {
            "local_datetime": f"2000-01-01T00:00:{second:02d}+01:00",
            "country": "FR",
            "display": f"[2000-01-01 00:00:{second:02d} FR]",
        },
        True,
    )


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------
# 1. The stamp itself
# --------------------------------------------------------------------------


def test_stamp_writes_once_and_is_byte_idempotent() -> None:
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "identify yourself"},
    ]
    assert stamp_user_turn_grounding(messages, grounding=_grounding(1)) is True
    stamped = messages[2]["content"]
    assert stamped.startswith("<runtime_metadata>")
    assert stamped.endswith("identify yourself")
    assert message_carries_grounding_envelope(messages[2])
    assert strip_turn_grounding(stamped) == "identify yourself"

    # Earlier turns are never touched: only the CURRENT user turn is stamped.
    assert messages[0] == {"role": "user", "content": "first"}

    # A second pass (next iteration of the same turn, a later normalization,
    # a resumed run) must change nothing at all — not one byte, not the second.
    assert stamp_user_turn_grounding(messages, grounding=_grounding(59)) is False
    assert messages[2]["content"] == stamped


def test_stamp_skips_when_there_is_no_plain_text_user_turn() -> None:
    assert stamp_user_turn_grounding([]) is False
    assert stamp_user_turn_grounding([{"role": "assistant", "content": "hi"}]) is False
    # Content-part lists (media turns) belong to the payload boundary.
    parts = [{"role": "user", "content": [{"type": "text", "text": "look"}]}]
    assert stamp_user_turn_grounding(parts) is False


# --------------------------------------------------------------------------
# 2. Normalization keeps what it finds
# --------------------------------------------------------------------------


def test_chat_shape_keeps_a_stamped_turn_byte_for_byte() -> None:
    """Two passes with DIFFERENT clocks must render the same bytes.

    Both the runtime ledger pass (`core/runtime.py`) and the LLM client pass
    normalize the same payload; a strip-and-reinject there made the ledger and
    the wire disagree, and made the same turn re-render differently next turn.
    """
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "what time is it?"},
    ]
    stamp_user_turn_grounding(messages, grounding=_grounding(5))
    stamped = messages[2]["content"]

    _, once = llm_client._normalize_turn_grounding(prompt="", messages=messages, grounding=_grounding(11))
    _, twice = llm_client._normalize_turn_grounding(prompt="", messages=once, grounding=_grounding(42))

    assert isinstance(once, list) and isinstance(twice, list)
    assert once[2]["content"] == stamped
    assert _dumps(twice) == _dumps(once)
    assert "00:00:11" not in _dumps(once) and "00:00:42" not in _dumps(twice)


def test_chat_shape_still_stamps_an_unstamped_turn() -> None:
    """Hosts that stamp nothing keep the old behaviour, one pass later."""
    messages = [{"role": "user", "content": "what time is it?"}]
    _, out = llm_client._normalize_turn_grounding(prompt="", messages=messages, grounding=_grounding(5))
    assert isinstance(out, list) and len(out) == 1
    assert str(out[0]["content"]).startswith("<runtime_metadata>")
    assert "00:00:05" in str(out[0]["content"])


def _tool_loop(user_content: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": "Checking the workspace.",
            "tool_calls": [{"type": "function", "id": "call_1", "function": {"name": "list_files", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "[list_files]: a.txt", "tool_call_id": "call_1"},
    ]


def test_tool_loop_with_a_stamped_turn_appends_no_trailer() -> None:
    """A stamped turn IS this turn's grounding — nothing rides the tail.

    That removes ~50 tokens of per-iteration entropy at the very END of the
    prompt, which is exactly where it costs the most: the end-of-prompt
    snapshot is the only one that restores a FULL prefix.
    """
    messages = _tool_loop("Create a project folder")
    stamp_user_turn_grounding(messages, grounding=_grounding(1))
    baseline = _dumps(messages)

    _, out = llm_client._normalize_turn_grounding(prompt="", messages=messages, grounding=_grounding(2))
    assert isinstance(out, list)
    assert len(out) == len(messages), "a stamped tool-loop turn must not grow the message list"
    assert _dumps(out) == baseline
    assert "00:00:02" not in _dumps(out)


def test_tool_loop_without_a_stamp_keeps_the_trailing_envelope() -> None:
    """Unstamped hosts keep the 0212 placement: entropy at the tail, never at
    message[0]."""
    messages = _tool_loop("Create a project folder")
    _, out = llm_client._normalize_turn_grounding(prompt="", messages=messages, grounding=_grounding(3))
    assert isinstance(out, list)
    assert len(out) == len(messages) + 1
    assert llm_client._is_runtime_grounding_only_user_message(out[-1])
    assert "<runtime_metadata>" not in str(out[0]["content"])


def test_stacked_legacy_artifacts_are_still_normalized_away() -> None:
    """"Already stamped" means ONE well-formed head envelope — not a pile."""
    doubled = (
        '<runtime_metadata>{"local_datetime":"a"}</runtime_metadata>\n'
        '<runtime_metadata>{"local_datetime":"b"}</runtime_metadata>\n'
        "real task"
    )
    messages = [{"role": "user", "content": doubled}]
    _, out = llm_client._normalize_turn_grounding(prompt="", messages=messages, grounding=_grounding(7))
    assert isinstance(out, list)
    content = str(out[0]["content"])
    assert content.count("<runtime_metadata>") == 1
    assert content.endswith("real task")
    assert "00:00:07" in content


# --------------------------------------------------------------------------
# 3. Session replay returns the stored bytes
# --------------------------------------------------------------------------


def _completed_turn(*, run_id: str, created_at: str, user_content: str, task: str, answer: str) -> RunState:
    return RunState(
        run_id=run_id,
        workflow_id="basic-agent",
        status=RunStatus.COMPLETED,
        current_node="done",
        vars={
            "context": {
                "task": task,
                # What the adapter durably stored for this turn — envelope included.
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ],
            }
        },
        output={"response": answer},
        error=None,
        created_at=created_at,
        updated_at=created_at,
        actor_id="tester",
        session_id="sess-bytes",
        parent_run_id=None,
        waiting=None,
    )


def test_session_replay_returns_the_stamped_bytes_not_the_display_form() -> None:
    task = "identify yourself"
    stamped = f'<runtime_metadata>{{"display":"[2000-01-01 00:00:01]"}}</runtime_metadata>\n{task}'
    run_store = InMemoryRunStore()
    run_store.save(
        _completed_turn(
            run_id="run-1",
            created_at="2026-01-01T00:00:00+00:00",
            user_content=stamped,
            task=task,
            answer="I am an agent.",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-bytes"
    )
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == stamped, "replay must re-send the bytes the turn was sent with"
    assert messages[1]["content"] == "I am an agent."


def test_session_replay_falls_back_to_the_display_prompt_when_nothing_was_stamped() -> None:
    run_store = InMemoryRunStore()
    run_store.save(
        _completed_turn(
            run_id="run-1",
            created_at="2026-01-01T00:00:00+00:00",
            user_content="identify yourself",
            task="identify yourself",
            answer="I am an agent.",
        )
    )
    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-bytes"
    )
    assert messages[0]["content"] == "identify yourself"


def test_session_replay_finds_the_stamped_turn_in_the_bundle_shape() -> None:
    """The gateway's basic-agent bundle runs its react loop in a SUBRUN.

    The root's own `vars.context.messages` then holds only the SEEDED history —
    the current turn is appended inside the subrun and never appears there. The
    agent node folds the loop's durable transcript back into the root's
    `output.scratchpad.messages`, which is already loaded, so replay reads the
    sent bytes from there at no extra I/O. Without this lane the hermetic
    gateway replayed every turn envelope-free and went `cold` on all 7 turns.
    """
    task = "identify yourself"
    stamped = f'<runtime_metadata>{{"display":"[2000-01-01 00:00:01]"}}</runtime_metadata>\n{task}'
    answer = "I am an agent."
    run = RunState(
        run_id="run-1",
        workflow_id="basic-agent@0.0.4:81795ea9",
        status=RunStatus.COMPLETED,
        current_node="end",
        vars={"context": {"task": task, "messages": []}},  # seeded history only
        output={
            "response": answer,
            "scratchpad": {
                "sub_run_id": "sub-1",
                "task": task,
                "messages": [
                    {"role": "user", "content": stamped},
                    {"role": "assistant", "content": answer},
                ],
            },
        },
        error=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        actor_id="tester",
        session_id="sess-bytes",
        parent_run_id=None,
        waiting=None,
    )
    run_store = InMemoryRunStore()
    run_store.save(run)

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-bytes"
    )
    assert messages[0]["content"] == stamped
    assert messages[1]["content"] == answer


def test_session_replay_never_substitutes_a_later_user_message_for_the_turn() -> None:
    """An ask_user reply or an operator interjection is a LATER user message in
    the same run. The verbatim channel must not hand it back as the turn's
    question — it only upgrades the message that IS the turn's question."""
    task = "identify yourself"
    run = _completed_turn(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        user_content=task,
        task=task,
        answer="I am an agent.",
    )
    run.vars["context"]["messages"].append(
        {"role": "user", "content": '<runtime_metadata>{"display":"[x]"}</runtime_metadata>\n[User response]: yes'}
    )
    run_store = InMemoryRunStore()
    run_store.save(run)

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-bytes"
    )
    assert messages[0]["content"] == task


# --------------------------------------------------------------------------
# 4. End to end: the prefix relation, without a model
# --------------------------------------------------------------------------


def test_turn_n_messages_are_an_exact_prefix_of_turn_n_plus_1() -> None:
    """The whole point, asserted on bytes.

    Turn 1 is built, stamped and normalized exactly as the payload boundary
    does it; the run is stored; turn 2 seeds its history through the SAME
    session-replay path the gateway uses (`_seed_session_history` ->
    `session_chat_messages`) and is normalized in turn. Turn 2's message list
    must start with turn 1's, byte for byte, envelope included.
    """
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()

    # --- turn 1 ---------------------------------------------------------
    task1 = "identify yourself"
    turn1: List[Dict[str, Any]] = [{"role": "user", "content": task1}]
    stamp_user_turn_grounding(turn1, grounding=_grounding(1))
    _, sent1 = llm_client._normalize_turn_grounding(prompt="", messages=turn1, grounding=_grounding(2))
    assert isinstance(sent1, list)

    answer1 = "I am an autonomous agent."
    run_store.save(
        _completed_turn(
            run_id="run-1",
            created_at="2026-01-01T00:00:00+00:00",
            user_content=turn1[0]["content"],
            task=task1,
            answer=answer1,
        )
    )

    # --- turn 2 (history reconstructed server-side, as the gateway does) ---
    task2 = "what is your key purpose?"
    seeded = session_chat_messages(run_store=run_store, ledger_store=ledger_store, session_id="sess-bytes")
    turn2 = [{k: v for k, v in m.items() if k in {"role", "content"}} for m in seeded]
    turn2.append({"role": "user", "content": task2})
    stamp_user_turn_grounding(turn2, grounding=_grounding(30))
    _, sent2 = llm_client._normalize_turn_grounding(prompt="", messages=turn2, grounding=_grounding(31))
    assert isinstance(sent2, list)

    assert len(sent2) == len(sent1) + 2
    assert _dumps(sent2[: len(sent1)]) == _dumps(sent1), "turn 1's payload must reappear byte-identical"
    assert sent2[len(sent1)] == {"role": "assistant", "content": answer1}
    assert str(sent2[-1]["content"]).endswith(task2)
    # …and the new turn carries its OWN clock, not turn 1's.
    assert "00:00:01" in str(sent1[0]["content"])
    assert "00:00:30" in str(sent2[-1]["content"])
