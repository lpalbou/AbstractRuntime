"""Grounding never lands on adapter-authored messages (mission A3, 2026-09-22).

Live run 081d8daa: the grounding envelope hopped onto a synthesized
`[unpaired tool result]` user carrier on every tool-loop iteration (the carrier
was the payload's last user message, so the payload was read as "chat shape"),
and the next iteration rebuilt that carrier WITHOUT it. Consecutive prompts
differed by exactly one envelope, thousands of tokens before the end, and every
iteration from the third on — and the whole next run — was cold. The hermetic
gateway then showed the same hop onto the `volatile` `[loop] iteration N of M.`
tail. These tests pin the structural rule with two different clocks, so a
re-injection can never hide inside one second.
"""

from __future__ import annotations

import copy
import json

from abstractruntime.integrations.abstractcore.llm_client import (
    _normalize_turn_grounding,
    _strip_synthetic_message_markers,
)
from abstractruntime.turn_grounding import (
    SYNTHETIC_MESSAGE_KEY,
    SYNTHETIC_TOOL_RESULT,
    message_carries_grounding_envelope,
    stamp_user_turn_grounding,
    transcript_carries_grounding_envelope,
)

CLOCK_A = {"display": "[2026-09-22 10:48:10]", "local_datetime": "2026-09-22T10:48:10+02:00"}
CLOCK_B = {"display": "[2026-09-22 10:48:58]", "local_datetime": "2026-09-22T10:48:58+02:00"}


def _d(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def _assistant(ids):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "id": i, "function": {"name": "web_search", "arguments": "{}"}} for i in ids],
    }


def _carrier(text):
    return {"role": "user", "content": f"[unpaired tool result]: {text}", SYNTHETIC_MESSAGE_KEY: SYNTHETIC_TOOL_RESULT}


def _stamped_task():
    msgs = [{"role": "user", "content": "research the debate"}]
    assert stamp_user_turn_grounding(msgs, grounding=CLOCK_A) is True
    return msgs


def test_stamp_refuses_a_carrier_and_targets_the_durable_turn() -> None:
    msgs = [{"role": "user", "content": "task"}, _assistant(["call_1"]), {"role": "tool", "tool_call_id": "call_1", "content": "r"}, _carrier("big")]
    before = copy.deepcopy(msgs[-1])
    assert stamp_user_turn_grounding(msgs, grounding=CLOCK_A) is True
    assert message_carries_grounding_envelope(msgs[0])
    assert msgs[-1] == before, "the carrier is rebuilt every iteration; a stamp in it is a stamp lost"
    assert stamp_user_turn_grounding(msgs, grounding=CLOCK_B) is False, "stamp once"


def test_carrier_is_byte_identical_across_two_reason_boundaries() -> None:
    task = _stamped_task()
    it_n = task + [_assistant(["call_1"]), {"role": "tool", "tool_call_id": "call_1", "content": "r1"}, _carrier("ghost-1")]
    it_n1 = copy.deepcopy(it_n) + [_assistant(["call_1"]), {"role": "tool", "tool_call_id": "call_1", "content": "r2"}, _carrier("ghost-2")]
    _, out_n = _normalize_turn_grounding(prompt="", messages=it_n, grounding=CLOCK_A)
    _, out_n1 = _normalize_turn_grounding(prompt="", messages=it_n1, grounding=CLOCK_B)
    assert _d(out_n) == _d(it_n), "a grounded tool-loop payload is left byte-for-byte alone"
    assert _d(out_n1[: len(out_n)]) == _d(out_n), "iteration N must be an exact message prefix of N+1"
    assert [i for i, m in enumerate(out_n1) if message_carries_grounding_envelope(m)] == [0]


def test_volatile_tail_is_never_the_turn() -> None:
    msgs = _stamped_task() + [_assistant(["call_1"]), {"role": "tool", "tool_call_id": "call_1", "content": "r"},
                              {"role": "user", "content": "[loop] iteration 2 of 8.", "volatile": True}]
    _, out = _normalize_turn_grounding(prompt="", messages=copy.deepcopy(msgs), grounding=CLOCK_B)
    assert _d(out) == _d(msgs)


def test_any_stamped_user_message_grounds_the_tool_loop_no_trailer() -> None:
    # The newest durable user message (an unstamped interjection) is not the one
    # that carries the envelope; the task several messages back does.
    msgs = _stamped_task() + [
        _assistant(["call_1"]),
        {"role": "tool", "tool_call_id": "call_1", "content": "r"},
        {"role": "user", "content": "[Operator guidance] also check X"},
        _assistant(["call_2"]),
        {"role": "tool", "tool_call_id": "call_2", "content": "r2"},
    ]
    assert transcript_carries_grounding_envelope(msgs)
    _, out = _normalize_turn_grounding(prompt="", messages=copy.deepcopy(msgs), grounding=CLOCK_B)
    assert _d(out) == _d(msgs), "no trailer, no rewrite: the turn is already grounded"


def test_an_unstamped_tool_loop_still_gets_its_trailer() -> None:
    # Hosts that never stamp keep the pre-mission behaviour (fresh trailer).
    msgs = [{"role": "user", "content": "task"}, _assistant(["call_1"]), {"role": "tool", "tool_call_id": "call_1", "content": "r"}]
    assert not transcript_carries_grounding_envelope(msgs)
    _, out = _normalize_turn_grounding(prompt="", messages=copy.deepcopy(msgs), grounding=CLOCK_B)
    assert len(out) == len(msgs) + 1 and message_carries_grounding_envelope(out[-1])


def test_a_stamped_carrier_does_not_count_as_grounding() -> None:
    msgs = [{"role": "user", "content": "task"}, _assistant(["call_1"]), {"role": "tool", "tool_call_id": "call_1", "content": "r"},
            {**_carrier("x"), "content": "<runtime_metadata>{}</runtime_metadata>\n[unpaired tool result]: x"}]
    assert not transcript_carries_grounding_envelope(msgs)


def test_markers_are_stripped_before_the_wire() -> None:
    msgs = [{"role": "user", "content": "t"}, _carrier("x"), {"role": "user", "content": "[loop] 2", SYNTHETIC_MESSAGE_KEY: "loop_tail"}]
    out = _strip_synthetic_message_markers(msgs)
    assert all(SYNTHETIC_MESSAGE_KEY not in m for m in out)
    assert SYNTHETIC_MESSAGE_KEY in msgs[1], "callers' structures are never mutated"


def test_an_envelope_merged_behind_the_task_still_grounds_the_loop() -> None:
    # Guidance drained before the first reply is stamped (it was the last user
    # message) and the payload merges it behind the task: the envelope then sits at
    # a paragraph boundary. The loop is grounded — no trailer, no user,user pair.
    msgs = [
        {"role": "user", "content": "Summarize the workspace\n\n<runtime_metadata>{\"display\":\"[x]\"}</runtime_metadata>\n[Operator guidance] be brief"},
        _assistant(["call_1"]),
        {"role": "tool", "tool_call_id": "call_1", "content": "r"},
        {"role": "user", "content": "[loop] iteration 2 of 20.", SYNTHETIC_MESSAGE_KEY: "loop_tail"},
    ]
    assert transcript_carries_grounding_envelope(msgs)
    _, out = _normalize_turn_grounding(prompt="", messages=copy.deepcopy(msgs), grounding=CLOCK_B)
    assert _d(out) == _d(msgs)


def test_the_legacy_trailer_never_creates_a_user_user_pair() -> None:
    msgs = [{"role": "user", "content": "task"}, _assistant(["call_1"]), {"role": "tool", "tool_call_id": "call_1", "content": "r"},
            {"role": "user", "content": "[loop] iteration 2 of 20.", SYNTHETIC_MESSAGE_KEY: "loop_tail"}]
    _, out = _normalize_turn_grounding(prompt="", messages=copy.deepcopy(msgs), grounding=CLOCK_B)
    roles = [m["role"] for m in out]
    assert all(not (a == b == "user") for a, b in zip(roles, roles[1:])), roles
    assert message_carries_grounding_envelope(out[-1])
