"""0064 fix 1 pins: tool_calls join the prompt-cache message fingerprint.

The three requirements from the backlog item, each pinned:
- NO FALSE MATCH: differing tool_calls never share a fingerprint.
- NO NEW FALSE MISMATCH: an unchanged already-sent message keeps its
  fingerprint across turns — including across the run-store JSON
  round-trip and dict key-order permutation (canonical serialization).
- ZERO CHURN for plain messages: a message without tool_calls keeps the
  pre-fix fingerprint shape (upgrade causes no rebuild for tool-less
  sessions).
"""
from __future__ import annotations

import hashlib
import json

from abstractruntime.integrations.abstractcore.llm_client import (
    _prompt_cache_message_fingerprint as fp,
)


def _tc(call_id: str = "call_1", args: str = '{"path": "a.txt"}') -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read_file", "arguments": args},
    }


def test_no_false_match_on_differing_tool_calls() -> None:
    base = {"role": "assistant", "content": ""}
    a = dict(base, tool_calls=[_tc(args='{"path": "a.txt"}')])
    b = dict(base, tool_calls=[_tc(args='{"path": "b.txt"}')])
    assert fp(a) != fp(b), "same content head, different calls - must differ"
    assert fp(base) != fp(a), "tool-less vs tool-carrying must differ"


def test_no_false_mismatch_across_json_round_trip_and_key_order() -> None:
    msg = {"role": "assistant", "content": "done", "tool_calls": [_tc()]}
    # The run-store replay path: durable JSON round-trip.
    round_tripped = json.loads(json.dumps(msg))
    assert fp(msg) == fp(round_tripped)
    # Key-order permutation (canonicalization kills order drift).
    permuted = {
        "tool_calls": [
            {"function": {"arguments": '{"path": "a.txt"}', "name": "read_file"},
             "type": "function", "id": "call_1"}
        ],
        "content": "done",
        "role": "assistant",
    }
    assert fp(msg) == fp(permuted)


def test_plain_messages_keep_the_pre_fix_shape() -> None:
    msg = {"role": "user", "content": "hello"}
    legacy_payload = {"role": "user", "content": "hello"}
    legacy = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert fp(msg) == legacy, "no churn for tool-less messages on upgrade"
    # Empty/None tool_calls read as absent (falsy) - same fingerprint.
    assert fp(dict(msg, tool_calls=None)) == legacy
    assert fp(dict(msg, tool_calls=[])) == legacy


def test_unserializable_tool_calls_degrade_deterministically() -> None:
    class Weird:
        def __str__(self) -> str:
            return "weird-call"

    a = {"role": "assistant", "content": "", "tool_calls": [Weird()]}
    b = {"role": "assistant", "content": "", "tool_calls": [Weird()]}
    assert fp(a) == fp(b), "default=str keeps exotic shapes stable"
