"""The history window over multimodal messages (0.7.1).

A message's content may be an OpenAI-style part list (`[{"type": "text", ...},
{"type": "image_url", ...}]`, the shape AbstractCore's providers and the
runtime's own media path build). The window's drop notice must go in as a TEXT
PART — prefixing it to the content used to turn the whole list into a string,
so the model received the image's base64 as text. And the window's token
estimate must not count that base64 as text either.
"""

from __future__ import annotations

import copy

from abstractruntime.memory.token_budget import MEDIA_PART_TOKEN_ESTIMATE, estimate_message_tokens
from abstractruntime.session_history import announce_dropped, fold_history_window, window_transcript

_B64 = "iVBORw0KGgo" + "A" * 200_000 + "=="
_IMAGE = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_B64}", "detail": "high"}}


def _report(dropped: int = 2) -> dict:
    return {**fold_history_window([], max_tokens=10)[1], "dropped_messages": dropped, "dropped_tokens": 9}


def _notice_of(report: dict) -> str:
    probe = [{"role": "user", "content": ""}]
    announce_dropped(probe, report, stamp_metadata=False)
    return probe[0]["content"].rstrip("\n")


def test_list_content_keeps_its_image_and_gains_one_notice_text_part() -> None:
    image = copy.deepcopy(_IMAGE)
    messages = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, image]}]
    announce_dropped(messages, _report(), stamp_metadata=False)

    content = messages[0]["content"]
    assert isinstance(content, list) and len(content) == 3
    notice, text, img = content
    assert notice["type"] == "text" and notice["text"].startswith("[#TRUNCATION: 2 earlier message(s)")
    assert set(notice) == {"type", "text"}
    assert text == {"type": "text", "text": "what is this?"}
    assert img == _IMAGE and img["image_url"]["url"].endswith(_B64)
    assert all(_B64 not in p.get("text", "") for p in content)


def test_list_content_with_a_stamped_head_keeps_the_envelope_first() -> None:
    from abstractruntime.turn_grounding import message_carries_grounding_envelope, stamp_user_turn_grounding

    stamped = [{"role": "user", "content": "hello"}]
    assert stamp_user_turn_grounding(stamped)
    head_text = stamped[0]["content"]
    envelope = head_text[: head_text.index("hello")]

    messages = [{"role": "user", "content": [{"type": "text", "text": head_text}, copy.deepcopy(_IMAGE)]}]
    report = _report()
    announce_dropped(messages, report, stamp_metadata=False)

    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[-1] == _IMAGE
    texts = [p["text"] for p in content if p.get("type") == "text"]
    # The envelope still heads the first text part, exactly once in the message.
    assert texts[0].startswith(envelope.rstrip("\n")) and "hello" not in texts[0]
    assert sum(t.count("<runtime_metadata>") for t in texts) == 1
    assert message_carries_grounding_envelope(messages[0])
    # One text part carries the notice, then the turn's own words.
    assert texts[1] == _notice_of(report)
    assert texts[2] == "hello"
    # Joined the way providers join text parts, it reads as the string path does.
    as_string = [{"role": "user", "content": head_text}]
    announce_dropped(as_string, report, stamp_metadata=False)
    assert "\n".join(texts) == as_string[0]["content"]


def test_string_content_is_unchanged() -> None:
    report = _report()
    messages = [{"role": "user", "content": "hello"}]
    announce_dropped(messages, report, stamp_metadata=False)
    assert messages[0]["content"] == (
        "[#TRUNCATION: 2 earlier message(s) of this session (~9 tokens) were dropped from replay by the "
        "history window (the most recent 10 tokens, whole turns; abstractruntime.session_history); "
        "this history starts mid-conversation]\nhello"
    )


def test_list_content_without_a_text_part_gains_the_notice_first() -> None:
    messages = [{"role": "user", "content": [copy.deepcopy(_IMAGE)]}]
    announce_dropped(messages, _report(), stamp_metadata=False)
    content = messages[0]["content"]
    assert len(content) == 2 and content[0]["type"] == "text" and content[1] == _IMAGE


def test_unknown_content_type_is_left_untouched() -> None:
    odd = {"custom": "shape"}
    messages = [{"role": "user", "content": odd}]
    announce_dropped(messages, _report())
    assert messages[0]["content"] is odd
    assert messages[0]["metadata"]["replay_truncated"] is True


def test_window_transcript_keeps_an_image_turn_as_parts() -> None:
    messages = [
        {"role": "user", "content": "old " * 2000},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": [{"type": "text", "text": "look"}, copy.deepcopy(_IMAGE)]},
        {"role": "assistant", "content": "a cat"},
    ]
    out = window_transcript(messages, max_tokens=2000)
    assert out.report["dropped_messages"] == 2
    head = out[0]["content"]
    assert isinstance(head, list) and head[-1] == _IMAGE
    assert head[0]["text"].startswith("[#TRUNCATION:")
    assert messages[2]["content"][0] == {"type": "text", "text": "look"}  # input not mutated
    assert len(messages[2]["content"]) == 2


def test_token_estimate_counts_an_image_part_once_not_its_base64() -> None:
    message = {"role": "user", "content": [{"type": "text", "text": "look"}, copy.deepcopy(_IMAGE)]}
    tokens = estimate_message_tokens(message)
    text_only = estimate_message_tokens({"role": "user", "content": "look"})
    assert tokens == text_only + MEDIA_PART_TOKEN_ESTIMATE
    # The base64 alone is ~50k tokens as text: it would push every older turn out.
    assert tokens < 1000


def test_token_estimate_counts_every_text_part() -> None:
    parts = [{"type": "text", "text": "alpha " * 100}, {"type": "text", "text": "beta " * 100}]
    many = estimate_message_tokens({"role": "user", "content": parts})
    one = estimate_message_tokens({"role": "user", "content": "alpha " * 100})
    assert many > one


def test_token_estimate_counts_a_non_media_part_by_its_text_not_flat() -> None:
    # An Anthropic-style tool_result or thinking part is text, however it is typed:
    # it must never be counted as a flat media part.
    big = "x " * 50_000
    tool_part = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]}
    as_text = estimate_message_tokens({"role": "user", "content": big})
    assert estimate_message_tokens(tool_part) >= as_text
    empty_text = {"role": "user", "content": [{"type": "text"}]}
    assert estimate_message_tokens(empty_text) < MEDIA_PART_TOKEN_ESTIMATE
