"""Regression: a session with NO attachments must never be told it has some.

2026-08-21 incident (session `acode-bc425138014f`, run `6673613e…`): the
session attachment index rendered its header and its "open them like this"
hint for an EMPTY entry list, and the ReAct system prompt turns that header
into an instruction ("If you see 'Stored session attachments' … use the
attachment-open tool with artifact_id"). The model was given no ids and no way
to list them, so it invented `a1`, `a2`, `a3` and the placeholder word
`attachment`; all 12 calls failed with a message that read as "wrong
identifier, try another one"; three identical batches then tripped the
stuck-streak detector and ended the turn at cycle 12 of a 50-iteration budget.

Every test here fails if the corresponding guard is removed.
"""

from __future__ import annotations

import hashlib

import pytest

from abstractruntime import Effect, EffectType, RunState
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.integrations.abstractcore.session_attachments import (
    execute_open_attachment,
    render_session_attachments_system_message,
    session_memory_owner_run_id,
)
from abstractruntime.storage.artifacts import InMemoryArtifactStore


def _register(store: InMemoryArtifactStore, sid: str, name: str, content: bytes) -> str:
    sha = hashlib.sha256(content).hexdigest()
    meta = store.store(
        content,
        content_type="text/plain",
        run_id=session_memory_owner_run_id(sid),
        tags={"kind": "attachment", "path": name, "filename": name, "session_id": sid, "sha256": sha},
    )
    return str(meta.artifact_id)


@pytest.mark.basic
def test_empty_index_renders_nothing_not_a_header() -> None:
    assert render_session_attachments_system_message([]) == ""
    # The hint is the invitation; it must not survive an empty list either.
    assert render_session_attachments_system_message([], include_open_attachment_hint=True) == ""


@pytest.mark.basic
def test_index_with_entries_still_renders_header_hint_and_entry() -> None:
    """Over-fixing guard: the empty case must go silent, the real case must not."""
    msg = render_session_attachments_system_message(
        [{"handle": "notes.txt", "artifact_id": "abc123", "content_type": "text/plain", "size_bytes": 12}]
    )
    assert msg.startswith("Stored session attachments")
    assert "open_attachment(artifact_id=" in msg
    assert "- notes.txt (id=abc123" in msg


@pytest.mark.basic
def test_entries_that_all_fail_to_render_produce_no_header() -> None:
    """Malformed items are skipped one by one; skipping them all is still empty."""
    assert render_session_attachments_system_message([None, 42, "nope"]) == ""


@pytest.mark.basic
def test_llm_call_injects_no_attachment_message_when_session_has_none() -> None:
    """The seam that actually bit: the injector gates on `if active_msg or session_msg`."""
    store = InMemoryArtifactStore()
    captured: dict = {}

    class _StubLLM:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return {"content": "ok", "metadata": {}}

    run = RunState.new(workflow_id="wf", entry_node="n1", session_id="s-empty", vars={})
    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "hello",
            "tools": [{"name": "open_attachment", "parameters": {}}],
            "params": {"temperature": 0.0},
        },
    )
    outcome = make_llm_call_handler(llm=_StubLLM(), artifact_store=store)(run, effect, None)
    assert outcome.status == "completed"

    msgs = captured.get("messages") or []
    joined = "\n".join(str(m.get("content") or "") for m in msgs if isinstance(m, dict))
    assert "Stored session attachments" not in joined


@pytest.mark.basic
def test_open_attachment_on_empty_session_says_so_and_closes_the_door() -> None:
    ok, out, err = execute_open_attachment(
        artifact_store=InMemoryArtifactStore(),
        session_id="s-empty",
        artifact_id="a1",
        handle="attachment",
        expected_sha256=None,
        start_line=1,
        end_line=220,
        max_chars=12000,
    )
    assert ok is False
    # Load-bearing: effect_handlers' read_file probe keys on this exact string
    # to fall through to a real filesystem read.
    assert err == "attachment not found"
    rendered = str((out or {}).get("rendered") or "")
    assert "no stored attachments right now" in rendered
    assert "stop retrying it with other identifiers" in rendered
    # It must name what the CALLER asked for, not the rewritten handle only.
    assert "a1" in rendered
    # And it must NOT claim a permanent truth: read_file registers what it
    # reads as a session attachment, so "never call this again in this
    # session" would be falsified by the very remedy this message names.
    assert "again in this session" not in rendered
    assert "nothing was ever attached" not in rendered


@pytest.mark.basic
def test_index_that_cannot_fit_reports_the_count_instead_of_going_silent() -> None:
    """Empty and "did not fit" are different facts. Silence is right for the
    first and a lie for the second — the model would conclude there is
    nothing to ask for."""
    entry = {"handle": "notes.txt", "artifact_id": "abc123", "content_type": "text/plain", "size_bytes": 12}
    msg = render_session_attachments_system_message([entry], max_chars=150)
    assert msg, "a session WITH attachments must never render nothing"
    assert "Stored session attachments: 1" in msg
    assert "did not fit" in msg
    assert len(msg) <= 150, "the fallback line respects the caller's budget too"
    # And when even the one-liner does not fit, silence beats a fragment.
    assert render_session_attachments_system_message([entry], max_chars=40) == ""
    # No invented identifiers to chase.
    assert "abc123" not in msg and "open_attachment(" not in msg


@pytest.mark.basic
def test_open_attachment_miss_lists_the_real_attachments() -> None:
    store = InMemoryArtifactStore()
    sid = "s2"
    aid = _register(store, sid, "notes.txt", b"hello\n")
    _register(store, sid, "plan.md", b"plan\n")

    ok, out, err = execute_open_attachment(
        artifact_store=store,
        session_id=sid,
        artifact_id="a1",
        handle="zzz-nothing-like-it",
        expected_sha256=None,
        start_line=1,
        end_line=10,
        max_chars=2000,
    )
    assert ok is False and err == "attachment not found"
    rendered = str((out or {}).get("rendered") or "")
    assert "This session's attachments:" in rendered
    assert "notes.txt" in rendered and "plan.md" in rendered
    assert aid in rendered


@pytest.mark.basic
def test_multiple_matches_lists_every_candidate_not_just_the_first() -> None:
    """The disambiguation `return` used to sit inside its own `for` loop."""
    store = InMemoryArtifactStore()
    sid = "s3"
    a1 = _register(store, sid, "dup.txt", b"one\n")
    a2 = _register(store, sid, "dup.txt", b"two\n")

    ok, out, err = execute_open_attachment(
        artifact_store=store,
        session_id=sid,
        artifact_id=None,
        handle="dup.txt",
        expected_sha256=None,
        start_line=1,
        end_line=10,
        max_chars=2000,
    )
    assert ok is False and err == "multiple matches"
    cands = (out or {}).get("candidates") or []
    assert len(cands) == 2
    ids = {str(c.get("artifact_id")) for c in cands}
    assert ids == {a1, a2}
    assert a1 in str((out or {}).get("rendered") or "")
