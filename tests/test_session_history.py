"""Durable session conversation replay — read-side contract tests.

Contract (agora `durable-sessions` v1): the run store is the single durable
source of a session's conversation; `session_chat_messages` reconstructs it
as chat messages for host-side seeding of new runs in the same session.
"""

from __future__ import annotations

from abstractruntime import (
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunState,
    RunStatus,
    SESSION_TURN_KIND,
    session_chat_messages,
)


def _turn_run(
    *,
    run_id: str,
    session_id: str = "sess-1",
    prompt: str = "",
    answer: str = "",
    status: RunStatus = RunStatus.COMPLETED,
    created_at: str = "2026-01-01T00:00:00+00:00",
    workflow_id: str = "wf_chat",
    parent_run_id: str | None = None,
) -> RunState:
    return RunState(
        run_id=run_id,
        workflow_id=workflow_id,
        status=status,
        current_node="done",
        vars={
            "prompt": prompt,
            "context": {"task": prompt, "messages": []},
        },
        output={"response": answer} if answer else {},
        error=None,
        created_at=created_at,
        updated_at=created_at,
        actor_id="tester",
        session_id=session_id,
        parent_run_id=parent_run_id,
        waiting=None,
    )


def test_session_chat_messages_reconstructs_turns_chronologically() -> None:
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(
            run_id="run-1",
            prompt="analyze the PDF",
            answer="The document concludes X.",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    run_store.save(
        _turn_run(
            run_id="run-2",
            prompt="explain simply",
            answer="In short: X.",
            created_at="2026-01-01T00:05:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        session_id="sess-1",
    )

    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages] == [
        "analyze the PDF",
        "The document concludes X.",
        "explain simply",
        "In short: X.",
    ]
    for m in messages:
        assert m["metadata"]["kind"] == SESSION_TURN_KIND
        assert m["metadata"]["run_id"] in {"run-1", "run-2"}
        assert m["metadata"]["ts"]


def test_session_chat_messages_strips_runtime_metadata_envelope() -> None:
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(
            run_id="run-env",
            prompt='<runtime_metadata>{"display":"[t]"}</runtime_metadata>\nreal question',
            answer="real answer",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )

    assert messages[0]["content"] == "real question"


def test_session_chat_messages_skips_incomplete_internal_and_child_runs() -> None:
    run_store = InMemoryRunStore()
    # Contributing turn.
    run_store.save(_turn_run(run_id="run-ok", prompt="q1", answer="a1"))
    # Failed run: half-turn stays invisible (contract v1, question (a)).
    run_store.save(
        _turn_run(
            run_id="run-failed",
            prompt="q-failed",
            answer="",
            status=RunStatus.FAILED,
            created_at="2026-01-01T00:01:00+00:00",
        )
    )
    # Completed but silent (no answer): invisible.
    run_store.save(
        _turn_run(
            run_id="run-silent",
            prompt="q-silent",
            answer="",
            created_at="2026-01-01T00:02:00+00:00",
        )
    )
    # Still RUNNING (mid-turn): invisible — it has not answered yet.
    run_store.save(
        _turn_run(
            run_id="run-running",
            prompt="q-running",
            answer="",
            status=RunStatus.RUNNING,
            created_at="2026-01-01T00:02:30+00:00",
        )
    )
    # Internal bookkeeping run: invisible.
    run_store.save(
        _turn_run(
            run_id="session_memory_sess-1",
            prompt="",
            answer="",
            workflow_id="__session_memory__",
            created_at="2026-01-01T00:03:00+00:00",
        )
    )
    # Child run of a turn: only ROOT runs are turns.
    run_store.save(
        _turn_run(
            run_id="run-child",
            prompt="child prompt",
            answer="child answer",
            parent_run_id="run-ok",
            created_at="2026-01-01T00:04:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )

    assert [m["metadata"]["run_id"] for m in messages] == ["run-ok", "run-ok"]
    # Pairs-only invariant (runtime review A3): strictly alternating
    # user/assistant, never a dangling half-turn.
    assert [m["role"] for m in messages] == ["user", "assistant"]


class _CountingLedgerStore(InMemoryLedgerStore):
    def __init__(self) -> None:
        super().__init__()
        self.list_calls = 0

    def list(self, run_id: str):  # type: ignore[override]
        self.list_calls += 1
        return super().list(run_id)


def test_session_chat_messages_reads_no_ledgers_when_outputs_carry_answers() -> None:
    """Runtime review A1 (cost contract): the seed read is O(turns) run
    loads — the ledger is touched only as a per-run fallback when a
    completed run's output carries no answer."""
    run_store = InMemoryRunStore()
    for i in range(3):
        run_store.save(
            _turn_run(
                run_id=f"run-{i}",
                prompt=f"q{i}",
                answer=f"a{i}",
                created_at=f"2026-01-01T00:{i:02d}:00+00:00",
            )
        )
    ledger_store = _CountingLedgerStore()

    messages = session_chat_messages(
        run_store=run_store, ledger_store=ledger_store, session_id="sess-1"
    )

    assert len(messages) == 6
    assert ledger_store.list_calls == 0


def test_session_chat_messages_works_on_json_file_store(tmp_path) -> None:
    """Same contract on the production file store (the live gateway's)."""
    from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

    run_store = JsonFileRunStore(str(tmp_path))
    run_store.save(_turn_run(run_id="run-1", prompt="q1", answer="a1"))
    run_store.save(
        _turn_run(
            run_id="run-2",
            prompt="q2",
            answer="a2",
            created_at="2026-01-01T00:05:00+00:00",
        )
    )
    run_store.save(
        _turn_run(
            run_id="run-failed",
            prompt="qf",
            answer="",
            status=RunStatus.FAILED,
            created_at="2026-01-01T00:06:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=JsonlLedgerStore(str(tmp_path)),
        session_id="sess-1",
    )

    assert [m["content"] for m in messages] == ["q1", "a1", "q2", "a2"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def test_session_chat_messages_caps_to_newest_window_without_splitting_turns() -> None:
    run_store = InMemoryRunStore()
    for i in range(10):
        run_store.save(
            _turn_run(
                run_id=f"run-{i}",
                prompt=f"q{i}",
                answer=f"a{i}",
                created_at=f"2026-01-01T00:{i:02d}:00+00:00",
            )
        )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        session_id="sess-1",
        max_messages=5,
    )

    # 5 would split a turn — the leading assistant half is dropped.
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert [m["content"] for m in messages] == ["q8", "a8", "q9", "a9"]


def test_session_chat_messages_truncates_long_content_with_label() -> None:
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(run_id="run-big", prompt="short q", answer="x" * 500)
    )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        session_id="sess-1",
        max_chars_per_message=100,
    )

    assistant = messages[1]
    assert assistant["content"].startswith("x" * 100)
    assert "#TRUNCATION" in assistant["content"]
    assert "run-big" in assistant["content"]


def test_session_chat_messages_total_char_budget_drops_oldest_whole_turns() -> None:
    """Audit #2: the cumulative budget is the ONLY input guard on the agent
    lane (ReAct disables downstream trimming) — oldest turns fall first,
    whole, and the newest turn survives even when it alone exceeds budget."""
    run_store = InMemoryRunStore()
    for i in range(3):
        run_store.save(
            _turn_run(
                run_id=f"run-{i}",
                prompt=f"q{i}",
                answer="a" * 300,
                created_at=f"2026-01-01T00:{i:02d}:00+00:00",
            )
        )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        session_id="sess-1",
        max_total_chars=700,
    )
    # Each turn ~303 chars; budget 700 keeps the newest two turns only.
    assert [m["metadata"]["run_id"] for m in messages] == [
        "run-1",
        "run-1",
        "run-2",
        "run-2",
    ]

    # A single over-budget newest turn still replays (never an empty seed).
    tight = session_chat_messages(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        session_id="sess-1",
        max_total_chars=100,
    )
    assert [m["metadata"]["run_id"] for m in tight] == ["run-2", "run-2"]


def test_session_chat_messages_skips_scheduled_turns() -> None:
    """Audit #4: scheduled wrapper runs are not conversation."""
    run_store = InMemoryRunStore()
    scheduled = _turn_run(
        run_id="run-sched",
        prompt="scheduled prompt",
        answer="scheduled answer",
        workflow_id="scheduled:abc123",
    )
    # No context.messages: without a chat-classified sibling the scheduled
    # root would otherwise survive the reconstruction's chat preference.
    scheduled.vars = {"prompt": "scheduled prompt"}
    run_store.save(scheduled)

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )
    assert messages == []


def test_session_chat_messages_orders_same_millisecond_turns_correctly() -> None:
    """Audit #7: sub-millisecond creation times must not reverse turn order
    (the stable sort used to keep the index's newest-first order on ties)."""
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(
            run_id="run-first",
            prompt="q-first",
            answer="a-first",
            created_at="2026-01-01T00:00:00.001000+00:00",
        )
    )
    run_store.save(
        _turn_run(
            run_id="run-second",
            prompt="q-second",
            answer="a-second",
            created_at="2026-01-01T00:00:00.001500+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )
    assert [m["content"] for m in messages] == [
        "q-first",
        "a-first",
        "q-second",
        "a-second",
    ]


def test_session_chat_messages_includes_tenant_catalog_workflow_turns() -> None:
    """Live-proof regression (2026-07-16): gateway catalog workflow ids are
    namespaced `__catalog__v2__...` — a bare startswith('__') internal check
    classified EVERY catalog turn internal, hiding all thin-client turns
    from session views and the durable replay. Internal = dunder BOTH ends
    (`__session_memory__`), never a mere prefix."""
    run_store = InMemoryRunStore()
    run_store.save(
        _turn_run(
            run_id="run-catalog",
            prompt="remember the codename",
            answer="noted.",
            workflow_id=(
                "__catalog__v2__tenant_catalog__ZGVmYXVsdA__"
                "YWJzdHJhY3Rhc3Npc3RhbnQtb3JjaGVzdHJhdG9y@0.0.1:d5f9fdd0"
            ),
        )
    )
    run_store.save(
        _turn_run(
            run_id="session_memory_sess-1",
            prompt="x",
            answer="y",
            workflow_id="__session_memory__",
            created_at="2026-01-01T00:01:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id="sess-1"
    )

    assert [m["metadata"]["run_id"] for m in messages] == ["run-catalog", "run-catalog"]
    assert messages[0]["content"] == "remember the codename"


def test_session_chat_messages_excludes_named_runs_and_empty_session() -> None:
    run_store = InMemoryRunStore()
    run_store.save(_turn_run(run_id="run-a", prompt="qa", answer="aa"))
    run_store.save(
        _turn_run(
            run_id="run-b",
            prompt="qb",
            answer="ab",
            created_at="2026-01-01T00:01:00+00:00",
        )
    )

    messages = session_chat_messages(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        session_id="sess-1",
        exclude_run_ids=["run-b"],
    )
    assert [m["metadata"]["run_id"] for m in messages] == ["run-a", "run-a"]

    assert (
        session_chat_messages(
            run_store=run_store, ledger_store=InMemoryLedgerStore(), session_id=""
        )
        == []
    )
