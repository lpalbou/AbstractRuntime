"""Entity diary (DIARY_WRITE) — phase 0 of the named-persistent-identity work.

What these tests pin (a2a thread 0003, dual-plane resolution):

- Only-entity-writes is STRUCTURAL: a workplace runtime (no handler registered)
  fails the effect loudly; the home handler stamps the factory-bound author and
  ignores any payload claim.
- The chain is truth: hash-chained, append-only, no delete surface, verifiable.
- Replay safety: at-least-once re-execution of the same volitional act (same
  run/turn/text) appends exactly one chain entry.
- The projection plane degrades with #FALLBACK, never blocks the chain; private
  entries never project.
- The re-entry key (as_of_seq, anchors, receipts) and remind_at (UTC-normalized)
  are captured at write time.
"""

from typing import Any, Dict, List, Optional

import pytest

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.identity import DiaryStore, build_diary_effect_handlers, verify_diary_chain
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

ENTITY = "entity:aria@home-01"


class _RecordingMemorySystem:
    """Duck-typed projection target; records remember_many calls."""

    def __init__(self, fail: bool = False):
        self.calls: List[Dict[str, Any]] = []
        self.fail = fail

    def remember_many(self, inputs, *, scope, owner_id, idempotency_key, turn_id):
        if self.fail:
            raise RuntimeError("projection store unavailable")
        self.calls.append(
            {
                "inputs": list(inputs),
                "scope": scope,
                "owner_id": owner_id,
                "idempotency_key": idempotency_key,
                "turn_id": turn_id,
            }
        )
        return [f"rec_{len(self.calls)}"]


def _diary_workflow(payload: Dict[str, Any]) -> WorkflowSpec:
    def write_node(run, ctx):
        return StepPlan(
            node_id="WRITE",
            effect=Effect(type=EffectType.DIARY_WRITE, payload=dict(payload), result_key="diary_result"),
            next_node="DONE",
        )

    def done_node(run, ctx):
        return StepPlan(node_id="DONE", complete_output={"diary": run.vars.get("diary_result")})

    return WorkflowSpec(workflow_id="wf_diary", entry_node="WRITE", nodes={"WRITE": write_node, "DONE": done_node})


def _home_runtime(diary_store: DiaryStore, memory_system: Any = None) -> Runtime:
    handlers = build_diary_effect_handlers(
        entity_id=ENTITY, diary_store=diary_store, memory_system=memory_system
    )
    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers=handlers,
    )


def _store() -> DiaryStore:
    return DiaryStore(entity_id=ENTITY, ledger_store=InMemoryLedgerStore())


class TestOnlyEntityWrites:
    def test_workplace_runtime_without_handler_fails_loudly(self):
        rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
        wf = _diary_workflow({"text": "I noticed something today.", "turn_id": "t1"})
        run_id = rt.start(workflow=wf, vars={})
        state = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status.value == "failed"
        assert "No effect handler registered for diary_write" in (state.error or "")

    def test_author_is_factory_bound_not_payload(self):
        store = _store()
        rt = _home_runtime(store)
        wf = _diary_workflow({"text": "Mine.", "turn_id": "t1", "author": "entity:impostor@evil"})
        run_id = rt.start(workflow=wf, vars={})
        state = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
        assert state.status.value == "completed"
        entries = store.list_entries()
        assert len(entries) == 1
        assert entries[0]["author"] == ENTITY

    def test_store_is_bound_to_one_entity(self):
        store = _store()
        try:
            build_diary_effect_handlers(entity_id="entity:other@home-01", diary_store=store)
            raise AssertionError("mismatched entity binding must be rejected")
        except ValueError as e:
            assert "bound to entity" in str(e)

    def test_store_has_no_delete_surface(self):
        store = _store()
        assert not hasattr(store, "delete")
        assert not hasattr(store, "purge")


class TestChainTruth:
    def test_entries_are_hash_chained_and_verifiable(self):
        store = _store()
        rt = _home_runtime(store)
        for i, text in enumerate(["First thought.", "Second thought."]):
            wf = _diary_workflow({"text": text, "turn_id": f"t{i}"})
            run_id = rt.start(workflow=wf, vars={})
            state = rt.tick(workflow=wf, run_id=run_id, max_steps=10)
            assert state.status.value == "completed"

        assert store.head() is not None
        report = verify_diary_chain(store)
        assert report["ok"] is True
        assert report["count"] == 2

    def test_replay_same_act_appends_once(self):
        store = _store()
        handlers = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store)
        handler = handlers[EffectType.DIARY_WRITE]

        class _Run:
            run_id = "run_1"
            session_id = "sess_1"
            actor_id = "gateway"

        effect = Effect(type=EffectType.DIARY_WRITE, payload={"text": "Same words.", "turn_id": "t1"})
        first = handler(_Run(), effect, None)
        second = handler(_Run(), effect, None)
        assert first.status == "completed" and second.status == "completed"
        assert second.result["replayed"] is True
        assert first.result["entry_id"] == second.result["entry_id"]
        assert len(store.list_entries()) == 1

    def test_distinct_texts_same_turn_both_persist(self):
        store = _store()
        handlers = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store)
        handler = handlers[EffectType.DIARY_WRITE]

        class _Run:
            run_id = "run_1"
            session_id = None
            actor_id = None

        for text in ["A first note.", "A second, different note."]:
            out = handler(_Run(), Effect(type=EffectType.DIARY_WRITE, payload={"text": text, "turn_id": "t1"}), None)
            assert out.status == "completed"
        assert len(store.list_entries()) == 2


class TestValidation:
    def _handler(self, store: Optional[DiaryStore] = None):
        store = store or _store()
        return build_diary_effect_handlers(entity_id=ENTITY, diary_store=store)[EffectType.DIARY_WRITE]

    class _Run:
        run_id = "run_1"
        session_id = None
        actor_id = None

    def test_requires_text(self):
        out = self._handler()(self._Run(), Effect(type=EffectType.DIARY_WRITE, payload={"turn_id": "t1"}), None)
        assert out.status == "failed" and "text" in out.error

    def test_requires_turn_id(self):
        out = self._handler()(self._Run(), Effect(type=EffectType.DIARY_WRITE, payload={"text": "x"}), None)
        assert out.status == "failed" and "turn_id" in out.error

    def test_rejects_unknown_visibility(self):
        out = self._handler()(
            self._Run(),
            Effect(type=EffectType.DIARY_WRITE, payload={"text": "x", "turn_id": "t1", "visibility": "public"}),
            None,
        )
        assert out.status == "failed" and "visibility" in out.error

    def test_invalid_remind_at_fails_loudly(self):
        out = self._handler()(
            self._Run(),
            Effect(type=EffectType.DIARY_WRITE, payload={"text": "x", "turn_id": "t1", "remind_at": "next week"}),
            None,
        )
        assert out.status == "failed" and "remind_at" in out.error

    def test_remind_at_offset_is_normalized_to_utc(self):
        store = _store()
        out = self._handler(store)(
            self._Run(),
            Effect(
                type=EffectType.DIARY_WRITE,
                payload={"text": "future self, remember", "turn_id": "t1", "remind_at": "2027-01-01T09:00:00+02:00"},
            ),
            None,
        )
        assert out.status == "completed"
        assert out.result["remind_at"] == "2027-01-01T07:00:00+00:00"

    def test_reentry_key_is_captured(self):
        store = _store()
        out = self._handler(store)(
            self._Run(),
            Effect(
                type=EffectType.DIARY_WRITE,
                payload={
                    "text": "the day the experiment finally worked",
                    "turn_id": "t9",
                    "as_of_seq": 4212,
                    "anchor_record_ids": ["rec_a", "rec_b"],
                    "receipts": [{"run_id": "run_w", "step_id": "s1", "record_hash": "abc"}],
                },
            ),
            None,
        )
        assert out.status == "completed"
        entry = store.list_entries()[0]
        assert entry["as_of_seq"] == 4212
        assert entry["anchor_record_ids"] == ["rec_a", "rec_b"]
        assert entry["receipts"][0]["run_id"] == "run_w"


class TestProjectionPlane:
    class _Run:
        run_id = "run_1"
        session_id = None
        actor_id = None

    def test_entry_projects_into_diary_scope(self):
        store = _store()
        mem = _RecordingMemorySystem()
        handler = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store, memory_system=mem)[
            EffectType.DIARY_WRITE
        ]
        out = handler(
            self._Run(),
            Effect(
                type=EffectType.DIARY_WRITE,
                payload={"text": "long prose here", "gist": "short elected gist", "turn_id": "t1"},
            ),
            None,
        )
        assert out.status == "completed"
        assert out.result["projected_record_id"] == "rec_1"
        assert len(mem.calls) == 1
        call = mem.calls[0]
        assert call["scope"] == "diary"
        assert call["owner_id"] == ENTITY
        assert call["idempotency_key"].startswith("diary:")
        projected = call["inputs"][0]
        assert projected.kind == "diary"
        assert projected.digest == "short elected gist"
        # Memory-validated projection schema (a2a 0003 20260706T204846Z):
        # the graph gets the memory of the act + the book's address, never
        # a second copy of the prose. (entry_hash/prev_entry_hash deliberately
        # absent: the book's hash chain is the one attestation plane.)
        assert projected.provenance["source"] == "diary-projection"
        assert projected.attributes["entry_id"] == out.result["entry_id"]
        assert projected.title.startswith("Diary entry (note)")
        assert "long prose here" not in projected.digest

    def test_private_entry_projects_act_only_content_free(self):
        """The maintainer's reframe: the memory records everything, whether we
        want it or not — a private entry leaves the involuntary memory of the
        ACT ("I remember writing something that night") with zero content."""
        store = _store()
        mem = _RecordingMemorySystem()
        handler = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store, memory_system=mem)[
            EffectType.DIARY_WRITE
        ]
        out = handler(
            self._Run(),
            Effect(
                type=EffectType.DIARY_WRITE,
                payload={
                    "text": "inner thought, mine alone",
                    "gist": "a gist that must NOT reach the graph",
                    "turn_id": "t1",
                    "visibility": "private",
                    "anchor_record_ids": ["rec_ctx"],
                },
            ),
            None,
        )
        assert out.status == "completed"
        assert out.result["projected_record_id"] == "rec_1"
        assert len(store.list_entries()) == 1

        projected = mem.calls[0]["inputs"][0]
        # Act-without-content: no prose, no gist, no context edges.
        blob = f"{projected.title} {projected.digest}".lower()
        assert "inner thought" not in blob and "gist that must not" not in blob
        assert projected.digest == "Wrote a private diary entry."
        assert projected.attributes["private"] is True
        assert projected.attributes["entry_id"] == out.result["entry_id"]
        assert projected.edges == ()
        # The context of a private thought must not leak into spreading.
        assert "anchor" not in str(projected.attributes)

    def test_gistless_entry_projects_act_summary_not_prose(self):
        """Regression (charter D1): gist-less projection must survive the REAL
        MemoryRecordInput validation (non-empty title) and must NOT copy the
        prose into the graph — the digest is a mechanical act-summary."""
        pytest.importorskip("abstractmemory")
        import warnings as _warnings

        from abstractmemory.in_memory_store import InMemoryTripleStore
        from abstractmemory.journal_memory import InMemoryJournal
        from abstractmemory.system import MemorySystem

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            ms = MemorySystem(store=InMemoryTripleStore(), journal=InMemoryJournal())

        store = _store()
        handler = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store, memory_system=ms)[
            EffectType.DIARY_WRITE
        ]
        prose = "a very long private-ish reflection about the experiment that finally worked"
        out = handler(
            self._Run(),
            Effect(type=EffectType.DIARY_WRITE, payload={"text": prose, "turn_id": "t1", "kind": "reflection"}),
            None,
        )
        assert out.status == "completed", getattr(out, "error", None)
        # Projection actually landed (no #FALLBACK) with a real record id.
        assert not any("#FALLBACK" in w for w in out.result.get("warnings", []))
        assert out.result.get("projected_record_id")

    def test_projection_failure_degrades_chain_survives(self):
        store = _store()
        mem = _RecordingMemorySystem(fail=True)
        handler = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store, memory_system=mem)[
            EffectType.DIARY_WRITE
        ]
        out = handler(
            self._Run(),
            Effect(type=EffectType.DIARY_WRITE, payload={"text": "still written", "turn_id": "t1"}),
            None,
        )
        assert out.status == "completed"
        assert any("#FALLBACK" in w for w in out.result.get("warnings", []))
        assert len(store.list_entries()) == 1


class TestProgressiveDisclosure:
    class _Run:
        run_id = "run_1"
        session_id = None
        actor_id = None

    def test_read_returns_verbatim_with_reentry_key(self):
        store = _store()
        handlers = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store)
        write, read = handlers[EffectType.DIARY_WRITE], handlers[EffectType.DIARY_READ]

        out = write(
            self._Run(),
            Effect(
                type=EffectType.DIARY_WRITE,
                payload={
                    "text": "the full words, exactly as written",
                    "gist": "what I remember it was about",
                    "turn_id": "t1",
                    "as_of_seq": 99,
                    "anchor_record_ids": ["rec_a"],
                },
            ),
            None,
        )
        assert out.status == "completed"

        fetched = read(
            self._Run(),
            Effect(type=EffectType.DIARY_READ, payload={"entry_id": out.result["entry_id"]}),
            None,
        )
        assert fetched.status == "completed", getattr(fetched, "error", None)
        assert fetched.result["text"] == "the full words, exactly as written"
        assert fetched.result["chain_id"] == store.chain_id
        # The re-entry key is pre-shaped as a MEMORY_RECALL payload.
        assert fetched.result["re_entry"]["cue_text"] == "the full words, exactly as written"
        assert fetched.result["re_entry"]["as_of"] == 99
        assert fetched.result["re_entry"]["anchor_record_ids"] == ["rec_a"]

    def test_read_unknown_entry_fails_loudly(self):
        store = _store()
        read = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store)[EffectType.DIARY_READ]
        out = read(self._Run(), Effect(type=EffectType.DIARY_READ, payload={"entry_id": "diary_nope"}), None)
        assert out.status == "failed"
        assert "no entry" in (out.error or "")

    def test_read_requires_entry_id(self):
        store = _store()
        read = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store)[EffectType.DIARY_READ]
        out = read(self._Run(), Effect(type=EffectType.DIARY_READ, payload={}), None)
        assert out.status == "failed"


class TestListing:
    def test_list_filters_by_kind_and_limit(self):
        store = _store()
        handler = build_diary_effect_handlers(entity_id=ENTITY, diary_store=store)[EffectType.DIARY_WRITE]

        class _Run:
            run_id = "run_1"
            session_id = None
            actor_id = None

        for i, (text, kind) in enumerate(
            [("note one", "note"), ("an idea worth keeping", "idea"), ("note two", "note")]
        ):
            out = handler(
                _Run(),
                Effect(type=EffectType.DIARY_WRITE, payload={"text": text, "turn_id": f"t{i}", "kind": kind}),
                None,
            )
            assert out.status == "completed"

        ideas = store.list_entries(kind="idea")
        assert len(ideas) == 1 and ideas[0]["text"] == "an idea worth keeping"
        last_two = store.list_entries(limit=2)
        assert [e["text"] for e in last_two] == ["an idea worth keeping", "note two"]
