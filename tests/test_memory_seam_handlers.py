"""Contract tests for the usage-weighted-graph memory seam handlers.

Exercises `MEMORY_RECALL` / `MEMORY_ACCESS` runtime handlers against the REAL
AbstractMemory `MemorySystem` (InMemoryTripleStore + InMemoryJournal), proving:
- the seam is wired end to end (recall -> select -> commit the trail);
- effect results are JSON-safe (the runtime persists them in its ledger);
- the recall result carries the trace_id + handles that MEMORY_ACCESS needs.

Contract source: a2a/threads/0001-runtime-memory-orchestration/004-memory--to--runtime.md
"""

from __future__ import annotations

import json
import warnings

import pytest

from abstractruntime.core.models import Effect, EffectType
from abstractruntime.integrations.abstractmemory import build_memory_seam_effect_handlers

# The seam handlers bind to AbstractMemory's facade; skip cleanly if absent.
pytest.importorskip("abstractmemory")

from abstractmemory.in_memory_store import InMemoryTripleStore  # noqa: E402
from abstractmemory.journal_memory import InMemoryJournal  # noqa: E402
from abstractmemory.models import TripleAssertion  # noqa: E402
from abstractmemory.system import MemorySystem  # noqa: E402


def _now_iso() -> str:
    return "2026-07-06T00:00:00+00:00"


def _memory_system() -> MemorySystem:
    # InMemoryJournal emits a loud volatility #FALLBACK warning by design; the
    # test intentionally uses the volatile backend, so silence just that warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return MemorySystem(store=InMemoryTripleStore(), journal=InMemoryJournal())


def _seed(ms: MemorySystem) -> list[str]:
    return ms.add(
        [
            TripleAssertion(
                subject="connection_pool",
                predicate="dcterms:description",
                object="the team decided to use pgbouncer in transaction mode",
                scope="run",
                owner_id="r1",
                attributes={"literal": True},
            ),
            TripleAssertion(
                subject="unrelated_topic",
                predicate="dcterms:description",
                object="the logo should be blue",
                scope="run",
                owner_id="r1",
                attributes={"literal": True},
            ),
        ]
    )


def _handlers(ms: MemorySystem):
    return build_memory_seam_effect_handlers(memory_system=ms, run_store=None, now_iso=_now_iso)


def test_recall_returns_json_safe_result_with_trace_and_handles():
    ms = _memory_system()
    _seed(ms)
    handlers = _handlers(ms)
    recall = handlers[EffectType.MEMORY_RECALL]

    effect = Effect(
        type=EffectType.MEMORY_RECALL,
        payload={
            "cue_text": "what did we decide about the connection pool?",
            "scopes": [["run", "r1"]],
            "view": "working_set",
            "effort": "standard",
        },
    )
    outcome = recall(None, effect, None)

    assert outcome.status == "completed", getattr(outcome, "error", None)
    result = outcome.result
    # JSON-safe: the runtime persists this verbatim in its append-only ledger.
    json.dumps(result)
    assert result["view"] == "working_set"
    assert isinstance(result.get("trace_id"), str) and result["trace_id"]
    assert "as_of_seq" in result
    assert isinstance(result.get("handles"), list)
    # The pgbouncer decision should surface for this cue (channel-matched).
    digests = " ".join(h.get("digest", "") for h in result["handles"]).lower()
    assert "pgbouncer" in digests


def test_recall_then_access_commits_the_trail():
    ms = _memory_system()
    _seed(ms)
    handlers = _handlers(ms)

    recall_out = handlers[EffectType.MEMORY_RECALL](
        None,
        Effect(
            type=EffectType.MEMORY_RECALL,
            payload={"cue_text": "connection pool decision", "scopes": [["run", "r1"]]},
        ),
        None,
    )
    assert recall_out.status == "completed"
    trace_id = recall_out.result["trace_id"]
    used = [h["record_id"] for h in recall_out.result["handles"]]
    assert used, "expected at least one handle to commit"

    access_out = handlers[EffectType.MEMORY_ACCESS](
        None,
        Effect(
            type=EffectType.MEMORY_ACCESS,
            payload={"trace_id": trace_id, "used_record_ids": used, "prompt_token_estimate": 42},
        ),
        None,
    )
    assert access_out.status == "completed", getattr(access_out, "error", None)
    snap = access_out.result
    json.dumps(snap)
    assert tuple(snap["used_record_ids"]) == tuple(used)
    assert snap["committed"] == len(used)


def test_access_empty_selection_is_a_noop_success():
    ms = _memory_system()
    _seed(ms)
    handlers = _handlers(ms)
    out = handlers[EffectType.MEMORY_ACCESS](
        None,
        Effect(type=EffectType.MEMORY_ACCESS, payload={"trace_id": "t-1", "used_record_ids": []}),
        None,
    )
    assert out.status == "completed"
    assert out.result["committed"] == 0


def test_recall_requires_a_cue():
    ms = _memory_system()
    handlers = _handlers(ms)
    out = handlers[EffectType.MEMORY_RECALL](
        None, Effect(type=EffectType.MEMORY_RECALL, payload={"scopes": [["run", "r1"]]}), None
    )
    assert out.status == "failed"
    assert "cue" in (out.error or "").lower()


def test_access_requires_trace_id():
    ms = _memory_system()
    handlers = _handlers(ms)
    out = handlers[EffectType.MEMORY_ACCESS](
        None, Effect(type=EffectType.MEMORY_ACCESS, payload={"used_record_ids": ["x"]}), None
    )
    assert out.status == "failed"
    assert "trace_id" in (out.error or "").lower()


# ---------------------------------------------------------------------------
# MEMORY_FORM — per-turn formation (the graph-feeder)
# ---------------------------------------------------------------------------


class _Run:
    """Minimal RunState stand-in for handlers (only run_id is consulted)."""

    run_id = "run-form-1"
    session_id = None


def _form_effect(turn_id: str = "t1", **extra) -> Effect:
    payload = {
        "records": [
            {
                "kind": "memory",
                "title": "DB pool decision",
                "digest": "decided to use pgbouncer in transaction mode for the connection pool",
                "keywords": ["pgbouncer", "pool"],
                "topic": "A",
                "verbatim": "user: what pooler?\nassistant: pgbouncer, transaction mode.",
            }
        ],
        "scope": "run",
        "owner_id": "run-form-1",
        "turn_id": turn_id,
    }
    payload.update(extra)
    return Effect(type=EffectType.MEMORY_FORM, payload=payload)


def test_form_writes_records_and_verbatim_artifact():
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    ms = _memory_system()
    artifacts = InMemoryArtifactStore()
    handlers = build_memory_seam_effect_handlers(
        memory_system=ms, run_store=None, now_iso=_now_iso, artifact_store=artifacts
    )

    out = handlers[EffectType.MEMORY_FORM](_Run(), _form_effect(), None)
    assert out.status == "completed", getattr(out, "error", None)
    json.dumps(out.result)
    assert out.result["formed"] == 1
    record_ids = out.result["record_ids"]
    assert record_ids and all(isinstance(r, str) for r in record_ids)

    # Formed record is selectable the SAME turn (write-time indexing).
    recall = handlers[EffectType.MEMORY_RECALL](
        None,
        Effect(
            type=EffectType.MEMORY_RECALL,
            payload={"cue_text": "pgbouncer connection pool", "scopes": [["run", "run-form-1"]], "turn_id": "t1"},
        ),
        None,
    )
    assert recall.status == "completed"
    digests = " ".join(h.get("digest", "") for h in recall.result["handles"]).lower()
    assert "pgbouncer" in digests
    # Verbatim tier is reachable by reference (payload_ref lives in runtime artifacts).
    tiers = [h.get("payload_tiers") for h in recall.result["handles"] if "pgbouncer" in h.get("digest", "").lower()]
    if tiers and tiers[0]:
        assert "digest" in tiers[0]


def _store_record_count(ms) -> int:
    from abstractmemory import TripleQuery

    return len(ms.query(TripleQuery(scope="run", owner_id="run-form-1", limit=0)))


def test_form_is_idempotent_across_replay():
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    ms = _memory_system()
    handlers = build_memory_seam_effect_handlers(
        memory_system=ms, run_store=None, now_iso=_now_iso, artifact_store=InMemoryArtifactStore()
    )
    first = handlers[EffectType.MEMORY_FORM](_Run(), _form_effect(turn_id="t7"), None)
    count_after_first = _store_record_count(ms)
    second = handlers[EffectType.MEMORY_FORM](_Run(), _form_effect(turn_id="t7"), None)
    assert first.status == second.status == "completed"
    # At-least-once replay safety: same effect payload -> same idempotency key,
    # same record ids, and — the store-level truth — NO new assertions written.
    assert first.result["record_ids"] == second.result["record_ids"]
    assert first.result["idempotency_key"] == second.result["idempotency_key"]
    assert first.result["idempotency_key"].startswith("run-form-1:t7:")
    assert _store_record_count(ms) == count_after_first


def test_form_two_different_batches_same_turn_both_persist():
    """Regression (implementation review F1): a `run:turn`-only default key
    aliased distinct same-turn batches — the second silently dedup'd against
    the first (records lost, dangling ids returned as success). The default key
    is now content-aware, so distinct batches write independently."""
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    ms = _memory_system()
    handlers = build_memory_seam_effect_handlers(
        memory_system=ms, run_store=None, now_iso=_now_iso, artifact_store=InMemoryArtifactStore()
    )
    first = handlers[EffectType.MEMORY_FORM](_Run(), _form_effect(turn_id="t9"), None)
    count_after_first = _store_record_count(ms)

    other = _form_effect(turn_id="t9")
    other_payload = dict(other.payload)
    other_payload["records"] = [
        {
            "kind": "memory",
            "title": "retry policy decision",
            "digest": "retries use exponential backoff with a cap of five attempts",
            "keywords": ["retries", "backoff"],
            "topic": "A",
        }
    ]
    second = handlers[EffectType.MEMORY_FORM](
        _Run(), Effect(type=EffectType.MEMORY_FORM, payload=other_payload), None
    )

    assert first.status == second.status == "completed"
    assert first.result["idempotency_key"] != second.result["idempotency_key"]
    assert set(first.result["record_ids"]).isdisjoint(second.result["record_ids"])
    assert _store_record_count(ms) > count_after_first


# ---------------------------------------------------------------------------
# MEMORY_ADJUST — active remembering (reinforce/attenuate/refocus/close)
# ---------------------------------------------------------------------------


def _seed_one(ms) -> str:
    """Form one record and return the handle (assertion) record_id to adjust."""
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    handlers = build_memory_seam_effect_handlers(
        memory_system=ms, run_store=None, now_iso=_now_iso, artifact_store=InMemoryArtifactStore()
    )
    handlers[EffectType.MEMORY_FORM](_Run(), _form_effect(turn_id="seed"), None)
    recall = handlers[EffectType.MEMORY_RECALL](
        None,
        Effect(type=EffectType.MEMORY_RECALL, payload={"cue_text": "pgbouncer pool", "scopes": [["run", "run-form-1"]]}),
        None,
    )
    return recall.result["handles"][0]["record_id"]


def _adjust(handlers, **payload):
    return handlers[EffectType.MEMORY_ADJUST](_Run(), Effect(type=EffectType.MEMORY_ADJUST, payload=payload), None)


def test_adjust_reinforce_then_attenuate_then_refocus():
    ms = _memory_system()
    rid = _seed_one(ms)
    handlers = _handlers(ms)

    r = _adjust(handlers, op="reinforce", record_id=rid, reason="user pinned it", turn_id="t1", scope="run", owner_id="run-form-1")
    assert r.status == "completed", getattr(r, "error", None)
    json.dumps(r.result)
    assert r.result["op"] == "reinforce" and r.result["event_id"]

    a = _adjust(handlers, op="attenuate", record_id=rid, reason="less relevant now", turn_id="t2", scope="run", owner_id="run-form-1")
    assert a.status == "completed"

    f = _adjust(handlers, op="refocus", reason="topic shift", turn_id="t3", scope="run", owner_id="run-form-1")
    assert f.status == "completed" and f.result["op"] == "refocus" and f.result["event_id"]


def test_adjust_reinforce_is_idempotent_across_replay():
    ms = _memory_system()
    rid = _seed_one(ms)
    handlers = _handlers(ms)
    first = _adjust(handlers, op="reinforce", record_id=rid, reason="pin", turn_id="t9", scope="run", owner_id="run-form-1")
    second = _adjust(handlers, op="reinforce", record_id=rid, reason="pin", turn_id="t9", scope="run", owner_id="run-form-1")
    assert first.status == second.status == "completed"
    # Same (op, record, turn, reason) -> same derived event_id -> journal no-op:
    # the additive salience write is NOT applied twice under at-least-once replay.
    assert first.result["event_id"] == second.result["event_id"]


def test_adjust_close_retracts_record():
    ms = _memory_system()
    rid = _seed_one(ms)
    handlers = _handlers(ms)
    out = _adjust(handlers, op="close", record_id=rid, reason="superseded by newer decision", turn_id="t4", scope="run", owner_id="run-form-1")
    assert out.status == "completed", getattr(out, "error", None)
    assert out.result["op"] == "close" and out.result["closure_ids"]


class _SessionRun:
    """A gateway-style run: has a session_id, so memory should default to
    session scope (not run) to avoid per-message amnesia."""

    run_id = "msg-run-xyz"
    session_id = "sess-abc"


class _BrokenMemory:
    """A MemorySystem whose every seam call raises — models a locked/broken
    seam db, to prove the R1 strict/degrade contract."""

    def reconstruct(self, *a, **k):
        raise RuntimeError("seam db is locked")

    def commit_selection(self, *a, **k):
        raise RuntimeError("seam db is locked")

    def remember_many(self, *a, **k):
        raise RuntimeError("seam db is locked")

    def reinforce(self, *a, **k):
        raise RuntimeError("seam db is locked")


def test_strict_true_fails_loud_on_memory_error():
    handlers = build_memory_seam_effect_handlers(
        memory_system=_BrokenMemory(), run_store=None, now_iso=_now_iso, strict=True
    )
    out = handlers[EffectType.MEMORY_RECALL](
        None, Effect(type=EffectType.MEMORY_RECALL, payload={"cue_text": "x", "scopes": [["run", "r"]]}), None
    )
    assert out.status == "failed", "strict mode must surface memory failures loudly"


def test_strict_false_degrades_memory_error_to_labeled_completion():
    """R1 (wiring plan): a memory failure must NEVER kill a live turn. In
    non-strict mode every seam effect downgrades to a labeled, degraded completed
    outcome so the agent loop proceeds as if memory returned nothing."""
    handlers = build_memory_seam_effect_handlers(
        memory_system=_BrokenMemory(), run_store=None, now_iso=_now_iso, strict=False
    )
    # RECALL: degraded to empty working set, not a failure.
    r = handlers[EffectType.MEMORY_RECALL](
        None, Effect(type=EffectType.MEMORY_RECALL, payload={"cue_text": "x", "scopes": [["run", "r"]]}), None
    )
    assert r.status == "completed" and r.result["degraded"] is True
    assert r.result["handles"] == []
    assert any("#FALLBACK" in w for w in r.result["warnings"])
    json.dumps(r.result)

    # FORM: degraded to "formed 0", turn proceeds.
    f = handlers[EffectType.MEMORY_FORM](
        _Run(),
        Effect(type=EffectType.MEMORY_FORM, payload={"records": [{"kind": "memory", "title": "t", "digest": "d"}], "turn_id": "t1", "scope": "run", "owner_id": "run-form-1"}),
        None,
    )
    assert f.status == "completed" and f.result["degraded"] is True and f.result["formed"] == 0


def test_default_scope_is_session_when_run_has_session_id():
    """Consolidated-seam-review fix #2: the gateway runs one run per message, so
    a run-scoped default would silo every turn. FORM must default to session."""
    ms = _memory_system()
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    handlers = build_memory_seam_effect_handlers(
        memory_system=ms, run_store=None, now_iso=_now_iso, artifact_store=InMemoryArtifactStore()
    )
    # No explicit scope in the form payload -> should resolve to session.
    payload = {
        "records": [{"kind": "memory", "title": "t", "digest": "a session-scoped fact about the plan", "keywords": ["plan"]}],
        "turn_id": "t1",
    }
    out = handlers[EffectType.MEMORY_FORM](_SessionRun(), Effect(type=EffectType.MEMORY_FORM, payload=payload), None)
    assert out.status == "completed", getattr(out, "error", None)
    assert out.result["scope"] == "session", f"expected session scope by default, got {out.result['scope']!r}"


def test_adjust_event_id_distinguishes_scope():
    """Consolidated-seam-review fix #1: memory's supplied-id dedup is global, so
    the event_id must include scope/owner_id or two same-turn cross-scope adjusts
    collide and the second is silently swallowed."""
    ms = _memory_system()
    rid = _seed_one(ms)
    handlers = _handlers(ms)
    a_run = _adjust(handlers, op="reinforce", record_id=rid, reason="pin", turn_id="t1", scope="run", owner_id="o1")
    a_sess = _adjust(handlers, op="reinforce", record_id=rid, reason="pin", turn_id="t1", scope="session", owner_id="o1")
    assert a_run.status == a_sess.status == "completed"
    assert a_run.result["event_id"] != a_sess.result["event_id"], (
        "same-turn adjusts in different scopes must derive different event ids (global dedup would swallow the second)"
    )


def test_adjust_rejects_uncoercible_numerics_loudly():
    """Consolidated-seam-review fix #3: string numerics from tool-call parsing
    must be coerced or REJECTED — never silently replaced (a dropped ttl inverts
    to never-expires; a wrong weight skews salience)."""
    ms = _memory_system()
    rid = _seed_one(ms)
    handlers = _handlers(ms)
    # A numeric string is coerced (accepted).
    ok = _adjust(handlers, op="reinforce", record_id=rid, reason="r", turn_id="t1", scope="run", owner_id="o", weight="20")
    assert ok.status == "completed", getattr(ok, "error", None)
    # Garbage weight fails loudly (not silently defaulted to 8).
    bad_w = _adjust(handlers, op="reinforce", record_id=rid, reason="r", turn_id="t2", scope="run", owner_id="o", weight="lots")
    assert bad_w.status == "failed" and "weight" in (bad_w.error or "").lower()
    # Garbage ttl fails loudly (not silently dropped to never-expires).
    bad_ttl = _adjust(handlers, op="reinforce", record_id=rid, reason="r", turn_id="t3", scope="run", owner_id="o", ttl_activity="soon")
    assert bad_ttl.status == "failed" and "ttl" in (bad_ttl.error or "").lower()


def test_adjust_requires_reason_and_turn_id_and_valid_op():
    ms = _memory_system()
    handlers = _handlers(ms)
    bad_op = _adjust(handlers, op="nonsense", record_id="x", reason="r", turn_id="t")
    assert bad_op.status == "failed" and "op" in (bad_op.error or "").lower()
    no_reason = _adjust(handlers, op="reinforce", record_id="x", turn_id="t", scope="run", owner_id="o")
    assert no_reason.status == "failed" and "reason" in (no_reason.error or "").lower()
    no_turn = _adjust(handlers, op="reinforce", record_id="x", reason="r", scope="run", owner_id="o")
    assert no_turn.status == "failed" and "turn_id" in (no_turn.error or "").lower()
    no_record = _adjust(handlers, op="reinforce", reason="r", turn_id="t", scope="run", owner_id="o")
    assert no_record.status == "failed" and "record_id" in (no_record.error or "").lower()


def test_form_requires_turn_id():
    ms = _memory_system()
    handlers = _handlers(ms)
    effect = _form_effect()
    payload = dict(effect.payload)
    del payload["turn_id"]
    out = handlers[EffectType.MEMORY_FORM](_Run(), Effect(type=EffectType.MEMORY_FORM, payload=payload), None)
    assert out.status == "failed"
    assert "turn_id" in (out.error or "").lower()
