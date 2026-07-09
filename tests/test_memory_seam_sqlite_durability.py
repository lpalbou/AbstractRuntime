"""Durability de-risk for the live wiring: the memory seam over the SQLite
store+journal pairing must survive a PROCESS RESTART.

The in-memory tests prove the contract shapes; they cannot prove that a formed
record + its deposited trail come back after the store/journal are closed and
reopened from the same db file. That round-trip is the honest v1 substrate the
memory agent endorsed (a2a 0001/014 §D: SQLiteTripleStore + SQLiteJournal, one
file; embedder-free HERE for determinism — production pairs SQLite with an
embedder when reachable), and it is the prerequisite the wiring charter said to test
FIRST — so this is written before any gateway/agent wiring.

Covers: form -> recall -> commit -> RESTART -> recall (persistence), and
idempotency across the restart boundary (a replayed MEMORY_FORM / MEMORY_ADJUST
after reopen must not duplicate records or double-apply salience).
"""

from __future__ import annotations

import json
import warnings

import pytest

from abstractruntime.core.models import Effect, EffectType
from abstractruntime.integrations.abstractmemory import build_memory_seam_effect_handlers
from abstractruntime.storage.artifacts import FileArtifactStore

pytest.importorskip("abstractmemory")

from abstractmemory import MemorySystem, SQLiteJournal, SQLiteTripleStore  # noqa: E402
from abstractmemory import TripleQuery  # noqa: E402

OWNER = "durable-run"


class _Run:
    run_id = OWNER
    session_id = None


def _open_system(db_path):
    """Construct MemorySystem over the SQLite pairing on ONE db file (the
    endorsed v1 substrate). Embedder-free: exact+keyword+STM+spreading only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        store = SQLiteTripleStore(db_path)
        journal = SQLiteJournal(db_path)
        return MemorySystem(store=store, journal=journal), store, journal


def _handlers(ms, artifacts):
    return build_memory_seam_effect_handlers(
        memory_system=ms, run_store=None, now_iso=lambda: "2026-07-06T00:00:00+00:00", artifact_store=artifacts
    )


def _form_payload(turn_id: str):
    return {
        "records": [{
            "kind": "memory", "title": "pool decision",
            "digest": "team chose pgbouncer in transaction mode for the connection pool",
            "keywords": ["pgbouncer", "pool"], "topic": "A",
            "verbatim": f"turn {turn_id}: which pooler? pgbouncer, transaction mode.",
        }],
        "scope": "run", "owner_id": OWNER, "turn_id": turn_id,
    }


def _recall(handlers, cue="pgbouncer connection pool"):
    out = handlers[EffectType.MEMORY_RECALL](
        None,
        Effect(type=EffectType.MEMORY_RECALL, payload={"cue_text": cue, "scopes": [["run", OWNER]], "view": "working_set"}),
        None,
    )
    assert out.status == "completed", getattr(out, "error", None)
    return out.result


def test_form_recall_commit_survive_process_restart(tmp_path):
    db = tmp_path / "memory.sqlite3"
    artifacts = FileArtifactStore(str(tmp_path / "artifacts"))

    # --- session 1: form, recall, commit ---
    ms1, store1, journal1 = _open_system(db)
    h1 = _handlers(ms1, artifacts)
    form = h1[EffectType.MEMORY_FORM](_Run(), Effect(type=EffectType.MEMORY_FORM, payload=_form_payload("t1")), None)
    assert form.status == "completed", getattr(form, "error", None)
    formed_ids = form.result["record_ids"]

    r1 = _recall(h1)
    used = [h["record_id"] for h in r1["handles"] if "pgbouncer" in h.get("digest", "")]
    assert used, "gold fact not recalled in session 1"
    acc = h1[EffectType.MEMORY_ACCESS](
        None, Effect(type=EffectType.MEMORY_ACCESS, payload={"trace_id": r1["trace_id"], "used_record_ids": used}), None
    )
    assert acc.status == "completed"
    # Simulate process exit.
    store1.close()
    journal1.close()

    # --- session 2: reopen from the SAME file; the memory must still be there ---
    ms2, store2, journal2 = _open_system(db)
    h2 = _handlers(ms2, artifacts)
    r2 = _recall(h2)
    digests = " ".join(h.get("digest", "") for h in r2["handles"]).lower()
    assert "pgbouncer" in digests, "formed record did not survive the restart"
    # The committed trail also persisted: the record carries base activation now.
    gold = [h for h in r2["handles"] if "pgbouncer" in h.get("digest", "")][0]
    assert (gold.get("activation") or {}).get("base_level", 0) > 0, "deposited trail did not survive restart"
    store2.close()
    journal2.close()


def test_form_idempotent_across_restart(tmp_path):
    db = tmp_path / "memory.sqlite3"
    artifacts = FileArtifactStore(str(tmp_path / "artifacts"))

    ms1, store1, journal1 = _open_system(db)
    first = _handlers(ms1, artifacts)[EffectType.MEMORY_FORM](
        _Run(), Effect(type=EffectType.MEMORY_FORM, payload=_form_payload("t1")), None
    )
    assert first.status == "completed"
    n_before = len(ms1.query(TripleQuery(scope="run", owner_id=OWNER, limit=0)))
    store1.close(); journal1.close()

    # Reopen and REPLAY the identical MEMORY_FORM (same content -> same content-aware
    # idempotency key): the append-only graph must not gain duplicate records.
    ms2, store2, journal2 = _open_system(db)
    second = _handlers(ms2, artifacts)[EffectType.MEMORY_FORM](
        _Run(), Effect(type=EffectType.MEMORY_FORM, payload=_form_payload("t1")), None
    )
    assert second.status == "completed"
    assert first.result["record_ids"] == second.result["record_ids"], "restart replay changed record ids"
    n_after = len(ms2.query(TripleQuery(scope="run", owner_id=OWNER, limit=0)))
    assert n_after == n_before, f"restart replay duplicated records ({n_before} -> {n_after})"
    store2.close(); journal2.close()


def test_adjust_reinforce_idempotent_across_restart(tmp_path):
    db = tmp_path / "memory.sqlite3"
    artifacts = FileArtifactStore(str(tmp_path / "artifacts"))

    ms1, store1, journal1 = _open_system(db)
    h1 = _handlers(ms1, artifacts)
    h1[EffectType.MEMORY_FORM](_Run(), Effect(type=EffectType.MEMORY_FORM, payload=_form_payload("t1")), None)
    rid = [h["record_id"] for h in _recall(h1)["handles"] if "pgbouncer" in h.get("digest", "")][0]
    adj1 = h1[EffectType.MEMORY_ADJUST](
        _Run(),
        Effect(type=EffectType.MEMORY_ADJUST, payload={"op": "reinforce", "record_id": rid, "reason": "user pinned", "turn_id": "t5", "scope": "run", "owner_id": OWNER}),
        None,
    )
    assert adj1.status == "completed", getattr(adj1, "error", None)
    event_id_1 = adj1.result["event_id"]
    store1.close(); journal1.close()

    # Reopen; replay the same reinforce (same op|record|turn|reason -> same event_id).
    ms2, store2, journal2 = _open_system(db)
    h2 = _handlers(ms2, artifacts)
    # record_id (assertion id) is stable across restart; re-resolve via recall to be safe.
    rid2 = [h["record_id"] for h in _recall(h2)["handles"] if "pgbouncer" in h.get("digest", "")][0]
    assert rid2 == rid, "assertion id changed across restart"
    adj2 = h2[EffectType.MEMORY_ADJUST](
        _Run(),
        Effect(type=EffectType.MEMORY_ADJUST, payload={"op": "reinforce", "record_id": rid, "reason": "user pinned", "turn_id": "t5", "scope": "run", "owner_id": OWNER}),
        None,
    )
    assert adj2.status == "completed"
    assert adj2.result["event_id"] == event_id_1, "reinforce replayed with a different event id across restart (would double-boost)"
    store2.close(); journal2.close()
