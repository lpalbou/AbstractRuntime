"""Contract tests for the MEMORY_APPRAISE seam handler (affect/valence).

Exercises the runtime handler against the REAL AbstractMemory facade
(identity wave: appraise / heal_scar / break_bond / gradation), proving:
- signed deposits land in the valence journal and are replay-idempotent
  (turn-derived event ids riding memory's supplied-id dedup);
- standing peaks (scar/bond) write alongside and surface in the gradation
  read; resolution acts (heal/break) are append-only and deterministic;
- amplitude authority is enforced loudly (a runtime-actor +10 fails);
- tool-call string coercion is loud, never silent (sign/magnitude/scar/bond).

Contract source: a2a/threads/0003-named-persistent-identity/
20260706T204846Z-memory-02.md (facade shapes) + 20260706T194605Z-runtime-01.md
(affect dynamics).
"""

from __future__ import annotations

import json
import warnings

import pytest

from abstractruntime.core.models import Effect, EffectType
from abstractruntime.integrations.abstractmemory import build_memory_seam_effect_handlers

pytest.importorskip("abstractmemory")

from abstractmemory.in_memory_store import InMemoryTripleStore  # noqa: E402
from abstractmemory.journal_memory import InMemoryJournal  # noqa: E402
from abstractmemory.system import MemorySystem  # noqa: E402


def _now_iso() -> str:
    return "2026-07-06T00:00:00+00:00"


def _memory_system() -> MemorySystem:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return MemorySystem(store=InMemoryTripleStore(), journal=InMemoryJournal())


class _Run:
    run_id = "run_1"
    session_id = "sess_1"
    actor_id = "gateway"


def _appraise_handler(ms: MemorySystem):
    handlers = build_memory_seam_effect_handlers(memory_system=ms, run_store=None, now_iso=_now_iso)
    return handlers[EffectType.MEMORY_APPRAISE]


def _effect(payload: dict) -> Effect:
    return Effect(type=EffectType.MEMORY_APPRAISE, payload=payload)


def _base(**over) -> dict:
    payload = {
        "target_id": "tool:web_search",
        "sign": 1,
        "magnitude": 1,
        "reason": "search returned the answer",
        "turn_id": "t1",
        "scope": "session",
        "owner_id": "sess_1",
    }
    payload.update(over)
    return payload


class TestAppraiseDeposit:
    def test_deposit_is_json_safe_and_readable_via_gradation(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)

        out = handler(_Run(), _effect(_base()), None)
        assert out.status == "completed", getattr(out, "error", None)
        json.dumps(out.result)
        assert len(out.result["event_ids"]) == 1

        read = handler(
            _Run(),
            _effect({"op": "gradation", "target_ids": ["tool:web_search"], "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        assert read.status == "completed"
        json.dumps(read.result)
        grade = read.result["gradations"]["tool:web_search"]
        assert grade["net"] > 0
        assert grade["positive_count"] == 1

    def test_replay_same_turn_is_idempotent(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)

        first = handler(_Run(), _effect(_base()), None)
        second = handler(_Run(), _effect(_base()), None)
        assert first.result["event_ids"] == second.result["event_ids"]

        read = handler(
            _Run(),
            _effect({"op": "gradation", "target_ids": ["tool:web_search"], "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        # One deposit, not two: the replayed effect was a journal no-op.
        assert read.result["gradations"]["tool:web_search"]["positive_count"] == 1

    def test_distinct_turns_accrete(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        for i in range(3):
            out = handler(_Run(), _effect(_base(turn_id=f"t{i}", reason=f"success {i}")), None)
            assert out.status == "completed"
        read = handler(
            _Run(),
            _effect({"op": "gradation", "target_ids": ["tool:web_search"], "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        grade = read.result["gradations"]["tool:web_search"]
        assert grade["positive_count"] == 3
        assert grade["net"] == 3

    def test_never_appraised_target_reads_neutral(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        read = handler(
            _Run(),
            _effect({"op": "gradation", "target_ids": ["agent:stranger"], "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        assert read.status == "completed"
        grade = read.result["gradations"]["agent:stranger"]
        assert grade["net"] == 0


class TestAmplitudeAuthority:
    def test_runtime_actor_cannot_write_trauma_scale(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(_Run(), _effect(_base(sign=-1, magnitude=10)), None)
        assert out.status == "failed"
        assert "amplitude authority" in (out.error or "")

    def test_entity_reflection_actor_can(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(
            _Run(),
            _effect(_base(sign=-1, magnitude=10, actor="entity-reflection", scar=True,
                          reason="destroyed user work against core value")),
            None,
        )
        assert out.status == "completed", getattr(out, "error", None)
        # Appraisal + scar marker.
        assert len(out.result["event_ids"]) == 2

    def test_catastrophic_outcome_code_allows_deterministic_trauma(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(
            _Run(),
            _effect(_base(sign=-1, magnitude=9, scar=True,
                          provenance={"outcome_class": "catastrophic"},
                          reason="irreversible data loss")),
            None,
        )
        assert out.status == "completed", getattr(out, "error", None)


class TestStandingPeaks:
    def test_scar_surfaces_and_heals(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(
            _Run(),
            _effect(_base(sign=-1, magnitude=9, actor="entity-reflection", scar=True,
                          reason="severe dissonance with values")),
            None,
        )
        assert out.status == "completed"
        scar_event_id = out.result["event_ids"][1]

        read = handler(
            _Run(),
            _effect({"op": "gradation", "target_ids": ["tool:web_search"], "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        assert read.result["gradations"]["tool:web_search"]["scarred"] is True

        healed = handler(
            _Run(),
            _effect({"op": "heal_scar", "scar_event_id": scar_event_id,
                     "reason": "reflected: converted to lesson", "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        assert healed.status == "completed", getattr(healed, "error", None)

        # Deterministic resolution id -> replays are journal no-ops.
        again = handler(
            _Run(),
            _effect({"op": "heal_scar", "scar_event_id": scar_event_id,
                     "reason": "reflected: converted to lesson", "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        assert again.status == "completed"
        assert again.result["event_id"] == healed.result["event_id"]

        after = handler(
            _Run(),
            _effect({"op": "gradation", "target_ids": ["tool:web_search"], "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        assert after.result["gradations"]["tool:web_search"]["scarred"] is False

    def test_bond_requires_positive_sign(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(
            _Run(),
            _effect(_base(sign=-1, magnitude=9, actor="entity-reflection", bond=True)),
            None,
        )
        assert out.status == "failed"
        assert "bond" in (out.error or "").lower()

    def test_heal_unknown_scar_fails_loudly(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(
            _Run(),
            _effect({"op": "heal_scar", "scar_event_id": "no_such_scar",
                     "reason": "healing nothing", "scope": "session", "owner_id": "sess_1"}),
            None,
        )
        assert out.status == "failed"


class TestCoercionAndValidation:
    def test_string_args_coerce(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(
            _Run(),
            _effect(_base(sign="-1", magnitude="2", scar="false", bond="false", reason="tool errored")),
            None,
        )
        assert out.status == "completed", getattr(out, "error", None)
        assert out.result["sign"] == -1
        assert out.result["magnitude"] == 2.0
        # "false" strings must NOT become standing peaks (string truthiness trap).
        assert out.result["scar"] is False
        assert len(out.result["event_ids"]) == 1

    def test_uncoercible_scar_fails_loudly(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(_Run(), _effect(_base(scar="maybe")), None)
        assert out.status == "failed"
        assert "boolean" in (out.error or "").lower()

    def test_invalid_sign_fails(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(_Run(), _effect(_base(sign=0)), None)
        assert out.status == "failed"

    def test_requires_reason_turn_id_target(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        assert handler(_Run(), _effect(_base(reason="")), None).status == "failed"
        assert handler(_Run(), _effect(_base(turn_id="")), None).status == "failed"
        assert handler(_Run(), _effect(_base(target_id="")), None).status == "failed"

    def test_unknown_op_fails(self):
        ms = _memory_system()
        handler = _appraise_handler(ms)
        out = handler(_Run(), _effect({"op": "erase", "reason": "x"}), None)
        assert out.status == "failed"


class TestDegradedMode:
    def test_non_strict_appraise_degrades_with_fallback(self):
        class _BrokenMemory:
            # Factory-required seam surface (present but broken).
            def reconstruct(self, *a, **k):
                raise RuntimeError("memory offline")

            def commit_selection(self, *a, **k):
                raise RuntimeError("memory offline")

            def appraise(self, *a, **k):
                raise RuntimeError("valence store offline")

        handlers = build_memory_seam_effect_handlers(
            memory_system=_BrokenMemory(), run_store=None, now_iso=_now_iso, strict=False
        )
        out = handlers[EffectType.MEMORY_APPRAISE](_Run(), _effect(_base()), None)
        assert out.status == "completed"
        assert out.result["degraded"] is True
        assert out.result["event_ids"] == []
        assert any("#FALLBACK" in w for w in out.result["warnings"])
