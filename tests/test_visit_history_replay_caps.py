"""History replay caps for the entity visit lane (ephemeral incident,
operator 2026-08-01).

The visit's react transcript (context.messages) is durable and replayed
WHOLE into every reason call; a single 494,932-char tool message (a 5MB
screenshot read as text) rode a live session into an upstream
context-window rejection and wedged it permanently. The repair is at the
packing seam (the adapter clamps per-message at the payload boundary,
honoring `_limits`); THIS suite pins the runtime side of the contract:

- BRIDGE seeds the entity lane's caps into `_limits` (soft — an
  operator/door-configured int wins, including explicit <= 0);
- the cap values genuinely derive from the seam arithmetic they cite
  (abstractmemory ENTITY_CONTEXT_RECOMMENDED at 4 chars/token) and admit
  every honestly-capped walled tool result whole.

Uses a contract-faithful STUB middle (no abstractagent import — the
dependency points the other way; agent's suite pins the same seam with the
real cycle from their side, test_history_monster_clamp.py).
"""
from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractruntime.core.models import Effect, EffectType, RunStatus, StepPlan  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    HARVEST_NODE,
    VISIT_HISTORY_MESSAGE_CAP_CHARS,
    VISIT_HISTORY_TOOL_RESULT_CAP_CHARS,
    VISITOR_WAIT_KEY,
    ReactMiddle,
    build_visit_workflow,
)


def test_caps_derive_from_the_seam_arithmetic() -> None:
    """The caps were SIZED against the 40k-era recommendation (the
    incident-day derivation: 20% and half of 160k working chars) and
    deliberately KEPT at those values when the same day's re-ruling moved
    the target to 50k — they are poison guards, not attention sizing, and
    the re-ruling touched no history caps. What must HOLD, at whatever the
    live recommendation is: the exact ruled values, the admit-honest-work
    bound, and the seam's own starvation bound."""
    from abstractmemory import ENTITY_CONTEXT_RECOMMENDED

    from abstractruntime.identity.tools import _EXEC_OUTPUT_CAP, WORKSPACE_TEXT_READ_CAP_CHARS

    working_chars = ENTITY_CONTEXT_RECOMMENDED * 4  # repo heuristic: 4 chars/token
    assert working_chars == 200_000  # 50k target (operator 2026-08-01 re-ruling)
    # Tool results: the ruled 32k (8k tokens — 20% of the 40k target it was
    # derived against; 16% of the live 50k recommendation)...
    assert VISIT_HISTORY_TOOL_RESULT_CAP_CHARS == 32_000
    # ...and every honestly-capped walled tool result fits WHOLE with
    # framing to spare — the clamp exists for monsters, never honest work.
    assert VISIT_HISTORY_TOOL_RESULT_CAP_CHARS >= _EXEC_OUTPUT_CAP + 1_000
    assert VISIT_HISTORY_TOOL_RESULT_CAP_CHARS >= WORKSPACE_TEXT_READ_CAP_CHARS + 1_000
    # Prose (entity/visitor words): the ruled 80k (half of the 40k-era
    # working chars) must stay AT OR UNDER the live starvation bound — the
    # seam's own arithmetic (token_fraction <= 0.5): half the recommended
    # context per single message.
    assert VISIT_HISTORY_MESSAGE_CAP_CHARS == 80_000
    assert VISIT_HISTORY_MESSAGE_CAP_CHARS <= working_chars // 2


def _make_home(tmp_path: Path, slug: str = "clampling") -> Path:
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}@home-test"
    spark = _copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = slug.capitalize()
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store, journal = SQLiteTripleStore(db), SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


def _limits_probe_middle(seen_limits: List[Dict[str, Any]]) -> ReactMiddle:
    """Contract-faithful one-iteration middle: records `_limits` as BRIDGE
    left them, answers, exits to HARVEST."""

    def reason(run: Any, ctx: Any) -> StepPlan:
        seen_limits.append(dict(run.vars.get("_limits") or {}))
        temp = run.vars.setdefault("_temp", {})
        temp["final_answer"] = "seen"
        temp["turn_captures"] = {"diary_entries": [], "act_only_warnings": []}
        return StepPlan(node_id="reason", next_node=HARVEST_NODE)

    return ReactMiddle(nodes={"reason": reason}, entry="reason")


def _drive_one_turn(tmp_path: Path, slug: str, *, start_vars: Dict[str, Any]) -> List[Dict[str, Any]]:
    home_dir = _make_home(tmp_path, slug=slug)

    def llm(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        return EffectOutcome.completed({"content": "…"})

    ert = open_entity_runtime(home_dir, extra_handlers={EffectType.LLM_CALL: llm})
    try:
        seen: List[Dict[str, Any]] = []
        wf = build_visit_workflow(
            ert.home,
            participants=["person:albou"],
            idle_seconds=3600,
            model_info={"provider": "test", "model": "scripted"},
            react_middle=_limits_probe_middle(seen),
        )
        run_id = ert.runtime.start(workflow=wf, vars=dict(start_vars), session_id="visit-caps")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        assert state.status == RunStatus.WAITING
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "hello", "speaker": "person:albou"}, max_steps=200,
        )
        assert state.status == RunStatus.WAITING
        assert seen, "the middle's reason node must have run"
        return seen
    finally:
        ert.close()


def test_bridge_seeds_replay_caps_when_unset(tmp_path: Path) -> None:
    seen = _drive_one_turn(tmp_path, "seeded", start_vars={})
    assert seen[0].get("max_tool_message_chars") == VISIT_HISTORY_TOOL_RESULT_CAP_CHARS
    assert seen[0].get("max_message_chars") == VISIT_HISTORY_MESSAGE_CAP_CHARS


def test_bridge_seeding_is_soft_configured_values_win(tmp_path: Path) -> None:
    """A door/operator-configured int stands — including an explicit
    'unbounded by choice' (<= 0); the adapter's shared 200k monster guard
    still floors that lane."""
    seen = _drive_one_turn(
        tmp_path, "cfged",
        start_vars={"_limits": {"max_tool_message_chars": 5_000, "max_message_chars": -1}},
    )
    assert seen[0].get("max_tool_message_chars") == 5_000
    assert seen[0].get("max_message_chars") == -1
