"""Every tool approval gets its OWN durable wait_key.

Defect (live 2026-07-31, multiagent/bugfix + the 36-run benchmark wave): an
agent node loops on the SAME node, and the TOOL_CALLS handler fell back to
`tool_calls:{run_id}:{node_id}` for the wait_key -- one constant key for every
approval round of the run. Any approver that deduplicates by key (a sane
idempotency measure, and what the flow editor's approval panel and both
AbstractFlow drivers do) answered approval #1 and then parked the run forever
on approval #2. The other branch used `tool_approval:{uuid4}`: distinct, but
re-randomised on every crash-replay, so a key handed out could not be
recomputed.

The contract these tests pin:
  1. distinct wait_key per approval instance, same run, same node;
  2. identical wait_key when the SAME approval is replayed after a crash;
  3. a run parked on an OLD-style key stays approvable (no silent bricking);
  4. an explicit payload/executor key still wins.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunStatus,
    StepPlan,
    WorkflowSpec,
)
from abstractruntime.core.event_keys import (
    TOOL_APPROVAL_WAIT_KEY_PREFIX,
    build_tool_approval_wait_key,
)
from abstractruntime.core.models import WaitReason, WaitState
from abstractruntime.core.runtime import Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.integrations.abstractcore.tool_executor import (
    ApprovalToolExecutor,
    MappingToolExecutor,
    ToolApprovalPolicy,
)


def _write_file(*, path: str, content: str) -> Dict[str, Any]:
    return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}


def _agent_like_workflow(*, rounds: int, workflow_id: str) -> WorkflowSpec:
    """One node ("act") that asks for the SAME tool batch `rounds` times.

    Deliberately byte-identical batches: that is the agent-loop shape that
    collapsed onto a single wait_key, and the shape a name/arguments digest
    alone would still collapse.
    """

    def act(run, ctx) -> StepPlan:
        del ctx
        done = int((run.vars.get("rounds_done") or 0))
        if done >= rounds:
            return StepPlan(node_id="act", complete_output={"rounds_done": done})
        run.vars["rounds_done"] = done + 1
        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": [{"name": "write_file", "arguments": {"path": "a.txt", "content": "hello"}}]},
                result_key="tool_results",
            ),
            next_node="act",
        )

    return WorkflowSpec(workflow_id=workflow_id, entry_node="act", nodes={"act": act})


def _runtime_with_approvals() -> tuple[Runtime, ApprovalToolExecutor]:
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"write_file": _write_file}), policy=ToolApprovalPolicy())
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )
    runtime.set_tool_executor_for_resume(tools)
    return runtime, tools


def test_sequential_approvals_get_distinct_wait_keys() -> None:
    runtime, _tools = _runtime_with_approvals()
    workflow = _agent_like_workflow(rounds=3, workflow_id="wait_key_distinct_test")

    run_id = runtime.start(workflow=workflow)
    seen: List[str] = []
    state = runtime.tick(workflow=workflow, run_id=run_id)
    for _ in range(3):
        assert state.status == RunStatus.WAITING
        assert state.waiting is not None
        key = state.waiting.wait_key
        assert isinstance(key, str) and key.startswith(f"{TOOL_APPROVAL_WAIT_KEY_PREFIX}:")
        assert run_id in key
        seen.append(key)
        state = runtime.resume(
            workflow=workflow, run_id=run_id, wait_key=key, payload={"approved": True}, max_steps=4
        )

    assert state.status == RunStatus.COMPLETED
    # THE regression: three approvals at one node of one run, three keys.
    assert len(set(seen)) == 3, f"approvals reused a wait_key: {seen}"


def test_replayed_approval_keeps_the_same_wait_key() -> None:
    """A crash-replay of the SAME approval must recompute the SAME key.

    The key's identity segment IS the effect's idempotency key -- the
    runtime's own at-most-once identity (run + node + normalized payload +
    the run's effect-issuance counter). That counter advances in the same save
    that lands a step, so a replay recomputes the same key by construction and
    a genuine later issuance does not. Asserting the tie to the LEDGERED
    idempotency key is stronger than re-ticking: it says the key is a pure
    function of durable state, which is exactly what replay-stability means.
    """
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"write_file": _write_file}), policy=ToolApprovalPolicy())
    ledger = InMemoryLedgerStore()
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)},
    )
    runtime.set_tool_executor_for_resume(tools)
    workflow = _agent_like_workflow(rounds=2, workflow_id="wait_key_replay_test")

    run_id = runtime.start(workflow=workflow)
    first = runtime.tick(workflow=workflow, run_id=run_id)
    assert first.waiting is not None
    key = first.waiting.wait_key

    idempotency_keys = {
        str(record.get("idempotency_key") or "")
        for record in ledger.list(run_id)
        if str(record.get("node_id") or "") == "act" and record.get("idempotency_key")
    }
    assert idempotency_keys, "the approval step must be ledgered with an idempotency key"
    assert any(key.endswith(ik) for ik in idempotency_keys), (
        f"wait_key {key!r} is not tied to the effect idempotency identity {idempotency_keys!r}"
    )

    replayed = runtime.tick(workflow=workflow, run_id=run_id)
    assert replayed.waiting is not None
    assert replayed.waiting.wait_key == key

    # And the key a fresh host reads back from the persisted state matches.
    reloaded = runtime.get_state(run_id)
    assert reloaded.waiting is not None
    assert reloaded.waiting.wait_key == key


def test_run_parked_on_a_legacy_wait_key_is_still_approvable() -> None:
    """Backward compatibility: no in-flight run is bricked by the upgrade.

    The key lives in the persisted WaitState; resume validates against THAT,
    never against a recomputation. A run parked on the pre-fix
    `tool_calls:{run_id}:{node_id}` key therefore resumes unchanged.
    """
    runtime, _tools = _runtime_with_approvals()
    workflow = _agent_like_workflow(rounds=1, workflow_id="wait_key_legacy_test")

    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id)
    assert state.waiting is not None

    legacy_key = f"tool_calls:{run_id}:act"
    state.waiting = WaitState(
        reason=WaitReason.USER,
        wait_key=legacy_key,
        resume_to_node=state.waiting.resume_to_node,
        result_key=state.waiting.result_key,
        details=state.waiting.details,
    )
    runtime.run_store.save(state)

    resumed = runtime.resume(
        workflow=workflow, run_id=run_id, wait_key=legacy_key, payload={"approved": True}, max_steps=6
    )
    assert resumed.status == RunStatus.COMPLETED


def test_explicit_payload_and_executor_keys_still_win() -> None:
    pinned_tools = ApprovalToolExecutor(
        delegate=MappingToolExecutor({"write_file": _write_file}),
        policy=ToolApprovalPolicy(),
        wait_key_factory=lambda: "executor-owned",
    )
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.TOOL_CALLS: make_tool_calls_handler(tools=pinned_tools)},
    )

    def act(run, ctx) -> StepPlan:
        del ctx, run
        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={
                    "tool_calls": [{"name": "write_file", "arguments": {"path": "a", "content": "b"}}],
                    "wait_key": "payload-owned",
                },
                result_key="tool_results",
            ),
            next_node="act",
        )

    workflow = WorkflowSpec(workflow_id="wait_key_explicit_test", entry_node="act", nodes={"act": act})
    run_id = runtime.start(workflow=workflow)
    state = runtime.tick(workflow=workflow, run_id=run_id)
    assert state.waiting is not None
    assert state.waiting.wait_key == "payload-owned"


def test_builder_is_unique_per_issuance_and_stable_per_replay() -> None:
    calls = [{"name": "write_file", "arguments": {"path": "a", "content": "b"}}]

    a = build_tool_approval_wait_key(run_id="r1", node_id="act", effect_seq=1, tool_calls=calls)
    a_again = build_tool_approval_wait_key(run_id="r1", node_id="act", effect_seq=1, tool_calls=calls)
    b = build_tool_approval_wait_key(run_id="r1", node_id="act", effect_seq=2, tool_calls=calls)

    assert a == a_again, "same issuance must recompute the same key"
    assert a != b, "a later issuance of an identical batch must get a new key"

    # The effect's own idempotency identity wins when supplied.
    keyed = build_tool_approval_wait_key(run_id="r1", node_id="act", effect_idempotency_key="deadbeef", tool_calls=calls)
    assert keyed.endswith("deadbeef")
    assert keyed.startswith(f"{TOOL_APPROVAL_WAIT_KEY_PREFIX}:r1:act:")

    # Different runs never collide.
    assert build_tool_approval_wait_key(run_id="r2", node_id="act", effect_seq=1, tool_calls=calls) != a

    with pytest.raises(ValueError):
        build_tool_approval_wait_key(run_id="", node_id="act", effect_seq=1, tool_calls=calls)
