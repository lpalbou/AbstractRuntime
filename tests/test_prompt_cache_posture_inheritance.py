"""`_runtime.prompt_cache` must cross the child-run hops, or `--no-prompt-cache`
is a no-op on the product path.

MEASURED (BENCH-B, 2026-08-03): the client provably serialized
`_runtime.prompt_cache=false` on the ROOT run (src/run_input.rs:239-240,
unit-tested), but the llm_calls live two levels down (run -> coding-agent child ->
per-round grandchildren). `_maybe_inject_prompt_cache_key` reads the EXECUTING
run's vars, found nothing there, re-defaulted to enabled — and all 48 "nocache"
llm_calls derived `session:` keys and reported `hit_extend`, silently invalidating
the A/B. Prior live witness: root afbfe618 carried the posture, child 0b468c2d
carried none.

Two spawn sites build child vars, and the posture must ride both:

- `Runtime._handle_start_subworkflow` (core/runtime.py) — the setdefault rider
  family (workspace keys, skills_block, tool_policy, operator_email);
- the VisualFlow compiler's Agent node (visualflow_compiler/compiler.py) — the
  fresh `_runtime` built for Agent subruns.

The posture differs from every prior rider in one load-bearing way: **False is a
meaningful value**, so the inheritance gate is PRESENCE with the consumer's types
(bool or dict), never truthiness. These tests assert `is False`, not falsy —
"missing" is also falsy, and missing is precisely the bug.
"""

from __future__ import annotations

from typing import Any, Dict

from abstractruntime import (
    Effect,
    EffectType,
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunState,
    RunStatus,
    Runtime,
    StepPlan,
    WaitReason,
    WorkflowRegistry,
    WorkflowSpec,
)
from abstractruntime.core.models import EffectType as VisualEffectType
from abstractruntime.integrations.abstractcore.effect_handlers import (
    _maybe_inject_prompt_cache_key,
)
from abstractruntime.visualflow_compiler.compiler import compile_visualflow


# ---------------------------------------------------------------------------
# Runtime hop: START_SUBWORKFLOW
# ---------------------------------------------------------------------------


def _spawn_child(parent_vars: Dict[str, Any], child_payload_vars: Dict[str, Any] | None = None):
    def child_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="child", complete_output={"ok": True})

    child = WorkflowSpec(workflow_id="child_wf", entry_node="child", nodes={"child": child_node})

    def parent_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="parent",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={
                    "workflow_id": "child_wf",
                    "vars": dict(child_payload_vars or {}),
                    "async": True,
                    "wait": True,
                },
                result_key="sub_result",
            ),
            next_node="after",
        )

    parent = WorkflowSpec(workflow_id="parent_wf", entry_node="parent", nodes={"parent": parent_node})

    reg = WorkflowRegistry()
    reg.register(child)
    reg.register(parent)

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    run_id = rt.start(workflow=parent, vars=parent_vars)

    st = rt.tick(workflow=parent, run_id=run_id, max_steps=1)
    assert st.status == RunStatus.WAITING
    assert st.waiting is not None and st.waiting.reason == WaitReason.SUBWORKFLOW
    sub_run_id = str(st.waiting.wait_key or "").split(":", 1)[1]
    child_run = rt.run_store.load(sub_run_id)
    assert child_run is not None
    return child_run


def _inject_params(child_run: RunState) -> Dict[str, Any]:
    """Run the REAL key derivation against the child run, exactly as the llm_call
    effect handler does, with everything else derivation needs (session id via
    trace_metadata, provider/model) present — so the ONLY variable is the
    inherited posture."""
    params: Dict[str, Any] = {
        "trace_metadata": {"session_id": "sess-bench", "workflow_id": "wf", "node_id": "n"},
    }
    _maybe_inject_prompt_cache_key(
        run=child_run, params=params, default_provider="mlx", default_model="qwen"
    )
    return params


def test_subworkflow_child_inherits_prompt_cache_false_and_derives_no_key() -> None:
    """THE BENCH-B PIN: posture false must reach the child, and an llm_call in the
    child must derive NO key."""
    child_run = _spawn_child({"_runtime": {"prompt_cache": False}})

    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("prompt_cache") is False, (
        "the disable posture must cross the hop — missing means the effect "
        "handler re-defaults to ENABLED and the nocache lane silently caches"
    )

    params = _inject_params(child_run)
    assert "prompt_cache_key" not in params, (
        f"nocache child still derived a key: {params.get('prompt_cache_key')!r}"
    )


def test_subworkflow_child_would_derive_a_key_when_enabled() -> None:
    """Control for the pin above: with the posture enabled the same child DOES
    derive a key — proving the empty-params assertion tests the posture, not a
    broken fixture."""
    child_run = _spawn_child({"_runtime": {"prompt_cache": True}})
    params = _inject_params(child_run)
    assert isinstance(params.get("prompt_cache_key"), str) and params["prompt_cache_key"]


def test_subworkflow_child_inherits_prompt_cache_dict_verbatim() -> None:
    posture = {"enabled": True, "namespace": "bench"}
    child_run = _spawn_child({"_runtime": {"prompt_cache": posture}})

    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("prompt_cache") == posture
    # A COPY crosses, never the parent's aliased dict (tool_policy precedent).
    assert child_rt["prompt_cache"] is not posture

    # The inherited dict drives derivation: the namespace reaches the key.
    params = _inject_params(child_run)
    key = params.get("prompt_cache_key")
    assert isinstance(key, str) and key.startswith("bench:"), key


def test_subworkflow_explicit_child_posture_wins() -> None:
    """setdefault semantics, exactly like every other rider."""
    child_run = _spawn_child(
        {"_runtime": {"prompt_cache": False}},
        child_payload_vars={"_runtime": {"prompt_cache": True}},
    )
    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("prompt_cache") is True


def test_subworkflow_absent_posture_stays_absent() -> None:
    """No posture on the parent -> none invented on the child (the default-ON
    decision stays where it lives, in the effect handler)."""
    child_run = _spawn_child({"_runtime": {"skills_block": "## S\n- x"}})
    child_rt = child_run.vars.get("_runtime") or {}
    assert "prompt_cache" not in child_rt


# ---------------------------------------------------------------------------
# Compiler hop: visual Agent node
# ---------------------------------------------------------------------------


def _agent_flow() -> object:
    return compile_visualflow(
        {
            "id": "test-flow",
            "name": "test",
            "nodes": [
                {
                    "id": "node-agent",
                    "type": "agent",
                    "data": {"agentConfig": {"provider": "lmstudio", "model": "dummy"}},
                }
            ],
            "edges": [],
            "entryNode": "node-agent",
        }
    )


def _agent_sub_runtime(parent_runtime: Dict[str, Any]) -> Dict[str, Any]:
    spec = _agent_flow()
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={
            "_runtime": dict(parent_runtime),
            "_last_output": {"prompt": "go", "include_context": False},
        },
    )
    plan = spec.nodes["node-agent"](run, None)
    assert plan.effect is not None
    assert plan.effect.type == VisualEffectType.START_SUBWORKFLOW
    sub_vars = dict(plan.effect.payload or {}).get("vars")
    assert isinstance(sub_vars, dict)
    sub_rt = sub_vars.get("_runtime")
    assert isinstance(sub_rt, dict)
    return sub_rt


def test_agent_node_child_runtime_carries_prompt_cache_false() -> None:
    sub_rt = _agent_sub_runtime({"prompt_cache": False})
    assert sub_rt.get("prompt_cache") is False, (
        "Agent-node children built a fresh _runtime without the posture, so "
        "their llm_calls re-defaulted to enabled — the --no-prompt-cache no-op"
    )


def test_agent_node_child_runtime_carries_prompt_cache_dict_verbatim() -> None:
    posture = {"enabled": True, "namespace": "bench"}
    sub_rt = _agent_sub_runtime({"prompt_cache": posture})
    assert sub_rt.get("prompt_cache") == posture
    assert sub_rt["prompt_cache"] is not posture      # copy, not alias


def test_agent_node_child_runtime_without_parent_posture_carries_none() -> None:
    sub_rt = _agent_sub_runtime({"audio_policy": "auto"})
    assert "prompt_cache" not in sub_rt
