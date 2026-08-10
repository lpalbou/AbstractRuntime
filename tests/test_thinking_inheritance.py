"""`_runtime.thinking` must cross the child-run hops, or the reasoning dial is
a no-op on the product path.

MEASURED (2026-08-03 handoff + two adversarial re-investigations): clients set
`_runtime.thinking="medium"` on the ROOT run, but bundle agents do their LLM
work in subflow-spawned CHILD runs (coder -> coding-agent -> verify-gates), and
the react wrappers that actually issue LLM_CALLs are grandchildren whose
visual-Agent spawn inherits from the EXECUTING run — so one stripped hop
propagates the loss all the way down. Store witness: roots `thinking="medium"`
(43f20ff7 / ff6b5240 / 011021ae), subflow children None (46c260da / c10a6081),
react grandchildren None (f211675f) — while the sibling spawned by the visual
Agent node from the MEDIUM root (68312896) carried medium. Five of eight bench
arms ran with reasoning OFF while every layer above reported medium.

Two spawn sites build child vars, and the value must ride both:

- `Runtime._handle_start_subworkflow` (core/runtime.py) — the setdefault rider
  family (workspace keys, skills_block, tool_policy, operator_email,
  prompt_cache); `thinking` is the sixth rider;
- the VisualFlow compiler's Agent node (visualflow_compiler/compiler.py) — the
  fresh `_runtime` built for Agent subruns already inherits
  (pin > agentConfig > parent), pinned here because no other test covered the
  parent leg.

Like prompt_cache, False is a MEANINGFUL value ("reasoning off" is a decision,
not an absence): the inheritance gate is PRESENCE with the consumer's types
(bool, or non-empty str), never truthiness — these tests assert `is False`.

Downstream consumer seams are pinned in their own repos: abstractagent
`runtime_llm_params` forwards the executing run's `_runtime.thinking` into
LLM_CALL params (tests/test_generation_params_media_policies.py), and
abstractcore's openai-compatible provider maps `thinking` to the OpenAI
`reasoning_effort` payload field
(tests/providers/test_openai_compatible_reasoning_effort_unit.py).
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


def test_subworkflow_child_inherits_thinking() -> None:
    """THE 2026-08-03 PIN: the requested reasoning level must reach the child,
    or every LLM call below the first subflow hop silently runs at the
    provider/relay default."""
    child_run = _spawn_child({"_runtime": {"thinking": "medium"}})

    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("thinking") == "medium", (
        "thinking must cross the hop — missing means the react wrapper "
        "grandchild inherits None and the wire carries no reasoning_effort"
    )


def test_subworkflow_explicit_child_thinking_wins() -> None:
    """setdefault semantics, exactly like every other rider."""
    child_run = _spawn_child(
        {"_runtime": {"thinking": "medium"}},
        child_payload_vars={"_runtime": {"thinking": "low"}},
    )
    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("thinking") == "low"


def test_subworkflow_child_inherits_thinking_false() -> None:
    """False is a decision ("reasoning off"), not an absence: the gate is
    presence with the consumer's types. `is False`, never falsy — missing is
    also falsy, and missing is precisely the bug."""
    child_run = _spawn_child({"_runtime": {"thinking": False}})
    child_rt = child_run.vars.get("_runtime") or {}
    assert child_rt.get("thinking") is False


def test_subworkflow_absent_thinking_stays_absent() -> None:
    """No value on the parent -> none invented on the child (the default lives
    downstream: capability-route default, else provider/relay default)."""
    child_run = _spawn_child({"_runtime": {"skills_block": "## S\n- x"}})
    child_rt = child_run.vars.get("_runtime") or {}
    assert "thinking" not in child_rt


def test_subworkflow_whitespace_thinking_stays_absent() -> None:
    """An empty/whitespace string is unset, not a value — the rider must not
    launder it into child state."""
    child_run = _spawn_child({"_runtime": {"thinking": "   "}})
    child_rt = child_run.vars.get("_runtime") or {}
    assert "thinking" not in child_rt


def test_subworkflow_thinking_crosses_two_hops() -> None:
    """The product shape is TWO subflow hops (coder -> coding-agent ->
    coding-verify-gates): the verifier's ABSENT-effort calls in the 2026-08-03
    wire data crossed both. One rider must heal every level."""

    def leaf_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="leaf", complete_output={"ok": True})

    leaf = WorkflowSpec(workflow_id="leaf_wf", entry_node="leaf", nodes={"leaf": leaf_node})

    def mid_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="mid",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={"workflow_id": "leaf_wf", "vars": {}, "async": True, "wait": True},
                result_key="leaf_result",
            ),
            next_node="after",
        )

    mid = WorkflowSpec(workflow_id="mid_wf", entry_node="mid", nodes={"mid": mid_node})

    def root_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(
            node_id="root",
            effect=Effect(
                type=EffectType.START_SUBWORKFLOW,
                payload={"workflow_id": "mid_wf", "vars": {}, "async": True, "wait": True},
                result_key="mid_result",
            ),
            next_node="after",
        )

    root = WorkflowSpec(workflow_id="root_wf", entry_node="root", nodes={"root": root_node})

    reg = WorkflowRegistry()
    for wf in (leaf, mid, root):
        reg.register(wf)

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), workflow_registry=reg)
    root_id = rt.start(workflow=root, vars={"_runtime": {"thinking": "medium"}})

    st = rt.tick(workflow=root, run_id=root_id, max_steps=1)
    assert st.status == RunStatus.WAITING and st.waiting is not None
    mid_id = str(st.waiting.wait_key or "").split(":", 1)[1]
    mid_run = rt.run_store.load(mid_id)
    assert mid_run is not None
    assert (mid_run.vars.get("_runtime") or {}).get("thinking") == "medium"

    st2 = rt.tick(workflow=mid, run_id=mid_id, max_steps=1)
    assert st2.status == RunStatus.WAITING and st2.waiting is not None
    leaf_id = str(st2.waiting.wait_key or "").split(":", 1)[1]
    leaf_run = rt.run_store.load(leaf_id)
    assert leaf_run is not None
    assert (leaf_run.vars.get("_runtime") or {}).get("thinking") == "medium", (
        "second hop stripped the value — the verifier lane stays reasoning-OFF"
    )


# ---------------------------------------------------------------------------
# Compiler hop: visual Agent node (pin > agentConfig > parent `_runtime`)
# ---------------------------------------------------------------------------


def _agent_flow(agent_config: Dict[str, Any] | None = None) -> object:
    cfg: Dict[str, Any] = {"provider": "lmstudio", "model": "dummy"}
    if agent_config:
        cfg.update(agent_config)
    return compile_visualflow(
        {
            "id": "test-flow",
            "name": "test",
            "nodes": [
                {
                    "id": "node-agent",
                    "type": "agent",
                    "data": {"agentConfig": cfg},
                }
            ],
            "edges": [],
            "entryNode": "node-agent",
        }
    )


def _agent_sub_runtime(
    parent_runtime: Dict[str, Any], agent_config: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    spec = _agent_flow(agent_config)
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


def test_agent_node_child_runtime_inherits_parent_thinking() -> None:
    """The inheriting path the 2026-08-03 store differential proved (run
    68312896 medium vs subflow siblings None) — previously unpinned."""
    sub_rt = _agent_sub_runtime({"thinking": "medium"})
    assert sub_rt.get("thinking") == "medium"


def test_agent_node_config_thinking_beats_parent() -> None:
    """Precedence: an explicit agentConfig level outranks inheritance."""
    sub_rt = _agent_sub_runtime({"thinking": "medium"}, agent_config={"thinking": "low"})
    assert sub_rt.get("thinking") == "low"


def test_agent_node_child_runtime_without_parent_thinking_carries_none() -> None:
    sub_rt = _agent_sub_runtime({"audio_policy": "auto"})
    assert "thinking" not in sub_rt


# ---------------------------------------------------------------------------
# LLM_CALL effect seam: every emitter, one injection point
# ---------------------------------------------------------------------------
#
# Wire witness (2026-08-04, probe session acode-74ac46b3b5b2): with the spawn
# hops fixed, the react loops ran 14/14 medium — but the Agent node's
# structured-output post-pass still emitted an ABSENT-effort call, because its
# params carry only the node pin/config and no downstream consumer applied the
# executing run's `_runtime.thinking`. The effect handler is the one seam every
# LLM_CALL crosses (the audio_policy precedent).

from abstractruntime.integrations.abstractcore.effect_handlers import (
    _maybe_inject_runtime_thinking,
)


def _run_with_runtime(runtime_ns: Dict[str, Any]) -> RunState:
    return RunState.new(workflow_id="wf", entry_node="n", vars={"_runtime": dict(runtime_ns)})


def test_effect_seam_injects_run_thinking_when_params_lack_it() -> None:
    params: Dict[str, Any] = {"temperature": 0.1}
    _maybe_inject_runtime_thinking(run=_run_with_runtime({"thinking": "medium"}), params=params)
    assert params.get("thinking") == "medium"


def test_effect_seam_explicit_pin_wins_including_false() -> None:
    """False is a decision: the gate is key-presence, never truthiness."""
    params: Dict[str, Any] = {"thinking": False}
    _maybe_inject_runtime_thinking(run=_run_with_runtime({"thinking": "medium"}), params=params)
    assert params["thinking"] is False


def test_effect_seam_runtime_false_crosses() -> None:
    params: Dict[str, Any] = {}
    _maybe_inject_runtime_thinking(run=_run_with_runtime({"thinking": False}), params=params)
    assert params.get("thinking") is False


def test_effect_seam_absent_stays_absent() -> None:
    params: Dict[str, Any] = {"temperature": 0.1}
    _maybe_inject_runtime_thinking(run=_run_with_runtime({"audio_policy": "auto"}), params=params)
    assert "thinking" not in params


def test_effect_seam_whitespace_is_not_a_value() -> None:
    params: Dict[str, Any] = {}
    _maybe_inject_runtime_thinking(run=_run_with_runtime({"thinking": "  "}), params=params)
    assert "thinking" not in params


# ---------------------------------------------------------------------------
# Call-site proof: the REAL handler must deliver the injection to the client
# ---------------------------------------------------------------------------
#
# The unit tests above pin the helper; the adversary (2026-08-04) proved that
# deleting the single call site left every suite green. These drive
# `make_llm_call_handler` with a capturing client so a removed or misplaced
# call site fails HERE.

from abstractruntime.core.models import Effect as CoreEffect, EffectType as CoreEffectType, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
from abstractruntime.integrations.abstractcore.llm_client import (
    _with_capability_default_reasoning,
)


class _CapturingLLM:
    def __init__(self, content: str = "ok"):
        self.calls: list[Dict[str, Any]] = []
        self._content = content

    def generate(self, *, prompt: str = "", messages=None, system_prompt=None,
                 media=None, tools=None, params=None) -> Dict[str, Any]:
        self.calls.append(dict(params or {}))
        return {"content": self._content, "finish_reason": "stop", "metadata": {}}


def _running_state(runtime_ns: Dict[str, Any]) -> RunState:
    return RunState(
        run_id="run-test",
        workflow_id="wf-test",
        status=RunStatus.RUNNING,
        current_node="node-llm",
        vars={"_runtime": dict(runtime_ns), "_limits": {"max_tokens": 65536}},
    )


def test_llm_call_handler_delivers_run_thinking_to_the_client() -> None:
    llm = _CapturingLLM()
    handler = make_llm_call_handler(llm=llm, artifact_store=None)
    out = handler(
        _running_state({"thinking": "medium"}),
        CoreEffect(type=CoreEffectType.LLM_CALL, payload={"prompt": "hello"}),
        None,
    )
    assert out.status == "completed"
    assert llm.calls and llm.calls[0].get("thinking") == "medium"


def test_llm_call_handler_explicit_false_pin_survives() -> None:
    llm = _CapturingLLM()
    handler = make_llm_call_handler(llm=llm, artifact_store=None)
    handler(
        _running_state({"thinking": "medium"}),
        CoreEffect(
            type=CoreEffectType.LLM_CALL,
            payload={"prompt": "hello", "params": {"thinking": False}},
        ),
        None,
    )
    assert llm.calls and llm.calls[0].get("thinking") is False


def test_structured_fallback_and_repair_retries_keep_run_thinking() -> None:
    """Adversary defect A (2026-08-04): the fallback/repair lanes re-issue
    generation from a params snapshot taken BEFORE the rider block — the
    run-level reasoning vanished exactly when the model was struggling. Every
    call the handler makes, first attempt or retry, must carry it."""
    llm = _CapturingLLM(content="this is not json")
    handler = make_llm_call_handler(llm=llm, artifact_store=None)
    handler(
        _running_state({"thinking": "medium"}),
        CoreEffect(
            type=CoreEffectType.LLM_CALL,
            payload={
                "prompt": "hello",
                "response_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                },
                "structured_output_fallback": True,
            },
        ),
        None,
    )
    assert len(llm.calls) >= 2, "fixture must exercise the fallback lane"
    for i, call in enumerate(llm.calls):
        assert call.get("thinking") == "medium", f"call {i} lost the run-level thinking: {call.keys()}"


def test_seam_then_capability_default_cascade_precedence() -> None:
    """Composed order pin > _runtime > route-default (each half is pinned in
    its own suite; this pins the composition)."""
    configured = {"output.text": {"key": "output.text", "provider": "lmstudio", "model": "q", "reasoning": "high"}}

    # _runtime wins over the configured route default.
    params: Dict[str, Any] = {}
    _maybe_inject_runtime_thinking(run=_run_with_runtime({"thinking": "medium"}), params=params)
    assert _with_capability_default_reasoning(params, configured) == "medium"

    # No _runtime: the configured route default applies.
    params2: Dict[str, Any] = {}
    _maybe_inject_runtime_thinking(run=_run_with_runtime({}), params=params2)
    assert _with_capability_default_reasoning(params2, configured) == "high"

    # An explicit False pin outranks both.
    params3: Dict[str, Any] = {"thinking": False}
    _maybe_inject_runtime_thinking(run=_run_with_runtime({"thinking": "medium"}), params=params3)
    assert _with_capability_default_reasoning(params3, configured) is False
