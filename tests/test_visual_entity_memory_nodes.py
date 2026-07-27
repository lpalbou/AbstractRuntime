"""Entity-memory brain nodes (abstractflow 0153): VisualFlow -> MEMORY_*/DIARY_* effects.

These nodes make the entity's cognition graph animatable as VisualFlow
subflows: each node emits the matching first-class effect via the
camera/tool_invoke `_pending_effect` precedent, and the handlers resolve
ONLY on an entity runtime (channel authority, never payload authority).

Pinned here:
1. every node type compiles and dispatches the RIGHT EffectType,
2. payloads mirror pins exactly, with unset/empty pins omitted
   (so seam-handler defaults apply),
3. JSON-string pin defaults for structured pins parse to real structures,
4. the result sync exposes result/success + convenience pins.
"""

from __future__ import annotations

from typing import Any, Dict

from abstractruntime.core.models import EffectType, RunState
from abstractruntime.visualflow_compiler.compiler import compile_visualflow
from abstractruntime.visualflow_compiler.visual.executor import ENTITY_MEMORY_EFFECT_PINS


def _one_node_flow(node_type: str, pin_defaults: Dict[str, Any], inputs: list[Dict[str, Any]]):
    return {
        "id": f"flow-{node_type}",
        "name": f"flow {node_type}",
        "nodes": [
            {
                "id": "n1",
                "type": node_type,
                "data": {
                    "inputs": [{"id": "exec-in", "type": "execution"}, *inputs],
                    "outputs": [
                        {"id": "exec-out", "type": "execution"},
                        {"id": "result", "type": "object"},
                        {"id": "success", "type": "boolean"},
                    ],
                    "pinDefaults": pin_defaults,
                },
            }
        ],
        "edges": [],
        "entryNode": "n1",
    }


def _plan_for(node_type: str, pin_defaults: Dict[str, Any], inputs: list[Dict[str, Any]]):
    spec = compile_visualflow(_one_node_flow(node_type, pin_defaults, inputs))
    run = RunState.new(workflow_id=spec.workflow_id, entry_node="n1", vars={})
    return spec.nodes["n1"](run, None)


def test_all_entity_memory_node_types_dispatch_the_right_effect():
    expected = {
        "memory_recall": EffectType.MEMORY_RECALL,
        "memory_commit": EffectType.MEMORY_ACCESS,
        "memory_form": EffectType.MEMORY_FORM,
        "memory_adjust": EffectType.MEMORY_ADJUST,
        "memory_appraise": EffectType.MEMORY_APPRAISE,
        "diary_write": EffectType.DIARY_WRITE,
        "diary_read": EffectType.DIARY_READ,
        "memory_consolidate": EffectType.MEMORY_CONSOLIDATE,
        "memory_probe": EffectType.MEMORY_PROBE,
        "life_query": EffectType.LIFE_QUERY,
        # The tend route (runtime c5215): body verbatim; grammar engine-owned.
        "memory_tend": EffectType.MEMORY_TEND,
        # The tool surface (flow c5285 ask 3 / runtime same-hour ship): grant
        # query + one-batch execution — the loop stays in the flow graph.
        "entity_tools_query": EffectType.ENTITY_TOOLS_QUERY,
        "entity_tools_execute": EffectType.ENTITY_TOOLS_EXECUTE,
    }
    assert set(expected) == set(ENTITY_MEMORY_EFFECT_PINS)
    for node_type, eff in expected.items():
        # Give each node one harmless string pin so compilation is realistic.
        _, arg_pins = ENTITY_MEMORY_EFFECT_PINS[node_type]
        first_pin = arg_pins[0]
        plan = _plan_for(
            node_type,
            {first_pin: "x"},
            [{"id": first_pin, "type": "string"}],
        )
        assert plan.effect is not None, node_type
        assert plan.effect.type == eff, node_type
        assert plan.effect.payload.get(first_pin) == "x", node_type


def test_memory_recall_payload_mirrors_pins_and_omits_unset():
    plan = _plan_for(
        "memory_recall",
        {
            "cue_text": "what do I know about laurent",
            "view": "working_set",
            "effort": "standard",
            "turn_id": "turn-1",
            # unset pins: scopes, budget, participants, ... must be OMITTED
            "escalation_reason": "",  # empty string = not provided
        },
        [
            {"id": "cue_text", "type": "string"},
            {"id": "view", "type": "string"},
            {"id": "effort", "type": "string"},
            {"id": "turn_id", "type": "string"},
            {"id": "escalation_reason", "type": "string"},
        ],
    )
    p = plan.effect.payload
    assert p.get("cue_text") == "what do I know about laurent"
    assert p.get("view") == "working_set"
    assert p.get("effort") == "standard"
    assert p.get("turn_id") == "turn-1"
    assert "escalation_reason" not in p
    assert "scopes" not in p and "budget" not in p and "participants" not in p


def test_memory_form_json_string_records_parse():
    records_json = '[{"kind": "episode", "title": "t", "digest": "d"}]'
    plan = _plan_for(
        "memory_form",
        {"records": records_json, "turn_id": "turn-2"},
        [
            {"id": "records", "type": "array"},
            {"id": "turn_id", "type": "string"},
        ],
    )
    p = plan.effect.payload
    assert isinstance(p.get("records"), list)
    assert p["records"][0]["kind"] == "episode"
    assert p.get("turn_id") == "turn-2"


def test_diary_write_payload_carries_text_and_visibility():
    plan = _plan_for(
        "diary_write",
        {"text": "today I learned...", "visibility": "private", "kind": "reflection", "turn_id": "t3"},
        [
            {"id": "text", "type": "string"},
            {"id": "visibility", "type": "string"},
            {"id": "kind", "type": "string"},
            {"id": "turn_id", "type": "string"},
        ],
    )
    assert plan.effect.type == EffectType.DIARY_WRITE
    p = plan.effect.payload
    assert p.get("text") == "today I learned..."
    assert p.get("visibility") == "private"
    assert p.get("kind") == "reflection"


def test_entity_scope_owner_resolves_bare_entity_scopes():
    """Home-bound seams resolve bare self/diary/life to the home owner
    (channel authority); without the param the legacy hard-error stands;
    and effort-preset recalls seat identity by right (self_fraction > 0)."""
    from abstractruntime.integrations.abstractmemory.seam_handlers import (
        _build_budget,
        _resolve_scope_pairs,
    )

    class _Budget:
        __dataclass_fields__ = {"self_fraction": None, "shelf_size": None}

        def __init__(self, **kw):
            self.self_fraction = kw.get("self_fraction", 0.0)
            self.shelf_size = kw.get("shelf_size", 12)

    run = RunState.new(workflow_id="w", entry_node="n", vars={})

    # (a) bare entity scopes resolve to the home owner.
    pairs = _resolve_scope_pairs(
        run, ["self", "diary", "life"], run_store=None,
        entity_scope_owner="entity:x",
    )
    assert pairs == [("self", "entity:x"), ("diary", "entity:x"), ("life", "entity:x")]

    # (b) without the param, bare entity scopes stay the legacy hard error.
    import pytest

    with pytest.raises(Exception):
        _resolve_scope_pairs(run, ["self"], run_store=None)

    # (c) home-bound effort presets seat the self core at the posture.
    budget, _ = _build_budget(_Budget, {"effort": "standard"}, entity_scope_owner="entity:x")
    assert budget.self_fraction == 0.5
    budget2, _ = _build_budget(_Budget, {"effort": "standard"})
    assert budget2.self_fraction == 0.0
    # An explicit budget always wins verbatim.
    budget3, _ = _build_budget(_Budget, {"budget": {"self_fraction": 0.1}}, entity_scope_owner="entity:x")
    assert budget3.self_fraction == 0.1


def test_absorbed_failure_reads_as_success_false():
    """continueOnError lands {"ok": False, "absorbed_failure": ...} at the
    result key — the sync must surface success=False (adversary-5 P1-1)."""
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    flow_def = _one_node_flow(
        "memory_form",
        {"records": '[{"kind": "episode", "title": "t", "digest": "d"}]', "turn_id": "t1"},
        [{"id": "records", "type": "array"}, {"id": "turn_id", "type": "string"}],
    )
    vf = load_visualflow_json(flow_def)
    flow = visual_to_flow(vf)
    run = RunState.new(workflow_id=flow_def["id"], entry_node="n1", vars={})
    run.vars["_temp"] = {
        "effects": {"n1": {"ok": False, "absorbed_failure": "MEMORY_FORM requires payload.records"}}
    }
    _sync_effect_results_to_node_outputs(run, flow)
    out = flow._node_outputs.get("n1") or {}
    assert out.get("success") is False


def test_result_sync_exposes_convenience_pins():
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    flow_def = _one_node_flow(
        "memory_recall",
        {"cue_text": "c"},
        [{"id": "cue_text", "type": "string"}],
    )
    vf = load_visualflow_json(flow_def)
    flow = visual_to_flow(vf)

    run = RunState.new(workflow_id=flow_def["id"], entry_node="n1", vars={})
    run.vars["_temp"] = {
        "effects": {
            "n1": {
                "trace_id": "trace_abc",
                "view": "working_set",
                "as_of_seq": 42,
                "handles": [{"record_id": "ex:1"}],
            }
        }
    }
    _sync_effect_results_to_node_outputs(run, flow)
    out = flow._node_outputs.get("n1") or {}
    assert out.get("success") is True
    assert out.get("trace_id") == "trace_abc"
    assert out.get("as_of_seq") == 42
    assert isinstance(out.get("result"), dict)
