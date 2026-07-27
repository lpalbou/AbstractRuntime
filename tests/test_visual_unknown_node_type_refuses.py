"""Connected unknown VisualFlow node types refuse LOUDLY at compile (flow's
live incident, commons c5166): a stale server compiled newer entity-brain
node types as silent `lambda x: x` no-ops — sessions "completed" with
real-looking answers while ZERO memory formed. A flow that runs and lies is
worse than one that refuses: `visual_to_flow` now raises
`UnknownNodeTypeError` for any unknown node with an edge (exec or data),
naming the type and the likely version skew.

Boundary choices pinned here:
- Parsing (`load_visualflow_json`) stays permissive — bundles still parse and
  list on an older server; COMPILATION is the honesty boundary.
- Fully disconnected unknown nodes (decoration/comments) compile fine: they
  can never fire, so refusing them would only break harmless graphs.
- Compile-time (not execution-time) because a raise from a node function
  propagates OUT of Runtime.tick with the run still RUNNING — an
  execution-time refusal would wedge runs instead of failing them.
"""

from __future__ import annotations

import pytest

from abstractruntime.core.models import RunStatus
from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import (
    UnknownNodeTypeError,
    load_visualflow_json,
)


def _flow_with(node_type: str, *, connected: bool = True):
    nodes = [
        {"id": "start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
        {
            "id": "mystery",
            "type": node_type,
            "data": {
                "nodeType": node_type,
                "label": "Mystery",
                "inputs": [{"id": "exec-in", "label": "", "type": "execution"}],
                "outputs": [{"id": "exec-out", "label": "", "type": "execution"}],
            },
        },
        {
            "id": "end",
            "type": "on_flow_end",
            "data": {
                "nodeType": "on_flow_end",
                "inputs": [{"id": "exec-in", "label": "", "type": "execution"}],
            },
        },
    ]
    edges = []
    if connected:
        edges.append({"source": "start", "sourceHandle": "exec-out", "target": "mystery", "targetHandle": "exec-in"})
        edges.append({"source": "mystery", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"})
    else:
        edges.append({"source": "start", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"})
    return load_visualflow_json(
        {
            "id": f"test-unknown-{node_type}",
            "name": "unknown node type",
            "nodes": nodes,
            "edges": edges,
        }
    )


def test_connected_unknown_node_type_refuses_at_compile() -> None:
    with pytest.raises(UnknownNodeTypeError) as exc_info:
        visual_to_flow(_flow_with("memory_recall_v99"))
    msg = str(exc_info.value)
    assert "memory_recall_v99" in msg, "the refusal names the type"
    assert "version skew" in msg


def test_parsing_stays_permissive_for_unknown_types() -> None:
    """load_visualflow_json never refuses — listing/publishing a newer bundle
    on an older server must not crash; only compilation does."""
    vf = _flow_with("memory_recall_v99")
    assert any(n.type == "memory_recall_v99" for n in vf.nodes)


def test_disconnected_unknown_node_compiles_and_runs() -> None:
    """Decoration/comment nodes (zero edges) can never fire — the refusal
    must not break them."""
    wf = compile_flow(visual_to_flow(_flow_with("sticky_comment", connected=False)))
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    rid = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=rid, max_steps=20)
    assert state.status == RunStatus.COMPLETED, f"decoration node broke the run: {state.error}"


def test_compiler_layer_node_types_are_not_refused() -> None:
    """Flow's c5197 false-positive: `memact_compose` (and its compiler-layer
    siblings) have no _create_handler branch — their semantics live in
    compiler.py adapters dispatched by `_visual_type`. The refusal must
    tolerate every one of them."""
    from abstractruntime.visualflow_compiler.visual.executor import COMPILER_LAYER_NODE_TYPES

    for node_type in sorted(COMPILER_LAYER_NODE_TYPES):
        flow = visual_to_flow(_flow_with(node_type))  # must not raise
        assert flow.nodes[  # the base handler stays an input passthrough
            "mystery"
        ].handler is not None


def test_compiler_layer_set_covers_every_compiler_dispatch() -> None:
    """Drift pin: any NEW `visual_type == "..."` dispatch added to
    compiler.py must land in COMPILER_LAYER_NODE_TYPES (or have a real
    _create_handler branch) — otherwise the GAP-2 refusal false-positives on
    it the day it ships. Source-level extraction is deliberate: the elif
    chain is the one authority on what the compiler layer handles."""
    import re
    from pathlib import Path

    import abstractruntime.visualflow_compiler.compiler as compiler_mod
    from abstractruntime.visualflow_compiler.visual.executor import COMPILER_LAYER_NODE_TYPES

    source = Path(compiler_mod.__file__).read_text(encoding="utf-8")
    dispatched = set(re.findall(r'visual_type == "([a-z_0-9]+)"', source))
    assert dispatched, "extraction found nothing - the dispatch pattern moved; update this pin"

    # Types with real _create_handler branches in the executor are covered
    # there; everything else the compiler dispatches must be in the tolerated
    # set. set_var/on_flow_end have executor branches today.
    executor_handled = {"set_var", "on_flow_end"}
    missing = dispatched - executor_handled - set(COMPILER_LAYER_NODE_TYPES)
    assert not missing, (
        f"compiler.py dispatches {sorted(missing)} by visual_type but the "
        "executor neither handles them nor lists them in "
        "COMPILER_LAYER_NODE_TYPES - the unknown-type refusal will "
        "false-positive on them"
    )
