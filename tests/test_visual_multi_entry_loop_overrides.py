from __future__ import annotations

from tempfile import TemporaryDirectory

import pytest

from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.multi_entry_lowering import lower_authoring_multi_entry
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json


def _run_until_waiting_prompt(vf, *, max_steps: int = 200) -> str:
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    wf = compile_flow(visual_to_flow(vf))
    run_id = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=max_steps)
    assert state.status.value == "waiting"
    assert state.waiting is not None
    return str(state.waiting.prompt)


def _make_multi_entry_ask_user_prompt_override_visualflow():
    """Authoring graph for a 2-turn AskUser loop.

    Requirements:
    - AskUser has 2 incoming exec edges (Start, If.false) => multi-entry.
    - AskUser.prompt uses pinDefaults ("start") on first entry.
    - AskUser.prompt uses AskUser.response on re-entry (per-entry override).

    The runtime should lower this authoring metadata into internal join_exec/path_mux nodes.
    """

    return load_visualflow_json(
        {
            "id": "test-multi-entry-ask-user-prompt-override",
            "name": "test-multi-entry-ask-user-prompt-override",
            "nodes": [
                {
                    "id": "node-start",
                    "type": "on_flow_start",
                    "data": {"nodeType": "on_flow_start"},
                },
                {
                    "id": "node-count",
                    "type": "var_decl",
                    "data": {
                        "nodeType": "var_decl",
                        "literalValue": {"name": "count", "type": "number", "default": 0},
                    },
                },
                {
                    "id": "node-two",
                    "type": "literal_number",
                    "data": {"nodeType": "literal_number", "literalValue": 2},
                },
                {
                    "id": "node-compare",
                    "type": "compare",
                    "data": {"nodeType": "compare", "pinDefaults": {"op": ">="}},
                },
                {
                    "id": "node-if",
                    "type": "if",
                    "data": {"nodeType": "if"},
                },
                {
                    "id": "node-ask",
                    "type": "ask_user",
                    "data": {
                        "nodeType": "ask_user",
                        "effectConfig": {"allowFreeText": True},
                        "pinDefaults": {"prompt": "start"},
                        "entryRoutes": [
                            {
                                "key": "node-start::exec-out",
                                "sourceNodeId": "node-start",
                                "sourceHandle": "exec-out",
                                "label": "Start",
                            },
                            {
                                "key": "node-if::false",
                                "sourceNodeId": "node-if",
                                "sourceHandle": "false",
                                "label": "Loop",
                            },
                        ],
                        "inputRouteOverrides": {
                            "prompt": {
                                "node-if::false": {
                                    "sourceNodeId": "node-ask",
                                    "sourceHandle": "response",
                                }
                            }
                        },
                    },
                },
                {"id": "node-one", "type": "literal_number", "data": {"nodeType": "literal_number", "literalValue": 1}},
                {"id": "node-add", "type": "add", "data": {"nodeType": "add"}},
                {"id": "node-set", "type": "set_var", "data": {"nodeType": "set_var"}},
                {"id": "node-end", "type": "on_flow_end", "data": {"nodeType": "on_flow_end"}},
            ],
            "edges": [
                # Exec
                {"source": "node-start", "sourceHandle": "exec-out", "target": "node-ask", "targetHandle": "exec-in"},
                {"source": "node-ask", "sourceHandle": "exec-out", "target": "node-set", "targetHandle": "exec-in"},
                {"source": "node-set", "sourceHandle": "exec-out", "target": "node-if", "targetHandle": "exec-in"},
                {"source": "node-if", "sourceHandle": "false", "target": "node-ask", "targetHandle": "exec-in"},
                {"source": "node-if", "sourceHandle": "true", "target": "node-end", "targetHandle": "exec-in"},
                # Data: condition
                {"source": "node-count", "sourceHandle": "value", "target": "node-compare", "targetHandle": "a"},
                {"source": "node-two", "sourceHandle": "value", "target": "node-compare", "targetHandle": "b"},
                {"source": "node-compare", "sourceHandle": "result", "target": "node-if", "targetHandle": "condition"},
                # Data: count increment
                {"source": "node-count", "sourceHandle": "name", "target": "node-set", "targetHandle": "name"},
                {"source": "node-count", "sourceHandle": "value", "target": "node-add", "targetHandle": "a"},
                {"source": "node-one", "sourceHandle": "value", "target": "node-add", "targetHandle": "b"},
                {"source": "node-add", "sourceHandle": "result", "target": "node-set", "targetHandle": "value"},
            ],
            "entryNode": "node-start",
        }
    )


def _make_same_predecessor_branch_visualflow(*, condition: bool):
    return load_visualflow_json(
        {
            "id": f"test-multi-entry-same-predecessor-{condition}",
            "name": "test-multi-entry-same-predecessor",
            "nodes": [
                {"id": "node-start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {"id": "node-cond", "type": "literal_boolean", "data": {"nodeType": "literal_boolean", "literalValue": condition}},
                {"id": "node-if", "type": "if", "data": {"nodeType": "if"}},
                {"id": "node-base-prompt", "type": "literal_string", "data": {"nodeType": "literal_string", "literalValue": "base prompt"}},
                {"id": "node-false-prompt", "type": "literal_string", "data": {"nodeType": "literal_string", "literalValue": "false prompt"}},
                {
                    "id": "node-ask",
                    "type": "ask_user",
                    "data": {
                        "nodeType": "ask_user",
                        "effectConfig": {"allowFreeText": True},
                        "entryRoutes": [
                            {
                                "key": "node-if::true",
                                "sourceNodeId": "node-if",
                                "sourceHandle": "true",
                                "label": "True route",
                            },
                            {
                                "key": "node-if::false",
                                "sourceNodeId": "node-if",
                                "sourceHandle": "false",
                                "label": "False route",
                            },
                        ],
                        "inputRouteOverrides": {
                            "prompt": {
                                "node-if::false": {
                                    "sourceNodeId": "node-false-prompt",
                                    "sourceHandle": "value",
                                }
                            }
                        },
                    },
                },
            ],
            "edges": [
                {"source": "node-start", "sourceHandle": "exec-out", "target": "node-if", "targetHandle": "exec-in"},
                {"source": "node-if", "sourceHandle": "true", "target": "node-ask", "targetHandle": "exec-in"},
                {"source": "node-if", "sourceHandle": "false", "target": "node-ask", "targetHandle": "exec-in"},
                {"source": "node-cond", "sourceHandle": "value", "target": "node-if", "targetHandle": "condition"},
                {"source": "node-base-prompt", "sourceHandle": "value", "target": "node-ask", "targetHandle": "prompt"},
            ],
            "entryNode": "node-start",
        }
    )


def _make_multi_entry_without_overrides_visualflow():
    return load_visualflow_json(
        {
            "id": "test-multi-entry-no-overrides",
            "name": "test-multi-entry-no-overrides",
            "nodes": [
                {"id": "node-start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {"id": "node-cond", "type": "literal_boolean", "data": {"nodeType": "literal_boolean", "literalValue": False}},
                {"id": "node-if", "type": "if", "data": {"nodeType": "if"}},
                {"id": "node-end", "type": "on_flow_end", "data": {"nodeType": "on_flow_end"}},
            ],
            "edges": [
                {"source": "node-start", "sourceHandle": "exec-out", "target": "node-if", "targetHandle": "exec-in"},
                {"source": "node-if", "sourceHandle": "true", "target": "node-end", "targetHandle": "exec-in"},
                {"source": "node-if", "sourceHandle": "false", "target": "node-end", "targetHandle": "exec-in"},
                {"source": "node-cond", "sourceHandle": "value", "target": "node-if", "targetHandle": "condition"},
            ],
            "entryNode": "node-start",
        }
    )


def _make_direct_effect_reentry_visualflow():
    return load_visualflow_json(
        {
            "id": "test-multi-entry-direct-effect-reentry",
            "name": "test-multi-entry-direct-effect-reentry",
            "nodes": [
                {"id": "node-start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {
                    "id": "node-ask",
                    "type": "ask_user",
                    "data": {
                        "nodeType": "ask_user",
                        "effectConfig": {"allowFreeText": True},
                        "pinDefaults": {"prompt": "start"},
                        "entryRoutes": [
                            {
                                "key": "node-start::exec-out",
                                "sourceNodeId": "node-start",
                                "sourceHandle": "exec-out",
                                "label": "Start",
                            },
                            {
                                "key": "node-ask::exec-out",
                                "sourceNodeId": "node-ask",
                                "sourceHandle": "exec-out",
                                "label": "Re-enter",
                            },
                        ],
                        "inputRouteOverrides": {
                            "prompt": {
                                "node-ask::exec-out": {
                                    "sourceNodeId": "node-ask",
                                    "sourceHandle": "response",
                                }
                            }
                        },
                    },
                },
            ],
            "edges": [
                {"source": "node-start", "sourceHandle": "exec-out", "target": "node-ask", "targetHandle": "exec-in"},
                {"source": "node-ask", "sourceHandle": "exec-out", "target": "node-ask", "targetHandle": "exec-in"},
            ],
            "entryNode": "node-start",
        }
    )


@pytest.mark.basic
def test_visual_multi_entry_loop_prompt_override_basic() -> None:
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    vf = _make_multi_entry_ask_user_prompt_override_visualflow()
    wf = compile_flow(visual_to_flow(vf))

    run_id = rt.start(workflow=wf, vars={})

    state1 = rt.tick(workflow=wf, run_id=run_id, max_steps=200)
    assert state1.status.value == "waiting"
    assert state1.waiting is not None
    assert state1.waiting.prompt == "start"

    state2 = rt.resume(
        workflow=wf,
        run_id=run_id,
        wait_key=state1.waiting.wait_key,
        payload={"response": "first"},
        max_steps=200,
    )
    assert state2.status.value == "waiting"
    assert state2.waiting is not None
    assert state2.waiting.prompt == "first"

    state3 = rt.resume(
        workflow=wf,
        run_id=run_id,
        wait_key=state2.waiting.wait_key,
        payload={"response": "second"},
        max_steps=200,
    )
    assert state3.status.value == "completed"


@pytest.mark.integration
def test_visual_multi_entry_loop_prompt_override_restart_simulation() -> None:
    vf = _make_multi_entry_ask_user_prompt_override_visualflow()

    with TemporaryDirectory() as td:
        run_store = JsonFileRunStore(td)
        ledger_store = JsonlLedgerStore(td)

        rt1 = Runtime(run_store=run_store, ledger_store=ledger_store)
        wf1 = compile_flow(visual_to_flow(vf))

        run_id = rt1.start(workflow=wf1, vars={})
        state1 = rt1.tick(workflow=wf1, run_id=run_id, max_steps=200)
        assert state1.status.value == "waiting"
        assert state1.waiting is not None
        assert state1.waiting.prompt == "start"

        # Restart simulation: new runtime instance reading the same file-backed stores.
        rt2 = Runtime(run_store=JsonFileRunStore(td), ledger_store=JsonlLedgerStore(td))
        wf2 = compile_flow(visual_to_flow(vf))

        state2 = rt2.resume(
            workflow=wf2,
            run_id=run_id,
            wait_key=state1.waiting.wait_key,
            payload={"response": "first"},
            max_steps=200,
        )
        assert state2.status.value == "waiting"
        assert state2.waiting is not None
        assert state2.waiting.prompt == "first"

        # Restart again to exercise durability of route selection + cache invalidation.
        rt3 = Runtime(run_store=JsonFileRunStore(td), ledger_store=JsonlLedgerStore(td))
        wf3 = compile_flow(visual_to_flow(vf))

        state3 = rt3.resume(
            workflow=wf3,
            run_id=run_id,
            wait_key=state2.waiting.wait_key,
            payload={"response": "second"},
            max_steps=200,
        )
        assert state3.status.value == "completed"


@pytest.mark.basic
def test_visual_multi_entry_disambiguates_same_predecessor_handles() -> None:
    true_prompt = _run_until_waiting_prompt(_make_same_predecessor_branch_visualflow(condition=True))
    false_prompt = _run_until_waiting_prompt(_make_same_predecessor_branch_visualflow(condition=False))

    assert true_prompt == "base prompt"
    assert false_prompt == "false prompt"


@pytest.mark.basic
def test_visual_multi_entry_direct_effect_reentry_uses_previous_response() -> None:
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    vf = _make_direct_effect_reentry_visualflow()
    wf = compile_flow(visual_to_flow(vf))

    run_id = rt.start(workflow=wf, vars={})
    state1 = rt.tick(workflow=wf, run_id=run_id, max_steps=200)
    assert state1.status.value == "waiting"
    assert state1.waiting is not None
    assert state1.waiting.prompt == "start"

    state2 = rt.resume(
        workflow=wf,
        run_id=run_id,
        wait_key=state1.waiting.wait_key,
        payload={"response": "first"},
        max_steps=200,
    )
    assert state2.status.value == "waiting"
    assert state2.waiting is not None
    assert state2.waiting.prompt == "first"


@pytest.mark.basic
def test_visual_multi_entry_lowering_rejects_stale_route_metadata() -> None:
    vf = _make_same_predecessor_branch_visualflow(condition=False)
    ask = next(n for n in vf.nodes if n.id == "node-ask")
    ask.data["entryRoutes"][1]["sourceHandle"] = "true"

    with pytest.raises(ValueError, match="entryRoutes mismatch"):
        visual_to_flow(vf)


@pytest.mark.basic
def test_visual_multi_entry_without_overrides_uses_join_only() -> None:
    vf = _make_multi_entry_without_overrides_visualflow()
    lowered = lower_authoring_multi_entry(vf)

    assert any(n.id == "__internal_join_exec__node-end" for n in lowered.nodes)
    assert not any(n.id.startswith("__internal_path_mux__") for n in lowered.nodes)

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    wf = compile_flow(visual_to_flow(vf))
    run_id = rt.start(workflow=wf, vars={})
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=200)
    assert state.status.value == "completed"


@pytest.mark.basic
def test_visual_multi_entry_rejects_duplicate_data_edges_to_same_pin() -> None:
    vf = load_visualflow_json(
        {
            "id": "test-duplicate-data-edge",
            "name": "test-duplicate-data-edge",
            "nodes": [
                {"id": "node-start", "type": "on_flow_start", "data": {"nodeType": "on_flow_start"}},
                {"id": "node-a", "type": "literal_string", "data": {"nodeType": "literal_string", "literalValue": "a"}},
                {"id": "node-b", "type": "literal_string", "data": {"nodeType": "literal_string", "literalValue": "b"}},
                {
                    "id": "node-ask",
                    "type": "ask_user",
                    "data": {
                        "nodeType": "ask_user",
                        "effectConfig": {"allowFreeText": True},
                    },
                },
            ],
            "edges": [
                {"source": "node-start", "sourceHandle": "exec-out", "target": "node-ask", "targetHandle": "exec-in"},
                {"source": "node-a", "sourceHandle": "value", "target": "node-ask", "targetHandle": "prompt"},
                {"source": "node-b", "sourceHandle": "value", "target": "node-ask", "targetHandle": "prompt"},
            ],
            "entryNode": "node-start",
        }
    )

    with pytest.raises(ValueError, match="Multiple data edges target the same input pin"):
        visual_to_flow(vf)
