from __future__ import annotations

from tempfile import TemporaryDirectory

import pytest

from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.visualflow_compiler.compiler import compile_flow
from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json


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

