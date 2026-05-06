from __future__ import annotations

from types import SimpleNamespace

from abstractruntime.visualflow_compiler.adapters.control_adapter import create_while_node_handler


def _run():
    return SimpleNamespace(vars={})


def test_while_true_routes_to_loop_even_when_done_is_connected() -> None:
    run = _run()
    handler = create_while_node_handler(
        node_id="while",
        loop_target="body",
        done_target="done",
        resolve_condition=lambda _run: True,
    )

    plan = handler(run, None)

    assert plan.next_node == "body"
    assert plan.complete_output is None
    assert run.vars["_temp"]["prev_exec_handle"] == "loop"


def test_while_false_routes_to_done_when_connected() -> None:
    run = _run()
    handler = create_while_node_handler(
        node_id="while",
        loop_target="body",
        done_target="done",
        resolve_condition=lambda _run: False,
    )

    plan = handler(run, None)

    assert plan.next_node == "done"
    assert plan.complete_output is None
    assert run.vars["_temp"]["prev_exec_handle"] == "done"


def test_while_false_without_done_completes_instead_of_looping() -> None:
    run = _run()
    handler = create_while_node_handler(
        node_id="while",
        loop_target="body",
        done_target=None,
        resolve_condition=lambda _run: False,
    )

    plan = handler(run, None)

    assert plan.next_node is None
    assert plan.complete_output == {"success": True, "result": None}
