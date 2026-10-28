from __future__ import annotations

from abstractruntime.visualflow_compiler.visual.builtins import get_builtin_handler


def test_string_is_empty_builtin() -> None:
    is_empty = get_builtin_handler("is_empty_string")
    assert is_empty is not None

    assert is_empty({"text": ""}) is True
    assert is_empty({}) is True
    assert is_empty({"text": None}) is True
    assert is_empty({"text": " "}) is False
    assert is_empty({"text": "hello"}) is False


def test_is_empty_accepts_lists_and_objects() -> None:
    """Operator QA 2026-10-28: Is Empty also takes a list (no elements = empty), so a list
    variable filled by a ForEach is checked with one node instead of Array Length + Compare."""
    is_empty = get_builtin_handler("is_empty_string")
    assert is_empty({"text": []}) is True
    assert is_empty({"text": ()}) is True
    assert is_empty({"text": {}}) is True
    assert is_empty({"text": ["a"]}) is False
    assert is_empty({"text": [None]}) is False
    assert is_empty({"text": [[]]}) is False
    assert is_empty({"text": {"k": 1}}) is False
    assert is_empty({"text": 0}) is False
    assert is_empty({"text": False}) is False


def test_string_contains_builtin() -> None:
    contains = get_builtin_handler("contains")
    assert contains is not None

    assert contains({"text": "hello world", "pattern": "world"}) is True
    assert contains({"text": "hello world", "pattern": "WORLD"}) is False

    # Safer than Python's default behavior where "" is contained in every string.
    assert contains({"text": "hello", "pattern": ""}) is False
    assert contains({"text": "hello", "pattern": None}) is False
    assert contains({"text": None, "pattern": "x"}) is False


def test_string_replace_builtin_modes() -> None:
    replace = get_builtin_handler("replace")
    assert replace is not None

    assert replace({"text": "a-b-c", "pattern": "-", "replacement": "_", "mode": "all"}) == "a_b_c"
    assert replace({"text": "a-b-c", "pattern": "-", "replacement": "_", "mode": "first"}) == "a_b-c"
    assert replace({"text": "a-b-c", "pattern": "-", "replacement": "_", "mode": None}) == "a_b_c"

    # Empty/missing pattern is a no-op.
    assert replace({"text": "a-b-c", "pattern": "", "replacement": "_", "mode": "all"}) == "a-b-c"
    assert replace({"text": "a-b-c", "pattern": None, "replacement": "_", "mode": "all"}) == "a-b-c"

    # Numeric best-effort support: replace first N occurrences.
    assert replace({"text": "a-b-c-d", "pattern": "-", "replacement": "_", "mode": "2"}) == "a_b_c-d"


def test_is_empty_on_a_list_variable_in_a_compiled_flow() -> None:
    """A list variable filled earlier, read by Get Variable into Is Empty, decides an If (the email-triage shape)."""
    from abstractruntime.core.runtime import Runtime
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
    from abstractruntime.visualflow_compiler.compiler import compile_flow
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    x, o = {"id": "exec-in", "type": "execution"}, {"id": "exec-out", "type": "execution"}

    def node(nid, ntype, inputs=(), outputs=(), **data):
        return {"id": nid, "type": ntype, "data": {"nodeType": ntype, "inputs": list(inputs), "outputs": list(outputs), **data}}

    def edge(s, sh, t, th):
        return {"source": s, "sourceHandle": sh, "target": t, "targetHandle": th}

    def run(items):
        nodes = [
            node("start", "on_flow_start", outputs=[o]),
            node("get", "get_var", [{"id": "name", "type": "string"}], [{"id": "value", "type": "array"}], pinDefaults={"name": "triage"}),
            node("empty", "is_empty_string", [{"id": "text", "type": "any"}], [{"id": "result", "type": "boolean"}]),
            node("if", "if", [x, {"id": "condition", "type": "boolean"}], [{"id": "true", "type": "execution"}, {"id": "false", "type": "execution"}]),
            node("none", "on_flow_end", [x, {"id": "response", "type": "string"}], pinDefaults={"response": "none"}),
            node("some", "on_flow_end", [x, {"id": "response", "type": "string"}], pinDefaults={"response": "some"}),
        ]
        edges = [
            edge("start", "exec-out", "if", "exec-in"),
            edge("get", "value", "empty", "text"),
            edge("empty", "result", "if", "condition"),
            edge("if", "true", "none", "exec-in"),
            edge("if", "false", "some", "exec-in"),
        ]
        vf = load_visualflow_json({"id": "t", "name": "t", "entryNode": "start", "nodes": nodes, "edges": edges})
        rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
        wf = compile_flow(visual_to_flow(vf))
        rid = rt.start(workflow=wf, vars={"triage": items})
        state = rt.tick(workflow=wf, run_id=rid, max_steps=50)
        assert state.status.value == "completed", state.error
        return state.output["response"]

    assert run([]) == "none"
    assert run([{"id": "m1"}, {"id": "m2"}]) == "some"
