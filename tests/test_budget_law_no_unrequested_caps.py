"""The budget law: never cap, clamp, shrink or truncate a budget the operator
did not ask for; when a caller-declared bound DOES cut, say so (ADR-0026 §1/§2).

Pinned here (adversarial budget audit, 2026-08-02):

1. `_limits` seeds PER KEY. Naming one budget must not suppress the others and
   leave downstream readers on their own fallback constants.
2. The OBSERVE FOLD (tool results -> parent context -> next prompt) carries
   tool evidence WHOLE by default; bounds are opt-in via `_limits` and label
   themselves when they fire.
3. Offloaded tool results are absent from the fold — that absence is stated.
4. The input-token budget only fires when a caller sets one, and a drop is
   never silent.
"""

from __future__ import annotations

from abstractruntime.core.config import RuntimeConfig
from abstractruntime.core.models import RunState, StepPlan
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.memory.token_budget import trim_messages_to_max_input_tokens
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.visualflow_compiler.compiler import compile_visualflow


# --------------------------------------------------------------------------
# 1) `_limits` seeding
# --------------------------------------------------------------------------


def _wf() -> WorkflowSpec:
    def done(run, ctx):
        return StepPlan(node_id="done", complete_output={"ok": True})

    return WorkflowSpec(workflow_id="wf-budget-law", entry_node="done", nodes={"done": done})


def _runtime(**config_kwargs) -> Runtime:
    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        config=RuntimeConfig(**config_kwargs),
    )


def test_partial_limits_do_not_shrink_the_unnamed_budgets() -> None:
    """A caller naming ONE budget must not lose the rest.

    Before the fix `start()` seeded all-or-nothing, so `{"max_iterations": 200}`
    produced a `_limits` holding only that key and every reader fell back to its
    own literal: `get_limit_status` reported a 32,768-token window for a model
    the config had resolved at 262,144.
    """
    rt = _runtime(model_capabilities={"max_tokens": 262_144, "max_output_tokens": 32_768})

    run_id = rt.start(workflow=_wf(), vars={"_limits": {"max_iterations": 200}})
    limits = rt.get_state(run_id).vars["_limits"]

    assert limits["max_iterations"] == 200  # the caller's word is untouched
    assert limits["max_tokens"] == 262_144  # ...and the rest is the config's, not a fallback
    assert limits["max_output_tokens"] == 32_768
    assert limits["max_history_messages"] == -1

    status = rt.get_limit_status(run_id)
    assert status["tokens"]["max"] == 262_144
    assert status["iterations"]["max"] == 200


def test_seeding_never_overwrites_a_narrower_caller_value() -> None:
    """Seeding may only ADD keys: a caller asking for LESS keeps less."""
    rt = _runtime(max_iterations=50, model_capabilities={"max_tokens": 262_144})
    run_id = rt.start(workflow=_wf(), vars={"_limits": {"max_iterations": 3, "max_tokens": 8_192}})
    limits = rt.get_state(run_id).vars["_limits"]
    assert limits["max_iterations"] == 3
    assert limits["max_tokens"] == 8_192


def test_seeding_keeps_the_operator_ceiling_semantics() -> None:
    """The ceiling still tells a DECLARED budget from a silent one.

    Seeding makes `max_iterations` always present, so the ceiling reads the
    caller's declaration captured before the merge — a silent workflow under a
    low ceiling clamps (the default is nobody's word), a declared one refuses.
    """
    rt = _runtime()
    run_id = rt.start(workflow=_wf(), vars={"_limits": {"max_iterations_ceiling": 10}})
    assert rt.get_state(run_id).vars["_limits"]["max_iterations"] == 10


# --------------------------------------------------------------------------
# 2/3) the observe fold
# --------------------------------------------------------------------------


def _agent_run(*, tool_output, limits=None, results=None) -> RunState:
    spec = compile_visualflow(
        {
            "id": "test-flow",
            "name": "test",
            "nodes": [{"id": "node-agent", "type": "agent", "data": {"agentConfig": {}}}],
            "edges": [],
            "entryNode": "node-agent",
        }
    )
    handler = spec.nodes["node-agent"]

    vars_in = {"context": {}}
    if limits is not None:
        vars_in["_limits"] = dict(limits)

    run = RunState.new(workflow_id=spec.workflow_id, entry_node="node-agent", vars=vars_in)
    run.vars["_temp"] = {
        "agent": {
            "node-agent": {
                "phase": "subworkflow",
                "resolved_inputs": {
                    "task": "Read the file.",
                    "context": {},
                    "provider": "lmstudio",
                    "model": "dummy",
                    "include_context": True,
                    "tools": ["read_file"],
                },
                "sub": {
                    "sub_run_id": "sub-1",
                    "output": {"answer": "Done.", "iterations": 2},
                    "node_traces": {
                        "act": {
                            "steps": [
                                {
                                    "ts": "2026-08-02T00:00:00+00:00",
                                    "node_id": "act",
                                    "status": "completed",
                                    "effect": {
                                        "type": "tool_calls",
                                        "payload": {
                                            "tool_calls": [
                                                {"name": "read_file", "arguments": {"path": "big.py"}, "call_id": "c1"}
                                            ]
                                        },
                                    },
                                    "result": {
                                        "results": results
                                        if results is not None
                                        else [
                                            {
                                                "call_id": "c1",
                                                "name": "read_file",
                                                "success": True,
                                                "output": tool_output,
                                                "error": None,
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    },
                },
            }
        }
    }
    handler(run, None)
    return run


def _observation_text(run: RunState) -> str:
    for m in run.vars["context"]["messages"]:
        meta = m.get("metadata") or {}
        if meta.get("kind") == "tool_observation":
            return str(m.get("content") or "")
    raise AssertionError("no tool observation was folded into the parent context")


def test_observe_fold_carries_the_whole_tool_result_by_default() -> None:
    """No unrequested cap on the evidence that becomes the next prompt.

    The old code cut every observation at a hardcoded 2,000 chars — a file read
    reached the outer loop as a stub, which is exactly ADR-0026 §2's forbidden
    case ("durable tool execution outputs used as inputs to later steps").
    """
    body = "LINE\n" * 3000  # 15,000 chars, far past the retired 2,000-char cut
    run = _agent_run(tool_output=body)
    text = _observation_text(run)
    assert body.strip() in text
    assert "#TRUNCATION" not in text


def test_observe_fold_bound_is_opt_in_and_labels_itself() -> None:
    """An operator who WANTS a bound sets it — and the cut names the key."""
    body = "LINE\n" * 3000
    run = _agent_run(tool_output=body, limits={"agent_observation_max_chars": 500})
    text = _observation_text(run)
    assert len(text) < len(body)
    assert "#TRUNCATION" in text
    assert "_limits.agent_observation_max_chars" in text
    assert "sub_run_id=sub-1" in text  # the full result is still reachable


def test_observe_fold_states_offloaded_results_instead_of_hiding_them() -> None:
    """A trace-bounded (`$artifact`) result is dropped from the fold — say so.

    `_extract_tool_activity_from_steps` skips refs, so an oversized result left
    NO trace at all: the outer loop could not tell "tool never ran" from "tool
    ran and its output was offloaded", and re-ran it forever.
    """
    run = _agent_run(tool_output="", results={"$artifact": "art-123"})
    contents = [str(m.get("content") or "") for m in run.vars["context"]["messages"]]
    notice = [c for c in contents if "#TRUNCATION" in c and "offloaded" in c]
    assert notice, contents
    assert "sub_run_id=sub-1" in notice[0]


# --------------------------------------------------------------------------
# 4) input-token budget
# --------------------------------------------------------------------------


def test_input_trim_is_a_no_op_without_a_caller_budget() -> None:
    messages = [{"role": "user", "content": "x" * 10_000}, {"role": "assistant", "content": "y" * 10_000}]
    for budget in (0, -1):
        assert trim_messages_to_max_input_tokens(messages, max_input_tokens=budget) == messages


def test_input_trim_says_what_it_dropped() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old " * 500},
        {"role": "assistant", "content": "older reply " * 500},
        {"role": "user", "content": "now"},
    ]
    out = trim_messages_to_max_input_tokens(messages, max_input_tokens=50)
    assert out[-1]["content"] == "now"  # newest turn always survives
    notices = [m for m in out if "#TRUNCATION" in str(m.get("content") or "")]
    assert len(notices) == 1
    assert "max_input_tokens=50" in notices[0]["content"]
    assert notices[0]["metadata"]["dropped_messages"] == 2
