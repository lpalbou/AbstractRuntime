"""Agent-node mentions passthrough (abstractgateway c26d38a2 + AbstractAgent ff1c4d1).

When the person's message mentions entities, the gateway writes `_runtime.mentions`
(`{entities: [{entity_id, name}], line}`) on the ROOT run and appends the line to its
`context.messages`. A visual Agent node runs its ReAct loop in a CHILD run whose `_runtime`
is built fresh, so the child never saw the mentions and retried an empty reply
("Your previous response was empty") instead of staying silent. These pins hold:

- the parent's `_runtime.mentions` rides into the Agent-node child, equal and as a copy;
- no mentions, no key;
- the line reaches the child's messages the way the conversation does: through the
  Agent node's `context` pin (the shipped basic-agent wiring), last, after the history.
"""

from __future__ import annotations

import copy

from abstractruntime.core.models import EffectType, RunState
from abstractruntime.visualflow_compiler.compiler import compile_visualflow

LINE = (
    "The mentioned entities (Castor) answer for themselves; answer only the part addressed to you, "
    "or say nothing with an empty reply."
)
MENTIONS = {"entities": [{"entity_id": "entity:castor", "name": "Castor"}], "line": LINE}


def _agent_only_flow() -> object:
    return compile_visualflow(
        {
            "id": "mentions-flow",
            "name": "test",
            "nodes": [
                {"id": "node-agent", "type": "agent", "data": {"agentConfig": {"provider": "lmstudio", "model": "dummy"}}}
            ],
            "edges": [],
            "entryNode": "node-agent",
        }
    )


def _child_vars(plan) -> dict:
    assert plan.effect is not None
    assert plan.effect.type == EffectType.START_SUBWORKFLOW
    sub_vars = dict(plan.effect.payload or {}).get("vars")
    assert isinstance(sub_vars, dict)
    return sub_vars


def test_parent_mentions_ride_into_the_agent_child_as_a_copy() -> None:
    spec = _agent_only_flow()
    parent_mentions = copy.deepcopy(MENTIONS)
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={"_runtime": {"mentions": parent_mentions}, "_last_output": {"prompt": "@Castor hi", "include_context": False}},
    )
    child_rt = _child_vars(spec.nodes["node-agent"](run, None))["_runtime"]
    assert child_rt.get("mentions") == MENTIONS, "the child's ReAct loop reads _runtime.mentions to stay silent"
    child_rt["mentions"]["entities"].append({"entity_id": "entity:pollux", "name": "Pollux"})
    assert parent_mentions == MENTIONS, "parent and child never share the dict"


def test_no_mentions_no_key() -> None:
    spec = _agent_only_flow()
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={"_runtime": {}, "_last_output": {"prompt": "hi", "include_context": False}},
    )
    assert "mentions" not in _child_vars(spec.nodes["node-agent"](run, None))["_runtime"]


def _basic_agent_shaped_flow() -> object:
    """On Flow Start's `context` wired into the Agent node's `context` pin (the shipped basic-agent)."""
    return compile_visualflow(
        {
            "id": "mentions-basic",
            "name": "basic",
            "entryNode": "start",
            "nodes": [
                {
                    "id": "start",
                    "type": "on_flow_start",
                    "data": {
                        "nodeType": "on_flow_start",
                        "outputs": [
                            {"id": "exec-out", "type": "execution"},
                            {"id": "context", "type": "object"},
                            {"id": "prompt", "type": "string"},
                            {"id": "provider", "type": "provider"},
                            {"id": "model", "type": "model"},
                        ],
                    },
                },
                {"id": "agent", "type": "agent", "data": {"nodeType": "agent", "agentConfig": {}}},
            ],
            "edges": [
                {"id": "e0", "source": "start", "sourceHandle": "exec-out", "target": "agent", "targetHandle": "exec-in"},
                {"id": "e1", "source": "start", "sourceHandle": "context", "target": "agent", "targetHandle": "context"},
                {"id": "e2", "source": "start", "sourceHandle": "prompt", "target": "agent", "targetHandle": "prompt"},
                {"id": "e3", "source": "start", "sourceHandle": "provider", "target": "agent", "targetHandle": "provider"},
                {"id": "e4", "source": "start", "sourceHandle": "model", "target": "agent", "targetHandle": "model"},
            ],
        }
    )


def test_the_line_reaches_the_child_messages_with_the_conversation_and_mentions_ride_along() -> None:
    spec = _basic_agent_shaped_flow()
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "an answer"}]
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="start",
        vars={
            "prompt": "@Castor does the bridge hold?",
            "provider": "lmstudio",
            "model": "dummy",
            # What the gateway's bundle host leaves on the root run: the history seed, then the line.
            "context": {"task": "@Castor does the bridge hold?", "messages": [*history, {"role": "system", "content": LINE}]},
            "_runtime": {"mentions": copy.deepcopy(MENTIONS)},
        },
    )
    spec.nodes["start"](run, None)
    child = _child_vars(spec.nodes["agent"](run, None))
    roles_contents = [(m.get("role"), m.get("content")) for m in child["context"]["messages"]]
    assert roles_contents == [("user", "earlier"), ("assistant", "an answer"), ("system", LINE)], roles_contents
    assert child["_runtime"].get("mentions") == MENTIONS
