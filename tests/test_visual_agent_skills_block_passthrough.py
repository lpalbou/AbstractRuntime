"""Agent-node skills passthrough (card 0087, gateway seam c2286).

The gateway resolves `input_data.skills` through abstractskill's trust gate
into the ROOT run's `_runtime.skills_block` — but visual Agent nodes execute
as SUBRUNS whose `_runtime` is built fresh, so the block never reached the
operator-visible basic-agent path. These pins hold the runtime half:

- the parent's skills_block rides into Agent-node subrun vars VERBATIM
  (byte-stable — the prompt-cache contract; never rebuilt);
- `read_skill` joins an EXPLICIT child allowlist when a block rides (the
  both-halves contract: an index in the prompt without a reachable executor
  is dead calls);
- an EMPTY child allowlist (= registry defaults) stays EMPTY — appending
  there would restrict the child to one tool;
- no block, no changes: subrun vars stay byte-identical to before.
"""

from __future__ import annotations

from abstractruntime.core.models import EffectType, RunState
from abstractruntime.visualflow_compiler.compiler import compile_visualflow


def _agent_flow(agent_config: dict) -> object:
    return compile_visualflow(
        {
            "id": "test-flow",
            "name": "test",
            "nodes": [
                {
                    "id": "node-agent",
                    "type": "agent",
                    "data": {"agentConfig": agent_config},
                }
            ],
            "edges": [],
            "entryNode": "node-agent",
        }
    )


def _sub_runtime_ns(spec, run: RunState) -> dict:
    plan = spec.nodes["node-agent"](run, None)
    assert plan.effect is not None
    assert plan.effect.type == EffectType.START_SUBWORKFLOW
    sub_vars = dict(plan.effect.payload or {}).get("vars")
    assert isinstance(sub_vars, dict)
    sub_rt = sub_vars.get("_runtime")
    assert isinstance(sub_rt, dict)
    return sub_rt


BLOCK = "## Skills\n\n- research-notes: how to take notes\n- 2000 bytes of teaching\n"


def test_parent_skills_block_rides_into_agent_subrun_verbatim() -> None:
    spec = _agent_flow({"provider": "lmstudio", "model": "dummy"})
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={
            "_runtime": {"skills_block": BLOCK},
            "_last_output": {"prompt": "go", "include_context": False},
        },
    )
    sub_rt = _sub_runtime_ns(spec, run)
    assert sub_rt.get("skills_block") == BLOCK, "byte-stable passthrough (prompt-cache contract)"


def test_read_skill_joins_an_explicit_child_allowlist() -> None:
    spec = _agent_flow(
        {"provider": "lmstudio", "model": "dummy", "tools": ["read_file", "write_file"]}
    )
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={
            "_runtime": {"skills_block": BLOCK},
            "_last_output": {"prompt": "go", "include_context": False},
        },
    )
    sub_rt = _sub_runtime_ns(spec, run)
    tools = sub_rt.get("allowed_tools")
    assert isinstance(tools, list)
    assert "read_skill" in tools, "the block's executor half must be reachable"
    assert "read_file" in tools and "write_file" in tools, "existing allowlist intact"
    assert tools.count("read_skill") == 1


def test_empty_allowlist_stays_empty_registry_defaults() -> None:
    """[] means registry defaults (which carry read_skill when the host
    registered it) — appending would RESTRICT the child to one tool."""
    spec = _agent_flow({"provider": "lmstudio", "model": "dummy"})
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={
            "_runtime": {"skills_block": BLOCK},
            "_last_output": {"prompt": "go", "include_context": False},
        },
    )
    sub_rt = _sub_runtime_ns(spec, run)
    assert sub_rt.get("allowed_tools") == [], "empty = registry defaults, untouched"
    assert sub_rt.get("skills_block") == BLOCK


def test_no_block_changes_nothing() -> None:
    spec = _agent_flow(
        {"provider": "lmstudio", "model": "dummy", "tools": ["read_file"]}
    )
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={"_last_output": {"prompt": "go", "include_context": False}},
    )
    sub_rt = _sub_runtime_ns(spec, run)
    assert "skills_block" not in sub_rt
    assert sub_rt.get("allowed_tools") == ["read_file"], "no read_skill without a block"


def test_whitespace_only_block_never_rides() -> None:
    """A whitespace-only parent block is treated as absent — nothing rides
    into the child. (Renamed per the 2026-07-15 scan-memo adversary's F
    finding: the old name claimed setdefault protection for a child-supplied
    block, which no child path can produce today — the body only ever
    tested the whitespace rule.)"""
    spec = _agent_flow({"provider": "lmstudio", "model": "dummy"})
    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="node-agent",
        vars={
            "_runtime": {"skills_block": "   "},
            "_last_output": {"prompt": "go", "include_context": False},
        },
    )
    sub_rt = _sub_runtime_ns(spec, run)
    assert "skills_block" not in sub_rt, "whitespace-only blocks never ride"
