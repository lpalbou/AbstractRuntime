"""An offloaded trace map must report UNKNOWN tool activity, never zero.

Live evidence 2026-08-21: run `e72c9edf` reported
`meta: {"iterations": 23, "tool_calls": 0, "tool_results": 0}` while making 19
tool calls and carrying 8 of that session's 17 failures. Its sibling
`1d59b704` reported 18/13 correctly.

Cause: when the subtree is offloaded, `scratchpad["node_traces"]` is
`{"$artifact": "<id>"}` — still a `dict`, so the extractor's isinstance guard
passes, the ref's own keys (`$artifact`, `bytes`) are iterated as if they were
node ids, every entry is skipped, and the function returns empty lists. The
count then reads 0.

A zero that means "I could not look" is indistinguishable from "nothing
happened" to every dashboard, cost attribution and triage query. The counts
are omitted instead, with a reason.
"""

from __future__ import annotations

from abstractruntime.storage.artifacts import is_artifact_ref
from abstractruntime.visualflow_compiler import compiler as C
from abstractruntime.visualflow_compiler.compiler import (
    _agent_tool_activity_is_unknown,
    _extract_agent_tool_activity,
)


def test_an_offloaded_trace_map_is_reported_as_unknown():
    """The behaviour, not the source: a ref must not read as a real trace map."""
    assert _agent_tool_activity_is_unknown({"node_traces": {"$artifact": "abc", "bytes": 12}}) is True
    assert _agent_tool_activity_is_unknown({"node_traces": {"n1": {"steps": []}}}) is False
    assert _agent_tool_activity_is_unknown({}) is False
    assert _agent_tool_activity_is_unknown(None) is False


def test_the_ref_keys_are_never_walked_as_node_ids():
    """Before the fix the extractor iterated `$artifact` and `bytes` as nodes."""
    calls, results = _extract_agent_tool_activity({"node_traces": {"$artifact": "abc", "bytes": 12}})
    assert calls == [] and results == []

    # …and a REAL trace map still extracts, so the guard cannot be a blanket.
    trace = {
        "n1": {
            "node_id": "n1",
            "steps": [
                {
                    "ts": "2026-08-21T00:00:00Z",
                    "effect": {
                        "type": "tool_calls",
                        "payload": {"tool_calls": [{"name": "read_file", "arguments": {}}]},
                    },
                    "result": {"results": [{"name": "read_file", "success": True}]},
                }
            ],
        }
    }
    calls2, results2 = _extract_agent_tool_activity({"node_traces": trace})
    assert len(calls2) == 1 and calls2[0]["name"] == "read_file"
    assert len(results2) == 1


def test_both_meta_sites_degrade_to_the_unknown():
    """Guarding one site and not the other leaves the lie half-fixed."""
    import inspect

    src = inspect.getsource(C)
    assert src.count('meta["tool_activity"] = "unknown: node traces were offloaded"') == 2
    assert 'meta.pop("tool_calls", None)' in src
