from __future__ import annotations

from abstractruntime.rendering import render_agent_trace_markdown


def test_render_agent_trace_markdown_includes_tool_args_and_results() -> None:
    scratchpad = {
        "sub_run_id": "sub-1",
        "workflow_id": "wf-1",
        "node_traces": {
            "reason": {
                "node_id": "reason",
                "steps": [
                    {
                        "ts": "2026-01-01T00:00:00Z",
                        "node_id": "reason",
                        "status": "completed",
                        "effect": {"type": "llm_call", "payload": {}, "result_key": "_temp.llm"},
                        "result": {
                            "content": "Hello",
                            "tool_calls": None,
                            "model": "test-model",
                            "finish_reason": "stop",
                        },
                    }
                ],
                "updated_at": "2026-01-01T00:00:00Z",
            },
            "act": {
                "node_id": "act",
                "steps": [
                    {
                        "ts": "2026-01-01T00:00:01Z",
                        "node_id": "act",
                        "status": "completed",
                        "effect": {
                            "type": "tool_calls",
                            "payload": {
                                "tool_calls": [
                                    {"name": "web_search", "arguments": {"query": "q"}, "call_id": "c1"}
                                ]
                            },
                            "result_key": "_temp.tools",
                        },
                        "result": {
                            "results": [
                                {
                                    "call_id": "c1",
                                    "name": "web_search",
                                    "success": True,
                                    "output": {"ok": 1},
                                    "error": None,
                                }
                            ]
                        },
                    }
                ],
                "updated_at": "2026-01-01T00:00:01Z",
            },
        },
    }

    md = render_agent_trace_markdown(scratchpad)
    assert "# Agent Trace Report" in md
    assert "`sub-1`" in md
    assert "`wf-1`" in md
    assert "Tool: `web_search` (call_id=c1)" in md
    assert '"query": "q"' in md
    assert '"ok": 1' in md
    assert "- **tool_calls_requested**: none" in md


