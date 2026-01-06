"""Agent scratchpad → Markdown report renderer.

Goal:
- Clear, complete, and token-efficient review artifact for agent runs.
- No truncation of tool call arguments or tool execution results.

Input shape:
The "scratchpad" passed around by hosts is expected to include runtime-owned node traces,
typically at `scratchpad["node_traces"]`, which is sourced from:
`RunState.vars["_runtime"]["node_traces"]` (ADR-0010).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .json_stringify import JsonStringifyMode, stringify_json


@dataclass(frozen=True)
class AgentTraceMarkdownReportConfig:
    """Rendering configuration.

    Keep defaults conservative to avoid bloating outputs.
    """

    include_timestamps: bool = False
    json_mode: JsonStringifyMode = JsonStringifyMode.BEAUTIFY


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _collect_trace_steps(node_traces: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Flatten node_traces into a chronologically sortable list of (ts, node_id, entry)."""
    out: list[tuple[str, str, Dict[str, Any]]] = []
    for node_id, trace in node_traces.items():
        t = _as_dict(trace)
        if not t:
            continue
        steps = t.get("steps")
        if not isinstance(steps, list):
            continue
        for entry in steps:
            e = _as_dict(entry)
            if not e:
                continue
            ts = e.get("ts")
            ts_s = ts if isinstance(ts, str) else ""
            out.append((ts_s, node_id, e))
    # ISO timestamps are lexicographically sortable.
    out.sort(key=lambda x: x[0])
    return out


def _code_block(value: Any, *, language: str) -> str:
    """Render a value inside a fenced code block (no truncation)."""
    if language == "json":
        text = stringify_json(value, mode=JsonStringifyMode.BEAUTIFY, sort_keys=False, parse_strings=False)
    else:
        text = "" if value is None else str(value)
    return f"```{language}\n{text}\n```"


def _render_tool_call(call: Dict[str, Any], result: Optional[Dict[str, Any]], *, cfg: AgentTraceMarkdownReportConfig) -> str:
    name = call.get("name")
    call_id = call.get("call_id") or call.get("id") or ""
    args = call.get("arguments", {})

    title = f"#### Tool: `{name}`"
    if isinstance(call_id, str) and call_id:
        title += f" (call_id={call_id})"

    lines: list[str] = [title, "", "**Arguments**", _code_block(args, language="json")]

    if result is None:
        lines.extend(["", "**Result**", "_missing tool result in trace entry_"])
        return "\n".join(lines)

    success = result.get("success") if isinstance(result.get("success"), bool) else None
    error = result.get("error")
    output = result.get("output")

    lines.append("")
    lines.append("**Result**")
    if success is not None:
        lines.append(f"- **success**: {str(success).lower()}")
    if error is not None:
        lines.append(f"- **error**: {error}")

    # Output can be string or JSON.
    if isinstance(output, (dict, list, bool, int, float)) or output is None:
        lines.append(_code_block(output, language="json"))
    else:
        lines.append(_code_block(output, language="text"))

    return "\n".join(lines)


def _index_tool_results_by_call_id(tool_results: Any) -> Dict[str, Dict[str, Any]]:
    """Build a call_id → result mapping from a TOOL_CALLS effect outcome."""
    if not isinstance(tool_results, dict):
        return {}
    results = tool_results.get("results")
    if not isinstance(results, list):
        return {}
    out: dict[str, Dict[str, Any]] = {}
    for r in results:
        rr = _as_dict(r)
        if not rr:
            continue
        call_id = rr.get("call_id")
        if isinstance(call_id, str) and call_id:
            out[call_id] = rr
    return out


def render_agent_trace_markdown(scratchpad: Any, *, config: Optional[AgentTraceMarkdownReportConfig] = None) -> str:
    """Render an agent scratchpad (runtime-owned node traces) into Markdown."""
    cfg = config or AgentTraceMarkdownReportConfig()

    sp = _as_dict(scratchpad)
    if sp is None:
        return "# Agent Trace Report\n\n_No scratchpad provided._\n"

    node_traces = sp.get("node_traces")
    if not isinstance(node_traces, dict):
        # Allow passing node_traces directly.
        if isinstance(scratchpad, dict) and "steps" in scratchpad and "node_id" in scratchpad:
            node_traces = {str(scratchpad.get("node_id")): scratchpad}
        else:
            return "# Agent Trace Report\n\n_No node_traces found in scratchpad._\n"

    header: list[str] = ["# Agent Trace Report"]

    sub_run_id = sp.get("sub_run_id")
    workflow_id = sp.get("workflow_id")
    if isinstance(sub_run_id, str) and sub_run_id:
        header.append(f"- **sub_run_id**: `{sub_run_id}`")
    if isinstance(workflow_id, str) and workflow_id:
        header.append(f"- **workflow_id**: `{workflow_id}`")

    header.append("")

    steps = _collect_trace_steps(node_traces)
    if not steps:
        return "\n".join(header + ["_No trace steps found._", ""])

    lines: list[str] = header + ["## Timeline", ""]

    for idx, (ts, node_id, entry) in enumerate(steps, start=1):
        status = entry.get("status")
        status_s = status if isinstance(status, str) else ""
        effect = _as_dict(entry.get("effect")) or {}
        effect_type = effect.get("type")
        effect_type_s = effect_type if isinstance(effect_type, str) else ""

        lines.append(f"### {idx}. `{node_id}` — `{effect_type_s}` ({status_s})")
        if cfg.include_timestamps and ts:
            lines.append(f"- **ts**: `{ts}`")
        duration_ms = entry.get("duration_ms")
        if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
            lines.append(f"- **duration_ms**: {float(duration_ms):.3f}")

        if status_s == "failed":
            err = entry.get("error")
            if err is not None:
                lines.append("")
                lines.append("**Error**")
                lines.append(_code_block(err, language="text"))
            lines.append("")
            continue

        result = _as_dict(entry.get("result"))

        if effect_type_s == "llm_call":
            # Keep it token-efficient: only show what the LLM produced + whether it asked for tools.
            content = result.get("content") if result else None
            tool_calls = result.get("tool_calls") if result else None
            model = result.get("model") if result else None
            finish_reason = result.get("finish_reason") if result else None

            if isinstance(model, str) and model:
                lines.append(f"- **model**: `{model}`")
            if isinstance(finish_reason, str) and finish_reason:
                lines.append(f"- **finish_reason**: `{finish_reason}`")

            if isinstance(tool_calls, list) and tool_calls:
                lines.append("- **tool_calls_requested**:")
                for c in tool_calls:
                    cc = _as_dict(c)
                    if not cc:
                        continue
                    nm = cc.get("name")
                    cid = cc.get("call_id") or ""
                    if isinstance(nm, str) and nm:
                        suffix = f" (call_id={cid})" if isinstance(cid, str) and cid else ""
                        lines.append(f"  - `{nm}`{suffix}")
            else:
                lines.append("- **tool_calls_requested**: none")

            if isinstance(content, str) and content.strip():
                lines.append("")
                lines.append("**Assistant content**")
                lines.append(_code_block(content, language="markdown"))
            lines.append("")
            continue

        if effect_type_s == "tool_calls":
            payload = _as_dict(effect.get("payload")) or {}
            calls = payload.get("tool_calls")
            calls_list: list[Any]
            if isinstance(calls, list):
                calls_list = calls
            elif calls is None:
                calls_list = []
            else:
                calls_list = [calls]

            results_by_id = _index_tool_results_by_call_id(result)
            if not calls_list:
                lines.append("- **tool_calls**: none")
                lines.append("")
                continue

            lines.append("")
            for call_any in calls_list:
                call = _as_dict(call_any)
                if not call:
                    continue
                call_id = call.get("call_id")
                call_id_s = call_id if isinstance(call_id, str) else ""
                r = results_by_id.get(call_id_s) if call_id_s else None
                lines.append(_render_tool_call(call, r, cfg=cfg))
                lines.append("")
            continue

        # Fallback: show a compact JSON of the result (still no truncation).
        if result is not None:
            lines.append("")
            lines.append("**Result (raw)**")
            lines.append(_code_block(result, language="json"))
            lines.append("")

    # Validate that report is JSON-safe when embedded (defensive, should always be true).
    try:
        json.dumps({"report": "\n".join(lines)})
    except Exception:
        pass

    return "\n".join(lines).rstrip() + "\n"


