"""Structured Active Memory (runtime-owned; JSON-safe).

This module implements a small, serializable "active memory" abstraction that
can be rendered into the next LLM prompt as *dynamically reconstructed context*.

Design goals:
- Store in `run.vars["_runtime"]["active_memory"]` so it is durable and snapshot-friendly.
- Keep canonical storage JSON-safe (dicts/lists/strings).
- Avoid mid-string truncation. If context must shrink, omit whole entries instead.

This is an early prototype for a future AbstractMemory "memory blocks" system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import uuid


def utc_now_iso_seconds() -> str:
    # Keep timestamps stable and readable (seconds precision).
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_runtime_ns(vars: Dict[str, Any]) -> Dict[str, Any]:
    runtime_ns = vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        runtime_ns = {}
        vars["_runtime"] = runtime_ns
    return runtime_ns


def get_active_memory(vars: Dict[str, Any]) -> Dict[str, Any]:
    runtime_ns = _ensure_runtime_ns(vars)
    mem = runtime_ns.get("active_memory")
    if not isinstance(mem, dict):
        mem = {}
        runtime_ns["active_memory"] = mem
    return mem


def ensure_active_memory(
    vars: Dict[str, Any],
    *,
    now_iso: Callable[[], str] = utc_now_iso_seconds,
) -> Dict[str, Any]:
    mem = get_active_memory(vars)

    mem.setdefault("version", 1)

    # Persistent blocks (Markdown)
    mem.setdefault("persona_md", _default_persona_md())
    mem.setdefault("memory_organization_md", _default_memory_organization_md())

    # Evolving blocks (canonical JSON; rendered as YAML-ish)
    if not isinstance(mem.get("tasks"), list):
        mem["tasks"] = []
    if not isinstance(mem.get("critical_insights"), list):
        mem["critical_insights"] = []
    if not isinstance(mem.get("key_history"), list):
        mem["key_history"] = []

    # Budget configuration (percentages of the *active prompt budget*)
    budgets = mem.get("budgets")
    if not isinstance(budgets, dict):
        budgets = {}
        mem["budgets"] = budgets
    budgets.setdefault("persona_pct", 0.10)
    budgets.setdefault("memory_organization_pct", 0.08)
    budgets.setdefault("current_tasks_pct", 0.35)
    budgets.setdefault("tools_pct", 0.10)
    budgets.setdefault("critical_insights_pct", 0.20)
    budgets.setdefault("key_history_pct", 0.17)

    mem.setdefault("updated_at", now_iso())
    return mem


def set_persona_md(vars: Dict[str, Any], persona_md: str, *, now_iso: Callable[[], str] = utc_now_iso_seconds) -> None:
    mem = ensure_active_memory(vars, now_iso=now_iso)
    mem["persona_md"] = str(persona_md or "").strip()
    mem["updated_at"] = now_iso()


def set_memory_organization_md(
    vars: Dict[str, Any], memory_organization_md: str, *, now_iso: Callable[[], str] = utc_now_iso_seconds
) -> None:
    mem = ensure_active_memory(vars, now_iso=now_iso)
    mem["memory_organization_md"] = str(memory_organization_md or "").strip()
    mem["updated_at"] = now_iso()


def upsert_task(
    vars: Dict[str, Any],
    *,
    task_id: Optional[str] = None,
    title: str,
    purpose: str = "",
    status: str = "todo",
    done: Optional[Sequence[str]] = None,
    next_steps: Optional[Sequence[str]] = None,
    issues: Optional[Sequence[str]] = None,
    now_iso: Callable[[], str] = utc_now_iso_seconds,
) -> str:
    mem = ensure_active_memory(vars, now_iso=now_iso)
    tasks = mem.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
        mem["tasks"] = tasks

    tid = str(task_id).strip() if isinstance(task_id, str) and task_id.strip() else f"t_{uuid.uuid4().hex[:10]}"
    ts = now_iso()

    normalized = {
        "task_id": tid,
        "title": str(title or "").strip(),
        "purpose": str(purpose or "").strip(),
        "status": str(status or "").strip() or "todo",
        "done": [str(x) for x in (done or []) if isinstance(x, str) and x.strip()],
        "next": [str(x) for x in (next_steps or []) if isinstance(x, str) and x.strip()],
        "issues": [str(x) for x in (issues or []) if isinstance(x, str) and x.strip()],
        "created_at": ts,
        "updated_at": ts,
    }

    for i, existing in enumerate(tasks):
        if not isinstance(existing, dict):
            continue
        if str(existing.get("task_id") or "") == tid:
            created_at = existing.get("created_at")
            if isinstance(created_at, str) and created_at.strip():
                normalized["created_at"] = created_at
            tasks[i] = normalized
            mem["updated_at"] = ts
            return tid

    tasks.append(normalized)
    # Keep at most 5 active tasks by default (caller may store more; prompt renderer will budget anyway).
    if len(tasks) > 5:
        # Prefer keeping non-done tasks; otherwise keep the most recent.
        non_done = [t for t in tasks if isinstance(t, dict) and str(t.get("status") or "").lower() != "done"]
        tasks[:] = (non_done or tasks)[-5:]

    mem["updated_at"] = ts
    return tid


def add_critical_insight(
    vars: Dict[str, Any],
    *,
    text: str,
    tags: Optional[Sequence[str]] = None,
    insight_id: Optional[str] = None,
    now_iso: Callable[[], str] = utc_now_iso_seconds,
) -> str:
    mem = ensure_active_memory(vars, now_iso=now_iso)
    insights = mem.get("critical_insights")
    if not isinstance(insights, list):
        insights = []
        mem["critical_insights"] = insights

    iid = (
        str(insight_id).strip()
        if isinstance(insight_id, str) and insight_id.strip()
        else f"i_{uuid.uuid4().hex[:10]}"
    )
    ts = now_iso()
    insights.append(
        {
            "insight_id": iid,
            "ts": ts,
            "text": str(text or "").strip(),
            "tags": [str(t) for t in (tags or []) if isinstance(t, str) and t.strip()],
        }
    )
    mem["updated_at"] = ts
    return iid


def add_key_history_event(
    vars: Dict[str, Any],
    *,
    kind: str,
    summary: str,
    refs: Optional[Sequence[Dict[str, Any]]] = None,
    event_id: Optional[str] = None,
    ts: Optional[str] = None,
    now_iso: Callable[[], str] = utc_now_iso_seconds,
) -> str:
    mem = ensure_active_memory(vars, now_iso=now_iso)
    history = mem.get("key_history")
    if not isinstance(history, list):
        history = []
        mem["key_history"] = history

    hid = (
        str(event_id).strip()
        if isinstance(event_id, str) and event_id.strip()
        else f"h_{uuid.uuid4().hex[:10]}"
    )
    when = str(ts).strip() if isinstance(ts, str) and ts.strip() else now_iso()
    history.append(
        {
            "event_id": hid,
            "ts": when,
            "kind": str(kind or "").strip(),
            "summary": str(summary or "").strip(),
            "refs": [dict(r) for r in (refs or []) if isinstance(r, dict)],
        }
    )
    mem["updated_at"] = now_iso()
    return hid


def render_active_memory_for_prompt(
    vars: Dict[str, Any],
    *,
    include_tools_summary: bool = True,
    max_chars: Optional[int] = None,
) -> str:
    """Render active memory into a prompt-friendly block.

    Important: This renderer never truncates entries mid-string. If content must
    shrink due to `max_chars`, it drops whole list entries (oldest first).
    """
    mem = ensure_active_memory(vars)
    budgets = mem.get("budgets") if isinstance(mem.get("budgets"), dict) else {}

    def pct(key: str, default: float) -> float:
        try:
            val = float(budgets.get(key, default))
            return max(0.0, min(1.0, val))
        except Exception:
            return default

    # If no max_chars is provided, render fully (callers may enforce limits upstream).
    total = int(max_chars) if isinstance(max_chars, int) and max_chars > 0 else None

    blocks: List[Tuple[str, str]] = []

    persona_md = str(mem.get("persona_md") or "").strip()
    if persona_md:
        blocks.append(("Persona (persistent)", persona_md))

    org_md = str(mem.get("memory_organization_md") or "").strip()
    if org_md:
        blocks.append(("Memory Organization (persistent)", org_md))

    tools_summary = ""
    if include_tools_summary:
        tools_summary = _render_tools_summary(vars)
        if tools_summary:
            blocks.append(("Tools (session)", tools_summary))

    tasks_yaml = _render_tasks_yaml(mem.get("tasks"))
    if tasks_yaml:
        blocks.append(("Current Tasks (evolving)", tasks_yaml))

    insights_yaml = _render_insights_yaml(mem.get("critical_insights"))
    if insights_yaml:
        blocks.append(("Critical Insights (evolving)", insights_yaml))

    history_yaml = _render_history_yaml(mem.get("key_history"))
    if history_yaml:
        blocks.append(("Key History (append-only)", history_yaml))

    rendered_blocks: List[str] = []

    # If we must fit into a max char budget, greedily drop old list items from
    # the YAML sections (tasks/insights/history) while keeping persona/org/tools intact.
    if total is not None:
        # Always keep the first N "persistent" blocks as-is.
        persistent_titles = {"Persona (persistent)", "Memory Organization (persistent)", "Tools (session)"}
        persistent_parts = [b for b in blocks if b[0] in persistent_titles]
        mutable_parts = [b for b in blocks if b[0] not in persistent_titles]

        base_text = _join_blocks(persistent_parts)
        remaining = max(0, total - len(base_text))

        # Budgets for mutable parts (rough; based on configured percentages).
        budgets_map = {
            "Current Tasks (evolving)": pct("current_tasks_pct", 0.35),
            "Critical Insights (evolving)": pct("critical_insights_pct", 0.20),
            "Key History (append-only)": pct("key_history_pct", 0.17),
        }
        # Normalize within the remaining window.
        total_pct = sum(budgets_map.get(title, 0.0) for title, _ in mutable_parts) or 1.0

        fitted: List[Tuple[str, str]] = []
        for title, content in mutable_parts:
            part_budget = int(remaining * (budgets_map.get(title, 0.0) / total_pct))
            fitted.append((title, _fit_yaml_section(title, content, max_chars=part_budget)))

        rendered_blocks = []
        if base_text.strip():
            rendered_blocks.append(base_text.strip())
        fitted_text = _join_blocks(fitted).strip()
        if fitted_text:
            rendered_blocks.append(fitted_text)
        return "\n\n".join([b for b in rendered_blocks if b]).strip()

    # No total budget: render as-is.
    return _join_blocks(blocks).strip()


def _default_persona_md() -> str:
    return (
        "You are an autonomous coding agent inside AbstractFramework.\n"
        "- You take action by calling tools (files/commands/web) when needed.\n"
        "- You verify changes with targeted checks/tests when possible.\n"
        "- You are truthful: only claim actions supported by tool outputs.\n"
    ).strip()


def _default_memory_organization_md() -> str:
    return (
        "Active context is reconstructed from these memory modules:\n"
        "1) Persona (persistent)\n"
        "2) Memory Organization (persistent)\n"
        "3) Current Tasks (evolving; up to 5)\n"
        "4) Tools (session-wide; allowlist + specs)\n"
        "5) Critical Insights (evolving)\n"
        "6) Key History (append-only)\n\n"
        "To inspect durable runtime state during a run:\n"
        "- Use `inspect_vars` to view `scratchpad`, `_runtime.node_traces`, and `_runtime.active_memory`.\n"
        "- Use `recall_memory` to rehydrate archived spans (when available).\n"
    ).strip()


def _render_tools_summary(vars: Dict[str, Any]) -> str:
    runtime_ns = vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        return ""
    allowed = runtime_ns.get("allowed_tools")
    toolset_id = runtime_ns.get("toolset_id")

    enabled: List[str] = []
    if isinstance(allowed, list):
        enabled = [str(t) for t in allowed if isinstance(t, str) and t.strip()]
    enabled.sort()

    parts: List[str] = []
    if isinstance(toolset_id, str) and toolset_id.strip():
        parts.append(f"toolset_id: {toolset_id.strip()}")
    if enabled:
        parts.append("enabled_tools:")
        for t in enabled:
            parts.append(f"  - {t}")
    return "\n".join(parts).strip()


def _join_blocks(blocks: Sequence[Tuple[str, str]]) -> str:
    out: List[str] = []
    for title, content in blocks:
        title = str(title or "").strip()
        content = str(content or "").rstrip()
        if not title or not content:
            continue
        out.append(f"## {title}\n{content}".rstrip())
    return "\n\n".join(out).strip()


def _yaml_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep it simple: always quote when it contains special chars.
    if any(c in text for c in (":", "#", "\n", '"', "'")) or text.strip() != text or not text:
        escaped = text.replace('"', '\\"')
        return f"\"{escaped}\""
    return text


def _render_list_yaml(name: str, items: Sequence[Any]) -> str:
    out: List[str] = [f"{name}:"]
    for x in items:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        out.append(f"  - {_yaml_escape(s)}")
    return "\n".join(out).strip()


def _render_tasks_yaml(tasks: Any) -> str:
    if not isinstance(tasks, list) or not tasks:
        return ""
    # Render newest first for quick relevance.
    items = [t for t in tasks if isinstance(t, dict)]
    items.sort(key=lambda d: str(d.get("updated_at") or d.get("created_at") or ""), reverse=True)
    out: List[str] = ["current_tasks:"]
    for t in items:
        out.append("  - task_id: " + _yaml_escape(t.get("task_id")))
        out.append("    status: " + _yaml_escape(t.get("status")))
        out.append("    title: " + _yaml_escape(t.get("title")))
        if str(t.get("purpose") or "").strip():
            out.append("    purpose: " + _yaml_escape(t.get("purpose")))
        if isinstance(t.get("created_at"), str) and t.get("created_at"):
            out.append("    created_at: " + _yaml_escape(t.get("created_at")))
        if isinstance(t.get("updated_at"), str) and t.get("updated_at"):
            out.append("    updated_at: " + _yaml_escape(t.get("updated_at")))
        done = t.get("done")
        if isinstance(done, list) and done:
            out.append("    done:")
            for d in done:
                out.append("      - " + _yaml_escape(d))
        nxt = t.get("next")
        if isinstance(nxt, list) and nxt:
            out.append("    next:")
            for n in nxt:
                out.append("      - " + _yaml_escape(n))
        issues = t.get("issues")
        if isinstance(issues, list) and issues:
            out.append("    issues:")
            for issue in issues:
                out.append("      - " + _yaml_escape(issue))
    return "\n".join(out).strip()


def _render_insights_yaml(insights: Any) -> str:
    if not isinstance(insights, list) or not insights:
        return ""
    items = [i for i in insights if isinstance(i, dict)]
    items.sort(key=lambda d: str(d.get("ts") or ""), reverse=True)
    out: List[str] = ["critical_insights:"]
    for it in items:
        out.append("  - insight_id: " + _yaml_escape(it.get("insight_id")))
        if isinstance(it.get("ts"), str) and it.get("ts"):
            out.append("    ts: " + _yaml_escape(it.get("ts")))
        out.append("    text: " + _yaml_escape(it.get("text")))
        tags = it.get("tags")
        if isinstance(tags, list) and tags:
            out.append("    tags:")
            for t in tags:
                out.append("      - " + _yaml_escape(t))
    return "\n".join(out).strip()


def _render_history_yaml(history: Any) -> str:
    if not isinstance(history, list) or not history:
        return ""
    items = [h for h in history if isinstance(h, dict)]
    items.sort(key=lambda d: str(d.get("ts") or ""), reverse=True)
    out: List[str] = ["key_history:"]
    for e in items:
        out.append("  - event_id: " + _yaml_escape(e.get("event_id")))
        if isinstance(e.get("ts"), str) and e.get("ts"):
            out.append("    ts: " + _yaml_escape(e.get("ts")))
        if str(e.get("kind") or "").strip():
            out.append("    kind: " + _yaml_escape(e.get("kind")))
        out.append("    summary: " + _yaml_escape(e.get("summary")))
        refs = e.get("refs")
        if isinstance(refs, list) and refs:
            out.append("    refs:")
            for r in refs:
                if not isinstance(r, dict) or not r:
                    continue
                out.append("      - " + _yaml_escape(r))
    return "\n".join(out).strip()


def _fit_yaml_section(title: str, yaml_text: str, *, max_chars: int) -> str:
    """Fit a rendered YAML-ish section into a char budget by dropping entries."""
    if max_chars <= 0:
        return ""
    text = str(yaml_text or "").strip()
    if len(text) <= max_chars:
        return text

    # Split on top-level list entries: lines starting with "  - ".
    lines = text.splitlines()
    if not lines:
        return ""

    header = lines[0]
    entries: List[List[str]] = []
    current: List[str] = []
    for line in lines[1:]:
        if line.startswith("  - "):
            if current:
                entries.append(current)
            current = [line]
        else:
            if current:
                current.append(line)
    if current:
        entries.append(current)

    # Keep newest entries (they were rendered newest-first).
    kept: List[List[str]] = []
    size = len(header) + 1
    for entry in entries:
        entry_text = "\n".join(entry)
        if size + 1 + len(entry_text) > max_chars:
            continue
        kept.append(entry)
        size += 1 + len(entry_text)
        if size >= max_chars:
            break

    if not kept:
        # If no entry fits, keep only the header (still valid YAML-ish).
        return header

    out_lines: List[str] = [header]
    for entry in kept:
        out_lines.extend(entry)
    return "\n".join(out_lines).strip()

