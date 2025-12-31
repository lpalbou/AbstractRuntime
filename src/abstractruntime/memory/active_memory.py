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

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
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

    mem.setdefault("version", 3)
    mem.setdefault("max_chars", 8000)
    mem.setdefault("max_tokens", 2000)

    # Persistent blocks (Markdown)
    mem.setdefault("persona_md", _default_persona_md())
    mem.setdefault("memory_organization_md", _default_memory_organization_md())

    # Evolving blocks (canonical JSON; rendered as YAML-ish)
    if not isinstance(mem.get("tasks"), list):
        mem["tasks"] = []
    if not isinstance(mem.get("current_context"), list):
        mem["current_context"] = []
    if not isinstance(mem.get("critical_insights"), list):
        mem["critical_insights"] = []
    if not isinstance(mem.get("key_history"), list):
        mem["key_history"] = []

    # Budget configuration (percentages of the *active prompt budget*)
    budgets = mem.get("budgets")
    if not isinstance(budgets, dict):
        budgets = {}
        mem["budgets"] = budgets
    # Defaults are chosen to:
    # - preserve stable identity (persona/org/tools),
    # - prioritize current work (tasks/context/insights),
    # - keep history bounded (history should not dominate; full fidelity lives in spans/artifacts).
    #
    # Sum ~= 1.0 (if users override and sum > 1.0, renderers normalize down).
    budgets.setdefault("persona_pct", 0.10)
    budgets.setdefault("memory_organization_pct", 0.07)
    budgets.setdefault("tools_pct", 0.10)
    budgets.setdefault("current_tasks_pct", 0.30)
    budgets.setdefault("current_context_pct", 0.08)
    budgets.setdefault("critical_insights_pct", 0.20)
    budgets.setdefault("key_history_pct", 0.15)

    # Conventional aliasing: treat some modules as "system prompt memory" and the rest
    # as "user prompt memory" for backward compatibility with UIs and code paths.
    #
    # Defaults:
    # - system: identity + coordination + tool affordances
    # - user: current work + insights + timeline
    #
    # If you want to move `key_history` into the system prompt, add it here.
    mem.setdefault("system_component_ids", ["persona", "memory_organization", "tools"])

    # Important: avoid calling `now_iso()` eagerly on every ensure.
    # `dict.setdefault("updated_at", now_iso())` would evaluate `now_iso()` even when the key exists.
    updated_at = mem.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.strip():
        mem["updated_at"] = now_iso()
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


def upsert_current_context_item(
    vars: Dict[str, Any],
    *,
    context_id: Optional[str] = None,
    title: str,
    summary: str = "",
    kind: str = "context",
    refs: Optional[Sequence[Dict[str, Any]]] = None,
    now_iso: Callable[[], str] = utc_now_iso_seconds,
) -> str:
    """Upsert a dynamic "current context" item.

    Current context is intentionally task-dependent and may be replaced frequently.
    Each item should remain small and reference deeper sources via `refs`.
    """
    mem = ensure_active_memory(vars, now_iso=now_iso)
    items = mem.get("current_context")
    if not isinstance(items, list):
        items = []
        mem["current_context"] = items

    cid = (
        str(context_id).strip()
        if isinstance(context_id, str) and context_id.strip()
        else f"c_{uuid.uuid4().hex[:10]}"
    )
    ts = now_iso()

    normalized = {
        "context_id": cid,
        "kind": str(kind or "").strip() or "context",
        "title": str(title or "").strip(),
        "summary": str(summary or "").strip(),
        "refs": [dict(r) for r in (refs or []) if isinstance(r, dict)],
        "created_at": ts,
        "updated_at": ts,
    }

    for i, existing in enumerate(items):
        if not isinstance(existing, dict):
            continue
        if str(existing.get("context_id") or "") == cid:
            created_at = existing.get("created_at")
            if isinstance(created_at, str) and created_at.strip():
                normalized["created_at"] = created_at
            items[i] = normalized
            mem["updated_at"] = ts
            return cid

    items.append(normalized)
    # Keep the list bounded (rendering will budget too, but this prevents unbounded growth in storage).
    if len(items) > 20:
        items[:] = [x for x in items if isinstance(x, dict)][-20:]

    mem["updated_at"] = ts
    return cid


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


_ACTIVE_MEMORY_COMPONENT_SPECS: List[Dict[str, str]] = [
    {"id": "persona", "title": "Persona (persistent)", "budget_key": "persona_pct", "kind": "markdown"},
    {
        "id": "memory_organization",
        "title": "Memory Organization (persistent)",
        "budget_key": "memory_organization_pct",
        "kind": "markdown",
    },
    {"id": "tools", "title": "Tools (session)", "budget_key": "tools_pct", "kind": "yaml"},
    {"id": "current_tasks", "title": "Current Tasks (evolving)", "budget_key": "current_tasks_pct", "kind": "yaml_list"},
    {
        "id": "current_context",
        "title": "Current Context (dynamic)",
        "budget_key": "current_context_pct",
        "kind": "yaml_list",
    },
    {
        "id": "critical_insights",
        "title": "Critical Insights (evolving)",
        "budget_key": "critical_insights_pct",
        "kind": "yaml_list",
    },
    {"id": "key_history", "title": "Key History (append-only)", "budget_key": "key_history_pct", "kind": "yaml_list"},
]


def _estimate_tokens_fast(text: str) -> int:
    """Fallback token estimate (≈4 chars/token)."""
    s = str(text or "")
    if not s:
        return 0
    return max(1, len(s) // 4)


def render_active_memory_blocks_for_prompt(
    vars: Dict[str, Any],
    *,
    include_tools_summary: bool = True,
    max_chars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Render Active Memory as fitted per-component blocks.

    Returns a list of dicts (ordered) with:
      - component_id
      - title
      - content (fitted; no mid-entry truncation)

    This is useful for UIs that want per-component accounting.
    """
    mem = ensure_active_memory(vars)
    budgets = mem.get("budgets") if isinstance(mem.get("budgets"), dict) else {}

    if max_chars is None:
        stored = mem.get("max_chars")
        if isinstance(stored, int) and stored > 0:
            max_chars = stored

    def pct(key: str) -> float:
        try:
            val = float(budgets.get(key, 0.0))
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0

    # Build raw content for each module (in the requested order).
    raw_by_id: Dict[str, str] = {
        "persona": str(mem.get("persona_md") or "").strip(),
        "memory_organization": str(mem.get("memory_organization_md") or "").strip(),
        "tools": _render_tools_yaml(vars).strip() if include_tools_summary else "",
        "current_tasks": _render_tasks_yaml(mem.get("tasks")).strip(),
        "current_context": _render_current_context_yaml(mem.get("current_context")).strip(),
        "critical_insights": _render_insights_yaml(mem.get("critical_insights")).strip(),
        "key_history": _render_history_yaml(mem.get("key_history")).strip(),
    }

    total = int(max_chars) if isinstance(max_chars, int) and max_chars > 0 else None

    specs: List[Dict[str, Any]] = []
    for spec in _ACTIVE_MEMORY_COMPONENT_SPECS:
        cid = spec["id"]
        content = raw_by_id.get(cid, "")
        if not isinstance(content, str) or not content.strip():
            continue
        specs.append(
            {
                "component_id": cid,
                "title": spec["title"],
                "budget_key": spec["budget_key"],
                "kind": spec["kind"],
                "content": content.strip(),
            }
        )

    if total is None:
        return [{"component_id": s["component_id"], "title": s["title"], "content": s["content"]} for s in specs]

    pcts = {s["component_id"]: pct(str(s.get("budget_key") or "")) for s in specs}
    total_pct = sum(pcts.values()) or 1.0
    scale = 1.0 / total_pct if total_pct > 1.0 else 1.0

    fitted: List[Dict[str, Any]] = []
    for s in specs:
        cid = str(s.get("component_id") or "")
        budget = int(total * (pcts.get(cid, 0.0) * scale))
        if budget <= 0:
            continue

        kind = str(s.get("kind") or "")
        content = str(s.get("content") or "")
        if kind == "markdown":
            out = _fit_markdown_section(content, max_chars=budget)
        elif kind == "yaml_list":
            out = _fit_yaml_section(str(s.get("title") or ""), content, max_chars=budget)
        else:
            out = _fit_tools_yaml_section(content, max_chars=budget)

        out = str(out or "").strip()
        if out:
            fitted.append({"component_id": cid, "title": s["title"], "content": out})

    # Best-effort final fit: drop least-critical blocks until within total.
    rendered = _join_blocks([(b["title"], b["content"]) for b in fitted]).strip()
    if len(rendered) <= total:
        return fitted

    drop_order = [
        "key_history",
        "critical_insights",
        "current_context",
        "current_tasks",
        "tools",
    ]
    remaining = list(fitted)
    for drop_id in drop_order:
        if len(rendered) <= total:
            break
        remaining = [b for b in remaining if b.get("component_id") != drop_id]
        rendered = _join_blocks([(b["title"], b["content"]) for b in remaining]).strip()
    if len(rendered) <= total:
        return remaining

    # At this point, keep only markdown identity blocks and fit them at line boundaries.
    keep_ids = {"persona", "memory_organization"}
    markdown_only = [b for b in remaining if b.get("component_id") in keep_ids]
    out_text = _fit_markdown_section(_join_blocks([(b["title"], b["content"]) for b in markdown_only]).strip(), max_chars=total)
    # Preserve per-component separation as best-effort: keep persona, then org if space remains.
    persona = next((b for b in markdown_only if b.get("component_id") == "persona"), None)
    org = next((b for b in markdown_only if b.get("component_id") == "memory_organization"), None)
    out: List[Dict[str, Any]] = []
    if isinstance(persona, dict) and persona.get("content"):
        out.append({"component_id": "persona", "title": persona["title"], "content": str(persona["content"])})
    if isinstance(org, dict) and org.get("content"):
        out.append({"component_id": "memory_organization", "title": org["title"], "content": str(org["content"])})
    # If our reconstructed blocks still exceed, fall back to a single combined block.
    if len(_join_blocks([(b["title"], b["content"]) for b in out]).strip()) > total:
        return [{"component_id": "persona", "title": "Persona (persistent)", "content": out_text}]
    return out


def render_active_memory_split_for_llm_request(
    vars: Dict[str, Any],
    *,
    include_tools_summary: bool = True,
    max_chars: Optional[int] = None,
) -> Dict[str, str]:
    """Render Active Memory split into (system_memory, user_memory).

    The split is a conventional alias:
    - system_memory: persona + memory_organization + tools (default)
    - user_memory: current_tasks + current_context + critical_insights + key_history

    The split uses a single fitted render pass so the combined result stays within `max_chars`.
    """
    mem = ensure_active_memory(vars)
    blocks = render_active_memory_blocks_for_prompt(vars, include_tools_summary=include_tools_summary, max_chars=max_chars)

    raw_ids = mem.get("system_component_ids")
    ids_list = (
        [str(x).strip() for x in raw_ids if isinstance(x, str) and str(x).strip()] if isinstance(raw_ids, list) else []
    )
    system_ids = set(ids_list or ["persona", "memory_organization", "tools"])

    system_blocks = [b for b in blocks if str(b.get("component_id") or "") in system_ids]
    user_blocks = [b for b in blocks if str(b.get("component_id") or "") and str(b.get("component_id") or "") not in system_ids]

    def _join(selected: List[Dict[str, Any]]) -> str:
        return _join_blocks([(str(b.get("title") or ""), str(b.get("content") or "")) for b in selected]).strip()

    return {"system_memory": _join(system_blocks), "user_memory": _join(user_blocks)}


def render_active_memory_for_system_prompt(
    vars: Dict[str, Any],
    *,
    include_tools_summary: bool = True,
    max_chars: Optional[int] = None,
) -> str:
    split = render_active_memory_split_for_llm_request(vars, include_tools_summary=include_tools_summary, max_chars=max_chars)
    return str(split.get("system_memory") or "").strip()


def render_active_memory_for_user_prompt(
    vars: Dict[str, Any],
    *,
    include_tools_summary: bool = True,
    max_chars: Optional[int] = None,
) -> str:
    split = render_active_memory_split_for_llm_request(vars, include_tools_summary=include_tools_summary, max_chars=max_chars)
    return str(split.get("user_memory") or "").strip()


def compute_active_memory_token_breakdown(
    vars: Dict[str, Any],
    *,
    token_counter: Optional[Callable[[str], int]] = None,
    include_tools_summary: bool = True,
) -> Dict[str, Any]:
    """Compute per-component token usage and budgets for Active Memory.

    Notes:
    - Uses `vars["_runtime"]["active_memory"].max_tokens` as the total Active Memory budget (default 2000).
    - If budgets sum > 1.0, the budget weights are normalized down.
    - Token counting is injected to avoid AbstractRuntime depending on AbstractCore.
    """
    mem = ensure_active_memory(vars)
    budgets = mem.get("budgets") if isinstance(mem.get("budgets"), dict) else {}

    total_budget = mem.get("max_tokens")
    if not isinstance(total_budget, int) or total_budget <= 0:
        max_chars = mem.get("max_chars")
        if isinstance(max_chars, int) and max_chars > 0:
            total_budget = max(1, max_chars // 4)
        else:
            total_budget = 2000

    def pct(key: str) -> float:
        try:
            val = float(budgets.get(key, 0.0))
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0

    weights: Dict[str, float] = {}
    for spec in _ACTIVE_MEMORY_COMPONENT_SPECS:
        cid = spec["id"]
        weights[cid] = pct(spec["budget_key"])

    sum_w = sum(weights.values()) or 1.0
    if sum_w > 1.0:
        weights = {k: (v / sum_w) for k, v in weights.items()}

    # Allocate integer budgets with stable rounding.
    floors: Dict[str, int] = {}
    fracs: List[Tuple[float, str]] = []
    allocated = 0
    for cid, w in weights.items():
        exact = float(total_budget) * float(w)
        base = int(exact)
        floors[cid] = base
        allocated += base
        fracs.append((exact - base, cid))

    remainder = max(0, int(total_budget) - allocated)
    fracs.sort(key=lambda x: x[0], reverse=True)
    for _, cid in fracs:
        if remainder <= 0:
            break
        floors[cid] += 1
        remainder -= 1

    count = token_counter or _estimate_tokens_fast

    blocks = render_active_memory_blocks_for_prompt(vars, include_tools_summary=include_tools_summary)
    used_by_id: Dict[str, int] = {spec["id"]: 0 for spec in _ACTIVE_MEMORY_COMPONENT_SPECS}
    for b in blocks:
        cid = str(b.get("component_id") or "")
        title = str(b.get("title") or "").strip()
        content = str(b.get("content") or "").strip()
        if not cid or not title or not content:
            continue
        section_text = f"## {title}\n{content}".strip()
        used_by_id[cid] = int(count(section_text) or 0)

    components: Dict[str, Dict[str, Any]] = {}
    for spec in _ACTIVE_MEMORY_COMPONENT_SPECS:
        cid = spec["id"]
        used = int(used_by_id.get(cid) or 0)
        cap = int(floors.get(cid) or 0)
        pct_used = float(used) / float(cap) if cap > 0 else (1.0 if used > 0 else 0.0)
        components[cid] = {
            "title": spec["title"],
            "used_tokens": used,
            "max_tokens": cap,
            "pct_used": pct_used,
        }

    return {
        "active_memory_max_tokens": int(total_budget),
        "components": components,
    }


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
    blocks = render_active_memory_blocks_for_prompt(vars, include_tools_summary=include_tools_summary, max_chars=max_chars)
    return _join_blocks([(b["title"], b["content"]) for b in blocks]).strip()


def _default_persona_md() -> str:
    return (
        "You are an autonomous coding agent inside AbstractFramework.\n"
        "- You take action by calling tools (files/commands/web) when needed.\n"
        "- You verify changes with targeted checks/tests when possible.\n"
        "- You are truthful: only claim actions supported by tool outputs.\n"
    ).strip()


def _default_memory_organization_md() -> str:
    return (
        "You must use Active Memory as your coordination layer.\n"
        "Read modules in this order (most important first):\n"
        "1) Persona (who you are; non-negotiable rules)\n"
        "2) Memory Organization (how to use memory)\n"
        "3) Tools (what actions are possible; ONLY call tools listed there)\n"
        "4) Current Tasks (what we are doing now; keep ≤5, actionable)\n"
        "5) Current Context (task-specific working set; references over payloads)\n"
        "6) Critical Insights (pitfalls/strategies to apply before acting)\n"
        "7) Key History (append-only timeline; use only when you need provenance/avoid repeats)\n\n"
        "System/user alias (backward compatibility):\n"
        "- System prompt memory ~= Persona + Memory Organization + Tools.\n"
        "- User prompt memory ~= Current Tasks + Current Context + Critical Insights + Key History.\n\n"
        "How to use each module:\n"
        "- Persona: keep identity + methodology stable. If a conflict appears, stop and resolve it.\n"
        "- Tools: choose the smallest allowed tool that advances the task.\n"
        "  - Never claim effects without tool outputs.\n"
        "  - If a tool you want is not listed, ask the host/user to enable it (e.g., AbstractCode `/tools`).\n"
        "- Current Tasks: keep the top items (≤5). Update status as you make progress.\n"
        "- Current Context: a *dynamic working set* rebuilt per task. Prefer short digests + refs (file paths, span_id, artifact_id).\n"
        "- Critical Insights: check this before repeating a failed approach or changing architecture.\n"
        "- Key History: consult only when needed (e.g., “did we already try X?”).\n\n"
        "Selective retrieval (when you are missing key info):\n"
        "1) Repo/source-of-truth → `search_files` then `read_file`.\n"
        "2) Durable run state → `inspect_vars` (start with keys_only=true), especially:\n"
        "   - `scratchpad` (agent loop state)\n"
        "   - `_runtime.node_traces` (tool calls + results)\n"
        "   - `_runtime.memory_spans` (archived span index)\n"
        "   - `_runtime.active_memory` (this structure)\n"
        "3) Compacted/archived conversation → `recall_memory(span_id=..., query=..., tags=..., since/until=...)`.\n"
        "4) If context is too large → `compact_memory` (keep provenance via span_id).\n"
        "5) Persist important facts/decisions → `remember` (with tags + sources).\n\n"
        "Current Context update rule (keep it small):\n"
        "- After any retrieval that produces useful signal, write a 1–3 line digest in your next message and include refs.\n"
        "- Avoid pasting long outputs; prefer ref handles (span_id/artifact_id/file path) so you can re-load on demand.\n\n"
        "Examples:\n"
        "- Need to adjust code behavior:\n"
        "  - Call `search_files(pattern=\"...\")` to find the right file/lines.\n"
        "  - Call `read_file(file_path=\"...\", start_line_one_indexed=..., end_line_one_indexed_inclusive=...)`.\n"
        "- Need exact past decision after compaction:\n"
        "  - Call `recall_memory(span_id=\"...\")`.\n"
        "  - Then proceed using: “Decision: ... (source span_id=...)”.\n"
        "- Need runtime trace of last tool execution:\n"
        "  - Call `inspect_vars(path=\"/_runtime/node_traces\", keys_only=true)` then drill down.\n"
        "- Need grounding in codebase:\n"
        "  - Call `search_files(pattern=\"ActiveContextPolicy\")` then `read_file(file_path=\"...\")`.\n"
    ).strip()


def _render_tools_yaml(vars: Dict[str, Any]) -> str:
    runtime_ns = vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        return ""
    toolset_id = str(runtime_ns.get("toolset_id") or "").strip()

    specs = runtime_ns.get("tool_specs")
    specs_list: List[Dict[str, Any]] = [dict(s) for s in specs if isinstance(s, dict)] if isinstance(specs, list) else []
    if not specs_list:
        allowed = runtime_ns.get("allowed_tools")
        enabled: List[str] = []
        if isinstance(allowed, list):
            enabled = [str(t) for t in allowed if isinstance(t, str) and t.strip()]
        enabled.sort()
        if not enabled and not toolset_id:
            return ""
        parts: List[str] = []
        if toolset_id:
            parts.append(f"toolset_id: {_yaml_escape(toolset_id)}")
        if enabled:
            parts.append("tools:")
            for t in enabled:
                parts.append(f"  - name: {_yaml_escape(t)}")
        return "\n".join(parts).strip()

    specs_list.sort(key=lambda s: str(s.get("name") or ""))

    parts: List[str] = []
    if toolset_id:
        parts.append(f"toolset_id: {_yaml_escape(toolset_id)}")
    parts.append("tools:")

    for spec in specs_list:
        name = str(spec.get("name") or "").strip()
        if not name:
            continue

        desc = str(spec.get("when_to_use") or spec.get("description") or "").strip()
        params = spec.get("parameters")
        params_dict = dict(params) if isinstance(params, dict) else {}

        required: List[str] = []
        for pname, pmeta in params_dict.items():
            if not isinstance(pname, str) or not pname.strip():
                continue
            if not isinstance(pmeta, dict):
                required.append(pname)
                continue
            # Convention: missing "default" means required (ToolDefinition.to_dict pattern in this repo).
            if "default" not in pmeta:
                required.append(pname)
        required.sort()

        example_call = _example_call_from_tool_spec(name=name, spec=spec, required=required)

        parts.append(f"  - name: {_yaml_escape(name)}")
        if desc:
            parts.append(f"    description: {_yaml_escape(desc)}")
        if required:
            parts.append("    required_args:")
            for r in required:
                parts.append(f"      - {_yaml_escape(r)}")
        if example_call:
            parts.append(f"    example: {_yaml_escape(example_call)}")

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


def _fit_markdown_section(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw
    import re
    import textwrap

    lines_in = raw.splitlines()

    # Word-wrap long lines (especially the first line) so we can keep content without
    # truncating mid-string. Preserve basic bullet/number list indentation.
    wrapped_lines: List[str] = []
    in_fence = False
    for line in lines_in:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            wrapped_lines.append(line)
            continue
        if in_fence or len(line) <= max_chars:
            wrapped_lines.append(line)
            continue

        indent_len = len(line) - len(line.lstrip(" "))
        indent = line[:indent_len]
        rest = line[indent_len:]

        m = re.match(r"([-*+] |\d+\. )", rest)
        if m:
            bullet = m.group(1)
            body = rest[len(bullet) :]
            initial = indent + bullet
            subsequent = indent + (" " * len(bullet))
            wrapped_lines.extend(
                textwrap.wrap(
                    body,
                    width=max_chars,
                    initial_indent=initial,
                    subsequent_indent=subsequent,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        else:
            wrapped_lines.extend(
                textwrap.wrap(
                    rest,
                    width=max_chars,
                    initial_indent=indent,
                    subsequent_indent=indent,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

    kept: List[str] = []
    size = 0
    for line in wrapped_lines:
        add = len(line) + (1 if kept else 0)
        if size + add > max_chars:
            break
        kept.append(line)
        size += add
    return "\n".join(kept).strip()


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


def _render_current_context_yaml(context_items: Any) -> str:
    if not isinstance(context_items, list) or not context_items:
        return ""
    items = [c for c in context_items if isinstance(c, dict)]
    items.sort(key=lambda d: str(d.get("updated_at") or d.get("created_at") or ""), reverse=True)
    out: List[str] = ["current_context:"]
    for c in items:
        out.append("  - context_id: " + _yaml_escape(c.get("context_id")))
        if str(c.get("kind") or "").strip():
            out.append("    kind: " + _yaml_escape(c.get("kind")))
        out.append("    title: " + _yaml_escape(c.get("title")))
        if str(c.get("summary") or "").strip():
            out.append("    summary: " + _yaml_escape(c.get("summary")))
        if isinstance(c.get("created_at"), str) and c.get("created_at"):
            out.append("    created_at: " + _yaml_escape(c.get("created_at")))
        if isinstance(c.get("updated_at"), str) and c.get("updated_at"):
            out.append("    updated_at: " + _yaml_escape(c.get("updated_at")))

        refs = c.get("refs")
        if isinstance(refs, list) and refs:
            out.append("    refs:")
            for r in refs:
                if not isinstance(r, dict) or not r:
                    continue
                _append_yaml_dict_list_item(out, r, base_indent="      ")
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
                _append_yaml_dict_list_item(out, r, base_indent="      ")
    return "\n".join(out).strip()


def _append_yaml_dict_list_item(lines: List[str], item: Dict[str, Any], *, base_indent: str) -> None:
    """Append a YAML list item for a small dict (single-line when possible)."""
    if not item:
        return
    indent = str(base_indent or "")
    keys = [k for k in item.keys() if isinstance(k, str) and k.strip()]
    keys.sort()
    if not keys:
        return
    first = keys[0]
    first_val = item.get(first)
    lines.append(f"{indent}- {first}: {_yaml_escape(first_val)}")
    for k in keys[1:]:
        lines.append(f"{indent}  {k}: {_yaml_escape(item.get(k))}")


def _example_call_from_tool_spec(*, name: str, spec: Dict[str, Any], required: Sequence[str]) -> str:
    """Render a compact example call string for a tool."""
    examples = spec.get("examples")
    if isinstance(examples, list):
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            args = ex.get("arguments")
            if isinstance(args, dict) and args:
                return _format_tool_call(name=name, arguments=args)

    # Fallback: minimal placeholder args for required params.
    args: Dict[str, Any] = {}
    for key in required:
        if not isinstance(key, str) or not key.strip():
            continue
        args[key] = "..."
    return _format_tool_call(name=name, arguments=args)


def _format_tool_call(*, name: str, arguments: Dict[str, Any]) -> str:
    if not arguments:
        return f"{name}()"

    parts: List[str] = []
    for k, v in arguments.items():
        if not isinstance(k, str) or not k.strip():
            continue
        key = k.strip()
        if isinstance(v, str):
            val = '"' + v.replace('"', '\\"') + '"'
        elif isinstance(v, bool):
            val = "true" if v else "false"
        elif v is None:
            val = "null"
        else:
            val = str(v)
        parts.append(f"{key}={val}")
    return f"{name}({', '.join(parts)})"


def _fit_tools_yaml_section(text: str, *, max_chars: int) -> str:
    """Fit a tools YAML mapping by dropping whole tool entries."""
    if max_chars <= 0:
        return ""
    raw = str(text or "").strip()
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw

    lines = raw.splitlines()
    if not lines:
        return ""

    # Keep header lines until "tools:" (inclusive).
    header: List[str] = []
    tool_lines: List[str] = []
    in_tools = False
    for line in lines:
        if not in_tools:
            header.append(line)
            if line.strip() == "tools:":
                in_tools = True
            continue
        tool_lines.append(line)

    if not in_tools:
        # No list to drop; just keep within budget at line boundaries.
        return _fit_markdown_section(raw, max_chars=max_chars)

    # Split tool entries on "  - name:".
    entries: List[List[str]] = []
    current: List[str] = []
    for line in tool_lines:
        if line.startswith("  - "):
            if current:
                entries.append(current)
            current = [line]
        else:
            if current:
                current.append(line)
    if current:
        entries.append(current)

    out_lines: List[str] = list(header)
    for entry in entries:
        candidate_lines = out_lines + entry
        candidate = "\n".join(candidate_lines).strip()
        if len(candidate) <= max_chars:
            out_lines = candidate_lines
    return "\n".join(out_lines).strip()


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
