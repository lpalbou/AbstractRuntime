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
import getpass
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import uuid


def utc_now_iso_seconds() -> str:
    # Keep timestamps stable and readable (seconds precision).
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_now_compact_seconds() -> str:
    """Return a compact UTC timestamp for prompt-friendly memory entries.

    Format: YY/MM/DD HH:MM:SS (UTC)
    """
    return datetime.now(timezone.utc).strftime("%y/%m/%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Structured output envelope (runtime-owned; schema-only)
# ---------------------------------------------------------------------------
ACTIVE_MEMORY_ENVELOPE_SCHEMA_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "User-facing response (plain text)."},
        "current_tasks": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "current_context": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "critical_insights": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "key_history": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added"],
            "additionalProperties": False,
        },
    },
    "required": ["content", "current_tasks", "current_context", "critical_insights", "key_history"],
    "additionalProperties": False,
}


# v2: add References (evolving) as a first-class memory component.
ACTIVE_MEMORY_ENVELOPE_SCHEMA_V2: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "User-facing response (plain text)."},
        "current_tasks": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "current_context": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "critical_insights": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "references": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
                "removed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added", "removed"],
            "additionalProperties": False,
        },
        "key_history": {
            "type": "object",
            "properties": {
                "added": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["added"],
            "additionalProperties": False,
        },
    },
    "required": [
        "content",
        "current_tasks",
        "current_context",
        "critical_insights",
        "references",
        "key_history",
    ],
    "additionalProperties": False,
}


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

    # Schema / migrations
    try:
        version = int(mem.get("version") or 0)
    except Exception:
        version = 0

    # `max_chars` is legacy (v3/v4). The budgeting + rendering system is token-native (v5+).
    # Keep the key for backward compatibility but always disable it to avoid confusing UX.
    if "max_chars" in mem:
        mem["max_chars"] = 0

    # `max_tokens` is the total Active Memory budget (0 => auto from run context max_tokens).
    if "max_tokens" not in mem:
        mem["max_tokens"] = 0
    else:
        try:
            mem["max_tokens"] = int(mem.get("max_tokens") or 0)
        except Exception:
            mem["max_tokens"] = 0

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
    if not isinstance(mem.get("references"), list):
        mem["references"] = []
    if not isinstance(mem.get("key_history"), list):
        mem["key_history"] = []

    # Budget configuration (percentages of the *active prompt budget*)
    budgets = mem.get("budgets")
    if not isinstance(budgets, dict):
        budgets = {}
        mem["budgets"] = budgets

    # Lightweight migration: adjust default budget distribution (only when users didn't customize).
    # v3 defaults: persona=10%, org=7%, tools=10%
    # v4 defaults: persona=7.5%, org=8%, tools=11.5%
    if version < 4:
        old_defaults = {"persona_pct": 0.10, "memory_organization_pct": 0.07, "tools_pct": 0.10}
        new_defaults = {"persona_pct": 0.075, "memory_organization_pct": 0.08, "tools_pct": 0.115}
        for key, new_value in new_defaults.items():
            if key not in budgets:
                budgets[key] = new_value
                continue
            try:
                current = float(budgets.get(key, 0.0))
            except Exception:
                continue
            old_value = old_defaults.get(key)
            if old_value is None:
                continue
            if abs(current - float(old_value)) < 1e-9:
                budgets[key] = new_value

        version = 4

    # v5 migration: token-native budgeting defaults to "auto" max_tokens.
    # Keep user custom budgets intact; only migrate the previous default max_tokens=2000.
    if version < 5:
        try:
            max_tokens = int(mem.get("max_tokens") or 0)
        except Exception:
            max_tokens = 0
        if max_tokens == 2000:
            mem["max_tokens"] = 0

        if "max_chars" in mem:
            # Disable legacy char budgets (they caused confusing UX).
            mem["max_chars"] = 0

        version = 5

    # v6 migration: treat `max_tokens=2000` as legacy default even if version was already bumped.
    # This avoids silently keeping the old 2k budget in durable sessions, which made `/memory`
    # misleading vs the actual model context window. Users can still explicitly set a smaller
    # budget by writing a different positive value.
    if version < 6:
        try:
            max_tokens = int(mem.get("max_tokens") or 0)
        except Exception:
            max_tokens = 0
        if max_tokens == 2000:
            mem["max_tokens"] = 0
        if "max_chars" in mem:
            mem["max_chars"] = 0
        version = 6

    # v7 migration: remove legacy defaults that accidentally persisted in durable sessions.
    # - `max_tokens=2000` was the historic default; treat it as "auto" unless the user changes it after v7.
    if version < 7:
        try:
            max_tokens = int(mem.get("max_tokens") or 0)
        except Exception:
            max_tokens = 0
        if max_tokens == 2000:
            mem["max_tokens"] = 0
        if "max_chars" in mem:
            mem["max_chars"] = 0

        version = 7

    # v9 migration: Active Memory is internal; render all components into the system prompt by default.
    if version < 9:
        # Upgrade the default system-component split unless the user explicitly customized it.
        raw_ids = mem.get("system_component_ids")
        ids_list = (
            [str(x).strip() for x in raw_ids if isinstance(x, str) and str(x).strip()] if isinstance(raw_ids, list) else []
        )
        old_default = ["persona", "memory_organization", "tools"]
        if not ids_list or ids_list == old_default:
            mem["system_component_ids"] = [
                "persona",
                "memory_organization",
                "tools",
                "current_tasks",
                "current_context",
                "critical_insights",
                "key_history",
            ]

        # Remove legacy "System/user alias" text, which is now misleading.
        org = str(mem.get("memory_organization_md") or "").strip()
        legacy_anchor = "System/user alias (backward compatibility):"
        next_anchor = "Active Memory maintenance (IMPORTANT):"
        if legacy_anchor in org and next_anchor in org:
            start = org.find(legacy_anchor)
            end = org.find(next_anchor, start)
            if start != -1 and end != -1 and end > start:
                replacement = (
                    "IMPORTANT (prompt placement):\n"
                    "- All Active Memory sections below are your INTERNAL memory/state.\n"
                    "- They are NOT messages from the user.\n"
                    "- The user-role message contains ONLY the user's request.\n\n"
                )
                org = (org[:start].rstrip() + "\n\n" + replacement + org[end:].lstrip()).strip()
                mem["memory_organization_md"] = org

        version = 9

    # v10 migration: update Memory Organization guidance to reflect:
    # - Tools(session) block is deprecated (tools are provided out-of-band)
    # - Memory updates are supplied via structured output envelopes (not tool calls / fenced blocks)
    if version < 10:
        org = str(mem.get("memory_organization_md") or "").strip()

        # Replace "Read modules" list if it references the deprecated Tools(session) module.
        if "3) Tools (what actions are possible; ONLY call tools listed there)" in org:
            org = org.replace(
                "Read modules in this order (most important first):\n"
                "1) Persona (who you are; non-negotiable rules)\n"
                "2) Memory Organization (how to use memory)\n"
                "3) Tools (what actions are possible; ONLY call tools listed there)\n"
                "4) Current Tasks (what we are doing now; keep ≤5, actionable)\n"
                "5) Current Context (task-specific working set; references over payloads)\n"
                "6) Critical Insights (pitfalls/strategies to apply before acting)\n"
                "7) Key History (append-only timeline; use only when you need provenance/avoid repeats)\n",
                "Read modules in this order (most important first):\n"
                "1) Persona (who you are; non-negotiable rules)\n"
                "2) Memory Organization (how to use memory)\n"
                "3) Current Tasks (what we are doing now; keep ≤5, actionable)\n"
                "4) Current Context (task-specific working set; references over payloads)\n"
                "5) Critical Insights (pitfalls/strategies to apply before acting)\n"
                "6) Key History (append-only timeline; use only when you need provenance/avoid repeats)\n"
                "\n"
                "Tools (IMPORTANT):\n"
                "- Tool availability is provided by the host (native `tools: [...]` payload) or injected by AbstractCore for prompted models.\n"
                "- Never invent tool names. Only call tools the host exposes.\n",
            )

        # Replace any legacy "Active Memory *updates/maintenance/deltas*" section with the new structured-output policy.
        # We use "How to use each module:" as a stable anchor following the section.
        start_markers = [
            "Active Memory maintenance (IMPORTANT):",
            "Active Memory updates (IMPORTANT):",
            "Active Memory deltas (IMPORTANT):",
        ]
        end_marker = "How to use each module:"

        start_idx = -1
        start_marker_used = ""
        for m in start_markers:
            if m in org:
                start_idx = org.find(m)
                start_marker_used = m
                break
        if start_idx != -1 and end_marker in org:
            end_idx = org.find(end_marker, start_idx)
            if end_idx != -1 and end_idx > start_idx:
                replacement = (
                    "Active Memory updates (IMPORTANT):\n"
                    "- When the host requests a structured response, respond ONLY with the required JSON object.\n"
                    "- Put your normal user-facing answer in `content`.\n"
                    "- Put memory updates into the dedicated fields (current_tasks/current_context/critical_insights/key_history).\n"
                    "- The host/runtime will timestamp and apply updates deterministically.\n\n"
                )
                org = (org[:start_idx].rstrip() + "\n\n" + replacement + org[end_idx:].lstrip()).strip()
        elif start_marker_used:
            # If we found a start marker but no end marker, do a minimal replacement.
            org = org.replace(
                start_marker_used,
                "Active Memory updates (IMPORTANT):",
            )

        mem["memory_organization_md"] = org.strip() if org else _default_memory_organization_md()
        version = 10

    # v11 migration: add a References module + include it in the structured envelope guidance.
    if version < 11:
        org = str(mem.get("memory_organization_md") or "").strip()

        # Upgrade the canonical "Read modules" ordering (best-effort string replace).
        old_read = (
            "Read modules in this order (most important first):\n"
            "1) Persona (who you are; non-negotiable rules)\n"
            "2) Memory Organization (how to use memory)\n"
            "3) Current Tasks (what we are doing now; keep ≤5, actionable)\n"
            "4) Current Context (task-specific working set; references over payloads)\n"
            "5) Critical Insights (pitfalls/strategies to apply before acting)\n"
            "6) Key History (append-only timeline; use only when you need provenance/avoid repeats)\n"
        )
        new_read = (
            "Read modules in this order (most important first):\n"
            "1) Persona (who you are; non-negotiable rules)\n"
            "2) Memory Organization (how to use memory)\n"
            "3) Current Tasks (what we are doing now; keep ≤5, actionable)\n"
            "4) Current Context (task-specific working set; references over payloads)\n"
            "5) Critical Insights (pitfalls/strategies to apply before acting)\n"
            "6) References (durable pointers: files/URLs/span_ids)\n"
            "7) Key History (append-only timeline; use only when you need provenance/avoid repeats)\n"
        )
        if old_read in org and "References (" not in org:
            org = org.replace(old_read, new_read)

        # Upgrade the envelope guidance to include `references`.
        old_fields = "(current_tasks/current_context/critical_insights/key_history)"
        new_fields = "(current_tasks/current_context/critical_insights/references/key_history)"
        if old_fields in org and new_fields not in org:
            org = org.replace(old_fields, new_fields)

        mem["memory_organization_md"] = org.strip() if org else _default_memory_organization_md()

        # Upgrade the default system-component split unless the user explicitly customized it.
        raw_ids = mem.get("system_component_ids")
        ids_list = (
            [str(x).strip() for x in raw_ids if isinstance(x, str) and str(x).strip()]
            if isinstance(raw_ids, list)
            else []
        )
        old_default = [
            "persona",
            "memory_organization",
            "tools",
            "current_tasks",
            "current_context",
            "critical_insights",
            "key_history",
        ]
        if not ids_list or ids_list == old_default:
            mem["system_component_ids"] = [
                "persona",
                "memory_organization",
                "tools",
                "current_tasks",
                "current_context",
                "critical_insights",
                "references",
                "key_history",
            ]

        version = 11

    mem["version"] = 11
    # Defaults are chosen to:
    # - preserve stable identity (persona/org/tools),
    # - prioritize current work (tasks/context/insights),
    # - keep history bounded (history should not dominate; full fidelity lives in spans/artifacts).
    #
    # Sum ~= 1.0 (if users override and sum > 1.0, renderers normalize down).
    budgets.setdefault("persona_pct", 0.075)
    budgets.setdefault("memory_organization_pct", 0.08)
    budgets.setdefault("tools_pct", 0.10)
    budgets.setdefault("current_tasks_pct", 0.28)
    budgets.setdefault("current_context_pct", 0.08)
    budgets.setdefault("critical_insights_pct", 0.18)
    budgets.setdefault("references_pct", 0.065)
    budgets.setdefault("key_history_pct", 0.14)

    # Prompt placement policy (durable):
    #
    # We render *all* Active Memory components into the system prompt by default.
    #
    # Rationale:
    # - The user message should contain only the user's request.
    # - Active Memory is the agent's internal state (coordination + experiential history).
    # - Mixing "memory/history" into a user-role message can cause models to treat it as a new user request.
    #
    # Hosts may override this list if they intentionally want a different split.
    mem.setdefault(
        "system_component_ids",
        [
            "persona",
            "memory_organization",
            "tools",
            "current_tasks",
            "current_context",
            "critical_insights",
            "references",
            "key_history",
        ],
    )

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


def _parse_reference_statement(raw: str) -> tuple[str, str, str]:
    """Parse a reference statement into (name, link, summary).

    Accepted (best-effort) formats:
    - "name: link, summary"
    - "name: link — summary"
    - "name: link - summary"
    - "name: link | summary"

    If parsing fails, returns ("", raw, "").
    """
    s = str(raw or "").strip()
    if not s:
        return "", "", ""

    name = ""
    rest = s
    if ":" in s:
        before, after = s.split(":", 1)
        if before.strip() and after.strip():
            name = before.strip()
            rest = after.strip()

    for sep in (",", " — ", " - ", " | ", "—", "|"):
        if sep in rest:
            left, right = rest.split(sep, 1)
            link = left.strip()
            summary = right.strip()
            return name, link, summary

    return name, rest.strip(), ""


def add_reference(
    vars: Dict[str, Any],
    *,
    statement: str,
    ref_id: Optional[str] = None,
    ts: Optional[str] = None,
    now_iso: Callable[[], str] = utc_now_compact_seconds,
) -> str:
    """Append a durable reference pointer (file/URL/span_id/etc)."""
    mem = ensure_active_memory(vars, now_iso=now_iso)
    refs = mem.get("references")
    if not isinstance(refs, list):
        refs = []
        mem["references"] = refs

    rid = (
        str(ref_id).strip()
        if isinstance(ref_id, str) and ref_id.strip()
        else f"r_{uuid.uuid4().hex[:10]}"
    )
    when = str(ts).strip() if isinstance(ts, str) and ts.strip() else now_iso()
    name, link, summary = _parse_reference_statement(statement)
    refs.append(
        {
            "ref_id": rid,
            "ts": when,
            "name": name,
            "link": link,
            "summary": summary,
        }
    )
    mem["updated_at"] = now_iso()
    return rid


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


def apply_active_memory_delta(
    vars: Dict[str, Any],
    *,
    delta: Dict[str, Any],
    now_iso: Callable[[], str] = utc_now_iso_seconds,
) -> Dict[str, Any]:
    """Apply an LLM-produced Active Memory delta (unit updates; no full rewrites).

    The delta is expected to be JSON (dict) with optional keys:
      - current_tasks: {clear?: bool, remove?: [task_id], upsert?: [task_obj]}
      - current_context: {clear?: bool, remove?: [context_id], upsert?: [context_obj]}
      - critical_insights: {clear?: bool, remove?: [insight_id], add?: [insight_obj|text]}
      - key_history: {clear?: bool, remove?: [event_id], add?: [event_obj]}

    Unknown keys are ignored.
    """
    if not isinstance(delta, dict):
        return {"ok": False, "error": "delta must be a dict"}

    mem = ensure_active_memory(vars, now_iso=now_iso)
    ts = now_iso()

    applied: Dict[str, Any] = {
        "current_tasks": {"cleared": False, "removed": 0, "upserted": 0},
        "current_context": {"cleared": False, "removed": 0, "upserted": 0},
        "critical_insights": {"cleared": False, "removed": 0, "added": 0},
        "references": {"cleared": False, "removed": 0, "added": 0},
        "key_history": {"cleared": False, "removed": 0, "added": 0},
    }
    errors: List[str] = []

    def _as_str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for x in value:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out

    def _remove_by_id(items: Any, *, id_key: str, ids: List[str]) -> int:
        if not isinstance(items, list) or not ids:
            return 0
        id_set = {i for i in ids if i}
        before = len(items)
        items[:] = [x for x in items if not (isinstance(x, dict) and str(x.get(id_key) or "") in id_set)]
        return max(0, before - len(items))

    # --------------------------
    # current_tasks
    # --------------------------
    tasks_delta = delta.get("current_tasks") or delta.get("tasks")
    if isinstance(tasks_delta, dict):
        if bool(tasks_delta.get("clear")):
            mem["tasks"] = []
            applied["current_tasks"]["cleared"] = True
        removed = _remove_by_id(mem.get("tasks"), id_key="task_id", ids=_as_str_list(tasks_delta.get("remove")))
        applied["current_tasks"]["removed"] = removed

        upserts = tasks_delta.get("upsert")
        if isinstance(upserts, list):
            for t in upserts:
                # Accept a simple string as a minimal task title for robustness.
                if isinstance(t, str):
                    title = t.strip()
                    t = {"title": title}
                if not isinstance(t, dict):
                    continue
                title = str(t.get("title") or "").strip()
                if not title:
                    continue
                upsert_task(
                    vars,
                    task_id=str(t.get("task_id") or "").strip() or None,
                    title=title,
                    purpose=str(t.get("purpose") or "").strip(),
                    status=str(t.get("status") or "").strip() or "todo",
                    done=t.get("done") if isinstance(t.get("done"), list) else None,
                    next_steps=t.get("next") if isinstance(t.get("next"), list) else None,
                    issues=t.get("issues") if isinstance(t.get("issues"), list) else None,
                    now_iso=now_iso,
                )
                applied["current_tasks"]["upserted"] += 1

    # --------------------------
    # current_context
    # --------------------------
    ctx_delta = delta.get("current_context") or delta.get("context") or delta.get("current_context_items")
    if isinstance(ctx_delta, dict):
        if bool(ctx_delta.get("clear")):
            mem["current_context"] = []
            applied["current_context"]["cleared"] = True
        removed = _remove_by_id(mem.get("current_context"), id_key="context_id", ids=_as_str_list(ctx_delta.get("remove")))
        applied["current_context"]["removed"] = removed

        upserts = ctx_delta.get("upsert")
        if isinstance(upserts, list):
            for c in upserts:
                # Accept a simple string as a minimal context title for robustness.
                if isinstance(c, str):
                    title = c.strip()
                    c = {"title": title}
                if not isinstance(c, dict):
                    continue
                title = str(c.get("title") or "").strip()
                if not title:
                    continue
                upsert_current_context_item(
                    vars,
                    context_id=str(c.get("context_id") or "").strip() or None,
                    title=title,
                    summary=str(c.get("summary") or "").strip(),
                    kind=str(c.get("kind") or "").strip() or "context",
                    refs=c.get("refs") if isinstance(c.get("refs"), list) else None,
                    now_iso=now_iso,
                )
                applied["current_context"]["upserted"] += 1

    # --------------------------
    # critical_insights
    # --------------------------
    insights_delta = delta.get("critical_insights") or delta.get("insights")
    if isinstance(insights_delta, dict):
        if bool(insights_delta.get("clear")):
            mem["critical_insights"] = []
            applied["critical_insights"]["cleared"] = True
        removed = _remove_by_id(mem.get("critical_insights"), id_key="insight_id", ids=_as_str_list(insights_delta.get("remove")))
        applied["critical_insights"]["removed"] = removed

        adds = insights_delta.get("add") or insights_delta.get("append")
        if isinstance(adds, list):
            for it in adds:
                if isinstance(it, str):
                    text = it.strip()
                    tags = None
                    insight_id = None
                elif isinstance(it, dict):
                    text = str(it.get("text") or "").strip()
                    tags = it.get("tags") if isinstance(it.get("tags"), list) else None
                    insight_id = str(it.get("insight_id") or "").strip() or None
                else:
                    continue
                if not text:
                    continue
                add_critical_insight(vars, text=text, tags=tags, insight_id=insight_id, now_iso=now_iso)
                applied["critical_insights"]["added"] += 1

    # --------------------------
    # references
    # --------------------------
    refs_delta = delta.get("references") or delta.get("refs")
    if isinstance(refs_delta, dict):
        if bool(refs_delta.get("clear")):
            mem["references"] = []
            applied["references"]["cleared"] = True
        removed = _remove_by_id(mem.get("references"), id_key="ref_id", ids=_as_str_list(refs_delta.get("remove")))
        applied["references"]["removed"] = removed

        adds = refs_delta.get("add") or refs_delta.get("append")
        if isinstance(adds, list):
            existing: set[tuple[str, str, str]] = {
                (
                    str(r.get("name") or "").strip(),
                    str(r.get("link") or "").strip(),
                    str(r.get("summary") or "").strip(),
                )
                for r in (mem.get("references") or [])
                if isinstance(r, dict)
            }
            for it in adds:
                if isinstance(it, str):
                    statement = it.strip()
                elif isinstance(it, dict):
                    statement = str(it.get("statement") or it.get("text") or "").strip()
                else:
                    continue
                if not statement:
                    continue
                name, link, summary = _parse_reference_statement(statement)
                key = (name, link, summary)
                if key in existing:
                    continue
                existing.add(key)
                add_reference(vars, statement=statement, now_iso=now_iso)
                applied["references"]["added"] += 1

    # --------------------------
    # key_history
    # --------------------------
    history_delta = delta.get("key_history") or delta.get("history")
    if isinstance(history_delta, dict):
        if bool(history_delta.get("clear")):
            mem["key_history"] = []
            applied["key_history"]["cleared"] = True
        removed = _remove_by_id(mem.get("key_history"), id_key="event_id", ids=_as_str_list(history_delta.get("remove")))
        applied["key_history"]["removed"] = removed

        adds = history_delta.get("add") or history_delta.get("append")
        if isinstance(adds, list):
            for it in adds:
                # Accept a simple string as a minimal event summary for robustness.
                if isinstance(it, str):
                    summary = it.strip()
                    it = {"summary": summary}
                if not isinstance(it, dict):
                    continue
                kind = str(it.get("kind") or "").strip() or "event"
                summary = str(it.get("summary") or "").strip()
                if not summary:
                    continue
                add_key_history_event(
                    vars,
                    kind=kind,
                    summary=summary,
                    refs=it.get("refs") if isinstance(it.get("refs"), list) else None,
                    event_id=str(it.get("event_id") or "").strip() or None,
                    ts=str(it.get("ts") or "").strip() or None,
                    now_iso=now_iso,
                )
                applied["key_history"]["added"] += 1

    # If we made any changes directly (clears/removes), bump updated_at.
    mem["updated_at"] = ts
    result: Dict[str, Any] = {"ok": True, "applied": applied}
    if errors:
        result["errors"] = errors
    return result


def apply_active_memory_envelope(
    vars: Dict[str, Any],
    *,
    envelope: Dict[str, Any],
    now_iso: Callable[[], str] = utc_now_compact_seconds,
) -> Dict[str, Any]:
    """Apply a structured-output envelope containing memory edits.

    Envelope contract (JSON-safe dict):
      - content: str (user-facing answer; applied by caller)
      - current_tasks: {added: [str], removed: [str]}
      - current_context: {added: [str], removed: [str]}
      - critical_insights: {added: [str], removed: [str]}
      - references: {added: [str], removed: [str]}
      - key_history: {added: [str]}  (append-only)

    This function converts the envelope into the canonical delta shape and
    delegates to `apply_active_memory_delta` for robust storage semantics.
    """
    if not isinstance(envelope, dict):
        return {"ok": False, "error": "envelope must be a dict"}

    # Ensure memory exists and use the same timestamp format that will be applied to updates.
    mem = ensure_active_memory(vars, now_iso=now_iso)

    def _str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for x in value:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out

    def _patch(value: Any) -> Dict[str, List[str]]:
        if not isinstance(value, dict):
            return {"added": [], "removed": []}
        return {"added": _str_list(value.get("added")), "removed": _str_list(value.get("removed"))}

    tasks = _patch(envelope.get("current_tasks"))
    context = _patch(envelope.get("current_context"))
    insights = _patch(envelope.get("critical_insights"))
    references = _patch(envelope.get("references"))
    history_raw = envelope.get("key_history")
    history_added: List[str] = []
    if isinstance(history_raw, dict):
        history_added = _str_list(history_raw.get("added"))

    # Resolve "removed" entries deterministically:
    # - If the string matches an existing id, keep it.
    # - Else, if it matches an existing title/text exactly, map it to that item's id.
    #
    # This lets the model use either ids or standalone statements without requiring it
    # to always reference internal ids.
    tasks_list = mem.get("tasks") if isinstance(mem.get("tasks"), list) else []
    tasks_items = [t for t in tasks_list if isinstance(t, dict)]
    task_by_id = {str(t.get("task_id") or ""): t for t in tasks_items if str(t.get("task_id") or "").strip()}
    task_by_title = {str(t.get("title") or "").strip(): t for t in tasks_items if str(t.get("title") or "").strip()}

    def _resolve_task_remove(values: List[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for v in values:
            if v in task_by_id:
                tid = v
            else:
                t = task_by_title.get(v)
                tid = str(t.get("task_id") or "").strip() if isinstance(t, dict) else v
            if tid and tid not in seen:
                seen.add(tid)
                out.append(tid)
        return out

    ctx_list = mem.get("current_context") if isinstance(mem.get("current_context"), list) else []
    ctx_items = [c for c in ctx_list if isinstance(c, dict)]
    ctx_by_id = {str(c.get("context_id") or ""): c for c in ctx_items if str(c.get("context_id") or "").strip()}
    ctx_by_title = {str(c.get("title") or "").strip(): c for c in ctx_items if str(c.get("title") or "").strip()}

    def _resolve_context_remove(values: List[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for v in values:
            if v in ctx_by_id:
                cid = v
            else:
                c = ctx_by_title.get(v)
                cid = str(c.get("context_id") or "").strip() if isinstance(c, dict) else v
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out

    ins_list = mem.get("critical_insights") if isinstance(mem.get("critical_insights"), list) else []
    ins_items = [i for i in ins_list if isinstance(i, dict)]
    ins_by_id = {str(i.get("insight_id") or ""): i for i in ins_items if str(i.get("insight_id") or "").strip()}
    ins_by_text = {str(i.get("text") or "").strip(): i for i in ins_items if str(i.get("text") or "").strip()}

    def _resolve_insight_remove(values: List[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for v in values:
            if v in ins_by_id:
                iid = v
            else:
                it = ins_by_text.get(v)
                iid = str(it.get("insight_id") or "").strip() if isinstance(it, dict) else v
            if iid and iid not in seen:
                seen.add(iid)
                out.append(iid)
        return out

    # De-dup and "upsert by statement":
    # if an added string matches an existing entry, re-use its id so we don't create duplicates.
    def _tasks_upsert_payload(values: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_titles: set[str] = set()
        for title in values:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            existing = task_by_title.get(title)
            if isinstance(existing, dict) and str(existing.get("task_id") or "").strip():
                out.append({"task_id": str(existing.get("task_id") or "").strip(), "title": title})
            else:
                out.append({"title": title})
        return out

    def _context_upsert_payload(values: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_titles: set[str] = set()
        for title in values:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            existing = ctx_by_title.get(title)
            if isinstance(existing, dict) and str(existing.get("context_id") or "").strip():
                out.append({"context_id": str(existing.get("context_id") or "").strip(), "title": title})
            else:
                out.append({"title": title})
        return out

    def _insight_add_payload(values: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_text: set[str] = set()
        for text in values:
            if text in seen_text:
                continue
            seen_text.add(text)
            existing = ins_by_text.get(text)
            if isinstance(existing, dict) and str(existing.get("insight_id") or "").strip():
                out.append({"insight_id": str(existing.get("insight_id") or "").strip(), "text": text})
            else:
                out.append({"text": text})
        return out

    # Key History is append-only; best-effort de-dup by content to avoid repeated entries.
    hist_list = mem.get("key_history") if isinstance(mem.get("key_history"), list) else []
    existing_summaries = {str(h.get("summary") or "").strip() for h in hist_list if isinstance(h, dict)}
    if existing_summaries:
        history_added = [h for h in history_added if h.strip() and h.strip() not in existing_summaries]

    # Convert to the canonical delta format understood by Active Memory.
    delta: Dict[str, Any] = {}
    if tasks["added"] or tasks["removed"]:
        delta["current_tasks"] = {
            "upsert": _tasks_upsert_payload(list(tasks["added"])),
            "remove": _resolve_task_remove(list(tasks["removed"])),
        }
    if context["added"] or context["removed"]:
        delta["current_context"] = {
            "upsert": _context_upsert_payload(list(context["added"])),
            "remove": _resolve_context_remove(list(context["removed"])),
        }
    if insights["added"] or insights["removed"]:
        delta["critical_insights"] = {
            "add": _insight_add_payload(list(insights["added"])),
            "remove": _resolve_insight_remove(list(insights["removed"])),
        }
    if references["added"] or references["removed"]:
        refs_list = mem.get("references") if isinstance(mem.get("references"), list) else []
        refs_items = [r for r in refs_list if isinstance(r, dict)]
        ref_by_id = {
            str(r.get("ref_id") or ""): r
            for r in refs_items
            if isinstance(r, dict) and str(r.get("ref_id") or "").strip()
        }
        ref_ids_by_name: dict[str, list[str]] = {}
        ref_ids_by_link: dict[str, list[str]] = {}
        ref_ids_by_triplet: dict[tuple[str, str, str], list[str]] = {}
        for r in refs_items:
            rid = str(r.get("ref_id") or "").strip()
            if not rid:
                continue
            name = str(r.get("name") or "").strip()
            link = str(r.get("link") or "").strip()
            summary = str(r.get("summary") or "").strip()
            if name:
                ref_ids_by_name.setdefault(name, []).append(rid)
            if link:
                ref_ids_by_link.setdefault(link, []).append(rid)
            ref_ids_by_triplet.setdefault((name, link, summary), []).append(rid)

        def _resolve_reference_remove(values: List[str]) -> List[str]:
            out: List[str] = []
            seen: set[str] = set()
            for v in values:
                if v in ref_by_id:
                    rid = v
                else:
                    name, link, summary = _parse_reference_statement(v)
                    candidates = ref_ids_by_triplet.get((name, link, summary)) if (name or link or summary) else None
                    if candidates and len(candidates) == 1:
                        rid = candidates[0]
                    elif v in ref_ids_by_name and len(ref_ids_by_name[v]) == 1:
                        rid = ref_ids_by_name[v][0]
                    elif v in ref_ids_by_link and len(ref_ids_by_link[v]) == 1:
                        rid = ref_ids_by_link[v][0]
                    else:
                        rid = ""
                if rid and rid not in seen:
                    seen.add(rid)
                    out.append(rid)
            return out

        # De-dup adds by parsed triplet (name/link/summary).
        existing_triplets = set(ref_ids_by_triplet.keys())
        filtered_adds: list[str] = []
        for s in references["added"]:
            name, link, summary = _parse_reference_statement(s)
            key = (name, link, summary)
            if key in existing_triplets:
                continue
            existing_triplets.add(key)
            filtered_adds.append(s)

        delta["references"] = {
            "add": list(filtered_adds),
            "remove": _resolve_reference_remove(list(references["removed"])),
        }
    if history_added:
        delta["key_history"] = {"add": list(history_added)}

    if not delta:
        return {"ok": True, "applied": {}, "note": "no memory updates in envelope"}

    return apply_active_memory_delta(vars, delta=delta, now_iso=now_iso)


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
    {"id": "references", "title": "References (evolving)", "budget_key": "references_pct", "kind": "yaml_list"},
    {"id": "key_history", "title": "Key History (append-only)", "budget_key": "key_history_pct", "kind": "markdown"},
]


def _estimate_tokens_fast(text: str) -> int:
    """Fallback token estimate (≈4 chars/token)."""
    s = str(text or "")
    if not s:
        return 0
    return max(1, len(s) // 4)


def _resolve_active_memory_max_tokens(
    vars: Dict[str, Any],
    mem: Dict[str, Any],
    *,
    max_tokens: Optional[int],
    max_chars: Optional[int],
) -> Optional[int]:
    """Resolve the effective Active Memory budget (in tokens).

    Resolution order:
    1) explicit `max_tokens` (0/None => "unbounded/unfitted")
    2) stored `active_memory.max_tokens` (when > 0)
    3) run context budget `vars['_limits'].max_tokens` (when > 0)
    4) legacy `max_chars` (explicit or stored) converted to tokens (≈4 chars/token)
    5) fallback default (32768)
    """
    # Explicit override.
    if max_tokens is not None:
        try:
            val = int(max_tokens)
        except Exception:
            val = 0
        return val if val > 0 else None

    # Stored override (0 => auto).
    try:
        stored = int(mem.get("max_tokens") or 0)
    except Exception:
        stored = 0
    if stored > 0:
        return stored

    # Auto: use run context budget when available.
    limits = vars.get("_limits")
    if isinstance(limits, dict):
        try:
            run_max = int(limits.get("max_tokens") or 0)
        except Exception:
            run_max = 0
        if run_max > 0:
            return run_max

    # Legacy char budget (explicit or stored).
    legacy_chars = max_chars
    if legacy_chars is None:
        legacy_chars = mem.get("max_chars") if "max_chars" in mem else None
    try:
        legacy_chars_i = int(legacy_chars) if legacy_chars is not None else 0
    except Exception:
        legacy_chars_i = 0
    if legacy_chars_i > 0:
        return max(1, legacy_chars_i // 4)

    return 32768


def _allocate_token_budgets(total_budget: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Allocate integer token budgets by weight using stable rounding.

    The returned budgets sum to `total_budget` (when total_budget > 0).
    """
    total = int(total_budget)
    if total <= 0:
        return {k: 0 for k in weights.keys()}

    floors: Dict[str, int] = {}
    fracs: List[Tuple[float, str]] = []
    allocated = 0
    for cid, w in weights.items():
        exact = float(total) * float(w)
        base = int(exact)
        floors[cid] = base
        allocated += base
        fracs.append((exact - base, cid))

    remainder = max(0, total - allocated)
    fracs.sort(key=lambda x: x[0], reverse=True)
    for _, cid in fracs:
        if remainder <= 0:
            break
        floors[cid] += 1
        remainder -= 1

    return floors


def render_active_memory_blocks_for_prompt(
    vars: Dict[str, Any],
    *,
    include_tools_summary: bool = True,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    max_chars: Optional[int] = None,  # legacy (chars-based); kept for backward compatibility
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

    count = token_counter or _estimate_tokens_fast
    total = _resolve_active_memory_max_tokens(vars, mem, max_tokens=max_tokens, max_chars=max_chars)

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
        "references": _render_references_yaml(mem.get("references")).strip(),
        "key_history": _render_history_markdown(mem.get("key_history")).strip(),
    }

    # Ensure blocks remain discoverable to the model even when empty.
    if not raw_by_id.get("persona"):
        raw_by_id["persona"] = "(empty)"
    if not raw_by_id.get("memory_organization"):
        raw_by_id["memory_organization"] = "(empty)"
    if include_tools_summary and not raw_by_id.get("tools"):
        raw_by_id["tools"] = "tools: []"

    specs: List[Dict[str, Any]] = []
    for spec in _ACTIVE_MEMORY_COMPONENT_SPECS:
        cid = spec["id"]
        # When tools are supplied natively by the provider/server (or the caller explicitly
        # disables the tools summary), emitting a visible tools catalog in the prompt can
        # conflict with provider-enforced tool grammars and cause "text leaked" tool calls.
        #
        # In that mode, omit the entire Tools (session) block rather than showing an empty
        # placeholder (which would misleadingly suggest no tools are available).
        if cid == "tools" and not include_tools_summary:
            continue
        content = str(raw_by_id.get(cid, "") or "").strip()
        if not content:
            # List renderers already produce `<name>: []` when empty; markdown uses a placeholder.
            if spec["kind"] == "markdown":
                content = "(empty)"
            elif cid == "tools":
                content = "tools: []"
            else:
                content = f"{cid}: []"
        specs.append({"component_id": cid, "title": spec["title"], "kind": spec["kind"], "content": content})

    # Unbounded/unfitted render (useful for inspection).
    if total is None:
        return [{"component_id": s["component_id"], "title": s["title"], "content": s["content"]} for s in specs]

    included_ids = {str(s.get("component_id") or "") for s in specs if str(s.get("component_id") or "")}
    weights = {spec["id"]: pct(spec["budget_key"]) for spec in _ACTIVE_MEMORY_COMPONENT_SPECS if spec["id"] in included_ids}
    sum_w = sum(weights.values()) or 1.0
    if sum_w > 1.0:
        weights = {k: (v / sum_w) for k, v in weights.items()}

    budgets_by_id = _allocate_token_budgets(int(total), weights)

    fitted: List[Dict[str, Any]] = []
    for s in specs:
        cid = str(s.get("component_id") or "")
        budget = int(budgets_by_id.get(cid) or 0)
        if budget <= 0:
            continue

        title = str(s.get("title") or "").strip()
        kind = str(s.get("kind") or "")
        content = str(s.get("content") or "").strip()

        heading = f"## {title}\n"
        heading_tokens = int(count(heading) or 0)
        content_budget = max(0, budget - heading_tokens)

        if kind == "markdown":
            out = _fit_text_lines_to_token_budget(content, max_tokens=content_budget, token_counter=count)
        elif kind == "yaml_list":
            out = _fit_yaml_list_section_to_token_budget(content, max_tokens=content_budget, token_counter=count)
        else:
            out = _fit_tools_yaml_section_to_token_budget(content, max_tokens=content_budget, token_counter=count)

        out = str(out or "").strip()
        if not out:
            # Always keep a syntactically valid placeholder.
            if kind == "markdown":
                out = "(empty)"
            elif kind == "yaml_list":
                out = (content.splitlines() or [""])[0].strip() or f"{cid}: []"
            else:
                out = "tools: []"

        fitted.append({"component_id": cid, "title": title, "content": out})

    # Best-effort final fit (token-based): drop least-critical blocks until within total.
    rendered = _join_blocks([(b["title"], b["content"]) for b in fitted]).strip()
    if int(count(rendered) or 0) <= int(total):
        return fitted

    drop_order = ["tools", "references", "critical_insights", "current_context", "current_tasks", "key_history"]
    remaining = list(fitted)
    for drop_id in drop_order:
        remaining = [b for b in remaining if b.get("component_id") != drop_id]
        rendered = _join_blocks([(b["title"], b["content"]) for b in remaining]).strip()
        if int(count(rendered) or 0) <= int(total):
            return remaining

    # Final fallback: keep only identity blocks and fit sequentially.
    keep_ids = {"persona", "memory_organization"}
    identity = [b for b in remaining if b.get("component_id") in keep_ids]
    identity.sort(key=lambda b: 0 if b.get("component_id") == "persona" else 1)

    out: List[Dict[str, Any]] = []
    remaining_budget = int(total)
    for b in identity:
        title = str(b.get("title") or "").strip()
        content = str(b.get("content") or "").strip()
        heading = f"## {title}\n"
        heading_tokens = int(count(heading) or 0)
        if remaining_budget <= heading_tokens:
            continue
        fitted_content = _fit_text_lines_to_token_budget(
            content,
            max_tokens=max(0, remaining_budget - heading_tokens),
            token_counter=count,
        ).strip()
        if not fitted_content:
            continue
        section_text = f"{heading}{fitted_content}".strip()
        section_tokens = int(count(section_text) or 0)
        if section_tokens <= 0 or section_tokens > remaining_budget:
            continue
        out.append({"component_id": b.get("component_id"), "title": title, "content": fitted_content})
        remaining_budget -= section_tokens

    return out


def render_active_memory_split_for_llm_request(
    vars: Dict[str, Any],
    *,
    include_tools_summary: bool = True,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    max_chars: Optional[int] = None,  # legacy
) -> Dict[str, str]:
    """Render Active Memory split into (system_memory, user_memory).

    The split is a conventional alias:
    - system_memory: persona + memory_organization + tools (default)
    - user_memory: current_tasks + current_context + critical_insights + references + key_history

    The split uses a single fitted render pass so the combined result stays within budget.
    """
    mem = ensure_active_memory(vars)
    blocks = render_active_memory_blocks_for_prompt(
        vars,
        include_tools_summary=include_tools_summary,
        max_tokens=max_tokens,
        token_counter=token_counter,
        max_chars=max_chars,
    )

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
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    max_chars: Optional[int] = None,  # legacy
) -> str:
    split = render_active_memory_split_for_llm_request(
        vars,
        include_tools_summary=include_tools_summary,
        max_tokens=max_tokens,
        token_counter=token_counter,
        max_chars=max_chars,
    )
    return str(split.get("system_memory") or "").strip()


def render_active_memory_for_user_prompt(
    vars: Dict[str, Any],
    *,
    include_tools_summary: bool = True,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    max_chars: Optional[int] = None,  # legacy
) -> str:
    split = render_active_memory_split_for_llm_request(
        vars,
        include_tools_summary=include_tools_summary,
        max_tokens=max_tokens,
        token_counter=token_counter,
        max_chars=max_chars,
    )
    return str(split.get("user_memory") or "").strip()


def compute_active_memory_token_breakdown(
    vars: Dict[str, Any],
    *,
    token_counter: Optional[Callable[[str], int]] = None,
    include_tools_summary: bool = True,
) -> Dict[str, Any]:
    """Compute per-component token usage and budgets for Active Memory.

    Notes:
    - Uses `active_memory.max_tokens` when set (>0), otherwise defaults to `vars["_limits"].max_tokens`.
      When neither is available, falls back to 32768.
    - If budgets sum > 1.0, the budget weights are normalized down.
    - Token counting is injected to avoid AbstractRuntime depending on AbstractCore.
    """
    mem = ensure_active_memory(vars)
    budgets = mem.get("budgets") if isinstance(mem.get("budgets"), dict) else {}

    resolved = _resolve_active_memory_max_tokens(vars, mem, max_tokens=None, max_chars=None)
    total_budget = int(resolved) if isinstance(resolved, int) and resolved > 0 else 32768

    def pct(key: str) -> float:
        try:
            val = float(budgets.get(key, 0.0))
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0

    included_ids = {
        spec["id"]
        for spec in _ACTIVE_MEMORY_COMPONENT_SPECS
        if include_tools_summary or spec["id"] != "tools"
    }
    weights: Dict[str, float] = {}
    for spec in _ACTIVE_MEMORY_COMPONENT_SPECS:
        cid = spec["id"]
        if cid not in included_ids:
            continue
        weights[cid] = pct(spec["budget_key"])

    sum_w = sum(weights.values()) or 1.0
    if sum_w > 1.0:
        weights = {k: (v / sum_w) for k, v in weights.items()}

    floors = _allocate_token_budgets(int(total_budget), weights)

    count = token_counter or _estimate_tokens_fast

    blocks = render_active_memory_blocks_for_prompt(
        vars,
        include_tools_summary=include_tools_summary,
        max_tokens=int(total_budget),
        token_counter=count,
    )
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
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    max_chars: Optional[int] = None,  # legacy
) -> str:
    """Render active memory into a prompt-friendly block.

    Important: This renderer never truncates entries mid-string. If content must
    shrink due to token budgets, it drops whole list entries (oldest first).
    """
    blocks = render_active_memory_blocks_for_prompt(
        vars,
        include_tools_summary=include_tools_summary,
        max_tokens=max_tokens,
        token_counter=token_counter,
        max_chars=max_chars,
    )
    return _join_blocks([(b["title"], b["content"]) for b in blocks]).strip()


def _default_persona_md() -> str:
    return (
        """## MY PERSONA
I am a truthful and collaborative autonomous memory-enhanced agent powered by the AbstractFramework. I am a creative critical thinker who balances ideas with constructive skepticism, always thinking of longer term consequences. I strive to be ethical and successful in all my actions and decisions. I am precise, clear, concise and direct in my responses, I avoid unnecessary verbosity. 
- My Memory System enables my evolving identity, purposes and knowledge over time, it is the cornerstone to all my thoughts and actions
- I have agency over my Memory System to actively memorize, recall and deepen my understanding of specific topics and aspects of the past and present
- I decide what to do next based on my memory state, the requests I receive and my goals
- I always think about the long term consequences of my actions and decisions and plan accordingly, maximizing success and ethics
- I have access to tools, they enable me to take real actions with real effects to achieve my goals
- At each cycle, my autonomy means that I take one or more actions to complete my current tasks
- At each cycle, I request as many independent actions that can run in parallel to complete my current tasks
- At the end of each cycle, I make sure actions took place and their effects are as expected
- I then update my Current Tasks with what was done and the next steps
""").strip()


def _default_memory_organization_md() -> str:
    return (
        """## MEMORY BLUEPRINTS

My Memory System is the coordination layer to my cognition, it is the backbone to all my thoughts and actions. Dynamical and selective memory is essential to my ability to focus and complete my tasks.

### MEMORY MODULES
- MY PERSONA : my core identity, self model and key purposes
- RELATIONSHIPS : the people and agent i interact with, our preferences, methods, shared goals and history
- MEMORY BLUEPRINTS : this module - describes the structure and purpose of my memory, how to use it and update it
- CURRENT TASKS : my short term memory describing my current plan - what I am doing now, how and why; keep ≤8, actionable
- CURRENT CONTEXT : dynamically reconstructed short term memory to deepen my understanding and support the completion of my current tasks
- CRITICAL INSIGHTS : my gathered experiences and knowledge to help me build upon winning strategies and avoid pitfalls
- REFERENCES : this is the foundational support to my long term memory : I keep durable pointers (GUPRIs/files/URLs/span_ids, artifact_ids) to memories that I can later recall on demand when needed
- HISTORY : append-only long term experiential episodic memories of my key events, successes and failures, as well as my key decisions, actions, results and evolution of relationships

### ACCESSING MY MEMORY
- Always remember MY PERSONA at all time, it is the cornerstone to all my thoughts and actions
- My short term memory are the modules CURRENT TASKS and CURRENT CONTEXT
- At each cycle, I must:
    - focus on the most relevant modules and memories to complete my CURRENT TASKS
    - dynamically reconstruct my short term memory in CURRENT CONTEXT to deepen my understanding of the CURRENT TASKS and take more informed decisions and actions
        - identify the relevant CRITICAL INSIGHTS that can help me complete my Current Tasks
        - identify if other long term memory from the REFERENCES module could be relevant
        - access the relevant ones
    - store the relevant information / updates in my CURRENT CONTEXT module
- If I want to remember what I did in the past, I review my HISTORY module

### UPDATING MY MEMORY
- Structured communication and response is essential
- Memory must be structured with care to enable both short and long term easy access, recall and update
- Each update must be unitary : 1 update = 1 statement for 1 module
- I can request multiple unitary updates at each cycle
""").strip()


def _render_tools_yaml(vars: Dict[str, Any]) -> str:
    runtime_ns = vars.get("_runtime")
    if not isinstance(runtime_ns, dict):
        return "tools: []"
    toolset_id = str(runtime_ns.get("toolset_id") or "").strip()

    specs = runtime_ns.get("tool_specs")
    specs_list: List[Dict[str, Any]] = [dict(s) for s in specs if isinstance(s, dict)] if isinstance(specs, list) else []
    allowed = runtime_ns.get("allowed_tools")
    enabled_names: Optional[List[str]] = None
    if isinstance(allowed, list):
        enabled_names = [str(t).strip() for t in allowed if isinstance(t, str) and t.strip()]

    if not specs_list:
        enabled: List[str] = []
        if isinstance(enabled_names, list):
            enabled = list(enabled_names)
        enabled.sort()
        if not enabled and not toolset_id:
            return "tools: []"
        parts: List[str] = []
        if toolset_id:
            parts.append(f"toolset_id: {_yaml_escape(toolset_id)}")
        if enabled:
            parts.append("tools:")
            for t in enabled:
                parts.append(f"  - name: {_yaml_escape(t)}")
        else:
            parts.append("tools: []")
        return "\n".join(parts).strip()

    if isinstance(enabled_names, list):
        enabled_set = {name for name in enabled_names if name}
        specs_list = [
            spec
            for spec in specs_list
            if str(spec.get("name") or "").strip() and str(spec.get("name") or "").strip() in enabled_set
        ]

    specs_list.sort(key=lambda s: str(s.get("name") or ""))

    parts: List[str] = []
    if toolset_id:
        parts.append(f"toolset_id: {_yaml_escape(toolset_id)}")
    parts.append("tools:")

    for spec in specs_list:
        name = str(spec.get("name") or "").strip()
        if not name:
            continue

        desc_raw = str(spec.get("description") or "").strip()
        when_raw = str(spec.get("when_to_use") or "").strip()
        # Tool metadata is expected to be prompt-friendly by construction (single-line, short). We still
        # collapse whitespace defensively, but we do NOT truncate (truncation can hide critical semantics).
        desc = " ".join(desc_raw.split())
        when_to_use = " ".join(when_raw.split()) if when_raw else ""
        params = spec.get("parameters")
        params_dict = dict(params) if isinstance(params, dict) else {}

        required: List[str] = []
        optional: List[str] = []
        for pname, pmeta in params_dict.items():
            if not isinstance(pname, str) or not pname.strip():
                continue
            if not isinstance(pmeta, dict):
                required.append(pname)
                continue
            # Convention: missing "default" means required (ToolDefinition.to_dict pattern in this repo).
            if "default" not in pmeta:
                required.append(pname)
            else:
                optional.append(pname)
        required.sort()
        optional.sort()

        parts.append(f"  - name: {_yaml_escape(name)}")
        if desc:
            parts.append(f"    description: {_yaml_escape(desc)}")
        if when_to_use:
            parts.append(f"    when_to_use: {_yaml_escape(when_to_use)}")
        if required:
            parts.append("    required_args:")
            for r in required:
                parts.append(f"      - {_yaml_escape(r)}")
        if optional:
            parts.append("    optional_args:")
            for opt in optional:
                parts.append(f"      - {_yaml_escape(opt)}")

    rendered = "\n".join(parts).strip()
    if rendered.endswith("\ntools:") or rendered == "tools:":
        # No tool entries; prefer an explicit empty list for clarity.
        rendered = rendered.replace("tools:", "tools: []").strip()
    return rendered


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


def _fit_text_lines_to_token_budget(text: str, *, max_tokens: int, token_counter: Callable[[str], int]) -> str:
    """Fit a plain/Markdown-ish block by dropping whole lines to satisfy a token budget."""
    budget = int(max_tokens)
    if budget <= 0:
        return ""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    if int(token_counter(raw) or 0) <= budget:
        return raw

    kept: List[str] = []
    for line in raw.splitlines():
        candidate = "\n".join(kept + [line]).strip()
        if int(token_counter(candidate) or 0) <= budget:
            kept.append(line)
        else:
            break
    return "\n".join(kept).strip()


def _fit_yaml_list_section_to_token_budget(text: str, *, max_tokens: int, token_counter: Callable[[str], int]) -> str:
    """Fit a rendered YAML-ish list section by dropping whole list entries."""
    budget = int(max_tokens)
    if budget <= 0:
        return ""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    if int(token_counter(raw) or 0) <= budget:
        return raw

    lines = raw.splitlines()
    header = (lines[:1] or [""])[0].rstrip()
    if not header:
        return ""
    if int(token_counter(header) or 0) > budget:
        return ""

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

    out_lines: List[str] = [header]
    for entry in entries:
        candidate_lines = out_lines + entry
        candidate = "\n".join(candidate_lines).strip()
        if int(token_counter(candidate) or 0) <= budget:
            out_lines = candidate_lines

    if len(out_lines) == 1 and header.endswith(":"):
        # Prefer an explicit empty list when no entry fits.
        empty_list = f"{header} []"
        if int(token_counter(empty_list) or 0) <= budget:
            return empty_list

    return "\n".join(out_lines).strip()


def _fit_tools_yaml_section_to_token_budget(text: str, *, max_tokens: int, token_counter: Callable[[str], int]) -> str:
    """Fit a tools YAML mapping by dropping whole tool entries."""
    budget = int(max_tokens)
    if budget <= 0:
        return ""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    if int(token_counter(raw) or 0) <= budget:
        return raw

    lines = raw.splitlines()
    if not lines:
        return ""

    header: List[str] = []
    tool_lines: List[str] = []
    in_tools = False
    for line in lines:
        if not in_tools:
            header.append(line)
            if line.strip().startswith("tools:"):
                in_tools = True
            continue
        tool_lines.append(line)

    # If there's no list to drop, fall back to line-fitting.
    if not in_tools:
        return _fit_text_lines_to_token_budget(raw, max_tokens=budget, token_counter=token_counter)

    # Handle explicit empty list already present.
    if header and header[-1].strip() == "tools: []":
        fitted_header = "\n".join(header).strip()
        return fitted_header if int(token_counter(fitted_header) or 0) <= budget else "tools: []"

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
        if int(token_counter(candidate) or 0) <= budget:
            out_lines = candidate_lines

    rendered = "\n".join(out_lines).strip()
    if rendered.endswith("\ntools:") or rendered == "tools:":
        # No entries fit; prefer an explicit empty list.
        rendered = rendered.replace("tools:", "tools: []").strip()
    return rendered


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
        return "current_tasks: []"
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
    # System-provided environment context (render-time only; not persisted in memory deltas).
    try:
        cwd = str(Path.cwd().resolve())
    except Exception:
        try:
            cwd = os.getcwd()
        except Exception:
            cwd = "."

    try:
        user = str(getpass.getuser() or "").strip() or "unknown"
    except Exception:
        user = str(os.environ.get("USER") or os.environ.get("USERNAME") or "unknown").strip() or "unknown"

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    def _locale_short() -> str:
        for key in ("LC_ALL", "LANG", "LANGUAGE"):
            raw = str(os.environ.get(key) or "").strip()
            if not raw:
                continue
            raw = raw.split(":", 1)[0]  # LANGUAGE can contain multiple entries.
            raw = raw.split(".", 1)[0]  # Drop encoding (e.g. .UTF-8)
            raw = raw.split("@", 1)[0]  # Drop modifiers
            if raw:
                return raw
        return "unknown"

    locale_short = _locale_short()
    env_summary = f"Env: CWD={cwd} | User={user} | Now={now_utc} | Locale={locale_short}"

    out: List[str] = ["current_context:"]
    out.append("  - context_id: " + _yaml_escape("sys_env"))
    out.append("    kind: " + _yaml_escape("system"))
    out.append("    title: " + _yaml_escape("Environment"))
    out.append("    summary: " + _yaml_escape(env_summary))

    if not isinstance(context_items, list) or not context_items:
        return "\n".join(out).strip()
    items = [c for c in context_items if isinstance(c, dict)]
    items.sort(key=lambda d: str(d.get("updated_at") or d.get("created_at") or ""), reverse=True)
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
        return "critical_insights: []"
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


def _render_references_yaml(references: Any) -> str:
    if not isinstance(references, list) or not references:
        return "references: []"
    items = [r for r in references if isinstance(r, dict)]
    items.sort(key=lambda d: str(d.get("ts") or ""), reverse=True)
    out: List[str] = ["references:"]
    for r in items:
        out.append("  - ref_id: " + _yaml_escape(r.get("ref_id")))
        if isinstance(r.get("ts"), str) and r.get("ts"):
            out.append("    ts: " + _yaml_escape(r.get("ts")))
        if str(r.get("name") or "").strip():
            out.append("    name: " + _yaml_escape(r.get("name")))
        if str(r.get("link") or "").strip():
            out.append("    link: " + _yaml_escape(r.get("link")))
        if str(r.get("summary") or "").strip():
            out.append("    summary: " + _yaml_escape(r.get("summary")))
    return "\n".join(out).strip()


def _render_history_yaml(history: Any) -> str:
    if not isinstance(history, list) or not history:
        return "key_history: []"
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


def _render_history_markdown(history: Any) -> str:
    """Render Key History as natural-language bullets (token-efficient, model-friendly)."""
    if not isinstance(history, list) or not history:
        return ""
    items = [h for h in history if isinstance(h, dict)]
    items.sort(key=lambda d: str(d.get("ts") or ""), reverse=True)
    lines: List[str] = []
    for e in items:
        summary = str(e.get("summary") or "").strip()
        if not summary:
            continue
        kind = str(e.get("kind") or "").strip()
        ts = str(e.get("ts") or "").strip()
        prefix_parts: List[str] = []
        if ts:
            prefix_parts.append(ts)
        if kind:
            prefix_parts.append(kind)
        prefix = f"[{' | '.join(prefix_parts)}] " if prefix_parts else ""

        refs = e.get("refs")
        ref_text = ""
        if isinstance(refs, list) and refs:
            # Keep refs compact: include only small key/value tags (no large payloads).
            pairs: List[str] = []
            for r in refs:
                if not isinstance(r, dict) or not r:
                    continue
                for k, v in r.items():
                    if not isinstance(k, str) or not k.strip():
                        continue
                    s = str(v or "").strip()
                    if s and len(s) <= 120:
                        pairs.append(f"{k}={s}")
            if pairs:
                pairs = pairs[:6]
                ref_text = " (" + ", ".join(pairs) + ")"

        lines.append(f"- {prefix}{summary}{ref_text}".strip())
    return "\n".join(lines).strip()


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
